"""Stage C: SAM2 masklets + reference cutouts for Phantom-Koala samples.

Input is stage B's ``extracted.jsonl`` (81-frame clips + per-subject reference frames).
Output is byte-for-byte schema-compatible with UltraVid57k so that stage D
(``ultravid_pipeline.stages.quality`` / ``stages.index``) and the training dataloader
consume it unchanged.

The one substantive difference from UltraVid's segmentation worker is **bidirectional
propagation**. UltraVid seeds every box on frame 0, so a single forward
``propagate_in_video`` covers the clip. Phantom seed frames sit anywhere in ``[0, 80]``
(empirically clustered at ~4 / ~40 / ~76), so each subject is propagated twice from the
same conditioning frame -- once forward to cover ``[seed, last]`` and once with
``reverse=True`` to cover ``[0, seed]`` -- and the two half-tracks are merged.

Each direction runs from a freshly ``reset_state``-ed inference state so the reverse
pass cannot read memory frames written by the forward pass (SAM2's reverse memory
lookup indexes ``non_cond_frame_outputs`` at ``frame_idx + t_diff``, which the forward
pass would otherwise have populated with its own predictions).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .. import canvas as canvas_module

STAGE = "segment"

# CLIP model used for ``ref_clip_score``. This is the checkpoint UltraVid57k actually
# scored with (verified against the ``ref_clip_model`` provenance field recorded in its
# produced bbox JSONs), NOT the clip-vit-large path written in its config yaml, which
# does not exist on this filesystem. Keeping ViT-B/32 keeps stage D's
# ``min_ref_clip_score=0.23`` threshold on the same calibration.
DEFAULT_CLIP_MODEL = (
    "/mnt/pfs/share/pretrained_model/.cache/huggingface/hub/"
    "models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = (
    "/mnt/pfs/users/yuanze/projects/2025/describe-anything/checkpoints/sam2.1_hiera_large.pt"
)
LINE_WIDTH = 3
#: Stage C's annotation-canvas hypothesis. ``H_768_long`` is what the built pilot dataset
#: was produced under; the convention itself is unresolved (see ``phantom_data.canvas``).
DEFAULT_HYPOTHESIS = canvas_module.H_768_long


# --------------------------------------------------------------------------------------
# pure helpers (unit tested)
# --------------------------------------------------------------------------------------


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    """Tight xyxy box around the set pixels, exclusive on max, or None when empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


#: A reference-mask component is speckle unless it holds at least this share of the
#: largest component's area. Swept over all 94 pilot subjects: 0.02/0.05/0.1/0.2 all
#: drive the fragmented count to 0/94, but 0.05 keeps the most true foreground in the
#: worst case (78% vs 67% at 0.2), and unlike keeping only the largest component it does
#: not amputate a subject that SAM2 splits across two blobs (worst case there kept 54%).
REF_MIN_COMPONENT_SHARE = 0.05
#: Components below this absolute pixel count are speckle regardless of the share rule;
#: guards the degenerate case of a mask that is *entirely* dust, where the largest
#: speckle would otherwise define the threshold.
REF_MIN_COMPONENT_PIXELS = 16


def largest_components(mask: np.ndarray, min_share: float = REF_MIN_COMPONENT_SHARE,
                       min_pixels: int = REF_MIN_COMPONENT_PIXELS) -> np.ndarray:
    """Drop speckle components from a binary mask, keeping the subject's body.

    SAM2's box-prompted single-image masks routinely carry hundreds of stray specks
    scattered across the frame (measured: up to 850 components on the pilot set). Those
    specks are individually tiny but spatially spread, so the mask's bounding box -- which
    :func:`write_reference` uses as the cutout's crop window -- inflates to nearly the
    whole frame. The cutout then reads as a mostly-white canvas with a small subject in
    it, and ``ref_mask_coverage`` collapses even though the subject itself was segmented
    fine. Removing the specks *before* the crop window is computed is what fixes the
    cutout; it is not a cosmetic touch-up.

    Components are kept when they hold ``min_share`` of the largest component's area, so a
    subject SAM2 splits into a couple of blobs (torso + arm) survives intact.
    """
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count <= 1:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background is not a component
    threshold = max(float(sizes.max()) * min_share, float(min_pixels))
    keep = np.nonzero(sizes >= threshold)[0]
    return np.isin(labels, keep)


def component_stats(mask: np.ndarray) -> dict[str, Any]:
    """Fragmentation diagnostics recorded alongside each reference cutout."""
    from scipy import ndimage

    foreground = int(np.count_nonzero(mask))
    if not foreground:
        return {"ref_mask_components": 0, "ref_mask_largest_share": 0.0}
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "ref_mask_components": int(count),
        "ref_mask_largest_share": round(float(sizes.max()) / foreground, 6),
    }


def scale_bbox_to_frame(box: Iterable[float], width: int, height: int,
                        hypothesis: canvas_module.Hypothesis = DEFAULT_HYPOTHESIS) -> list[float]:
    """Map a Phantom annotation box onto a real ``(width, height)`` frame, then clamp.

    ``hypothesis`` selects the annotation canvas; the default is ``H_768_long``, which is
    the behaviour the built pilot dataset depends on. The convention is unresolved -- see
    :mod:`phantom_data.canvas`. Clamping into frame bounds stays unconditional, since
    annotation boxes routinely overhang the frame edge.
    """
    mapped = canvas_module.map_box(
        [float(value) for value in box], width, height, hypothesis
    )
    return clamp_to_frame(mapped, width, height)


#: Which coordinate system a manifest row's boxes are already in. **This is the single most
#: dangerous field in stage C's input contract**, because both values are plausible numbers of
#: the same magnitude and applying the wrong one corrupts every box silently.
#:
#: ``BOX_SPACE_ANNOTATION`` -- raw Phantom annotation coordinates (the ``_768`` fields written by
#: stage A/B). These must be projected through the annotation-canvas hypothesis before use.
#: This is the historical behaviour and stays the default when the field is absent, so every
#: existing ``extracted.jsonl`` keeps working untouched.
#:
#: ``BOX_SPACE_FRAME`` -- already real pixel coordinates of the decoded frame. Written by
#: ``tools/gate_apply.py``: its boxes come from Grounding DINO, which looked at the frame
#: itself, so there is no canvas in the story at all. Mapping them again would rescale by
#: ``max(W, H) / 768`` a second time -- on a 1920x1080 clip that is a factor of 2.5, which does
#: not fail loudly, it just puts every box in the wrong place.
#:
#: The choice is carried **by the data, not by a CLI flag**, deliberately. A flag can be pointed
#: at the wrong input file; a tag travels with the rows it describes and cannot be mismatched.
BOX_SPACE_ANNOTATION = "annotation"
BOX_SPACE_FRAME = "frame"
BOX_SPACES = (BOX_SPACE_ANNOTATION, BOX_SPACE_FRAME)


def resolve_box(box: Iterable[float], width: int, height: int, box_space: str,
                hypothesis: canvas_module.Hypothesis = DEFAULT_HYPOTHESIS) -> list[float]:
    """Put one box into frame coordinates, dispatching on which space it is already in.

    The dispatch is **total and explicit**: an unrecognised ``box_space`` raises. It is not
    defaulted to either branch, because the two failure modes are asymmetric in the worst way --
    guessing ``annotation`` on frame boxes rescales them, guessing ``frame`` on annotation boxes
    leaves them unmapped, and neither raises on its own. A typo in a manifest writer must stop
    the run, not produce a dataset that looks fine and trains wrong.

    Clamping is unconditional in both branches. Annotation boxes overhang the frame edge
    routinely (14% overflow the canvas on the x axis), and a detector box can sit a fraction of
    a pixel outside the frame from its own coordinate rounding.
    """
    if box_space == BOX_SPACE_ANNOTATION:
        return scale_bbox_to_frame(box, width, height, hypothesis)
    if box_space == BOX_SPACE_FRAME:
        return clamp_to_frame(box, width, height)
    raise ValueError(
        f"unknown box_space {box_space!r}: expected one of {', '.join(BOX_SPACES)}. "
        f"Refusing to guess -- mapping a frame box through the annotation canvas (or failing "
        f"to map an annotation box) silently misplaces every box in the sample."
    )


def clamp_to_frame(box: Iterable[float], width: int, height: int) -> list[float]:
    """Order the corners and clip into ``[0, width] x [0, height]``. No coordinate mapping.

    Split out of :func:`scale_bbox_to_frame` so the frame-space branch of :func:`resolve_box`
    shares exactly the same clamping arithmetic as the annotation branch, rather than a second
    copy of it that could drift.
    """
    x1, y1, x2, y2 = (float(value) for value in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        float(min(max(x1, 0.0), width)),
        float(min(max(y1, 0.0), height)),
        float(min(max(x2, 0.0), width)),
        float(min(max(y2, 0.0), height)),
    ]


def box_is_degenerate(box: Iterable[float], min_side: float = 2.0) -> bool:
    x1, y1, x2, y2 = box
    return (x2 - x1) < min_side or (y2 - y1) < min_side


def pack_masks(masks: np.ndarray) -> np.ndarray:
    """``(F, H, W)`` bool -> ``(F, H, ceil(W/8))`` uint8, matching UltraVid's layout."""
    return np.packbits(np.asarray(masks, dtype=bool), axis=-1)


def unpack_masks(packed: np.ndarray, width: int) -> np.ndarray:
    """Inverse of :func:`pack_masks`; ``width`` is the pre-packbits frame width."""
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1)[..., :width].astype(bool)


def mask_stats(masks: np.ndarray) -> dict[str, Any]:
    """Per-subject aggregate stats in UltraVid's field vocabulary."""
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    visible = np.nonzero(areas)[0]
    frame_area = int(masks.shape[-2]) * int(masks.shape[-1])
    return {
        "visible_frame_count": int(visible.size),
        "mask_frame_count": int(visible.size),
        "first_mask_frame": int(visible[0]) if visible.size else None,
        "last_mask_frame": int(visible[-1]) if visible.size else None,
        "full_clip_covered": bool(visible.size == masks.shape[0]),
        "max_mask_area": int(areas.max()) if areas.size else 0,
        "max_mask_area_ratio": round(float(areas.max()) / frame_area, 6) if areas.size else 0.0,
    }


def merge_directional_masks(
    forward: dict[int, np.ndarray], reverse: dict[int, np.ndarray], frame_count: int,
    height: int, width: int, despeckle: bool = True,
) -> np.ndarray:
    """Stitch the two half-tracks into one ``(frame_count, H, W)`` bool volume.

    The seed frame is produced by both passes; they agree there (it is a conditioning
    frame in each), and forward wins by construction.

    Each frame is despeckled with :func:`largest_components` for the same reason the
    reference cutouts are, but with a consequence specific to the video track: the per-frame
    ``bboxes_xyxy`` written by stage C is the mask's bounding box, and *that box is the
    training condition signal*. A single stray one-pixel speck can therefore drag a box edge
    hundreds of pixels off the subject. Measured over 11,109 visible frames of the 138-sample
    inspection set: 7.9% of frames had a box inflated >15% by speckle (worst case 9.7x, a
    171px-wide subject boxed at 1190px), affecting 46% of samples. Speckle is invisible to
    every area-based metric -- it was ~0.7% of mask area in the case that surfaced this --
    so the bad boxes passed all existing quality gates.

    Safety on the same set: 64.3% of frames are returned unchanged, 35.7% lose only speckle
    (median 0.119% of area), and exactly one frame out of 11,109 lost more than 30% -- a
    581-component herd of cattle, where a multi-blob subject is the ground truth rather than
    an artefact. Pass ``despeckle=False`` to recover the raw SAM2 output.
    """
    masks = np.zeros((frame_count, height, width), dtype=bool)
    for index, mask in reverse.items():
        if 0 <= index < frame_count:
            masks[index] = mask
    for index, mask in forward.items():
        if 0 <= index < frame_count:
            masks[index] = mask
    if despeckle:
        for index in range(frame_count):
            if masks[index].any():
                cleaned = largest_components(masks[index])
                # Never let despeckling empty a frame that had a subject: an empty mask
                # reads downstream as "not visible", which is a different claim entirely.
                if cleaned.any():
                    masks[index] = cleaned
    return masks


def coverage_gaps(forward: dict[int, np.ndarray], reverse: dict[int, np.ndarray],
                  frame_count: int) -> list[int]:
    """Frames no propagation direction wrote. Non-empty means a real tracking bug."""
    seen = set(forward) | set(reverse)
    return [index for index in range(frame_count) if index not in seen]


# --------------------------------------------------------------------------------------
# input schema
# --------------------------------------------------------------------------------------


class SampleRejected(Exception):
    """The sample is unusable for data reasons, not code/IO reasons.

    Kept distinct from a plain failure so the marker records ``rejected``: stage C will
    not retry it on the next pass (MarkerStore treats rejected as terminal), and it is
    not counted as a pipeline error. Degenerate-after-clamp seed boxes and empty
    masklets land here; a missing clip or a malformed manifest row does not.
    """

    def __init__(self, message: str, reasons: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []


def _require(payload: dict[str, Any], keys: tuple[str, ...], where: str) -> Any:
    """First present key among ``keys``; raises with context when all are absent.

    Stage B's exact field naming was still in flux while this stage was written, so a
    small set of aliases is accepted -- but never a silent default, because a missing
    seed box or seed frame would otherwise degrade into a wrong-but-plausible masklet.
    """
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    raise KeyError(f"{where}: missing required field (tried {', '.join(keys)})")


@dataclass(frozen=True)
class SubjectSpec:
    """One subject out of stage B's manifest.

    The ``_768`` suffixes are **misnomers kept for on-disk compatibility**: the annotation
    canvas is unresolved (see :mod:`phantom_data.canvas`), but these are the field names
    stage A wrote into the already-built pilot manifests, so renaming them would break
    stage C's input contract. They mean "raw annotation coordinates", nothing more.
    """

    subject_id: int
    prompt: str
    seed_frame_index: int
    seed_bbox_768: list[float]
    ref_frame: str
    ref_bbox_768: list[float]
    extra: dict[str, Any]
    #: Which coordinate space the two boxes above are in -- see :data:`BOX_SPACE_ANNOTATION`.
    #: Defaults to ``annotation`` so a row from an existing ``extracted.jsonl``, which has no
    #: such field, keeps its historical treatment.
    box_space: str = BOX_SPACE_ANNOTATION


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    video_id: str
    video: str
    frame_count: int
    caption: str
    subjects: tuple[SubjectSpec, ...]


def parse_sample(row: dict[str, Any]) -> SampleSpec:
    sample_id = str(_require(row, ("sample_id",), "sample"))
    where = f"sample {sample_id}"
    subjects: list[SubjectSpec] = []
    raw_subjects = _require(row, ("subjects",), where)
    if not raw_subjects:
        raise ValueError(f"{where}: no subjects")
    # Row-level, not per-subject: a manifest is written by one producer, so mixing the two
    # coordinate spaces within a sample would mean two producers wrote the same row.
    # Absent means ``annotation``; an unknown value is rejected here rather than at the point
    # of use, so the whole run stops on the first bad row instead of per sample.
    box_space = str(row.get("box_space") or BOX_SPACE_ANNOTATION)
    if box_space not in BOX_SPACES:
        raise ValueError(f"{where}: unknown box_space {box_space!r} "
                         f"(expected one of {', '.join(BOX_SPACES)})")
    for position, item in enumerate(raw_subjects):
        sid = int(_require(item, ("subject_id",), f"{where} subject#{position}"))
        scope = f"{where} subject {sid}"
        ref = _require(item, ("ref", "reference"), scope)
        subjects.append(
            SubjectSpec(
                subject_id=sid,
                prompt=str(_require(item, ("phrase", "prompt", "query"), scope)),
                seed_frame_index=int(_require(item, ("seed_frame_index",), scope)),
                seed_bbox_768=[float(v) for v in _require(item, ("seed_bbox_768", "bbox_768"), scope)],
                ref_frame=str(_require(ref, ("frame", "image", "path"), f"{scope} ref")),
                ref_bbox_768=[float(v) for v in _require(ref, ("bbox_768", "bbox"), f"{scope} ref")],
                extra={k: v for k, v in item.items() if k not in {"ref", "reference"}},
                box_space=box_space,
            )
        )
    return SampleSpec(
        sample_id=sample_id,
        video_id=str(_require(row, ("video_id",), where)),
        video=str(_require(row, ("video", "clip"), where)),
        frame_count=int(_require(row, ("frame_count",), where)),
        caption=str(row.get("caption") or row.get("clip_prompt") or ""),
        subjects=tuple(subjects),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------


class Models:
    """Lazily-built SAM2 video/image predictors and CLIP scorer, loaded once."""

    def __init__(self, sam2_config: str, sam2_checkpoint: str, clip_model: str,
                 device: str = "cuda") -> None:
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.clip_model = clip_model
        self.device = device
        self._video = None
        self._image = None
        self._clip = None

    @property
    def video(self):
        if self._video is None:
            from sam2.build_sam import build_sam2_video_predictor

            self._video = build_sam2_video_predictor(
                self.sam2_config, self.sam2_checkpoint, device=self.device
            )
        return self._video

    @property
    def image(self):
        """Image predictor sharing the video predictor's weights.

        ``SAM2VideoPredictor`` subclasses ``SAM2Base``, and ``SAM2ImagePredictor`` only
        touches the image encoder / prompt encoder / mask decoder, so wrapping the same
        instance avoids holding a second ~900MB copy of the backbone.
        """
        if self._image is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._image = SAM2ImagePredictor(self.video)
        return self._image

    @property
    def clip(self):
        if self._clip is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            processor = CLIPProcessor.from_pretrained(self.clip_model, local_files_only=True)
            model = CLIPModel.from_pretrained(self.clip_model, local_files_only=True)
            self._clip = (processor, model.eval().to(self.device), torch)
        return self._clip

    def clip_score(self, image, text: str) -> float:
        processor, model, torch = self.clip
        inputs = processor(text=[text], images=[image], return_tensors="pt",
                           padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            image_features = model.get_image_features(pixel_values=inputs.pixel_values)
            text_features = model.get_text_features(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return round(float((image_features * text_features).sum().item()), 6)


def _autocast(device: str):
    import contextlib

    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------------------
# propagation
# --------------------------------------------------------------------------------------


def _run_direction(predictor, state, frame_index: int, box: np.ndarray,
                   reverse: bool) -> dict[int, np.ndarray]:
    """One clean propagation pass from ``frame_index`` in a single time direction."""
    predictor.reset_state(state)
    predictor.add_new_points_or_box(state, frame_idx=frame_index, obj_id=1, box=box)
    out: dict[int, np.ndarray] = {}
    for index, _object_ids, logits in predictor.propagate_in_video(
        state, start_frame_idx=frame_index, reverse=reverse
    ):
        out[int(index)] = (logits[0] > 0).detach().cpu().numpy().reshape(
            state["video_height"], state["video_width"]
        )
    return out


def propagate_bidirectional(predictor, state, seed_frame_index: int, box: np.ndarray,
                            frame_count: int, device: str = "cuda") -> tuple[np.ndarray, dict[str, Any]]:
    """Track a box seeded at an arbitrary frame across the whole clip.

    Returns the merged ``(frame_count, H, W)`` mask volume plus a diagnostics dict
    recording how many frames each direction contributed.
    """
    height, width = int(state["video_height"]), int(state["video_width"])
    with _autocast(device):
        forward = _run_direction(predictor, state, seed_frame_index, box, reverse=False)
        # ``reverse=True`` is a no-op when the seed is frame 0 (SAM2 returns an empty
        # processing order); the forward pass has already covered the clip in that case.
        reverse = (
            {} if seed_frame_index == 0
            else _run_direction(predictor, state, seed_frame_index, box, reverse=True)
        )
    masks = merge_directional_masks(forward, reverse, frame_count, height, width)
    gaps = coverage_gaps(forward, reverse, frame_count)
    return masks, {
        "seed_frame_index": seed_frame_index,
        "forward_frames": len(forward),
        "reverse_frames": len(reverse),
        "uncovered_frames": gaps,
    }


# --------------------------------------------------------------------------------------
# reference cutout
# --------------------------------------------------------------------------------------


def segment_reference(models: Models, image_rgb: np.ndarray, box: list[float],
                      device: str = "cuda") -> np.ndarray:
    """Box-prompted single-image segmentation of the reference frame.

    ``multimask_output=False`` is SAM2's recommendation for box prompts and is kept: a
    sweep over the pilot set showed selecting by ``iou_predictions.argmax()`` over the
    three multimask tokens is *worse* here (fragmented in 51/94 subjects vs 42/94), and
    SAM2's IoU head is poorly calibrated on these frames (predicted IoU 0.36 for a mask
    the same model scores 0.94 from a centre-point prompt on the identical frame).

    The speckle removal is the substantive step -- see :func:`largest_components`.
    """
    predictor = models.image
    with _autocast(device):
        predictor.set_image(image_rgb)
        masks, _scores, _low_res = predictor.predict(
            box=np.asarray(box, dtype=np.float32)[None, :], multimask_output=False
        )
    mask = np.asarray(masks).reshape(-1, *image_rgb.shape[:2])[0] > 0
    return largest_components(mask)


def write_reference(dataset: Path, sample_id: str, sid: int, image_rgb: np.ndarray,
                    mask: np.ndarray) -> dict[str, Any] | None:
    """Write the white-matted jpg + RGBA png cutout, cropped to the mask box."""
    from PIL import Image

    box = bbox_from_mask(mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    rgb = image_rgb[y1:y2, x1:x2]
    alpha = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
    composite = np.where(alpha[..., None] > 0, rgb, 255).astype(np.uint8)
    rgb_rel = f"object_reference/{sample_id}_subj{sid:02d}.jpg"
    alpha_rel = f"object_reference_alpha/{sample_id}_subj{sid:02d}.png"
    for relative in (rgb_rel, alpha_rel):
        (dataset / relative).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite).save(dataset / rgb_rel, quality=95)
    Image.fromarray(np.dstack((rgb, alpha)), "RGBA").save(dataset / alpha_rel)
    return {
        "object_reference": rgb_rel,
        "object_reference_alpha": alpha_rel,
        "ref_crop_window_xyxy": [x1, y1, x2, y2],
        "ref_mask_coverage": round(float(mask[y1:y2, x1:x2].mean()), 6),
        **component_stats(mask),
        "composite": composite,
    }


# --------------------------------------------------------------------------------------
# per-sample driver
# --------------------------------------------------------------------------------------


def decode_frames(video: Path, directory: Path) -> list[np.ndarray]:
    """Decode to RGB arrays and to the numbered jpg dir SAM2's video loader wants."""
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(video)
    frames: list[np.ndarray] = []
    try:
        for index, frame in enumerate(reader):
            rgb = np.asarray(frame)[..., :3]
            frames.append(rgb)
            Image.fromarray(rgb).save(directory / f"{index:05d}.jpg", quality=95)
    finally:
        reader.close()
    return frames


def segment_sample(spec: SampleSpec, dataset: Path, models: Models,
                   selfcheck: bool = False, device: str = "cuda",
                   selfcheck_subdir: str = STAGE) -> dict[str, Any]:
    """Produce bbox json, masklets npz, reference cutouts and the samples.jsonl row."""
    from PIL import Image

    video_path = dataset / spec.video
    if not video_path.is_file():
        raise FileNotFoundError(f"clip missing: {video_path}")

    with tempfile.TemporaryDirectory(prefix="phantom-sam2-") as directory:
        frame_dir = Path(directory)
        frames = decode_frames(video_path, frame_dir)
        if not frames:
            raise ValueError(f"{spec.sample_id}: decoded 0 frames")
        if len(frames) != spec.frame_count:
            raise ValueError(
                f"{spec.sample_id}: decoded {len(frames)} frames but extracted.jsonl "
                f"declares {spec.frame_count}"
            )
        height, width = frames[0].shape[:2]
        state = models.video.init_state(video_path=str(frame_dir))

        objects: list[dict[str, Any]] = []
        packed_masks: list[np.ndarray] = []
        dropped: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        sheet_payload: list[dict[str, Any]] = []

        for subject in spec.subjects:
            scope = f"{spec.sample_id} subj {subject.subject_id}"
            if not 0 <= subject.seed_frame_index < len(frames):
                dropped.append({"subject_id": subject.subject_id,
                                "reason": "seed_frame_out_of_range",
                                "value": subject.seed_frame_index})
                continue
            seed_box = resolve_box(subject.seed_bbox_768, width, height, subject.box_space)
            if box_is_degenerate(seed_box):
                dropped.append({"subject_id": subject.subject_id,
                                "reason": "degenerate_seed_box", "value": seed_box})
                continue

            masks, diagnostic = propagate_bidirectional(
                models.video, state, subject.seed_frame_index,
                np.asarray(seed_box, dtype=np.float32), len(frames), device=device,
            )
            diagnostic["subject_id"] = subject.subject_id
            diagnostics.append(diagnostic)
            if diagnostic["uncovered_frames"]:
                raise RuntimeError(f"{scope}: propagation left frames uncovered: "
                                   f"{diagnostic['uncovered_frames'][:8]}")
            stats = mask_stats(masks)
            if not stats["visible_frame_count"]:
                dropped.append({"subject_id": subject.subject_id, "reason": "empty_masklet"})
                continue

            ref_path = dataset / subject.ref_frame
            if not ref_path.is_file():
                raise FileNotFoundError(f"{scope}: reference frame missing: {ref_path}")
            ref_image = np.asarray(Image.open(ref_path).convert("RGB"))
            ref_height, ref_width = ref_image.shape[:2]
            # The reference box maps against the *reference frame's* dimensions, not the
            # clip's: the two are different images and routinely different sizes.
            ref_box = resolve_box(subject.ref_bbox_768, ref_width, ref_height,
                                  subject.box_space)
            if box_is_degenerate(ref_box):
                dropped.append({"subject_id": subject.subject_id,
                                "reason": "degenerate_ref_box", "value": ref_box})
                continue
            ref_mask = segment_reference(models, ref_image, ref_box, device=device)
            written = write_reference(dataset, spec.sample_id, subject.subject_id,
                                      ref_image, ref_mask)
            if written is None:
                dropped.append({"subject_id": subject.subject_id, "reason": "empty_ref_mask"})
                continue
            composite = written.pop("composite")

            objects.append({
                **subject.extra,
                "subject_id": subject.subject_id,
                "prompt": subject.prompt,
                "seed_frame_index": subject.seed_frame_index,
                "seed_bbox_xyxy": [round(value, 3) for value in seed_box],
                # Provenance: which space the input boxes arrived in, so a built sample can be
                # traced back to whether it came straight from stage B or through the gate.
                "box_space": subject.box_space,
                "bboxes_xyxy": [bbox_from_mask(mask) for mask in masks],
                **stats,
                **written,
                "ref_frame": subject.ref_frame,
                "ref_bbox_xyxy": [round(value, 3) for value in ref_box],
                "ref_clip_prompt": subject.prompt,
                "ref_clip_score": models.clip_score(Image.fromarray(composite), subject.prompt),
                "ref_clip_model": models.clip_model,
                "propagation": diagnostic,
            })
            packed_masks.append(pack_masks(masks))
            if selfcheck:
                sheet_payload.append({"subject": objects[-1], "masks": masks,
                                      "composite": composite})

        if not objects:
            raise SampleRejected(
                f"{spec.sample_id}: every subject was dropped", dropped
            )

        if selfcheck:
            write_contact_sheet(dataset, spec, frames, sheet_payload,
                                subdir=selfcheck_subdir)

    from ultravid_pipeline.state import atomic_write_json

    mask_rel = f"masklets/{spec.sample_id}.npz"
    bbox_rel = f"bbox/{spec.sample_id}.json"
    (dataset / mask_rel).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset / mask_rel,
        subject_masks_packed=np.asarray(packed_masks, dtype=np.uint8),
        source_subject_ids=np.asarray([item["subject_id"] for item in objects], dtype=np.int64),
        mask_width=np.asarray(width),
        mask_format_version=np.asarray(2),
    )
    payload = {
        "sample_id": spec.sample_id, "video_id": spec.video_id,
        "width": width, "height": height, "frame_count": len(frames),
        "line_width": LINE_WIDTH, "objects": objects,
    }
    if dropped:
        payload["dropped_subjects"] = dropped
    atomic_write_json(dataset / bbox_rel, payload)

    # ``subjects`` intentionally carries the same dicts as bbox ``objects`` minus the
    # bulky per-frame boxes: stage D reads ref_clip_score / visible_frame_count off the
    # samples row, and re-reads bboxes from the bbox json when it needs them.
    slim = [{k: v for k, v in item.items() if k != "bboxes_xyxy"} for item in objects]
    return {
        "status": "built", "sample_id": spec.sample_id, "video_id": spec.video_id,
        "video": spec.video, "bbox": bbox_rel, "masklets": mask_rel,
        "width": width, "height": height, "frame_count": len(frames),
        "clip_prompt": spec.caption, "num_subjects": len(objects),
        "subjects": slim, "dropped_subjects": dropped,
        "propagation": diagnostics,
    }


# --------------------------------------------------------------------------------------
# self-check contact sheet
# --------------------------------------------------------------------------------------


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """1px boundary of a binary mask, via 4-neighbour erosion difference."""
    inner = np.ones_like(mask)
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def write_contact_sheet(dataset: Path, spec: SampleSpec, frames: list[np.ndarray],
                        payload: list[dict[str, Any]], columns: int = 6,
                        cell_width: int = 320, subdir: str = STAGE) -> Path:
    """Grid of sampled frames with boxes + mask outlines, plus the ref cutouts."""
    from PIL import Image, ImageDraw

    colors = [(255, 64, 64), (64, 200, 255), (140, 255, 100), (255, 200, 60),
              (220, 120, 255), (255, 140, 40)]
    count = len(frames)
    seeds = {item["subject"]["seed_frame_index"] for item in payload}
    picks = sorted(seeds | set(np.linspace(0, count - 1, columns, dtype=int).tolist()))

    height, width = frames[0].shape[:2]
    cell_height = max(1, int(round(cell_width * height / width)))
    tiles: list[Image.Image] = []
    for index in picks:
        canvas = np.array(frames[index], dtype=np.uint8)
        for position, item in enumerate(payload):
            color = colors[position % len(colors)]
            outline = mask_outline(item["masks"][index])
            canvas[outline] = color
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        for position, item in enumerate(payload):
            color = colors[position % len(colors)]
            box = bbox_from_mask(item["masks"][index])
            if box is not None:
                draw.rectangle(tuple(box), outline=color, width=3)
        is_seed = index in seeds
        label = f"f{index:03d}" + ("  <== SEED" if is_seed else "")
        draw.rectangle((0, 0, image.width - 1, 18), fill=(0, 0, 0))
        draw.text((4, 4), label, fill=(255, 255, 0) if is_seed else (255, 255, 255))
        if is_seed:
            draw.rectangle((0, 0, image.width - 1, image.height - 1),
                           outline=(255, 255, 0), width=6)
        tiles.append(image.resize((cell_width, cell_height)))

    rows = (len(tiles) + columns - 1) // columns
    ref_height = cell_height
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows + ref_height),
                      (20, 20, 20))
    for position, tile in enumerate(tiles):
        sheet.paste(tile, ((position % columns) * cell_width,
                           (position // columns) * cell_height))
    offset = 0
    for position, item in enumerate(payload):
        cutout = Image.fromarray(item["composite"])
        cutout.thumbnail((cell_width - 8, ref_height - 26))
        sheet.paste(cutout, (offset + 4, rows * cell_height + 22))
        draw = ImageDraw.Draw(sheet)
        subject = item["subject"]
        draw.text((offset + 4, rows * cell_height + 4),
                  f"subj{subject['subject_id']:02d} clip={subject['ref_clip_score']:.3f} "
                  f"vis={subject['visible_frame_count']}/{len(frames)} "
                  f"{subject['prompt'][:40]}",
                  fill=colors[position % len(colors)])
        offset += cell_width

    # Namespaced under _selfcheck/<subdir>/ because other stages write self-check
    # artifacts into _selfcheck/ too.
    target = dataset / "_selfcheck" / subdir / f"{spec.sample_id}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=90)
    return target


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def run(dataset: Path, input_path: Path, models: Models, limit: int | None = None,
        selfcheck_every: int = 0, force: bool = False, device: str = "cuda",
        selfcheck_subdir: str = STAGE, output: str = "segmented.jsonl") -> dict[str, Any]:
    from ultravid_pipeline.state import MarkerStore, atomic_write_text

    markers = MarkerStore(dataset)
    rows = read_jsonl(input_path)
    if limit:
        rows = rows[:limit]

    results: list[dict[str, Any]] = []
    counts = {"passed": 0, "rejected": 0, "failed": 0, "skipped": 0}
    started = time.time()
    for position, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or f"row{position}")
        existing = None if force else markers.get(STAGE, sample_id)
        if existing is not None:
            counts["skipped"] += 1
            if existing.get("sample"):
                results.append(existing["sample"])
            continue
        selfcheck = bool(selfcheck_every) and position % selfcheck_every == 0
        clock = time.time()
        try:
            spec = parse_sample(row)
            sample = segment_sample(spec, dataset, models, selfcheck=selfcheck, device=device,
                                    selfcheck_subdir=selfcheck_subdir)
            sample["elapsed_sec"] = round(time.time() - clock, 3)
            markers.put(STAGE, sample_id, {"status": "passed", "sample": sample})
            results.append(sample)
            counts["passed"] += 1
            print(f"[{position + 1}/{len(rows)}] {sample_id} ok "
                  f"subjects={sample['num_subjects']} {sample['elapsed_sec']}s", flush=True)
        except SampleRejected as error:
            counts["rejected"] += 1
            markers.put(STAGE, sample_id, {
                "status": "rejected", "sample_id": sample_id,
                "error": str(error), "reasons": error.reasons,
            })
            print(f"[{position + 1}/{len(rows)}] {sample_id} REJECTED {error}",
                  file=sys.stderr, flush=True)
        except Exception as error:  # noqa: BLE001 - one bad sample must not stop the run
            counts["failed"] += 1
            markers.put(STAGE, sample_id, {
                "status": "failed", "sample_id": sample_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            print(f"[{position + 1}/{len(rows)}] {sample_id} FAILED "
                  f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)

    atomic_write_text(dataset / output,
                      "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    summary = {
        **counts, "total": len(rows), "elapsed_sec": round(time.time() - started, 2),
        "sec_per_sample": round((time.time() - started) / max(counts["passed"], 1), 2),
        "output": str(dataset / output),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage C: SAM2 masklets + reference cutouts")
    parser.add_argument("--dataset", required=True, type=Path, help="dataset root")
    parser.add_argument("--input", default="extracted.jsonl",
                        help="stage B manifest, relative to --dataset or absolute")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--selfcheck-every", type=int, default=0,
                        help="write a contact sheet every N samples (0 disables)")
    parser.add_argument("--selfcheck-subdir", default=STAGE,
                        help="subdirectory under _selfcheck/ for contact sheets")
    parser.add_argument("--output", default="segmented.jsonl",
                        help="manifest filename written under --dataset")
    parser.add_argument("--force", action="store_true", help="ignore existing markers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-config", default=os.environ.get("SAM2_CONFIG", DEFAULT_SAM2_CONFIG))
    parser.add_argument("--sam2-checkpoint",
                        default=os.environ.get("SAM2_CHECKPOINT", DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--clip-model", default=os.environ.get("CLIP_MODEL", DEFAULT_CLIP_MODEL))
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = dataset / input_path
    if not input_path.is_file():
        parser.error(f"input manifest not found: {input_path}")

    models = Models(args.sam2_config, args.sam2_checkpoint, args.clip_model, device=args.device)
    summary = run(dataset, input_path, models, limit=args.limit,
                  selfcheck_every=args.selfcheck_every, force=args.force, device=args.device,
                  selfcheck_subdir=args.selfcheck_subdir, output=args.output)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
