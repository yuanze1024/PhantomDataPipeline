"""Read-only inspection renderer for a built Phantom-Koala sample.

Answers two questions the stage C contact sheet cannot, because it draws boxes and mask
outlines into the same small tile:

1. **Is the box right?** ``render_target`` draws boxes only -- no mask -- so the annotation
   box can be judged against the object without a mask outline hugging it and hiding the
   error. The Phantom seed box (yellow) and SAM2's per-frame mask box (cyan) are drawn in
   different colours so a good box with bad tracking is distinguishable from the reverse.
2. **Is the mask clean?** ``render_mask`` renders the mask *alone*, with the interior holes
   ``scipy.binary_fill_holes`` would close painted red. Holes are invisible in an outline
   rendering; they are the loudest thing in this one.

Everything here reads stage B/C artefacts already on disk (clip mp4, ref jpg, masklet npz,
bbox json). It never runs SAM2 and never rewrites a stage output, so it can be re-run over
the pilot dataset without touching it.

Hole ratios are reported, not fixed. Some holes are real geometry (a wheel rim, a gap
between railings) and filling them unconditionally would make the mask wrong, so the
threshold is a decision to take after looking at these renders.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import canvas as canvas_module
from .build.segment import bbox_from_mask, scale_bbox_to_frame, unpack_masks

#: Tiles per row in the frame grids. ``render_target`` and ``render_mask`` must lay out the
#: same frames in the same order -- a red hole blob that does not line up with the frame
#: beside it is worse than no rendering at all -- so both take their frame list from
#: :func:`pick_frames`.
GRID_COLUMNS = 6
TILE_WIDTH = 420

BOX_ANNOTATION = (255, 48, 48)      # red: Phantom's annotation box, mapped + clamped
BOX_MASK = (0, 224, 255)            # cyan: bbox of SAM2's mask on this frame
SEED_MARK = (255, 214, 0)           # yellow: seed-tile border + label, never a box
HOLE = (255, 32, 32)                # red: pixels binary_fill_holes would add
FOREGROUND = (245, 245, 245)
BACKGROUND = (48, 48, 48)

#: The annotation-canvas long edge whose overflow we flag. Boxes beyond this are the ~14%
#: that ``scale_bbox_to_frame`` clamps onto the frame edge; see :mod:`phantom_data.canvas`.
CANVAS_LONG_EDGE = canvas_module.CANVAS


# --------------------------------------------------------------------------------------
# pure helpers (unit tested)
# --------------------------------------------------------------------------------------


def pick_frames(frame_count: int, seed_frames: Iterable[int],
                columns: int = GRID_COLUMNS) -> list[int]:
    """Frame indices to render: every seed frame, plus an even spread, sorted unique.

    The seed frames are mandatory -- they are the only frames where the annotation box
    exists, so dropping one would remove the evidence for whether the box is right.
    """
    if frame_count <= 0:
        return []
    picks = {int(index) for index in seed_frames if 0 <= int(index) < frame_count}
    picks.update(int(v) for v in np.linspace(0, frame_count - 1, columns, dtype=int))
    return sorted(picks)


def hole_mask(mask: np.ndarray) -> np.ndarray:
    """Pixels that ``binary_fill_holes`` would turn on: interior background only.

    A hole is background fully enclosed by foreground. Background touching the frame edge
    is not a hole, which is exactly what ``binary_fill_holes`` encodes, so this is its
    difference against the input rather than a hand-rolled flood fill.
    """
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    return ndimage.binary_fill_holes(mask) & ~mask


def hole_ratio(mask: np.ndarray) -> float:
    """Hole pixels / foreground pixels. 0.0 for an empty mask (no foreground to spoil)."""
    mask = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask))
    if not area:
        return 0.0
    return float(np.count_nonzero(hole_mask(mask))) / area


def hole_summary(masks: np.ndarray, frames: Iterable[int] | None = None) -> dict[str, Any]:
    """Per-frame hole ratios reduced to median / p90 / worst, over *visible* frames only.

    Invisible frames (empty mask) have ratio 0 by definition; including them would drag
    the median to 0 for any subject that leaves frame, hiding the holes on the frames where
    the subject is actually present.
    """
    indices = list(range(len(masks))) if frames is None else [int(f) for f in frames]
    pairs = [(index, hole_ratio(masks[index])) for index in indices
             if np.count_nonzero(masks[index])]
    if not pairs:
        return {"frames_measured": 0, "median": 0.0, "p90": 0.0,
                "worst": 0.0, "worst_frame": None}
    ratios = np.array([ratio for _, ratio in pairs], dtype=float)
    worst_index, worst_value = max(pairs, key=lambda item: item[1])
    return {
        "frames_measured": len(pairs),
        "median": round(float(np.median(ratios)), 6),
        "p90": round(float(np.quantile(ratios, 0.9)), 6),
        "worst": round(float(worst_value), 6),
        "worst_frame": int(worst_index),
    }


def clamp_report(raw_box: Iterable[float], width: int, height: int,
                 hypothesis: canvas_module.Hypothesis | None = None) -> dict[str, Any]:
    """How much of an annotation box the frame-bounds clamp ate, and on which side.

    ``scale_bbox_to_frame`` (stage C) both maps and clamps in one step, so the mapped-but-
    unclamped box is not recorded anywhere on disk. Recomputing it here is what makes the
    overflow visible instead of silently absorbed into the frame edge.
    """
    hypothesis = hypothesis or canvas_module.H_768_long
    raw = [float(value) for value in raw_box]
    mapped = canvas_module.map_box(raw, width, height, hypothesis)
    clamped = scale_bbox_to_frame(raw, width, height, hypothesis)
    overflow = {
        "left": round(max(0.0, -mapped[0]), 2),
        "top": round(max(0.0, -mapped[1]), 2),
        "right": round(max(0.0, mapped[2] - width), 2),
        "bottom": round(max(0.0, mapped[3] - height), 2),
    }
    mapped_area = max(0.0, mapped[2] - mapped[0]) * max(0.0, mapped[3] - mapped[1])
    clamped_area = max(0.0, clamped[2] - clamped[0]) * max(0.0, clamped[3] - clamped[1])
    return {
        "raw": [round(value, 2) for value in raw],
        "mapped": [round(value, 2) for value in mapped],
        "clamped": [round(value, 2) for value in clamped],
        "frame": [int(width), int(height)],
        "overflow_px": overflow,
        "clamped_any": any(value > 0.01 for value in overflow.values()),
        "area_lost_pct": (
            0.0 if mapped_area <= 0
            else round(100.0 * (mapped_area - clamped_area) / mapped_area, 2)
        ),
        # The canvas-overflow signature from phantom_data.canvas: y obeys the 768 fit,
        # x does not. Recorded per box so the two axes can be told apart by eye.
        "raw_x2_over_canvas": round(max(0.0, raw[2] - CANVAS_LONG_EDGE), 2),
        "raw_y2_over_canvas": round(max(0.0, raw[3] - CANVAS_LONG_EDGE), 2),
    }


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def discard_partial(out_dir: Path) -> int:
    """Remove a half-written sample directory. Returns how many files were dropped."""
    import shutil

    if not out_dir.is_dir():
        return 0
    count = sum(1 for entry in out_dir.iterdir() if entry.is_file())
    shutil.rmtree(out_dir, ignore_errors=True)
    return count


def load_masks(dataset: Path, masklet_relpath: str) -> tuple[np.ndarray, list[int]]:
    """``(subjects, frames, H, W)`` bool volume + the subject ids it is indexed by."""
    with np.load(dataset / masklet_relpath) as archive:
        packed = np.asarray(archive["subject_masks_packed"], dtype=np.uint8)
        width = int(np.asarray(archive["mask_width"]))
        ids = [int(value) for value in np.asarray(archive["source_subject_ids"])]
    return unpack_masks(packed, width), ids


def decode_frames(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        return [np.asarray(frame)[..., :3] for frame in reader]
    finally:
        reader.close()


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write via a temp file + rename so a full disk cannot leave a truncated artifact.

    A 0-byte ``metrics.json`` is worse than a missing one: the viewer's ``json.loads``
    raises on it and takes down the whole page, and a 0-byte PNG reads as a corrupt image
    rather than as absent. Observed for real when the filesystem filled mid-render.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_save_image(image, path: Path) -> None:
    """PIL save through :func:`atomic_write_bytes` (PIL writes in place otherwise)."""
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    atomic_write_bytes(path, buffer.getvalue())


def _grid(tiles: list, columns: int, background=(20, 20, 20)):
    from PIL import Image

    if not tiles:
        raise ValueError("no tiles to lay out")
    cell_width, cell_height = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), background)
    for position, tile in enumerate(tiles):
        sheet.paste(tile, ((position % columns) * cell_width,
                           (position // columns) * cell_height))
    return sheet


def _label(image, text: str, colour=(255, 255, 255)) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, 20), fill=(0, 0, 0))
    draw.text((4, 5), text, fill=colour)


def render_target(frames: list[np.ndarray], picks: list[int],
                  subjects: list[dict[str, Any]], tile_width: int = TILE_WIDTH,
                  columns: int = GRID_COLUMNS, show_annotation_box: bool = False):
    """Frame grid with the SAM2 mask box (cyan) on every frame; no mask drawn.

    With ``show_annotation_box=True`` Phantom's original annotation box is drawn in **red**
    alongside the cyan detector-derived box, so "did the box actually move" is answerable by
    eye. The two are no longer near-duplicates: under the reordered pipeline the cyan box
    comes from Grounding DINO, and the seed boxes disagree with Phantom's at a median IoU of
    0.87 -- which is the whole reason to show both.

    Red is drawn *under* cyan and thicker, so where they nearly coincide the red still shows
    as a halo instead of being hidden. It is drawn on every frame that has a mask box, not
    only the seed frame: the annotation box is a fixed reference, and seeing the mask box
    drift away from it across the window is the point.

    The mask itself is never drawn here: an outline hugging the object makes any box look
    plausible, which is exactly what the stage C contact sheet could not disentangle.
    """
    from PIL import Image, ImageDraw

    height, width = frames[0].shape[:2]
    tile_height = max(1, int(round(tile_width * height / width)))
    # Tiles are downscaled into the grid, so a hairline box vanishes. Scale the strokes with
    # the shrink factor -- an earlier pass drew width=5 into a 420px tile and the box was
    # invisible, which read as "the box was never drawn".
    shrink = max(1.0, width / float(tile_width))
    mask_stroke = max(3, int(round(3 * shrink)))
    annotation_stroke = max(5, int(round(6 * shrink)))
    tiles = []
    for index in picks:
        image = Image.fromarray(np.asarray(frames[index], dtype=np.uint8))
        draw = ImageDraw.Draw(image)
        seeds_here = []
        for subject in subjects:
            box = subject["mask_boxes"].get(index)
            phantom = subject.get("phantom_box")
            if show_annotation_box and phantom and box is not None:
                draw.rectangle(tuple(float(v) for v in phantom),
                               outline=BOX_ANNOTATION, width=annotation_stroke)
            if box is not None:
                draw.rectangle(tuple(box), outline=BOX_MASK, width=mask_stroke)
            if int(subject["seed_frame_index"]) == index:
                seeds_here.append(subject["subject_id"])
        label = f"f{index:03d}"
        if seeds_here:
            label += f"  SEED subj{','.join(str(v) for v in seeds_here)}"
        if show_annotation_box:
            label += "   RED=phantom  CYAN=grounding-dino"
        # The seed tile's border and label use their own colour: they used to reuse the
        # annotation colour, which now means "Phantom's box" and would read as a box.
        _label(image, label, colour=SEED_MARK if seeds_here else (255, 255, 255))
        if seeds_here:
            draw.rectangle((0, 0, image.width - 1, image.height - 1),
                           outline=SEED_MARK, width=max(6, int(round(6 * shrink))))
        tiles.append(image.resize((tile_width, tile_height)))
    return _grid(tiles, columns)


def mask_panel(mask: np.ndarray) -> np.ndarray:
    """Three-colour RGB view of one mask: foreground, holes (red), background."""
    mask = np.asarray(mask, dtype=bool)
    panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
    panel[...] = BACKGROUND
    panel[mask] = FOREGROUND
    panel[hole_mask(mask)] = HOLE
    return panel


def render_mask(masks_by_subject: list[dict[str, Any]], picks: list[int],
                frame_shape: tuple[int, int], ref_alphas: list[dict[str, Any]] | None = None,
                tile_width: int = TILE_WIDTH, columns: int = GRID_COLUMNS):
    """Mask-only grid over the SAME ``picks`` as :func:`render_target`, holes in red.

    Ref-frame alphas are appended as extra tiles because the ref branch is the one with no
    hole filling at all (``SAM2ImagePredictor`` defaults ``max_hole_area=0``, while the
    video predictor is built with ``fill_hole_area=8``), so its holes are the ones most
    likely to be a code artefact rather than real geometry.
    """
    from PIL import Image

    height, width = frame_shape
    tile_height = max(1, int(round(tile_width * height / width)))
    tiles = []
    for index in picks:
        combined = np.zeros((height, width), dtype=bool)
        for subject in masks_by_subject:
            combined |= subject["masks"][index]
        image = Image.fromarray(mask_panel(combined))
        ratio = hole_ratio(combined)
        _label(image, f"f{index:03d}  holes {100 * ratio:.2f}%",
               colour=HOLE if ratio > 0.02 else (255, 255, 255))
        tiles.append(image.resize((tile_width, tile_height)))
    for reference in ref_alphas or []:
        alpha = np.asarray(reference["alpha"], dtype=bool)
        image = Image.fromarray(mask_panel(alpha))
        image.thumbnail((tile_width, tile_height))
        padded = Image.new("RGB", (tile_width, tile_height), (20, 20, 20))
        padded.paste(image, ((tile_width - image.width) // 2,
                             (tile_height - image.height) // 2))
        ratio = hole_ratio(alpha)
        _label(padded, f"REF subj{reference['subject_id']:02d}  holes {100 * ratio:.2f}%",
               colour=HOLE if ratio > 0.02 else (255, 255, 255))
        tiles.append(padded)
    return _grid(tiles, columns)


def render_reference(references: list[dict[str, Any]], tile_width: int = TILE_WIDTH * 2):
    """Per subject: the FULL ref frame with both boxes, beside the white-matte cutout.

    The whole frame matters here. The stored ``object_reference`` jpg is already cropped to
    the mask, so it cannot show whether the box landed on the right object -- only what was
    inside it once the decision was made.

    ``reference["box"]`` is the box the crop was actually taken with -- Grounding DINO's under
    the reordered pipeline -- so it is drawn cyan, matching the target sheet. Phantom's own ref
    box is drawn red underneath when available. They are not near-duplicates: on 8% of subjects
    the two ref boxes do not overlap at all, which is exactly what this sheet has to expose.
    """
    from PIL import Image, ImageDraw

    rows = []
    for reference in references:
        frame = np.asarray(reference["frame"], dtype=np.uint8)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        # Strokes are scaled like the target sheet's: this frame gets downscaled into the
        # row, and a hairline box disappears in the resize.
        shrink = max(1.0, image.width / float(tile_width))
        phantom = reference.get("phantom_box")
        if phantom:
            draw.rectangle(tuple(float(v) for v in phantom),
                           outline=BOX_ANNOTATION, width=max(5, int(round(6 * shrink))))
        draw.rectangle(tuple(round(v) for v in reference["box"]),
                       outline=BOX_MASK, width=max(4, int(round(5 * shrink))))
        label = (f"subj{reference['subject_id']:02d}  ref frame {image.width}x"
                 f"{image.height}  '{reference['prompt'][:48]}'")
        if phantom:
            label += "   RED=phantom  CYAN=grounding-dino"
        _label(image, label, colour=SEED_MARK)
        scale = tile_width / image.width
        image = image.resize((tile_width, max(1, int(round(image.height * scale)))))

        cutout = Image.fromarray(np.asarray(reference["cutout"], dtype=np.uint8))
        cutout.thumbnail((image.height, image.height))
        row = Image.new("RGB", (image.width + image.height, image.height), (20, 20, 20))
        row.paste(image, (0, 0))
        row.paste(cutout, (image.width + (image.height - cutout.width) // 2,
                           (image.height - cutout.height) // 2))
        rows.append(row)

    total_width = max(row.width for row in rows)
    sheet = Image.new("RGB", (total_width, sum(row.height for row in rows)), (20, 20, 20))
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset))
        offset += row.height
    return sheet


# --------------------------------------------------------------------------------------
# per-sample driver
# --------------------------------------------------------------------------------------


def inspect_sample(dataset: Path, sample: dict[str, Any], extracted: dict[str, Any],
                   out_dir: Path, columns: int = GRID_COLUMNS,
                   show_annotation_box: bool = False,
                   phantom_boxes: dict[int, list[float]] | None = None,
                   phantom_ref_boxes: dict[int, list[float]] | None = None) -> dict[str, Any]:
    """Render the three sheets for one sample and return its metrics dict.

    ``phantom_boxes`` / ``phantom_ref_boxes`` map subject id to Phantom's original annotation
    box on the seed frame and on the ref frame, both in frame coordinates, for the
    annotation-vs-detector overlay. Supplied by the caller from a stage-2 report because the
    bbox JSON no longer carries them -- it holds only the box SAM2 was seeded with. The two
    sides are separate maps because the ref frame is a different frame with its own
    dimensions, so a seed box drawn on it would land in the wrong place.
    """
    from PIL import Image

    sample_id = sample["sample_id"]
    bbox_payload = json.loads(
        (dataset / sample["bbox"]).read_text(encoding="utf-8"))
    masks, mask_ids = load_masks(dataset, sample["masklets"])
    frames = decode_frames(dataset / sample["video"])
    if len(frames) != masks.shape[1]:
        raise ValueError(f"{sample_id}: {len(frames)} frames but masklet has "
                         f"{masks.shape[1]}")
    height, width = frames[0].shape[:2]
    extracted_subjects = {int(item["subject_id"]): item
                          for item in extracted.get("subjects") or []}

    subjects: list[dict[str, Any]] = []
    for entry in bbox_payload["objects"]:
        sid = int(entry["subject_id"])
        volume = masks[mask_ids.index(sid)]
        boxes = {index: box for index, box in enumerate(entry["bboxes_xyxy"])
                 if box is not None}
        subjects.append({
            "subject_id": sid,
            "prompt": entry.get("prompt"),
            "seed_frame_index": int(entry["seed_frame_index"]),
            "seed_box": entry["seed_bbox_xyxy"],
            # Phantom's original annotation box, for the side-by-side. It is deliberately not
            # in the bbox JSON: under the reordered pipeline that file carries only the box
            # SAM2 was actually seeded with (``box_space: frame`` = Grounding DINO's), so the
            # annotation has to come from the stage-2 report or not at all. None when no
            # report is passed, which is the normal case for a plain render.
            "phantom_box": (phantom_boxes or {}).get(sid),
            "mask_boxes": boxes,
            "masks": volume,
            "entry": entry,
            "spec": extracted_subjects.get(sid) or {},
        })

    picks = pick_frames(len(frames), [s["seed_frame_index"] for s in subjects],
                        columns=columns)

    references: list[dict[str, Any]] = []
    ref_alphas: list[dict[str, Any]] = []
    for subject in subjects:
        entry = subject["entry"]
        ref_frame = np.asarray(
            Image.open(dataset / entry["ref_frame"]).convert("RGB"))
        cutout = np.asarray(
            Image.open(dataset / entry["object_reference"]).convert("RGB"))
        alpha_image = Image.open(dataset / entry["object_reference_alpha"])
        alpha = np.asarray(alpha_image.split()[-1]) > 0
        references.append({
            "subject_id": subject["subject_id"],
            "prompt": subject["prompt"] or "",
            "frame": ref_frame,
            "box": entry["ref_bbox_xyxy"],
            # Phantom's ref box, from the stage-2 report; None without --gate-report. Kept
            # separate from ``box`` because that one is whatever the crop was taken with.
            "phantom_box": (phantom_ref_boxes or {}).get(int(subject["subject_id"]))
                           if show_annotation_box else None,
            "cutout": cutout,
        })
        ref_alphas.append({"subject_id": subject["subject_id"], "alpha": alpha})

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_image(render_target(frames, picks, subjects, columns=columns,
                                    show_annotation_box=show_annotation_box),
                      out_dir / "target.png")
    atomic_save_image(render_mask(subjects, picks, (height, width),
                                  ref_alphas=ref_alphas, columns=columns),
                      out_dir / "mask.png")
    atomic_save_image(render_reference(references), out_dir / "reference.png")

    metrics = {
        "sample_id": sample_id,
        "video_id": sample.get("video_id"),
        "phantom_video_id": extracted.get("phantom_video_id"),
        "resolution": [width, height],
        "frame_count": len(frames),
        "caption": sample.get("clip_prompt") or extracted.get("caption"),
        "frames_rendered": picks,
        "source": extracted.get("source") or {},
        "subjects": [],
    }
    for subject, reference in zip(subjects, ref_alphas):
        entry = subject["entry"]
        spec = subject["spec"]
        raw_seed = spec.get("seed_bbox_768")
        raw_ref = ((spec.get("ref") or {}).get("bbox_768"))
        ref_dimensions = (
            int((spec.get("ref") or {}).get("ref_frame_width") or 0),
            int((spec.get("ref") or {}).get("ref_frame_height") or 0),
        )
        metrics["subjects"].append({
            "subject_id": subject["subject_id"],
            "prompt": subject["prompt"],
            "seed_frame_index": subject["seed_frame_index"],
            "ref_clip_score": entry.get("ref_clip_score"),
            "visible_frame_count": entry.get("visible_frame_count"),
            "max_mask_area_ratio": entry.get("max_mask_area_ratio"),
            "ref_mask_coverage": entry.get("ref_mask_coverage"),
            "ref_mask_components": entry.get("ref_mask_components"),
            "ref_mask_largest_share": entry.get("ref_mask_largest_share"),
            "holes_video": hole_summary(subject["masks"]),
            "holes_video_rendered_frames": hole_summary(subject["masks"], picks),
            "holes_ref_alpha": round(hole_ratio(reference["alpha"]), 6),
            "seed_box_clamp": (
                None if raw_seed is None else clamp_report(raw_seed, width, height)),
            "ref_box_clamp": (
                None if raw_ref is None or not all(ref_dimensions)
                else clamp_report(raw_ref, *ref_dimensions)),
            "ref_time_sec": (spec.get("ref") or {}).get("abs_time"),
            "ref_phantom_vid": (spec.get("ref") or {}).get("phantom_vid"),
        })
    # metrics.json is written last and is the marker of a complete sample: the viewer skips
    # any directory without it, so a partial render is invisible rather than fatal.
    atomic_write_bytes(
        out_dir / "metrics.json",
        (json.dumps(metrics, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return metrics


def read_phantom_boxes(path: Path, field: str = "box_seed_phantom",
                       ) -> dict[str, dict[int, list[float]]]:
    """``{sample_id: {subject_id: phantom box}}`` from a stage-2 ``gate_report.json``.

    The bbox JSON carries only the box SAM2 was seeded with, so the annotation box for the
    overlay has to be recovered from the report that chose between them. Both
    ``box_seed_phantom`` and ``box_ref_phantom`` are already mapped into their own frame's
    coordinates and clamped -- the same space as ``bboxes_xyxy`` / ``ref_bbox_xyxy`` -- so
    either can be drawn on the matching frame without conversion. Pass ``field`` to pick a
    side; they must not be interchanged, since the ref frame has its own dimensions.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    boxes: dict[str, dict[int, list[float]]] = {}
    for subject in report.get("subjects") or []:
        box = subject.get(field)
        if box:
            boxes.setdefault(str(subject["sample_id"]), {})[int(subject["subject_id"])] = box
    return boxes


def run(dataset: Path, manifest: str = "segmented.jsonl", out_root: str = "_inspect",
        limit: int | None = None, only: list[str] | None = None,
        columns: int = GRID_COLUMNS, show_annotation_box: bool = False,
        gate_report: Path | None = None) -> dict[str, Any]:
    samples = read_jsonl(dataset / manifest)
    extracted = {row["sample_id"]: row
                 for row in read_jsonl(dataset / "extracted.jsonl")}
    phantom_boxes = read_phantom_boxes(gate_report) if gate_report else {}
    phantom_ref_boxes = (read_phantom_boxes(gate_report, "box_ref_phantom")
                         if gate_report else {})
    if show_annotation_box and not phantom_boxes:
        print("warning: --show-annotation-box without --gate-report; Phantom's box is not "
              "in the bbox JSON, so no annotation box can be drawn", flush=True)
    if only:
        wanted = set(only)
        samples = [row for row in samples if row["sample_id"] in wanted]
    if limit:
        samples = samples[:limit]

    root = dataset / out_root
    done: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        try:
            metrics = inspect_sample(
                dataset, sample, extracted.get(sample_id) or {},
                root / sample_id, columns=columns,
                show_annotation_box=show_annotation_box,
                phantom_boxes=phantom_boxes.get(sample_id),
                phantom_ref_boxes=phantom_ref_boxes.get(sample_id))
            done.append(metrics)
            worst = max((s["holes_video"]["worst"] for s in metrics["subjects"]),
                        default=0.0)
            worst_ref = max((s["holes_ref_alpha"] for s in metrics["subjects"]),
                            default=0.0)
            print(f"[{position}/{len(samples)}] {sample_id} ok  "
                  f"worst_video_hole={100 * worst:.2f}%  "
                  f"worst_ref_hole={100 * worst_ref:.2f}%", flush=True)
        except Exception as error:  # noqa: BLE001 - one bad sample must not stop the run
            failures.append({"sample_id": sample_id,
                             "error": f"{type(error).__name__}: {error}"})
            # Drop whatever this sample managed to write. Half a sheet set is not useful for
            # inspection and a stale sheet next to fresh ones would be actively misleading.
            leftover = discard_partial(root / sample_id)
            print(f"[{position}/{len(samples)}] {sample_id} FAILED "
                  f"{type(error).__name__}: {error}"
                  + (f" (discarded {leftover} partial file(s))" if leftover else ""),
                  flush=True)

    summary = {
        "dataset": str(dataset),
        "samples": len(samples),
        "rendered": len(done),
        "failed": len(failures),
        "failures": failures,
        "out_root": str(root),
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        root / "summary.json",
        (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render inspection sheets (boxes / masks+holes / reference) per sample")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", default="segmented.jsonl")
    parser.add_argument("--out-root", default="_inspect")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these sample ids")
    parser.add_argument("--columns", type=int, default=GRID_COLUMNS)
    parser.add_argument("--show-annotation-box", action="store_true",
                        help="also draw Phantom's annotation box in red next to the cyan "
                             "detector box; needs --gate-report to know where it was")
    parser.add_argument("--gate-report", type=Path, default=None,
                        help="stage-2 gate_report.json to read Phantom's boxes from "
                             "(they are not in the bbox JSON)")
    args = parser.parse_args(argv)
    summary = run(args.dataset.resolve(), manifest=args.manifest, out_root=args.out_root,
                  limit=args.limit, only=args.only, columns=args.columns,
                  show_annotation_box=args.show_annotation_box,
                  gate_report=args.gate_report)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
