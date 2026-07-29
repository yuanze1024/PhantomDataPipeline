"""Text-free box refinement: Phantom's box -> SAM2 single-frame mask -> tight box.

The chain this replaces used Grounding DINO to *re-detect* the subject from an LLM-written
phrase, then compared its box to Phantom's by IoU. Three things went wrong with that, all
measured on the 140-subject pilot:

* **The phrase was the single point of failure.** It was both the detector's query and the CLIP
  judge's text, so when the ``_enrich`` cache was missed (140/140 subjects fell back to Phantom's
  bare ``phrase``, median 2 words) two of the three judges degraded together.
* **Long phrases hijack the detector.** ``max``-over-tokens is what
  ``post_process_grounded_object_detection`` reports as confidence, so a phrase mentioning parts
  ("wrinkled face", "expressive eyes") makes Grounding DINO emit separate boxes for the face and
  the eyes, each scoring ~0.31 on its own tokens. The whole-subject box won at 0.69 -- but only
  because it happened to score higher, not because the aggregation prefers subjects to parts.
* **IoU between two fallible boxes cannot establish correctness.** Sometimes Phantom is wrong,
  sometimes the detector is. Measured: identity and IoU are statistically independent on this
  data (Spearman +0.008), so the IoU gate was not filtering what it appeared to filter.

This module drops text entirely. Phantom's box is *semantically* trusted -- it names the right
object -- and only geometrically distrusted, so it is used as a SAM2 prompt rather than as an
answer. SAM2 segments the instance the box points at, and :func:`segment.bbox_from_mask` reads
the tight box off the mask. Nothing here reads a phrase, a caption, or a detector.

The cost argument for doing this inside stage C rather than as a separate pass: the mask is
already being computed. ``segment_reference`` (segment.py:540) runs the identical single-frame
call for the reference cutout at a measured 0.149 s/subject, against 3.3 s for the 81-frame
propagation. Tight boxes are a by-product, not a new expense.

What replaces the IoU gate is *geometric self-consistency* of the mask, in
:func:`tighten_diagnostics`: how much of the prompt box the mask fills, whether the mask
fragmented, whether it runs off the frame edge, and how far the tight box moved. Which of those
become gates rather than recorded numbers is deliberately not decided here -- that needs the
human labels, and hard-coding a threshold now would repeat the mistake this module exists to
undo.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from phantom_data.boxes import clamp_box, iou
from phantom_data.build.segment import bbox_from_mask, component_stats, largest_components

#: A tight box smaller than this fraction of the prompt box means SAM2 latched onto a part
#: (a face inside a person, a logo on a shirt) rather than the instance. Recorded, not enforced.
SUSPECT_SHRINK = 0.25

#: Mask filling less than this share of its own tight box is a thin or scattered shape --
#: a limb, a railing, or leftover speckle that survived component filtering.
SUSPECT_FILL = 0.15

# --------------------------------------------------------------------------------------
# prompt dilation
# --------------------------------------------------------------------------------------
# Measured cause of the "box cuts off an arm" failures, on the reviewer's first 16 labels:
# it is not despeckling and it is not SAM2's ceiling. Growing the prompt box 15% per side and
# re-segmenting enlarged the resulting box by a median 25% (max 83%) on the four subjects the
# reviewer failed, against 2.2% on the twelve they passed. Re-running without despeckling moved
# the failures by 0.6%, which rules that out as the driver.
#
# The mechanism: SAM2 treats a box prompt as a near-hard boundary, so a limb outside Phantom's
# box is learned as background. The model was not blind to it -- it was told not to look.

#: Fraction of each side length added per dilation round.
DILATE_STEP = 0.15

#: How many rounds to allow. Each round is a full ``set_image`` + ``predict``, so this bounds the
#: added cost at 4x the single-pass 0.149 s/side. Three rounds compound to ~1.5x the original box
#: per axis, beyond which a prompt has stopped describing Phantom's annotation.
DILATE_MAX_ROUNDS = 3

#: A mask edge within this many pixels of the prompt edge counts as pinned against it. Not zero:
#: SAM2's mask boundary is computed at 1/4 resolution and upsampled, so an unconstrained mask
#: still lands a pixel or two off the prompt edge.
EDGE_SLACK = 2


def dilate_box(box: list[int], width: int, height: int,
               step: float = DILATE_STEP) -> list[int] | None:
    """Grow ``box`` by ``step`` of its own size on every side, clamped to the frame."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * step, (y2 - y1) * step
    return clamp_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], width, height)


def pinned_edges(tight: list[int], prompt: list[int], width: int, height: int,
                 slack: int = EDGE_SLACK) -> list[str]:
    """Which prompt edges the mask is pressed against -- the signal that it is still clipped.

    An edge only counts when it is *not* the frame boundary. A subject genuinely running off the
    screen pins that edge forever, and treating it as evidence of clipping would dilate every
    edge-cropped subject to the whole frame -- 78 of 140 pilot subjects touch an edge, so this
    distinction is the difference between a working test and a runaway.
    """
    tx1, ty1, tx2, ty2 = tight
    px1, py1, px2, py2 = prompt
    pinned = []
    if tx1 - px1 <= slack and px1 > 0:
        pinned.append("left")
    if ty1 - py1 <= slack and py1 > 0:
        pinned.append("top")
    if px2 - tx2 <= slack and px2 < width:
        pinned.append("right")
    if py2 - ty2 <= slack and py2 < height:
        pinned.append("bottom")
    return pinned


def tighten_box(models, frame_rgb: np.ndarray, box: Any, device: str = "cuda",
                max_rounds: int = DILATE_MAX_ROUNDS,
                step: float = DILATE_STEP) -> tuple[np.ndarray | None, list[int] | None,
                                                    dict[str, Any]]:
    """Segment the instance ``box`` points at, dilating the prompt while the mask stays clipped.

    Delegates each segmentation to :func:`segment.segment_reference` rather than re-implementing
    the predictor call, so the despeckling policy cannot drift between the reference cutout and
    this path -- they are the same operation on the same weights and must stay identical.

    The loop: segment, then ask whether the mask is pressed against any non-frame edge of the
    prompt (:func:`pinned_edges`). Pinned means the subject continues past where Phantom drew the
    line, so the prompt grows and it tries again. Unpinned means SAM2 found the object's own
    boundary inside the prompt, and the box is final. This is per-subject adaptive rather than a
    fixed margin because a fixed 15% is too little for a badly-clipped subject and gratuitous
    risk for an accurate one -- and every extra pixel of prompt is a chance to swallow a
    neighbouring instance, which is a real hazard on this data's crowd scenes.

    Returns ``(mask, tight box, trace)``. ``(None, None, trace)`` when the box is degenerate or
    SAM2 returns nothing: both mean there is no instance to tighten to, and inventing a box would
    ship an unverified coordinate.
    """
    from phantom_data.build.segment import segment_reference

    height, width = frame_rgb.shape[:2]
    prompt = clamp_box(box, width, height)
    trace: dict[str, Any] = {"rounds": 0, "dilated": False, "history": []}
    if prompt is None:
        trace["stop_reason"] = "degenerate prompt"
        return None, None, trace
    if prompt[2] - prompt[0] < 2 or prompt[3] - prompt[1] < 2:
        trace["stop_reason"] = "degenerate prompt"
        return None, None, trace

    original = list(prompt)
    best_mask: np.ndarray | None = None
    best_tight: list[int] | None = None

    for round_index in range(max_rounds + 1):
        mask = segment_reference(models, frame_rgb, prompt, device=device)
        if mask is None or not mask.any():
            # A dilated prompt that yields nothing is a worse answer than the previous round's
            # mask, so keep whatever the tighter prompt produced rather than discarding it.
            trace["stop_reason"] = "empty mask" if best_mask is None else "empty after dilation"
            break
        tight = bbox_from_mask(mask)
        best_mask, best_tight = mask, tight
        pinned = pinned_edges(tight, prompt, width, height)
        trace["rounds"] = round_index
        trace["history"].append({"prompt": list(prompt), "tight": list(tight),
                                 "pinned": list(pinned)})
        if not pinned:
            trace["stop_reason"] = "converged: mask inside prompt on every free edge"
            break
        if round_index == max_rounds:
            trace["stop_reason"] = f"hit max rounds ({max_rounds}) still pinned on {pinned}"
            break
        grown = dilate_box(prompt, width, height, step=step)
        if grown is None or grown == prompt:
            trace["stop_reason"] = "cannot grow further (frame bounds)"
            break
        prompt = grown
        trace["dilated"] = True

    trace["final_prompt"] = list(prompt)
    trace["original_prompt"] = original
    if best_tight is not None:
        trace["prompt_growth_area"] = round(
            max(1, (prompt[2] - prompt[0]) * (prompt[3] - prompt[1]))
            / max(1, (original[2] - original[0]) * (original[3] - original[1])), 4)
    return best_mask, best_tight, trace


def tighten_diagnostics(mask: np.ndarray | None, tight: list[int] | None,
                        prompt: Any, frame_shape: tuple[int, int]) -> dict[str, Any]:
    """Geometric evidence about one tightening, in place of an IoU-vs-annotation score.

    Every field is a property of *this* box and *this* mask -- nothing is compared against a
    second fallible annotation. That is the point: the previous gate's IoU could be low because
    Phantom was wrong, because the detector was wrong, or because the object simply moved, and
    the number could not distinguish those. These can only be low for reasons involving the mask
    itself.

    ``box_shift_iou`` is still an IoU, but a different claim: it measures how far the refinement
    moved the box, which is a *description* of what this pass did, not a judgement of whether
    Phantom was right.
    """
    height, width = frame_shape[:2]
    clamped = clamp_box(prompt, width, height)
    out: dict[str, Any] = {
        "prompt_box": None if clamped is None else list(clamped),
        "tight_box": None if tight is None else list(tight),
        "tightened": bool(tight is not None),
    }
    if mask is None or tight is None or clamped is None:
        out["fail_reason"] = "degenerate prompt box" if clamped is None else "empty mask"
        return out

    px1, py1, px2, py2 = clamped
    tx1, ty1, tx2, ty2 = tight
    prompt_area = max(1, (px2 - px1) * (py2 - py1))
    tight_area = max(0, (tx2 - tx1) * (ty2 - ty1))
    mask_pixels = int(np.count_nonzero(mask))

    out.update({
        # How much of the prompt box the mask actually occupied. Low means Phantom's box was
        # much larger than the instance -- the case this pass exists to fix.
        "mask_share_of_prompt": round(mask_pixels / prompt_area, 6),
        # How densely the mask fills its own tight box. Low means a thin or scattered shape.
        "mask_fill_of_tight": round(mask_pixels / tight_area, 6) if tight_area else 0.0,
        # Area ratio of the refinement. <1 shrank, >1 grew (SAM2 found more of the object than
        # Phantom's box covered, which happens with a partially-boxed subject).
        "area_ratio": round(tight_area / prompt_area, 6),
        "box_shift_iou": (lambda v: None if v is None else round(v, 6))(iou(clamped, tight)),
        "mask_pixels": mask_pixels,
        # Whether the instance is cut off by the frame; a cropped subject is legitimate but its
        # tight box understates the object, which matters when the box becomes conditioning.
        "touches_frame_edge": bool(tx1 <= 0 or ty1 <= 0 or tx2 >= width or ty2 >= height),
        "tight_area_frac_of_frame": round(tight_area / max(1, width * height), 6),
    })
    out.update(component_stats(mask))
    # Flags, not verdicts: they name the two failure shapes worth eyeballing first when
    # reviewing, and the labels will say whether either actually predicts a bad sample.
    out["suspect_shrink"] = bool(out["area_ratio"] < SUSPECT_SHRINK)
    out["suspect_thin"] = bool(out["mask_fill_of_tight"] < SUSPECT_FILL)
    return out


def tighten_subject(models, ref_frame: np.ndarray, ref_box: Any,
                    seed_frame: np.ndarray, seed_box: Any, device: str = "cuda",
                    max_rounds: int = DILATE_MAX_ROUNDS) -> dict[str, Any]:
    """Both sides of one subject. Returns the two tight boxes plus per-side diagnostics.

    Kept as one function because the two sides are always refined together and a subject with
    only one tightened side is not usable: the reference crop and the target box have to describe
    the same instance for the pair to train anything.
    """
    ref_mask, ref_tight, ref_trace = tighten_box(models, ref_frame, ref_box, device=device,
                                                 max_rounds=max_rounds)
    seed_mask, seed_tight, seed_trace = tighten_box(models, seed_frame, seed_box, device=device,
                                                    max_rounds=max_rounds)
    ref_diag = tighten_diagnostics(ref_mask, ref_tight, ref_box, ref_frame.shape[:2])
    seed_diag = tighten_diagnostics(seed_mask, seed_tight, seed_box, seed_frame.shape[:2])
    # The dilation trace rides along with the diagnostics rather than in a separate structure:
    # every question about a box ("why is it bigger than Phantom's?") is answered by the same
    # record, and a reviewer looking at one suspicious subject should not have to join two files.
    ref_diag["dilation"] = ref_trace
    seed_diag["dilation"] = seed_trace
    return {
        "ref": ref_diag,
        "seed": seed_diag,
        "both_tightened": bool(ref_tight is not None and seed_tight is not None),
        "_ref_mask": ref_mask,
        "_seed_mask": seed_mask,
    }
