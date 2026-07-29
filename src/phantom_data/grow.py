"""Grow a mask past the box that seeded it, by feeding it back to SAM2 without the box.

The problem this solves, in the reviewer's words: after ID-Sim picks the best candidate box, that
box may still cover only part of the instance -- so the mask derived from it is partial, and the
tight box inherits the truncation. Measured on the pilot: growing the *prompt box* and
re-segmenting made things worse (81% of boxes enlarged, median identity 0.659 -> 0.408), because a
wider box invites SAM2 to annex background. The mistake there was arguing with SAM2 in the
language of boxes.

``SAM2ImagePredictor.predict`` takes a ``mask_input`` -- "a low resolution mask input to the
model, typically coming from a previous prediction iteration" -- and every prompt is optional. So
the second round can pass **the mask alone, with no box at all**. SAM2 is then unconstrained: it
refines the region it was handed, and the refinement is free to extend past the original box
because no box is telling it where the object ends. That is the difference between this and
prompt dilation: the evidence handed back is *the object's own shape*, not a bigger rectangle.

Convergence is on mask area, not on box edges. The dilation attempt failed partly because
"mask touches the prompt edge" is both the symptom of clipping and the normal state of a mask
that fills its box, so it could not tell them apart -- a positive feedback loop dressed as a
convergence test. Area change has no such ambiguity: the mask either stopped growing or it did
not, and a runaway is visible as growth that never settles rather than as a plausible-looking
"converged".
"""
from __future__ import annotations

from typing import Any

import numpy as np

from phantom_data.boxes import clamp_box
from phantom_data.build.segment import bbox_from_mask, largest_components

#: Stop once a round changes the mask area by less than this fraction. 2% is below the
#: frame-to-frame jitter of SAM2's own boundary, so a smaller value would spend rounds chasing
#: noise.
AREA_TOLERANCE = 0.02

#: Rounds of feedback after the initial box-prompted pass. Each is one predict() on an image whose
#: embedding is already computed, so the marginal cost is small next to set_image.
MAX_GROW_ROUNDS = 3

#: Refuse a round that inflates the mask beyond this multiple of the previous one. A single round
#: that doubles the mask has almost certainly leaked into the background or a neighbouring
#: instance rather than found a limb, and keeping the previous mask is the safe answer.
RUNAWAY_RATIO = 2.0

#: Foreground points sampled from the current mask and passed alongside it. The mask alone is a
#: weak prompt -- ``mask_input`` is 256x256 and lossy -- and a handful of interior points anchors
#: the identity of the thing being refined so a round cannot silently drift to another object.
GROW_POINTS = 4


def _sample_interior_points(mask: np.ndarray, count: int = GROW_POINTS) -> np.ndarray | None:
    """Points spread across the mask, biased away from its boundary.

    Deterministic: uses a fixed stride through the sorted foreground pixels rather than random
    sampling, so a rerun of the pipeline produces the same boxes. Non-determinism here would make
    every downstream comparison (v1 vs v2, before vs after a fix) untrustworthy.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    if len(xs) <= count:
        return np.stack([xs, ys], axis=1).astype(np.float32)
    # Order by distance from the mask centroid so the stride samples the body, not the fringe.
    cx, cy = float(xs.mean()), float(ys.mean())
    order = np.argsort((xs - cx) ** 2 + (ys - cy) ** 2)
    picked = order[:: max(1, len(order) // (count * 2))][:count]
    return np.stack([xs[picked], ys[picked]], axis=1).astype(np.float32)


def grow_mask(models, frame_rgb: np.ndarray, box: Any, device: str = "cuda",
              max_rounds: int = MAX_GROW_ROUNDS,
              tolerance: float = AREA_TOLERANCE) -> tuple[np.ndarray | None,
                                                          list[int] | None,
                                                          dict[str, Any]]:
    """Segment ``box``, then let SAM2 refine its own mask with the box removed.

    Round 0 is an ordinary box-prompted segmentation -- the box is what identifies *which*
    instance is wanted, so it has to be there once. Rounds 1..n pass the previous round's low-res
    logits as ``mask_input`` plus a few interior points, and **no box**, which is what allows the
    mask to exceed the original rectangle.

    Returns ``(mask, tight box, trace)``. The trace records every round's area and box so a
    suspicious result can be attributed to a specific round instead of taken on trust.
    """
    from phantom_data.build.segment import _autocast

    height, width = frame_rgb.shape[:2]
    prompt = clamp_box(box, width, height)
    trace: dict[str, Any] = {"rounds": 0, "grew": False, "history": []}
    if prompt is None or prompt[2] - prompt[0] < 2 or prompt[3] - prompt[1] < 2:
        trace["stop_reason"] = "degenerate box"
        return None, None, trace

    predictor = models.image
    best_mask: np.ndarray | None = None
    best_tight: list[int] | None = None
    low_res: np.ndarray | None = None

    with _autocast(device):
        predictor.set_image(frame_rgb)
        for round_index in range(max_rounds + 1):
            if round_index == 0:
                masks, _scores, low = predictor.predict(
                    box=np.asarray(prompt, dtype=np.float32)[None, :],
                    multimask_output=False)
            else:
                points = _sample_interior_points(best_mask)
                if points is None:
                    trace["stop_reason"] = "no interior points to anchor the refinement"
                    break
                # No box: this is the whole mechanism. SAM2 refines the region described by the
                # mask and the points, and nothing constrains it to the original rectangle.
                masks, _scores, low = predictor.predict(
                    point_coords=points,
                    point_labels=np.ones(len(points), dtype=np.int32),
                    mask_input=low_res,
                    box=None,
                    multimask_output=False)

            raw = np.asarray(masks).reshape(-1, height, width)[0] > 0
            if not raw.any():
                trace["stop_reason"] = ("empty mask" if best_mask is None
                                        else "empty after refinement")
                break
            mask = largest_components(raw)
            if not mask.any():
                trace["stop_reason"] = "despeckling emptied the mask"
                break

            area = int(mask.sum())
            previous = int(best_mask.sum()) if best_mask is not None else 0
            if previous and area > previous * RUNAWAY_RATIO:
                # Leaked into the background or a neighbour. Keep the previous round.
                trace["stop_reason"] = (f"runaway: round {round_index} inflated the mask "
                                        f"{area / previous:.1f}x; kept the previous one")
                break

            best_mask, best_tight = mask, bbox_from_mask(mask)
            low_res = np.asarray(low).reshape(-1, *np.asarray(low).shape[-2:])[:1]
            trace["rounds"] = round_index
            trace["history"].append({"round": round_index, "area": area,
                                     "box": list(best_tight)})

            if previous and abs(area - previous) <= previous * tolerance:
                trace["stop_reason"] = f"converged: area changed <{tolerance:.0%}"
                break
            if round_index > 0:
                trace["grew"] = trace["grew"] or area > previous
            if round_index == max_rounds:
                trace["stop_reason"] = f"hit max rounds ({max_rounds})"

    if best_tight is not None:
        seed_area = max(1, (prompt[2] - prompt[0]) * (prompt[3] - prompt[1]))
        final_area = max(1, (best_tight[2] - best_tight[0]) * (best_tight[3] - best_tight[1]))
        trace["box_vs_seed_area"] = round(final_area / seed_area, 4)
        trace["escaped_seed_box"] = bool(
            best_tight[0] < prompt[0] or best_tight[1] < prompt[1]
            or best_tight[2] > prompt[2] or best_tight[3] > prompt[3])
        trace["seed_box"] = list(prompt)
    trace.setdefault("stop_reason", "completed")
    return best_mask, best_tight, trace
