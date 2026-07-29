"""Tests for mask-feedback growth (:mod:`phantom_data.grow`).

Growth is a feedback loop over a segmentation model, so the interesting properties are the safety
ones. The previous attempt at this problem -- dilating the *prompt box* -- failed by running away:
it enlarged 81% of boxes and dropped median identity from 0.659 to 0.408 while reporting
"converged", because its stop test could not distinguish a mask filling its box from a mask being
clipped by it. These tests pin down the properties that failure lacked: convergence on area,
refusal of a round that inflates the mask, and a trace that says which round did what.
"""
from __future__ import annotations

import numpy as np

from phantom_data import grow


class StubPredictor:
    """Returns a scripted sequence of masks, recording whether each call passed a box.

    The box/no-box distinction is the mechanism under test: round 0 must prompt with the box (it
    identifies which instance is wanted), and later rounds must not (that is what lets the mask
    exceed it).
    """

    def __init__(self, masks, frame_shape=(120, 160)):
        self.masks = masks
        self.frame_shape = frame_shape
        self.calls = []

    def set_image(self, image):
        self.calls.append({"op": "set_image"})

    def predict(self, box=None, point_coords=None, point_labels=None, mask_input=None,
                multimask_output=False):
        index = len([c for c in self.calls if c["op"] == "predict"])
        self.calls.append({"op": "predict", "box_given": box is not None,
                           "mask_input_given": mask_input is not None,
                           "points_given": point_coords is not None})
        mask = self.masks[min(index, len(self.masks) - 1)]
        return mask[None, ...], np.array([0.9]), np.zeros((1, 256, 256), dtype=np.float32)


class StubModels:
    def __init__(self, predictor):
        self.image = predictor
        self.device = "cpu"


def rect(shape, x1, y1, x2, y2):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def run(masks, box, frame_shape=(120, 160), **kwargs):
    predictor = StubPredictor(masks, frame_shape)
    frame = np.zeros((*frame_shape, 3), dtype=np.uint8)
    mask, tight, trace = grow.grow_mask(StubModels(predictor), frame, box,
                                        device="cpu", **kwargs)
    return mask, tight, trace, predictor


def test_round_zero_uses_the_box_and_later_rounds_do_not():
    shape = (120, 160)
    masks = [rect(shape, 20, 20, 60, 60), rect(shape, 20, 20, 80, 60),
             rect(shape, 20, 20, 81, 60)]
    _mask, _tight, _trace, predictor = run(masks, [20, 20, 60, 60])
    predicts = [c for c in predictor.calls if c["op"] == "predict"]
    assert predicts[0]["box_given"] is True, "the box identifies which instance is wanted"
    assert all(not c["box_given"] for c in predicts[1:]), "refinement must be unconstrained"
    assert all(c["mask_input_given"] and c["points_given"] for c in predicts[1:])


def test_the_mask_is_allowed_to_grow_past_the_seed_box():
    # The entire point. The seeded box ends at x=60; the object continues to x=90, which is 1.75x
    # the seeded area -- under the runaway ceiling, so it is accepted.
    shape = (120, 160)
    masks = [rect(shape, 20, 20, 60, 60), rect(shape, 20, 20, 90, 60),
             rect(shape, 20, 20, 91, 60)]
    _mask, tight, trace, _predictor = run(masks, [20, 20, 60, 60])
    assert tight[2] > 60
    assert trace["escaped_seed_box"] is True
    assert trace["box_vs_seed_area"] > 1.0


def test_recovery_beyond_the_runaway_ratio_is_refused_in_one_round():
    """A limb worth more than RUNAWAY_RATIO of the mask cannot be recovered in a single round.

    Worth pinning down because it is a real limit, not a bug: a mask that more than doubles in one
    step is indistinguishable from a leak into the background, and the guard exists because the
    previous approach to this problem failed exactly that way. Multi-round growth still reaches
    such an object -- each round may add up to the ratio -- so the ceiling is per round, not
    absolute. If the reviewer finds genuinely halved subjects being refused, RUNAWAY_RATIO is the
    knob, and the ``runaway`` stop_reason makes those subjects findable in the report.
    """
    shape = (120, 160)
    masks = [rect(shape, 20, 20, 60, 60), rect(shape, 20, 20, 120, 60)]  # 3.0x in one step
    _mask, tight, trace, _predictor = run(masks, [20, 20, 60, 60])
    assert tight == [20, 20, 60, 60]
    assert "runaway" in trace["stop_reason"]

    # Reached in two smaller steps instead: 1.75x then 1.71x, neither over the ceiling.
    stepped = [rect(shape, 20, 20, 60, 60), rect(shape, 20, 20, 90, 60),
               rect(shape, 20, 20, 120, 60), rect(shape, 20, 20, 121, 60)]
    _mask, tight2, trace2, _p = run(stepped, [20, 20, 60, 60], max_rounds=3)
    # 121, not 120: the final round's 1-pixel change is inside AREA_TOLERANCE, so it is accepted
    # as the converged answer rather than discarded.
    assert tight2[2] >= 120, "incremental growth reaches the same place"
    assert "runaway" not in trace2["stop_reason"]
    assert "converged" in trace2["stop_reason"]


def test_growth_stops_when_the_area_settles():
    shape = (120, 160)
    stable = rect(shape, 20, 20, 60, 60)
    _mask, _tight, trace, predictor = run([stable, stable, stable], [20, 20, 60, 60])
    assert "converged" in trace["stop_reason"]
    # Two predicts: the box-prompted pass, then one refinement that changed nothing.
    assert len([c for c in predictor.calls if c["op"] == "predict"]) == 2


def test_a_round_that_doubles_the_mask_is_refused_and_the_previous_one_kept():
    # A mask that suddenly doubles has leaked into the background or a neighbouring instance --
    # exactly the failure that sank prompt dilation. The safe answer is the previous round.
    shape = (120, 160)
    good = rect(shape, 20, 20, 60, 60)
    leaked = rect(shape, 0, 0, 160, 120)
    _mask, tight, trace, _predictor = run([good, leaked], [20, 20, 60, 60])
    assert tight == [20, 20, 60, 60], "the pre-leak box is what ships"
    assert "runaway" in trace["stop_reason"]


def test_a_degenerate_box_is_declined_without_touching_the_model():
    for bad in (None, [10, 10, 10, 10], "not a box"):
        _mask, tight, trace, predictor = run([rect((120, 160), 0, 0, 10, 10)], bad)
        assert tight is None
        assert trace["stop_reason"] == "degenerate box"
        assert predictor.calls == []


def test_an_empty_first_mask_yields_no_box():
    _mask, tight, trace, _predictor = run([np.zeros((120, 160), dtype=bool)],
                                          [20, 20, 60, 60])
    assert tight is None
    assert trace["stop_reason"] == "empty mask"


def test_an_empty_refinement_keeps_the_first_rounds_answer():
    shape = (120, 160)
    _mask, tight, trace, _predictor = run(
        [rect(shape, 20, 20, 60, 60), np.zeros(shape, dtype=bool)], [20, 20, 60, 60])
    assert tight == [20, 20, 60, 60]
    assert trace["stop_reason"] == "empty after refinement"


def test_the_trace_records_every_round():
    shape = (120, 160)
    masks = [rect(shape, 20, 20, 60, 60), rect(shape, 20, 20, 75, 60),
             rect(shape, 20, 20, 90, 60), rect(shape, 20, 20, 105, 60)]
    _mask, _tight, trace, _predictor = run(masks, [20, 20, 60, 60], max_rounds=3)
    assert len(trace["history"]) == 4
    areas = [step["area"] for step in trace["history"]]
    assert areas == sorted(areas), "areas are recorded in the order they were produced"
    assert "max rounds" in trace["stop_reason"]


def test_interior_points_are_deterministic():
    # A rerun of the pipeline must produce identical boxes, or every before/after comparison in
    # the project becomes untrustworthy.
    mask = rect((120, 160), 20, 20, 90, 70)
    first = grow._sample_interior_points(mask)
    second = grow._sample_interior_points(mask)
    assert np.array_equal(first, second)
    assert len(first) == grow.GROW_POINTS


def test_interior_points_lie_inside_the_mask():
    mask = rect((120, 160), 20, 20, 90, 70)
    for x, y in grow._sample_interior_points(mask):
        assert mask[int(y), int(x)], "a background point would prompt for the wrong thing"


def test_interior_points_handle_a_tiny_mask():
    mask = np.zeros((120, 160), dtype=bool)
    mask[10, 10] = True
    points = grow._sample_interior_points(mask)
    assert points is not None and len(points) == 1
