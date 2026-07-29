"""Tests for the text-free box refinement chain.

The properties under test are the ones whose failure would be invisible in aggregate. Prompt
dilation is a feedback loop over a segmentation model: if the convergence test is wrong in the
lenient direction limbs stay cut off (the bug it was written to fix), and if it is wrong in the
greedy direction every edge-cropped subject dilates to the whole frame and the boxes silently
become useless. 78 of 140 pilot subjects touch a frame edge, so the second failure mode is the
more dangerous one and gets the most tests.
"""
from __future__ import annotations

import numpy as np

from phantom_data import tighten


# ---------------------------------------------------------------------------------------
# dilate_box
# ---------------------------------------------------------------------------------------

def test_dilation_grows_every_side_by_a_fraction_of_the_box():
    grown = tighten.dilate_box([100, 100, 200, 200], 1000, 1000, step=0.1)
    assert grown == [90, 90, 210, 210]


def test_dilation_scales_with_the_box_not_by_a_fixed_pixel_count():
    # Relative growth so a 40px subject and a 400px subject dilate proportionally; a fixed
    # margin would be imperceptible on one and reckless on the other.
    small = tighten.dilate_box([100, 100, 140, 140], 1000, 1000, step=0.25)
    large = tighten.dilate_box([100, 100, 500, 500], 1000, 1000, step=0.25)
    assert small == [90, 90, 150, 150]
    assert large == [0, 0, 600, 600]


def test_dilation_is_clamped_to_the_frame():
    grown = tighten.dilate_box([5, 5, 95, 95], 100, 100, step=0.5)
    assert grown == [0, 0, 100, 100]


# ---------------------------------------------------------------------------------------
# pinned_edges -- the convergence test
# ---------------------------------------------------------------------------------------

def test_a_mask_well_inside_the_prompt_pins_nothing():
    assert tighten.pinned_edges([120, 120, 180, 180], [100, 100, 200, 200], 500, 500) == []


def test_a_mask_touching_a_prompt_edge_pins_that_edge():
    assert tighten.pinned_edges([100, 120, 180, 180], [100, 100, 200, 200], 500, 500) == ["left"]


def test_all_four_edges_can_pin_at_once():
    pinned = tighten.pinned_edges([100, 100, 200, 200], [100, 100, 200, 200], 500, 500)
    assert pinned == ["left", "top", "right", "bottom"]


def test_slack_absorbs_the_mask_upsampling_error():
    # SAM2 computes the mask at 1/4 resolution and upsamples, so an unconstrained boundary still
    # lands a pixel or two short. Without slack, converged masks would read as pinned forever.
    assert tighten.pinned_edges([102, 120, 180, 180], [100, 100, 200, 200], 500, 500,
                                slack=2) == ["left"]
    assert tighten.pinned_edges([103, 120, 180, 180], [100, 100, 200, 200], 500, 500,
                                slack=2) == []


def test_a_prompt_edge_at_the_frame_boundary_never_pins():
    # The runaway guard. A subject genuinely running off screen presses that edge no matter how
    # far the prompt grows, so counting it would dilate to the full frame and never converge.
    # This is the majority case on the pilot: 78 of 140 subjects touch an edge.
    assert tighten.pinned_edges([0, 0, 100, 100], [0, 0, 100, 100], 100, 100) == []


def test_only_the_frame_touching_edges_are_exempt_not_the_others():
    # A subject cropped on the left but with room on the right: the left must be ignored and the
    # right must still pin, or a half-off-screen subject would stop dilating too early.
    pinned = tighten.pinned_edges([0, 50, 200, 150], [0, 50, 200, 300], 200, 400)
    assert "left" not in pinned and "right" not in pinned
    assert "top" in pinned and "bottom" not in pinned


# ---------------------------------------------------------------------------------------
# tighten_box -- the loop, against a stub segmenter
# ---------------------------------------------------------------------------------------

class StubSegmenter:
    """Stands in for SAM2: returns a fixed rectangle of mask, clipped to the prompt.

    Reproduces the behaviour the measurement identified as the root cause -- SAM2 treats a box
    prompt as a near-hard boundary, so the mask is the true object intersected with the prompt.
    That makes the loop's job testable without a GPU: the "object" is known, so whether dilation
    recovers it is a definite question.
    """

    def __init__(self, object_box, frame_shape=(400, 600)):
        self.object_box = object_box
        self.frame_shape = frame_shape
        self.calls: list[list[int]] = []

    def __call__(self, models, frame, prompt, device="cuda"):
        self.calls.append(list(prompt))
        mask = np.zeros(self.frame_shape, dtype=bool)
        ox1, oy1, ox2, oy2 = self.object_box
        px1, py1, px2, py2 = prompt
        x1, y1 = max(ox1, px1), max(oy1, py1)
        x2, y2 = min(ox2, px2), min(oy2, py2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
        return mask


def run_loop(monkeypatch, object_box, prompt, frame_shape=(400, 600), **kwargs):
    stub = StubSegmenter(object_box, frame_shape)
    monkeypatch.setattr("phantom_data.build.segment.segment_reference", stub)
    frame = np.zeros((*frame_shape, 3), dtype=np.uint8)
    mask, tight, trace = tighten.tighten_box(None, frame, prompt, device="cpu", **kwargs)
    return mask, tight, trace, stub


def test_a_prompt_that_already_contains_the_object_does_not_dilate(monkeypatch):
    _mask, tight, trace, stub = run_loop(monkeypatch, [150, 150, 250, 250],
                                         [100, 100, 300, 300])
    assert tight == [150, 150, 250, 250]
    assert trace["dilated"] is False
    assert len(stub.calls) == 1, "converged on the first pass, so only one segmentation"
    assert "converged" in trace["stop_reason"]


def test_a_clipped_object_is_recovered_by_dilation(monkeypatch):
    # The bug this whole mechanism exists for: the prompt cuts the object off at x=200, so the
    # first mask stops there and the box loses everything to the right.
    _mask, tight, trace, stub = run_loop(monkeypatch, [150, 150, 240, 250],
                                         [100, 100, 200, 300])
    assert tight == [150, 150, 240, 250], "the object's true extent is recovered"
    assert trace["dilated"] is True
    assert "converged" in trace["stop_reason"]
    assert len(stub.calls) > 1


def test_recovery_is_bounded_by_max_rounds_and_reported_honestly(monkeypatch):
    """A badly clipped object is only partly recovered, and the trace must not claim otherwise.

    Growth compounds at 15% of the *current* prompt per round, so three rounds widen a prompt by
    roughly 1.5x per axis. An object extending far past that is recovered partially -- more than
    the single pass managed, less than the truth. This is a real ceiling of the mechanism, not a
    defect, and the value of testing it is that ``stop_reason`` distinguishes "this is the
    object's boundary" from "I ran out of rounds", which is what makes the incomplete boxes
    findable in the report instead of indistinguishable from good ones.
    """
    object_box = [150, 150, 400, 250]
    _mask, tight, trace, _stub = run_loop(monkeypatch, object_box, [100, 100, 200, 300])
    assert tight[2] > 200, "it must make progress past the original prompt edge"
    assert tight[2] < object_box[2], "but three rounds cannot reach this far"
    assert "max rounds" in trace["stop_reason"]
    assert trace["history"][-1]["pinned"], "and the final round records the edge still pinned"

    # More rounds do recover it, which confirms the ceiling is the round budget rather than the
    # convergence test refusing to grow.
    _mask, tight_more, trace_more = tighten.tighten_box(
        None, np.zeros((400, 600, 3), dtype=np.uint8), [100, 100, 200, 300],
        device="cpu", max_rounds=8)
    assert tight_more == object_box
    assert "converged" in trace_more["stop_reason"]


def test_the_loop_stops_at_max_rounds_and_says_so(monkeypatch):
    # An object far larger than the prompt cannot be recovered in the allowed rounds. The trace
    # must record that the answer is incomplete rather than presenting it as converged.
    _mask, _tight, trace, stub = run_loop(monkeypatch, [0, 0, 600, 400],
                                          [280, 190, 320, 210], max_rounds=2)
    assert trace["rounds"] == 2
    assert "max rounds" in trace["stop_reason"]
    assert len(stub.calls) == 3, "initial pass plus two dilations"


def test_an_object_filling_the_whole_frame_does_not_dilate_forever(monkeypatch):
    # The other half of the runaway guard, end to end. frame_shape is (height, width) = 400x600,
    # so a prompt covering the entire frame is [0, 0, 600, 400] -- every edge is a frame boundary,
    # the mask presses all four, and none may count as pinned. Getting the two axes backwards
    # here would leave a genuinely free edge untested and the guard unproven.
    frame_shape = (400, 600)
    full = [0, 0, 600, 400]
    _mask, tight, trace, stub = run_loop(monkeypatch, full, full, frame_shape=frame_shape)
    assert trace["dilated"] is False
    assert len(stub.calls) == 1
    assert tight == full


def test_a_subject_cropped_on_one_side_only_dilates_on_its_free_edges(monkeypatch):
    # The realistic majority case: 78 of 140 pilot subjects touch an edge. Such a subject must
    # still dilate towards open space while its cropped edge is exempt -- otherwise the guard
    # would either run away or switch off refinement entirely for over half the dataset.
    # The object extends to x=280, within the ~1.5x that three rounds can reach from a 200-wide
    # prompt (230 -> 264 -> 304); see the max-rounds test for what happens beyond that.
    frame_shape = (400, 600)
    _mask, tight, trace, _stub = run_loop(monkeypatch, [0, 100, 280, 300],
                                          [0, 100, 200, 300], frame_shape=frame_shape)
    assert tight == [0, 100, 280, 300], "grew right, stayed pinned to the frame on the left"
    assert trace["dilated"] is True
    assert "converged" in trace["stop_reason"]
    assert trace["final_prompt"][0] == 0, "the frame-edge side never moved"


def test_the_trace_records_how_much_the_prompt_grew(monkeypatch):
    _mask, _tight, trace, _stub = run_loop(monkeypatch, [100, 100, 500, 350],
                                           [200, 150, 300, 250])
    assert trace["prompt_growth_area"] > 1.0
    assert trace["original_prompt"] == [200, 150, 300, 250]
    assert trace["final_prompt"] != trace["original_prompt"]
    # Every round is on the record, so a suspicious box can be traced back to the prompt that
    # produced it rather than being taken on faith.
    assert len(trace["history"]) == trace["rounds"] + 1
    assert all({"prompt", "tight", "pinned"} <= set(step) for step in trace["history"])


def test_a_degenerate_prompt_returns_no_box_rather_than_inventing_one(monkeypatch):
    for bad in (None, [10, 10, 10, 10], "not a box"):
        _mask, tight, trace, stub = run_loop(monkeypatch, [0, 0, 100, 100], bad)
        assert tight is None
        assert trace["stop_reason"] == "degenerate prompt"
        assert stub.calls == [], "must not call the segmenter on a box it rejected"


def test_an_empty_mask_yields_no_box(monkeypatch):
    # Object entirely outside the prompt: the stub returns nothing, and the chain must decline
    # rather than emit the prompt as if it were a measured box.
    _mask, tight, trace, _stub = run_loop(monkeypatch, [500, 350, 550, 380],
                                          [10, 10, 40, 40], max_rounds=0)
    assert tight is None
    assert trace["stop_reason"] == "empty mask"


def test_dilation_keeps_the_last_good_mask_if_a_later_round_returns_nothing(monkeypatch):
    """A dilated prompt that yields an empty mask must not discard the tighter prompt's answer."""
    frame_shape = (400, 600)

    class Flaky(StubSegmenter):
        def __call__(self, models, frame, prompt, device="cuda"):
            mask = super().__call__(models, frame, prompt, device=device)
            return np.zeros(frame_shape, dtype=bool) if len(self.calls) > 1 else mask

    stub = Flaky([150, 150, 400, 250], frame_shape)
    monkeypatch.setattr("phantom_data.build.segment.segment_reference", stub)
    frame = np.zeros((*frame_shape, 3), dtype=np.uint8)
    _mask, tight, trace = tighten.tighten_box(None, frame, [100, 100, 200, 300], device="cpu")
    assert tight is not None, "the first round's box must survive a later empty round"
    assert trace["stop_reason"] == "empty after dilation"


# ---------------------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------------------

def test_diagnostics_describe_the_refinement_against_phantoms_original_box():
    mask = np.zeros((400, 600), dtype=bool)
    mask[150:250, 150:250] = True
    diag = tighten.tighten_diagnostics(mask, [150, 150, 250, 250],
                                       [100, 100, 300, 300], (400, 600))
    assert diag["tightened"] is True
    assert diag["area_ratio"] == 0.25          # 100x100 tight inside a 200x200 prompt
    assert diag["mask_fill_of_tight"] == 1.0   # a solid rectangle fills its own box
    assert diag["touches_frame_edge"] is False


def test_diagnostics_flag_a_mask_that_collapsed_onto_a_part():
    mask = np.zeros((400, 600), dtype=bool)
    mask[150:170, 150:170] = True
    diag = tighten.tighten_diagnostics(mask, [150, 150, 170, 170],
                                       [100, 100, 300, 300], (400, 600))
    assert diag["suspect_shrink"] is True


def test_diagnostics_report_a_failure_rather_than_raising_on_a_missing_mask():
    diag = tighten.tighten_diagnostics(None, None, [100, 100, 200, 200], (400, 600))
    assert diag["tightened"] is False
    assert diag["fail_reason"] == "empty mask"
