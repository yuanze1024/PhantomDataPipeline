"""Tests for per-side box selection in :func:`phantom_data.redetect.pick_side`.

Box *selection* is this file; which pairs survive is ``test_decide.py``. The two modes exist
for a reason that is about coordinate systems rather than quality, and the tests are written to
hold that reason down:

* ``trust_detector=True`` -- Grounding DINO returns **real frame pixel coordinates**, so its box
  needs no projection and cannot be misplaced by a wrong one. Phantom's box has to go through an
  unresolved annotation canvas (``H_768_long``, measurably wrong on the x axis: 14% of boxes
  overflow it). The detector's box therefore wins outright, and "no detection" is a filter-out
  rather than a fallback to a box we do not trust.
* ``trust_detector=False`` -- the historical rule, kept **byte-for-byte** because it produced the
  pilot's ``gate_report.json`` and is the control that report stays reproducible against. The
  parity test below asserts the reason strings too, not just the chosen box set: the reasons are
  written into the report and the viewer shows them, so a reworded reason is a changed artifact.

Why the mode matters at all: box correction moved to *before* SAM2, so the chosen box is now
what gets segmented and what ships as the training condition, rather than a report annotation.
Keeping a box already believed to be skewed stopped being free.
"""
from __future__ import annotations

import itertools

import pytest

from phantom_data import redetect
from phantom_data.redetect import FROM_DIS, NO_BOX, PHANTOM


# ----- trust_detector=True: the detector's box always wins ----------------------------


@pytest.mark.parametrize("phantom_clip", [None, 0.0, 0.15, 0.21, 0.30, 0.9])
@pytest.mark.parametrize("box_iou", [None, 0.0, 0.4, 0.75, 0.99])
def test_trust_detector_takes_the_new_box_whatever_phantom_scored(phantom_clip, box_iou) -> None:
    """No branch on Phantom's numbers: frame coordinates beat canvas-mapped ones, full stop."""
    pick, _reason = redetect.pick_side(phantom_clip, box_iou, True, trust_detector=True)
    assert pick == FROM_DIS


def test_trust_detector_takes_the_new_box_where_the_old_rule_protected_phantom() -> None:
    """The case the two modes disagree on, named explicitly.

    Phantom's crop scored well (0.30 >= 0.21) and the new box moved far (IoU 0.40 < 0.75). The
    old rule reads that as "the detector wandered off" and keeps Phantom's box; the new one reads
    it as "the two boxes disagree, and only one of them is in coordinates we trust".
    """
    assert redetect.pick_side(0.30, 0.40, True, trust_detector=True)[0] == FROM_DIS
    assert redetect.pick_side(0.30, 0.40, True, trust_detector=False)[0] == PHANTOM


def test_trust_detector_filters_out_when_no_box_was_found() -> None:
    """``NO_BOX``, not ``PHANTOM``. A fallback here would ship the untrusted box."""
    pick, reason = redetect.pick_side(0.9, None, False, trust_detector=True)
    assert pick == NO_BOX
    assert "filtered out" in reason


def test_no_box_is_not_one_of_the_box_sets() -> None:
    """It is an outcome, not a third set of coordinates.

    Load-bearing for ``flat_scores``, which builds ``box_<side>_<set>`` keys from ``BOX_SETS``
    and must find no key to copy for a filtered side (chosen box comes out None).
    """
    assert NO_BOX not in redetect.BOX_SETS


# ----- trust_detector=False: the four historical branches ------------------------------


def test_phantom_good_and_iou_high_refines() -> None:
    pick, reason = redetect.pick_side(0.25, 0.80, True, trust_detector=False)
    assert pick == FROM_DIS
    assert "refines" in reason


def test_phantom_good_and_iou_low_keeps_phantom() -> None:
    pick, reason = redetect.pick_side(0.25, 0.40, True, trust_detector=False)
    assert pick == PHANTOM
    assert "moved too far" in reason


def test_phantom_bad_takes_the_new_box_whatever_the_iou() -> None:
    for box_iou in (None, 0.0, 0.5, 0.99):
        assert redetect.pick_side(0.10, box_iou, True, trust_detector=False)[0] == FROM_DIS


def test_no_box_falls_back_to_phantom_in_the_historical_mode() -> None:
    pick, reason = redetect.pick_side(0.25, 0.80, False, trust_detector=False)
    assert pick == PHANTOM
    assert reason == "detector found nothing"


def test_a_missing_phantom_clip_counts_as_bad() -> None:
    """None means the crop could not be scored, which is not evidence the box is fine."""
    assert redetect.pick_side(None, 0.99, True, trust_detector=False)[0] == FROM_DIS


def test_thresholds_are_inclusive_in_the_historical_mode() -> None:
    """``>=`` on both gates, asserted at the boundary because the report's counts sit on it."""
    assert redetect.pick_side(0.21, 0.75, True, trust_detector=False)[0] == FROM_DIS
    assert redetect.pick_side(0.20999, 0.75, True, trust_detector=False)[0] == FROM_DIS
    assert redetect.pick_side(0.21, 0.74999, True, trust_detector=False)[0] == PHANTOM


# ----- the control must not drift ------------------------------------------------------


def _historical_pick_side(phantom_clip, box_iou, has_new_box, clip_min=redetect.CLIP_MIN,
                          iou_min=redetect.IOU_MIN):
    """The rule exactly as it stood before ``trust_detector`` existed.

    Duplicated here on purpose. This is the only copy of the pre-change behaviour that is not
    the implementation under test, so the parity assertion below compares against something
    that cannot be updated by editing ``redetect.py`` -- which is the entire point of pinning a
    control. Reason strings included: they are written into ``gate_report.json``.
    """
    if not has_new_box:
        return PHANTOM, "detector found nothing"
    if phantom_clip is not None and phantom_clip >= clip_min:
        if box_iou is not None and box_iou >= iou_min:
            return FROM_DIS, f"phantom box was fine; new box refines it (IoU >= {iou_min})"
        return PHANTOM, (f"phantom box was fine and the new box moved too far "
                         f"(IoU < {iou_min})")
    return FROM_DIS, f"phantom crop scored below {clip_min}, so the new box is used"


@pytest.mark.parametrize("phantom_clip,box_iou,has_new_box", list(itertools.product(
    [None, 0.0, 0.15, 0.20999, 0.21, 0.25, 0.9],
    [None, 0.0, 0.5, 0.74999, 0.75, 0.9, 1.0],
    [True, False],
)))
def test_trust_detector_false_reproduces_the_historical_rule(phantom_clip, box_iou,
                                                             has_new_box) -> None:
    assert (redetect.pick_side(phantom_clip, box_iou, has_new_box, trust_detector=False)
            == _historical_pick_side(phantom_clip, box_iou, has_new_box))


# ----- a filtered side must not be rescued by the other side's numbers ----------------


def _scores(**overrides):
    """A subject that would sail through: identity, clip and IoU all pass on both sides."""
    base = {
        "pick_ref": FROM_DIS, "pick_seed": FROM_DIS,
        "pick_ref_reason": "detector box preferred", "pick_seed_reason": "detector box preferred",
        "dino_cos_chosen": 0.80,
        "chosen_clip_ref": 0.30, "chosen_clip_seed": 0.29,
        "iou_dis_vs_phantom": 0.90, "iou_seed_dis_vs_phantom": 0.88,
    }
    return {**base, **overrides}


def test_decide_keeps_the_control_subject() -> None:
    """Guards the test below: without this, a broken fixture would make it pass vacuously."""
    assert redetect.decide(_scores())["verdict"] == redetect.KEEP


@pytest.mark.parametrize("side", ["ref", "seed"])
@pytest.mark.parametrize("rule", redetect.RULES)
def test_a_no_box_side_drops_the_subject_under_both_rules(side, rule) -> None:
    """The guard that stops ``iou_stands`` keeping a pair with one box missing.

    Not redundant with the identity gate: every gate value is a ``max`` over the two sides, so a
    subject that lost only its reference box still carries the target side's IoU of 0.88, and
    ``iou_stands`` would keep it on that number alone. The rule has to read the *pick*.
    """
    ruling = redetect.decide(_scores(**{f"pick_{side}": NO_BOX,
                                        f"chosen_clip_{side}": None}), rule)
    assert ruling["verdict"] == redetect.DROP
    assert ruling["no_box_sides"] == [side]
    assert ruling["rescued_by_iou"] is False
    assert "no usable box" in ruling["reason"]


def test_no_box_sides_is_always_present() -> None:
    """Empty, not absent, on the normal path -- callers read it without a default."""
    assert redetect.decide(_scores())["no_box_sides"] == []
