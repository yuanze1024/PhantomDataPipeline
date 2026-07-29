"""Tests for the keep/drop rule in :func:`phantom_data.redetect.decide`.

Box *selection* is :func:`pick_side`'s job; this file is only about which pairs survive. Each
part of the rule rests on a stated reason, so the tests name the reason rather than just the
arithmetic:

* Identity is mandatory -- a reference that is not the same object as the target is unusable
  whatever else passes. 0.6 was chosen for a low false-positive rate, accepting that some
  genuine same-object pairs get dropped.
* CLIP and IoU are alternatives, not both-required. They establish the same fact by
  independent means; once one side is confirmed and identity says both sides match, a low
  CLIP score on the other side is evidence about CLIP (these crops keep their background,
  which depresses scores) rather than about the box.
"""
from __future__ import annotations

import pytest

from phantom_data import redetect
from phantom_data.redetect import DROP, KEEP


def scores(**overrides):
    """A subject whose low Phantom scores put both sides on the detector's boxes."""
    base = {
        "pick_ref": "dis", "pick_seed": "dis",
        "pick_ref_reason": "phantom crop scored low", "pick_seed_reason": "phantom crop low",
        "dino_cos_chosen": 0.70,
        "crop_clip_ref_phantom": 0.18, "crop_clip_seed_phantom": 0.19,
        "crop_clip_ref_dis": 0.25, "crop_clip_seed_dis": 0.24,
        "chosen_clip_ref": 0.25, "chosen_clip_seed": 0.24,
        "box_ref_dis": [1, 1, 9, 9], "box_seed_dis": [2, 2, 9, 9],
        "iou_dis_vs_phantom": 0.40, "iou_seed_dis_vs_phantom": 0.45,
    }
    return {**base, **overrides}


def on_phantom(**overrides):
    """A subject that stayed on Phantom's boxes -- so it has no IoU to fall back on."""
    base = {
        "pick_ref": "phantom", "pick_seed": "phantom",
        "pick_ref_reason": "detector found nothing",
        "pick_seed_reason": "detector found nothing",
        "dino_cos_chosen": 0.70,
        "crop_clip_ref_phantom": 0.25, "crop_clip_seed_phantom": 0.24,
        "chosen_clip_ref": 0.25, "chosen_clip_seed": 0.24,
        "iou_dis_vs_phantom": None, "iou_seed_dis_vs_phantom": None,
    }
    return {**base, **overrides}


# ----- the identity gate ---------------------------------------------------------------


def test_low_identity_drops_however_good_everything_else_is() -> None:
    """Under the identity-required rule a different object cannot be rescued by any score."""
    result = redetect.decide(scores(dino_cos_chosen=0.30, chosen_clip_ref=0.40,
                                    iou_dis_vs_phantom=0.99,
                                    iou_seed_dis_vs_phantom=0.99),
                             redetect.RULE_IDENTITY_REQUIRED)
    assert result["verdict"] == DROP
    assert "not the same object" in result["reason"]


def test_the_iou_rule_keeps_the_same_subject() -> None:
    """The one difference between the rules, asserted head to head on one subject."""
    subject = scores(dino_cos_chosen=0.30, chosen_clip_ref=0.40,
                     iou_dis_vs_phantom=0.99, iou_seed_dis_vs_phantom=0.99)
    result = redetect.decide(subject, redetect.RULE_IOU_STANDS)
    assert result["verdict"] == KEEP
    assert result["rescued_by_iou"] is True


def test_identity_is_inclusive_at_the_threshold() -> None:
    assert redetect.decide(scores(dino_cos_chosen=0.6))["verdict"] == KEEP


def test_identity_just_below_the_threshold_drops() -> None:
    assert redetect.decide(scores(dino_cos_chosen=0.599))["verdict"] == DROP


def test_a_missing_identity_score_drops() -> None:
    """Unmeasured is not the same as passing."""
    assert redetect.decide(scores(dino_cos_chosen=None))["verdict"] == DROP


# ----- the semantic / agreement alternative -------------------------------------------


def test_clip_alone_is_enough() -> None:
    result = redetect.decide(scores(chosen_clip_ref=0.22, chosen_clip_seed=0.10,
                                    iou_dis_vs_phantom=0.10,
                                    iou_seed_dis_vs_phantom=0.10))
    assert result["verdict"] == KEEP and result["clip_ok"] and not result["iou_ok"]


def test_iou_alone_is_enough_even_with_a_low_clip_score() -> None:
    """The stated reason: background-inclusive crops depress CLIP, so IoU can carry it."""
    result = redetect.decide(scores(chosen_clip_ref=0.05, chosen_clip_seed=0.05,
                                    iou_dis_vs_phantom=0.80,
                                    iou_seed_dis_vs_phantom=0.10),
                             redetect.RULE_IDENTITY_REQUIRED)
    assert result["verdict"] == KEEP and result["iou_ok"] and not result["clip_ok"]
    assert "the box agrees with Phantom's" in result["reason"]


def test_neither_passing_drops() -> None:
    result = redetect.decide(scores(chosen_clip_ref=0.05, chosen_clip_seed=0.05,
                                    iou_dis_vs_phantom=0.10,
                                    iou_seed_dis_vs_phantom=0.10))
    assert result["verdict"] == DROP
    assert "nothing confirms" in result["reason"]


def test_clip_uses_the_better_side_not_the_worse() -> None:
    """One confirmed side plus a passing identity check already pins both."""
    result = redetect.decide(scores(chosen_clip_ref=0.30, chosen_clip_seed=0.02,
                                    iou_dis_vs_phantom=0.1,
                                    iou_seed_dis_vs_phantom=0.1))
    assert result["clip"] == pytest.approx(0.30)
    assert result["verdict"] == KEEP


def test_iou_uses_the_better_side_too() -> None:
    result = redetect.decide(scores(chosen_clip_ref=0.05, chosen_clip_seed=0.05,
                                    iou_dis_vs_phantom=0.10,
                                    iou_seed_dis_vs_phantom=0.90))
    assert result["iou"] == pytest.approx(0.90)
    assert result["verdict"] == KEEP


def test_clip_is_inclusive_at_the_threshold() -> None:
    result = redetect.decide(scores(chosen_clip_ref=0.21, chosen_clip_seed=0.05,
                                    iou_dis_vs_phantom=0.1,
                                    iou_seed_dis_vs_phantom=0.1))
    assert result["verdict"] == KEEP


def test_clip_just_below_the_threshold_needs_the_iou() -> None:
    below = scores(chosen_clip_ref=0.209, chosen_clip_seed=0.05,
                   iou_dis_vs_phantom=0.1, iou_seed_dis_vs_phantom=0.1)
    assert redetect.decide(below)["verdict"] == DROP
    assert redetect.decide({**below, "iou_dis_vs_phantom": 0.8})["verdict"] == KEEP


def test_iou_is_inclusive_at_the_threshold() -> None:
    result = redetect.decide(scores(chosen_clip_ref=0.05, chosen_clip_seed=0.05,
                                    iou_dis_vs_phantom=0.75,
                                    iou_seed_dis_vs_phantom=0.10))
    assert result["verdict"] == KEEP


# ----- subjects still on Phantom's boxes ----------------------------------------------


def test_a_phantom_subject_passes_on_clip_alone() -> None:
    """With no re-detected box there is no IoU, so CLIP is its only route."""
    result = redetect.decide(on_phantom())
    assert result["verdict"] == KEEP and result["iou"] is None


def test_a_phantom_subject_with_low_identity_still_drops() -> None:
    assert redetect.decide(on_phantom(dino_cos_chosen=0.4))["verdict"] == DROP


def test_a_phantom_subject_below_clip_has_no_second_chance() -> None:
    result = redetect.decide(on_phantom(chosen_clip_ref=0.10, chosen_clip_seed=0.10))
    assert result["verdict"] == DROP


# ----- reporting ----------------------------------------------------------------------


def test_box_set_reads_phantom_when_both_sides_kept_it() -> None:
    assert redetect.decide(on_phantom())["box_set"] == "phantom"


def test_box_set_reads_dis_when_both_sides_were_replaced() -> None:
    assert redetect.decide(scores())["box_set"] == "dis"


def test_box_set_names_both_sides_when_they_differ() -> None:
    """A mixed selection must not be reported as if one box set won outright."""
    mixed = redetect.decide(scores(pick_ref="phantom"))
    assert mixed["box_set"] == "ref=phantom, target=dis"


def test_the_reason_carries_both_sides_selection_reasons() -> None:
    reason = redetect.decide(scores())["box_reason"]
    assert "ref:" in reason and "seed:" in reason


# ----- thresholds are parameters -------------------------------------------------------


def test_thresholds_can_be_overridden() -> None:
    subject = scores(dino_cos_chosen=0.55)
    assert redetect.decide(subject)["verdict"] == DROP
    assert redetect.decide(subject, identity_min=0.5)["verdict"] == KEEP


def test_defaults_match_the_agreed_rule() -> None:
    assert (redetect.IDENTITY_MIN, redetect.CLIP_MIN, redetect.IOU_MIN) == (0.6, 0.21, 0.75)


# --------------------------------------------------------------------------------------
# iou_both_sides: the rule that will not let one side vouch for the other
# --------------------------------------------------------------------------------------

BOTH = redetect.RULE_IOU_BOTH_SIDES


def decide_both(**overrides):
    return redetect.decide(scores(**overrides), rule=BOTH)


def test_both_sides_passing_keeps() -> None:
    assert decide_both(iou_dis_vs_phantom=0.80,
                       iou_seed_dis_vs_phantom=0.90)["verdict"] == KEEP


@pytest.mark.parametrize("ref_iou, seed_iou", [(0.80, 0.40), (0.40, 0.80)])
def test_one_weak_side_drops_whichever_side_it_is(ref_iou: float, seed_iou: float) -> None:
    """The whole point: a strong side must not carry a weak one.

    ``iou_stands`` takes the ``max`` and would keep both of these on the strong side alone --
    which is how a detector that wandered off on one side shipped.
    """
    ruling = decide_both(iou_dis_vs_phantom=ref_iou, iou_seed_dis_vs_phantom=seed_iou)
    assert ruling["verdict"] == DROP
    assert "both sides" in ruling["reason"]


def test_max_of_sides_would_have_kept_what_this_rule_drops() -> None:
    """Pins the difference between the rules rather than asserting one in isolation."""
    one_sided = scores(iou_dis_vs_phantom=0.00, iou_seed_dis_vs_phantom=0.95)
    assert redetect.decide(one_sided, rule=redetect.RULE_IOU_STANDS)["verdict"] == KEEP
    assert redetect.decide(one_sided, rule=BOTH)["verdict"] == DROP


def test_both_sides_are_inclusive_at_the_threshold() -> None:
    at = redetect.IOU_MIN
    assert decide_both(iou_dis_vs_phantom=at, iou_seed_dis_vs_phantom=at)["verdict"] == KEEP


def test_a_hair_below_on_one_side_drops() -> None:
    at = redetect.IOU_MIN
    assert decide_both(iou_dis_vs_phantom=at,
                       iou_seed_dis_vs_phantom=at - 0.001)["verdict"] == DROP


@pytest.mark.parametrize("missing", ["iou_dis_vs_phantom", "iou_seed_dis_vs_phantom"])
def test_a_missing_side_is_a_failure_not_a_skip(missing: str) -> None:
    """An unmeasured side must not read as an agreeing one."""
    ruling = redetect.decide(scores(**{missing: None}, **{
        "iou_seed_dis_vs_phantom" if missing == "iou_dis_vs_phantom"
        else "iou_dis_vs_phantom": 0.95}), rule=BOTH)
    assert ruling["verdict"] == DROP
    assert ruling["iou_both_ok"] is False


def test_high_identity_cannot_rescue_a_disagreeing_box() -> None:
    """The measured failure mode: the detector wanders to the same wrong object on both
    sides, so identity is high and clip passes, and only the IoUs give it away."""
    ruling = decide_both(dino_cos_chosen=0.95, chosen_clip_ref=0.30, chosen_clip_seed=0.30,
                         iou_dis_vs_phantom=0.0, iou_seed_dis_vs_phantom=0.0)
    assert ruling["verdict"] == DROP
    assert ruling["identity_ok"] and ruling["clip_ok"]


def test_identity_and_clip_are_still_required() -> None:
    strong_iou = {"iou_dis_vs_phantom": 0.95, "iou_seed_dis_vs_phantom": 0.95}
    assert decide_both(dino_cos_chosen=0.10, **strong_iou)["verdict"] == DROP
    assert decide_both(chosen_clip_ref=0.05, chosen_clip_seed=0.05,
                       **strong_iou)["verdict"] == DROP


def test_per_side_ious_are_reported_under_every_rule() -> None:
    """So a threshold can be re-tuned from a stored report, with no GPU pass."""
    for rule in redetect.RULES:
        ruling = redetect.decide(scores(iou_dis_vs_phantom=0.31,
                                        iou_seed_dis_vs_phantom=0.62), rule=rule)
        assert ruling["iou_ref"] == 0.31
        assert ruling["iou_seed"] == 0.62
        assert ruling["iou_both_ok"] is False


def test_the_new_rule_is_registered_but_not_the_default() -> None:
    assert BOTH in redetect.RULES
    assert redetect.DEFAULT_RULE == redetect.RULE_IOU_STANDS


# --------------------------------------------------------------------------------------
# RULE_IOU_FLOOR_PEAK -- both sides clear a floor, one side reaches the peak
# --------------------------------------------------------------------------------------
#
# The premise, which none of the other rules encodes: Phantom's annotation is taken to be
# semantically right and merely offset, so Grounding DINO normalises it rather than voting on
# which object the phrase names. That makes two failure modes worth separating, and a single
# threshold cannot separate them:
#
#   detector left the object -> near-zero IoU on one side  -> the floor catches it
#   box is merely offset     -> middling IoU on both sides -> should survive, and does
#
# The peak then asks that the normalisation be anchored somewhere: at least one side has to land
# where the detector and the annotator genuinely agree.

FLOOR_PEAK = redetect.RULE_IOU_FLOOR_PEAK


def decide_fp(*, iou_floor: float = redetect.IOU_FLOOR_MIN, **overrides):
    return redetect.decide(scores(**overrides), rule=FLOOR_PEAK, iou_floor=iou_floor)


def test_offset_on_one_side_is_kept() -> None:
    """The case this rule exists for: anchored on one side, offset on the other."""
    ruling = decide_fp(iou_dis_vs_phantom=0.90, iou_seed_dis_vs_phantom=0.60)
    assert ruling["verdict"] == KEEP
    assert ruling["iou_floor_ok"] and ruling["iou_peak_ok"]


def test_this_is_exactly_what_the_strict_two_sided_rule_drops() -> None:
    """Pins the difference between the two two-sided rules instead of asserting one alone."""
    offset = scores(iou_dis_vs_phantom=0.90, iou_seed_dis_vs_phantom=0.60)
    assert redetect.decide(offset, rule=BOTH)["verdict"] == DROP
    assert redetect.decide(offset, rule=FLOOR_PEAK)["verdict"] == KEEP


def test_a_wandered_detector_is_dropped_by_both_two_sided_rules() -> None:
    """Near-zero on one side means a different object, and no floor in the useful band
    tolerates it. Asserted at 0.3 and 0.5 because the pilot's 11 wandered subjects all sit at
    or below 0.06, so the choice of floor inside that band must not change this verdict."""
    wandered = {"iou_dis_vs_phantom": 0.95, "iou_seed_dis_vs_phantom": 0.02}
    assert redetect.decide(scores(**wandered), rule=BOTH)["verdict"] == DROP
    for floor in (0.3, 0.5):
        ruling = decide_fp(iou_floor=floor, **wandered)
        assert ruling["verdict"] == DROP
        assert ruling["iou_floor_ok"] is False


def test_both_sides_loose_is_dropped_even_though_the_floor_passes() -> None:
    """What the peak adds. Nothing anchors a pair of middling boxes, so 'in the area' is not
    the same as 'normalised'. On the pilot this is 6 subjects (e.g. ref 0.544 / target 0.587)."""
    ruling = decide_fp(iou_dis_vs_phantom=0.60, iou_seed_dis_vs_phantom=0.55)
    assert ruling["verdict"] == DROP
    assert ruling["iou_floor_ok"] is True
    assert ruling["iou_peak_ok"] is False
    assert "neither reaches" in ruling["reason"]


def test_the_floor_and_the_peak_are_both_inclusive() -> None:
    ruling = decide_fp(iou_floor=0.5, iou_dis_vs_phantom=redetect.IOU_MIN,
                       iou_seed_dis_vs_phantom=0.5)
    assert ruling["verdict"] == KEEP


def test_a_hair_below_the_floor_drops() -> None:
    ruling = decide_fp(iou_floor=0.5, iou_dis_vs_phantom=0.95,
                       iou_seed_dis_vs_phantom=0.499)
    assert ruling["verdict"] == DROP
    assert ruling["iou_floor_ok"] is False


@pytest.mark.parametrize("missing", ["iou_dis_vs_phantom", "iou_seed_dis_vs_phantom"])
def test_a_missing_side_fails_the_floor_and_the_peak(missing: str) -> None:
    """The one-sided pass this rule refuses. Were the peak computed as ``max`` over the
    *present* values, a single strong side would satisfy it on its own -- so the peak has to be
    guarded on both sides being measured, not just the floor."""
    other = ("iou_seed_dis_vs_phantom" if missing == "iou_dis_vs_phantom"
             else "iou_dis_vs_phantom")
    ruling = redetect.decide(scores(**{missing: None, other: 0.99}), rule=FLOOR_PEAK)
    assert ruling["verdict"] == DROP
    assert ruling["iou_floor_ok"] is False
    assert ruling["iou_peak_ok"] is False


def test_floor_peak_still_requires_identity_and_clip() -> None:
    good_iou = {"iou_dis_vs_phantom": 0.90, "iou_seed_dis_vs_phantom": 0.60}
    assert decide_fp(dino_cos_chosen=0.10, **good_iou)["verdict"] == DROP
    assert decide_fp(chosen_clip_ref=0.05, chosen_clip_seed=0.05,
                     **good_iou)["verdict"] == DROP


def test_floor_equal_to_peak_reproduces_the_strict_rule() -> None:
    """The degenerate case the CLI and the viewer clamp to, asserted rather than assumed:
    at floor == peak the two-sided rules must agree on every subject."""
    for ref, seed in ((0.90, 0.60), (0.80, 0.80), (0.95, 0.02), (0.60, 0.55)):
        subject = scores(iou_dis_vs_phantom=ref, iou_seed_dis_vs_phantom=seed)
        strict = redetect.decide(subject, rule=BOTH)
        clamped = redetect.decide(subject, rule=FLOOR_PEAK, iou_floor=redetect.IOU_MIN)
        assert clamped["verdict"] == strict["verdict"], (ref, seed)


def test_the_floor_is_weaker_than_the_strict_rule_never_stronger() -> None:
    """Ordering property the front-end comparison depends on: with floor <= peak, anything the
    strict rule keeps this rule must also keep. If that ever breaks, the 'gained' column in the
    viewer would be hiding losses behind a net count."""
    for ref in (0.0, 0.3, 0.6, 0.75, 0.9, 1.0):
        for seed in (0.0, 0.3, 0.6, 0.75, 0.9, 1.0):
            subject = scores(iou_dis_vs_phantom=ref, iou_seed_dis_vs_phantom=seed)
            if redetect.decide(subject, rule=BOTH)["verdict"] == KEEP:
                assert redetect.decide(subject, rule=FLOOR_PEAK,
                                       iou_floor=0.5)["verdict"] == KEEP, (ref, seed)


def test_floor_and_peak_flags_are_reported_under_every_rule() -> None:
    """So a floor can be chosen from a stored report with no GPU pass, whatever rule produced
    it -- the same reason the per-side IoUs are always reported."""
    for rule in redetect.RULES:
        ruling = redetect.decide(scores(iou_dis_vs_phantom=0.90,
                                        iou_seed_dis_vs_phantom=0.60), rule=rule)
        assert ruling["iou_floor_ok"] is True
        assert ruling["iou_peak_ok"] is True
        assert ruling["iou_floor_peak_ok"] is True


def test_no_box_short_circuits_before_the_floor() -> None:
    """A filtered side has no box to segment from, so it must drop on ``no_box`` regardless of
    what the other side's IoU says."""
    ruling = redetect.decide(
        scores(pick_ref=redetect.NO_BOX, iou_dis_vs_phantom=0.99,
               iou_seed_dis_vs_phantom=0.99), rule=FLOOR_PEAK)
    assert ruling["verdict"] == DROP
    assert ruling["no_box_sides"] == ["ref"]
    assert "iou_floor_ok" in ruling and "iou_peak_ok" in ruling


def test_the_floor_default_sits_below_the_peak() -> None:
    assert redetect.IOU_FLOOR_MIN < redetect.IOU_MIN


def test_floor_peak_is_registered_but_not_the_default() -> None:
    assert FLOOR_PEAK in redetect.RULES
    assert redetect.DEFAULT_RULE == redetect.RULE_IOU_STANDS


# ---------------------------------------------------------------------------------------
# RULE_IDENTITY_ONLY -- the text-free chain's rule
# ---------------------------------------------------------------------------------------

def text_free_scores(identity=0.7, **overrides):
    """A report row as tools/tighten_run.py writes it: identity and picks, no clip, no IoU.

    Built explicitly rather than by deleting keys from the detector-era fixture, because the
    absences are the point of these tests -- a fixture that grew a clip score later would make
    them pass for the wrong reason.
    """
    scores = {
        "dino_cos_chosen": identity,
        "pick_ref": redetect.FROM_DIS,
        "pick_seed": redetect.FROM_DIS,
        "pick_ref_reason": "sam2 tight box from phantom prompt",
        "pick_seed_reason": "sam2 tight box from phantom prompt",
    }
    scores.update(overrides)
    return scores


def test_identity_only_keeps_on_identity_alone_with_no_clip_or_iou_present():
    verdict = redetect.decide(text_free_scores(identity=0.8),
                              rule=redetect.RULE_IDENTITY_ONLY, identity_min=0.6)
    assert verdict["verdict"] == redetect.KEEP
    assert verdict["clip"] is None and verdict["iou"] is None


def test_identity_only_drops_below_the_threshold():
    verdict = redetect.decide(text_free_scores(identity=0.4),
                              rule=redetect.RULE_IDENTITY_ONLY, identity_min=0.6)
    assert verdict["verdict"] == redetect.DROP
    assert "0.4" in verdict["reason"]


def test_identity_only_drops_when_there_is_no_identity_score_at_all():
    # An unmeasured pair is not an endorsed one. This is the case that would silently ship
    # garbage if absence were treated as satisfaction.
    verdict = redetect.decide(text_free_scores(identity=None),
                              rule=redetect.RULE_IDENTITY_ONLY, identity_min=0.6)
    assert verdict["verdict"] == redetect.DROP
    assert "no identity score" in verdict["reason"]


def test_identity_only_still_drops_a_subject_missing_a_box():
    verdict = redetect.decide(
        text_free_scores(identity=0.95, pick_seed=redetect.NO_BOX),
        rule=redetect.RULE_IDENTITY_ONLY, identity_min=0.6)
    assert verdict["verdict"] == redetect.DROP
    assert verdict["no_box_sides"] == ["seed"]


def test_the_older_rules_drop_everything_on_a_text_free_row():
    # The protection that justifies a separate rule: each older rule ANDs a clip score, and on a
    # text-free report there is none. They must read that as failure, not as a free pass --
    # measured on the real report, all four return keep=0/140.
    for rule in (redetect.RULE_IDENTITY_REQUIRED, redetect.RULE_IOU_STANDS,
                 redetect.RULE_IOU_BOTH_SIDES, redetect.RULE_IOU_FLOOR_PEAK):
        verdict = redetect.decide(text_free_scores(identity=0.99), rule=rule,
                                  identity_min=0.6)
        assert verdict["verdict"] == redetect.DROP, rule


def test_identity_only_is_registered_and_byte_equivalent_for_the_other_rules():
    # Adding a fifth rule must not perturb the four that were already calibrated.
    assert redetect.RULE_IDENTITY_ONLY in redetect.RULES
    assert redetect.DEFAULT_RULE != redetect.RULE_IDENTITY_ONLY
    row = scores()
    for rule in (redetect.RULE_IDENTITY_REQUIRED, redetect.RULE_IOU_STANDS,
                 redetect.RULE_IOU_BOTH_SIDES, redetect.RULE_IOU_FLOOR_PEAK):
        assert redetect.decide(dict(row), rule=rule)["verdict"] in (
            redetect.KEEP, redetect.DROP)
