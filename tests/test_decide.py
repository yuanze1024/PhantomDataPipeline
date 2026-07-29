"""Tests for the keep/drop rule in :func:`phantom_data.redetect.decide`.

Box *selection* is the candidate pool's job (see test_candidates.py); this file is only about
which pairs survive. The chain has one judge -- is the reference the same instance as the target --
because the box now comes from SAM2 segmenting what a proposal pointed at, so "is the box on the
intended object" is answered by construction rather than by a confirming score.

The properties worth pinning down are the conservative ones. Every case where the rule lacks
evidence must drop rather than keep: a missing box, a missing score, an unknown rule name. Four
earlier rules ANDed a CLIP text score and an IoU-vs-annotation, and the failure mode that made
them dangerous once those judges were withdrawn was silently reading *absent* as *satisfied*.
"""
from __future__ import annotations

import pytest

from phantom_data import redetect
from phantom_data.redetect import DROP, KEEP


def scores(**overrides):
    """A subject whose two sides both ship a chain-derived box."""
    base = {
        "pick_ref": redetect.FROM_DIS, "pick_seed": redetect.FROM_DIS,
        "pick_ref_reason": "phantom_box, ranked 1 of 2 by id-sim",
        "pick_seed_reason": "detector_box, ranked 1 of 3 by id-sim",
        "dino_cos_chosen": 0.70,
        "chosen_box_ref": [1, 1, 9, 9], "chosen_box_seed": [2, 2, 9, 9],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------------------
# the identity gate
# ---------------------------------------------------------------------------------------

def test_a_pair_above_the_threshold_is_kept() -> None:
    verdict = redetect.decide(scores(dino_cos_chosen=0.8), identity_min=0.2)
    assert verdict["verdict"] == KEEP
    assert verdict["identity_ok"] is True


def test_a_pair_below_the_threshold_is_dropped() -> None:
    verdict = redetect.decide(scores(dino_cos_chosen=0.1), identity_min=0.2)
    assert verdict["verdict"] == DROP
    assert "0.100" in verdict["reason"]


def test_the_threshold_is_inclusive() -> None:
    assert redetect.decide(scores(dino_cos_chosen=0.2), identity_min=0.2)["verdict"] == KEEP


def test_just_below_the_threshold_drops() -> None:
    assert redetect.decide(scores(dino_cos_chosen=0.199), identity_min=0.2)["verdict"] == DROP


def test_a_missing_identity_score_drops() -> None:
    # An unmeasured pair is not an endorsed one. This is the case that would ship garbage if
    # absence were treated as satisfaction.
    verdict = redetect.decide(scores(dino_cos_chosen=None), identity_min=0.2)
    assert verdict["verdict"] == DROP
    assert "no identity score" in verdict["reason"]


def test_a_negative_similarity_drops() -> None:
    # ID-Sim similarity is 1 - distance and can go below zero; two pilot subjects did.
    assert redetect.decide(scores(dino_cos_chosen=-0.11), identity_min=0.2)["verdict"] == DROP


# ---------------------------------------------------------------------------------------
# missing boxes outrank the score
# ---------------------------------------------------------------------------------------

def test_a_side_with_no_box_is_dropped_however_good_the_identity() -> None:
    # Guarded on the *pick*, not the scores: a subject missing only its reference box would
    # otherwise carry a fine identity score and ship as a pair with one box missing.
    verdict = redetect.decide(
        scores(dino_cos_chosen=0.99, pick_seed=redetect.NO_BOX), identity_min=0.2)
    assert verdict["verdict"] == DROP
    assert verdict["no_box_sides"] == ["seed"]


def test_both_sides_missing_are_both_reported() -> None:
    verdict = redetect.decide(
        scores(pick_ref=redetect.NO_BOX, pick_seed=redetect.NO_BOX), identity_min=0.2)
    assert verdict["no_box_sides"] == ["ref", "seed"]


def test_a_kept_pair_carries_no_no_box_key() -> None:
    assert "no_box_sides" not in redetect.decide(scores(), identity_min=0.2)


# ---------------------------------------------------------------------------------------
# the on-disk contract
# ---------------------------------------------------------------------------------------

def test_the_withdrawn_judges_remain_as_nulls() -> None:
    # Stage C carries this dict through as provenance without reading it, so the shape is part of
    # the on-disk contract even though the CLIP and IoU judges are gone. Dropping the keys would
    # change the manifest shape for reports already written.
    verdict = redetect.decide(scores(), identity_min=0.2)
    for key in ("clip", "clip_ok", "iou", "iou_ok"):
        assert key in verdict and verdict[key] is None


def test_the_verdict_records_which_box_set_shipped() -> None:
    verdict = redetect.decide(scores(), identity_min=0.2)
    assert verdict["box_set"] == redetect.FROM_DIS
    assert "ranked 1 of" in verdict["box_reason"]


def test_mixed_box_sources_are_named_per_side() -> None:
    verdict = redetect.decide(
        scores(pick_ref=redetect.PHANTOM), identity_min=0.2)
    assert verdict["box_set"] == "ref=phantom, target=dis"


def test_the_rule_name_is_echoed() -> None:
    assert redetect.decide(scores())["rule"] == redetect.RULE_IDENTITY_ONLY


# ---------------------------------------------------------------------------------------
# the rule registry
# ---------------------------------------------------------------------------------------

def test_identity_only_is_the_default_and_the_only_rule() -> None:
    assert redetect.DEFAULT_RULE == redetect.RULE_IDENTITY_ONLY
    assert redetect.RULES == (redetect.RULE_IDENTITY_ONLY,)


def test_an_unknown_rule_is_rejected_rather_than_silently_defaulted() -> None:
    # The four withdrawn rules' names are the likely inputs here -- an old command line or a
    # stale report. Falling back to the default would apply a different rule than asked for.
    for name in ("iou_stands", "identity_required", "iou_both_sides", "iou_floor_peak", ""):
        with pytest.raises(ValueError, match="unknown rule"):
            redetect.decide(scores(), rule=name)


def test_withdrawn_threshold_arguments_are_accepted_and_ignored() -> None:
    # gate_apply records every threshold it was invoked with; passing them must not error, and
    # must not change the verdict.
    verdict = redetect.decide(scores(dino_cos_chosen=0.3), identity_min=0.2,
                              clip_min=0.21, iou_min=0.75, iou_floor=0.5)
    assert verdict["verdict"] == KEEP
    assert verdict["clip"] is None and verdict["iou"] is None
