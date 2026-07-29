"""Tests for row shaping in ``tools/gate_apply.py`` -- the step that writes boxes back.

Pure, no GPU, no report on disk: the row construction is factored out of the CLI precisely so
this file can assert it. What is load-bearing:

* **The output must be shape-compatible with stage C's ``parse_sample``.** That is asserted by
  actually calling ``parse_sample`` on the gated rows rather than by listing field names, so a
  change to stage C's requirements fails here instead of at the next 8-hour SAM2 run.
* **``box_space: "frame"`` must be on every row.** The chosen boxes are already frame pixels; if
  stage C maps them through the annotation canvas again they are rescaled 2.5x on HD and nothing
  raises. The tag is the only thing standing between the corrected boxes and that.
* **Dropped subjects must not leak through.** A subject with no report record is dropped, not
  passed along with its original box -- passing it would ship exactly the uncorrected box the
  pipeline exists to remove.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from phantom_data import redetect
from phantom_data.build import segment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import gate_apply  # noqa: E402


def subject(subject_id: int = 1) -> dict:
    """One stage B subject, with the raw annotation boxes it arrives with."""
    return {
        "subject_id": subject_id,
        "phrase": "a woman in a red coat",
        "bbox_cls": "woman",
        "seed_frame_index": 10,
        "seed_abs_time": 193.7313,
        "seed_bbox_768": [226.0, 14.0, 797.0, 394.0],
        "ref": {
            "frame": f"ref_frames/vid_subj{subject_id:02d}.jpg",
            "bbox_768": [171.0, 0.0, 649.0, 432.0],
            "bbox_cls": "woman",
            "ref_frame_width": 1920,
            "ref_frame_height": 1080,
        },
    }


def row(*subject_ids: int) -> dict:
    """A stage B manifest row."""
    return {
        "sample_id": "vid_w000000001", "video_id": "vid", "video": "clips/vid.mp4",
        "caption": "a clip", "width": 1280, "height": 720, "frame_count": 81,
        "subjects": [subject(sid) for sid in (subject_ids or (1,))],
    }


def record(subject_id: int = 1, verdict_inputs: dict | None = None) -> dict:
    """A gate_report subject row that ``decide`` keeps: identity, clip and IoU all pass.

    The chosen boxes are deliberately *unlike* the annotation boxes in ``subject()`` and sized
    like real frame coordinates on a 1280x720 clip, so a test that confuses the two can be seen.
    """
    base = {
        "sample_id": "vid_w000000001", "subject_id": subject_id,
        "dis": "a woman in a red coat",
        "pick_ref": redetect.FROM_DIS, "pick_seed": redetect.FROM_DIS,
        "pick_ref_reason": "detector box preferred",
        "pick_seed_reason": "detector box preferred",
        "chosen_box_ref": [426.8, 4.1, 1534.5, 1075.5],
        "chosen_box_seed": [263.0, 35.9, 1179.2, 675.1],
        "chosen_clip_ref": 0.30, "chosen_clip_seed": 0.29,
        "dino_cos_chosen": 0.80,
        "iou_dis_vs_phantom": 0.90, "iou_seed_dis_vs_phantom": 0.88,
        "rule_identity": 0.80, "rule_clip": 0.30, "rule_iou": 0.90,
    }
    return {**base, **(verdict_inputs or {})}


# ----- corrected_subject ---------------------------------------------------------------


def test_the_corrected_boxes_replace_the_annotation_boxes() -> None:
    result = gate_apply.corrected_subject(subject(), record())
    assert result["seed_bbox_768"] == [263.0, 35.9, 1179.2, 675.1]
    assert result["ref"]["bbox_768"] == [426.8, 4.1, 1534.5, 1075.5]


def test_the_original_subject_is_not_mutated() -> None:
    """The caller still holds the stage B row; writing through it would corrupt the drop list."""
    original = subject()
    gate_apply.corrected_subject(original, record())
    assert original["seed_bbox_768"] == [226.0, 14.0, 797.0, 394.0]
    assert original["ref"]["bbox_768"] == [171.0, 0.0, 649.0, 432.0]


def test_the_rest_of_the_subject_carries_through() -> None:
    """Stage C needs these and they are not ours to change."""
    result = gate_apply.corrected_subject(subject(), record())
    assert result["seed_frame_index"] == 10
    assert result["phrase"] == "a woman in a red coat"
    assert result["ref"]["frame"] == "ref_frames/vid_subj01.jpg"
    assert result["ref"]["ref_frame_width"] == 1920


def test_the_gate_numbers_are_attached_for_provenance() -> None:
    gate = gate_apply.corrected_subject(subject(), record())["gate"]
    assert gate["identity"] == 0.80
    assert gate["pick_ref"] == redetect.FROM_DIS
    assert gate["dis"] == "a woman in a red coat"


# ----- gated_row ----------------------------------------------------------------------


def test_a_kept_subject_produces_a_row_tagged_frame() -> None:
    gated, dropped = gate_apply.gated_row(row(), [record()])
    assert gated["box_space"] == segment.BOX_SPACE_FRAME
    assert dropped == []


def test_the_gated_row_parses_as_a_stage_c_sample() -> None:
    """The contract assertion: stage C itself accepts the row, and reads the corrected box.

    ``parse_sample`` plus ``resolve_box`` together are the whole path from manifest to the box
    SAM2 is prompted with, so running both is what proves the round trip. The seed box comes out
    clamped to the 1280x720 clip and *not* rescaled.
    """
    gated, _ = gate_apply.gated_row(row(), [record()])
    spec = segment.parse_sample(gated)
    assert spec.subjects[0].box_space == segment.BOX_SPACE_FRAME
    assert segment.resolve_box(spec.subjects[0].seed_bbox_768, 1280, 720,
                              spec.subjects[0].box_space) == [263.0, 35.9, 1179.2, 675.1]


def test_a_dropped_subject_is_removed_with_its_gate_named() -> None:
    """Identity does nearly all the dropping (62/62 on the pilot), so it is the case to name."""
    gated, dropped = gate_apply.gated_row(row(), [record(verdict_inputs={
        "dino_cos_chosen": 0.30, "rule_identity": 0.30,
        "iou_dis_vs_phantom": 0.10, "iou_seed_dis_vs_phantom": 0.10, "rule_iou": 0.10})])
    assert gated is None
    assert [item["gate"] for item in dropped] == ["identity"]


def test_a_sample_with_no_surviving_subject_yields_no_row() -> None:
    """Stage C raises on a subject-less sample, so emitting one turns a filter into a failure."""
    gated, _ = gate_apply.gated_row(row(), [])
    assert gated is None


def test_a_subject_without_a_report_record_is_dropped_not_passed_through() -> None:
    """The silent-corruption case: passing it through ships the uncorrected Phantom box."""
    gated, dropped = gate_apply.gated_row(row(1, 2), [record(1)])
    assert [s["subject_id"] for s in gated["subjects"]] == [1]
    assert [(d["subject_id"], d["gate"]) for d in dropped] == [(2, "no_record")]


def test_a_missing_chosen_box_is_dropped_even_if_the_rule_kept_it() -> None:
    """Belt and braces: no row with a None coordinate can reach SAM2."""
    gated, dropped = gate_apply.gated_row(row(), [record(verdict_inputs={
        "chosen_box_ref": None})])
    assert gated is None
    assert [item["gate"] for item in dropped] == ["no_box"]


def test_a_no_box_pick_is_dropped_and_counted_as_no_box() -> None:
    gated, dropped = gate_apply.gated_row(row(), [record(verdict_inputs={
        "pick_ref": redetect.NO_BOX, "chosen_box_ref": None, "chosen_clip_ref": None})])
    assert gated is None
    assert [item["gate"] for item in dropped] == ["no_box"]


def test_previously_dropped_subjects_are_preserved_alongside_the_new_ones() -> None:
    """Stage B's own ``dropped_subjects`` (e.g. seed_outside_window) is not overwritten."""
    source = {**row(1, 2), "dropped_subjects": [{"subject_id": 9, "reason": "seed_outside_window"}]}
    gated, _ = gate_apply.gated_row(source, [record(1)])
    reasons = [item.get("reason") for item in gated["dropped_subjects"]]
    assert "seed_outside_window" in reasons
    assert len(gated["dropped_subjects"]) == 2


def test_the_rule_and_thresholds_are_honoured() -> None:
    """A subject kept under ``iou_stands`` and dropped under ``identity_required``.

    Identity fails at 0.30 but both IoUs pass, which is exactly the substitution the two rules
    disagree about -- so it also proves the rule argument is actually threaded through.
    """
    failing_identity = record(verdict_inputs={"dino_cos_chosen": 0.30, "rule_identity": 0.30,
                                              "chosen_clip_ref": 0.10, "chosen_clip_seed": 0.10,
                                              "rule_clip": 0.10})
    kept, _ = gate_apply.gated_row(row(), [failing_identity], rule=redetect.RULE_IOU_STANDS)
    dropped_row, _ = gate_apply.gated_row(row(), [failing_identity],
                                          rule=redetect.RULE_IDENTITY_REQUIRED)
    assert kept is not None
    assert dropped_row is None


# ----- apply_gate ---------------------------------------------------------------------


def test_apply_gate_counts_the_funnel() -> None:
    rows = [row(1, 2), {**row(1), "sample_id": "vid_w000000002"}]
    records = [record(1), record(2, verdict_inputs={"dino_cos_chosen": 0.20,
                                                   "rule_identity": 0.20, "rule_iou": 0.10,
                                                   "iou_dis_vs_phantom": 0.10,
                                                   "iou_seed_dis_vs_phantom": 0.10})]
    result = gate_apply.apply_gate(rows, records)
    summary = result["summary"]
    assert summary["samples_in"] == 2
    # The second sample's subjects have no records for that sample_id, so it empties out.
    assert summary["samples_out"] == 1
    assert summary["subjects_in"] == 3
    assert summary["subjects_out"] == 1
    assert summary["dropped_by_gate"] == {"identity": 1, "no_record": 1}


def test_apply_gate_matches_records_by_sample_not_just_subject_id() -> None:
    """subject_id is only unique within a sample; keying globally would cross-contaminate."""
    rows = [row(1), {**row(1), "sample_id": "other"}]
    result = gate_apply.apply_gate(rows, [record(1)])
    assert [r["sample_id"] for r in result["rows"]] == ["vid_w000000001"]


def test_every_emitted_row_is_tagged_and_parses() -> None:
    result = gate_apply.apply_gate([row(1, 2)], [record(1), record(2)])
    for emitted in result["rows"]:
        assert emitted["box_space"] == segment.BOX_SPACE_FRAME
        segment.parse_sample(emitted)


# ----- gate_reason --------------------------------------------------------------------


@pytest.mark.parametrize("ruling,expected", [
    ({"no_box_sides": ["ref"], "identity_ok": True, "clip_ok": True, "iou_ok": True}, "no_box"),
    ({"identity_ok": False, "clip_ok": True, "iou_ok": True}, "identity"),
    ({"identity_ok": True, "clip_ok": False, "iou_ok": False}, "clip_and_iou"),
    ({"identity_ok": True, "clip_ok": False, "iou_ok": True}, "clip"),
    ({"identity_ok": True, "clip_ok": True, "iou_ok": False}, "iou"),
    ({"identity_ok": True, "clip_ok": True, "iou_ok": True}, "unknown"),
])
def test_gate_reason_names_the_gate_that_rejected(ruling, expected) -> None:
    """``no_box`` outranks identity: an abstention is not a measurement of the pair."""
    assert gate_apply.gate_reason(ruling) == expected


# ----- --keep-all (inspection mode) ---------------------------------------------------
#
# The flag exists so a human can look at the mask a corrected box produced *before* the gate
# decides anything: a low identity score can mean "genuinely a different object" or "the crop
# was bad", and filtering first removes exactly the evidence that separates those. What must
# hold is that it changes only *which* subjects are emitted -- never the boxes, never the
# ``box_space`` tag, never the default behaviour -- and that the verdict travels with the row so
# the renderer can label what the gate would have dropped.


def rejected_record(subject_id: int = 1) -> dict:
    """A record ``decide`` drops under either rule: identity, clip and IoU all fail."""
    return record(subject_id, verdict_inputs={
        "dino_cos_chosen": 0.30, "rule_identity": 0.30,
        "chosen_clip_ref": 0.10, "chosen_clip_seed": 0.10, "rule_clip": 0.10,
        "iou_dis_vs_phantom": 0.10, "iou_seed_dis_vs_phantom": 0.10, "rule_iou": 0.10})


def test_keep_all_emits_a_subject_the_rule_rejected() -> None:
    gated, dropped = gate_apply.gated_row(row(), [rejected_record()], keep_all=True)
    assert gated is not None
    assert [s["subject_id"] for s in gated["subjects"]] == [1]
    assert [item["gate"] for item in dropped] == ["identity"]


def test_keep_all_records_the_verdict_on_the_emitted_subject() -> None:
    """The renderer's only source for "the gate would have dropped this one"."""
    gated, _ = gate_apply.gated_row(row(), [rejected_record()], keep_all=True)
    gate = gated["subjects"][0]["gate"]
    assert gate["verdict"] == redetect.DROP
    assert gate["gate_reason"] == "identity"
    assert "identity" in gate["verdict_reason"]


def test_keep_all_marks_a_kept_subject_as_kept() -> None:
    gated, dropped = gate_apply.gated_row(row(), [record()], keep_all=True)
    gate = gated["subjects"][0]["gate"]
    assert gate["verdict"] == redetect.KEEP
    assert gate["gate_reason"] is None
    assert dropped == []


def test_keep_all_flags_the_drop_entry_as_kept_anyway() -> None:
    """The funnel must stay countable: the verdict is still recorded, with an override note."""
    _, dropped = gate_apply.gated_row(row(), [rejected_record()], keep_all=True)
    assert dropped[0]["kept_anyway"] is True


def test_keep_all_still_uses_the_corrected_boxes_and_the_frame_tag() -> None:
    """The whole point of the run: rejected subjects are segmented from corrected boxes too.

    If ``box_space`` were lost here, stage C would re-map these frame coordinates through the
    annotation canvas and silently misplace every box.
    """
    gated, _ = gate_apply.gated_row(row(), [rejected_record()], keep_all=True)
    assert gated["box_space"] == segment.BOX_SPACE_FRAME
    spec = segment.parse_sample(gated)
    assert spec.subjects[0].seed_bbox_768 == [263.0, 35.9, 1179.2, 675.1]
    assert segment.resolve_box(spec.subjects[0].seed_bbox_768, 1280, 720,
                               spec.subjects[0].box_space) == [263.0, 35.9, 1179.2, 675.1]


def test_keep_all_does_not_resurrect_a_subject_without_a_report_record() -> None:
    """``no_record`` has no corrected box at all, so keeping it would ship the Phantom box."""
    gated, dropped = gate_apply.gated_row(row(1, 2), [record(1)], keep_all=True)
    assert [s["subject_id"] for s in gated["subjects"]] == [1]
    assert [(d["subject_id"], d["gate"]) for d in dropped] == [(2, "no_record")]


def test_keep_all_does_not_resurrect_a_subject_with_a_missing_chosen_box() -> None:
    """Structural, not a threshold: there is no box to segment from, kept or not."""
    missing = record(verdict_inputs={"pick_ref": redetect.NO_BOX, "chosen_box_ref": None,
                                     "chosen_clip_ref": None})
    gated, dropped = gate_apply.gated_row(row(), [missing], keep_all=True)
    assert gated is None
    assert [item["gate"] for item in dropped] == ["no_box"]
    assert not any(item.get("kept_anyway") for item in dropped)


def test_keep_all_leaves_the_default_path_untouched() -> None:
    """Same input, flag off: byte-identical to the pre-flag behaviour."""
    gated, dropped = gate_apply.gated_row(row(), [rejected_record()])
    assert gated is None
    assert [item["gate"] for item in dropped] == ["identity"]
    assert "kept_anyway" not in dropped[0]


def test_the_default_row_carries_no_verdict_fields() -> None:
    """``gate`` keeps its original shape when the flag is off, so nothing downstream shifts."""
    gated, _ = gate_apply.gated_row(row(), [record()])
    assert set(gated["subjects"][0]["gate"]) == {"pick_ref", "pick_seed", "identity", "clip",
                                                 "iou", "dis"}


def test_apply_gate_keep_all_counts_verdicts_separately_from_withheld_subjects() -> None:
    result = gate_apply.apply_gate([row(1, 2)], [record(1), rejected_record(2)], keep_all=True)
    summary = result["summary"]
    assert summary["keep_all"] is True
    assert summary["subjects_out"] == 2
    assert summary["subjects_dropped"] == 1
    assert summary["subjects_kept_anyway"] == 1


def test_apply_gate_default_reports_keep_all_false_and_no_overrides() -> None:
    result = gate_apply.apply_gate([row(1, 2)], [record(1), rejected_record(2)])
    summary = result["summary"]
    assert summary["keep_all"] is False
    assert summary["subjects_out"] == 1
    assert summary["subjects_kept_anyway"] == 0


def test_keep_all_emits_every_subject_the_report_has_boxes_for() -> None:
    """The run's actual requirement: nothing gated out, so the user sees the dropped ones too."""
    rows = [row(1, 2), {**row(1), "sample_id": "vid_w000000002"}]
    records = [record(1), rejected_record(2),
               {**rejected_record(1), "sample_id": "vid_w000000002"}]
    result = gate_apply.apply_gate(rows, records, keep_all=True)
    assert result["summary"]["subjects_in"] == result["summary"]["subjects_out"] == 3
    assert result["summary"]["samples_out"] == 2
