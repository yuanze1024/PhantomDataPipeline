"""Tests for the human label store.

These labels are the only ground truth the pipeline will ever have, and they are produced by a
person spending real time. So the properties worth testing are the ones that would silently lose
that work: a relabel that appends a second contradictory record instead of replacing the first,
a corrupt file that takes down the page mid-session, a filename built from an unvalidated id.
"""
from __future__ import annotations

import json

import pytest

from phantom_data import labels


def subject(sample_id: str = "a" * 32 + "_w000012345", subject_id: int = 1,
            **extra) -> dict:
    return {"sample_id": sample_id, "subject_id": subject_id, **extra}


# ---------------------------------------------------------------------------------------
# record shape
# ---------------------------------------------------------------------------------------

def test_a_record_carries_the_verdict_and_the_blind_flag():
    record = labels.make_record("abc", 2, labels.PASS, blind=True)
    assert record["verdict"] == labels.PASS
    assert record["blind"] is True
    assert record["subject_id"] == 2
    assert record["schema"] == labels.SCHEMA_VERSION


def test_blind_defaults_to_false_so_an_unmarked_label_is_not_claimed_as_blind():
    # The conservative default: mislabelling a score-influenced label as blind would overstate
    # the independence of the whole set.
    assert labels.make_record("abc", 0, labels.FAIL)["blind"] is False


def test_an_unknown_verdict_is_rejected_at_construction():
    with pytest.raises(ValueError):
        labels.make_record("abc", 0, "maybe")


def test_scores_are_snapshotted_because_the_report_is_going_to_be_regenerated():
    # The pilot's dis text was degraded (phantom_fallback on 140/140), so rerunning redetect
    # will move every clip score. Without the snapshot there is no way to tell which numbers a
    # label was written against.
    record = labels.make_record("abc", 0, labels.PASS,
                                scores={"rule_identity": 0.61, "rule_clip": 0.24})
    assert record["scores_at_label"]["rule_identity"] == 0.61


def test_scores_of_takes_the_report_field_names_and_skips_missing_ones():
    picked = labels.scores_of(subject(rule_identity=0.5, iou_dis_vs_phantom=0.9,
                                      candidates_ref_dis=3))
    assert picked == {"rule_identity": 0.5, "iou_dis_vs_phantom": 0.9,
                      "candidates_ref_dis": 3}
    assert "rule_clip" not in picked


# ---------------------------------------------------------------------------------------
# filenames
# ---------------------------------------------------------------------------------------

def test_the_filename_zero_pads_the_subject_id_like_the_enrich_cache():
    assert labels.label_name("xyz", 3) == "xyz_subj03.json"


def test_a_path_traversing_sample_id_cannot_become_a_filename():
    # A manifest is data, not code; a stray separator must not write outside the label dir.
    for hostile in ("../../etc/passwd", "a/b", "a\x00b", ""):
        with pytest.raises(ValueError):
            labels.label_name(hostile, 0)


# ---------------------------------------------------------------------------------------
# round trip and idempotence
# ---------------------------------------------------------------------------------------

def test_write_then_read_round_trips(tmp_path):
    labels.write_label(tmp_path, labels.make_record("s1", 0, labels.FAIL, note="box on the dog"))
    got = labels.read_label(tmp_path, "s1", 0)
    assert got["verdict"] == labels.FAIL
    assert got["note"] == "box on the dog"


def test_relabelling_replaces_rather_than_accumulating(tmp_path):
    # The property that makes one-file-per-subject worth the inode: an appended jsonl would
    # leave two contradictory records and no rule for which wins.
    labels.write_label(tmp_path, labels.make_record("s1", 0, labels.PASS))
    labels.write_label(tmp_path, labels.make_record("s1", 0, labels.FAIL))
    assert labels.read_label(tmp_path, "s1", 0)["verdict"] == labels.FAIL
    assert len(labels.load_labels(tmp_path)) == 1


def test_clearing_a_label_returns_the_subject_to_unlabelled(tmp_path):
    labels.write_label(tmp_path, labels.make_record("s1", 0, labels.PASS))
    assert labels.delete_label(tmp_path, "s1", 0) is True
    assert labels.read_label(tmp_path, "s1", 0) is None
    assert labels.delete_label(tmp_path, "s1", 0) is False


def test_reading_a_missing_label_is_none_not_an_error(tmp_path):
    assert labels.read_label(tmp_path, "nope", 0) is None
    assert labels.load_labels(tmp_path / "does_not_exist") == {}


# ---------------------------------------------------------------------------------------
# robustness: one bad file must not end a labelling session
# ---------------------------------------------------------------------------------------

def test_a_corrupt_file_is_skipped_rather_than_raising(tmp_path):
    (tmp_path / "broken_subj00.json").write_text("{not json", encoding="utf-8")
    labels.write_label(tmp_path, labels.make_record("good", 0, labels.PASS))
    assert list(labels.load_labels(tmp_path)) == [("good", 0)]


def test_a_record_from_a_future_schema_is_skipped_not_guessed_at(tmp_path):
    (tmp_path / "future_subj00.json").write_text(json.dumps(
        {"schema": labels.SCHEMA_VERSION + 1, "sample_id": "future", "subject_id": 0,
         "verdict": labels.PASS}), encoding="utf-8")
    assert labels.load_labels(tmp_path) == {}
    assert labels.read_label(tmp_path, "future", 0) is None


def test_a_record_with_an_unknown_verdict_is_skipped(tmp_path):
    (tmp_path / "weird_subj00.json").write_text(json.dumps(
        {"schema": labels.SCHEMA_VERSION, "sample_id": "weird", "subject_id": 0,
         "verdict": "probably"}), encoding="utf-8")
    assert labels.load_labels(tmp_path) == {}


# ---------------------------------------------------------------------------------------
# progress reporting
# ---------------------------------------------------------------------------------------

def test_the_summary_counts_passes_fails_and_blind_labels(tmp_path):
    labels.write_label(tmp_path, labels.make_record("s1", 0, labels.PASS, blind=True))
    labels.write_label(tmp_path, labels.make_record("s2", 0, labels.FAIL, blind=True))
    labels.write_label(tmp_path, labels.make_record("s3", 0, labels.FAIL))
    summary = labels.label_summary(labels.load_labels(tmp_path), total=140)
    assert summary == {"labelled": 3, "pass": 1, "fail": 2, "blind": 2,
                       "total": 140, "remaining": 137, "pass_rate": round(1 / 3, 4)}


def test_the_summary_omits_the_pass_rate_when_nothing_is_labelled():
    assert "pass_rate" not in labels.label_summary({})


# ---------------------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------------------

def test_next_unlabelled_wraps_so_a_resumed_session_finds_work_behind_it(tmp_path):
    subjects = [subject("s1", 0), subject("s2", 0), subject("s3", 0)]
    labels.write_label(tmp_path, labels.make_record("s2", 0, labels.PASS))
    labels.write_label(tmp_path, labels.make_record("s3", 0, labels.PASS))
    store = labels.load_labels(tmp_path)
    # Starting past the end of the list, the only unlabelled subject is at index 0.
    assert labels.next_unlabelled(subjects, store, start=2) == 0


def test_next_unlabelled_is_none_when_everything_is_done(tmp_path):
    subjects = [subject("s1", 0)]
    labels.write_label(tmp_path, labels.make_record("s1", 0, labels.PASS))
    assert labels.next_unlabelled(subjects, labels.load_labels(tmp_path)) is None


def test_next_unlabelled_handles_an_empty_subject_list():
    assert labels.next_unlabelled([], {}) is None


def test_labels_are_keyed_per_subject_not_per_sample(tmp_path):
    # A multi-subject sample must be labellable one subject at a time; keying on sample_id
    # alone would have the second subject overwrite the first.
    labels.write_label(tmp_path, labels.make_record("same", 0, labels.PASS))
    labels.write_label(tmp_path, labels.make_record("same", 1, labels.FAIL))
    store = labels.load_labels(tmp_path)
    assert store[("same", 0)]["verdict"] == labels.PASS
    assert store[("same", 1)]["verdict"] == labels.FAIL
