"""Tests for the threshold-picking browser.

The page exists to answer one question -- where along the score axis does good turn into bad -- so
the properties worth testing are the ones that would quietly misinform that decision: a window
that shows the wrong subjects for a cut, or a histogram whose "cost of cutting here" column is off
by a bin. Both would look plausible on screen.
"""
from __future__ import annotations

import numpy as np

from phantom_data import threshold_viewer as tv


def subject(score, sample="s", sid=0):
    return {"sample_id": sample, "subject_id": sid, "rule_identity": score,
            "dis": "thing", "ranking": {"margin": 0.1, "used_detector": False}}


def report(scores):
    return {"subjects": [subject(v, sample=f"s{i}") for i, v in enumerate(scores)]}


# ---------------------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------------------

def test_subjects_are_ordered_worst_first():
    # Ascending on purpose: the threshold's job is to cut off a tail, and a page opening on the
    # best subjects shows nothing about where that tail starts.
    ordered = tv.scored_subjects(report([0.9, 0.1, 0.5]))
    assert [s["rule_identity"] for s in ordered] == [0.1, 0.5, 0.9]


def test_subjects_without_a_score_are_excluded():
    data = report([0.5])
    data["subjects"].append({"sample_id": "x", "subject_id": 0, "rule_identity": None})
    assert len(tv.scored_subjects(data)) == 1


# ---------------------------------------------------------------------------------------
# the window around the cut
# ---------------------------------------------------------------------------------------

def test_the_window_splits_on_the_cut_with_the_boundary_inclusive_above():
    # `>= cut` keeps, matching redetect.decide, so a subject exactly at the cut must appear on the
    # kept side. Off-by-one here would show the reviewer a pair that ships as though it were being
    # discarded.
    subjects = tv.scored_subjects(report([0.3, 0.49, 0.5, 0.7]))
    below, above = tv.window_around(subjects, 0.5, window=4)
    assert [s["rule_identity"] for s in below] == [0.49, 0.3]
    assert [s["rule_identity"] for s in above] == [0.5, 0.7]


def test_both_sides_read_outward_from_the_boundary():
    # Nearest-first in both directions: the pairs that decide whether a cut is right are the ones
    # closest to it, so they must be the ones the eye lands on first.
    subjects = tv.scored_subjects(report([0.1, 0.2, 0.3, 0.6, 0.7, 0.8]))
    below, above = tv.window_around(subjects, 0.5, window=2)
    assert [s["rule_identity"] for s in below] == [0.3, 0.2]
    assert [s["rule_identity"] for s in above] == [0.6, 0.7]


def test_the_window_is_capped_at_the_requested_size():
    subjects = tv.scored_subjects(report([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]))
    below, above = tv.window_around(subjects, 0.5, window=2)
    assert len(below) == 2 and len(above) == 2


def test_a_cut_past_every_score_leaves_the_kept_side_empty():
    subjects = tv.scored_subjects(report([0.1, 0.2]))
    below, above = tv.window_around(subjects, 1.0, window=4)
    assert len(below) == 2 and above == []


def test_a_cut_below_every_score_leaves_the_discarded_side_empty():
    subjects = tv.scored_subjects(report([0.1, 0.2]))
    below, above = tv.window_around(subjects, 0.0, window=4)
    assert below == [] and len(above) == 2


# ---------------------------------------------------------------------------------------
# histogram and the cost column
# ---------------------------------------------------------------------------------------

def test_each_score_lands_in_exactly_one_bin():
    subjects = tv.scored_subjects(report([0.04, 0.05, 0.99, 1.0]))
    rows = tv.histogram(subjects, bins=20)
    assert sum(row["subjects"] for row in rows) == 4, "no score lost, none double counted"


def test_a_perfect_score_is_not_dropped_from_the_last_bin():
    # 1.0 falls on the upper edge; a half-open last bin would silently omit it.
    rows = tv.histogram(tv.scored_subjects(report([1.0])), bins=20)
    assert rows[-1]["subjects"] == 1


def test_the_cost_column_counts_everything_strictly_below_the_bin():
    # This column converts a bin count into the actual cost of cutting there, which is the number
    # the decision turns on -- an off-by-one bin would misstate the discard rate.
    subjects = tv.scored_subjects(report([0.02, 0.07, 0.12, 0.9]))
    rows = tv.histogram(subjects, bins=20)
    assert rows[0]["cut here → discarded"].startswith("0 ")   # cut at 0.00 discards nothing
    assert rows[1]["cut here → discarded"].startswith("1 ")   # cut at 0.05 discards the 0.02
    assert rows[2]["cut here → discarded"].startswith("2 ")   # cut at 0.10 discards 0.02, 0.07


def test_the_histogram_handles_no_subjects_without_dividing_by_zero():
    rows = tv.histogram([], bins=5)
    assert len(rows) == 5
    assert all(row["subjects"] == 0 for row in rows)


# ---------------------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------------------

def test_the_crop_is_clipped_to_the_frame():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    crop = tv.crop_box(frame, [40, 40, 200, 200])
    assert crop.shape == (10, 10, 3)


def test_a_degenerate_or_missing_box_yields_no_crop():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert tv.crop_box(frame, None) is None
    assert tv.crop_box(frame, [10, 10, 10, 10]) is None
    assert tv.crop_box(frame, [60, 60, 70, 70]) is None
