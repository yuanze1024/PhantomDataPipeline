"""Tests for the masklet browser's overlay and temporal geometry.

The overlay tests exist because a subtle overlay fails silently: it renders, the page loads, and
the reviewer concludes the mask is fine when they simply could not see it. A neutral-grey veil was
measured to change a mid-grey pixel by exactly 0 -- invisible on precisely the mid-tone indoor
frames this dataset is full of -- so visibility is asserted numerically rather than eyeballed.

The frame_series tests cover the temporal failures a still frame hides: a track that drops and is
reacquired, a mask that partly dissolves, and a mask that leaks onto a neighbouring object.
"""
from __future__ import annotations

import numpy as np

from phantom_data import masklet_viewer as mv


def square_mask(shape=(40, 40), x1=10, y1=10, x2=30, y2=30):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# ---------------------------------------------------------------------------------------
# overlay visibility
# ---------------------------------------------------------------------------------------

def test_the_veil_is_visible_at_every_frame_brightness():
    # The regression this guards: a neutral grey veil is a no-op on a mid-grey frame, because
    # blending 128 towards 128 changes nothing. A tinted veil shifts hue as well as luminance.
    mask = square_mask()
    for value in (5, 60, 128, 200, 250):
        frame = np.full((40, 40, 3), value, dtype=np.uint8)
        out = mv.draw_overlay(frame, mask, None)
        inside = out[20, 20].astype(int)
        outside = out[2, 2].astype(int)
        assert np.abs(inside - outside).sum() >= 30, f"veil too subtle at brightness {value}"


def test_pixels_outside_the_mask_are_untouched():
    frame = np.full((40, 40, 3), 77, dtype=np.uint8)
    out = mv.draw_overlay(frame, square_mask(), None)
    assert (out[0:9, 0:9] == 77).all()


def test_alpha_zero_is_a_no_op_and_alpha_one_is_a_solid_fill():
    frame = np.full((40, 40, 3), 100, dtype=np.uint8)
    mask = square_mask()
    assert mv.draw_overlay(frame, mask, None, alpha=0.0)[20, 20][0] == 100
    assert tuple(mv.draw_overlay(frame, mask, None, alpha=1.0)[20, 20]) == mv.MASK_FILL


def test_the_box_is_drawn_in_its_own_colour():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    out = mv.draw_overlay(frame, np.zeros((40, 40), bool), [10, 10, 30, 30])
    assert tuple(out[10, 20]) == mv.BOX_COLOUR, "top edge"
    assert tuple(out[20, 10]) == mv.BOX_COLOUR, "left edge"


def test_an_empty_mask_renders_without_raising():
    frame = np.full((40, 40, 3), 50, dtype=np.uint8)
    out = mv.draw_overlay(frame, np.zeros((40, 40), bool), None)
    assert (out == 50).all()


def test_a_box_flush_with_the_frame_edge_does_not_overflow():
    # bbox_from_mask is exclusive on max, so a mask touching the edge yields x2 == width.
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    mv.draw_overlay(frame, square_mask(x1=0, y1=0, x2=40, y2=40), [0, 0, 40, 40])


# ---------------------------------------------------------------------------------------
# temporal geometry
# ---------------------------------------------------------------------------------------

def test_a_steady_masklet_reports_full_coverage_and_flat_area():
    masks = np.stack([square_mask() for _ in range(10)])
    series = mv.frame_series(masks)
    assert series["frames_with_mask"] == 10
    assert series["interior_gaps"] == []
    assert series["area_min_ratio"] == 1.0 and series["area_max_ratio"] == 1.0


def test_an_interior_gap_is_reported_because_it_means_the_track_was_reacquired():
    # Distinct from a subject that simply leaves at the end: a hole in the middle means SAM2 lost
    # the object and found it again, and what it found again may not be the same object.
    masks = np.stack([square_mask(), np.zeros((40, 40), bool), square_mask()])
    series = mv.frame_series(masks)
    assert series["interior_gaps"] == [1]
    assert series["frames_with_mask"] == 2


def test_a_subject_leaving_at_the_end_is_not_an_interior_gap():
    masks = np.stack([square_mask(), square_mask(), np.zeros((40, 40), bool)])
    series = mv.frame_series(masks)
    assert series["interior_gaps"] == []
    assert series["last_frame"] == 1


def test_a_partly_dissolved_mask_shows_a_low_min_ratio():
    # The failure mode found on the real 30-sample run: one subject's mask collapsed to 3% of its
    # median area on some frames while never disappearing, so a presence check passed it.
    big = square_mask()
    tiny = square_mask(x1=10, y1=10, x2=12, y2=12)
    series = mv.frame_series(np.stack([big, big, tiny, big]))
    assert series["area_min_ratio"] < 0.1
    assert series["frames_with_mask"] == 4, "presence alone would call this healthy"


def test_a_leaked_mask_shows_a_high_max_ratio():
    small = square_mask()
    huge = square_mask(x1=0, y1=0, x2=40, y2=40)
    series = mv.frame_series(np.stack([small, small, huge]))
    assert series["area_max_ratio"] > 2.0


def test_ratios_are_relative_to_the_median_not_the_max():
    # One leaked frame would inflate the max and make the rest of the series look stable by
    # comparison, hiding a genuine dissolve elsewhere in the same masklet.
    normal = square_mask()
    huge = square_mask(x1=0, y1=0, x2=40, y2=40)
    series = mv.frame_series(np.stack([normal, normal, normal, huge]))
    assert series["area_median"] == float(normal.sum())


def test_the_union_box_spans_every_frame():
    left = square_mask(x1=0, y1=10, x2=15, y2=30)
    right = square_mask(x1=25, y1=10, x2=40, y2=30)
    series = mv.frame_series(np.stack([left, right]))
    assert series["union_box"] == [0, 10, 40, 30]


def test_an_all_empty_masklet_reports_nothing_present_without_dividing_by_zero():
    series = mv.frame_series(np.zeros((5, 40, 40), dtype=bool))
    assert series["frames_with_mask"] == 0
    assert series["first_frame"] is None
    assert series["area_min_ratio"] == 0.0
    assert series["union_box"] is None


# ---------------------------------------------------------------------------------------
# strip selection
# ---------------------------------------------------------------------------------------

def test_the_strip_always_includes_the_seed_frame():
    # The seed frame is where the box came from, so it is the one frame that must be shown.
    for seed in (0, 7, 40, 80):
        assert seed in mv.strip_indices(81, seed, count=8)


def test_the_strip_is_sorted_spans_the_clip_and_respects_the_count():
    picks = mv.strip_indices(81, None, count=8)
    assert picks == sorted(picks)
    assert picks[0] == 0 and picks[-1] == 80
    assert len(picks) == 8


def test_a_short_clip_yields_every_frame():
    assert mv.strip_indices(5, 2, count=8) == [0, 1, 2, 3, 4]
