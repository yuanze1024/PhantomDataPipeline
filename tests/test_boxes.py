"""Pure tests for :mod:`phantom_data.boxes`.

Two things are load-bearing and asserted explicitly:

* :func:`crop_box` must never return an empty array, whatever the box. An empty crop
  reaches CLIP as a zero-size PIL image and raises deep inside the processor, which in the
  cascade would look like a model failure rather than a bad annotation.
* :func:`box_fraction` must *not* clamp. It is the field that makes an oversized annotation
  visible in the report; clamping would report 1.0 for every overflowing box and hide it.
"""
from __future__ import annotations

import numpy as np
import pytest

from phantom_data.boxes import box_fraction, clamp_box, crop_box, iou, is_box


def frame(width: int = 100, height: int = 50) -> np.ndarray:
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


# ----- is_box -------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, [], [1, 2, 3], [1, 2, 3, 4, 5], "1,2,3,4",
                                   [1, 2, 3, "4"], [True, 2, 3, 4], 4])
def test_is_box_rejects_non_boxes(value) -> None:
    assert is_box(value) is False


@pytest.mark.parametrize("value", [[0, 0, 1, 1], (0.5, 1.5, 2.5, 3.5), [-10, -10, 999, 999]])
def test_is_box_accepts_four_numbers(value) -> None:
    assert is_box(value) is True


# ----- clamp_box ----------------------------------------------------------------------


def test_clamp_box_leaves_an_interior_box_alone() -> None:
    assert clamp_box([10, 20, 30, 40], 100, 50) == [10, 20, 30, 40]


def test_clamp_box_rounds_to_nearest_pixel() -> None:
    assert clamp_box([10.4, 20.6, 30.5, 39.49], 100, 50) == [10, 21, 30, 39]


def test_clamp_box_clips_overflow_to_the_frame() -> None:
    assert clamp_box([-20, -5, 400, 900], 100, 50) == [0, 0, 100, 50]


def test_clamp_box_keeps_a_zero_area_box_non_empty() -> None:
    """x2 == x1 would slice to nothing; the contract is x2 > x1 always."""
    assert clamp_box([30, 20, 30, 20], 100, 50) == [30, 20, 31, 21]


def test_clamp_box_survives_an_inverted_box() -> None:
    x1, y1, x2, y2 = clamp_box([80, 40, 10, 5], 100, 50)
    assert x2 > x1 and y2 > y1


def test_clamp_box_survives_a_box_entirely_off_frame() -> None:
    assert clamp_box([500, 500, 600, 600], 100, 50) == [99, 49, 100, 50]


@pytest.mark.parametrize("value", [None, [1, 2, 3], "box"])
def test_clamp_box_returns_none_for_non_boxes(value) -> None:
    assert clamp_box(value, 100, 50) is None


# ----- crop_box -----------------------------------------------------------------------


def test_crop_box_returns_the_boxed_pixels() -> None:
    source = frame()
    crop = crop_box(source, [10, 20, 30, 40])
    assert crop.shape == (20, 20, 3)
    assert np.array_equal(crop, source[20:40, 10:30])


@pytest.mark.parametrize("box", [[30, 20, 30, 20], [-50, -50, -10, -10], [500, 500, 600, 600],
                                 [80, 40, 10, 5], [0, 0, 1000, 1000]])
def test_crop_box_is_never_empty(box) -> None:
    """Whatever the annotation says, CLIP downstream must get a real image."""
    crop = crop_box(frame(), box)
    assert crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0


def test_crop_box_returns_none_for_a_missing_box() -> None:
    assert crop_box(frame(), None) is None


def test_crop_box_output_is_uint8() -> None:
    assert crop_box(frame().astype(np.float32), [0, 0, 5, 5]).dtype == np.uint8


# ----- box_fraction -------------------------------------------------------------------


def test_box_fraction_measures_against_frame_area() -> None:
    # 20x10 box in a 100x50 frame -> 200 / 5000.
    assert box_fraction([0, 0, 20, 10], frame()) == pytest.approx(0.04)


def test_box_fraction_does_not_clamp() -> None:
    """An annotation twice the frame's size reports >1.0 rather than looking like a fit."""
    assert box_fraction([0, 0, 200, 50], frame()) == pytest.approx(2.0)


def test_box_fraction_of_an_inverted_box_is_zero() -> None:
    assert box_fraction([50, 40, 10, 5], frame()) == 0.0


def test_box_fraction_returns_none_for_a_missing_box() -> None:
    assert box_fraction(None, frame()) is None


# ----- iou ----------------------------------------------------------------------------


def test_iou_of_identical_boxes_is_one() -> None:
    assert iou([10, 10, 20, 20], [10, 10, 20, 20]) == 1.0


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0


def test_iou_half_overlap() -> None:
    # inter = 10x20 = 200, union = 400 + 400 - 200 = 600.
    assert iou([0, 0, 20, 20], [10, 0, 30, 20]) == pytest.approx(200 / 600, abs=1e-4)


def test_iou_of_nested_boxes() -> None:
    assert iou([0, 0, 20, 20], [5, 5, 15, 15]) == pytest.approx(100 / 400, abs=1e-4)


def test_iou_is_symmetric() -> None:
    a, b = [0, 0, 20, 30], [7, 11, 40, 33]
    assert iou(a, b) == iou(b, a)


@pytest.mark.parametrize("a,b", [(None, [0, 0, 1, 1]), ([0, 0, 1, 1], None), (None, None),
                                 ([1, 2, 3], [0, 0, 1, 1])])
def test_iou_returns_none_when_a_box_is_missing(a, b) -> None:
    assert iou(a, b) is None


def test_iou_of_two_degenerate_boxes_is_none() -> None:
    """Zero union is undefined, not zero -- reporting 0.0 would read as "disagree"."""
    assert iou([5, 5, 5, 5], [9, 9, 9, 9]) is None
