"""Pure tests for the crop policies in ``tools/compare_dino_crop.py``.

The measurement itself needs a GPU; the geometry does not, and the geometry is what the
conclusion rests on. Two properties are load-bearing:

* :func:`expand_to_aspect` **only ever grows**. If it could shrink the long side it would be
  the centre crop it exists to replace, and the comparison would measure nothing.
* padding is a **last resort**. The policy's claim is "take real surrounding pixels"; a
  version that pads a box sitting against the frame edge instead of sliding inwards would
  quietly become the letterbox variant and destroy the contrast between the two.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_dino_crop import (  # noqa: E402
    PAD_RGB, elongation, expand_to_aspect, letterbox, pad_image, pearson, spearman,
    view_expand, view_letterbox,
)


def frame(width: int = 200, height: int = 100) -> np.ndarray:
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


# ----- expand_to_aspect ---------------------------------------------------------------


@pytest.mark.parametrize("box", [None, [1, 2, 3], "0,0,1,1", 4])
def test_expand_rejects_non_boxes(box) -> None:
    assert expand_to_aspect(box, 100, 100) is None


def test_expand_widens_a_tall_box_with_real_pixels() -> None:
    # 20 wide, 60 tall, in the middle of a wide frame: the deficit is all available.
    (x1, y1, x2, y2), pads = expand_to_aspect([90, 20, 110, 80], 200, 100)
    assert (x2 - x1, y2 - y1) == (60, 60)
    assert pads == (0, 0, 0, 0)
    assert (y1, y2) == (20, 80)  # the long axis is untouched: nothing is cut


def test_expand_heightens_a_wide_box() -> None:
    (x1, y1, x2, y2), pads = expand_to_aspect([40, 40, 120, 60], 200, 100)
    assert (x2 - x1, y2 - y1) == (80, 80)
    assert pads == (0, 0, 0, 0)
    assert (x1, x2) == (40, 120)


def test_expand_never_shrinks_the_long_side() -> None:
    box = [0, 0, 10, 90]
    (x1, y1, x2, y2), _pads = expand_to_aspect(box, 200, 100)
    assert y2 - y1 >= 90 and x2 - x1 >= 10


def test_expand_slides_inwards_at_a_frame_edge_instead_of_padding() -> None:
    # Box hugs the left edge; the missing width must come from the right, not from padding.
    (x1, y1, x2, y2), pads = expand_to_aspect([0, 20, 20, 80], 200, 100)
    assert (x1, x2) == (0, 60)
    assert pads == (0, 0, 0, 0)


def test_expand_pads_only_when_the_frame_itself_is_too_small() -> None:
    # A frame narrower than the box is tall: 40 px of width can never cover a 90 px height.
    (x1, y1, x2, y2), pads = expand_to_aspect([0, 5, 40, 95], 40, 100)
    assert (x1, x2) == (0, 40)
    assert y2 - y1 == 90
    assert sum(pads) == 50 and pads[1] == pads[3] == 0  # padded on x only


def test_expand_result_plus_padding_is_square() -> None:
    for box, w, h in ([[10, 10, 30, 90], 200, 100], [[0, 0, 199, 5], 200, 100],
                      [[150, 60, 199, 99], 200, 100], [[0, 0, 3, 99], 40, 100]):
        (x1, y1, x2, y2), (pl, pt, pr, pb) = expand_to_aspect(box, w, h)
        assert (x2 - x1) + pl + pr == (y2 - y1) + pt + pb


def test_expand_honours_a_non_square_aspect() -> None:
    (x1, y1, x2, y2), pads = expand_to_aspect([80, 20, 100, 80], 200, 100, aspect=2.0)
    assert pads == (0, 0, 0, 0)
    assert (x2 - x1) == 2 * (y2 - y1)


def test_expand_stays_inside_the_frame() -> None:
    for box in ([-50, -50, 10, 10], [190, 90, 400, 400], [0, 0, 200, 100]):
        (x1, y1, x2, y2), _ = expand_to_aspect(box, 200, 100)
        assert 0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 100


# ----- letterbox ----------------------------------------------------------------------


def test_letterbox_pads_the_short_axis_only() -> None:
    assert letterbox(60, 20) == (20, 0, 20, 0)
    assert letterbox(20, 60) == (0, 20, 0, 20)


def test_letterbox_is_a_noop_on_a_square() -> None:
    assert letterbox(32, 32) == (0, 0, 0, 0)


def test_letterbox_puts_the_odd_pixel_on_the_far_side() -> None:
    assert letterbox(11, 4) == (3, 0, 4, 0)


# ----- pad_image ----------------------------------------------------------------------


def test_pad_image_keeps_the_original_pixels_and_fills_the_rest() -> None:
    image = frame(4, 6)
    out = pad_image(image, (2, 1, 3, 0))
    assert out.shape == (7, 9, 3)
    assert np.array_equal(out[1:7, 2:6], image)
    assert tuple(out[0, 0]) == PAD_RGB


def test_pad_image_returns_the_input_when_nothing_is_padded() -> None:
    image = frame(4, 4)
    assert pad_image(image, (0, 0, 0, 0)) is image


# ----- views --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [224, 336, 448])
def test_views_are_square_at_the_requested_size(size: int) -> None:
    assert view_expand(frame(), [90, 20, 110, 80], size).shape == (size, size, 3)
    assert view_letterbox(frame(), [90, 20, 110, 80], size).shape == (size, size, 3)


def test_views_reject_a_missing_box() -> None:
    assert view_expand(frame(), None) is None
    assert view_letterbox(frame(), None) is None


def test_letterbox_view_has_pad_bands_where_expand_has_pixels() -> None:
    # Same tall box: letterbox must show the pad colour at the left edge, expand must not.
    box = [90, 20, 110, 80]
    lb, ex = view_letterbox(frame(), box, 224), view_expand(frame(), box, 224)
    assert tuple(lb[112, 2]) == PAD_RGB
    assert tuple(ex[112, 2]) != PAD_RGB


# ----- elongation and correlations ----------------------------------------------------


def test_elongation_is_symmetric_and_at_least_one() -> None:
    assert elongation([0, 0, 10, 30]) == 3.0
    assert elongation([0, 0, 30, 10]) == 3.0
    assert elongation([0, 0, 10, 10]) == 1.0


@pytest.mark.parametrize("box", [None, [0, 0, 0, 10], [0, 0, 10, 0], [1, 2, 3]])
def test_elongation_none_for_degenerate(box) -> None:
    assert elongation(box) is None


def test_pearson_and_spearman_on_a_monotone_nonlinear_pair() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [1.0, 4.0, 9.0, 16.0, 25.0]
    assert spearman(xs, ys) == 1.0
    assert 0.9 < pearson(xs, ys) < 1.0


def test_correlations_are_none_when_undefined() -> None:
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert spearman([1.0, 2.0], [1.0, 2.0]) is None
