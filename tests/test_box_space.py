"""Tests for the coordinate-space dispatch in :mod:`phantom_data.build.segment`.

This is the most dangerous field in stage C's input contract, and the tests are shaped around
*why*: both spaces hold plausible numbers of the same magnitude, and applying the wrong branch
raises nothing. It just puts every box somewhere else.

* ``annotation`` -- raw Phantom coordinates, projected through the annotation canvas. What every
  existing ``extracted.jsonl`` holds, which is why an absent tag must mean this one.
* ``frame`` -- already real frame pixels, from Grounding DINO via ``tools/gate_apply.py``.
  Mapping these again rescales by ``max(W, H) / 768`` -- 2.5x on a 1920x1080 clip.

So the assertions to keep are: the frame branch clamps and does **not** rescale, the annotation
branch is bit-identical to what stage C always did, and an unrecognised value is a loud error
rather than a guess in either direction.
"""
from __future__ import annotations

import numpy as np
import pytest

from phantom_data.build import segment
from phantom_data.build.segment import BOX_SPACE_ANNOTATION, BOX_SPACE_FRAME

#: A 16:9 HD frame, where ``H_768_long`` gives ``scale = 1920 / 768 = 2.5`` -- a factor big
#: enough that a mistakenly re-mapped box lands nowhere near the right place.
HD = (1920, 1080)


# ----- the frame branch: clamp only ----------------------------------------------------


def test_a_frame_box_inside_the_frame_is_returned_unchanged() -> None:
    """No scaling, no rounding, no canvas. The detector already looked at these pixels."""
    assert segment.resolve_box([100, 50, 300, 200], *HD, BOX_SPACE_FRAME) == [
        100.0, 50.0, 300.0, 200.0]


def test_a_frame_box_is_not_rescaled_by_the_canvas_factor() -> None:
    """The corruption this tag exists to prevent, asserted as a difference.

    2.5x on this frame: the annotation branch turns [100, 50, 300, 200] into
    [250, 125, 750, 500]. If the dispatch ever fell through to that branch for frame boxes,
    every box in a gated dataset would be wrong and nothing would fail.
    """
    frame_box = segment.resolve_box([100, 50, 300, 200], *HD, BOX_SPACE_FRAME)
    annotation_box = segment.resolve_box([100, 50, 300, 200], *HD, BOX_SPACE_ANNOTATION)
    assert frame_box != annotation_box
    assert annotation_box == [250.0, 125.0, 750.0, 500.0]


def test_a_frame_box_is_still_clamped_to_the_frame() -> None:
    """Clamping stays unconditional: detector coordinates can sit a hair outside the frame."""
    assert segment.resolve_box([-10, -5, 5000, 9000], *HD, BOX_SPACE_FRAME) == [
        0.0, 0.0, 1920.0, 1080.0]


def test_a_frame_box_with_swapped_corners_is_ordered() -> None:
    assert segment.resolve_box([300, 200, 100, 50], *HD, BOX_SPACE_FRAME) == [
        100.0, 50.0, 300.0, 200.0]


def test_a_frame_box_exactly_on_the_edge_survives() -> None:
    """The clamp is inclusive of the frame bounds, so a full-frame box is not shrunk."""
    assert segment.resolve_box([0, 0, 1920, 1080], *HD, BOX_SPACE_FRAME) == [
        0.0, 0.0, 1920.0, 1080.0]


# ----- the annotation branch: unchanged behaviour --------------------------------------


@pytest.mark.parametrize("box", [[0, 0, 768, 432], [226, 14, 797, 394], [-5, 0, 900, 500],
                                 [171.5, 0.0, 649.25, 432.0]])
@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (720, 720), (480, 640)])
def test_the_annotation_branch_matches_scale_bbox_to_frame(box, size) -> None:
    """Bit-identical to the function stage C has always called.

    ``scale_bbox_to_frame`` was refactored to share the clamp with the frame branch, so this is
    the assertion that the refactor changed no output -- the built pilot dataset depends on it.
    """
    assert (segment.resolve_box(box, *size, BOX_SPACE_ANNOTATION)
            == segment.scale_bbox_to_frame(box, *size))


# ----- the dispatch is total -----------------------------------------------------------


@pytest.mark.parametrize("value", ["768", "Frame", "annotations", "", "pixel", None, 768])
def test_an_unknown_box_space_raises(value) -> None:
    """Never a silent default. Both wrong guesses are undetectable downstream."""
    with pytest.raises(ValueError, match="unknown box_space"):
        segment.resolve_box([1, 2, 3, 4], *HD, value)


def test_the_error_names_the_accepted_values() -> None:
    with pytest.raises(ValueError) as error:
        segment.resolve_box([1, 2, 3, 4], *HD, "768")
    assert BOX_SPACE_ANNOTATION in str(error.value)
    assert BOX_SPACE_FRAME in str(error.value)


# ----- parse_sample carries the tag ---------------------------------------------------


def _row(**overrides):
    """A minimal stage B row: exactly the fields ``parse_sample`` requires."""
    base = {
        "sample_id": "vid_w000000001", "video_id": "vid", "video": "clips/vid.mp4",
        "frame_count": 81,
        "subjects": [{
            "subject_id": 1, "phrase": "a woman in a red coat", "seed_frame_index": 10,
            "seed_bbox_768": [226, 14, 797, 394],
            "ref": {"frame": "ref_frames/vid_subj01.jpg", "bbox_768": [171, 0, 649, 432]},
        }],
    }
    return {**base, **overrides}


def test_an_absent_box_space_means_annotation() -> None:
    """The compatibility guarantee: every existing extracted.jsonl keeps its behaviour."""
    assert segment.parse_sample(_row()).subjects[0].box_space == BOX_SPACE_ANNOTATION


def test_box_space_frame_reaches_the_subject_spec() -> None:
    spec = segment.parse_sample(_row(box_space=BOX_SPACE_FRAME))
    assert spec.subjects[0].box_space == BOX_SPACE_FRAME


def test_parse_sample_rejects_an_unknown_box_space() -> None:
    """Rejected at parse time so a bad manifest stops the run on its first row."""
    with pytest.raises(ValueError, match="unknown box_space"):
        segment.parse_sample(_row(box_space="bogus"))


def test_every_subject_of_a_row_shares_the_row_level_space() -> None:
    """Row-level, not per-subject: one manifest has one producer."""
    row = _row(box_space=BOX_SPACE_FRAME)
    row["subjects"] = [dict(row["subjects"][0], subject_id=sid) for sid in (1, 2, 3)]
    spec = segment.parse_sample(row)
    assert [s.box_space for s in spec.subjects] == [BOX_SPACE_FRAME] * 3


# ---------------------------------------------------------------------------------------
# mask_stats: presence versus health
# ---------------------------------------------------------------------------------------
#
# The UltraVid fields all measure presence. Presence is not health: on the pilot, one masklet
# reported 81/81 frames present while its mask collapsed to 3% of its own median area on some of
# them, and a tight box read off that frame is garbage. These tests cover the stability numbers
# that make such a subject findable.


def masklet(*areas, shape=(20, 20)):
    """A masklet whose frame k has ``areas[k]`` set pixels, laid out row-major."""
    frames = []
    for area in areas:
        flat = np.zeros(shape[0] * shape[1], dtype=bool)
        flat[:area] = True
        frames.append(flat.reshape(shape))
    return np.stack(frames)


def test_a_steady_masklet_reports_flat_area_ratios() -> None:
    stats = segment.mask_stats(masklet(100, 100, 100))
    assert stats["area_min_ratio"] == 1.0
    assert stats["area_max_ratio"] == 1.0
    assert stats["interior_gap_frames"] == []


def test_a_partly_dissolved_mask_shows_a_low_min_ratio() -> None:
    # The real failure: present on every frame, so every presence check passes it.
    stats = segment.mask_stats(masklet(100, 100, 3, 100))
    assert stats["visible_frame_count"] == 4
    assert stats["full_clip_covered"] is True
    assert stats["area_min_ratio"] == 0.03


def test_a_leaked_mask_shows_a_high_max_ratio() -> None:
    stats = segment.mask_stats(masklet(100, 100, 400))
    assert stats["area_max_ratio"] == 4.0


def test_ratios_are_relative_to_the_median_not_the_max() -> None:
    # One leaked frame must not rescale the whole series and hide a dissolve elsewhere.
    stats = segment.mask_stats(masklet(100, 100, 100, 400))
    assert stats["area_median"] == 100
    assert stats["area_max_ratio"] == 4.0


def test_an_interior_gap_is_reported_separately_from_a_subject_leaving() -> None:
    # A hole in the middle means the track dropped and was reacquired; what came back may not be
    # the same object. A subject that simply exits at the end is not the same claim.
    assert segment.mask_stats(masklet(100, 0, 100))["interior_gap_frames"] == [1]
    assert segment.mask_stats(masklet(100, 100, 0))["interior_gap_frames"] == []
    assert segment.mask_stats(masklet(0, 100, 100))["interior_gap_frames"] == []


def test_an_empty_masklet_reports_zeros_without_dividing_by_zero() -> None:
    stats = segment.mask_stats(masklet(0, 0))
    assert stats["visible_frame_count"] == 0
    assert stats["area_min_ratio"] == 0.0
    assert stats["area_max_ratio"] == 0.0
    assert stats["area_median"] == 0
    assert stats["interior_gap_frames"] == []


def test_the_ultravid_presence_fields_are_unchanged() -> None:
    # The stability fields are additive: stage D reads the presence vocabulary and must not see a
    # changed value for any of it.
    stats = segment.mask_stats(masklet(0, 50, 100, 0))
    assert stats["visible_frame_count"] == 2
    assert stats["first_mask_frame"] == 1
    assert stats["last_mask_frame"] == 2
    assert stats["full_clip_covered"] is False
    assert stats["max_mask_area"] == 100
