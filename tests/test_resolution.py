"""Tests for stage B's storage geometry (:mod:`phantom_data.build.resolution`).

Pure arithmetic, no network, no encode backend. The shape of these tests follows the shape of
the risk, which is **not** "is the maths right" -- it is "do the stored pixels and the stored
metadata agree". Stage B writes ``width``/``height`` into ``extracted.jsonl``, and every
downstream box lands via those two numbers (``segment.scale_bbox_to_frame`` derives its factor
from them). Disagreement misplaces every box in the dataset and raises nothing.

So the assertions worth keeping are:

* the three measured pilot resolutions land where the design says (1920x1080 and 1280x720 both
  to 854x480; 1440x1080 to 640x480, height intact);
* 16:9 storage gives training ``scale == 1.0`` -- the whole reason for 854 over 832;
* portrait sources are *flagged*, loudly and countably, not silently mangled;
* stored dimensions are always even, because ``macro_block_size=2`` would otherwise round the
  encode up and desynchronise pixels from metadata;
* scaling disabled returns the source verbatim, which is what reproduces the pilot data.
"""
from __future__ import annotations

import pytest

from phantom_data.build import resolution
from phantom_data.build.resolution import (
    CROP_DISCARD_WARN,
    DEFAULT_TARGET_HEIGHT,
    DEFAULT_TARGET_WIDTH,
    scale_box,
    snap,
    storage_plan,
    stored_dims,
    training_crop,
)

#: The measured pilot distribution: 102 samples at 1920x1080, 35 at 1280x720, 1 at 1440x1080.
PILOT = [(1920, 1080), (1280, 720), (1440, 1080)]
#: Training's frame, confirmed from the running jobs' argv rather than a script default.
TARGET = (DEFAULT_TARGET_WIDTH, DEFAULT_TARGET_HEIGHT)


# ----- stored dimensions ---------------------------------------------------------------


@pytest.mark.parametrize("source", [(1920, 1080), (1280, 720)])
def test_sixteen_by_nine_sources_all_store_at_854x480(source) -> None:
    """Both dominant pilot resolutions collapse to one stored size, since both are 16:9.

    854 rather than 853: 1920*480/1080 = 853.33 and the stored size must be even.
    """
    assert stored_dims(*source, DEFAULT_TARGET_HEIGHT) == (854, 480)


def test_the_four_by_three_source_keeps_its_full_height() -> None:
    """1440x1080 -> 640x480. The height is anchored, so none of it is lost at storage time.

    This is the case that rules out storing 832x480: at that size the 4:3 frame would have had
    144px (23%) of its height cropped away permanently, potentially cutting the subject out.
    """
    assert stored_dims(1440, 1080, DEFAULT_TARGET_HEIGHT) == (640, 480)


def test_the_aspect_ratio_survives_the_downscale() -> None:
    """Isotropy, asserted as a ratio rather than as a pair of numbers.

    Tolerance is one part in 400, which is what snapping to an even width costs at this size
    (854/480 = 1.7792 against 16/9 = 1.7778).
    """
    for source_width, source_height in PILOT:
        width, height = stored_dims(source_width, source_height, DEFAULT_TARGET_HEIGHT)
        assert width / height == pytest.approx(source_width / source_height, rel=0.0025)


@pytest.mark.parametrize("source", PILOT + [(1080, 1920), (720, 1280), (1440, 1079)])
def test_stored_dimensions_are_always_even(source) -> None:
    """``macro_block_size=2`` rounds odd dimensions up inside the ffmpeg writer.

    An odd width in the manifest would therefore describe a file that is one pixel wider --
    the exact pixels-vs-metadata disagreement this module exists to prevent. 1440x1079 is in
    the list as a deliberately odd source, to prove the snap is on the *output*.
    """
    width, height = stored_dims(*source, DEFAULT_TARGET_HEIGHT)
    assert width % 2 == 0 and height % 2 == 0


def test_a_source_already_at_the_target_height_is_passed_through() -> None:
    assert stored_dims(854, 480, DEFAULT_TARGET_HEIGHT) == (854, 480)


def test_a_source_smaller_than_the_target_is_never_upscaled() -> None:
    """Upscaling would spend bytes on invented pixels and then hand training a doubly
    resampled frame -- it upscales by ``max(832/w, 480/h)`` itself, from the sharper original.
    """
    assert stored_dims(640, 360, DEFAULT_TARGET_HEIGHT) == (640, 360)


def test_bad_source_dimensions_raise_rather_than_produce_a_zero_sized_clip() -> None:
    with pytest.raises(ValueError, match="bad source dimensions"):
        stored_dims(0, 1080, DEFAULT_TARGET_HEIGHT)


# ----- scaling disabled: the pilot-reproduction path ------------------------------------


@pytest.mark.parametrize("source", PILOT + [(1080, 1920), (1441, 1079)])
def test_target_height_zero_returns_the_source_verbatim(source) -> None:
    """The reproduction path for the existing pilot data.

    Verbatim, not snapped: 1441x1079 comes back odd. Snapping here would resize an
    odd-dimensioned source, which is a behaviour change however harmless it looks, and this
    path's entire job is to be byte-identical to what already shipped.
    """
    assert stored_dims(*source, 0) == source


def test_scaling_disabled_reports_itself_as_unscaled_at_scale_one() -> None:
    plan = storage_plan(1920, 1080, target_height=0)
    assert plan["scaled"] is False
    assert plan["scale"] == 1.0
    assert (plan["width"], plan["height"]) == (1920, 1080)


def test_a_negative_target_height_also_disables_scaling() -> None:
    """``<= 0``, not ``== 0``: an off-by-one in a sweep script must not silently mangle sizes."""
    assert stored_dims(1920, 1080, -1) == (1920, 1080)


# ----- what training then does ---------------------------------------------------------


@pytest.mark.parametrize("source", [(1920, 1080), (1280, 720)])
def test_storing_854x480_makes_training_resample_not_at_all(source) -> None:
    """``crop_scale == 1.0`` on 16:9 -- the point of the whole design.

    Training computes ``max(832/854, 480/480) = 1.0``, so it resizes by nothing and only
    performs its centre crop. The clip is therefore resampled exactly once, here.
    """
    width, height = stored_dims(*source, DEFAULT_TARGET_HEIGHT)
    assert training_crop(width, height, *TARGET)["crop_scale"] == 1.0


def test_the_crop_on_a_16x9_clip_is_the_expected_10px_a_side() -> None:
    """854 -> 832 is 22px total, 2.6% of the width. Small, and reversible: the pixels either
    side of the crop are still on disk, which storing 832x480 would not have allowed.
    """
    crop = training_crop(854, 480, *TARGET)
    assert crop["resized_width"] - DEFAULT_TARGET_WIDTH == 22
    assert crop["discard_width"] == pytest.approx(22 / 854, abs=1e-6)
    assert crop["discard_height"] == 0.0


def test_the_4x3_clip_loses_height_and_is_flagged() -> None:
    """640x480 upscales by 832/640 = 1.3 and then loses 23% of its height.

    Note this is a property of *source aspect vs target aspect*, not of the storage decision:
    the same 23% is lost whether the clip is stored at 1440x1080 or at 640x480. The flag
    surfaces a pre-existing property of the data rather than creating one.
    """
    plan = storage_plan(1440, 1080)
    assert plan["train_crop_scale"] == pytest.approx(1.3)
    assert plan["train_discard_height"] == pytest.approx(0.2308, abs=1e-4)
    assert plan["crop_discard_excessive"] is True


def test_a_portrait_source_is_flagged_and_not_silently_mangled() -> None:
    """1080x1920 height-anchors to 270x480; training then upscales 3.08x and throws away 68%
    of the height. The pilot is 100% landscape so this never surfaced, but 126k samples will
    contain portrait video, and the contract is: flag it, count it, do not drop it here.
    """
    plan = storage_plan(1080, 1920)
    assert (plan["width"], plan["height"]) == (270, 480)
    assert plan["train_discard_height"] > 0.6
    assert plan["crop_discard_excessive"] is True


def test_the_landscape_pilot_resolutions_are_not_flagged() -> None:
    """The threshold has to be quiet on the intended case, or the funnel category is noise."""
    for source in [(1920, 1080), (1280, 720)]:
        assert storage_plan(*source)["crop_discard_excessive"] is False


def test_the_threshold_sits_between_the_16x9_and_4x3_losses() -> None:
    """Stated as an ordering rather than as the constant, so the reason survives a re-tune."""
    assert storage_plan(1920, 1080)["train_discard_width"] < CROP_DISCARD_WARN
    assert storage_plan(1440, 1080)["train_discard_height"] > CROP_DISCARD_WARN


def test_discard_is_never_negative_when_rounding_leaves_a_side_short() -> None:
    """A resized side can land a pixel under the target, where torchvision's ``center_crop``
    pads instead of cropping. A negative "discarded fraction" would be a nonsense number in
    the manifest and could not trip the threshold in the direction it means.
    """
    for stored in [(832, 480), (833, 480), (1, 1), (832, 481)]:
        crop = training_crop(*stored, *TARGET)
        assert crop["discard_width"] >= 0.0 and crop["discard_height"] >= 0.0


# ----- snapping ------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (853.33, 854), (853.0, 854), (852.0, 852), (426.5, 426), (427.5, 428), (0.4, 2), (1.0, 2),
])
def test_snap_rounds_to_even_and_never_to_zero(value, expected) -> None:
    """Halves go upward and consistently -- :func:`round` is banker's rounding, so it would
    turn 426.5 into 426 and 427.5 into 428, a dimension that depends on the parity of the
    quotient. And a zero-width clip is not encodable, so the floor is one step, not zero.
    """
    assert snap(value) == expected


def test_snap_with_multiple_one_is_plain_rounding_with_a_floor_of_one() -> None:
    assert snap(853.33, 1) == 853
    assert snap(0.1, 1) == 1


# ----- box scaling ---------------------------------------------------------------------


def test_a_frame_space_box_at_the_edge_stays_at_the_edge() -> None:
    """The property that matters more than any particular coordinate: a box touching the frame
    border must still touch it after the rescale, or subjects at the edge get clipped.
    """
    scaled = scale_box([0, 0, 1920, 1080], 1920, 1080, 854, 480)
    assert scaled == [0.0, 0.0, 854.0, 480.0]


def test_box_scaling_preserves_the_boxs_aspect_ratio_when_the_frames_do() -> None:
    """Isotropy of the frame rescale, observed through a box rather than through the dims."""
    x1, y1, x2, y2 = scale_box([200, 100, 800, 500], 1920, 1080, 854, 480)
    source_ratio = (800 - 200) / (500 - 100)
    assert (x2 - x1) / (y2 - y1) == pytest.approx(source_ratio, rel=0.003)


def test_box_scaling_matches_the_frame_scale_on_each_axis() -> None:
    """Coordinates move by exactly the stored/source ratio -- the same factor the pixels moved
    by. Any other factor is the silent misalignment case.
    """
    width, height = stored_dims(1920, 1080, DEFAULT_TARGET_HEIGHT)
    scaled = scale_box([100, 200, 300, 400], 1920, 1080, width, height)
    assert scaled == [100 * width / 1920, 200 * height / 1080,
                      300 * width / 1920, 400 * height / 1080]


def test_box_scaling_is_the_identity_when_the_dimensions_are_unchanged() -> None:
    """So the scaling-disabled path leaves frame-space boxes untouched too, not merely close."""
    assert scale_box([12.5, 7.25, 99.0, 33.75], 1920, 1080, 1920, 1080) == [
        12.5, 7.25, 99.0, 33.75]


# ----- the annotation-coordinate decision ----------------------------------------------


def test_stage_b_leaves_raw_annotation_boxes_alone_and_the_stored_dims_carry_the_change() -> None:
    """The load-bearing consequence of *not* pre-scaling ``seed_bbox_768``.

    Those are raw annotation coordinates in an unresolved canvas, projected onto a frame by
    ``segment.scale_bbox_to_frame``, which derives its factor from the frame dimensions it is
    handed. Storing the new dims is therefore sufficient *and* pre-multiplying the coordinates
    would be a double application: this test pins the 2.25x gap between the two, which is the
    size of the corruption that would be invisible in every log.
    """
    from phantom_data.build import segment

    # A real pilot subject's seed box, narrowed to [226, 14, 600, 394] so it stays inside the
    # frame after projection. The original x2 of 797 projects to 886 on an 854-wide frame and is
    # clamped there, which would make this test measure the clamp instead of the double-scaling.
    annotation_box = [226.0, 14.0, 600.0, 394.0]
    stored_width, stored_height = stored_dims(1920, 1080, DEFAULT_TARGET_HEIGHT)

    correct = segment.scale_bbox_to_frame(annotation_box, stored_width, stored_height)
    doubled = segment.scale_bbox_to_frame(
        scale_box(annotation_box, 1920, 1080, stored_width, stored_height),
        stored_width, stored_height)

    assert correct[2] / doubled[2] == pytest.approx(1920 / stored_width, rel=0.003)
    # And the correct projection is the source-resolution projection, shrunk by the frame's own
    # ratio: the box tracks the pixels because both are driven by the same dims.
    at_source = segment.scale_bbox_to_frame(annotation_box, 1920, 1080)
    assert correct[2] == pytest.approx(at_source[2] * stored_width / 1920, rel=0.003)


@pytest.mark.parametrize("hypothesis_id", resolution.COMMUTING_HYPOTHESES)
def test_the_canvas_map_commutes_with_the_downscale_for_the_hypotheses_in_use(
        hypothesis_id) -> None:
    """**The property that lets stage B store scaled frames and change nothing downstream.**

    ``map(box, W', H') == map(box, W, H) * (W'/W)``. Because it holds for stage C's default
    (``H_768_long``), projecting an untouched annotation box against the *stored* dimensions
    gives exactly the source-resolution answer shrunk by the frame's own ratio -- so
    ``segment.scale_bbox_to_frame`` and ``redetect_run`` stay correct with no edit at all.

    Tolerance, not equality: snapping the width to an even number makes the achieved scale
    marginally anisotropic (kx 0.444792 vs ky 0.444444 on 1920x1080), so commutation is exact in
    principle and sub-pixel in practice. The displacement is bounded separately below.
    """
    from phantom_data import canvas

    hypothesis = canvas.get(hypothesis_id)
    box = [226.0, 14.0, 600.0, 394.0]
    stored_width, stored_height = stored_dims(1920, 1080, DEFAULT_TARGET_HEIGHT)

    direct = canvas.map_box(box, stored_width, stored_height, hypothesis)
    via_source = scale_box(canvas.map_box(box, 1920, 1080, hypothesis),
                           1920, 1080, stored_width, stored_height)
    assert direct == pytest.approx(via_source, rel=resolution.COMMUTATION_TOLERANCE)


@pytest.mark.parametrize("source", PILOT + [(1080, 1920), (1000, 563), (1920, 800)])
def test_the_even_width_snap_costs_less_than_one_pixel_at_the_frame_edge(source) -> None:
    """The honest bound on the only inexactness the design introduces.

    The snap means kx and ky differ slightly, so a box coordinate can land a fraction of a pixel
    from where pure isotropy would put it. Asserted as *displacement at the frame edge* rather
    than as a ratio, because that is the quantity that would actually misalign a box -- and it
    stays under a pixel, an order of magnitude below the annotation noise these boxes carry (the
    canvas convention itself is unresolved to within tens of pixels on x).
    """
    source_width, source_height = source
    width, height = stored_dims(source_width, source_height, DEFAULT_TARGET_HEIGHT)
    kx, ky = width / source_width, height / source_height
    assert abs(kx - ky) * source_width < 1.0


def test_the_qwen_hypothesis_does_not_commute_and_the_gap_is_recorded() -> None:
    """The landmine, pinned as a number so it cannot be rediscovered the expensive way.

    ``H_qwen_smart``'s canvas is a function of the frame's *size* (each side rounded to a
    multiple of 28 under an absolute area budget), not of its shape: 1920x1080 annotates in a
    1316x728 canvas but 854x480 in an 840x476 one. So mapping against the stored dimensions
    inflates every box by ~1.57x. If the canvas question ever resolves in favour of a
    resolution-dependent hypothesis, the map must run against ``storage_geometry.source_*`` and
    then be brought into stored pixels with :func:`scale_box` -- which is exactly why those two
    fields are recorded.
    """
    from phantom_data import canvas

    hypothesis = canvas.get("H_qwen_smart")
    assert "H_qwen_smart" not in resolution.COMMUTING_HYPOTHESES
    box = [226.0, 14.0, 600.0, 394.0]
    stored_width, stored_height = stored_dims(1920, 1080, DEFAULT_TARGET_HEIGHT)

    naive = canvas.map_box(box, stored_width, stored_height, hypothesis)
    correct = scale_box(canvas.map_box(box, 1920, 1080, hypothesis),
                        1920, 1080, stored_width, stored_height)
    assert naive[2] / correct[2] == pytest.approx(1.567, abs=0.01)


def test_the_source_dimensions_needed_to_fix_that_are_in_every_row() -> None:
    """The recovery path above is only available if the source size survives in the manifest."""
    plan = storage_plan(1920, 1080)
    assert (plan["source_width"], plan["source_height"]) == (1920, 1080)


def test_the_geometry_record_names_both_resolutions_so_a_clip_stays_interpretable() -> None:
    """A stored clip is only interpretable together with the source it came from: the funnel
    needs to know a 854x480 clip was 1920x1080, and a re-run needs to know what target it was
    produced under.
    """
    plan = storage_plan(1920, 1080)
    assert plan["source_width"] == 1920 and plan["source_height"] == 1080
    assert plan["width"] == 854 and plan["height"] == 480
    assert plan["target_height"] == 480 and plan["target_width"] == 832
    assert plan["scaled"] is True
    assert plan["scale"] == pytest.approx(480 / 1080, abs=1e-6)


def test_the_plan_is_json_serialisable_because_it_goes_into_the_manifest() -> None:
    """Numpy scalars leaking out of the decode would make ``json.dumps`` fail *after* the clip
    was already written and the marker put -- a run that burns decode time and stores nothing.
    """
    import json

    assert json.loads(json.dumps(storage_plan(1920, 1080))) == storage_plan(1920, 1080)


def test_the_reported_scale_reflects_what_happened_not_what_was_asked() -> None:
    """A 640x360 source is passed through, so its scale is 1.0 even though 480 was requested.

    Reporting the requested ratio there would claim a 1.33x downscale that never happened.
    """
    assert storage_plan(640, 360)["scale"] == 1.0
    assert storage_plan(640, 360)["scaled"] is False


# ----- the projected saving, as arithmetic ---------------------------------------------


def test_the_pixel_reduction_on_the_dominant_source_is_about_five_fold() -> None:
    """Not a byte claim -- H.264 bitrate is not proportional to area -- but the pixel ratio is
    the ceiling on what the measured saving can be, and it is worth pinning at 5.06x so a
    regression in the anchor rule shows up here rather than in a BOS bill.
    """
    width, height = stored_dims(1920, 1080, DEFAULT_TARGET_HEIGHT)
    assert (1920 * 1080) / (width * height) == pytest.approx(5.06, abs=0.02)


def test_module_defaults_match_the_running_training_jobs() -> None:
    """480/832 are read off the two running jobs' actual argv. If these ever drift from the
    trainer, every clip in the dataset is stored for a resolution nobody trains at.
    """
    assert (resolution.DEFAULT_TARGET_HEIGHT, resolution.DEFAULT_TARGET_WIDTH) == (480, 832)
