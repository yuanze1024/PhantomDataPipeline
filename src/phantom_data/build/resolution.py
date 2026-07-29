"""Stage B storage geometry: how large a clip is stored, and what training then does to it.

**Why this exists.** Stage B originally stored clips at *source* resolution -- measured over the
138-sample pilot: 102 at 1920x1080, 35 at 1280x720, 1 at 1440x1080, averaging 8.5 MiB per clip.
Training only ever consumes 832x480 (verified against the two running jobs' actual argv:
``--height 480 --width 832 --num_frames 81``), so at ~126k samples that is ~1.1 TiB on BOS with
99% of it in pixels nobody reads. Downscaling here is the single biggest cost lever in the
pipeline, which is why the arithmetic is factored out and unit tested rather than inlined.

**The rule: isotropic, height-anchored, uncropped.** ``scale = target_height / source_height``,
width follows the aspect ratio. So 1920x1080 -> 854x480 and 1280x720 -> 854x480 (both 16:9), and
1440x1080 -> 640x480 (4:3, kept whole).

Deliberately **not** 832x480. Training's own loader (``ImageCropAndResize`` in
``diffsynth/core/data/operators.py``, applied per frame inside ``LoadVideo``'s decode loop) does

    scale = max(target_w / w, target_h / h);  resize;  center_crop(target_h, target_w)

so on a 16:9 source it rescales by max(832/1920, 480/1080) = 0.4444 and then shaves ~10px off
each side. Storing 832x480 would bake that crop in permanently and -- far worse -- on the 4:3
source it would crop away 144px of *height* (23%), which can put the subject out of frame. At
854x480 training computes ``scale = 1.0``: zero resampling, it just performs its own crop, and
the crop stays reversible because the pixels either side of it are still on disk. The training
resolution is therefore never welded into the dataset.

**Height as the anchor is only optimal while sources are wider than 832/480 = 1.733.** A portrait
source (1080x1920) height-anchors to 270x480, and training then upscales by 832/270 = 3.08 and
throws away 67% of the height. The pilot is 100% landscape so this never surfaced; 126k samples
will contain portrait video. :func:`storage_plan` therefore *computes* what training will discard
and flags it, per sample, instead of silently mangling it -- see :data:`CROP_DISCARD_WARN`.

Note the discard fraction is a property of *source aspect vs target aspect*, not of the storage
decision: the 1440x1080 pilot sample loses the same 23% of height whether training reads it at
1440x1080 or at 640x480. This diagnostic surfaces a pre-existing property of the data; it does
not create one.

Pure module on purpose -- no numpy, no PIL, no I/O -- so the geometry is testable without an
encode backend (``cv2.VideoWriter`` has no working encoder in this image: ``isOpened()`` is False
for every fourcc, so all encoding goes through imageio/PIL).
"""
from __future__ import annotations

import math

#: Training's frame size, and so the default target here. Confirmed from the running jobs' argv,
#: not from a script default.
DEFAULT_TARGET_HEIGHT = 480
DEFAULT_TARGET_WIDTH = 832

#: Stored dimensions are snapped to a multiple of this. **This is a correctness constraint, not a
#: tidiness one.** ``encode_mp4`` passes ``macro_block_size=2`` to the imageio ffmpeg writer, which
#: silently rounds odd dimensions *up* to the next multiple of 2 via a scale filter. 16:9 sources
#: land on 1920*480/1080 = 853.33 -> 853, odd, so the writer would store 854 while the manifest
#: recorded 853. Stored pixels disagreeing with stored dimensions is the one failure this whole
#: module has to avoid: every box is then misplaced and nothing raises.
SIZE_MULTIPLE = 2

#: Flag a sample when training's centre crop would discard more than this share of either axis.
#: 0.20 sits above the 2.6% a 16:9 source loses on width and below the 23% a 4:3 source loses on
#: height, so it catches the aspect mismatches worth a human decision (portrait, extreme 4:3)
#: without firing on the intended landscape case.
CROP_DISCARD_WARN = 0.20


def snap(value: float, multiple: int = SIZE_MULTIPLE) -> int:
    """Round ``value`` to the nearest positive multiple of ``multiple``, halves upward.

    ``math.floor(x + 0.5)`` rather than :func:`round`: Python's round is banker's rounding, so
    ``round(426.5)`` is 426 and ``round(427.5)`` is 428. A dimension that changes with the parity
    of the quotient is not something to debug later from a byte-size histogram.
    """
    if multiple <= 1:
        return max(1, int(math.floor(value + 0.5)))
    steps = max(1, int(math.floor(value / multiple + 0.5)))
    return steps * multiple


def stored_dims(source_width: int, source_height: int, target_height: int,
                multiple: int = SIZE_MULTIPLE) -> tuple[int, int]:
    """``(width, height)`` a clip is stored at. Isotropic, anchored on height, never upscaled.

    ``target_height <= 0`` disables scaling and returns the source dimensions **verbatim** --
    not snapped. That path has to reproduce the existing pilot data byte-for-byte, and snapping
    would resize any odd-dimensioned source, which is a change however harmless it looks. (Every
    pilot resolution is even, so the latent odd-source hazard described at :data:`SIZE_MULTIPLE`
    is pre-existing on that path and untouched here.)

    Sources at or below the target height are passed through rather than upscaled. Upscaling would
    spend bytes inventing pixels and then hand training a doubly-resampled frame: training
    upscales by ``max(832/w, 480/h)`` on its own, from the sharper original.
    """
    source_width, source_height = int(source_width), int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"bad source dimensions {source_width}x{source_height}")
    if target_height <= 0 or source_height <= target_height:
        return source_width, source_height
    scale = target_height / source_height
    return snap(source_width * scale, multiple), snap(source_height * scale, multiple)


def training_crop(stored_width: int, stored_height: int,
                  target_width: int = DEFAULT_TARGET_WIDTH,
                  target_height: int = DEFAULT_TARGET_HEIGHT) -> dict[str, float]:
    """What ``ImageCropAndResize`` will do to a stored frame, and what it throws away.

    Mirrors ``diffsynth/core/data/operators.py`` line for line, including its use of plain
    :func:`round` on the resize -- this is a prediction of another module's behaviour, so matching
    its rounding matters more than matching :func:`snap`'s.

    ``discard_width`` / ``discard_height`` are the fraction of the *resized* frame the centre crop
    removes on each axis. Clamped at 0 because rounding can leave a resized side a pixel short of
    the target, in which case torchvision's ``center_crop`` pads instead of cropping.
    """
    scale = max(target_width / stored_width, target_height / stored_height)
    resized_width = round(stored_width * scale)
    resized_height = round(stored_height * scale)
    return {
        "crop_scale": round(float(scale), 6),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "discard_width": round(max(0.0, (resized_width - target_width) / resized_width), 6),
        "discard_height": round(max(0.0, (resized_height - target_height) / resized_height), 6),
    }


def storage_plan(source_width: int, source_height: int,
                 target_height: int = DEFAULT_TARGET_HEIGHT,
                 target_width: int = DEFAULT_TARGET_WIDTH,
                 discard_warn: float = CROP_DISCARD_WARN) -> dict[str, object]:
    """One sample's full storage geometry, as it is recorded in ``extracted.jsonl``.

    The record is the point: a stored clip is only interpretable together with the source it came
    from and the target it was scaled for, and the funnel needs the discard numbers per sample to
    decide about portrait sources. ``crop_discard_excessive`` is the counted category -- flagged,
    never dropped here.

    ``scale`` is reported as the achieved height ratio (``stored_height / source_height``), not the
    requested one, so it reflects the snapping and the never-upscale rule rather than intent.
    """
    width, height = stored_dims(source_width, source_height, target_height)
    crop = training_crop(width, height, target_width, target_height)
    return {
        "source_width": int(source_width),
        "source_height": int(source_height),
        "width": width,
        "height": height,
        "target_width": int(target_width),
        "target_height": int(target_height),
        "scaled": (width, height) != (int(source_width), int(source_height)),
        "scale": round(height / float(source_height), 6),
        "train_crop_scale": crop["crop_scale"],
        "train_discard_width": crop["discard_width"],
        "train_discard_height": crop["discard_height"],
        "crop_discard_excessive": bool(
            crop["discard_width"] > discard_warn or crop["discard_height"] > discard_warn
        ),
    }


#: Canvas hypotheses whose annotation->frame map **commutes** with an isotropic frame downscale,
#: i.e. ``map(box, W', H') == map(box, W, H) * (W'/W)``. Verified by construction: their scale
#: factor is a ratio of the frame dimensions, so shrinking the frame shrinks the mapped box by
#: exactly the same factor. Stage C's default (``H_768_long``) is one of these, which is the
#: reason stage B can store scaled frames while leaving raw annotation coordinates untouched, and
#: why neither ``segment.py`` nor ``redetect_run.py`` needed a change.
#:
#: **``H_qwen_smart`` does NOT commute, and this is a live landmine for whoever resolves the
#: canvas question.** Its canvas comes from Qwen2.5-VL's ``smart_resize``, which rounds each side
#: to a multiple of 28 under an *absolute* area budget -- so it is a function of the frame's size,
#: not of its shape. Measured: 1920x1080 -> canvas 1316x728 (sx 1.459), while 854x480 -> canvas
#: 840x476 (sx 1.017). Mapping an annotation box against the *stored* 854x480 therefore inflates
#: it by 1.567x versus the correct answer, because the canvas Qwen actually annotated in was
#: derived from the 1920x1080 source it was shown.
#:
#: The information needed to do this correctly is preserved: ``storage_geometry.source_width`` /
#: ``source_height`` are in every manifest row. A future switch to a resolution-dependent
#: hypothesis must map against those **source** dimensions and then apply
#: :func:`scale_box` to bring the result into stored pixels -- it must not simply hand the stored
#: dimensions to ``scale_bbox_to_frame``. ``H_norm1000`` and ``H_832x480_aniso`` are anisotropic
#: and so commute only up to the ~0.08% the even-width snap introduces; they are excluded here
#: because "almost commutes" is not a property to build on.
COMMUTING_HYPOTHESES = ("H_768_long", "H_1024_long", "H_832_iso", "H_y_anchored")

#: How far commutation actually holds, given that :data:`SIZE_MULTIPLE` snaps the width. The snap
#: makes the achieved scale marginally anisotropic -- 1920x1080 -> 854x480 is kx = 0.444792 against
#: ky = 0.444444, a relative gap of 0.078% -- so the commutation above is exact in principle and
#: sub-pixel in practice. Worst case measured across the pilot resolutions plus synthetic odd
#: sources: **0.67px of displacement at the frame edge**, which is an order of magnitude below the
#: annotation noise these boxes already carry (the canvas convention itself is unresolved to
#: within tens of pixels on the x axis). Snapping is not negotiable -- see :data:`SIZE_MULTIPLE` --
#: so this residual is the price, recorded rather than hidden.
COMMUTATION_TOLERANCE = 0.001


def scale_box(box, source_width: int, source_height: int,
              stored_width: int, stored_height: int) -> list[float]:
    """Rescale a **real-frame-pixel** box from source dimensions to stored dimensions.

    Provided for callers holding frame-space boxes (Grounding DINO output, ``box_space="frame"``)
    that were measured on a differently-sized decode of the same frame.

    **Stage B does not call this, and that is the important part.** Stage B's ``seed_bbox_768`` /
    ``ref.bbox_768`` are raw *annotation* coordinates in an unresolved canvas (the ``_768`` name is
    a documented misnomer), and they are projected onto a frame by
    ``segment.scale_bbox_to_frame``, which derives its own factor from the frame dimensions it is
    handed. Pre-multiplying the raw coordinates by the downscale factor would make that projection
    apply the shrink a *second* time -- 0.44 * 1.11 instead of 1.11, i.e. every box at 40% of its
    correct size, silently. The raw coordinates are left exactly as stage A wrote them; storing
    the new ``width``/``height`` alongside them is what keeps them correct.
    """
    x1, y1, x2, y2 = (float(value) for value in box)
    sx = stored_width / float(source_width)
    sy = stored_height / float(source_height)
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
