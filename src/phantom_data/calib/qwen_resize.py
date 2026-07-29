"""Qwen2.5-VL ``smart_resize``, reimplemented for offline canvas calibration.

Phantom-Data's bboxes were produced by Qwen2.5-VL grounding (paper arXiv 2506.18851
section 4.2.1) on 3 sampled frames. Qwen2.5-VL does NOT resize to an isotropic long
edge: its image processor aligns EACH side independently to a multiple of
``patch_size * merge_size`` (14 * 2 = 28) under an area budget
``min_pixels <= h*w <= max_pixels``. That per-side alignment is the only published
mechanism that could make the x axis of the annotations land on a small set of walls
(768 / 798 / 800 / 832) while the y axis lands somewhere else, so this module lets the
probe test the hypothesis numerically instead of by eyeball.

Kept dependency-free and pure so it is unit-testable on the dev box; the probe
cross-checks it against the real ``qwen_vl_utils`` / ``transformers`` implementation at
runtime when one is importable and records the verdict in its report.
"""
from __future__ import annotations

import math

#: ``patch_size * merge_size`` for Qwen2.5-VL: every side is rounded to a multiple of it.
FACTOR = 28

#: Defaults from the Qwen2.5-VL image processor.
MIN_PIXELS = 56 * 56
MAX_PIXELS = 14 * 14 * 4 * 1280

#: Guard from upstream: refuse absurd aspect ratios instead of producing a 0-size side.
MAX_RATIO = 200


def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Return ``(height, width)`` rescaled the way Qwen2.5-VL rescales an image.

    Faithful transcription of the upstream function:

    1. each side is independently rounded to the nearest multiple of ``factor``;
    2. if that exceeds ``max_pixels`` in area, both sides are scaled down by
       ``beta = sqrt(h*w / max_pixels)`` and floored to a multiple of ``factor``;
    3. if it undershoots ``min_pixels``, both are scaled up and ceiled instead.

    Note the asymmetry that matters for calibration: the area branches divide the
    ORIGINAL ``h`` and ``w`` by ``beta`` (not the already-rounded ``hbar``/``wbar``),
    and they floor/ceil rather than round. Raises :class:`ValueError` for aspect ratios
    above :data:`MAX_RATIO`, matching upstream.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            "absolute aspect ratio must be smaller than "
            f"{MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def isotropic_long_edge_canvas(width: int, height: int, long_edge: int = 768) -> tuple[float, float]:
    """The competing hypothesis: one isotropic canvas whose long edge is ``long_edge``.

    Returns ``(canvas_w, canvas_h)`` as floats (they are generally not integers: 16:9
    gives 768x432 but 2.22:1 gives 768x345.6). This is what the y axis of the
    annotations was already shown to obey.
    """
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    scale = float(long_edge) / float(max(width, height))
    return width * scale, height * scale
