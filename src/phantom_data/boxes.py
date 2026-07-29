"""Frame-space box arithmetic, shared by the renderers and the re-detection cascade.

Everything here works in **frame pixels**, after ``canvas.map_box`` has already taken an
annotation off Phantom's 768-long-edge canvas. That is the one distinction worth keeping in
mind: :mod:`phantom_data.canvas` converts between coordinate systems, this module never
does -- it only clips, measures, and compares boxes that are already in the frame's own
system.

These helpers previously existed three times over -- in two renderers and inline in the
cascade -- with the clamp arithmetic written out each time. A crop that is off by a pixel in
one copy and not another silently changes a CLIP score, so they live in one place with tests.

Clamping convention: ``x1``/``y1`` are clipped into ``[0, size - 1]`` and ``x2``/``y2`` into
``[x1 + 1, size]``, which guarantees a non-empty crop for any input -- including boxes that
are entirely off-frame, inverted, or zero-area. Callers that need to *know* a box was
degenerate should check the box itself; :func:`crop_box` will not tell them.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

Box = Sequence[float]


def is_box(box: Any) -> bool:
    """A box is four numbers. Anything else (None, a 2-tuple, a string) is not."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)


def clamp_box(box: Box | None, width: int, height: int) -> list[int] | None:
    """Integer pixel box clipped into a ``width`` x ``height`` frame, or None if not a box.

    The returned box always satisfies ``x2 > x1`` and ``y2 > y1``, so it can be used to
    slice an array without a further emptiness check.
    """
    if not is_box(box):
        return None
    x1 = max(0, min(int(round(box[0])), width - 1))
    y1 = max(0, min(int(round(box[1])), height - 1))
    x2 = max(x1 + 1, min(int(round(box[2])), width))
    y2 = max(y1 + 1, min(int(round(box[3])), height))
    return [x1, y1, x2, y2]


def crop_box(frame: np.ndarray, box: Box | None) -> np.ndarray | None:
    """Contents of ``box`` in ``frame``, clamped to the frame. None if ``box`` is not a box."""
    height, width = frame.shape[:2]
    clamped = clamp_box(box, width, height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    return np.asarray(frame[y1:y2, x1:x2], dtype=np.uint8)


def box_fraction(box: Box | None, frame: np.ndarray) -> float | None:
    """Box area as a fraction of frame area, measured on the box *as given*.

    Deliberately not clamped: a box hanging off the frame edge should report the area it
    claims, because that is what makes an over-large annotation visible in the report.
    Clamping first would make every overflowing box look like it fitted.
    """
    if not is_box(box):
        return None
    height, width = frame.shape[:2]
    area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    return round(area / float(width * height), 5)


def iou(a: Box | None, b: Box | None) -> float | None:
    """Intersection over union. None if either side is not a box or both are degenerate."""
    if not is_box(a) or not is_box(b):
        return None
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return round(inter / union, 4) if union > 0 else None
