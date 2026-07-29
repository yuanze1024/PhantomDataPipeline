"""Candidate coordinate systems ("canvases") for Phantom-Data bbox annotations.

**The convention is UNRESOLVED.** The package historically assumed the annotation
canvas is "long edge scaled to 768", i.e. ``scale = max(W, H) / 768``. That assumption
is falsified by measurement over ~41k target boxes joined to their true source
resolution: the *y* axis obeys the 768 fit exactly (1:1 sources cap at exactly 768,
2.22:1 sources at exactly 345 = 768*1080/2400), while *x* overshoots and clamps at four
distinct values -- 768 / 798 / 800 / 832 -- with an observed maximum of 981. A single
canvas cannot produce four x-clamps, so the annotations were most likely produced under
several resolution buckets (832x480-style video-generation buckets are the prime
suspect). The mixture is per-box, not per-shard: 93 of 983 multi-box source videos carry
both an ``x2 > 768`` box and a separate ``y2 == 432`` box, and table A has no provenance
column to bucket by.

This module is the single home for the competing hypotheses so that call sites take a
*named parameter* instead of a hardcoded constant. It is deliberately pure: no numpy, no
I/O, and no import of :mod:`phantom_data.dataset` (the dependency runs the other way --
``dataset.scale_bbox`` is a thin wrapper over :func:`map_box` under
:data:`H_768_long`).

Anisotropy is the core capability here. The legacy ``scale_bbox`` is isotropic-only and
therefore cannot even express half of the hypotheses below, which is why the mixture was
never testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

#: Long edge of the historically-assumed canvas. Kept as the default everywhere so that
#: every existing code path -- and the already-built pilot dataset -- stays bit-identical.
CANVAS = 768.0

#: Type of a scale-factor function: ``(src_w, src_h) -> (sx, sy)``.
ScaleFn = Callable[[float, float], "tuple[float, float]"]


@dataclass(frozen=True)
class Hypothesis:
    """One candidate annotation canvas.

    ``scale_fn`` returns the *independent* ``(sx, sy)`` divisor-free multipliers that map
    an annotation coordinate onto a real source frame of size ``(src_w, src_h)``.
    ``formula`` is human-readable and is what the calibration frontend shows per panel.
    """

    id: str
    label: str
    formula: str
    scale_fn: ScaleFn = field(repr=False)
    note: str = ""

    def scales(self, src_w: float, src_h: float) -> tuple[float, float]:
        """``(sx, sy)`` for this hypothesis on a ``(src_w, src_h)`` source frame."""
        return self.scale_fn(float(src_w), float(src_h))

    def map_box(self, box: Iterable[float], src_w: float, src_h: float) -> list[float]:
        return map_box(box, src_w, src_h, self)

    def unmap_box(self, box: Iterable[float], src_w: float, src_h: float) -> list[float]:
        return unmap_box(box, src_w, src_h, self)


# --------------------------------------------------------------------------------------
# constructors (the registry stays cheap to extend from measured numbers)
# --------------------------------------------------------------------------------------


def long_edge(id: str, label: str, canvas: float, note: str = "") -> Hypothesis:
    """Isotropic: the source's long edge was fitted to ``canvas``."""
    def scale_fn(src_w: float, src_h: float) -> tuple[float, float]:
        scale = max(src_w, src_h) / canvas
        return scale, scale

    return Hypothesis(
        id=id, label=label, formula=f"s = max(W,H)/{canvas:g}; sx = sy = s",
        scale_fn=scale_fn, note=note,
    )


def isotropic_width(id: str, label: str, canvas_w: float, note: str = "") -> Hypothesis:
    """Isotropic, but pinned on the width: the canvas is ``canvas_w`` wide."""
    def scale_fn(src_w: float, src_h: float) -> tuple[float, float]:
        scale = src_w / canvas_w
        return scale, scale

    return Hypothesis(
        id=id, label=label, formula=f"s = W/{canvas_w:g}; sx = sy = s",
        scale_fn=scale_fn, note=note,
    )


def anisotropic(id: str, label: str, canvas_w: float, canvas_h: float,
                note: str = "") -> Hypothesis:
    """Independent axes: the annotation frame was ``canvas_w x canvas_h``, aspect broken.

    This is the constructor a data-driven hypothesis should be built with once the
    estimator reports per-aspect spike values.
    """
    def scale_fn(src_w: float, src_h: float) -> tuple[float, float]:
        return src_w / canvas_w, src_h / canvas_h

    return Hypothesis(
        id=id, label=label,
        formula=f"sx = W/{canvas_w:g}; sy = H/{canvas_h:g}",
        scale_fn=scale_fn, note=note,
    )


def qwen_smart_resize(id: str, label: str, note: str = "") -> Hypothesis:
    """The canvas Qwen2.5-VL actually feeds its vision tower, per side.

    Unlike every other entry here this is not a guessed constant: Phantom-Data's boxes come
    from Qwen2.5-VL grounding (arXiv 2506.18851 s4.2.1), whose image processor rounds each
    side independently to a multiple of 28 under an area budget. That per-side rounding is
    the only published mechanism that can put the x axis on several walls while y sits
    elsewhere, which is exactly the measured signature.
    """
    from .calib.qwen_resize import smart_resize

    def scale_fn(src_w: float, src_h: float) -> tuple[float, float]:
        canvas_h, canvas_w = smart_resize(int(round(src_h)), int(round(src_w)))
        return src_w / canvas_w, src_h / canvas_h

    return Hypothesis(
        id=id, label=label,
        formula="(ch,cw) = qwen_smart_resize(H,W); sx = W/cw; sy = H/ch",
        scale_fn=scale_fn, note=note,
    )


def y_anchored(id: str, label: str, canvas: float, note: str = "") -> Hypothesis:
    """Trust the y axis: the canvas *height* is the ``canvas``-long-edge fit of the source.

    ``sy = H / (canvas * H / max(W,H))``, and ``sx = sy`` (isotropic). Note this is
    algebraically ``max(W,H)/canvas`` -- see :data:`H_y_anchored`.
    """
    def scale_fn(src_w: float, src_h: float) -> tuple[float, float]:
        canvas_h = canvas * src_h / max(src_w, src_h)
        scale = src_h / canvas_h
        return scale, scale

    return Hypothesis(
        id=id, label=label,
        formula=f"canvas_h = {canvas:g}*H/max(W,H); sy = H/canvas_h; sx = sy",
        scale_fn=scale_fn, note=note,
    )


# --------------------------------------------------------------------------------------
# mapping
# --------------------------------------------------------------------------------------


def scales(hyp: Hypothesis, src_w: float, src_h: float) -> tuple[float, float]:
    """``(sx, sy)`` alone, for display next to a rendered panel."""
    return hyp.scales(src_w, src_h)


def map_box(box: Iterable[float], src_w: float, src_h: float,
            hyp: Hypothesis) -> list[float]:
    """Annotation xyxy -> real-frame xyxy under ``hyp``, x and y scaled independently.

    No clamping and no corner normalisation happen here; callers that need frame bounds
    (stage C) apply them afterwards.
    """
    x1, y1, x2, y2 = box
    sx, sy = hyp.scales(src_w, src_h)
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def unmap_box(box: Iterable[float], src_w: float, src_h: float,
              hyp: Hypothesis) -> list[float]:
    """Real-frame xyxy -> annotation xyxy: the exact inverse of :func:`map_box`."""
    x1, y1, x2, y2 = box
    sx, sy = hyp.scales(src_w, src_h)
    return [x1 / sx, y1 / sy, x2 / sx, y2 / sy]


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------

H_768_long = long_edge(
    "H_768_long", "long edge = 768", CANVAS,
    note="The historical assumption and today's behaviour of every code path; it is what "
         "the already-built pilot dataset was produced with. Fits y exactly, cannot "
         "explain x > 768.",
)
H_1024_long = long_edge(
    "H_1024_long", "long edge = 1024", 1024.0,
    note="Already falsified: 1080x1080 sources cap at y2 = 768, not 1024. Kept so the "
         "calibration UI can show it failing.",
)
H_832_iso = isotropic_width(
    "H_832_iso", "width = 832 (aspect preserved)", 832.0,
    note="Takes the dominant x-clamp as the canvas width while keeping the aspect ratio; "
         "predicts y2 up to 468 on 16:9, close to the observed 465-498.",
)
H_832x480_aniso = anisotropic(
    "H_832x480_aniso", "832 x 480 (aspect broken)", 832.0, 480.0,
    note="A standard video-generation bucket. Explains the 832 x-clamp and the observed "
         "y2 maxima of 465-498 at the same time.",
)
H_y_anchored = y_anchored(
    "H_y_anchored", "y-anchored 768 fit", CANVAS,
    note="Trusts the axis the measurement vindicates. Under isotropy (sx = sy) this "
         "reduces to H_768_long for EVERY aspect ratio, not just 16:9 -- kept as an "
         "explicit record of the y-axis reading, and as evidence that no isotropic "
         "hypothesis can be y-correct and x-correct at once.",
)
H_norm1000 = anisotropic(
    "H_norm1000", "0-1000 normalised", 1000.0, 1000.0,
    note="The common VLM convention: coordinates are per-mille of width/height, so the "
         "aspect ratio is broken by construction.",
)
H_qwen_smart = qwen_smart_resize(
    "H_qwen_smart", "Qwen2.5-VL smart_resize (per-side, x28)",
    note="Mechanism-derived rather than guessed: the annotator was Qwen2.5-VL, whose "
         "processor rounds each side to a multiple of 28 under an area budget. Predicts "
         "1280x720 -> 1288x728 (scales ~0.99, i.e. annotations are almost literal pixels) "
         "and 1920x1080 -> 1316x728. The per-side rounding is the only published mechanism "
         "that can produce several x walls with y elsewhere.",
)

#: Ordered id -> hypothesis. Insertion order is the display order in the calibration UI.
HYPOTHESES: dict[str, Hypothesis] = {
    hypothesis.id: hypothesis
    for hypothesis in (
        H_768_long,
        H_1024_long,
        H_832_iso,
        H_832x480_aniso,
        H_y_anchored,
        H_norm1000,
        H_qwen_smart,
    )
}

# ``H_adaptive`` -- a per-aspect anisotropic() built from the measured spike values -- lands
# here once /mnt/pfs/users/yuanze/datasets/phantom_canvas_calib_v1/canvas_estimator.json exists.


def get(hypothesis_id: str) -> Hypothesis:
    """Look a hypothesis up by id, with the valid ids in the error message."""
    try:
        return HYPOTHESES[hypothesis_id]
    except KeyError:
        raise KeyError(
            f"unknown canvas hypothesis {hypothesis_id!r}; known: {', '.join(HYPOTHESES)}"
        ) from None
