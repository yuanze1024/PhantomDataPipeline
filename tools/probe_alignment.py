"""Probe: are the Phantom-Data x-clamps a per-side ALIGNMENT rule (Qwen2.5-VL smart_resize)?

What the three earlier scans (``phantom_data.calib.estimate``, ``tools/probe_long_edge.py``,
``tools/probe_xclamp_by_resolution.py``) already established:

* the **y** axis strictly obeys ONE isotropic "long edge = 768" canvas in every aspect
  bucket (16:9 -> 432, 4:3 -> 576, 1:1 -> 768, 2.22:1 -> 345.6),
* the **x** axis does not: 14.4% of 16:9 boxes have ``x2 > 768``, clamping at
  768 / 798 / 800 / 832 (4:3 clamps at ~806, 2.22:1 never overflows),
* "mixture of isotropic canvases" is falsified (51.7% of ``x2==832`` boxes sit on
  ``y2==432``, the *768* canvas height, so x and y clamp independently),
* the clamp is NOT a function of the source resolution.

New hypothesis under test here. The paper says the boxes came from **Qwen2.5-VL**
grounding. Qwen2.5-VL's image processor does not use an isotropic long edge: its
``smart_resize`` aligns EACH side independently to a multiple of
``patch_size * merge_size`` = 28, under a ``min_pixels``/``max_pixels`` area budget.
Per-side alignment is the only published mechanism that would let x land on a small set
of walls while y lands elsewhere. And 768 = 32*24, 800 = 32*25, 832 = 32*26 are three
consecutive multiples of 32, so 32-alignment is a live variant.

Four things are measured, over the full table, per aspect bucket and per kind:

1. residue histograms of ``round(x2) mod 28`` / ``mod 32`` and the same for ``y2``, both
   for ALL boxes and for the overflow subpopulation (``x2`` beyond the 768-fit canvas
   width). Flat => no alignment; a spike => alignment. Reported as top-3 residues plus
   the max/min bin ratio.
2. the sharper test: alignment says the CANVAS WIDTH is a multiple of 28 (or 32), not
   every box coordinate. So the distinct ``x2`` values carrying >= 0.5% of a cell (the
   candidate clamps) are each divided by 28 and 32 and the verdict is stated plainly.
3. the decisive test: :func:`phantom_data.calib.qwen_resize.smart_resize` is run for
   every distinct source resolution under a grid of ``max_pixels`` settings, and each
   resulting canvas is checked against BOTH the observed x clamps and the observed y
   walls. A setting only survives if it explains both axes at once.
4. the y cross-check: is the y wall a multiple of 28 / 32 at all?

Streaming only (``iter_batches`` via :func:`phantom_data.calib.join.iter_boxes`); table A
is one 1.09 GB row group and is never materialized.

    PYTHONPATH=third_party/PhantomData/src \
    python third_party/PhantomData/tools/probe_alignment.py \
        --out /mnt/pfs/users/yuanze/datasets/phantom_canvas_calib_v1/alignment_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict

from phantom_data.calib.join import (
    FILTERED_PARQUET,
    META_PARQUET,
    JoinStats,
    aspect_bucket,
    build_vid_wh,
    iter_boxes,
)
from phantom_data.calib.qwen_resize import (
    isotropic_long_edge_canvas,
    smart_resize,
)

#: The canvas the y axis was shown to obey; overflow is measured against its width.
LONG_EDGE = 768

#: The two alignment moduli under test. 28 = Qwen2.5-VL patch*merge; 32 because
#: 768/800/832 are consecutive multiples of it.
MODULI = (28, 32)

#: ``max_pixels`` settings for the smart_resize sweep. 1003520 = 1280*28*28 is the
#: Qwen2.5-VL default; 12845056 is the HF ``Qwen2VLImageProcessor`` default.
MAX_PIXELS_GRID = (
    ("640*28*28", 640 * 28 * 28),
    ("768*28*28", 768 * 28 * 28),
    ("1024*28*28", 1024 * 28 * 28),
    ("1280*28*28 (qwen2.5-vl default)", 1280 * 28 * 28),
    ("12845056 (hf default)", 12845056),
)

#: A distinct x2 value must carry at least this share of a cell to count as a candidate
#: clamp (item 2). 0.5% over a >=10k-box cell is well above histogram noise.
CANDIDATE_MIN_FRAC = 0.005

#: Tolerance when calling a predicted canvas edge a "match" for an observed wall.
MATCH_TOL = 1.0


def fit_canvas_width(src_w: int, src_h: int) -> float:
    """Width of the 768-long-edge isotropic canvas for this source resolution."""
    return isotropic_long_edge_canvas(src_w, src_h, LONG_EDGE)[0]


def fit_canvas_height(src_w: int, src_h: int) -> float:
    """Height of the 768-long-edge isotropic canvas (the observed y wall)."""
    return isotropic_long_edge_canvas(src_w, src_h, LONG_EDGE)[1]


class ResidueHist:
    """Residue counters for one coordinate under every modulus in :data:`MODULI`."""

    def __init__(self) -> None:
        self.n = 0
        self.by_modulus: dict[int, Counter] = {modulus: Counter() for modulus in MODULI}

    def add(self, value: float) -> None:
        rounded = int(round(value))
        self.n += 1
        for modulus, counter in self.by_modulus.items():
            counter[rounded % modulus] += 1

    def summarize(self) -> dict:
        out: dict[str, dict] = {}
        for modulus, counter in self.by_modulus.items():
            # Every residue gets an entry, including the empty ones: a spike is only
            # meaningful next to the bins it beat, and min_bin==0 is itself a signal.
            counts = [counter.get(residue, 0) for residue in range(modulus)]
            total = sum(counts)
            top = sorted(
                ((residue, count) for residue, count in enumerate(counts)),
                key=lambda item: (-item[1], item[0]),
            )[:3]
            uniform = (total / modulus) if total else 0.0
            out[str(modulus)] = {
                "n": total,
                "counts": counts,
                "fracs": [(count / total) if total else 0.0 for count in counts],
                "top3": [
                    {
                        "residue": residue,
                        "n": count,
                        "frac": (count / total) if total else 0.0,
                        "x_uniform": (count / uniform) if uniform else 0.0,
                    }
                    for residue, count in top
                ],
                "max_bin": max(counts) if counts else 0,
                "min_bin": min(counts) if counts else 0,
                # inf when some residue is unoccupied: reported as None to stay JSON-safe.
                "max_over_min": (max(counts) / min(counts)) if counts and min(counts) else None,
                "frac_residue_zero": (counts[0] / total) if total else 0.0,
            }
        return out


class Cell:
    """Per-``(aspect_bucket, kind)`` accumulator."""

    def __init__(self) -> None:
        self.n = 0
        self.n_overflow = 0
        self.x2_all = ResidueHist()
        self.y2_all = ResidueHist()
        self.x2_overflow = ResidueHist()
        self.y2_overflow = ResidueHist()
        self.x2_hist: Counter = Counter()
        self.y2_hist: Counter = Counter()
        self.x2_overflow_hist: Counter = Counter()
        self.x2_noninteger = 0
        self.y2_noninteger = 0
        self.resolutions: Counter = Counter()

    def add(self, box, src_w: int, src_h: int) -> None:
        x2, y2 = float(box[2]), float(box[3])
        self.n += 1
        self.resolutions[(src_w, src_h)] += 1
        self.x2_all.add(x2)
        self.y2_all.add(y2)
        self.x2_hist[int(round(x2))] += 1
        self.y2_hist[int(round(y2))] += 1
        if abs(x2 - round(x2)) > 1e-6:
            self.x2_noninteger += 1
        if abs(y2 - round(y2)) > 1e-6:
            self.y2_noninteger += 1
        if x2 > fit_canvas_width(src_w, src_h) + 1e-6:
            self.n_overflow += 1
            self.x2_overflow.add(x2)
            self.y2_overflow.add(y2)
            self.x2_overflow_hist[int(round(x2))] += 1

    def candidates(self, hist: Counter, min_frac: float) -> list[dict]:
        """Distinct values carrying >= ``min_frac`` of the cell, with mod-28/32 arithmetic."""
        total = self.n
        out = []
        for value, count in sorted(hist.items(), key=lambda item: -item[1]):
            frac = (count / total) if total else 0.0
            if frac < min_frac:
                continue
            out.append(
                {
                    "value": value,
                    "n": count,
                    "frac": frac,
                    "mod_28": value % 28,
                    "mod_32": value % 32,
                    "div_28": value / 28.0,
                    "div_32": value / 32.0,
                    "is_mult_28": value % 28 == 0,
                    "is_mult_32": value % 32 == 0,
                }
            )
        return out

    def summarize(self, min_frac: float) -> dict:
        x_candidates = self.candidates(self.x2_hist, min_frac)
        y_candidates = self.candidates(self.y2_hist, min_frac)
        return {
            "n": self.n,
            "n_overflow": self.n_overflow,
            "frac_overflow": (self.n_overflow / self.n) if self.n else 0.0,
            "x2_noninteger_n": self.x2_noninteger,
            "y2_noninteger_n": self.y2_noninteger,
            "residues": {
                "x2_all": self.x2_all.summarize(),
                "x2_overflow": self.x2_overflow.summarize(),
                "y2_all": self.y2_all.summarize(),
                "y2_overflow": self.y2_overflow.summarize(),
            },
            "x2_candidates": x_candidates,
            "y2_candidates": y_candidates,
            "x2_candidate_verdict": candidate_verdict(x_candidates),
            "y2_candidate_verdict": candidate_verdict(y_candidates),
            "top_resolutions": [
                {"resolution": f"{w}x{h}", "n": count}
                for (w, h), count in self.resolutions.most_common(6)
            ],
        }


def candidate_verdict(candidates: list[dict]) -> str:
    """Plain-language verdict for item 2: all ==0 mod 28, mod 32, mixed, or neither."""
    if not candidates:
        return "no candidate values"
    all_28 = all(item["is_mult_28"] for item in candidates)
    all_32 = all(item["is_mult_32"] for item in candidates)
    any_28 = any(item["is_mult_28"] for item in candidates)
    any_32 = any(item["is_mult_32"] for item in candidates)
    if all_28 and all_32:
        return "ALL are multiples of BOTH 28 and 32"
    if all_28:
        return "ALL are multiples of 28"
    if all_32:
        return "ALL are multiples of 32"
    if any_28 or any_32:
        parts = []
        if any_28:
            parts.append(
                "mult-of-28: " + ",".join(str(i["value"]) for i in candidates if i["is_mult_28"])
            )
        if any_32:
            parts.append(
                "mult-of-32: " + ",".join(str(i["value"]) for i in candidates if i["is_mult_32"])
            )
        return "MIXED (" + "; ".join(parts) + ")"
    return "NEITHER: no candidate is a multiple of 28 or 32"


# ---------------------------------------------------------------------------
# item 3: smart_resize sweep
# ---------------------------------------------------------------------------


def observed_walls(cells: dict[tuple[str, str], Cell], min_frac: float) -> dict[str, dict]:
    """Per-bucket observed x clamps (union over kinds) and the 768-canvas y wall."""
    per_bucket: dict[str, dict] = {}
    for (bucket, _kind), cell in cells.items():
        entry = per_bucket.setdefault(bucket, {"x_clamps": set(), "n": 0, "y_walls": Counter()})
        entry["n"] += cell.n
        for item in cell.candidates(cell.x2_hist, min_frac):
            entry["x_clamps"].add(item["value"])
        for (w, h), count in cell.resolutions.items():
            entry["y_walls"][round(fit_canvas_height(w, h), 2)] += count
    return {
        bucket: {
            "n": entry["n"],
            "x_clamps": sorted(entry["x_clamps"]),
            "y_wall": entry["y_walls"].most_common(1)[0][0] if entry["y_walls"] else None,
        }
        for bucket, entry in per_bucket.items()
    }


def sweep_smart_resize(
    resolutions: Counter,
    walls: dict[str, dict],
    factors=(28, 32),
    grid=MAX_PIXELS_GRID,
    top_n: int = 12,
) -> dict:
    """Run smart_resize over (resolution x factor x max_pixels) and score each setting.

    A setting is scored on the box mass whose predicted canvas *width* lands on one of
    that bucket's observed x clamps, and separately on the mass whose predicted canvas
    *height* lands on that bucket's observed y wall. The hypothesis needs BOTH.
    """
    top = resolutions.most_common(top_n)
    total_mass = sum(count for _res, count in top)
    settings = []
    for factor in factors:
        for label, max_pixels in grid:
            rows = []
            width_hit = 0
            height_hit = 0
            both_hit = 0
            for (src_w, src_h), count in top:
                bucket = aspect_bucket(src_w, src_h)
                wall = walls.get(bucket, {})
                clamps = wall.get("x_clamps") or []
                y_wall = wall.get("y_wall")
                try:
                    canvas_h, canvas_w = smart_resize(
                        src_h, src_w, factor=factor, max_pixels=max_pixels
                    )
                    error = None
                except ValueError as exc:  # pragma: no cover - guarded resolutions only
                    canvas_h = canvas_w = None
                    error = str(exc)
                w_ok = canvas_w is not None and any(
                    abs(canvas_w - clamp) <= MATCH_TOL for clamp in clamps
                )
                h_ok = (
                    canvas_h is not None
                    and y_wall is not None
                    and abs(canvas_h - y_wall) <= MATCH_TOL
                )
                if w_ok:
                    width_hit += count
                if h_ok:
                    height_hit += count
                if w_ok and h_ok:
                    both_hit += count
                rows.append(
                    {
                        "resolution": f"{src_w}x{src_h}",
                        "aspect_bucket": bucket,
                        "n_boxes": count,
                        "canvas_w": canvas_w,
                        "canvas_h": canvas_h,
                        "error": error,
                        "observed_x_clamps": clamps,
                        "observed_y_wall": y_wall,
                        "width_matches_a_clamp": w_ok,
                        "height_matches_y_wall": h_ok,
                        "width_residual_to_nearest_clamp": (
                            min((abs(canvas_w - clamp) for clamp in clamps), default=None)
                            if canvas_w is not None
                            else None
                        ),
                        "height_residual_to_y_wall": (
                            abs(canvas_h - y_wall)
                            if canvas_h is not None and y_wall is not None
                            else None
                        ),
                    }
                )
            settings.append(
                {
                    "factor": factor,
                    "max_pixels_label": label,
                    "max_pixels": max_pixels,
                    "mass_width_match": (width_hit / total_mass) if total_mass else 0.0,
                    "mass_height_match": (height_hit / total_mass) if total_mass else 0.0,
                    "mass_both_match": (both_hit / total_mass) if total_mass else 0.0,
                    "per_resolution": rows,
                }
            )
    settings.sort(key=lambda item: (-item["mass_both_match"], -item["mass_width_match"]))
    best = settings[0] if settings else None
    return {
        "total_box_mass_considered": total_mass,
        "resolutions_considered": [f"{w}x{h}" for (w, h), _count in top],
        "settings": settings,
        "best": (
            {
                "factor": best["factor"],
                "max_pixels_label": best["max_pixels_label"],
                "mass_both_match": best["mass_both_match"],
                "mass_width_match": best["mass_width_match"],
                "mass_height_match": best["mass_height_match"],
            }
            if best
            else None
        ),
        "decisive_verdict": (
            "SUPPORTED"
            if best and best["mass_both_match"] >= 0.5
            else ("PARTIAL" if best and best["mass_both_match"] > 0.0 else "FALSIFIED")
        ),
    }


def y_wall_alignment_check(walls: dict[str, dict]) -> list[dict]:
    """Item 4: is each bucket's y wall a multiple of 28 or 32 at all?"""
    out = []
    for bucket, entry in sorted(walls.items(), key=lambda item: -item[1]["n"]):
        wall = entry["y_wall"]
        if wall is None:
            continue
        out.append(
            {
                "aspect_bucket": bucket,
                "n": entry["n"],
                "y_wall": wall,
                "is_integer": abs(wall - round(wall)) < 1e-6,
                "div_28": wall / 28.0,
                "div_32": wall / 32.0,
                "mod_28": round(wall % 28, 4),
                "mod_32": round(wall % 32, 4),
                "is_mult_28": abs(wall % 28) < 1e-6,
                "is_mult_32": abs(wall % 32) < 1e-6,
            }
        )
    return out


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def verify_against_upstream(log=print) -> dict:
    """Compare the local ``smart_resize`` to a real one if any is importable.

    Tries ``qwen_vl_utils.vision_process`` then the transformers Qwen2-VL image
    processor. A missing import is not a failure -- it is recorded so the report can
    say the local transcription was used unverified.
    """
    cases = [
        (1080, 1920),
        (720, 1280),
        (2160, 3840),
        (1440, 2560),
        (768, 1024),
        (1080, 1080),
        (486, 1080),
        (30, 40),
    ]

    def call(function, height, width, factor, max_pixels):
        """``(result, error_type)`` -- upstream raises on some inputs ours accepts.

        The transformers copy adds a ``height < factor`` guard that the paper snippet
        does not have; a raise on ONE side is a documented difference, not a numeric
        disagreement, so it is recorded separately from a value mismatch.
        """
        try:
            return tuple(function(height, width, factor=factor, max_pixels=max_pixels)), None
        except TypeError:  # different kwarg names upstream
            try:
                return tuple(function(height, width)), None
            except Exception as exc:
                return None, type(exc).__name__
        except Exception as exc:
            return None, type(exc).__name__
    for module_name, attr in (
        ("qwen_vl_utils.vision_process", "smart_resize"),
        ("transformers.models.qwen2_vl.image_processing_qwen2_vl", "smart_resize"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr])
            reference = getattr(module, attr)
        except Exception as exc:  # ImportError, or transformers exploding on import
            log(f"[align] no {module_name}: {type(exc).__name__}: {exc}")
            continue
        mismatches = []
        raise_only = []
        compared = 0
        for height, width in cases:
            for factor in (28, 32):
                for _label, max_pixels in MAX_PIXELS_GRID:
                    mine, mine_error = call(smart_resize, height, width, factor, max_pixels)
                    theirs, theirs_error = call(reference, height, width, factor, max_pixels)
                    record = {
                        "hw": [height, width],
                        "factor": factor,
                        "max_pixels": max_pixels,
                        "mine": list(mine) if mine else None,
                        "theirs": list(theirs) if theirs else None,
                        "mine_error": mine_error,
                        "theirs_error": theirs_error,
                    }
                    if (mine is None) != (theirs is None):
                        raise_only.append(record)
                        continue
                    if mine is None:
                        continue
                    compared += 1
                    if mine != theirs:
                        mismatches.append(record)
        return {
            "verified_against": module_name,
            "cases": len(cases),
            "numeric_comparisons": compared,
            "mismatches": mismatches,
            "guard_only_differences": raise_only,
            "agrees": not mismatches,
        }
    return {
        "verified_against": None,
        "note": "no qwen_vl_utils / transformers smart_resize importable in this "
        "environment; local transcription used as written",
        "agrees": None,
    }


def scan(
    filtered_parquet: str,
    meta_parquet: str,
    max_rows: int,
    batch_size: int,
    meta_batch_size: int,
    candidate_min_frac: float,
    log_every: int,
    log=print,
) -> dict:
    started = time.time()
    verification = verify_against_upstream(log=log)
    log(f"[align] building vid->(W,H) from {meta_parquet}")
    vid_wh = build_vid_wh(meta_parquet, batch_size=meta_batch_size)
    log(f"[align] vid_wh entries={len(vid_wh)} ({time.time() - started:.1f}s)")

    stats = JoinStats()
    cells: dict[tuple[str, str], Cell] = defaultdict(Cell)
    all_resolutions: Counter = Counter()
    next_log = log_every

    for kind, box, src_w, src_h, _vid, _phrase, _cls in iter_boxes(
        filtered_parquet, vid_wh, batch_size=batch_size, max_rows=max_rows, stats=stats
    ):
        bucket = aspect_bucket(src_w, src_h)
        cells[(bucket, kind)].add(box, src_w, src_h)
        all_resolutions[(src_w, src_h)] += 1
        if log_every and stats.rows_read >= next_log:
            next_log = stats.rows_read + log_every
            log(
                f"[align] rows={stats.rows_read} boxes={stats.boxes_emitted} "
                f"cells={len(cells)} elapsed={time.time() - started:.0f}s"
            )

    log(
        f"[align] scan done rows={stats.rows_read} boxes={stats.boxes_emitted} "
        f"cells={len(cells)} elapsed={time.time() - started:.0f}s"
    )

    walls = observed_walls(cells, candidate_min_frac)
    entries = [
        {"aspect_bucket": bucket, "kind": kind, **cell.summarize(candidate_min_frac)}
        for (bucket, kind), cell in cells.items()
    ]
    entries.sort(key=lambda item: -item["n"])

    return {
        "params": {
            "filtered_parquet": filtered_parquet,
            "meta_parquet": meta_parquet,
            "max_rows": max_rows,
            "batch_size": batch_size,
            "long_edge": LONG_EDGE,
            "moduli": list(MODULI),
            "candidate_min_frac": candidate_min_frac,
            "match_tol": MATCH_TOL,
            "max_pixels_grid": [{"label": label, "value": value} for label, value in MAX_PIXELS_GRID],
            "residue_note": "residues computed on round(coord); non-integer coordinate "
            "counts reported per cell",
        },
        "global": {
            "rows_read": stats.rows_read,
            "boxes": stats.boxes_emitted,
            "counters": stats.as_dict(),
            "vid_wh_entries": len(vid_wh),
            "distinct_resolutions": len(all_resolutions),
            "wall_clock_sec": round(time.time() - started, 1),
        },
        "smart_resize_verification": verification,
        "observed_walls": walls,
        "cells": entries,
        "smart_resize_sweep": sweep_smart_resize(all_resolutions, walls),
        "y_wall_alignment": y_wall_alignment_check(walls),
    }


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------


def _residue_line(label: str, summary: dict) -> str:
    parts = []
    for modulus in MODULI:
        entry = summary[str(modulus)]
        top3 = ", ".join(
            f"r{item['residue']}={item['frac'] * 100:.1f}%({item['x_uniform']:.1f}x)"
            for item in entry["top3"]
        )
        ratio = entry["max_over_min"]
        ratio_text = f"{ratio:.1f}" if ratio is not None else "inf(empty bin)"
        parts.append(f"mod{modulus}: n={entry['n']} top3[{top3}] max/min={ratio_text}")
    return f"    {label:<12} " + "  |  ".join(parts)


def format_report(report: dict) -> str:
    lines: list[str] = []
    glob = report["global"]
    lines.append("=" * 120)
    lines.append("ALIGNMENT PROBE -- do the x-clamps come from a per-side (28 / 32) alignment rule?")
    lines.append("=" * 120)
    lines.append(f"rows read      : {glob['rows_read']}")
    lines.append(f"boxes analyzed : {glob['boxes']}")
    lines.append(f"counters       : {glob['counters']}")
    lines.append(f"resolutions    : {glob['distinct_resolutions']} distinct (W,H)")
    lines.append(f"wall clock     : {glob['wall_clock_sec']}s")
    verification = report["smart_resize_verification"]
    if verification.get("verified_against"):
        lines.append(
            f"smart_resize   : cross-checked against {verification['verified_against']} on "
            f"{verification['numeric_comparisons']} inputs -> "
            f"{'AGREES' if verification['agrees'] else 'MISMATCH ' + json.dumps(verification['mismatches'][:3])}"
            f"; guard-only differences: {len(verification['guard_only_differences'])}"
        )
    else:
        lines.append(f"smart_resize   : {verification['note']}")
    lines.append("")

    sweep = report["smart_resize_sweep"]
    lines.append("### ITEM 3 (DECISIVE): can ONE smart_resize setting explain x clamps AND y walls?")
    lines.append(f"  verdict: {sweep['decisive_verdict']}")
    if sweep["best"]:
        best = sweep["best"]
        lines.append(
            f"  best setting: factor={best['factor']} max_pixels={best['max_pixels_label']} "
            f"-> both-axes mass {best['mass_both_match'] * 100:.2f}%, "
            f"width-only {best['mass_width_match'] * 100:.2f}%, "
            f"height-only {best['mass_height_match'] * 100:.2f}%"
        )
    lines.append("")
    lines.append("  per-setting mass matched (of the top resolutions by box count):")
    header = f"  {'factor':>6} {'max_pixels':<34}{'width%':>9}{'height%':>9}{'both%':>9}"
    lines.append("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    for setting in sweep["settings"]:
        lines.append(
            f"  {setting['factor']:>6} {setting['max_pixels_label']:<34}"
            f"{setting['mass_width_match'] * 100:>8.2f}%"
            f"{setting['mass_height_match'] * 100:>8.2f}%"
            f"{setting['mass_both_match'] * 100:>8.2f}%"
        )
    lines.append("-" * len(header))
    lines.append("")
    lines.append("  predicted canvases per (resolution, setting):")
    for setting in sweep["settings"]:
        lines.append(
            f"    factor={setting['factor']} max_pixels={setting['max_pixels_label']}"
        )
        for row in setting["per_resolution"]:
            if row["error"]:
                lines.append(f"      {row['resolution']:>11} ERROR {row['error']}")
                continue
            lines.append(
                f"      {row['resolution']:>11} {row['aspect_bucket']:<8} n={row['n_boxes']:<7} "
                f"-> canvas {row['canvas_w']}x{row['canvas_h']}  "
                f"obs x clamps={row['observed_x_clamps']} y wall={row['observed_y_wall']}  "
                f"dx={row['width_residual_to_nearest_clamp']} dy={row['height_residual_to_y_wall']}  "
                f"{'W-HIT' if row['width_matches_a_clamp'] else 'w-miss'}/"
                f"{'H-HIT' if row['height_matches_y_wall'] else 'h-miss'}"
            )
    lines.append("")

    lines.append("### ITEM 2: candidate clamp values (>= "
                 f"{report['params']['candidate_min_frac'] * 100:.1f}% of a cell) vs 28 / 32")
    for cell in report["cells"]:
        lines.append(
            f"  {cell['aspect_bucket']:<8} {cell['kind']:<7} n={cell['n']:<7} "
            f"overflow={cell['frac_overflow'] * 100:.2f}%"
        )
        for item in cell["x2_candidates"]:
            lines.append(
                f"    x2={item['value']:<6} {item['frac'] * 100:>6.2f}%  "
                f"mod28={item['mod_28']:<3} mod32={item['mod_32']:<3} "
                f"/28={item['div_28']:<9.4f} /32={item['div_32']:<9.4f}"
            )
        lines.append(f"    verdict(x2): {cell['x2_candidate_verdict']}")
        for item in cell["y2_candidates"]:
            lines.append(
                f"    y2={item['value']:<6} {item['frac'] * 100:>6.2f}%  "
                f"mod28={item['mod_28']:<3} mod32={item['mod_32']:<3} "
                f"/28={item['div_28']:<9.4f} /32={item['div_32']:<9.4f}"
            )
        lines.append(f"    verdict(y2): {cell['y2_candidate_verdict']}")
    lines.append("")

    lines.append("### ITEM 1: residue histograms (top-3 residues, x_uniform = bin/uniform, max/min)")
    for cell in report["cells"]:
        lines.append(
            f"  {cell['aspect_bucket']:<8} {cell['kind']:<7} n={cell['n']:<7} "
            f"overflow n={cell['n_overflow']:<7} "
            f"noninteger x2={cell['x2_noninteger_n']} y2={cell['y2_noninteger_n']}"
        )
        residues = cell["residues"]
        lines.append(_residue_line("x2 ALL", residues["x2_all"]))
        lines.append(_residue_line("x2 OVERFLOW", residues["x2_overflow"]))
        lines.append(_residue_line("y2 ALL", residues["y2_all"]))
        lines.append(_residue_line("y2 OVERFLOW", residues["y2_overflow"]))
    lines.append("")

    lines.append("### ITEM 4: is the y wall alignment-quantized at all?")
    for entry in report["y_wall_alignment"]:
        lines.append(
            f"  {entry['aspect_bucket']:<8} n={entry['n']:<7} y_wall={entry['y_wall']:<8} "
            f"/28={entry['div_28']:<9.4f} /32={entry['div_32']:<9.4f} "
            f"mod28={entry['mod_28']:<8} mod32={entry['mod_32']:<8} "
            f"mult28={entry['is_mult_28']} mult32={entry['is_mult_32']}"
        )
    lines.append("")
    lines.append("### observed walls per bucket (x clamps = union over kinds)")
    for bucket, entry in sorted(report["observed_walls"].items(), key=lambda item: -item[1]["n"]):
        lines.append(
            f"  {bucket:<8} n={entry['n']:<7} y_wall={entry['y_wall']} "
            f"x_clamps={entry['x_clamps']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/probe_alignment.py",
        description="Test whether the Phantom-Data bbox x-clamps are explained by a "
        "per-side multiple-of-28/32 alignment rule (Qwen2.5-VL smart_resize).",
    )
    parser.add_argument("--out", required=True, help="path to write alignment_probe.json")
    parser.add_argument("--filtered-parquet", default=FILTERED_PARQUET)
    parser.add_argument("--meta-parquet", default=META_PARQUET)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1_000_000,
        help="table A has 651,031 rows; the default exhausts it",
    )
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--meta-batch-size", type=int, default=131072)
    parser.add_argument(
        "--candidate-min-frac",
        type=float,
        default=CANDIDATE_MIN_FRAC,
        help="minimum share of a cell for a distinct x2/y2 value to count as a candidate clamp",
    )
    parser.add_argument("--log-every", type=int, default=100000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    report = scan(
        filtered_parquet=args.filtered_parquet,
        meta_parquet=args.meta_parquet,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        meta_batch_size=args.meta_batch_size,
        candidate_min_frac=args.candidate_min_frac,
        log_every=args.log_every,
        log=log,
    )
    text = format_report(report)
    print(text, flush=True)
    report["text_summary"] = text
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
    log(f"[align] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
