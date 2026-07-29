"""Estimate the canvas (annotation image size) behind the Phantom-Data bboxes.

Background: existing code assumes the boxes were measured on a canvas whose long
edge is 768 (``scale = max(W,H)/768``). Measurement falsified that: the y axis obeys
the 768 fit exactly (1:1 sources cap at exactly y2=768, 2.22:1 sources at exactly
345), but x overshoots -- 16:9 sources produce x2 up to 928 against a 768-wide
canvas, with hard spikes at 768 / 798 / 800 / 832. A single canvas produces one
edge, not four, so the annotations were probably produced under several resolution
buckets (video-generation buckets like 832x480 are the prime suspect). And the
mixture is per-box, not per-shard: 93 of 983 multi-box source videos contain both an
``x2 > 768`` box and a separate ``y2 == 432`` box. Table A has no provenance column
to split on.

What this module decides:

* If the discovered ``(x2 spike, y2 spike)`` pairs come out cleanly paired, the
  convention is a small set of fixed buckets and a per-box classifier is viable.
* If the pairs mix freely, this annotation set cannot be trusted for pixel-accurate
  boxes and has to be discounted.

Run it (streaming, ~1-2 CPU, no writes except ``--out``)::

    python -m phantom_data.calib.estimate --out outputs/calib/canvas_estimator.json

Sampling is rare-bucket driven, not row driven: the common 16:9 bucket saturates
within a few thousand rows, so the scan stops as soon as each of 1:1, 4:3 and
>=2.2:1 has ``--rare-target`` boxes (or ``--max-rows`` rows have been read). The
achieved count and an explicit ``sufficient`` flag are recorded per bucket -- a
spike histogram over 46 boxes is not evidence and is not presented as one.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict

from phantom_data.calib.join import (
    FILTERED_PARQUET,
    META_PARQUET,
    RARE_BUCKETS,
    JoinStats,
    aspect_bucket,
    aspect_ratio,
    build_vid_wh,
    iter_boxes,
)

# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def rare_buckets_saturated(
    bucket_kind_counts: dict[str, dict[str, int]],
    rare_target: int,
    rare_buckets: tuple[str, ...] = RARE_BUCKETS,
    kinds: tuple[str, ...] = ("target", "ref"),
) -> bool:
    """True when every rare bucket has ``rare_target`` boxes *of each kind*.

    Factored out as a pure predicate so the early-stop decision is testable without
    touching parquet.

    The gate is per kind, not per bucket, because the analysis cell is
    ``(bucket, kind)``: target and ref boxes are summarized separately, since whether
    the two agree is itself a finding. Gating on the combined count would stop the
    scan exactly when each cell holds only half the target, so every rare cell would
    be flagged insufficient the moment the scan declared itself saturated.
    """
    if rare_target <= 0:
        return True
    for name in rare_buckets:
        counts = bucket_kind_counts.get(name) or {}
        for kind in kinds:
            if counts.get(kind, 0) < rare_target:
                return False
    return True


def fit_canvas(w: int, h: int, long_edge: float) -> tuple[float, float]:
    """Scale ``(w, h)`` so its long edge equals ``long_edge``. No rounding."""
    if w <= 0 or h <= 0:
        return (0.0, 0.0)
    scale = long_edge / float(max(w, h))
    return (w * scale, h * scale)


def find_spikes(
    hist: dict[int, int],
    window: int = 4,
    factor: float = 5.0,
    top_decile_only: bool = True,
    min_distinct: int = 20,
    isolated_min_share: float = 0.005,
) -> list[dict]:
    """Locate canvas-edge candidates in an integer histogram.

    A value ``v`` is a candidate when ``count(v) > factor * baseline(v)`` and ``v``
    lies in the top decile of the axis' observed range. That is what converts "max x2
    = 928" into "clamps at 768 / 798 / 800 / 832": a canvas edge shows up as a
    pile-up of clipped boxes sitting far above its surroundings.

    ``baseline(v)`` is the median of ``count(v-window .. v+window)``, but **restricted
    to the axis' observed range**. That restriction matters: the most important clamp
    is often the axis maximum itself (1:1 sources cap at exactly y2=768), and for the
    maximum, half of a symmetric window is structurally empty. Including those
    phantom zeros drags the median to 0 and would discard the sharpest evidence in
    the dataset. So the maximum is judged against its lower side only.

    Two fallbacks keep genuinely isolated clamps detectable without dividing by zero:

    * neighbourhood median 0 -> fall back to the mean density over the whole top
      decile (``rule="top_decile_mean"``);
    * that mean also 0, i.e. the value stands completely alone -> call it a spike only
      if it carries real mass (``>= max(2, isolated_min_share * total)``,
      ``rule="isolated"``). A lone value with a couple of hits is tail noise; a lone
      value holding 1% of the axis is a hard clamp.

    Fewer than ``min_distinct`` distinct values returns ``[]`` -- not enough shape to
    call a spike at all.

    Returns spikes ordered by count descending, each
    ``{"value", "count", "baseline", "ratio", "rule"}``.
    """
    if not hist or len(hist) < min_distinct:
        return []
    values = sorted(hist)
    lo, hi = values[0], values[-1]
    span = hi - lo
    threshold = lo + 0.9 * span if top_decile_only else lo
    total = sum(hist.values())
    decile_lo = int(math.floor(threshold))

    spikes: list[dict] = []
    for value in values:
        if value < threshold:
            continue
        count = hist[value]
        neighbours = [
            hist.get(neighbour, 0)
            for neighbour in range(value - window, value + window + 1)
            if neighbour != value and lo <= neighbour <= hi
        ]
        baseline = statistics.median(neighbours) if neighbours else 0.0
        rule = "neighbor_median"
        if baseline <= 0:
            region = [
                hist.get(other, 0)
                for other in range(decile_lo, hi + 1)
                if other != value
            ]
            baseline = (sum(region) / len(region)) if region else 0.0
            rule = "top_decile_mean"
        if baseline <= 0:
            if count >= max(2.0, isolated_min_share * total):
                spikes.append(
                    {
                        "value": int(value),
                        "count": int(count),
                        "baseline": 0.0,
                        # Not float('inf'): json.dump would emit bare `Infinity`,
                        # which is invalid JSON and breaks strict readers.
                        "ratio": None,
                        "rule": "isolated",
                    }
                )
            continue
        if count > factor * baseline:
            spikes.append(
                {
                    "value": int(value),
                    "count": int(count),
                    "baseline": float(baseline),
                    "ratio": float(count) / float(baseline),
                    "rule": rule,
                }
            )
    spikes.sort(key=lambda spike: (-spike["count"], spike["value"]))
    return spikes


def percentiles(sorted_values: list[float], quantiles=(50, 90, 99, 99.9)) -> dict[str, float]:
    """Nearest-rank percentiles plus max, from an already-sorted list."""
    if not sorted_values:
        return {}
    out: dict[str, float] = {}
    n = len(sorted_values)
    for q in quantiles:
        rank = min(n - 1, max(0, int(math.ceil(q / 100.0 * n)) - 1))
        out[f"p{q:g}"] = float(sorted_values[rank])
    out["max"] = float(sorted_values[-1])
    return out


def canvas_score(
    boxes_xy: list[tuple[float, float]], canvas_w: float, canvas_h: float
) -> dict[str, float]:
    """Containment and tightness of ``(x2, y2)`` pairs against one candidate canvas.

    * ``containment`` -- fraction with ``x2 <= Xc and y2 <= Yc``.
    * ``tightness`` -- ``mean(max(x2/Xc, y2/Yc))``. A correctly sized canvas has many
      boxes touching an edge, so tightness sits near 1.0; an oversized canvas
      contains everything at tightness far below 1. Containment alone would pick the
      largest candidate, so both numbers are reported and both matter.
    """
    if not boxes_xy or canvas_w <= 0 or canvas_h <= 0:
        return {"containment": 0.0, "tightness": 0.0, "n": 0}
    contained = 0
    total = 0.0
    for x2, y2 in boxes_xy:
        if x2 <= canvas_w and y2 <= canvas_h:
            contained += 1
        total += max(x2 / canvas_w, y2 / canvas_h)
    n = len(boxes_xy)
    return {
        "containment": contained / n,
        "tightness": total / n,
        "n": n,
    }


def crosstab_is_clean(crosstab: dict[str, int], min_share: float = 0.9) -> tuple[bool, float]:
    """Judge whether ``(x_spike, y_spike)`` pairing is functional.

    ``crosstab`` maps ``"<x>x<y>"`` to a count. Clean means: for every x spike, one y
    spike accounts for at least ``min_share`` of that x's mass (and vice versa). The
    returned float is the worst per-spike dominant share, so the caller can report
    how close to clean a "dirty" result actually is.
    """
    if not crosstab:
        return False, 0.0
    by_x: dict[int, Counter] = defaultdict(Counter)
    by_y: dict[int, Counter] = defaultdict(Counter)
    for key, count in crosstab.items():
        x_str, _, y_str = key.partition("x")
        try:
            x, y = int(x_str), int(y_str)
        except ValueError:
            continue
        by_x[x][y] += count
        by_y[y][x] += count
    if not by_x:
        return False, 0.0
    worst = 1.0
    for counter in list(by_x.values()) + list(by_y.values()):
        total = sum(counter.values())
        if total <= 0:
            continue
        dominant = counter.most_common(1)[0][1]
        worst = min(worst, dominant / total)
    return worst >= min_share, worst


# ---------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------


class CellAccumulator:
    """Per-``(aspect_bucket, kind)`` accumulator. Keeps only what the report needs."""

    def __init__(self) -> None:
        self.n = 0
        self.x2_hist: Counter = Counter()
        self.y2_hist: Counter = Counter()
        self.x1_min = math.inf
        self.y1_min = math.inf
        self.x1_neg = 0
        self.y1_neg = 0
        self.x2_values: list[float] = []
        self.y2_values: list[float] = []
        # (x2, y2) kept for canvas scoring and the cross-tab; bounded by the sample.
        self.pairs: list[tuple[float, float]] = []
        self.resolutions: Counter = Counter()
        # 768-long-edge fit overflow, computed per box against that box's own W/H
        # (sources inside one aspect bucket can differ in absolute size).
        self.over_x_768 = 0
        self.over_y_768 = 0

    def add(self, box: list[float], src_w: int, src_h: int) -> None:
        x1, y1, x2, y2 = box
        self.n += 1
        self.x2_hist[int(round(x2))] += 1
        self.y2_hist[int(round(y2))] += 1
        self.x1_min = min(self.x1_min, x1)
        self.y1_min = min(self.y1_min, y1)
        if x1 < 0:
            self.x1_neg += 1
        if y1 < 0:
            self.y1_neg += 1
        self.x2_values.append(x2)
        self.y2_values.append(y2)
        self.pairs.append((x2, y2))
        self.resolutions[(src_w, src_h)] += 1
        fit_w, fit_h = fit_canvas(src_w, src_h, 768.0)
        if fit_w and x2 > fit_w:
            self.over_x_768 += 1
        if fit_h and y2 > fit_h:
            self.over_y_768 += 1


def candidate_canvases(
    resolutions: Counter,
    x_spikes: list[int],
    y_spikes: list[int],
) -> list[tuple[str, float, float]]:
    """Enumerate ``(label, Xc, Yc)`` canvases worth scoring for one cell.

    Covers the hypotheses on the table: the 768- and 1024-long-edge fits for the
    bucket's dominant resolution, the 832x480 video-generation bucket, 832 paired
    with the 768-fit height (the "x was re-bucketed, y wasn't" hypothesis), a
    1000x1000-scaled fit, and every discovered ``(x spike, y spike)`` pair.
    """
    out: list[tuple[str, float, float]] = []
    seen: set[tuple[int, int]] = set()

    def push(label: str, cw: float, ch: float) -> None:
        if cw <= 0 or ch <= 0:
            return
        key = (int(round(cw)), int(round(ch)))
        if key in seen:
            return
        seen.add(key)
        out.append((label, float(cw), float(ch)))

    dominant = resolutions.most_common(1)[0][0] if resolutions else (0, 0)
    src_w, src_h = dominant
    for long_edge in (768.0, 1024.0):
        cw, ch = fit_canvas(src_w, src_h, long_edge)
        push(f"fit{int(long_edge)}({src_w}x{src_h})", cw, ch)
    push("832x480", 832, 480)
    fit768_w, fit768_h = fit_canvas(src_w, src_h, 768.0)
    push("832x fit768-h", 832, fit768_h)
    cw, ch = fit_canvas(src_w, src_h, 1000.0)
    push(f"fit1000({src_w}x{src_h})", cw, ch)
    for x in x_spikes:
        for y in y_spikes:
            push(f"spike {x}x{y}", x, y)
    return out


def significant_spikes(spikes: list[dict], n: int, min_share: float = 0.005, top_k: int = 8) -> list[int]:
    """Spike values large enough to be load-bearing for the cross-tab.

    The spike rule is relative to a local neighbourhood, so in a sparse tail a
    handful of boxes can clear it. Those are fine to report, but they must not drive
    the pairing verdict: one count-1 pair would otherwise flip a clean result to
    "mixes". Keep spikes holding at least ``min_share`` of the cell's boxes, capped
    at ``top_k`` by count.
    """
    floor = max(2.0, min_share * n)
    return [spike["value"] for spike in spikes if spike["count"] >= floor][:top_k]


def summarize_cell(cell: CellAccumulator, rare_target: int, is_rare: bool) -> dict:
    """Turn one accumulator into the JSON-serializable per-cell report."""
    x_spikes = find_spikes(dict(cell.x2_hist))
    y_spikes = find_spikes(dict(cell.y2_hist))
    x_spike_values = significant_spikes(x_spikes, cell.n)
    y_spike_values = significant_spikes(y_spikes, cell.n)

    x_spike_set = set(x_spike_values)
    y_spike_set = set(y_spike_values)
    crosstab: Counter = Counter()
    for x2, y2 in cell.pairs:
        xi, yi = int(round(x2)), int(round(y2))
        if xi in x_spike_set and yi in y_spike_set:
            crosstab[f"{xi}x{yi}"] += 1
    clean, worst_share = crosstab_is_clean(dict(crosstab))

    scored = []
    for label, cw, ch in candidate_canvases(cell.resolutions, x_spike_values, y_spike_values):
        score = canvas_score(cell.pairs, cw, ch)
        scored.append(
            {
                "label": label,
                "canvas": [round(cw, 3), round(ch, 3)],
                "containment": round(score["containment"], 6),
                "tightness": round(score["tightness"], 6),
            }
        )
    # Winner: containment first, but bucketed to 3 decimals so a 0.001 edge does not
    # outrank a canvas that is dramatically tighter. Within a containment tier, the
    # tightness closest to 1.0 wins -- that is what separates "correct size" from
    # "big enough to contain anything".
    scored.sort(
        key=lambda item: (-round(item["containment"], 3), abs(1.0 - item["tightness"]))
    )
    best = scored[0] if scored else None

    return {
        "n": cell.n,
        "sufficient": (cell.n >= rare_target) if is_rare else True,
        "rare_bucket": is_rare,
        "resolutions": {f"{w}x{h}": count for (w, h), count in cell.resolutions.most_common()},
        "x1_min": None if cell.x1_min is math.inf else cell.x1_min,
        "y1_min": None if cell.y1_min is math.inf else cell.y1_min,
        "x1_negative": cell.x1_neg,
        "y1_negative": cell.y1_neg,
        "x2_percentiles": percentiles(sorted(cell.x2_values)),
        "y2_percentiles": percentiles(sorted(cell.y2_values)),
        "x2_spikes": x_spikes[:12],
        "y2_spikes": y_spikes[:12],
        "spike_crosstab": dict(crosstab.most_common()),
        "crosstab_clean": clean,
        "crosstab_worst_dominant_share": round(worst_share, 6),
        "frac_x2_over_fit768": (cell.over_x_768 / cell.n) if cell.n else 0.0,
        "frac_y2_over_fit768": (cell.over_y_768 / cell.n) if cell.n else 0.0,
        "canvas_candidates": scored,
        "best_canvas": best,
        # Full integer histograms are large; keep the top of each axis only, which is
        # where canvas edges live. The spike list already carries the exact counts.
        "x2_hist_top": dict(Counter(cell.x2_hist).most_common(40)),
        "y2_hist_top": dict(Counter(cell.y2_hist).most_common(40)),
    }


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def scan(
    filtered_parquet: str,
    meta_parquet: str,
    max_rows: int,
    rare_target: int,
    batch_size: int,
    meta_batch_size: int,
    log_every: int,
    log=print,
) -> dict:
    started = time.time()
    log(f"[estimate] building vid->(W,H) from {meta_parquet}")
    vid_wh = build_vid_wh(meta_parquet, batch_size=meta_batch_size)
    log(f"[estimate] vid_wh entries={len(vid_wh)} ({time.time() - started:.1f}s)")

    stats = JoinStats()
    cells: dict[tuple[str, str], CellAccumulator] = defaultdict(CellAccumulator)
    bucket_counts: Counter = Counter()
    bucket_kind_counts: dict[str, Counter] = defaultdict(Counter)
    ratio_examples: dict[str, Counter] = defaultdict(Counter)
    stop_reason = "exhausted"
    next_log = log_every

    for kind, box, src_w, src_h, _vid, _phrase, _cls in iter_boxes(
        filtered_parquet, vid_wh, batch_size=batch_size, max_rows=max_rows, stats=stats
    ):
        bucket = aspect_bucket(src_w, src_h)
        cells[(bucket, kind)].add(box, src_w, src_h)
        bucket_counts[bucket] += 1
        bucket_kind_counts[bucket][kind] += 1
        ratio_examples[bucket][(src_w, src_h, round(aspect_ratio(src_w, src_h), 4))] += 1

        if log_every and stats.rows_read >= next_log:
            next_log = stats.rows_read + log_every
            log(
                f"[estimate] rows={stats.rows_read} boxes={stats.boxes_emitted} "
                + " ".join(
                    f"{name}=t{bucket_kind_counts[name].get('target', 0)}"
                    f"/r{bucket_kind_counts[name].get('ref', 0)}"
                    for name in RARE_BUCKETS
                )
                + f" elapsed={time.time() - started:.0f}s"
            )
        if rare_buckets_saturated(bucket_kind_counts, rare_target):
            stop_reason = "rare_buckets_saturated"
            break
    else:
        if max_rows is not None and stats.rows_read >= max_rows:
            stop_reason = "max_rows"

    if stop_reason == "exhausted" and max_rows is not None and stats.rows_read >= max_rows:
        stop_reason = "max_rows"

    log(
        f"[estimate] scan done rows={stats.rows_read} boxes={stats.boxes_emitted} "
        f"reason={stop_reason} elapsed={time.time() - started:.0f}s"
    )

    report_cells: dict[str, dict] = {}
    for (bucket, kind), cell in sorted(cells.items()):
        is_rare = bucket in RARE_BUCKETS
        report_cells[f"{bucket}|{kind}"] = summarize_cell(cell, rare_target, is_rare)

    return {
        "params": {
            "filtered_parquet": filtered_parquet,
            "meta_parquet": meta_parquet,
            "max_rows": max_rows,
            "rare_target": rare_target,
            "batch_size": batch_size,
            "meta_batch_size": meta_batch_size,
            "rare_buckets": list(RARE_BUCKETS),
            "spike_rule": "count(v) > 5 * median(count(v-4..v+4)) and v in top decile",
        },
        "global": {
            "rows_read": stats.rows_read,
            "boxes": stats.boxes_emitted,
            "stop_reason": stop_reason,
            "counters": stats.as_dict(),
            "vid_wh_entries": len(vid_wh),
            "bucket_box_counts": dict(bucket_counts.most_common()),
            "bucket_kind_counts": {
                bucket: dict(counter) for bucket, counter in sorted(bucket_kind_counts.items())
            },
            "bucket_resolutions": {
                bucket: {f"{w}x{h}@{ratio}": count for (w, h, ratio), count in counter.most_common()}
                for bucket, counter in sorted(ratio_examples.items())
            },
            # Sufficiency is per (bucket, kind) -- the analysis cell -- so a bucket
            # only counts as sufficient when BOTH its target and ref cells cleared
            # the target.
            "rare_bucket_sufficient": {
                name: min(
                    bucket_kind_counts[name].get("target", 0),
                    bucket_kind_counts[name].get("ref", 0),
                )
                >= rare_target
                for name in RARE_BUCKETS
            },
            "wall_clock_sec": round(time.time() - started, 1),
        },
        "cells": report_cells,
    }


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------


def format_report(report: dict) -> str:
    lines: list[str] = []
    glob = report["global"]
    params = report["params"]
    lines.append("=" * 100)
    lines.append("CANVAS ESTIMATOR -- Phantom-Data bbox coordinate system")
    lines.append("=" * 100)
    lines.append(
        f"rows read      : {glob['rows_read']}  (stop: {glob['stop_reason']}, "
        f"max_rows={params['max_rows']})"
    )
    lines.append(f"boxes analyzed : {glob['boxes']}")
    lines.append(f"rare target    : {params['rare_target']} boxes per rare bucket")
    lines.append(f"counters       : {glob['counters']}")
    lines.append(f"wall clock     : {glob['wall_clock_sec']}s")
    lines.append("")
    lines.append("boxes per aspect bucket (target + ref):")
    for bucket, count in glob["bucket_box_counts"].items():
        flag = ""
        if bucket in params["rare_buckets"]:
            flag = "  OK" if glob["rare_bucket_sufficient"].get(bucket) else "  *** UNDER TARGET ***"
        lines.append(f"  {bucket:<14} {count:>8}{flag}")
    lines.append("")
    lines.append("raw resolutions per bucket:")
    for bucket, resolutions in glob["bucket_resolutions"].items():
        top = list(resolutions.items())[:6]
        rendered = ", ".join(f"{key}:{count}" for key, count in top)
        more = "" if len(resolutions) <= 6 else f" (+{len(resolutions) - 6} more)"
        lines.append(f"  {bucket:<14} {rendered}{more}")
    lines.append("")

    header = (
        f"{'bucket|kind':<22}{'n':>7}{'suff':>6}  {'top-3 x2 spikes':<28}"
        f"{'top-3 y2 spikes':<28}{'best canvas':<26}{'cont':>7}{'tight':>7}"
    )
    lines.append("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    for key, cell in report["cells"].items():
        x_spikes = ",".join(f"{s['value']}({s['count']})" for s in cell["x2_spikes"][:3]) or "-"
        y_spikes = ",".join(f"{s['value']}({s['count']})" for s in cell["y2_spikes"][:3]) or "-"
        best = cell["best_canvas"]
        best_text = "-" if best is None else f"{best['label']} {best['canvas']}"
        cont = "-" if best is None else f"{best['containment']:.3f}"
        tight = "-" if best is None else f"{best['tightness']:.3f}"
        lines.append(
            f"{key:<22}{cell['n']:>7}{('yes' if cell['sufficient'] else 'NO'):>6}  "
            f"{x_spikes:<28}{y_spikes:<28}{best_text:<26}{cont:>7}{tight:>7}"
        )
    lines.append("-" * len(header))
    lines.append("")

    lines.append("768-long-edge fit overflow (compare against the falsifying table):")
    for key, cell in report["cells"].items():
        lines.append(
            f"  {key:<22} frac(x2>Wc)={cell['frac_x2_over_fit768']:.4f}  "
            f"frac(y2>Hc)={cell['frac_y2_over_fit768']:.4f}  "
            f"x2 max={cell['x2_percentiles'].get('max', 0):.0f}  "
            f"y2 max={cell['y2_percentiles'].get('max', 0):.0f}  "
            f"x1 min={cell['x1_min']}  y1 min={cell['y1_min']}"
        )
    lines.append("")

    lines.append("(x2 spike, y2 spike) cross-tab -- the deciding artifact:")
    for key, cell in report["cells"].items():
        crosstab = cell["spike_crosstab"]
        if not crosstab:
            lines.append(f"  {key:<22} no box hit a spike on both axes -> undecidable here")
            continue
        top = ", ".join(f"{pair}:{count}" for pair, count in list(crosstab.items())[:8])
        verdict = (
            "PAIRS CLEANLY -> fixed buckets, per-box classifier viable"
            if cell["crosstab_clean"]
            else "MIXES -> no functional (x,y) pairing; discount this annotation set"
        )
        note = "" if cell["sufficient"] else "  [sample under target: weak evidence]"
        lines.append(f"  {key:<22} {top}")
        lines.append(
            f"  {'':<22} worst dominant share={cell['crosstab_worst_dominant_share']:.3f} "
            f"-> {verdict}{note}"
        )
    lines.append("")

    decisive = [
        (key, cell)
        for key, cell in report["cells"].items()
        if cell["spike_crosstab"] and cell["sufficient"]
    ]
    if not decisive:
        lines.append(
            "OVERALL: no cell reached both a sufficient sample and a two-axis spike hit. "
            "Verdict deferred -- raise --rare-target / --max-rows before concluding."
        )
    else:
        clean = [key for key, cell in decisive if cell["crosstab_clean"]]
        lines.append(
            f"OVERALL: {len(clean)}/{len(decisive)} sufficiently-sampled cells pair cleanly "
            f"({', '.join(clean) if clean else 'none'})."
        )
        if len(clean) == len(decisive):
            lines.append(
                "  => Annotation canvas is a small fixed set of buckets. A per-box classifier "
                "keyed on (x2, y2) clamp values is viable."
            )
        elif clean:
            lines.append(
                "  => Mixed: some cells pair, others do not. Per-box classification is only "
                "safe inside the clean cells; treat the rest as unusable."
            )
        else:
            lines.append(
                "  => Spikes mix freely. There is no recoverable per-box canvas; this "
                "annotation set should be discounted for pixel-accurate boxes."
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m phantom_data.calib.estimate",
        description="Estimate the annotation canvas behind the Phantom-Data bboxes.",
    )
    parser.add_argument("--out", required=True, help="path to write canvas_estimator.json")
    parser.add_argument("--filtered-parquet", default=FILTERED_PARQUET)
    parser.add_argument("--meta-parquet", default=META_PARQUET)
    parser.add_argument("--max-rows", type=int, default=80000)
    parser.add_argument(
        "--rare-target",
        type=int,
        default=300,
        help="stop once each of 1:1 / 4:3 / >=2.2:1 has this many boxes",
    )
    parser.add_argument("--batch-size", type=int, default=2000, help="table A iter_batches size")
    parser.add_argument("--meta-batch-size", type=int, default=131072)
    parser.add_argument("--log-every", type=int, default=5000, help="rows between progress lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    report = scan(
        filtered_parquet=args.filtered_parquet,
        meta_parquet=args.meta_parquet,
        max_rows=args.max_rows,
        rare_target=args.rare_target,
        batch_size=args.batch_size,
        meta_batch_size=args.meta_batch_size,
        log_every=args.log_every,
        log=log,
    )
    text = format_report(report)
    print(text, flush=True)
    report["text_summary"] = text
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
    log(f"[estimate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
