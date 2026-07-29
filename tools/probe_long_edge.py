"""Probe: is the Phantom-Data bbox canvas a MIXTURE of isotropic long-edge canvases?

The marginal histograms from ``phantom_data.calib.estimate`` show x2 piling up at 768 /
798 / 800 / 832 while y2 piles up at 432 (the 768-long-edge 16:9 height). Two readings
fit those marginals equally well:

* **mixture of isotropic canvases** -- some boxes were measured on a long-edge-768
  canvas (16:9 -> 768x432), others on a long-edge-832 canvas (16:9 -> 832x468). Then
  ``x2 == 832`` boxes should carry y2 up to ~468, not ~432.
* **one anisotropic canvas** -- x and y use different scales; x2 and y2 clamp
  independently, so ``x2 == 832`` boxes cap at the same y2 == 432 as everyone else.

The marginals cannot tell these apart; the *conditional* can. This probe computes, per
box, the smallest isotropic canvas long edge that contains it
(:func:`phantom_data.calib.join.long_edge_needed`) and reports:

1. histogram of ``round(L_needed)`` (top values + pile/gap range buckets),
2. percentiles of ``L_needed``,
3. **the conditional**: for boxes with ``x2`` exactly on each candidate clamp, the
   max/p99/p90 of ``y2`` and of the y-implied long edge ``y2 * max(W,H)/H``,
4. per-source-video internal consistency: ``max(L_needed) - min(L_needed)`` over each
   video's boxes,
5. per candidate ``L``: containment fraction and exact edge-touching mass.

Streaming only (``iter_batches``); never materializes table A.

    PYTHONPATH=third_party/PhantomData/src python third_party/PhantomData/tools/probe_long_edge.py \
        --out /mnt/pfs/users/yuanze/datasets/phantom_canvas_calib_v1/long_edge_probe.json
"""
from __future__ import annotations

import argparse
import json
import math
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
    long_edge_needed,
)

#: Candidate isotropic long edges. 768/832 are the leading hypotheses; 798/800 are the
#: other x2 spikes; the rest bound the tail so containment does not saturate silently.
CANDIDATE_L = (768, 798, 800, 832, 864, 928, 1000, 1024)

#: x2 values to condition on (item 3). Exact integer match on ``round(x2)``.
CLAMP_X2 = (768, 798, 800, 832)

#: Ranges that make the pile-and-gap structure directly visible.
RANGE_BUCKETS = (
    ("...-767", None, 767),
    ("768", 768, 768),
    ("769-797", 769, 797),
    ("798", 798, 798),
    ("799-800", 799, 800),
    ("801-831", 801, 831),
    ("832", 832, 832),
    ("833-...", 833, None),
)

QUANTILES = (50, 75, 90, 95, 99, 99.9)


def percentiles(sorted_values: list[float], quantiles=QUANTILES) -> dict[str, float]:
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


def range_buckets(hist: dict[int, int]) -> dict[str, int]:
    """Collapse an integer histogram into :data:`RANGE_BUCKETS` counts."""
    out: dict[str, int] = {}
    for label, lo, hi in RANGE_BUCKETS:
        total = 0
        for value, count in hist.items():
            if lo is not None and value < lo:
                continue
            if hi is not None and value > hi:
                continue
            total += count
        out[label] = total
    return out


def _y_zone(y_long: float) -> str:
    """Which isotropic canvas the y axis alone demands: <=768, (768,832], or >832."""
    if y_long <= 768.0 + 1e-6:
        return "y_fits_768"
    if y_long <= 832.0 + 1e-6:
        return "y_needs_768to832"
    return "y_needs_over_832"


def y_implied_long_edge(y2: float, w: int, h: int) -> float:
    """Long edge implied by the y axis alone: ``y2 * max(w,h) / h``."""
    if w <= 0 or h <= 0:
        return 0.0
    return float(y2) * float(max(w, h)) / float(h)


class Cell:
    """Per-``(aspect_bucket, kind)`` accumulator."""

    def __init__(self) -> None:
        self.n = 0
        self.l_hist: Counter = Counter()
        self.l_values: list[float] = []
        # item 3: conditional on round(x2) == clamp
        self.cond_y2: dict[int, list[float]] = {clamp: [] for clamp in CLAMP_X2}
        self.cond_yl: dict[int, list[float]] = {clamp: [] for clamp in CLAMP_X2}
        # ...and the same for ALL boxes, as the baseline the conditionals must beat.
        self.all_y2: list[float] = []
        self.all_yl: list[float] = []
        # Where each clamp's y2 sits relative to the 768- and 832-long-edge fit heights.
        # Percentiles blur this; the three-way split does not.
        self.cond_yzone: dict[int, Counter] = {clamp: Counter() for clamp in CLAMP_X2}
        self.all_yzone: Counter = Counter()
        # Percentiles can hide whether the conditional y2 has a WALL (one value holding
        # visible mass) or just a smear, so keep the conditional histogram too.
        self.cond_y2_hist: dict[int, Counter] = {clamp: Counter() for clamp in CLAMP_X2}
        # Reverse conditional: boxes whose y axis alone overflows the 768 fit. If they
        # come from a larger isotropic canvas, their x2 must be scaled up too.
        self.over_y_x2: list[float] = []
        self.over_y_x2_hist: Counter = Counter()
        self.over_y_n = 0
        # item 5
        self.contained: Counter = Counter()  # L -> count(L_needed <= L + eps)
        self.touching: Counter = Counter()  # L -> count(round(L_needed) == L)
        self.resolutions: Counter = Counter()

    def add(self, box, src_w: int, src_h: int) -> None:
        need = long_edge_needed(box, src_w, src_h)
        self.n += 1
        self.l_values.append(need)
        rounded = int(round(need))
        self.l_hist[rounded] += 1
        self.resolutions[(src_w, src_h)] += 1

        x2, y2 = float(box[2]), float(box[3])
        y_long = y_implied_long_edge(y2, src_w, src_h)
        zone = _y_zone(y_long)
        self.all_y2.append(y2)
        self.all_yl.append(y_long)
        self.all_yzone[zone] += 1
        clamp = int(round(x2))
        if clamp in self.cond_y2:
            self.cond_y2[clamp].append(y2)
            self.cond_yl[clamp].append(y_long)
            self.cond_yzone[clamp][zone] += 1
            self.cond_y2_hist[clamp][int(round(y2))] += 1
        if y_long > 768.0 + 1e-6:
            self.over_y_n += 1
            self.over_y_x2.append(x2)
            self.over_y_x2_hist[int(round(x2))] += 1

        for candidate in CANDIDATE_L:
            # 1e-6 slack: L_needed is float arithmetic, a box exactly on the edge must
            # count as contained rather than losing to rounding dust.
            if need <= candidate + 1e-6:
                self.contained[candidate] += 1
            if rounded == candidate:
                self.touching[candidate] += 1

    def summarize(self) -> dict:
        sorted_l = sorted(self.l_values)
        def zone_fracs(counter: Counter, total: int) -> dict[str, float]:
            return {
                name: (counter.get(name, 0) / total) if total else 0.0
                for name in ("y_fits_768", "y_needs_768to832", "y_needs_over_832")
            }

        conditional = {
            "all": {
                "n": self.n,
                "y2": percentiles(sorted(self.all_y2), (90, 99, 99.9)),
                "y_implied_long_edge": percentiles(sorted(self.all_yl), (90, 99, 99.9)),
                "y_zone_frac": zone_fracs(self.all_yzone, self.n),
            }
        }
        for clamp in CLAMP_X2:
            y2s = sorted(self.cond_y2[clamp])
            yls = sorted(self.cond_yl[clamp])
            conditional[str(clamp)] = {
                "n": len(y2s),
                "y2": percentiles(y2s, (90, 99, 99.9)),
                "y_implied_long_edge": percentiles(yls, (90, 99, 99.9)),
                "y_zone_frac": zone_fracs(self.cond_yzone[clamp], len(y2s)),
                "y2_hist_top": dict(Counter(self.cond_y2_hist[clamp]).most_common(8)),
            }
        return {
            "n": self.n,
            "resolutions": {
                f"{w}x{h}": count for (w, h), count in self.resolutions.most_common(6)
            },
            "l_percentiles": percentiles(sorted_l),
            "l_hist_top": dict(Counter(self.l_hist).most_common(25)),
            "l_range_buckets": range_buckets(dict(self.l_hist)),
            "conditional_on_x2": conditional,
            "reverse_conditional_y_over_768": {
                "n": self.over_y_n,
                "frac_of_cell": (self.over_y_n / self.n) if self.n else 0.0,
                "x2_percentiles": percentiles(sorted(self.over_y_x2), (50, 90, 99)),
                "x2_hist_top": dict(Counter(self.over_y_x2_hist).most_common(12)),
            },
            "candidates": {
                str(candidate): {
                    "containment": (self.contained[candidate] / self.n) if self.n else 0.0,
                    "touching_count": self.touching[candidate],
                    "touching_frac": (self.touching[candidate] / self.n) if self.n else 0.0,
                }
                for candidate in CANDIDATE_L
            },
        }


def video_consistency(
    per_video: dict[str, list[float]], tolerance: float, candidates=CANDIDATE_L
) -> dict:
    """Item 4: how internally consistent is each source video's ``L_needed``?

    A video is "single-canvas" when its span (max - min of ``L_needed``) is <= the
    tolerance. Separately, classify every video by which candidate canvases its boxes
    need: a video whose boxes span two candidates (some need <= 768, some need > 768 but
    <= 832) is genuinely mixing signatures, which no per-video scale can explain.
    """
    spans: list[float] = []
    span_hist: Counter = Counter()
    signature_hist: Counter = Counter()
    straddle = 0
    consistent = 0
    for values in per_video.values():
        if len(values) < 2:
            continue
        span = max(values) - min(values)
        spans.append(span)
        span_hist[_span_bin(span)] += 1
        if span <= tolerance:
            consistent += 1
        # Smallest candidate that contains each box; >1 distinct = straddles candidates.
        signature = sorted({_smallest_candidate(value, candidates) for value in values})
        signature_hist["+".join(str(item) for item in signature)] += 1
        if len(signature) > 1:
            straddle += 1
    total = len(spans)
    return {
        "videos_with_2plus_boxes": total,
        "tolerance_px": tolerance,
        "consistent_within_tolerance": consistent,
        "consistent_frac": (consistent / total) if total else 0.0,
        "straddling_two_candidates": straddle,
        "straddling_frac": (straddle / total) if total else 0.0,
        "span_percentiles": percentiles(sorted(spans)),
        "span_bins": dict(sorted(span_hist.items(), key=lambda item: -item[1])),
        "signature_hist_top": dict(signature_hist.most_common(15)),
    }


def _span_bin(span: float) -> str:
    for edge in (0.5, 2.0, 8.0, 32.0, 64.0, 128.0):
        if span <= edge:
            return f"<={edge:g}"
    return ">128"


def _smallest_candidate(value: float, candidates=CANDIDATE_L) -> str:
    for candidate in candidates:
        if value <= candidate + 1e-6:
            return str(candidate)
    return f">{candidates[-1]}"


def scan(
    filtered_parquet: str,
    meta_parquet: str,
    max_rows: int,
    batch_size: int,
    meta_batch_size: int,
    tolerance: float,
    log_every: int,
    log=print,
) -> dict:
    started = time.time()
    log(f"[probe] building vid->(W,H) from {meta_parquet}")
    vid_wh = build_vid_wh(meta_parquet, batch_size=meta_batch_size)
    log(f"[probe] vid_wh entries={len(vid_wh)} ({time.time() - started:.1f}s)")

    stats = JoinStats()
    cells: dict[tuple[str, str], Cell] = defaultdict(Cell)
    # Per-video L_needed lists, target boxes only: a ref box comes from a different clip
    # (and is keyed by its own vid), so mixing kinds would not test the same thing.
    per_video: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    next_log = log_every

    for kind, box, src_w, src_h, vid, _phrase, _cls in iter_boxes(
        filtered_parquet, vid_wh, batch_size=batch_size, max_rows=max_rows, stats=stats
    ):
        bucket = aspect_bucket(src_w, src_h)
        cells[(bucket, kind)].add(box, src_w, src_h)
        per_video[kind][vid].append(long_edge_needed(box, src_w, src_h))
        if log_every and stats.rows_read >= next_log:
            next_log = stats.rows_read + log_every
            log(
                f"[probe] rows={stats.rows_read} boxes={stats.boxes_emitted} "
                f"elapsed={time.time() - started:.0f}s"
            )

    log(
        f"[probe] scan done rows={stats.rows_read} boxes={stats.boxes_emitted} "
        f"elapsed={time.time() - started:.0f}s"
    )

    return {
        "params": {
            "filtered_parquet": filtered_parquet,
            "meta_parquet": meta_parquet,
            "max_rows": max_rows,
            "batch_size": batch_size,
            "candidate_long_edges": list(CANDIDATE_L),
            "clamp_x2_values": list(CLAMP_X2),
            "tolerance_px": tolerance,
            "l_needed_rule": "L_needed = max(x2*max(W,H)/W, y2*max(W,H)/H)",
        },
        "global": {
            "rows_read": stats.rows_read,
            "boxes": stats.boxes_emitted,
            "counters": stats.as_dict(),
            "vid_wh_entries": len(vid_wh),
            "wall_clock_sec": round(time.time() - started, 1),
        },
        "cells": {
            f"{bucket}|{kind}": cell.summarize()
            for (bucket, kind), cell in sorted(cells.items())
        },
        "video_consistency": {
            kind: video_consistency(videos, tolerance)
            for kind, videos in sorted(per_video.items())
        },
    }


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------


def format_report(report: dict) -> str:
    lines: list[str] = []
    glob = report["global"]
    lines.append("=" * 108)
    lines.append("LONG-EDGE PROBE -- mixture of isotropic canvases vs one anisotropic canvas")
    lines.append("=" * 108)
    lines.append(f"rule           : {report['params']['l_needed_rule']}")
    lines.append(f"rows read      : {glob['rows_read']}")
    lines.append(f"boxes analyzed : {glob['boxes']}")
    lines.append(f"counters       : {glob['counters']}")
    lines.append(f"wall clock     : {glob['wall_clock_sec']}s")
    lines.append("")

    lines.append("### ITEM 3 (DECISIVE): y2 and y-implied long edge, CONDITIONED on round(x2)")
    lines.append(
        "If a long-edge-832 canvas exists, x2==832 boxes carry y2 up to ~468 (16:9) and "
        "y-implied L up to ~832."
    )
    lines.append(
        "If x/y clamp independently (anisotropic), x2==832 boxes cap at the SAME y2 as "
        "everybody else."
    )
    header = (
        f"{'bucket|kind':<26}{'x2==':>7}{'n':>8}"
        f"{'y2 p99':>9}{'y2 p99.9':>10}{'y2 max':>9}"
        f"{'yL p99':>9}{'yL p99.9':>10}{'yL max':>9}"
        f"{'y<=768':>9}{'y769-832':>10}{'y>832':>8}"
    )
    lines.append("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    for key, cell in report["cells"].items():
        # Cells with almost no boxes cannot support a conditional; skip the clutter.
        if cell["n"] < 50:
            continue
        first = True
        for clamp, entry in cell["conditional_on_x2"].items():
            label = f"{key} n={cell['n']}" if first else ""
            first = False
            if not entry["n"]:
                lines.append(f"{label:<26}{clamp:>7}{0:>8}   (no box on this clamp)")
                continue
            y2 = entry["y2"]
            yl = entry["y_implied_long_edge"]
            zone = entry["y_zone_frac"]
            lines.append(
                f"{label:<26}{clamp:>7}{entry['n']:>8}"
                f"{y2['p99']:>9.1f}{y2['p99.9']:>10.1f}{y2['max']:>9.1f}"
                f"{yl['p99']:>9.1f}{yl['p99.9']:>10.1f}{yl['max']:>9.1f}"
                f"{zone['y_fits_768']:>9.4f}{zone['y_needs_768to832']:>10.4f}"
                f"{zone['y_needs_over_832']:>8.4f}"
            )
    lines.append("-" * len(header))
    lines.append("")

    lines.append("### ITEM 3a: conditional y2 histogram (is there a WALL or just a smear?)")
    for key, cell in report["cells"].items():
        if cell["n"] < 1000:
            continue
        for clamp, entry in cell["conditional_on_x2"].items():
            if clamp == "all" or not entry["n"]:
                continue
            lines.append(
                f"  {key:<16} x2=={clamp:<5} n={entry['n']:<7} "
                f"top y2: {entry['y2_hist_top']}"
            )
    lines.append("")

    lines.append("### ITEM 3b (REVERSE): boxes whose y axis ALONE overflows the 768 fit")
    lines.append(
        "Under the mixture reading these are the large-canvas boxes, so their x2 should "
        "sit at/near the large clamp."
    )
    for key, cell in report["cells"].items():
        if cell["n"] < 50:
            continue
        entry = cell["reverse_conditional_y_over_768"]
        if not entry["n"]:
            lines.append(f"  {key:<22} none")
            continue
        pcts = entry["x2_percentiles"]
        lines.append(
            f"  {key:<22} n={entry['n']} ({entry['frac_of_cell']:.4f} of cell)  "
            f"x2 p50={pcts['p50']:.1f} p90={pcts['p90']:.1f} p99={pcts['p99']:.1f} "
            f"max={pcts['max']:.1f}"
        )
        lines.append(f"  {'':<22} top x2: {entry['x2_hist_top']}")
    lines.append("")

    lines.append("### ITEM 1+2: L_needed histogram / range buckets / percentiles")
    for key, cell in report["cells"].items():
        if cell["n"] < 50:
            continue
        pcts = cell["l_percentiles"]
        lines.append(f"  {key}  n={cell['n']}  resolutions={cell['resolutions']}")
        lines.append(
            "    percentiles: "
            + "  ".join(f"{name}={value:.1f}" for name, value in pcts.items())
        )
        top = ", ".join(
            f"{value}:{count}" for value, count in list(cell["l_hist_top"].items())[:25]
        )
        lines.append(f"    top round(L_needed): {top}")
        lines.append(
            "    ranges: "
            + "  ".join(f"{label}={count}" for label, count in cell["l_range_buckets"].items())
        )
    lines.append("")

    lines.append("### ITEM 5: containment and edge-touching mass per candidate L")
    header5 = f"{'bucket|kind':<20}{'L':>7}{'containment':>13}{'touch n':>10}{'touch frac':>12}"
    lines.append("-" * len(header5))
    lines.append(header5)
    lines.append("-" * len(header5))
    for key, cell in report["cells"].items():
        if cell["n"] < 50:
            continue
        first = True
        for candidate, entry in cell["candidates"].items():
            label = key if first else ""
            first = False
            lines.append(
                f"{label:<20}{candidate:>7}{entry['containment']:>13.5f}"
                f"{entry['touching_count']:>10}{entry['touching_frac']:>12.5f}"
            )
    lines.append("-" * len(header5))
    lines.append("")

    lines.append("### ITEM 4: per-source-video internal consistency of L_needed")
    for kind, entry in report["video_consistency"].items():
        lines.append(
            f"  kind={kind}  videos(>=2 boxes)={entry['videos_with_2plus_boxes']}  "
            f"consistent within {entry['tolerance_px']}px="
            f"{entry['consistent_within_tolerance']} ({entry['consistent_frac']:.3f})  "
            f"straddling 2+ candidates={entry['straddling_two_candidates']} "
            f"({entry['straddling_frac']:.3f})"
        )
        lines.append(
            "    span percentiles: "
            + "  ".join(f"{name}={value:.1f}" for name, value in entry["span_percentiles"].items())
        )
        lines.append(f"    span bins: {entry['span_bins']}")
        lines.append(f"    candidate signatures: {entry['signature_hist_top']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/probe_long_edge.py",
        description="Test whether Phantom-Data bboxes come from a mixture of isotropic "
        "long-edge canvases or from one anisotropic canvas.",
    )
    parser.add_argument("--out", required=True, help="path to write long_edge_probe.json")
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
        "--tolerance",
        type=float,
        default=2.0,
        help="px span within which a video counts as single-canvas",
    )
    parser.add_argument("--log-every", type=int, default=50000)
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
        tolerance=args.tolerance,
        log_every=args.log_every,
        log=log,
    )
    text = format_report(report)
    print(text, flush=True)
    report["text_summary"] = text
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
    log(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
