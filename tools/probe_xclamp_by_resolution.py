"""Probe: is the Phantom-Data x-clamp a property of the SOURCE RESOLUTION?

Established by the two earlier scans (``phantom_data.calib.estimate`` and
``tools/probe_long_edge.py``):

* the y axis obeys one isotropic "long edge = 768" canvas in every aspect bucket
  (16:9 -> 432, 4:3 -> 576, 1:1 -> 768),
* the x axis does not: in 16:9 it clamps at 768 / 798 / 800 / 832, in 4:3 at 806,
* those x-clamps share ONE y wall (51.7% of the ``x2==832`` boxes sit on ``y2==432``,
  the 768-canvas height, not the 832-canvas 468), so x and y clamp independently.

Remaining question: the 16:9 bucket is a mix of source resolutions (1920x1080,
1280x720, 3840x2160, 2560x1440). If each resolution owns ONE x clamp, the annotation
frame is recoverable from ``(W, H)``. If every resolution shows all four clamps in the
same proportions, the x axis was re-bucketed by something invisible in this data and is
not recoverable.

So this probe keys its accumulators by the exact ``(src_w, src_h, kind)`` tuple and, per
resolution, reports:

1. the top-8 ``round(x2)`` values with counts + fractions, plus explicit counts for the
   known clamps (768 / 798 / 800 / 832 / 806) and every value carrying >= 1%,
2. ``x2`` percentiles p50/p90/p99/p99.9/max and ``y2`` p90/p99/p99.9/max,
3. the fraction of boxes with ``x2 > 768``,
4. a one-line scannable summary ending in a concentration verdict: of the overflow mass
   (``x2 > 768``), how much sits on that resolution's single biggest clamp.

Streaming only (``iter_batches`` via :func:`phantom_data.calib.join.iter_boxes`); table A
is one 1.09 GB row group and is never materialized.

    PYTHONPATH=third_party/PhantomData/src \
    python third_party/PhantomData/tools/probe_xclamp_by_resolution.py \
        --out /mnt/pfs/users/yuanze/datasets/phantom_canvas_calib_v1/xclamp_by_resolution.json
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
)

#: Known x clamps, reported explicitly for every resolution even when the count is 0.
KNOWN_CLAMPS = (768, 798, 800, 806, 832)

#: The "no overflow" reference edge. Everything above it is the anomaly being chased.
BASE_EDGE = 768

X_QUANTILES = (50, 90, 99, 99.9)
Y_QUANTILES = (90, 99, 99.9)


def hist_percentiles(hist: dict[int, int], quantiles) -> dict[str, float]:
    """Nearest-rank percentiles over an integer histogram (counts, not values).

    Values are ``round(coord)``, so percentiles are integer-resolution. The exact
    (unrounded) maximum is tracked separately by :class:`ResCell` and reported as
    ``max_exact``; ``max`` here is the rounded one, kept for consistency with the
    histogram the clamp counts come from.
    """
    total = sum(hist.values())
    if not total:
        return {}
    ordered = sorted(hist.items())
    out: dict[str, float] = {}
    for q in quantiles:
        rank = min(total, max(1, int(math.ceil(q / 100.0 * total))))
        seen = 0
        for value, count in ordered:
            seen += count
            if seen >= rank:
                out[f"p{q:g}"] = float(value)
                break
    out["max"] = float(ordered[-1][0])
    return out


class ResCell:
    """Per-``(src_w, src_h, kind)`` accumulator.

    Histograms rather than value lists: ~1M boxes across a few hundred cells, and the
    questions asked (clamp mass, percentiles, overflow fraction) are all answerable from
    ``round(coord)`` counts. Exact maxima are kept as floats so a non-integer wall would
    still be visible.
    """

    def __init__(self) -> None:
        self.n = 0
        self.x2_hist: Counter = Counter()
        self.y2_hist: Counter = Counter()
        self.x2_max_exact = float("-inf")
        self.y2_max_exact = float("-inf")
        self.x2_over_base = 0
        self.x2_noninteger = 0

    def add(self, box) -> None:
        x2, y2 = float(box[2]), float(box[3])
        self.n += 1
        self.x2_hist[int(round(x2))] += 1
        self.y2_hist[int(round(y2))] += 1
        if x2 > self.x2_max_exact:
            self.x2_max_exact = x2
        if y2 > self.y2_max_exact:
            self.y2_max_exact = y2
        if x2 > BASE_EDGE + 1e-6:
            self.x2_over_base += 1
        if abs(x2 - round(x2)) > 1e-6:
            self.x2_noninteger += 1

    def summarize(self, top_k: int = 8, min_frac: float = 0.01) -> dict:
        n = self.n
        frac = (lambda count: (count / n) if n else 0.0)
        top = [
            {"x2": value, "n": count, "frac": frac(count)}
            for value, count in self.x2_hist.most_common(top_k)
        ]
        # Every value carrying >= min_frac, whether or not it made the top-k cut.
        heavy = [
            {"x2": value, "n": count, "frac": frac(count)}
            for value, count in sorted(
                (item for item in self.x2_hist.items() if frac(item[1]) >= min_frac),
                key=lambda item: -item[1],
            )
        ]
        known = {
            str(clamp): {"n": self.x2_hist.get(clamp, 0), "frac": frac(self.x2_hist.get(clamp, 0))}
            for clamp in KNOWN_CLAMPS
        }
        # Overflow structure: does the >768 mass sit on ONE value or smear over many?
        overflow_hist = Counter(
            {value: count for value, count in self.x2_hist.items() if value > BASE_EDGE}
        )
        overflow_n = sum(overflow_hist.values())
        overflow_top = [
            {"x2": value, "n": count, "frac_of_overflow": (count / overflow_n) if overflow_n else 0.0}
            for value, count in overflow_hist.most_common(5)
        ]
        biggest = overflow_top[0] if overflow_top else None
        return {
            "n": n,
            "x2_top": top,
            "x2_heavy_ge_1pct": heavy,
            "x2_known_clamps": known,
            "x2_percentiles": hist_percentiles(self.x2_hist, X_QUANTILES),
            "y2_percentiles": hist_percentiles(self.y2_hist, Y_QUANTILES),
            "x2_max_exact": self.x2_max_exact if n else None,
            "y2_max_exact": self.y2_max_exact if n else None,
            "x2_noninteger_n": self.x2_noninteger,
            "frac_x2_over_768": frac(self.x2_over_base),
            "overflow": {
                "n": overflow_n,
                "distinct_values": len(overflow_hist),
                "top": overflow_top,
                "dominant_value": biggest["x2"] if biggest else None,
                "dominant_share_of_overflow": biggest["frac_of_overflow"] if biggest else 0.0,
            },
        }


def summary_line(res_key: str, kind: str, bucket: str, cell: dict) -> str:
    """One compact scannable line per resolution+kind (item 4)."""
    clamps = " ".join(
        f"{clamp}={cell['x2_known_clamps'][str(clamp)]['frac'] * 100:.1f}%"
        for clamp in KNOWN_CLAMPS
        if cell["x2_known_clamps"][str(clamp)]["n"]
    ) or "(none)"
    overflow = cell["overflow"]
    if overflow["n"]:
        shape = (
            f"overflow n={overflow['n']} on {overflow['distinct_values']} values, "
            f"top={overflow['dominant_value']} holds "
            f"{overflow['dominant_share_of_overflow'] * 100:.1f}%"
        )
    else:
        shape = "overflow none"
    x2max = cell["x2_max_exact"]
    return (
        f"{res_key:>11} {kind:<7} {bucket:<8} n={cell['n']:<7} "
        f"x2max={x2max:<8.6g} clamps: {clamps:<44} "
        f"x2>768={cell['frac_x2_over_768'] * 100:.1f}%  {shape}"
    )


def scan(
    filtered_parquet: str,
    meta_parquet: str,
    max_rows: int,
    batch_size: int,
    meta_batch_size: int,
    min_boxes: int,
    json_min_boxes: int,
    log_every: int,
    log=print,
) -> dict:
    started = time.time()
    log(f"[xclamp] building vid->(W,H) from {meta_parquet}")
    vid_wh = build_vid_wh(meta_parquet, batch_size=meta_batch_size)
    log(f"[xclamp] vid_wh entries={len(vid_wh)} ({time.time() - started:.1f}s)")

    stats = JoinStats()
    cells: dict[tuple[int, int, str], ResCell] = defaultdict(ResCell)
    next_log = log_every

    for kind, box, src_w, src_h, _vid, _phrase, _cls in iter_boxes(
        filtered_parquet, vid_wh, batch_size=batch_size, max_rows=max_rows, stats=stats
    ):
        cells[(src_w, src_h, kind)].add(box)
        if log_every and stats.rows_read >= next_log:
            next_log = stats.rows_read + log_every
            log(
                f"[xclamp] rows={stats.rows_read} boxes={stats.boxes_emitted} "
                f"cells={len(cells)} elapsed={time.time() - started:.0f}s"
            )

    log(
        f"[xclamp] scan done rows={stats.rows_read} boxes={stats.boxes_emitted} "
        f"cells={len(cells)} elapsed={time.time() - started:.0f}s"
    )

    entries = []
    for (src_w, src_h, kind), cell in cells.items():
        if cell.n < json_min_boxes:
            continue
        summary = cell.summarize()
        entries.append(
            {
                "src_w": src_w,
                "src_h": src_h,
                "resolution": f"{src_w}x{src_h}",
                "kind": kind,
                "aspect_bucket": aspect_bucket(src_w, src_h),
                "reported_in_lines": cell.n >= min_boxes,
                **summary,
            }
        )
    entries.sort(key=lambda item: (-item["n"],))

    return {
        "params": {
            "filtered_parquet": filtered_parquet,
            "meta_parquet": meta_parquet,
            "max_rows": max_rows,
            "batch_size": batch_size,
            "known_clamps": list(KNOWN_CLAMPS),
            "base_edge": BASE_EDGE,
            "min_boxes_for_lines": min_boxes,
            "min_boxes_for_json": json_min_boxes,
            "percentile_note": "percentiles computed over round(coord) histograms; "
            "x2_max_exact / y2_max_exact are unrounded",
        },
        "global": {
            "rows_read": stats.rows_read,
            "boxes": stats.boxes_emitted,
            "counters": stats.as_dict(),
            "vid_wh_entries": len(vid_wh),
            "distinct_resolution_kind_cells": len(cells),
            "wall_clock_sec": round(time.time() - started, 1),
        },
        "resolutions": entries,
    }


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------


def format_report(report: dict) -> str:
    lines: list[str] = []
    glob = report["global"]
    lines.append("=" * 132)
    lines.append("X-CLAMP BY SOURCE RESOLUTION -- is the x clamp recoverable from (W, H)?")
    lines.append("=" * 132)
    lines.append(f"rows read      : {glob['rows_read']}")
    lines.append(f"boxes analyzed : {glob['boxes']}")
    lines.append(f"counters       : {glob['counters']}")
    lines.append(f"cells          : {glob['distinct_resolution_kind_cells']} (W,H,kind)")
    lines.append(f"wall clock     : {glob['wall_clock_sec']}s")
    lines.append("")

    shown = [entry for entry in report["resolutions"] if entry["reported_in_lines"]]

    lines.append(
        f"### ITEM 4: compact per-resolution lines (n >= "
        f"{report['params']['min_boxes_for_lines']}), sorted by n"
    )
    for entry in shown:
        lines.append("  " + summary_line(entry["resolution"], entry["kind"], entry["aspect_bucket"], entry))
    lines.append("")

    lines.append("### ITEM 1: top-8 round(x2) values per resolution (+ every value >= 1%)")
    for entry in shown:
        top = ", ".join(
            f"{item['x2']}:{item['n']}({item['frac'] * 100:.1f}%)" for item in entry["x2_top"]
        )
        lines.append(f"  {entry['resolution']:>11} {entry['kind']:<7} n={entry['n']:<7} top8: {top}")
        heavy_only = [
            item for item in entry["x2_heavy_ge_1pct"]
            if item["x2"] not in {other["x2"] for other in entry["x2_top"]}
        ]
        if heavy_only:
            extra = ", ".join(
                f"{item['x2']}:{item['n']}({item['frac'] * 100:.1f}%)" for item in heavy_only
            )
            lines.append(f"  {'':>11} {'':<7} also >=1%: {extra}")
    lines.append("")

    lines.append("### ITEM 1b: known clamps, explicit counts (0 shown too)")
    header = (
        f"  {'resolution':>11} {'kind':<7} {'n':>8}"
        + "".join(f"{clamp:>14}" for clamp in KNOWN_CLAMPS)
    )
    lines.append("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    for entry in shown:
        cells_text = "".join(
            f"{entry['x2_known_clamps'][str(clamp)]['n']:>8}"
            f"{entry['x2_known_clamps'][str(clamp)]['frac'] * 100:>6.1f}"
            for clamp in KNOWN_CLAMPS
        )
        lines.append(
            f"  {entry['resolution']:>11} {entry['kind']:<7} {entry['n']:>8}{cells_text}"
        )
    lines.append("-" * len(header))
    lines.append("")

    lines.append("### ITEM 2+3: x2 / y2 percentiles and overflow fraction")
    header2 = (
        f"  {'resolution':>11} {'kind':<7} {'n':>8}"
        f"{'x2 p50':>9}{'x2 p90':>9}{'x2 p99':>9}{'x2 p99.9':>10}{'x2 max':>10}"
        f"{'y2 p90':>9}{'y2 p99':>9}{'y2 p99.9':>10}{'y2 max':>10}{'x2>768':>9}"
    )
    lines.append("-" * len(header2))
    lines.append(header2)
    lines.append("-" * len(header2))
    for entry in shown:
        x = entry["x2_percentiles"]
        y = entry["y2_percentiles"]
        lines.append(
            f"  {entry['resolution']:>11} {entry['kind']:<7} {entry['n']:>8}"
            f"{x['p50']:>9.1f}{x['p90']:>9.1f}{x['p99']:>9.1f}{x['p99.9']:>10.1f}"
            f"{entry['x2_max_exact']:>10.1f}"
            f"{y['p90']:>9.1f}{y['p99']:>9.1f}{y['p99.9']:>10.1f}"
            f"{entry['y2_max_exact']:>10.1f}"
            f"{entry['frac_x2_over_768'] * 100:>8.2f}%"
        )
    lines.append("-" * len(header2))
    lines.append("")

    lines.append("### 4:3 and non-landscape sources (who owns 806? does portrait/square overflow?)")
    for entry in report["resolutions"]:
        bucket = entry["aspect_bucket"]
        portrait_or_square = entry["src_h"] >= entry["src_w"]
        if bucket != "4:3" and not portrait_or_square:
            continue
        overflow = entry["overflow"]
        lines.append(
            f"  {entry['resolution']:>11} {entry['kind']:<7} {bucket:<8} n={entry['n']:<7} "
            f"x2max={entry['x2_max_exact']:<8.6g} x2>768={entry['frac_x2_over_768'] * 100:.2f}% "
            f"806={entry['x2_known_clamps']['806']['n']} "
            f"overflow_top={[(item['x2'], item['n']) for item in overflow['top']]}"
        )
    lines.append("")

    lines.append("### CROSS-RESOLUTION CONTRAST (the actual question)")
    lines.append(
        "  Per-resolution rule  => each (W,H) concentrates its overflow on ONE value and the "
        "values DIFFER across resolutions."
    )
    lines.append(
        "  Not recoverable      => the same clamp set appears at similar proportions in every "
        "resolution."
    )
    for entry in shown:
        overflow = entry["overflow"]
        if not overflow["n"]:
            continue
        lines.append(
            f"  {entry['resolution']:>11} {entry['kind']:<7} overflow values "
            f"{[(item['x2'], round(item['frac_of_overflow'], 4)) for item in overflow['top']]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/probe_xclamp_by_resolution.py",
        description="Break the Phantom-Data bbox x2 clamps down by exact source "
        "resolution to test whether the annotation frame width is recoverable from (W, H).",
    )
    parser.add_argument("--out", required=True, help="path to write xclamp_by_resolution.json")
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
        "--min-boxes",
        type=int,
        default=200,
        help="minimum boxes for a resolution to get its own text line",
    )
    parser.add_argument(
        "--json-min-boxes",
        type=int,
        default=1,
        help="minimum boxes for a resolution to appear in the JSON at all",
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
        min_boxes=args.min_boxes,
        json_min_boxes=args.json_min_boxes,
        log_every=args.log_every,
        log=log,
    )
    text = format_report(report)
    print(text, flush=True)
    report["text_summary"] = text
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
    log(f"[xclamp] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
