"""Report + diagnostic plots for the shotcheck stage (see phantom_data.build.shotcheck).

Reads ``transnet_summary.json`` / ``transnet/<sample_id>.json`` and prints the
distribution, the top-N table, the edge-vs-interior split, and the cross-tabulation of
detections against "is the 81-frame window flush against the Phantom clip boundary".

Plots (one PNG per sample) go to ``_selfcheck/shotcheck/``: per-frame TransNet
probability curve plus the frames around the peak/boundary, so the cut is visible.

    python tools/shotcheck_report.py --dataset <root> --top 15 --plots 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from phantom_data.build.shotcheck import shot_reject_reasons

FLUSH_EPS = 0.05  # <=50 ms from the clip edge counts as flush


def load(dataset: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = json.loads((dataset / "transnet_summary.json").read_text(encoding="utf-8"))
    records = {}
    for path in sorted((dataset / "transnet").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["sample_id"]] = record
    return summary, records


def is_flush(row: dict[str, Any]) -> tuple[bool, bool]:
    head = row.get("head_margin_sec")
    tail = row.get("tail_margin_sec")
    return (head is not None and head <= FLUSH_EPS, tail is not None and tail <= FLUSH_EPS)


def report(dataset: Path, top: int, threshold: float) -> dict[str, Any]:
    summary, records = load(dataset)
    rows = summary["rows"]
    print(f"clips={len(rows)} threshold={summary['threshold']} "
          f"boundary_padding={summary['boundary_padding_frames']}")
    print(json.dumps(summary["distribution"], indent=2))

    print(f"\n=== top {top} by max_prob ===")
    print(f"{'sample_id':<46} {'maxP':>7} {'f':>3} {'intMaxP':>8} {'manyMax':>8} "
          f"{'trFr':>5} {'resolution':>10} {'boundaries':<12} {'head':>7} {'tail':>7} "
          f"{'reject':<28}")
    for row in rows[:top]:
        reasons = ",".join(shot_reject_reasons(row)) or "-"
        print(f"{row['sample_id']:<46} {row['max_prob']:>7.4f} "
              f"{row['max_prob_frame']:>3} {row['interior_max_prob']:>8.4f} "
              f"{row.get('many_hot_max', 0.0):>8.4f} "
              f"{row.get('transition_frames', 0):>5} {row['resolution']:>10} "
              f"{str(row['boundaries']):<12} "
              f"{_fmt(row.get('head_margin_sec')):>7} {_fmt(row.get('tail_margin_sec')):>7} "
              f"{reasons:<28}")

    rejected = [row for row in rows if shot_reject_reasons(row)]
    print(f"\n=== default policy rejections ({len(rejected)}/{len(rows)}) ===")
    for row in rejected:
        print(f"  {row['sample_id']:<46} {','.join(shot_reject_reasons(row)):<30} "
              f"singleMax={row['max_prob']:.4f} manyMax={row.get('many_hot_max', 0.0):.4f} "
              f"span={row.get('transition_span')} "
              f"({row.get('transition_frames', 0)} frames = "
              f"{row.get('transition_frames', 0) / 16:.2f}s)")

    # Edge vs interior detections.
    edge_frames: dict[int, int] = {}
    interior_frames: dict[int, int] = {}
    for row in rows:
        for frame in row["edge_boundaries"]:
            edge_frames[frame] = edge_frames.get(frame, 0) + 1
        for frame in row["interior_boundaries"]:
            interior_frames[frame] = interior_frames.get(frame, 0) + 1
    print("\n=== boundary frame positions ===")
    print(f"clips with any boundary        : {sum(1 for r in rows if r['boundaries'])}")
    print(f"clips with edge boundary (0-1/79-80 at padding=2): "
          f"{sum(1 for r in rows if r['edge_boundaries'])}")
    print(f"clips with interior boundary   : "
          f"{sum(1 for r in rows if r['interior_boundaries'])}")
    print(f"edge boundary frame histogram    : {dict(sorted(edge_frames.items()))}")
    print(f"interior boundary frame histogram: {dict(sorted(interior_frames.items()))}")

    # How close are interior boundaries to the window ends?
    near = {"first_10": 0, "last_10": 0, "middle": 0}
    for row in rows:
        for frame in row["interior_boundaries"]:
            count = row["frame_count"]
            if frame < 10:
                near["first_10"] += 1
            elif frame >= count - 10:
                near["last_10"] += 1
            else:
                near["middle"] += 1
    print(f"interior boundary position      : {near}")

    # Flush-boundary cross tabulation.
    print("\n=== window flush against Phantom clip boundary vs detections ===")
    groups = {"flush": [], "interior_window": []}
    for row in rows:
        head, tail = is_flush(row)
        groups["flush" if (head or tail) else "interior_window"].append(row)
    table = {}
    for name, group in groups.items():
        probabilities = [r["max_prob"] for r in group]
        table[name] = {
            "clips": len(group),
            "rejected_by_policy": sum(1 for r in group if shot_reject_reasons(r)),
            "with_interior_boundary": sum(1 for r in group if r["interior_boundaries"]),
            "with_any_boundary": sum(1 for r in group if r["boundaries"]),
            "with_transition_span": sum(1 for r in group
                                        if r.get("transition_frames", 0) > 0),
            "max_prob_gt_0.3": sum(1 for v in probabilities if v > 0.3),
            "max_prob_gt_0.1": sum(1 for v in probabilities if v > 0.1),
            "max_prob_max": round(max(probabilities), 4) if probabilities else 0.0,
            "max_prob_median": round(float(np.median(probabilities)), 4) if probabilities else 0.0,
            "max_prob_mean": round(float(np.mean(probabilities)), 4) if probabilities else 0.0,
        }
    print(json.dumps(table, indent=2))

    head_only = [r for r in rows if is_flush(r)[0] and not is_flush(r)[1]]
    tail_only = [r for r in rows if is_flush(r)[1] and not is_flush(r)[0]]
    both = [r for r in rows if all(is_flush(r))]
    for name, group in (("head_flush_only", head_only), ("tail_flush_only", tail_only),
                        ("both_flush", both)):
        probabilities = [r["max_prob"] for r in group] or [0.0]
        print(f"{name:<18} n={len(group):<3} maxP max={max(probabilities):.4f} "
              f"median={float(np.median(probabilities)):.4f}")

    # Margin-cost estimate for a window.py safety margin.
    print("\n=== safety-margin cost estimate ===")
    slack = []
    for row in rows:
        head, tail = row.get("head_margin_sec"), row.get("tail_margin_sec")
        if head is None or tail is None:
            continue
        slack.append(head + tail)  # total movable+spare room inside the clip
    for margin in (0.1, 0.2, 0.25, 0.3, 0.5):
        need = 2 * margin
        short = sum(1 for value in slack if value < need)
        print(f"margin={margin:>4}s per side -> needs {need:.2f}s spare; "
              f"{short}/{len(slack)} clips cannot afford it "
              f"({100.0 * short / max(len(slack), 1):.1f}%)")
    return {"summary": summary, "records": records, "flush_table": table}


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def simulate_margin(specs_path: Path, margins: tuple[float, ...]) -> None:
    """Re-run ``window.choose_window`` with a safety margin and count the real cost.

    Counting "spare seconds" understates the cost, because shrinking the allowed window
    range can also make a *seed* uncoverable, dropping a subject rather than the clip.
    This replays the actual stage-A selection.
    """
    from phantom_data.build.window import WINDOW_SEC, choose_window, sample_id_for

    specs = [json.loads(line) for line in
             specs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\n=== window.py safety-margin simulation ({len(specs)} specs) ===")
    baseline = {}
    for spec in specs:
        source = spec["source"]
        seeds = [float(s["seed_abs_time"]) for s in spec["subjects"]]
        baseline[spec["sample_id"]] = (source, seeds, len(spec["subjects"]))

    print(f"{'margin':>7} {'kept':>6} {'lost_clip':>10} {'fewer_subjects':>15} "
          f"{'subjects_kept':>14} {'window_moved':>13}")
    total_subjects = sum(count for _, _, count in baseline.values())
    for margin in margins:
        kept = lost = fewer = moved = 0
        subjects_kept = 0
        for sample_id, (source, seeds, count) in baseline.items():
            start = float(source["clip_start_sec"]) + margin
            end = float(source["clip_end_sec"]) - margin
            plan = choose_window(start, end, seeds) if end - start >= WINDOW_SEC else None
            if plan is None:
                lost += 1
                continue
            kept += 1
            subjects_kept += len(plan.covered)
            if len(plan.covered) < count:
                fewer += 1
            if sample_id_for(source["key"], plan.window_start) != sample_id_for(
                    source["key"], float(source["window_start_sec"])):
                moved += 1
        print(f"{margin:>7.2f} {kept:>6} {lost:>10} {fewer:>15} "
              f"{subjects_kept:>7}/{total_subjects:<6} {moved:>13}")


def plot(dataset: Path, records: dict[str, dict[str, Any]], sample_ids: list[str],
         out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for sample_id in sample_ids:
        record = records[sample_id]
        predictions = np.asarray(record["predictions"], dtype=np.float32)
        boundaries = record["boundaries"]
        focus = boundaries[0] if boundaries else int(np.argmax(predictions))
        window = [f for f in range(focus - 2, focus + 3) if 0 <= f < len(predictions)]

        reader = imageio.get_reader(dataset / record["video"])
        try:
            frames = {index: np.asarray(frame) for index, frame in enumerate(reader)
                      if index in set(window)}
        finally:
            reader.close()

        figure = plt.figure(figsize=(13, 6.2), dpi=110)
        grid = figure.add_gridspec(2, len(window), height_ratios=[1.15, 1.0], hspace=0.32)
        axis = figure.add_subplot(grid[0, :])
        axis.plot(predictions, color="#2b6cb0", linewidth=1.6, label="single-frame P(cut)")
        axis.plot(record["many_hot"], color="#a0aec0", linewidth=1.0, linestyle="--",
                  label="many-hot P")
        axis.axhline(record["threshold"], color="#c53030", linewidth=1.0, linestyle=":",
                     label=f"threshold {record['threshold']}")
        padding = record["boundary_padding_frames"]
        axis.axvspan(-0.5, padding - 0.5, color="#edf2f7", zorder=0)
        axis.axvspan(len(predictions) - padding - 0.5, len(predictions) - 0.5,
                     color="#edf2f7", zorder=0, label="padding-affected edge")
        for frame in boundaries:
            axis.axvline(frame, color="#dd6b20", linewidth=1.0, alpha=0.8)
        axis.set_xlim(-0.5, len(predictions) - 0.5)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("frame")
        axis.set_ylabel("P(shot boundary)")
        head, tail = record["window"]["head_margin_sec"], record["window"]["tail_margin_sec"]
        axis.set_title(
            f"{sample_id}  {record['width']}x{record['height']}\n"
            f"max P={record['max_prob']:.4f} @f{record['max_prob_frame']}  "
            f"interior max={record['interior_max_prob']:.4f}  "
            f"boundaries={boundaries}  clip margins head={_fmt(head)}s tail={_fmt(tail)}s",
            fontsize=9,
        )
        axis.legend(fontsize=7, loc="upper right", ncol=2)

        for column, frame_index in enumerate(window):
            cell = figure.add_subplot(grid[1, column])
            cell.imshow(frames[frame_index])
            cell.set_xticks([])
            cell.set_yticks([])
            marker = " (peak)" if frame_index == focus else ""
            cell.set_title(f"f{frame_index}  P={predictions[frame_index]:.3f}{marker}",
                           fontsize=8)
            if frame_index == focus:
                for side in cell.spines.values():
                    side.set_color("#dd6b20")
                    side.set_linewidth(2.5)

        path = out_dir / f"{sample_id}.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
        print(f"wrote {path}", flush=True)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--plots", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--specs", type=Path, default=None,
                        help="stage-A specs jsonl, for the margin simulation")
    args = parser.parse_args(argv)
    result = report(args.dataset, args.top, args.threshold)
    if args.specs:
        simulate_margin(args.specs, (0.1, 0.2, 0.25, 0.3, 0.5))
    if args.plots:
        rows = result["summary"]["rows"][:args.plots]
        plot(args.dataset.resolve(), result["records"], [r["sample_id"] for r in rows],
             args.dataset / "_selfcheck" / "shotcheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
