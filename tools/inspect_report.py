"""Aggregate a rendered inspection set into the numbers that decide "is this trainable".

Reads only ``_inspect/*/metrics.json``, so it is safe to re-run and needs no GPU.

The columns are chosen around the failure modes found by eye on the first 10 samples:
holes are mostly benign (real geometry), while a *wrong-object* reference cutout is the
error that matters and shows up as low ``ref_mask_coverage`` rather than as holes. The
report therefore cross-tabulates coverage against the ``min_ref_clip_score=0.23`` gate, to
show whether the funnel actually catches the bad ones.

Usage: python tools/inspect_report.py --dataset <root> [--out-root _inspect]
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path
from typing import Any, Callable

CLIP_GATE = 0.23
COVERAGE_SUSPECT = 0.20


def collect(dataset: Path, out_root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((dataset / out_root).glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for subject in metrics.get("subjects") or []:
            seed = subject.get("seed_box_clamp") or {}
            ref = subject.get("ref_box_clamp") or {}
            rows.append({
                "sample_id": metrics["sample_id"],
                "resolution": tuple(metrics.get("resolution") or ()),
                "video_worst": 100 * subject["holes_video"]["worst"],
                "video_median": 100 * subject["holes_video"]["median"],
                "ref_hole": 100 * subject["holes_ref_alpha"],
                "coverage": subject.get("ref_mask_coverage"),
                "components": subject.get("ref_mask_components"),
                "clip": subject.get("ref_clip_score"),
                "visible": subject.get("visible_frame_count"),
                "seed_clamped": bool(seed.get("clamped_any")),
                "ref_clamped": bool(ref.get("clamped_any")),
                "prompt": subject.get("prompt") or "",
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-root", default="_inspect")
    args = parser.parse_args(argv)

    rows = collect(args.dataset.resolve(), args.out_root)
    if not rows:
        print("no metrics.json found")
        return 1
    total = len(rows)
    samples = len({row["sample_id"] for row in rows})

    def share(predicate: Callable[[dict[str, Any]], bool]) -> str:
        hits = sum(1 for row in rows if predicate(row))
        return f"{hits} ({100 * hits / total:.0f}%)"

    def quantile(key: str, q: float) -> float:
        values = sorted(row[key] for row in rows if row[key] is not None)
        return values[min(len(values) - 1, int(q * len(values)))]

    print(f"=== {total} subjects over {samples} samples ===\n")

    print("HOLES (video masklet)")
    print(f"  worst-frame ratio: median {quantile('video_worst', .5):.2f}%  "
          f"p90 {quantile('video_worst', .9):.2f}%  "
          f"max {max(r['video_worst'] for r in rows):.2f}%")
    for cut in (2, 5, 10):
        print(f"  worst >{cut}%: {share(lambda r, c=cut: r['video_worst'] > c)}")
    print("HOLES (reference cutout alpha)")
    print(f"  median {quantile('ref_hole', .5):.2f}%  "
          f"max {max(r['ref_hole'] for r in rows):.2f}%  "
          f">2%: {share(lambda r: r['ref_hole'] > 2)}\n")

    print("REFERENCE CUTOUT (the failure holes cannot see)")
    print(f"  ref_mask_coverage: p10 {quantile('coverage', .1):.3f}  "
          f"median {quantile('coverage', .5):.3f}  p90 {quantile('coverage', .9):.3f}")
    print(f"  coverage <{COVERAGE_SUSPECT} (suspect wrong object): "
          f"{share(lambda r: (r['coverage'] or 1) < COVERAGE_SUSPECT)}")
    print(f"  multi-component: {share(lambda r: (r['components'] or 1) > 1)}\n")

    print(f"CLIP SCORE vs the {CLIP_GATE} gate")
    print(f"  median {quantile('clip', .5):.3f}  p10 {quantile('clip', .1):.3f}  "
          f"p90 {quantile('clip', .9):.3f}")
    print(f"  below gate (rejected): {share(lambda r: (r['clip'] or 1) < CLIP_GATE)}\n")

    suspect = [r for r in rows if (r["coverage"] or 1) < COVERAGE_SUSPECT]
    caught = [r for r in suspect if (r["clip"] or 1) < CLIP_GATE]
    print("DOES THE GATE CATCH THE BAD CUTOUTS?")
    print(f"  suspect (low coverage): {len(suspect)}   "
          f"of those rejected by clip<{CLIP_GATE}: {len(caught)}")
    leaked = [r for r in suspect if r not in caught]
    if leaked:
        print(f"  LEAKED INTO TRAINING ({len(leaked)}):")
        for row in leaked:
            print(f"    cov={row['coverage']:.3f} clip={row['clip']:.3f} "
                  f"{row['sample_id'][:12]} '{row['prompt'][:30]}'")
    print()

    print("BOX CLAMP")
    print(f"  seed box clamped: {share(lambda r: r['seed_clamped'])}")
    print(f"  ref box clamped:  {share(lambda r: r['ref_clamped'])}")
    print(f"  visible <20 frames: {share(lambda r: (r['visible'] or 81) < 20)}\n")

    print("WORST 12 BY ref_mask_coverage (inspect these in the viewer)")
    for row in sorted(rows, key=lambda r: r["coverage"] or 1)[:12]:
        verdict = ("REJ  " if (row["clip"] or 1) < CLIP_GATE else "PASS ")
        print(f"  {verdict} cov={row['coverage']:.3f} clip={row['clip']:.3f} "
              f"comp={row['components']} hole={row['video_worst']:.1f}% "
              f"{row['sample_id'][:12]} '{row['prompt'][:26]}'")
    print()
    print("WORST 6 BY video hole")
    for row in sorted(rows, key=lambda r: -r["video_worst"])[:6]:
        verdict = ("REJ  " if (row["clip"] or 1) < CLIP_GATE else "PASS ")
        print(f"  {verdict} hole={row['video_worst']:.1f}% "
              f"(median {row['video_median']:.2f}%) cov={row['coverage']:.3f} "
              f"{row['sample_id'][:12]} '{row['prompt'][:26]}'")

    resolutions: dict[tuple, int] = {}
    for row in rows:
        resolutions[row["resolution"]] = resolutions.get(row["resolution"], 0) + 1
    print("\nRESOLUTIONS")
    for resolution, count in sorted(resolutions.items(), key=lambda kv: -kv[1]):
        width, height = resolution
        ratio = width / height
        flag = "" if abs(ratio - 16 / 9) < 0.02 else "   <-- not 16:9, center-crop cuts height"
        print(f"  {width}x{height}  n={count}  ar={ratio:.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
