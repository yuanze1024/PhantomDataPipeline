"""Measure what stage B's downscale actually saves, on real pilot clips.

Not an estimate: H.264 bitrate is nowhere near proportional to pixel area (it tracks residual
entropy, so a static talking-head shot compresses far better than a pan), and the decision to
re-extract 126k samples rests on the real ratio. So this decodes shipped pilot clips and
re-encodes them both ways through the *same* :func:`extract.encode_mp4` the pipeline uses.

**The control matters.** Comparing the shipped clip's size against a downscaled re-encode would
confound the downscale with a second generation of lossy encoding. Both arms here are therefore
re-encodes of the same decoded frames -- one at source resolution, one downscaled -- so the only
difference between them is the resize. The shipped size is reported alongside as a sanity check
that the source-resolution re-encode reproduces it.

Read-only on its input. Writes nothing but a JSON summary to ``--out``.

    python tools/measure_downscale_bytes.py --dataset <pilot root> --limit 8 --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from phantom_data.build import extract
from phantom_data.build.resolution import DEFAULT_TARGET_HEIGHT, storage_plan


def measure_clip(path: Path, target_height: int) -> dict[str, object]:
    """Both arms for one clip: source-resolution re-encode vs downscaled re-encode."""
    import imageio.v2 as imageio
    import numpy as np

    reader = imageio.get_reader(path)
    try:
        frames = [np.asarray(frame)[..., :3] for frame in reader]
    finally:
        reader.close()
    height, width = frames[0].shape[:2]
    plan = storage_plan(width, height, target_height)

    started = time.time()
    source_bytes = len(extract.encode_mp4(frames, 16))
    control_seconds = time.time() - started

    started = time.time()
    scaled = extract.resize_frames(frames, int(plan["width"]), int(plan["height"]))
    resize_seconds = time.time() - started
    scaled_bytes = len(extract.encode_mp4(scaled, 16))
    encode_seconds = time.time() - started - resize_seconds

    return {
        "clip": path.name,
        "frames": len(frames),
        "source": f"{width}x{height}",
        "stored": f"{plan['width']}x{plan['height']}",
        "shipped_bytes": path.stat().st_size,
        "source_reencode_bytes": source_bytes,
        "scaled_bytes": scaled_bytes,
        "ratio": round(source_bytes / scaled_bytes, 3),
        "pixel_ratio": round((width * height) / (int(plan["width"]) * int(plan["height"])), 3),
        "resize_seconds": round(resize_seconds, 2),
        "scaled_encode_seconds": round(encode_seconds, 2),
        "control_encode_seconds": round(control_seconds, 2),
        "crop_discard_excessive": plan["crop_discard_excessive"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--target-height", type=int, default=DEFAULT_TARGET_HEIGHT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    clips = sorted((args.dataset / "clips").glob("*.mp4"))[: args.limit or None]
    rows = []
    for position, clip in enumerate(clips, 1):
        row = measure_clip(clip, args.target_height)
        rows.append(row)
        print(f"[{position}/{len(clips)}] {row['source']}->{row['stored']} "
              f"shipped={row['shipped_bytes'] / 1048576:.2f} "
              f"ctrl={row['source_reencode_bytes'] / 1048576:.2f} "
              f"scaled={row['scaled_bytes'] / 1048576:.2f} MiB  x{row['ratio']}", flush=True)

    summary = {
        "clips": len(rows),
        "target_height": args.target_height,
        "shipped_mib_mean": round(sum(r["shipped_bytes"] for r in rows) / len(rows) / 1048576, 3),
        "source_reencode_mib_mean": round(
            sum(r["source_reencode_bytes"] for r in rows) / len(rows) / 1048576, 3),
        "scaled_mib_mean": round(sum(r["scaled_bytes"] for r in rows) / len(rows) / 1048576, 3),
        # Ratio of the totals, which is the number that predicts the dataset's size. The mean and
        # median of the per-clip ratios are reported too: they differ, and the spread is the
        # honest statement of how much a per-clip prediction can be trusted.
        "ratio_of_totals": round(sum(r["source_reencode_bytes"] for r in rows)
                                 / sum(r["scaled_bytes"] for r in rows), 3),
        "ratio_vs_shipped": round(sum(r["shipped_bytes"] for r in rows)
                                  / sum(r["scaled_bytes"] for r in rows), 3),
        "ratio_mean": round(statistics.mean(r["ratio"] for r in rows), 3),
        "ratio_median": round(statistics.median(r["ratio"] for r in rows), 3),
        "ratio_min": min(r["ratio"] for r in rows),
        "ratio_max": max(r["ratio"] for r in rows),
        "resize_seconds_mean": round(statistics.mean(r["resize_seconds"] for r in rows), 2),
        "scaled_encode_seconds_mean": round(
            statistics.mean(r["scaled_encode_seconds"] for r in rows), 2),
        "control_encode_seconds_mean": round(
            statistics.mean(r["control_encode_seconds"] for r in rows), 2),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
