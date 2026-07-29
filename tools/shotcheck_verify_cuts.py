"""Independently verify TransNet's flagged clips: is the scene actually different?

TransNet says "transition here". That can also fire on fast motion or a whip pan, so this
script checks the thing a shot cut implies and motion does not: the content *before* the
span and the content *after* it belong to different scenes.

For each flagged clip it compares frame ``span_start - k`` against ``span_end + k`` with two
metrics that do not share TransNet's features:

* ``hist_corr``   correlation of 3x32-bin RGB histograms (global appearance)
* ``tile_corr``   mean correlation of per-tile grayscale histograms on a 4x4 grid, which
                  survives a global brightness shift but not a scene change

As a control, the same pair distance is computed for equally spaced frame pairs elsewhere
in the same clip. A real cut should sit far outside that clip's own baseline; fast motion
should not.

    python tools/shotcheck_verify_cuts.py --dataset <root> [--max-prob 0.1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rgb_hist(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    parts = [np.histogram(frame[..., channel], bins=bins, range=(0, 256))[0]
             for channel in range(3)]
    vector = np.concatenate(parts).astype(np.float64)
    return vector / max(vector.sum(), 1.0)


def tile_hist(frame: np.ndarray, grid: int = 4, bins: int = 32) -> np.ndarray:
    gray = (0.299 * frame[..., 0] + 0.587 * frame[..., 1]
            + 0.114 * frame[..., 2]).astype(np.float32)
    height, width = gray.shape
    parts = []
    for row in range(grid):
        for column in range(grid):
            tile = gray[row * height // grid:(row + 1) * height // grid,
                        column * width // grid:(column + 1) * width // grid]
            counts = np.histogram(tile, bins=bins, range=(0, 256))[0].astype(np.float64)
            parts.append(counts / max(counts.sum(), 1.0))
    return np.stack(parts)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt((left ** 2).sum() * (right ** 2).sum())
    return float((left * right).sum() / denominator) if denominator > 0 else 1.0


def compare(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    hist = correlation(rgb_hist(first), rgb_hist(second))
    tiles_a, tiles_b = tile_hist(first), tile_hist(second)
    tile = float(np.mean([correlation(a, b) for a, b in zip(tiles_a, tiles_b)]))
    return hist, tile


def main(argv: list[str] | None = None) -> int:
    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--max-prob", type=float, default=0.1,
                        help="verify clips whose single-frame OR many-hot max exceeds this")
    parser.add_argument("--gap", type=int, default=3,
                        help="frames outside the span to sample from")
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    summary = json.loads((dataset / "transnet_summary.json").read_text(encoding="utf-8"))
    selected = [row for row in summary["rows"]
                if max(row["max_prob"], row.get("many_hot_max", 0.0)) > args.max_prob]
    print(f"verifying {len(selected)} clips (max_prob or many_hot_max > {args.max_prob})\n")
    print(f"{'sample_id':<46} {'span':<10} {'histCorr':>9} {'tileCorr':>9} "
          f"{'baseHist':>9} {'baseTile':>9} {'verdict':<22}")

    for row in selected:
        record = json.loads(
            (dataset / "transnet" / f"{row['sample_id']}.json").read_text(encoding="utf-8"))
        predictions = np.asarray(record["predictions"])
        span = record.get("transition_span")
        if not span:
            centre = int(predictions.argmax())
            span = [centre, centre]
        start = max(0, span[0] - args.gap)
        end = min(len(predictions) - 1, span[1] + args.gap)
        width = end - start

        reader = imageio.get_reader(dataset / record["video"])
        try:
            frames = {index: np.asarray(frame) for index, frame in enumerate(reader)}
        finally:
            reader.close()

        hist, tile = compare(frames[start], frames[end])
        # Baseline: same temporal distance, elsewhere in the clip, avoiding the span.
        # Subsampled every 4th anchor; full-res histograms are the cost here, and the
        # median of ~15 pairs is already stable for a "far outside baseline" test.
        baseline = []
        for anchor in range(0, len(predictions) - width, 4):
            if anchor + width < span[0] - args.gap or anchor > span[1] + args.gap:
                baseline.append(compare(frames[anchor], frames[anchor + width]))
        base_hist = float(np.median([h for h, _ in baseline])) if baseline else float("nan")
        base_tile = float(np.median([t for _, t in baseline])) if baseline else float("nan")

        # A cut: pair correlation collapses relative to the clip's own baseline.
        if tile < base_tile - 0.25 or tile < 0.35:
            verdict = "scene change"
        elif tile < base_tile - 0.10:
            verdict = "partial change"
        else:
            verdict = "no scene change"
        print(f"{row['sample_id']:<46} {f'{span[0]}-{span[1]}':<10} {hist:>9.4f} "
              f"{tile:>9.4f} {base_hist:>9.4f} {base_tile:>9.4f} {verdict:<22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
