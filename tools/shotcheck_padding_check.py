"""Validate the 81-frame padding path of phantom_data.build.shotcheck against synthetics.

Three controls, each 81 frames at TransNet's 48x27 input size:

* ``static``      one constant image -> a correct padding path must produce ~0 everywhere,
                  in particular at frames 0-2 and 78-80 (the padded edges)
* ``cut@40``      first half image A, second half image B -> a spike at frame 40 and
                  nothing at the edges
* ``cut@2``       cut two frames in, i.e. a real boundary that lives *inside* the region
                  UltraVid's boundary_padding_frames would mask; tells us what the edge
                  mask costs us

Also compares the padded windowing against feeding the bare 81 frames, which is what a
naive implementation would do, to quantify the difference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from phantom_data.build.shotcheck import load_model, score_clip


def synth(kind: str, count: int = 81) -> np.ndarray:
    rng = np.random.default_rng(0)
    left = rng.integers(0, 90, size=(27, 48, 3), dtype=np.uint8)
    right = rng.integers(160, 256, size=(27, 48, 3), dtype=np.uint8)
    frames = np.empty((count, 27, 48, 3), dtype=np.uint8)
    if kind == "static":
        frames[:] = left
        return frames
    at = int(kind.split("@")[1])
    frames[:at] = left
    frames[at:] = right
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    model = load_model(args.source, args.weights, args.device)
    for kind in ("static", "cut@40", "cut@2", "cut@78"):
        frames = synth(kind)
        with torch.inference_mode():
            padded_pred, _ = score_clip(model, frames)
            # naive: bare 81 frames, no padding, no windowing
            output = model.model(torch.from_numpy(frames[None]).to(args.device))
            logits = output[0] if isinstance(output, tuple) else output
            naive = torch.sigmoid(logits)[0, :, 0].float().cpu().numpy()
        padded = np.asarray(padded_pred)
        edge = np.concatenate([padded[:3], padded[-3:]])
        print(f"\n[{kind}]")
        print(f"  padded : argmax=f{int(padded.argmax())} max={padded.max():.4f} "
              f"edge(0-2,78-80) max={edge.max():.4f} "
              f"frames>=0.5={[int(i) for i in np.where(padded >= 0.5)[0]]}")
        print(f"  naive  : argmax=f{int(naive.argmax())} max={naive.max():.4f} "
              f"edge max={max(naive[:3].max(), naive[-3:].max()):.4f} "
              f"frames>=0.5={[int(i) for i in np.where(naive >= 0.5)[0]]}")
        print(f"  max|padded-naive|={np.abs(padded - naive).max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
