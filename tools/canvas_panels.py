"""Render one frame once per canvas hypothesis so the right protocol can be picked by eye.

The numbers cannot settle this. Every hypothesis in :mod:`phantom_data.canvas` produces an
in-frame box for a typical annotation, so "does it fit inside the frame" does not
discriminate; only "does the box sit on the object" does, and that is a visual judgement.

Reads the seed frame straight out of the already-extracted clip (stage B output), so it
needs no BOS access and no GPU. One PNG per sample: a grid of panels, each panel the same
frame with that hypothesis's box drawn, labelled with the id, the (sx, sy) it implies and
the mapped coordinates.

Usage:
  python tools/canvas_panels.py --dataset <root> --sample <sample_id> [--out <dir>]
  python tools/canvas_panels.py --dataset <root> --all --limit 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from phantom_data import canvas as canvas_module
from phantom_data.inspect import atomic_save_image, decode_frames, read_jsonl

BOX = (255, 214, 0)
BOX_BAD = (255, 80, 80)
PANEL_WIDTH = 640


def draw_panel(frame: np.ndarray, box: list[float], title: str, subtitle: str,
               width: int = PANEL_WIDTH):
    """One frame with one candidate box, labelled. Out-of-frame boxes are drawn in red."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    height, frame_width = frame.shape[:2]
    inside = (box[0] >= 0 and box[1] >= 0
              and box[2] <= frame_width and box[3] <= height)
    draw = ImageDraw.Draw(image)
    # clip only for drawing; the label carries the true (unclipped) numbers
    drawn = [max(0, min(box[0], frame_width - 1)), max(0, min(box[1], height - 1)),
             max(0, min(box[2], frame_width - 1)), max(0, min(box[3], height - 1))]
    if drawn[2] > drawn[0] and drawn[3] > drawn[1]:
        draw.rectangle(tuple(round(v) for v in drawn),
                       outline=BOX if inside else BOX_BAD, width=6)
    scale = width / image.width
    image = image.resize((width, max(1, int(round(image.height * scale)))))

    bar = 42
    canvas = Image.new("RGB", (width, image.height + bar), (16, 16, 16))
    canvas.paste(image, (0, bar))
    label = ImageDraw.Draw(canvas)
    label.text((6, 4), title, fill=BOX if inside else BOX_BAD)
    label.text((6, 22), subtitle, fill=(200, 200, 200))
    return canvas


def render_sample(dataset: Path, sample: dict[str, Any], extracted: dict[str, Any],
                  out_dir: Path, columns: int = 4) -> dict[str, Any]:
    from PIL import Image

    sample_id = sample["sample_id"]
    frames = decode_frames(dataset / sample["video"])
    height, width = frames[0].shape[:2]
    subjects = extracted.get("subjects") or []
    if not subjects:
        raise ValueError(f"{sample_id}: no subjects in the stage B manifest")

    report: dict[str, Any] = {"sample_id": sample_id, "resolution": [width, height],
                             "subjects": []}
    for subject in subjects:
        subject_id = int(subject["subject_id"])
        raw = [float(v) for v in subject["seed_bbox_768"]]
        seed = int(subject["seed_frame_index"])
        frame = frames[min(seed, len(frames) - 1)]

        panels, entries = [], []
        for hid, hypothesis in canvas_module.HYPOTHESES.items():
            sx, sy = hypothesis.scales(width, height)
            box = hypothesis.map_box(raw, width, height)
            inside = (box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height)
            panels.append(draw_panel(
                frame, box, f"{hid}   {'in frame' if inside else 'OUT OF FRAME'}",
                f"sx={sx:.3f} sy={sy:.3f}  ->  "
                f"[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]"))
            entries.append({"hypothesis": hid, "formula": hypothesis.formula,
                            "scales": [round(sx, 4), round(sy, 4)],
                            "mapped": [round(v, 1) for v in box], "in_frame": inside})

        cell_w, cell_h = panels[0].size
        rows = (len(panels) + columns - 1) // columns
        sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (16, 16, 16))
        for position, panel in enumerate(panels):
            sheet.paste(panel, ((position % columns) * cell_w,
                                (position // columns) * cell_h))
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_image(sheet, out_dir / f"canvas_subj{subject_id:02d}.png")

        report["subjects"].append({
            "subject_id": subject_id,
            "phrase": subject.get("phrase"),
            "seed_frame_index": seed,
            "raw_annotation": raw,
            "candidates": entries,
        })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sample", action="append", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-root", default="_canvas")
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    samples = read_jsonl(dataset / "segmented.jsonl")
    extracted = {row["sample_id"]: row
                 for row in read_jsonl(dataset / "extracted.jsonl")}
    if args.sample:
        wanted = set(args.sample)
        samples = [row for row in samples if row["sample_id"] in wanted]
    elif not args.all:
        parser.error("pass --sample <id> (repeatable) or --all")
    if args.limit:
        samples = samples[:args.limit]

    root = dataset / args.out_root
    reports, failures = [], []
    for position, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        try:
            report = render_sample(dataset, sample, extracted.get(sample_id) or {},
                                   root / sample_id, columns=args.columns)
            reports.append(report)
            print(f"[{position}/{len(samples)}] {sample_id} ok", flush=True)
        except Exception as error:  # noqa: BLE001
            failures.append({"sample_id": sample_id,
                             "error": f"{type(error).__name__}: {error}"})
            print(f"[{position}/{len(samples)}] {sample_id} FAILED "
                  f"{type(error).__name__}: {error}", flush=True)

    root.mkdir(parents=True, exist_ok=True)
    (root / "canvas_report.json").write_text(
        json.dumps({"hypotheses": list(canvas_module.HYPOTHESES),
                    "samples": reports, "failures": failures},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nrendered {len(reports)}, failed {len(failures)} -> {root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
