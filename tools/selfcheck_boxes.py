"""Draw the planned boxes back onto the produced artifacts to prove the math.

For each checked sample it decodes ``seed_frame_index`` out of the *produced* mp4, maps
``seed_bbox_768`` under the currently-assumed hypothesis ``scale = max(W, H) / 768``
(``H_768_long`` in ``phantom_data.canvas``; the real annotation canvas is unresolved and
under calibration) and draws it; then does the same for the reference jpg with
``ref.bbox_768``. If the window arithmetic or the coordinate scaling were wrong, the box
would not land on the object.

Run inside the pod:
    python tools/selfcheck_boxes.py <dataset root> [count]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from decord import VideoReader
from PIL import Image, ImageDraw

from phantom_data.dataset import scale_bbox

GREEN = (0, 255, 0)
CYAN = (0, 255, 255)


def annotate(image: Image.Image, box: list[float], color, label: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width = max(3, image.width // 320)
    draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=width)
    draw.text((box[0] + 4, max(0, box[1] - 14)), label, fill=color)
    return image


def main(dataset_root: str, count: int = 3) -> int:
    dataset = Path(dataset_root)
    out_dir = dataset / "_selfcheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (dataset / "extracted.jsonl").read_text().splitlines() if line.strip()]

    # prefer showing the multi-subject samples: they exercise the window-coverage logic
    rows.sort(key=lambda row: (-len(row["subjects"]), row["sample_id"]))
    report = []
    for row in rows[:count]:
        sample_id = row["sample_id"]
        reader = VideoReader(str(dataset / row["video"]))
        frames = len(reader)
        height, width = reader[0].asnumpy().shape[:2]
        assert (width, height) == (row["width"], row["height"]), "manifest resolution mismatch"

        for subject in row["subjects"]:
            index = int(subject["seed_frame_index"])
            frame = reader[min(index, frames - 1)].asnumpy()
            box = scale_bbox([float(v) for v in subject["seed_bbox_768"]], width, height)
            image = annotate(
                Image.fromarray(frame),
                box,
                GREEN,
                f"{subject['bbox_cls']} f{index}",
            )
            target_out = out_dir / f"{sample_id}_subj{subject['subject_id']:02d}_seed_f{index:03d}.jpg"
            image.save(target_out, quality=92)

            reference = subject["ref"]
            ref_image = Image.open(dataset / reference["frame"]).convert("RGB")
            ref_box = scale_bbox(
                [float(v) for v in reference["bbox_768"]], ref_image.width, ref_image.height
            )
            ref_out = out_dir / f"{sample_id}_subj{subject['subject_id']:02d}_ref.jpg"
            annotate(ref_image, ref_box, CYAN, str(reference["bbox_cls"])).save(ref_out, quality=92)

            report.append(
                {
                    "sample_id": sample_id,
                    "subject_id": subject["subject_id"],
                    "phrase": subject["phrase"],
                    "clip_frames": frames,
                    "clip_wh": [width, height],
                    "seed_frame_index": index,
                    "seed_box_scaled": [round(v, 1) for v in box],
                    "seed_box_inside_frame": bool(
                        0 <= box[0] < box[2] <= width + 1 and 0 <= box[1] < box[3] <= height + 1
                    ),
                    "seed_box_area_frac": round(
                        (box[2] - box[0]) * (box[3] - box[1]) / float(width * height), 4
                    ),
                    "ref_wh": [ref_image.width, ref_image.height],
                    "ref_box_scaled": [round(v, 1) for v in ref_box],
                    "ref_box_inside_frame": bool(
                        0 <= ref_box[0] < ref_box[2] <= ref_image.width + 1
                        and 0 <= ref_box[1] < ref_box[3] <= ref_image.height + 1
                    ),
                    "target_jpg": str(target_out),
                    "ref_jpg": str(ref_out),
                }
            )
    (out_dir / "selfcheck.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    bad = [item for item in report if not (item["seed_box_inside_frame"] and item["ref_box_inside_frame"])]
    print(f"checked={len(report)} out_of_frame={len(bad)} -> {out_dir}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
