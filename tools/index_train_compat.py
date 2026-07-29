"""Check a stage D index against what bboxref_train.py actually reads off a CSV row.

Loads metadata via the trainer's own path (pandas -> per-row dict, as
diffsynth UnifiedDataset.load_metadata does), then asserts every column the
training/eval code touches is present and every referenced asset resolves under
--dataset-base-path. Catches a missing column before it becomes a launch-time crash.

Usage: python tools/index_train_compat.py <dataset_root> <index_name>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Columns bboxref_train.py reads (grep: row["prompt"], row["bbox"],
# row["object_reference_images"], row.get("video_id"), data["frame_count"], "video").
REQUIRED = ["sample_id", "video_id", "video", "bbox", "prompt",
            "object_reference_images", "frame_count"]
ULTRAVID_HEADER = ("sample_id,video_id,video,vace_video,bbox,prompt,"
                   "object_reference_images,frame_count")


def check(dataset: Path, index_name: str) -> int:
    import pandas

    index = dataset / "indexes" / index_name
    problems: list[str] = []
    for split in ("train", "eval"):
        path = index / f"metadata_{split}.csv"
        header = path.read_text(encoding="utf-8").splitlines()[0]
        if header != ULTRAVID_HEADER:
            problems.append(f"{split}: header differs from UltraVid index\n"
                            f"  got      {header}\n  expected {ULTRAVID_HEADER}")
        frame = pandas.read_csv(path)
        rows = [frame.iloc[i].to_dict() for i in range(len(frame))]
        for column in REQUIRED:
            if column not in frame.columns:
                problems.append(f"{split}: missing column {column}")
        refs = 0
        for row in rows:
            for column in REQUIRED:
                value = row.get(column)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    problems.append(f"{split}/{row.get('sample_id')}: empty {column}")
            for key in ("video", "bbox"):
                if not (dataset / str(row[key])).is_file():
                    problems.append(f"{split}/{row['sample_id']}: {key} not found: {row[key]}")
            paths = json.loads(row["object_reference_images"])
            if not paths:
                problems.append(f"{split}/{row['sample_id']}: no object references")
            for relative in paths:
                refs += 1
                if not (dataset / relative).is_file():
                    problems.append(f"{split}/{row['sample_id']}: ref not found: {relative}")
            # The trainer samples a window of num_frames from frame_count.
            if int(row["frame_count"]) < 1:
                problems.append(f"{split}/{row['sample_id']}: bad frame_count")
        print(f"{split}: {len(rows)} rows, {refs} refs, frame_counts="
              f"{sorted({int(r['frame_count']) for r in rows})}")

    train = pandas.read_csv(index / "metadata_train.csv")
    evaluation = pandas.read_csv(index / "metadata_eval.csv")
    overlap = set(train["video_id"]) & set(evaluation["video_id"])
    if overlap:
        problems.append(f"train/eval share video_id: {sorted(overlap)[:5]}")

    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    print(f"OK {index_name}: schema + assets + source-disjoint split all check out")
    return 0


if __name__ == "__main__":
    raise SystemExit(check(Path(sys.argv[1]), sys.argv[2]))
