"""One-off probe: is a decord reader stable across a big get_batch then a far seek?

Stage B decodes 81 target frames from a source video and then grabs a single reference
frame from a different part of the SAME file through the same cached reader. If decord's
seek were unreliable after a large batch read, every reference frame would silently be
the wrong frame. This compares the shared-reader grab against a grab from a freshly
opened reader, pixel for pixel.

Run inside the pod. Not part of the package; no assertions about the dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from decord import VideoReader

from phantom_data.bos import FrameGrabber, load_aksk, make_client
from phantom_data.build.extract import frame_indices_for_window


def main(specs_path: str, count: int = 3) -> int:
    specs = [json.loads(line) for line in Path(specs_path).read_text().splitlines() if line.strip()]
    ak, sk = load_aksk()
    grabber = FrameGrabber(make_client(ak, sk))
    failures = 0
    checked = 0
    for spec in specs:
        if checked >= count:
            break
        source = spec["source"]
        subject = spec["subjects"][0]
        reference = subject["ref"]
        same_file = (reference["bucket"], reference["key"]) == (source["bucket"], source["key"])

        # shared reader: batch-read the 81 target frames first, then seek to the ref time
        try:
            reader = grabber.reader(source["bucket"], source["key"])
        except Exception as error:  # some containers cannot be opened over http at all
            print(f"{spec['sample_id']} SKIP open failed: {type(error).__name__}: {error}", flush=True)
            continue
        fps_source = float(reader.get_avg_fps())
        indices = frame_indices_for_window(
            float(source["window_start_sec"]), int(source["num_frames"]),
            int(source["fps"]), fps_source, len(reader),
        )
        batch = reader.get_batch(indices).asnumpy()
        shared = grabber.grab(reference["bucket"], reference["key"], float(reference["abs_time"]))

        # fresh reader, nothing read before the seek
        url = grabber.presigned_url(reference["bucket"], reference["key"])
        fresh_reader = VideoReader(url)
        frame_no = min(int(round(float(reference["abs_time"]) * fresh_reader.get_avg_fps())),
                       len(fresh_reader) - 1)
        fresh = fresh_reader[frame_no].asnumpy()

        identical = shared.shape == fresh.shape and bool(np.array_equal(shared, fresh))
        difference = (
            float(np.abs(shared.astype(np.int32) - fresh.astype(np.int32)).mean())
            if shared.shape == fresh.shape else float("nan")
        )
        print(
            f"{spec['sample_id']} same_file={same_file} batch={batch.shape} "
            f"ref_frame_no={frame_no} identical={identical} mean_abs_diff={difference:.4f}",
            flush=True,
        )
        if not identical:
            failures += 1
        checked += 1

        # also confirm the batch itself is stable: re-read one index from a fresh reader
        probe = len(indices) // 2
        fresh_target = VideoReader(grabber.presigned_url(source["bucket"], source["key"]))
        one = fresh_target[indices[probe]].asnumpy()
        target_identical = bool(np.array_equal(batch[probe], one))
        target_difference = float(
            np.abs(batch[probe].astype(np.int32) - one.astype(np.int32)).mean()
        )
        print(
            f"  target frame {probe} (src idx {indices[probe]}): identical={target_identical} "
            f"mean_abs_diff={target_difference:.4f}",
            flush=True,
        )
        if not target_identical:
            failures += 1
    print(f"checked={checked} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
