"""Stage B: sample specs -> 81-frame mp4 clips + reference jpgs.

Decoding is done with decord straight off presigned BOS URLs (no download of the full
source video, and no ffmpeg CLI: the static ffmpeg binary shipped by imageio-ffmpeg
segfaults on https presigned URLs). Artifacts go through a storage backend so the
same code can later write to BOS instead of the local data disk.

    python -m phantom_data.build.extract --specs specs.jsonl --dataset <root> --workers 4

Resumability: one marker per sample under ``<root>/_stages/extract/``, reusing
``ultravid_pipeline.state.MarkerStore``. Samples with a marker are skipped.

**Clips are stored downscaled** to ``--target-height`` (default 480), isotropically and
uncropped, since training only ever consumes 832x480 and source-resolution storage was ~99%
of a projected ~1.1 TiB at 126k samples. All of that geometry -- and the reason the stored
size is *not* 832x480 -- lives in :mod:`phantom_data.build.resolution`. ``--target-height 0``
disables scaling and reproduces the existing pilot data.

Reference jpgs are **not** scaled: they are the source of the identity signal (stage C mattes
its cutout out of them and stage D's CLIP score reads that cutout), and they are 45 MiB of the
pilot's 1.25 GiB, so there are no meaningful bytes to win and real quality to lose. Their own
``ref.ref_frame_width/height`` stay the decoded jpg's dimensions, independent of the clip's.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

from ultravid_pipeline.state import MarkerStore, atomic_write_json

from ..bos import FrameGrabber, load_aksk, make_client
from .resolution import DEFAULT_TARGET_HEIGHT, DEFAULT_TARGET_WIDTH, storage_plan
from .storage import StorageBackend, make_storage

STAGE = "extract"
MAX_CACHED_READERS = 2

#: Wall-clock budget for one sample's decode + encode. A healthy sample takes ~5 s; the
#: cap exists because decord's threaded decoder can wedge permanently on a malformed BOS
#: source (observed: one sample pinned a worker for 65 min at 0% CPU and 38 GB RSS, with no
#: open socket, while the other workers had long finished). At 500k samples an unbounded
#: wait is a guaranteed stall, so a hung sample is recorded as failed and the run moves on.
DEFAULT_SAMPLE_TIMEOUT = 300.0


def frame_indices_for_window(
    window_start_sec: float,
    num_frames: int,
    fps: int,
    fps_source: float,
    total_frames: int,
) -> list[int]:
    """Source-video frame numbers for the ``num_frames`` samples of the target window."""
    indices = []
    for offset in range(num_frames):
        time_sec = window_start_sec + offset / float(fps)
        indices.append(min(int(round(time_sec * fps_source)), total_frames - 1))
    return indices


def resize_frames(frames, width: int, height: int) -> list:
    """Downscale decoded frames to ``(width, height)``, or return them untouched.

    PIL's ``LANCZOS``, not nearest and not bilinear. This is a ~0.44x downscale, where the
    filter choice is not cosmetic: nearest point-samples and aliases every high-frequency edge
    (and the boxes are drawn around *edges*), while bilinear's 2x2 support cannot cover a 2.25px
    footprint and softens the result. Lanczos is the standard antialiased choice and PIL applies
    it as a proper resampling kernel over the full footprint.

    **Not cv2.** ``cv2.INTER_AREA`` would be an equally defensible filter, but this image has no
    working OpenCV video/encode path at all (``cv2.VideoWriter.isOpened()`` is False for every
    fourcc), so the encode side is already imageio/PIL; keeping the resize in the same library
    avoids adding a second imaging dependency to the hot loop for no gain.

    Identity short-circuit rather than a no-op resize: an unscaled run must produce the same
    array objects it always did, so ``--target-height 0`` is byte-identical and not merely
    equivalent-looking.
    """
    import numpy as np
    from PIL import Image

    if not len(frames):
        return list(frames)
    if (int(frames[0].shape[1]), int(frames[0].shape[0])) == (int(width), int(height)):
        return list(frames)
    return [
        np.asarray(
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
                (int(width), int(height)), resample=Image.LANCZOS
            )
        )
        for frame in frames
    ]


def encode_mp4(frames, fps: int) -> bytes:
    """H.264 mp4 bytes at whatever resolution ``frames`` already are.

    The imageio ffmpeg plugin muxes through a real file (it cannot target a BytesIO), so
    encoding goes to a scratch file that is read back and handed to the storage backend.
    Keeping bytes as the backend currency is what lets the BOS backend drop in later.

    Scaling happens in :func:`resize_frames` before this call, deliberately *not* here via
    ffmpeg's ``-s``: ``macro_block_size=2`` means the writer would round an odd dimension up on
    its own and the manifest would then record a size the file does not have. Committing to the
    exact stored dimensions in numpy, in one place, is what keeps pixels and metadata agreeing.
    """
    import imageio.v2 as imageio

    handle, temporary = tempfile.mkstemp(prefix="phantom_clip.", suffix=".mp4")
    os.close(handle)
    try:
        writer = imageio.get_writer(
            temporary, format="ffmpeg", fps=fps, codec="libx264", quality=8, macro_block_size=2
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return Path(temporary).read_bytes()
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def encode_jpeg(frame, quality: int = 95) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def clip_relpath(sample_id: str) -> str:
    return f"clips/{sample_id}.mp4"


def ref_relpath(sample_id: str, subject_id: int) -> str:
    return f"ref_frames/{sample_id}_subj{subject_id:02d}.jpg"


def extract_sample(
    spec: dict[str, Any],
    grabber: FrameGrabber,
    storage: StorageBackend,
    jpeg_quality: int = 95,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    target_width: int = DEFAULT_TARGET_WIDTH,
) -> dict[str, Any]:
    """Decode + persist one sample; returns the ``extracted.jsonl`` row.

    The row's ``width``/``height`` are the **stored** clip dimensions, and every downstream
    consumer derives box coordinates from them (``segment.scale_bbox_to_frame`` /
    ``resolve_box``), so they are the one field that must agree with the encoded pixels. The
    subjects' raw annotation boxes are passed through untouched -- see
    :func:`resolution.scale_box` for why pre-scaling them would double-apply the shrink.
    """
    sample_id = spec["sample_id"]
    source = spec["source"]
    num_frames, fps = int(source["num_frames"]), int(source["fps"])

    reader = grabber.reader(source["bucket"], source["key"])
    fps_source = float(reader.get_avg_fps())
    if not fps_source or fps_source != fps_source:  # 0 or NaN
        raise ValueError(f"bad source fps {fps_source!r} for {source['key']}")
    total = len(reader)
    indices = frame_indices_for_window(
        float(source["window_start_sec"]), num_frames, fps, fps_source, total
    )
    batch = reader.get_batch(indices).asnumpy()
    if batch.shape[0] != num_frames:
        raise ValueError(f"decoded {batch.shape[0]} frames, expected {num_frames}")
    source_height, source_width = int(batch.shape[1]), int(batch.shape[2])
    plan = storage_plan(source_width, source_height, target_height, target_width)
    width, height = int(plan["width"]), int(plan["height"])

    video_path = clip_relpath(sample_id)
    clip_bytes = encode_mp4(resize_frames(list(batch), width, height), fps)
    storage.write_bytes(video_path, clip_bytes)

    subjects: list[dict[str, Any]] = []
    ref_bytes = 0
    for subject in spec["subjects"]:
        reference = subject["ref"]
        frame = grabber.grab(reference["bucket"], reference["key"], float(reference["abs_time"]))
        relative = ref_relpath(sample_id, int(subject["subject_id"]))
        jpeg = encode_jpeg(frame, quality=jpeg_quality)
        ref_bytes += len(jpeg)
        storage.write_bytes(relative, jpeg)
        # ``ref.frame`` is the RAW (un-matted) reference frame. Deliberately NOT named
        # ``object_reference``: in the UltraVid schema that key means the white-background
        # cutout, which is a downstream (segmentation) product, not this.
        subjects.append(
            {
                **subject,
                "ref": {
                    **reference,
                    "frame": relative,
                    "ref_frame_width": int(frame.shape[1]),
                    "ref_frame_height": int(frame.shape[0]),
                },
            }
        )

    return {
        "sample_id": sample_id,
        "video_id": spec["video_id"],
        "phantom_video_id": spec["phantom_video_id"],
        "video": video_path,
        "caption": spec["caption"],
        "prompt": spec["caption"],
        # STORED dimensions, which on a scaled run are not the source's. Everything downstream
        # that turns an annotation box into pixels reads these two numbers.
        "width": width,
        "height": height,
        "frame_count": int(batch.shape[0]),
        "fps": fps,
        "source": {**source, "fps_source": fps_source, "source_total_frames": total},
        # The geometry record: source dims, target, achieved scale, and the fraction of each
        # axis training's centre crop will discard (with ``crop_discard_excessive`` as the
        # counted flag). Additive metadata -- no existing consumer reads it -- but it is what
        # makes a stored clip interpretable and what puts portrait sources in the funnel.
        "storage_geometry": plan,
        #: Encoded clip size, so the BOS cost model runs on measured bytes rather than on an
        #: average of the pilot. Excludes the reference jpgs, which are per subject.
        "clip_bytes": len(clip_bytes),
        #: All reference jpgs of this sample summed. Kept separate from ``clip_bytes`` because
        #: the two scale differently with the dataset: one clip per sample, but 1-6 refs.
        "ref_bytes": ref_bytes,
        "subjects": subjects,
        "dropped_subjects": spec.get("dropped_subjects") or [],
    }


def read_specs(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(row)
    return rows


def run(
    specs_path: Path,
    dataset: Path,
    workers: int = 4,
    storage_kind: str = "local",
    limit: int | None = None,
    force: bool = False,
    sample_timeout: float = DEFAULT_SAMPLE_TIMEOUT,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    target_width: int = DEFAULT_TARGET_WIDTH,
) -> dict[str, Any]:
    specs = read_specs(specs_path)
    if limit:
        specs = specs[:limit]
    dataset.mkdir(parents=True, exist_ok=True)
    storage = make_storage(storage_kind, dataset)
    markers = MarkerStore(dataset)

    pending = [
        spec for spec in specs if force or markers.get(STAGE, spec["sample_id"]) is None
    ]
    print(f"{len(specs)} specs, {len(specs) - len(pending)} already done, {len(pending)} to do", flush=True)

    local = threading.local()
    lock = threading.Lock()
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    results: dict[str, dict[str, Any]] = {}
    # Rows are appended here the moment a sample finishes. The final extracted.jsonl is
    # still rewritten at the end, but a run killed mid-flight (decord can hang forever on
    # a bad BOS read) leaves its completed rows recoverable: markers alone cannot rebuild
    # them, because a marker records only shape, not the per-subject boxes and ref paths.
    partial_path = dataset / f"{STAGE}.partial.jsonl"
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_handle = open(partial_path, "a", encoding="utf-8")

    ak, sk = load_aksk()

    def grabber_for_thread() -> FrameGrabber:
        # decord readers are not thread safe: one grabber (and reader cache) per worker.
        if getattr(local, "grabber", None) is None:
            local.grabber = FrameGrabber(make_client(ak, sk))
        grabber = local.grabber
        # Within a sample, target and ref usually share the source file, so the reader
        # cache is worth keeping; across samples it would pin one 1080p reader per source.
        if len(grabber._reader_cache) > MAX_CACHED_READERS:
            grabber._reader_cache.clear()
        return grabber

    def work(spec: dict[str, Any]) -> None:
        sample_id = spec["sample_id"]
        started = time.time()
        try:
            row = extract_sample(spec, grabber_for_thread(), storage,
                                 target_height=target_height, target_width=target_width)
        except Exception as error:  # noqa: BLE001 - failures are recorded, never silent
            detail = f"{type(error).__name__}: {error}"
            markers.put(STAGE, sample_id, {"status": "failed", "error": detail})
            with lock:
                counts["failed"] += 1
                failures.append({"sample_id": sample_id, "error": detail})
            print(f"FAILED {sample_id}: {detail}", flush=True)
            return
        elapsed = round(time.time() - started, 3)
        markers.put(
            STAGE,
            sample_id,
            {
                "status": "passed",
                "video": row["video"],
                "width": row["width"],
                "height": row["height"],
                "frame_count": row["frame_count"],
                "subjects": len(row["subjects"]),
                # Bytes and geometry live in the marker too, not only in the manifest row:
                # ``tools/recover_extract_manifest.py`` rebuilds rows from markers alone when a
                # run's partial log is gone, and a rebuilt row that lost its stored dimensions
                # would be a row whose boxes cannot be resolved.
                "clip_bytes": row["clip_bytes"],
                "ref_bytes": row["ref_bytes"],
                "storage_geometry": row["storage_geometry"],
                "seconds": elapsed,
            },
        )
        with lock:
            counts["passed"] += 1
            results[sample_id] = row
            done = counts["passed"] + counts["failed"]
            # fsync so a SIGKILL (or an OOM kill) cannot lose an already-finished sample
            partial_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            partial_handle.flush()
            os.fsync(partial_handle.fileno())
        geometry = row["storage_geometry"]
        print(
            f"[{done}/{len(pending)}] {sample_id} "
            f"{geometry['source_width']}x{geometry['source_height']}->"
            f"{row['width']}x{row['height']} {row['frame_count']}f "
            f"{row['clip_bytes'] / 1048576:.2f}MiB subj={len(row['subjects'])} {elapsed}s"
            + ("  CROP-DISCARD" if geometry["crop_discard_excessive"] else ""),
            flush=True,
        )

    started = time.time()
    abandoned: list[str] = []
    if pending:
        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        futures = {pool.submit(work, spec): spec["sample_id"] for spec in pending}
        for future in futures:
            try:
                future.result(timeout=sample_timeout)
            except FuturesTimeout:
                # A wedged decord decode cannot be interrupted from Python -- it is blocked
                # inside a C thread, so neither cancel() nor a signal reaches it. Record it
                # and move on; main() exits with os._exit so the dead thread cannot hold
                # the interpreter open at join time.
                sample_id = futures[future]
                detail = (f"TimeoutError: exceeded {sample_timeout:g}s "
                          f"(worker wedged, abandoned)")
                markers.put(STAGE, sample_id, {"status": "failed", "error": detail})
                with lock:
                    counts["failed"] += 1
                    failures.append({"sample_id": sample_id, "error": detail})
                abandoned.append(sample_id)
                print(f"ABANDONED {sample_id}: {detail}", flush=True)
        pool.shutdown(wait=False)
    wall = round(time.time() - started, 2)

    partial_handle.close()

    # extracted.jsonl is rebuilt from the union of this run and any previous run's rows,
    # so a resumed run never loses earlier samples. The partial log is folded in too, which
    # is what makes a killed run recoverable.
    manifest = dataset / "extracted.jsonl"
    merged: dict[str, dict[str, Any]] = {}
    if manifest.is_file():
        for row in read_specs(manifest):
            merged[row["sample_id"]] = row
    if partial_path.is_file():
        for row in read_specs(partial_path):
            merged[row["sample_id"]] = row
    merged.update(results)
    ordered = [merged[spec["sample_id"]] for spec in specs if spec["sample_id"] in merged]
    ordered += [row for key, row in merged.items() if key not in {s["sample_id"] for s in specs}]
    tmp = manifest.with_suffix(".jsonl.tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8"
    )
    tmp.replace(manifest)

    # Byte and geometry aggregates over *this run's* rows only, not over the merged manifest:
    # a resumed run's older rows may have been produced at a different --target-height, and
    # averaging across two geometries would give a cost-per-sample that describes neither.
    sizes = sorted(int(row["clip_bytes"]) for row in results.values() if "clip_bytes" in row)
    flagged = [sample for sample, row in results.items()
               if (row.get("storage_geometry") or {}).get("crop_discard_excessive")]
    resolutions = Counter(f"{row['width']}x{row['height']}" for row in results.values())
    summary = {
        "specs": len(specs),
        "attempted": len(pending),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": len(specs) - len(pending),
        "manifest_rows": len(ordered),
        "abandoned": abandoned,
        "seconds": wall,
        "storage": storage.root_uri,
        "target_height": target_height,
        "target_width": target_width,
        "stored_resolutions": dict(resolutions.most_common()),
        # The numbers the BOS cost model needs. Median as well as mean, because the mean of a
        # few hundred clips is dragged by the long tail of high-motion scenes.
        "clip_bytes_total": sum(sizes),
        "clip_bytes_mean": int(sum(sizes) / len(sizes)) if sizes else 0,
        "clip_bytes_median": sizes[len(sizes) // 2] if sizes else 0,
        "ref_bytes_total": sum(int(row.get("ref_bytes") or 0) for row in results.values()),
        # Flagged, counted, and named -- never dropped here. These are the samples where
        # training's centre crop discards >20% of an axis (portrait sources, extreme 4:3); the
        # decision to keep or drop them is the user's and belongs in the funnel, not in stage B.
        "crop_discard_excessive": len(flagged),
        "crop_discard_excessive_samples": sorted(flagged)[:50],
        "failures": failures[:50],
    }
    atomic_write_json(dataset / "_stages" / f"{STAGE}.summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "failures"}, indent=2))
    if failures:
        print(f"{len(failures)} FAILURES:", flush=True)
        for item in failures:
            print(f"  {item['sample_id']}: {item['error']}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract Phantom-Koala clips and ref frames")
    parser.add_argument("--specs", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workers", type=int, default=4)
    # ``bos`` targets PHANTOM_BOS_BUCKET/PHANTOM_BOS_PREFIX (default vast-yz/koala-ref-n-box);
    # --dataset is still required either way, for markers and the merged manifest.
    parser.add_argument("--storage", default="local", choices=["local", "bos"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="re-extract samples that have markers")
    parser.add_argument("--sample-timeout", type=float, default=DEFAULT_SAMPLE_TIMEOUT,
                        help="per-sample wall-clock budget in seconds (healthy: ~5s)")
    # Height is the scaling anchor; width is only used to predict training's centre crop, so
    # changing it changes the recorded diagnostic and nothing about the stored pixels.
    parser.add_argument("--target-height", type=int, default=DEFAULT_TARGET_HEIGHT,
                        help="store clips isotropically scaled to this height, uncropped "
                             "(0 disables scaling and stores source resolution, which is what "
                             "the existing pilot data was built with)")
    parser.add_argument("--target-width", type=int, default=DEFAULT_TARGET_WIDTH,
                        help="training's frame width; used only for the crop-loss diagnostic")
    args = parser.parse_args(argv)
    summary = run(
        specs_path=Path(args.specs),
        dataset=Path(args.dataset),
        workers=args.workers,
        storage_kind=args.storage,
        limit=args.limit or None,
        force=args.force,
        sample_timeout=args.sample_timeout,
        target_height=args.target_height,
        target_width=args.target_width,
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    code = main()
    # os._exit, not SystemExit: an abandoned sample leaves a decord thread wedged in C, and
    # a normal exit would block forever joining it. Everything is already fsynced by here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
