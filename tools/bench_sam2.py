"""Benchmark stage C (SAM2 masklets) at 832x480 to size a 100k-scale run.

Answers two questions the pilot's 10.15 s/sample cannot:

1. **How many samples fit in one A800-hour.** The pilot ran one stream at 1080p. SAM2's
   hiera-l config pins ``image_size: 1024``, so every frame is letterboxed to 1024x1024
   inside the model regardless of the source resolution -- downscaling to 832x480 buys
   decode time, VRAM, and disk, but *not* encoder time. Single-stream latency is therefore
   expected to stay near the pilot number, and the only lever on throughput is running
   several independent streams over the same GPU. This script sweeps that lever.

2. **What a sample costs on disk** at 832x480: the encoded mp4 plus the packed mask npz,
   against the 1080p originals.

Concurrency is **separate OS processes**, never threads: SAM2's inference state is mutated
in place and decord's readers are documented as not thread safe, so a threaded sweep would
measure a race rather than a throughput ceiling. Each worker builds its own predictor and
takes a disjoint slice of the sample list.

The timed region is production stage C's code path, imported rather than reimplemented
(``segment.decode_frames`` / ``propagate_bidirectional`` / ``pack_masks``), so the numbers
transfer. Two deliberate exclusions, reported separately rather than folded in:

* **mp4 downscaling is prep, not stage C.** In production the 832x480 clip is what stage B
  writes, so stage C reads it off disk. It is built once up front here (and its bytes
  recorded) instead of being charged to every concurrency level.
* **reference cutouts + CLIP scoring** are timed under ``ref_seg_sec`` but kept out of the
  masklet throughput number, because the reference frame is a single 1080p image whose cost
  does not scale with clip length.

Boxes come from the Grounding-DINO-corrected gate report, seed side, restricted to the
subjects :func:`redetect.decide` keeps -- the population production would actually segment.
Seeding SAM2 with Phantom's raw (often skewed) boxes, as the pilot did, changes how much
of the frame the tracker has to explain and is one of the three variables being fixed here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

DATASET = Path("/mnt/pfs/data/yuanze/phantom_koala_inspect100_v1")
GATE_REPORT = DATASET / "_redetect100" / "gate_report.json"
#: Target geometry: 81 frames at 16 fps is 5.0625 s, the training clip length.
TARGET_WIDTH, TARGET_HEIGHT = 832, 480
TARGET_FPS = 16
#: Pod-local scratch. Never the shared FS: the point of the exercise is to measure GPU
#: throughput, and juicefs latency would show up as SAM2 being slow.
SCRATCH = Path(os.environ.get("BENCH_SCRATCH", "/tmp/sam2_bench"))
DEFAULT_LEVELS = (1, 2, 4, 6)
#: nvidia-smi poll period. Fast enough to catch the propagation peak, slow enough that the
#: sampler itself is free.
GPU_POLL_SEC = 2.0


# --------------------------------------------------------------------------------------
# sample selection
# --------------------------------------------------------------------------------------


def gate_passing_subjects(report_path: Path = GATE_REPORT,
                          rule: str | None = None) -> list[dict[str, Any]]:
    """Seed-side boxes for the subjects the gate keeps, in report order.

    ``chosen_box_seed`` is the box :func:`redetect.pick_side` settled on for the target
    frame -- already in that frame's pixel coordinates, so no canvas mapping is needed
    (unlike ``seed_bbox_768`` in stage B's manifest, which is raw annotation space).

    The verdict is recomputed from the flat scores rather than read off ``verdict``, which
    exposes a discrepancy worth being explicit about: ``redetect.DEFAULT_RULE`` is
    ``iou_stands`` (keeps 134/140 here), but the ``verdict`` field stored in this report was
    written under ``identity_required`` (78/140). ``rule`` defaults to the module default so
    "the default gate" means what :func:`redetect.decide` means by it; pass
    ``identity_required`` to reproduce the stored verdicts. Which one is right is a data
    question, not a throughput one -- see the ``gate_rule_sensitivity`` note in the output.
    """
    from phantom_data import redetect

    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for record in report["subjects"]:
        ruling = redetect.decide(
            record,
            rule=rule or redetect.DEFAULT_RULE,
            identity_min=redetect.IDENTITY_MIN,
            clip_min=redetect.CLIP_MIN,
            iou_min=redetect.IOU_MIN,
        )
        if ruling["verdict"] != redetect.KEEP:
            continue
        box = record.get("chosen_box_seed")
        if not box:
            continue
        selected.append(
            {
                "sample_id": record["sample_id"],
                "subject_id": int(record["subject_id"]),
                "seed_frame_index": int(record["seed_frame_index"]),
                "box_seed_1080p": [float(v) for v in box],
                "box_source": record.get("pick_seed"),
                "phrase": record.get("phrase"),
                "ref_frame": record.get("ref_frame"),
                "chosen_box_ref": record.get("chosen_box_ref"),
            }
        )
    return selected


# --------------------------------------------------------------------------------------
# prep: 832x480 clips in pod-local scratch
# --------------------------------------------------------------------------------------


def scale_box(box: list[float], src_w: int, src_h: int,
              dst_w: int = TARGET_WIDTH, dst_h: int = TARGET_HEIGHT) -> list[float]:
    """Map a box through an anisotropic resize, then clamp into the target frame.

    The axes are scaled independently because the resize is: 1920x1080 is 16:9 and
    832x480 is 1.733:1, so a single factor would leave the box off the subject vertically.
    """
    sx, sy = dst_w / float(src_w), dst_h / float(src_h)
    x1, y1, x2, y2 = box
    out = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    out[0], out[2] = sorted((out[0], out[2]))
    out[1], out[3] = sorted((out[1], out[3]))
    return [
        float(min(max(out[0], 0.0), dst_w)),
        float(min(max(out[1], 0.0), dst_h)),
        float(min(max(out[2], 0.0), dst_w)),
        float(min(max(out[3], 0.0), dst_h)),
    ]


def downscale_clip(source: Path, target: Path, width: int = TARGET_WIDTH,
                   height: int = TARGET_HEIGHT, fps: int = TARGET_FPS) -> dict[str, Any]:
    """Re-encode ``source`` at ``width x height`` and report both files' sizes.

    ``cv2.VideoWriter`` is unusable in this image -- it was built without an encode
    backend, so ``isOpened()`` is False for every fourcc including mp4v and avc1, silently
    producing 0-byte files. Encoding therefore goes through imageio-ffmpeg with the same
    settings ``build/extract.py`` writes the pilot clips with (libx264, quality=8,
    macro_block_size=2), which keeps the measured bytes comparable to what stage B would
    actually store.
    """
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(str(source))
    frames: list[np.ndarray] = []
    try:
        for frame in reader:
            rgb = np.asarray(frame)[..., :3]
            if not frames:
                source_shape = rgb.shape[:2]
            frames.append(
                np.asarray(Image.fromarray(rgb).resize((width, height), Image.BILINEAR))
            )
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"decoded 0 frames from {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(target), format="ffmpeg", fps=fps, codec="libx264",
        quality=8, macro_block_size=2,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return {
        "source_height": int(source_shape[0]),
        "source_width": int(source_shape[1]),
        "frame_count": len(frames),
        "mp4_bytes_1080p": source.stat().st_size,
        "mp4_bytes_832x480": target.stat().st_size,
    }


def prepare(subjects: list[dict[str, Any]], scratch: Path) -> list[dict[str, Any]]:
    """Build every 832x480 clip once and scale each seed box to match.

    Done up front, single stream, because it is stage B's work: charging it to each
    concurrency level would inflate every sec/sample by the same constant and hide the
    thing being measured.
    """
    clips = scratch / "clips832"
    clips.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
    for position, subject in enumerate(subjects, 1):
        source = DATASET / "clips" / f"{subject['sample_id']}.mp4"
        if not source.is_file():
            print(f"  skip {subject['sample_id']}: clip missing", flush=True)
            continue
        target = clips / f"{subject['sample_id']}.mp4"
        started = time.time()
        try:
            info = downscale_clip(source, target)
        except Exception as error:  # noqa: BLE001 - a bad clip must not stop prep
            print(f"  FAILED prep {subject['sample_id']}: "
                  f"{type(error).__name__}: {error}", flush=True)
            continue
        box = scale_box(subject["box_seed_1080p"], info["source_width"], info["source_height"])
        prepared.append({**subject, **info, "clip832": str(target),
                         "box_seed_832x480": [round(v, 2) for v in box],
                         "prep_sec": round(time.time() - started, 3)})
        print(f"  [{position}/{len(subjects)}] {subject['sample_id'][:16]} "
              f"{info['source_width']}x{info['source_height']} -> {TARGET_WIDTH}x{TARGET_HEIGHT} "
              f"{info['mp4_bytes_1080p'] / 1e6:.2f}MB -> "
              f"{info['mp4_bytes_832x480'] / 1e6:.2f}MB {prepared[-1]['prep_sec']}s", flush=True)
    return prepared


# --------------------------------------------------------------------------------------
# worker: one process, one SAM2, a disjoint slice
# --------------------------------------------------------------------------------------


def run_worker(manifest_path: Path, output_path: Path, do_ref: bool = True) -> int:
    """Segment every sample in ``manifest_path``, writing per-sample timings as JSONL.

    Kept as a subcommand of this same file rather than a separate module so the worker and
    the orchestrator cannot drift apart.
    """
    from phantom_data.build import segment

    samples = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = segment.Models(
        segment.DEFAULT_SAM2_CONFIG, segment.DEFAULT_SAM2_CHECKPOINT,
        segment.DEFAULT_CLIP_MODEL, device="cuda",
    )
    load_started = time.time()
    predictor = models.video  # force the weight load out of the first sample's timing
    load_sec = round(time.time() - load_started, 3)

    handle = output_path.open("w", encoding="utf-8")
    handle.write(json.dumps({"event": "ready", "model_load_sec": load_sec,
                             "pid": os.getpid(), "samples": len(samples)}) + "\n")
    handle.flush()

    for position, sample in enumerate(samples, 1):
        record: dict[str, Any] = {"event": "sample", "sample_id": sample["sample_id"],
                                  "subject_id": sample["subject_id"], "pid": os.getpid()}
        started = time.time()
        try:
            with tempfile.TemporaryDirectory(prefix="bench-sam2-", dir=str(SCRATCH)) as directory:
                frame_dir = Path(directory)
                clock = time.time()
                frames = segment.decode_frames(Path(sample["clip832"]), frame_dir)
                record["decode_sec"] = round(time.time() - clock, 3)
                height, width = frames[0].shape[:2]

                clock = time.time()
                state = predictor.init_state(video_path=str(frame_dir))
                record["init_state_sec"] = round(time.time() - clock, 3)

                clock = time.time()
                masks, diagnostic = segment.propagate_bidirectional(
                    predictor, state, sample["seed_frame_index"],
                    np.asarray(sample["box_seed_832x480"], dtype=np.float32),
                    len(frames), device="cuda",
                )
                record["propagate_sec"] = round(time.time() - clock, 3)
                record["propagation"] = diagnostic
                record.update(segment.mask_stats(masks))

                clock = time.time()
                packed = segment.pack_masks(masks)
                boxes = [segment.bbox_from_mask(mask) for mask in masks]
                npz_path = SCRATCH / "npz" / f"{sample['sample_id']}.npz"
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    npz_path,
                    subject_masks_packed=np.asarray([packed], dtype=np.uint8),
                    source_subject_ids=np.asarray([sample["subject_id"]], dtype=np.int64),
                    mask_width=np.asarray(width),
                    mask_format_version=np.asarray(2),
                )
                record["pack_sec"] = round(time.time() - clock, 3)
                record["npz_bytes_832x480"] = npz_path.stat().st_size
                record["mask_shape"] = [len(frames), height, width]
                record["boxes_written"] = sum(1 for box in boxes if box is not None)

                if do_ref and sample.get("ref_frame") and sample.get("chosen_box_ref"):
                    from PIL import Image

                    ref_path = DATASET / "_redetect100" / sample["ref_frame"]
                    if ref_path.is_file():
                        clock = time.time()
                        ref_image = np.asarray(Image.open(ref_path).convert("RGB"))
                        ref_mask = segment.segment_reference(
                            models, ref_image, sample["chosen_box_ref"], device="cuda")
                        record["ref_seg_sec"] = round(time.time() - clock, 3)
                        record["ref_mask_pixels"] = int(np.count_nonzero(ref_mask))

            record["total_sec"] = round(time.time() - started, 3)
            record["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - one bad sample must not kill the level
            import traceback

            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
            record["total_sec"] = round(time.time() - started, 3)

        handle.write(json.dumps(record) + "\n")
        handle.flush()
        print(f"[pid {os.getpid()}] [{position}/{len(samples)}] {sample['sample_id'][:16]} "
              f"{record['status']} {record['total_sec']}s", flush=True)

    handle.write(json.dumps({"event": "done"}) + "\n")
    handle.close()
    return 0


# --------------------------------------------------------------------------------------
# GPU sampling
# --------------------------------------------------------------------------------------


class GpuSampler:
    """Background ``nvidia-smi`` poller, one sample per :data:`GPU_POLL_SEC`.

    Utilisation is read rather than inferred: a level whose mean utilisation is already
    near 100% cannot be sped up by adding streams, which is the whole question here. VRAM
    is read as the ceiling on how many streams fit at all.
    """

    def __init__(self, period: float = GPU_POLL_SEC) -> None:
        self.period = period
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                raw = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20, check=True,
                ).stdout.strip().splitlines()
                # Index 0: the pod sees exactly the GPU it was allocated.
                utilisation, memory = (part.strip() for part in raw[0].split(","))
                self.samples.append((float(utilisation), float(memory)))
            except Exception as error:  # noqa: BLE001 - a lost sample is not a failed run
                if len(self.errors) < 5:
                    self.errors.append(f"{type(error).__name__}: {error}")
            self._stop.wait(self.period)

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.period * 3)
        if not self.samples:
            return {"gpu_samples": 0, "mean_gpu_util_pct": None, "peak_vram_mib": None,
                    "sampler_errors": self.errors}
        utilisations = [value for value, _ in self.samples]
        memories = [value for _, value in self.samples]
        return {
            "gpu_samples": len(self.samples),
            "mean_gpu_util_pct": round(sum(utilisations) / len(utilisations), 1),
            "median_gpu_util_pct": round(float(np.median(utilisations)), 1),
            "min_gpu_util_pct": round(min(utilisations), 1),
            "peak_vram_mib": int(max(memories)),
            "mean_vram_mib": int(sum(memories) / len(memories)),
            "sampler_errors": self.errors,
        }


# --------------------------------------------------------------------------------------
# one concurrency level
# --------------------------------------------------------------------------------------


def slices(items: list[Any], parts: int) -> list[list[Any]]:
    """Round-robin split, so a slow sample cannot land all in one worker's share."""
    out: list[list[Any]] = [[] for _ in range(parts)]
    for position, item in enumerate(items):
        out[position % parts].append(item)
    return out


def read_worker_output(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse a worker's JSONL, tolerating a truncated final line from a killed process."""
    records: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    if not path.is_file():
        return records, meta
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "sample":
            records.append(payload)
        elif payload.get("event") == "ready":
            meta = payload
    return records, meta


def run_level(concurrency: int, samples: list[dict[str, Any]], scratch: Path,
              do_ref: bool = True) -> dict[str, Any]:
    """Launch ``concurrency`` worker processes over a disjoint split and time the whole set.

    Wall time is measured from after every worker reports ``ready`` -- i.e. once the ~0.9 GB
    checkpoint is on the GPU -- to the last worker exiting. Model load is a fixed startup
    cost that a 100k-sample run pays once per worker, not once per sample, so leaving it in
    would understate throughput by more at high concurrency (where each worker has fewer
    samples to amortise it over) and invert the comparison.
    """
    level_dir = scratch / f"level{concurrency}"
    level_dir.mkdir(parents=True, exist_ok=True)
    shards = slices(samples, concurrency)

    processes = []
    outputs = []
    for index, shard in enumerate(shards):
        manifest = level_dir / f"shard{index}.json"
        manifest.write_text(json.dumps(shard), encoding="utf-8")
        output = level_dir / f"shard{index}.jsonl"
        outputs.append(output)
        log = (level_dir / f"shard{index}.log").open("w")
        command = [sys.executable, os.path.abspath(__file__), "worker",
                   "--manifest", str(manifest), "--output", str(output)]
        if not do_ref:
            command.append("--no-ref")
        processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))

    # Wait for every worker to finish loading before starting the clock.
    load_deadline = time.time() + 900
    ready = [False] * concurrency
    while not all(ready) and time.time() < load_deadline:
        for index, output in enumerate(outputs):
            if not ready[index]:
                _records, meta = read_worker_output(output)
                ready[index] = bool(meta)
        if all(ready):
            break
        if any(process.poll() is not None for process, _ in processes):
            break  # a worker died during load; fall through and report it
        time.sleep(1.0)

    sampler = GpuSampler()
    sampler.start()
    started = time.time()
    exit_codes = []
    for process, log in processes:
        exit_codes.append(process.wait())
        log.close()
    wall = time.time() - started
    gpu = sampler.stop()

    records: list[dict[str, Any]] = []
    load_times: list[float] = []
    for output in outputs:
        shard_records, meta = read_worker_output(output)
        records.extend(shard_records)
        if meta.get("model_load_sec") is not None:
            load_times.append(meta["model_load_sec"])

    ok = [record for record in records if record.get("status") == "ok"]
    failed = [record for record in records if record.get("status") != "ok"]

    def mean(key: str) -> float | None:
        values = [record[key] for record in ok if record.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    result: dict[str, Any] = {
        "concurrency": concurrency,
        "samples_submitted": len(samples),
        "samples_processed": len(ok),
        "samples_failed": len(failed),
        "wall_sec": round(wall, 2),
        "sec_per_sample": round(wall / len(ok), 3) if ok else None,
        "samples_per_gpu_hour": round(3600.0 * len(ok) / wall, 1) if ok and wall else None,
        "mean_latency_sec": mean("total_sec"),
        "mean_decode_sec": mean("decode_sec"),
        "mean_init_state_sec": mean("init_state_sec"),
        "mean_propagate_sec": mean("propagate_sec"),
        "mean_pack_sec": mean("pack_sec"),
        "mean_ref_seg_sec": mean("ref_seg_sec"),
        "mean_model_load_sec": round(sum(load_times) / len(load_times), 2) if load_times else None,
        "worker_exit_codes": exit_codes,
        **gpu,
    }
    if failed:
        result["failures"] = [
            {"sample_id": record.get("sample_id"), "error": record.get("error")}
            for record in failed[:10]
        ]
    # A worker killed by the OOM killer or by CUDA OOM leaves no per-sample record for the
    # sample it died on, so a nonzero exit with fewer records than submitted is reported as
    # a level-wide failure rather than quietly shrinking the denominator.
    if any(code != 0 for code in exit_codes):
        result["worker_failure"] = True
        result["worker_logs"] = {
            f"shard{index}": (level_dir / f"shard{index}.log").read_text(
                encoding="utf-8", errors="replace")[-3000:]
            for index, code in enumerate(exit_codes) if code != 0
        }
    return result, records


# --------------------------------------------------------------------------------------
# bytes
# --------------------------------------------------------------------------------------


def distribution(values: list[float], label: str) -> dict[str, Any]:
    if not values:
        return {f"{label}_n": 0}
    array = np.asarray(sorted(values), dtype=float)
    return {
        f"{label}_n": int(array.size),
        f"{label}_mean": int(round(float(array.mean()))),
        f"{label}_median": int(round(float(np.median(array)))),
        f"{label}_p95": int(round(float(np.percentile(array, 95)))),
        f"{label}_min": int(array.min()),
        f"{label}_max": int(array.max()),
    }


def bytes_section(prepared: list[dict[str, Any]],
                  records: list[dict[str, Any]]) -> dict[str, Any]:
    """mp4 and npz size distributions at both resolutions, plus the per-sample total.

    The npz numbers come from the segmentation records rather than from a directory listing
    so that a sample which failed to segment cannot contribute a stale file's size.
    """
    seen: dict[str, int] = {}
    for record in records:
        if record.get("status") == "ok" and record.get("npz_bytes_832x480"):
            seen[record["sample_id"]] = record["npz_bytes_832x480"]
    section: dict[str, Any] = {
        **distribution([item["mp4_bytes_1080p"] for item in prepared], "mp4_bytes_source"),
        **distribution([item["mp4_bytes_832x480"] for item in prepared], "mp4_bytes_832x480"),
        **distribution(list(seen.values()), "npz_bytes_832x480"),
    }
    # The 1080p npz baseline is the already-built pilot dataset's own masklets, which is a
    # real measurement of the same subjects at the old resolution -- not a rescaling.
    source_npz = []
    for item in prepared:
        path = DATASET / "masklets" / f"{item['sample_id']}.npz"
        if path.is_file():
            source_npz.append(path.stat().st_size)
    section.update(distribution(source_npz, "npz_bytes_source_pilot"))
    if section.get("mp4_bytes_832x480_mean") and section.get("npz_bytes_832x480_mean"):
        section["per_sample_bytes_832x480_mean"] = (
            section["mp4_bytes_832x480_mean"] + section["npz_bytes_832x480_mean"])
        section["per_sample_bytes_832x480_median"] = (
            section["mp4_bytes_832x480_median"] + section["npz_bytes_832x480_median"])
        section["per_sample_mb_832x480_mean"] = round(
            section["per_sample_bytes_832x480_mean"] / 1e6, 3)
        section["tb_per_100k_samples"] = round(
            section["per_sample_bytes_832x480_mean"] * 100_000 / 1e12, 3)
    if section.get("mp4_bytes_source_mean") and section.get("npz_bytes_source_pilot_mean"):
        section["per_sample_bytes_source_mean"] = (
            section["mp4_bytes_source_mean"] + section["npz_bytes_source_pilot_mean"])
        section["tb_per_100k_samples_source"] = round(
            section["per_sample_bytes_source_mean"] * 100_000 / 1e12, 3)
    return section


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------


def orchestrate(levels: tuple[int, ...], per_level: int, scratch: Path,
                output_path: Path, do_ref: bool = True,
                rule: str | None = None) -> dict[str, Any]:
    subjects = gate_passing_subjects(rule=rule)
    print(f"gate-passing subjects: {len(subjects)} (rule={rule or 'module default'})",
          flush=True)
    # Every level runs ``prepared[:per_level]``, so preparing more than that is pure waste.
    # A small margin covers subjects whose clip is missing or fails to re-encode.
    subjects = subjects[: per_level + 4]
    print(f"preparing {TARGET_WIDTH}x{TARGET_HEIGHT} clips in {scratch} "
          f"(up to {len(subjects)}) ...", flush=True)
    prep_started = time.time()
    prepared = prepare(subjects, scratch)
    prep_wall = time.time() - prep_started
    print(f"prepared {len(prepared)} clips in {prep_wall:.1f}s "
          f"({prep_wall / max(len(prepared), 1):.2f}s/clip, single stream)", flush=True)
    if not prepared:
        raise SystemExit("no clips prepared; nothing to benchmark")

    per_level = min(per_level, len(prepared))
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "target": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
            "fps": TARGET_FPS,
            "frames_per_clip": 81,
            "clip_seconds": round(81 / TARGET_FPS, 4),
            "levels": list(levels),
            "samples_per_level": per_level,
            "gate": {"identity_min": 0.6, "clip_min": 0.21, "iou_min": 0.75,
                     "rule": rule or "module default (iou_stands)",
                     "note": "the report's stored `verdict` field was written under "
                             "identity_required (78/140 keep); decide()'s default rule is "
                             "iou_stands (134/140 keep)"},
            "box_source": "gate_report chosen_box_seed (GroundingDINO-corrected, seed side)",
            "sam2_config": "configs/sam2.1/sam2.1_hiera_l.yaml (image_size=1024)",
            "ref_cutout_timed_separately": do_ref,
            "scratch": str(scratch),
        },
        "prep": {
            "clips_prepared": len(prepared),
            "wall_sec": round(prep_wall, 2),
            "sec_per_clip_single_stream": round(prep_wall / len(prepared), 3),
            "note": "stage B work (decode 1080p + resize + libx264 encode), CPU-bound; "
                    "excluded from the stage C throughput numbers",
        },
        "levels": [],
    }

    all_records: list[dict[str, Any]] = []
    for concurrency in levels:
        chosen = prepared[:per_level]
        print(f"\n=== concurrency {concurrency} | {len(chosen)} samples ===", flush=True)
        result, records = run_level(concurrency, chosen, scratch, do_ref=do_ref)
        all_records.extend(records)
        payload["levels"].append(result)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {"worker_logs", "failures"}}, indent=2), flush=True)
        if result.get("worker_failure"):
            print(f"!! concurrency {concurrency} had a worker failure; "
                  f"see bench_sam2.json", flush=True)
        # Written after every level so a crash at concurrency 6 does not lose 1/2/4.
        payload["bytes"] = bytes_section(prepared, all_records)
        payload["samples"] = [
            {key: value for key, value in item.items() if key != "prep_sec"}
            for item in prepared[:per_level]
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = max((level for level in payload["levels"]
                if level.get("samples_per_gpu_hour")),
               key=lambda level: level["samples_per_gpu_hour"], default=None)
    if best:
        payload["headline"] = {
            "best_concurrency": best["concurrency"],
            "samples_per_gpu_hour": best["samples_per_gpu_hour"],
            "sec_per_sample": best["sec_per_sample"],
            "mean_gpu_util_pct": best["mean_gpu_util_pct"],
            "peak_vram_mib": best["peak_vram_mib"],
            "per_sample_bytes_832x480_mean": payload["bytes"].get(
                "per_sample_bytes_832x480_mean"),
        }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {output_path}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    worker = sub.add_parser("worker", help="internal: segment one shard")
    worker.add_argument("--manifest", required=True, type=Path)
    worker.add_argument("--output", required=True, type=Path)
    worker.add_argument("--no-ref", dest="ref", action="store_false", default=True)

    bench = sub.add_parser("bench", help="run the concurrency sweep (default)")
    bench.add_argument("--levels", default=",".join(str(v) for v in DEFAULT_LEVELS))
    bench.add_argument("--samples-per-level", type=int, default=24)
    bench.add_argument("--scratch", type=Path, default=SCRATCH)
    bench.add_argument("--output", type=Path, default=Path(
        "/mnt/pfs/users/yuanze/projects/2026/BboxCondition/outputs/bench_sam2.json"))
    bench.add_argument("--no-ref", dest="ref", action="store_false", default=True)
    bench.add_argument("--rule", default=None,
                       help="gate rule: iou_stands (decide() default) or identity_required "
                            "(what the report's stored verdict field used)")

    args = parser.parse_args(argv or (sys.argv[1:] or ["bench"]))
    if args.command == "worker":
        SCRATCH.mkdir(parents=True, exist_ok=True)
        return run_worker(args.manifest, args.output, do_ref=args.ref)

    scratch = args.scratch
    scratch.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(value) for value in str(args.levels).split(",") if value.strip())
    orchestrate(levels, args.samples_per_level, scratch, args.output, do_ref=args.ref,
                rule=args.rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
