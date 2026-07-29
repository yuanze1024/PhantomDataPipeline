"""Stage 2 of the box pipeline: re-detect every subject, write the frames and the numbers.

**Reads stage B (``extracted.jsonl``), not stage C.** This is the change that lets box
correction run *before* SAM2:

    A plan -> B extract -> 1 enrich -> 2 redetect -> 3 gate(apply) -> C segment -> D index

The old order put this pass after segmentation, which meant every masklet was cut from
Phantom's box and the corrected box was never written back anywhere. Measured on the pilot,
93/140 reference boxes and 107/140 target boxes should have been replaced -- so the shipped
masklets were mostly cut from boxes already known to be skewed, and that box *is* the training
conditioning signal. Running before SAM2 also stops paying for segmentation on the 44% of
subjects the identity gate drops.

Nothing here needs SAM2 to have run. The three inputs come from stage B directly:

* the reference frame is ``subject.ref.frame``, the raw jpg stage B already wrote;
* the target frame is decoded from the clip at ``seed_frame_index``, as before;
* the two boxes are stage B's raw annotation coordinates put into frame space with
  :func:`segment.scale_bbox_to_frame` -- a pure canvas-map-then-clamp function with no model
  in it. Stage C used to precompute these into ``bbox/<id>.json``; that was the only reason
  this pass ever read stage C's output.

What the new order gives up: stage C's ``ref_clip_score`` and ``ref_mask_coverage`` do not
exist yet, so those two report fields come out None. Neither ever fed a decision -- they were
display context -- and ``ref_clip_score`` was never comparable to the ``crop_clip_*`` numbers
anyway (it scores SAM2's white-matte cutout, these score the plain crop).

Draws nothing, on purpose. Boxes burnt into an image make every presentation change -- a
colour, a line width, hiding one box -- a reason to re-run Grounding DINO over the whole
dataset. So this writes the two frames *unannotated* and puts every box coordinate in
``gate_report.json``; :mod:`phantom_data.gate_viewer` overlays them at page-render time.

Per subject it writes:

  ``<sample>/subj<NN>_ref.jpg``     the reference frame, as decoded
  ``<sample>/subj<NN>_target.jpg``  the clip's seed frame, as decoded

Frames are shared between subjects of the same sample in content but written per subject, so a
subject's record is self-contained -- the alternative saves a little disk and makes the viewer
resolve which subject shares which file.

All the arithmetic lives in :mod:`phantom_data.redetect`; this file decodes, calls it, saves
frames, and writes the report.

Usage: python tools/redetect_run.py --dataset <root> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from phantom_data import enrich, redetect
from phantom_data.build import segment
from phantom_data.inspect import atomic_write_bytes, decode_frames, read_jsonl

STAGE = "redetect"
DEFAULT_OUT_ROOT = "_redetect100"
DEFAULT_CACHE = "_enrich"
DEFAULT_INPUT = "extracted.jsonl"

#: Frames are saved at full decoded resolution: the viewer crops from them, so a small box
#: needs the original pixels to have anything to show. JPEG rather than PNG because that is
#: 280 full-frame images -- PNG measured ~8x larger for no visible benefit at this quality,
#: and these are display copies of frames that already exist as video on disk.
JPEG_QUALITY = 92


def save_frame(frame: np.ndarray, path: Path) -> None:
    """Write one frame, unannotated, atomically.

    Not :func:`atomic_save_image` -- that one is PNG-only. Same temp-file-then-rename
    guarantee via :func:`atomic_write_bytes`, so a full disk cannot leave a truncated JPEG
    that the viewer would read as a corrupt image rather than as absent.
    """
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB").save(
        buffer, format="JPEG", quality=JPEG_QUALITY)
    atomic_write_bytes(path, buffer.getvalue())


def subject_boxes(subject: dict[str, Any], clip_width: int, clip_height: int,
                  ref_width: int, ref_height: int) -> tuple[list[float], list[float]]:
    """Stage B's raw annotation boxes for one subject, in frame coordinates.

    Returns ``(ref_box, seed_box)``. Two separate frames with two separate sizes, which is the
    whole reason this is not one call: the reference jpg is a frame of a *different* video at a
    different resolution from the clip (1920x1080 references against 1280x720 clips are common),
    so mapping the reference box against the clip's dimensions would shift it by the ratio of
    the two.

    Identical arithmetic to what stage C applies, by calling the same function rather than
    reimplementing it -- these boxes are what the report's IoU-vs-Phantom column compares
    against, so any divergence would make that number measure the wrong thing.
    """
    ref = subject.get("ref") or {}
    return (
        segment.scale_bbox_to_frame(ref["bbox_768"], ref_width, ref_height),
        segment.scale_bbox_to_frame(subject["seed_bbox_768"], clip_width, clip_height),
    )


def process_subject(dataset: Path, sample_id: str, subject: dict[str, Any],
                    seed_frame: np.ndarray, clip_size: tuple[int, int],
                    texts: dict[str, Any], models: redetect.Models, out_root: Path,
                    trust_detector: bool = True) -> dict[str, Any]:
    """Re-detect one subject and return its report row."""
    from PIL import Image

    subject_id = int(subject["subject_id"])
    ref_relative = (subject.get("ref") or {})["frame"]
    ref_frame = np.asarray(Image.open(dataset / ref_relative).convert("RGB"))
    ref_height, ref_width = ref_frame.shape[:2]
    clip_width, clip_height = clip_size
    ref_box, seed_box = subject_boxes(subject, clip_width, clip_height, ref_width, ref_height)

    analysis = redetect.analyse_subject(models, ref_frame, seed_frame, ref_box, seed_box,
                                        texts["dis"], trust_detector=trust_detector)
    analysis["text_source"] = texts.get("text_source")

    frames = {"ref": f"{sample_id}/subj{subject_id:02d}_ref.jpg",
              "seed": f"{sample_id}/subj{subject_id:02d}_target.jpg"}
    save_frame(ref_frame, out_root / frames["ref"])
    save_frame(seed_frame, out_root / frames["seed"])
    # ``entry`` carries provenance only. ``ref_clip_score`` / ``ref_mask_coverage`` are stage C
    # numbers and stage C has not run yet in this order, so they are simply absent -- see the
    # module docstring for why that costs nothing.
    return redetect.subject_record(sample_id, subject_id, analysis, frames,
                                   {"seed_frame_index": subject.get("seed_frame_index")})


def load_texts(dataset: Path, cache_dir: str,
               input_name: str = DEFAULT_INPUT) -> dict[tuple[str, int], dict[str, Any]]:
    """The enriched phrases, keyed by subject, with Phantom's own phrase as the fallback.

    Reading the cache rather than calling the gateway keeps this pass free to repeat: the
    phrases were paid for once by ``tools/enrich_subjects.py``.
    """
    cache = dataset / cache_dir
    texts: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(dataset / input_name):
        for subject in row.get("subjects") or []:
            subject_id = int(subject["subject_id"])
            phrase = str(subject.get("phrase") or subject.get("bbox_cls") or "")
            ref_phrase = str((subject.get("ref") or {}).get("bbox_cls") or "")
            path = cache / enrich.cache_name(row["sample_id"], subject_id)
            entry = None
            if path.is_file():
                try:
                    entry = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                    entry = None
            # Validity is judged on ``dis``: it is the only text this pass uses, both as the
            # detector query and as the CLIP text.
            texts[(row["sample_id"], subject_id)] = (
                entry if isinstance(entry, dict) and entry.get("dis")
                else enrich.fallback(phrase, ref_phrase))
    return texts


def recover_partial(path: Path, done: set[str]) -> list[dict[str, Any]]:
    """Subject rows from a previous run's partial file, for samples that finished.

    Filtered by ``done`` (the marker set) rather than trusted wholesale: a run killed between
    appending a row and writing its marker leaves rows for a sample that will be re-processed,
    and admitting those would duplicate subjects in the report. Marker-then-rows would just move
    the race, so the marker is treated as the authority and the rows as its recoverable payload.

    Deduplicated on ``(sample_id, subject_id)``, last write winning, so a ``--force`` re-run
    appending to an existing partial file cannot produce two rows for one subject.
    """
    if not path.is_file():
        return []
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A row torn by a SIGKILL mid-write. The sample has no marker in that case, so it
            # is about to be re-processed anyway; skipping the fragment is the whole recovery.
            continue
        if isinstance(row, dict) and str(row.get("sample_id")) in done:
            rows[(str(row.get("sample_id")), int(row.get("subject_id", -1)))] = row
    return list(rows.values())


def run(dataset: Path, models: redetect.Models, out_root: str = DEFAULT_OUT_ROOT,
        cache_dir: str = DEFAULT_CACHE, input_name: str = DEFAULT_INPUT,
        limit: int = 0, force: bool = False,
        trust_detector: bool = True) -> dict[str, Any]:
    """Re-detect every subject of every sample, resumably, and write ``gate_report.json``.

    Resume is the same two-part mechanism ``build/extract.py`` uses, deliberately not a new one:
    a :class:`MarkerStore` marker per sample says *whether* it is done, and an append+fsync
    partial jsonl holds *what* it produced. Markers alone cannot rebuild the report, because a
    marker records shape, not the dozens of coordinates and scores per subject.

    The store is rooted at ``out_root`` rather than at the dataset, so ``_redetect100`` and a
    second ``--out-root`` (a different rule, a different phrase set) each keep their own resume
    state instead of one silently satisfying the other's markers.

    fsync on every append: 140 subjects take 3 minutes and none of this matters, but at 100k+
    subjects a crash near the end of a multi-hour GPU pass must not cost the whole pass, and an
    OOM kill gives no chance to flush.
    """
    from ultravid_pipeline.state import MarkerStore

    root = dataset / out_root
    root.mkdir(parents=True, exist_ok=True)
    markers = MarkerStore(root)

    samples = read_jsonl(dataset / input_name)[:limit or None]
    texts = load_texts(dataset, cache_dir, input_name)

    done = set() if force else {str(s["sample_id"]) for s in samples
                                if markers.get(STAGE, str(s["sample_id"])) is not None}
    records: list[dict[str, Any]] = recover_partial(root / f"{STAGE}.partial.jsonl", done)
    pending = [s for s in samples if str(s["sample_id"]) not in done]
    print(f"{len(samples)} samples, {len(samples) - len(pending)} already done "
          f"({len(records)} subject rows recovered), {len(pending)} to do", flush=True)

    partial = open(root / f"{STAGE}.partial.jsonl", "a", encoding="utf-8")
    started = time.time()
    failures: list[dict[str, Any]] = []
    try:
        for position, sample in enumerate(pending, 1):
            sample_id = str(sample["sample_id"])
            try:
                frames = decode_frames(dataset / sample["video"])
                if not frames:
                    raise ValueError("decoded 0 frames")
                height, width = frames[0].shape[:2]
                produced: list[dict[str, Any]] = []
                for subject in sample.get("subjects") or []:
                    index = min(int(subject["seed_frame_index"]), len(frames) - 1)
                    subject_texts = texts.get((sample_id, int(subject["subject_id"]))) or \
                        enrich.fallback(str(subject.get("phrase") or ""))
                    produced.append(process_subject(
                        dataset, sample_id, subject, frames[index], (width, height),
                        subject_texts, models, root, trust_detector=trust_detector))
                # Rows first, then the marker: the marker is what makes the rows count, so it
                # is written last and a crash in between costs a re-run of one sample rather
                # than a report with half a sample in it.
                for row in produced:
                    partial.write(json.dumps(row, ensure_ascii=False) + "\n")
                partial.flush()
                os.fsync(partial.fileno())
                markers.put(STAGE, sample_id, {"status": "passed", "subjects": len(produced)})
                records.extend(produced)
                print(f"[{position}/{len(pending)}] {sample_id} ok "
                      f"({len(produced)} subj, {time.time() - started:.0f}s)", flush=True)
            except Exception as error:  # noqa: BLE001 - one bad sample must not stop the run
                detail = f"{type(error).__name__}: {error}"
                failures.append({"sample_id": sample_id, "error": detail})
                # No marker on failure, so the next run retries it. MarkerStore.get only
                # honours passed/rejected, so writing a failed marker would not skip it either;
                # leaving it absent keeps the report's failure list the single account.
                print(f"[{position}/{len(pending)}] {sample_id} FAILED {detail}", flush=True)
    finally:
        partial.close()

    summary = redetect.summarise(records)
    report = {
        "rule": {"identity_min": redetect.IDENTITY_MIN, "clip_min": redetect.CLIP_MIN,
                 "iou_min": redetect.IOU_MIN},
        # Recorded because it changes what the boxes in this report *are*, not just how they
        # were scored: under trust_detector the detector's box wins outright and a side with no
        # detection is filtered out instead of falling back to Phantom's box.
        "trust_detector": trust_detector,
        "input": input_name,
        "summary": summary, "subjects": records, "failures": failures,
    }
    atomic_write_bytes(root / "gate_report.json",
                       (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"\n{len(records)} subjects in {time.time() - started:.0f}s, "
          f"{len(failures)} samples failed")
    for key, value in summary.items():
        print(f"  {key:30s} {value}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="stage B manifest name under --dataset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true",
                        help="ignore resume markers and re-process every sample")
    # Two modes rather than a tunable: see redetect.pick_side. The default trusts the detector
    # because its boxes are in real frame coordinates; --no-trust-detector reproduces the
    # pilot's gate_report.json exactly and exists as the control.
    parser.add_argument("--no-trust-detector", dest="trust_detector", action="store_false",
                        help="use the historical pick_side rule (phantom's box wins when its "
                             "crop scored well and the new box moved too far)")
    parser.set_defaults(trust_detector=True)
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    models = redetect.Models(device=args.device)
    report = run(dataset, models, out_root=args.out_root, cache_dir=args.cache_dir,
                 input_name=args.input, limit=args.limit, force=args.force,
                 trust_detector=args.trust_detector)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
