"""Stage D: finalize a Phantom-Koala dataset into a training index.

Reads stage C's ``segmented.jsonl`` and writes ``<root>/indexes/<name>/`` with the same
layout the training launcher expects from UltraVid57k
(``indexes/bboxref_clean_dedup_iou50``): ``metadata_train.csv`` / ``metadata_eval.csv``
with an identical column set, ``funnel.json``, ``funnel_lists/*.jsonl`` and a README.

Every *decision* is delegated to UltraVidPipeline so the two datasets stay on one
calibration: ``stages.quality.filter_sample`` (mask area q75 / visible frames /
ref CLIP score), ``stages.index.deduplicate_sample`` (per-subject mask IoU),
``stages.index.audit_dataset`` (asset presence) and ``stages.index._is_eval``
(split hash). Only the orchestration is local, because ``finalize_dataset`` is bound to
a config object and to the ``samples.jsonl`` filename.

Threshold note: ``min_ref_clip_score`` is exposed on the CLI because Phantom prompts are
short noun phrases ("woman", "cat") while UltraVid prompts are VLM sentences, so Phantom
CLIP scores sit systematically lower. Relaxing it produces an index that is NOT on
UltraVid's calibration; that fact is recorded in ``funnel.json`` and the README.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Iterable

from ultravid_pipeline.schema import read_jsonl
from ultravid_pipeline.state import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file
from ultravid_pipeline.stages.index import _is_eval, audit_dataset, deduplicate_sample
from ultravid_pipeline.stages.quality import filter_sample

STAGE = "index"
DEFAULT_INPUT = "segmented.jsonl"

# Column set of UltraVid's metadata_train.csv, reproduced exactly. ``vace_video`` is
# written empty on purpose: the training dataset renders VACE control online from the
# bbox JSON (bboxref_train.py's RenderBBoxControlWindow overwrites the column), and the
# main-path setting never reads it at all. Keeping the column preserves header parity
# with the UltraVid index so a launcher cannot notice the difference.
CSV_FIELDS = (
    "sample_id", "video_id", "video", "vace_video", "bbox", "prompt",
    "object_reference_images", "frame_count",
)

# UltraVid57k config (configs/ultravid57k_v1.yaml, quality + split sections).
ULTRAVID_THRESHOLDS = {
    "max_mask_area_q75": 0.70,
    "min_visible_frames": 20,
    "min_ref_clip_score": 0.23,
    "dedup_max_mask_iou": 0.50,
}
DEFAULT_EVAL_FRACTION = 0.05
DEFAULT_SPLIT_SEED = 20260712

PENDING_FILTERS = [
    {"name": "insightface", "rule": "face quality detection not yet applied"},
    {"name": "ref_augmentation", "rule": "FLUX pose/light ref variants not generated"},
    {"name": "caption_rewrite",
     "rule": "prompt is Phantom's source-clip video_caption, not a window-scoped caption"},
]


# --------------------------------------------------------------------------------------
# pure helpers (unit tested)
# --------------------------------------------------------------------------------------


def resolve_thresholds(**overrides: Any) -> dict[str, Any]:
    """UltraVid thresholds with explicit overrides applied; unknown keys are rejected."""
    unknown = sorted(set(overrides) - set(ULTRAVID_THRESHOLDS))
    if unknown:
        raise ValueError(f"unknown threshold(s): {', '.join(unknown)}")
    resolved = dict(ULTRAVID_THRESHOLDS)
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def threshold_deltas(thresholds: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keys where ``thresholds`` departs from UltraVid, for provenance reporting."""
    return {
        key: {"ultravid": ULTRAVID_THRESHOLDS[key], "this_index": value}
        for key, value in thresholds.items()
        if float(value) != float(ULTRAVID_THRESHOLDS[key])
    }


def rejection_codes(decision: dict[str, Any]) -> list[str]:
    """Distinct ``filter_sample`` reason codes, sorted for stable reporting."""
    return sorted({str(reason["code"]) for reason in decision.get("reasons") or []})


def csv_row(sample: dict[str, Any]) -> dict[str, Any]:
    """One loader row. ``prompt`` is the clip-level caption, matching UltraVid's column.

    Phantom's ``clip_prompt`` is the source clip's ``video_caption``; our 81-frame window
    is a sub-range of that clip, so the caption can describe action outside the window.
    That is recorded as a pending filter rather than papered over here.
    """
    refs = [subject["object_reference"] for subject in sample.get("subjects") or []]
    return {
        "sample_id": sample["sample_id"],
        "video_id": sample["video_id"],
        "video": sample["video"],
        "vace_video": "",
        "bbox": sample["bbox"],
        "prompt": sample.get("clip_prompt") or sample.get("prompt") or "",
        "object_reference_images": json.dumps(refs),
        "frame_count": sample["frame_count"],
    }


def assign_splits(rows: Iterable[dict[str, Any]], fraction: float,
                  seed: int) -> dict[str, list[dict[str, Any]]]:
    """Split CSV rows by ``video_id`` hash. Source-disjoint by construction; asserted."""
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    videos: dict[str, set[str]] = {"train": set(), "eval": set()}
    for row in rows:
        name = "eval" if _is_eval(str(row["video_id"]), fraction, seed) else "train"
        splits[name].append(row)
        videos[name].add(str(row["video_id"]))
    if not videos["train"].isdisjoint(videos["eval"]):
        overlap = sorted(videos["train"] & videos["eval"])[:5]
        raise AssertionError(f"source video overlap between train and eval: {overlap}")
    splits["train_videos"] = sorted(videos["train"])  # type: ignore[assignment]
    splits["eval_videos"] = sorted(videos["eval"])  # type: ignore[assignment]
    return splits


def funnel_stages(counts: dict[str, int]) -> list[dict[str, Any]]:
    """Human/machine readable funnel rows, in pipeline order."""
    return [
        {"name": "segmented samples", "clips": counts["source"],
         "notes": "stage C output rows"},
        {"name": "stage C built", "clips": counts["built"],
         "notes": "status=built, has canonical training assets"},
        {"name": "stage C non-built", "clips": counts["source"] - counts["built"],
         "notes": "not indexed for training"},
        {"name": "quality passed", "clips": counts["quality_passed"],
         "notes": "mask_area_q75 + visible_frames + ref_clip"},
        {"name": "quality removed", "clips": counts["quality_removed"],
         "notes": "see funnel_lists/quality_removed.jsonl"},
        {"name": "dedup changed", "clips": counts["dedup_changed"],
         "notes": "subjects dropped for overlapping masklets"},
        {"name": "train index", "clips": counts["train"],
         "notes": "final loader rows"},
        {"name": "eval index", "clips": counts["eval"],
         "notes": "source-video-disjoint hash split"},
    ]


def format_readme(summary: dict[str, Any]) -> str:
    """Human-readable index README, structured like UltraVid's."""
    lines = [
        "# Phantom-Koala BBoxRef Training Index",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "> Built from Phantom-Data annotations over Koala-36M source videos on BOS.",
        "> Schema-identical to the UltraVid57k index so the same launcher consumes it.",
        "",
    ]
    deltas = summary.get("threshold_deltas") or {}
    if deltas:
        lines += [
            "> **Different calibration from the UltraVid index.** The thresholds below",
            "> depart from `configs/ultravid57k_v1.yaml`:",
            "",
        ]
        for key, value in sorted(deltas.items()):
            lines.append(f"> - `{key}`: UltraVid `{value['ultravid']}` -> this index "
                         f"`{value['this_index']}`")
        lines += [
            ">",
            "> `min_ref_clip_score` in particular is Phantom-specific: Phantom prompts are",
            "> short noun phrases (\"woman\", \"cat\"), which score lower under CLIP than",
            "> UltraVid's VLM sentences (median 0.257 vs 0.279). A relaxed value keeps",
            "> samples whose *cutout* is fine and whose *prompt* is merely terse; it is NOT",
            "> comparable to the UltraVid index funnel.",
            "",
        ]
    lines += ["## Funnel", "", "| Stage | Clips | Notes |", "| --- | ---: | --- |"]
    for stage in summary["stages"]:
        lines.append(f"| {stage['name']} | {stage['clips']:,} | {stage['notes']} |")
    lines += ["", "## Rejection Reasons", ""]
    reasons = summary.get("quality_rejection_clips") or {}
    if reasons:
        lines += ["| Code | Clips |", "| --- | ---: |"]
        for code, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{code}` | {count:,} |")
    else:
        lines.append("None; every built clip passed.")
    lines += [
        "",
        "## Thresholds",
        "",
    ]
    for key, value in sorted(summary["thresholds"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Pending Filters", ""]
    for item in summary["pending_filters"]:
        lines.append(f"- `{item['name']}`: {item['rule']}")
    lines += [
        "",
        "## Temporal Sampling",
        "",
        "Clips are 81 frames at fps 16 (5.0625 s), stored at source resolution.",
        "Training samples one random contiguous `num_frames` window per `__getitem__`;",
        "RGB video and the online-rendered bbox control share the same start frame.",
        "With `num_frames=81` the whole clip is used and no window is sampled.",
        "A subject may be absent for part of the window; those frames carry zero bbox",
        "coverage and are intentionally retained.",
        "",
        "## Prompt Column",
        "",
        "`prompt` is Phantom's `video_caption` for the **source clip**, carried through",
        "stage B/C as `clip_prompt`. It is the clip-level caption, matching what UltraVid",
        "puts in this column. Caveat: the caption describes the full Phantom clip, while",
        "our window is a 5.0625 s sub-range of it, so it may mention action outside the",
        "window. Per-subject noun phrases live in the bbox JSON (`objects[].prompt`) and",
        "are used for `ref_clip_score`, not for the training prompt.",
        "",
        "## Provenance",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Source manifest: `{summary['source_manifest']}`",
        f"- Source SHA256: `{summary['source_manifest_sha256']}`",
        f"- Split seed: `{summary['split']['seed']}`",
        f"- Eval fraction: `{summary['split']['eval_fraction']}`",
        f"- Split key: `video_id` (Koala source-video uuid), source-disjoint",
        "",
    ]
    if summary.get("pilot_scale_warning"):
        lines += [
            "## Scale Caveat",
            "",
            summary["pilot_scale_warning"],
            "",
        ]
    lines += ["## Stage Lists", "",
              "One JSONL per funnel stage; rejected rows carry their reason metadata."]
    for name, path in sorted(summary["stage_lists"].items()):
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _materialize_dedup(dataset: Path, output: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Rewrite bbox JSON without the dropped subjects, mirroring UltraVid's behaviour."""
    sample = copy.deepcopy(result["sample"])
    if not result["removed"]:
        return sample
    kept = set(result["kept_ids"])
    sample["subjects"] = [x for x in sample["subjects"] if int(x["subject_id"]) in kept]
    payload = json.loads((dataset / sample["bbox"]).read_text(encoding="utf-8"))
    key = "objects" if "objects" in payload else "subjects"
    payload[key] = [x for x in payload.get(key) or [] if int(x["subject_id"]) in kept]
    payload["subject_dedup"] = {
        "index": output.name, "kept_subject_ids": result["kept_ids"],
        "removed": result["removed"],
    }
    target = output / "bbox" / f"{sample['sample_id']}.json"
    atomic_write_json(target, payload)
    sample["bbox"] = target.relative_to(dataset).as_posix()
    sample["subject_dedup"] = payload["subject_dedup"]
    return sample


def build_index(dataset: Path, output_name: str, input_name: str = DEFAULT_INPUT,
                thresholds: dict[str, Any] | None = None,
                eval_fraction: float = DEFAULT_EVAL_FRACTION,
                seed: int = DEFAULT_SPLIT_SEED,
                skip_audit: bool = False) -> dict[str, Any]:
    dataset = dataset.resolve()
    output = (dataset / "indexes" / output_name).resolve()
    output.relative_to(dataset)  # refuse to write outside the dataset root
    thresholds = thresholds or dict(ULTRAVID_THRESHOLDS)
    manifest = dataset / input_name
    if not manifest.is_file():
        raise FileNotFoundError(f"stage C manifest not found: {manifest}")

    audit = {"skipped": True}
    if not skip_audit:
        audit = audit_dataset(dataset, manifest)
        if not audit["passed"]:
            raise RuntimeError(f"asset audit failed: {audit['errors']}")

    source = read_jsonl(manifest)
    built = [row for row in source if (row.get("status") or "built") == "built"]

    # Quality gate. Decisions for *every* built sample are persisted so downstream
    # tooling (the data viewer) can show why a clip was dropped without reopening npz.
    decisions: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for sample in built:
        decision = filter_sample(sample, dataset, thresholds)
        record = {
            "sample_id": sample["sample_id"],
            "video_id": sample.get("video_id"),
            "passed": bool(decision["passed"]),
            "codes": rejection_codes(decision),
            "reasons": decision["reasons"],
            "mask_area_q75": {str(k): v for k, v in decision["mask_area_q75"].items()},
        }
        decisions.append(record)
        if decision["passed"]:
            passed.append(sample)
        else:
            removed.append(record)

    dedup = [deduplicate_sample(sample, dataset, float(thresholds["dedup_max_mask_iou"]))
             for sample in passed]
    final = [_materialize_dedup(dataset, output, result) for result in dedup]
    events = [{"sample_id": item["sample"]["sample_id"], "removed": item["removed"],
               "kept_subject_ids": item["kept_ids"]} for item in dedup if item["removed"]]

    splits = assign_splits((csv_row(sample) for sample in final), eval_fraction, seed)
    train_rows, eval_rows = splits["train"], splits["eval"]

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "metadata_train.csv", train_rows)
    _write_csv(output / "metadata_eval.csv", eval_rows)
    atomic_write_jsonl(output / "samples.jsonl", final)
    atomic_write_jsonl(output / "quality_decisions.jsonl", decisions)
    lists = output / "funnel_lists"
    list_rows = {
        "source_all": [{"sample_id": x.get("sample_id"), "video_id": x.get("video_id"),
                        "status": x.get("status") or "built"} for x in source],
        "built": [{"sample_id": x["sample_id"], "video_id": x.get("video_id"),
                   "num_subjects": len(x.get("subjects") or [])} for x in built],
        "quality_passed": [{"sample_id": x["sample_id"], "video_id": x.get("video_id")}
                           for x in passed],
        "quality_removed": removed,
        "dedup_changed": events,
        "train": [{"sample_id": x["sample_id"], "video_id": x["video_id"]} for x in train_rows],
        "eval": [{"sample_id": x["sample_id"], "video_id": x["video_id"]} for x in eval_rows],
    }
    for name, rows in list_rows.items():
        atomic_write_jsonl(lists / f"{name}.jsonl", rows)

    counts = {
        "source": len(source), "built": len(built),
        "quality_passed": len(passed), "quality_removed": len(removed),
        "dedup_changed": len(events), "train": len(train_rows), "eval": len(eval_rows),
    }
    reason_clips: dict[str, int] = {}
    for record in removed:
        for code in record["codes"]:
            reason_clips[code] = reason_clips.get(code, 0) + 1
    reason_subjects: dict[str, int] = {}
    for record in removed:
        for reason in record["reasons"]:
            code = str(reason["code"])
            reason_subjects[code] = reason_subjects.get(code, 0) + 1
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "index_name": output_name,
        "dataset": str(dataset),
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "code_version": _code_version(),
        "thresholds": thresholds,
        "threshold_deltas": threshold_deltas(thresholds),
        "counts": counts,
        "stages": funnel_stages(counts),
        "quality_rejection_clips": reason_clips,
        "quality_rejection_subjects": reason_subjects,
        "final_subjects": sum(len(x.get("subjects") or []) for x in final),
        "frame_count_distribution": _histogram(x["frame_count"] for x in final),
        "subject_count_histogram": _histogram(len(x.get("subjects") or []) for x in final),
        "pending_filters": PENDING_FILTERS,
        "split": {
            "key": "video_id", "seed": seed, "eval_fraction": eval_fraction,
            "train_clips": len(train_rows), "eval_clips": len(eval_rows),
            "train_videos": len(splits["train_videos"]),
            "eval_videos": len(splits["eval_videos"]),
            "video_overlap": 0,
        },
        "outputs": {
            "train": str(output / "metadata_train.csv"),
            "eval": str(output / "metadata_eval.csv"),
            "samples": str(output / "samples.jsonl"),
            "quality_decisions": str(output / "quality_decisions.jsonl"),
        },
        "stage_lists": {name: str(lists / f"{name}.jsonl") for name in list_rows},
        "asset_audit": audit,
        "csv_fields": list(CSV_FIELDS),
    }
    if len(final) < 1000:
        plural = "clip" if len(eval_rows) == 1 else "clips"
        summary["pilot_scale_warning"] = (
            f"Pilot scale ({len(final)} clips, one clip per source video). The "
            f"{eval_fraction:.0%} eval split holds only {len(eval_rows)} {plural}, which "
            "exists to validate the pipeline end to end and must NOT be used as an "
            "evaluation set -- it has no statistical meaning at this scale."
        )
    atomic_write_json(output / "funnel.json", summary)
    atomic_write_text(output / "README.md", format_readme(summary))
    return {**summary, "output": str(output)}


def _histogram(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _code_version() -> str:
    try:
        import phantom_data

        return getattr(phantom_data, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage D: finalize a training index")
    parser.add_argument("--dataset", required=True, type=Path, help="dataset root")
    parser.add_argument("--output-name", required=True,
                        help="index directory name under <dataset>/indexes/")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="stage C manifest, relative to --dataset")
    parser.add_argument("--min-ref-clip-score", type=float, default=None,
                        help=f"default {ULTRAVID_THRESHOLDS['min_ref_clip_score']} "
                             "(UltraVid calibration)")
    parser.add_argument("--min-visible-frames", type=int, default=None)
    parser.add_argument("--max-mask-area-q75", type=float, default=None)
    parser.add_argument("--dedup-max-mask-iou", type=float, default=None)
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--skip-audit", action="store_true",
                        help="skip the asset-presence audit (debug only)")
    args = parser.parse_args(argv)

    thresholds = resolve_thresholds(
        min_ref_clip_score=args.min_ref_clip_score,
        min_visible_frames=args.min_visible_frames,
        max_mask_area_q75=args.max_mask_area_q75,
        dedup_max_mask_iou=args.dedup_max_mask_iou,
    )
    summary = build_index(
        args.dataset, args.output_name, input_name=args.input, thresholds=thresholds,
        eval_fraction=args.eval_fraction, seed=args.seed, skip_audit=args.skip_audit,
    )
    print(json.dumps({
        "output": summary["output"], "thresholds": summary["thresholds"],
        "threshold_deltas": summary["threshold_deltas"], "counts": summary["counts"],
        "quality_rejection_clips": summary["quality_rejection_clips"],
        "quality_rejection_subjects": summary["quality_rejection_subjects"],
        "split": summary["split"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
