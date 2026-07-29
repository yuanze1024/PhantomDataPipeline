"""Stage A: Phantom filtered parquet -> sample specs jsonl.

One spec = one 81-frame @ 16fps window of one Koala source video, with the subjects
(target seed box + reference frame pointer) that fall inside that window.

Only one row per source uuid is kept, so a train/eval split on ``video_id`` is
source-disjoint by construction and no two samples share visual content.

    python -m phantom_data.build.plan --num-sources 100 --out specs.jsonl

Sharding (``--shard-id`` / ``--num-shards``) exists so several pods can plan and then
build disjoint slices of the 183k available sources in parallel. See
:func:`shard_of` for the hash and :func:`build_specs` for the ``--num-sources``
interaction, which is **per shard**, not global.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from ..dataset import (
    AUDIT_CSV,
    FILTERED_PARQUET,
    META_PARQUET,
    PhantomIndex,
    frame_time,
    parse_vid,
)
from .window import (
    FPS,
    WINDOW_FRAMES,
    WINDOW_SEC,
    choose_window,
    sample_id_for,
    seed_frame_index,
    source_order_key,
)

DEFAULT_SEED = 20260725


def read_filtered_rows(parquet_path: str | Path) -> list[dict[str, Any]]:
    """Single-threaded read of the filtered table (shared FS: never fan out threads)."""
    import pyarrow

    pyarrow.set_cpu_count(1)
    table = pq.read_table(
        str(parquet_path),
        columns=["video_id", "video_caption", "cross_pair"],
        use_threads=False,
    )
    video_ids = table["video_id"].to_pylist()
    captions = table["video_caption"].to_pylist()
    cross_pairs = table["cross_pair"].to_pylist()
    return [
        {"video_id": video_id, "video_caption": caption, "cross_pair": cross_pair}
        for video_id, caption, cross_pair in zip(video_ids, captions, cross_pairs)
    ]


def pick_one_row_per_source(
    rows: Iterable[dict[str, Any]], seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Deterministically keep one row per source uuid, ordered by a seeded hash.

    Within a source the row whose full ``video_id`` hashes lowest wins; sources are then
    ordered by the hash of the uuid so that ``--num-sources`` takes a stable random-ish
    subset rather than a parquet-order prefix.
    """
    best: dict[str, tuple[str, dict[str, Any]]] = {}
    for row in rows:
        uuid = str(row["video_id"]).rsplit("_", 2)[0]
        key = source_order_key(str(row["video_id"]), seed)
        current = best.get(uuid)
        if current is None or key < current[0]:
            best[uuid] = (key, row)
    ordered = sorted(best.items(), key=lambda item: source_order_key(item[0], seed))
    return [row for _, (_, row) in ordered]


def source_uuid_of(video_id: str) -> str:
    """The Koala source uuid behind a Phantom ``<uuid>_<start>_<end>`` id.

    Same ``rsplit("_", 2)[0]`` as :func:`pick_one_row_per_source` uses, factored out so the
    shard hash and the one-row-per-source dedup provably key on the same string.
    """
    return str(video_id).rsplit("_", 2)[0]


def shard_of(source_uuid: str, num_shards: int) -> int:
    """Which shard a source uuid belongs to. Stable, independent of every other row.

    Deliberately hashed on the **source uuid** rather than assigned by position in the
    ordered list (``index % num_shards``), because position depends on the whole input:
    re-plan after Phantom ships a new parquet, or with a different ``--seed``, and a
    positional assignment reshuffles every shard, so pods that already extracted their
    slice would have to redo it. The uuid hash keeps a source in the same shard forever.

    Independent of :data:`DEFAULT_SEED` on purpose too. ``--seed`` controls the *order*
    within a shard (via :func:`source_order_key`) and hence which sources ``--num-sources``
    takes first; folding it into the shard hash as well would mean changing the seed also
    migrates sources between shards, mixing two unrelated knobs.

    ``num_shards <= 1`` short-circuits to 0 so the unsharded path never even hashes.
    """
    if num_shards <= 1:
        return 0
    digest = hashlib.sha256(f"shard:{source_uuid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % int(num_shards)


def select_shard(
    rows: Iterable[dict[str, Any]], shard_id: int = 0, num_shards: int = 1
) -> list[dict[str, Any]]:
    """Keep only the rows whose source uuid hashes into ``shard_id``, order preserved.

    With ``num_shards=1`` this returns the input list unchanged (identity), which is what
    makes the default path byte-identical to the pre-sharding behaviour.
    """
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"--shard-id must be in [0, {num_shards}), got {shard_id}")
    if num_shards == 1:
        return list(rows)
    return [
        row
        for row in rows
        if shard_of(source_uuid_of(str(row["video_id"])), num_shards) == shard_id
    ]


def _subject_candidates(cross_pair: dict[str, Any], row_vid: str) -> list[dict[str, Any]]:
    """Flatten ``cross_pair`` into per-phrase (target, ref) candidates.

    Phantom guarantees exactly one target and one reference per phrase; anything that
    breaks that shape is reported through ``malformed`` so it is never silently dropped.
    """
    candidates: list[dict[str, Any]] = []
    for phrase, payload in cross_pair.items():
        targets = payload.get("obj_from_tgt_video") or []
        references = [ref for group in (payload.get("refer_result") or []) for ref in group]
        if len(targets) != 1 or len(references) != 1:
            candidates.append({"phrase": phrase, "malformed": f"targets={len(targets)},refs={len(references)}"})
            continue
        target, reference = targets[0], references[0]
        if str(target.get("vid")) != row_vid:
            candidates.append({"phrase": phrase, "malformed": "target_vid_mismatch"})
            continue
        candidates.append({"phrase": phrase, "target": target, "reference": reference})
    return candidates


def plan_row(
    row: dict[str, Any],
    index: PhantomIndex,
    num_frames: int = WINDOW_FRAMES,
    fps: int = FPS,
) -> dict[str, Any]:
    """Turn one filtered row into ``{"spec": ...}`` or ``{"reason": ...}``."""
    row_vid = str(row["video_id"])
    uuid, start, end = parse_vid(row_vid)
    window_sec = num_frames / fps
    if end - start < window_sec - 1e-9:
        return {"reason": "clip_too_short"}

    cross_pair = json.loads(row["cross_pair"])
    candidates = _subject_candidates(cross_pair, row_vid)
    usable = [item for item in candidates if "malformed" not in item]
    dropped: list[dict[str, Any]] = [
        {"phrase": item["phrase"], "reason": "malformed_annotation", "detail": item["malformed"]}
        for item in candidates
        if "malformed" in item
    ]
    if not usable:
        return {"reason": "no_usable_subject"}

    seed_times = [
        frame_time(start, end, item["target"]["frame_idx"]) for item in usable
    ]
    plan = choose_window(start, end, seed_times, window_sec=window_sec)
    if plan is None:
        return {"reason": "no_window_covers_any_seed"}
    for position in plan.dropped:
        dropped.append(
            {
                "phrase": usable[position]["phrase"],
                "reason": "seed_outside_window",
                "seed_abs_time": seed_times[position],
            }
        )

    source = index.resolve_bos_key(row_vid)
    if source is None:
        return {"reason": "bos_unresolved"}

    subjects: list[dict[str, Any]] = []
    for offset, position in enumerate(plan.covered, start=1):
        item = usable[position]
        target, reference = item["target"], item["reference"]
        ref_vid = str(reference["vid"])
        ref_source = index.resolve_bos_key(ref_vid)
        if ref_source is None:
            dropped.append({"phrase": item["phrase"], "reason": "ref_bos_unresolved"})
            continue
        _, ref_start, ref_end = parse_vid(ref_vid)
        ref_box = reference["bbox_loc"]
        while isinstance(ref_box, list) and ref_box and isinstance(ref_box[0], list):
            ref_box = ref_box[0]
        subjects.append(
            {
                "subject_id": offset,
                "phrase": item["phrase"],
                "bbox_cls": target.get("bbox_cls"),
                "seed_frame_index": seed_frame_index(
                    seed_times[position], plan.window_start, fps=fps, num_frames=num_frames
                ),
                "seed_abs_time": round(seed_times[position], 6),
                # The ``_768`` suffix here and on ``ref.bbox_768`` below is a MISNOMER kept
                # for on-disk compatibility: these are raw annotation coordinates whose
                # canvas is unresolved (see ``phantom_data.canvas``). Stage C's manifest
                # contract and the built pilot dataset use these names; do not rename.
                "seed_bbox_768": [float(value) for value in target["bbox_loc"]],
                "ref": {
                    "bucket": ref_source[0],
                    "key": ref_source[1],
                    "abs_time": round(frame_time(ref_start, ref_end, reference["frame_idx"]), 6),
                    "bbox_768": [float(value) for value in ref_box],
                    "bbox_cls": reference.get("bbox_cls"),
                    "phantom_vid": ref_vid,
                    "score": reference.get("scores"),
                },
            }
        )
    if not subjects:
        return {"reason": "no_subject_with_resolved_ref"}

    # subject_id must be dense after ref-resolution drops
    for offset, subject in enumerate(subjects, start=1):
        subject["subject_id"] = offset

    return {
        "spec": {
            "sample_id": sample_id_for(uuid, plan.window_start),
            "video_id": uuid,
            "phantom_video_id": row_vid,
            "source": {
                "bucket": source[0],
                "key": source[1],
                "window_start_sec": round(plan.window_start, 6),
                "window_end_sec": round(plan.window_start + window_sec, 6),
                "clip_start_sec": start,
                "clip_end_sec": end,
                "num_frames": num_frames,
                "fps": fps,
            },
            "caption": row["video_caption"],
            "subjects": subjects,
            "dropped_subjects": dropped,
        }
    }


def build_specs(
    num_sources: int | None,
    seed: int = DEFAULT_SEED,
    filtered_parquet: str | Path = FILTERED_PARQUET,
    meta_parquet: str | Path = META_PARQUET,
    audit_csv: str | Path = AUDIT_CSV,
    num_frames: int = WINDOW_FRAMES,
    fps: int = FPS,
    shard_id: int = 0,
    num_shards: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Plan specs for one shard of the source pool.

    ``num_sources`` is **per shard**, applied after the shard filter: ``--num-sources 1000
    --num-shards 8`` gives 8 pods that each plan 1000 samples, 8000 total. The alternative
    reading ("take 1000 globally, then split") was rejected because it makes parallelism
    useless -- 8 pods would each get ~125 samples, and the number a pod must build would
    depend on how the hash happened to divide, so no pod could be sized or timed in
    advance. Per-shard also means a pod's workload is knowable from its own arguments
    alone, with no cross-shard coordination.

    Consequence worth stating plainly: a shard's output is *not* a subset of what the same
    ``--num-sources`` would produce unsharded, and the union of N shards is ~N x
    ``num_sources`` samples. Whenever ``num_sources`` truncates, the union of all shards is
    also not the unsharded set -- each shard walks its own sources in its own order and
    stops at its own limit. Only the untruncated case (``num_sources=None``) partitions the
    pool exactly, which is what ``tests/test_shard.py`` asserts disjointness against.

    Rejection tallies in ``stats`` are likewise shard-local; :mod:`phantom_data.build.funnel`
    sums them across shards.
    """
    rows = read_filtered_rows(filtered_parquet)
    unique = pick_one_row_per_source(rows, seed=seed)
    sources_in_pool = len(unique)
    # Shard *after* the one-row-per-source dedup: dedup is a property of the source, so
    # doing it first keeps every shard's view of a source identical to the unsharded view.
    unique = select_shard(unique, shard_id=shard_id, num_shards=num_shards)
    index = PhantomIndex(filtered_parquet, meta_parquet, audit_csv)

    specs: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    subject_counts: Counter[int] = Counter()
    dropped_reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    considered = 0
    for row in unique:
        if num_sources is not None and len(specs) >= num_sources:
            break
        considered += 1
        outcome = plan_row(row, index, num_frames=num_frames, fps=fps)
        if "spec" in outcome:
            spec = outcome["spec"]
            if spec["sample_id"] in seen_ids:
                reasons["duplicate_sample_id"] += 1
                continue
            seen_ids.add(spec["sample_id"])
            specs.append(spec)
            subject_counts[len(spec["subjects"])] += 1
            for item in spec["dropped_subjects"]:
                dropped_reasons[item["reason"]] += 1
        else:
            reasons[outcome["reason"]] += 1

    stats = {
        "seed": seed,
        "filtered_parquet": str(filtered_parquet),
        "source_rows": len(rows),
        # Always the *global* post-dedup pool, not this shard's slice, so the number means
        # the same thing in every shard's stats file. Unsharded the two are equal, which is
        # what keeps this file byte-identical to pre-sharding runs.
        "unique_sources": sources_in_pool,
        "sources_considered": considered,
        "requested_sources": num_sources,
        "samples": len(specs),
        "rejected": dict(reasons),
        "subjects_total": sum(len(spec["subjects"]) for spec in specs),
        "subject_count_histogram": {str(key): value for key, value in sorted(subject_counts.items())},
        "dropped_subject_reasons": dict(dropped_reasons),
        "window": {"num_frames": num_frames, "fps": fps, "seconds": num_frames / fps},
    }
    if num_shards > 1:
        # Emitted only when sharding is actually on, so an unsharded stats file has exactly
        # the keys it always had (byte-identical). Readers must treat a missing "shard"
        # block as shard 0 of 1 -- funnel.py does.
        stats["shard"] = {
            "shard_id": shard_id,
            "num_shards": num_shards,
            "sources_in_shard": len(unique),
            "num_sources_is_per_shard": True,
        }
    return specs, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan Phantom-Koala training samples")
    parser.add_argument("--num-sources", type=int, default=100,
                        help="stop after this many planned samples (one per source video); 0 = all")
    parser.add_argument("--out", required=True, help="output specs jsonl path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--filtered-parquet", default=FILTERED_PARQUET)
    parser.add_argument("--meta-parquet", default=META_PARQUET)
    parser.add_argument("--audit-csv", default=AUDIT_CSV)
    parser.add_argument("--num-frames", type=int, default=WINDOW_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split the source pool across this many parallel pods; "
                             "1 (default) = no sharding")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="which shard to plan, in [0, --num-shards). --num-sources is "
                             "PER SHARD, so N pods produce ~N x --num-sources samples")
    args = parser.parse_args(argv)

    specs, stats = build_specs(
        num_sources=args.num_sources or None,
        seed=args.seed,
        filtered_parquet=args.filtered_parquet,
        meta_parquet=args.meta_parquet,
        audit_csv=args.audit_csv,
        num_frames=args.num_frames,
        fps=args.fps,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs), encoding="utf-8"
    )
    stats_path = Path(str(out) + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {len(specs)} specs -> {out}")
    print(f"wrote stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
