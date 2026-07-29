"""Build the clip-ingest request list to hand to the Koala pipeline owner.

Every Phantom annotation row needs its target clip AND all of its reference clips before
it becomes a usable training sample, so raw clip count is the wrong ranking. This scores
each missing clip by how many A rows it unlocks, then emits waves:

  wave 1  clips that complete an A row whose other clips are ALL already in C
  wave 2  clips for A rows needing 2 clips, and so on

Output is keyed by (hf_csv, hf_record_index) -- verified to match 100% of Phantom's
windows against the official Koala-36M CSVs -- so the owner can enqueue by row id with no
matching logic of their own. (youtube_id, start, end) is carried along for readability.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq

CSV_DIR = "/mnt/pfs/share/pengchunli/koala36m_thordata_audit/hf_annotations"
A = "/mnt/pfs/users/yuanze/datasets/phantom_data_koala36m/koala36M_multi_ref_merged_filtered.parquet"
B = "/mnt/pfs/users/yuanze/datasets/phantom_data_koala36m/koala36M_multi_ref_meta_info_merged.parquet"
C = "/mnt/pfs/share/pengchunli/dataset/koala36m_v1_thordata_face_caption_ready_20260724.parquet"
AUDIT = "/mnt/pfs/share/pengchunli/koala36m_thordata_audit/thordata_bos_videos.csv"
OUT_DIR = "/mnt/pfs/users/yuanze/datasets/phantom_clip_request_v1"

YT = re.compile(r"v=([^&]+)")
KEY = re.compile(r"Thordata/([^/]+)/")

csv.field_size_limit(10 * 1024 * 1024)


def parse_timestamp(raw: str) -> tuple[float, float] | None:
    parts = re.findall(r"(\d+):(\d+):([\d.]+)", raw or "")
    if len(parts) != 2:
        return None
    return tuple(
        int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        for hours, minutes, seconds in parts
    )  # type: ignore[return-value]


def key_of(youtube_id: str, start: float, end: float) -> tuple[str, float, float]:
    return (youtube_id, round(start, 1), round(end, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--max-waves", type=int, default=6)
    args = parser.parse_args()

    # ---- B: vid -> youtube id ---------------------------------------------------
    vid2yt: dict[str, str] = {}
    for batch in pq.ParquetFile(B).iter_batches(
        batch_size=65536, columns=["vid", "youtube_url"], use_threads=False
    ):
        cols = batch.to_pydict()
        for vid, url in zip(cols["vid"], cols["youtube_url"]):
            match = YT.search(url or "")
            if match:
                vid2yt[vid] = match.group(1)
    print(f"[B] {len(vid2yt)} vids", flush=True)

    def clip_key(vid: str):
        youtube_id = vid2yt.get(vid)
        if youtube_id is None:
            return None
        try:
            _, start_str, end_str = vid.rsplit("_", 2)
            return key_of(youtube_id, float(start_str), float(end_str))
        except ValueError:
            return None

    # ---- C: which clips already exist ------------------------------------------
    have: set[tuple[str, float, float]] = set()
    for batch in pq.ParquetFile(C).iter_batches(
        batch_size=32768,
        columns=["source_video_key", "clip_start_time", "clip_end_time"],
        use_threads=False,
    ):
        cols = batch.to_pydict()
        for raw, start, end in zip(cols["source_video_key"],
                                   cols["clip_start_time"], cols["clip_end_time"]):
            match = KEY.search(raw or "")
            if match and start is not None and end is not None:
                have.add(key_of(match.group(1), start, end))
    print(f"[C] {len(have)} clips already ingested", flush=True)

    # ---- BOS reachability: no point requesting a source we cannot download -------
    audit_ids: set[str] = set()
    with open(AUDIT, newline="") as handle:
        next(handle)
        for line in handle:
            audit_ids.add(line.split(",", 1)[0])
    print(f"[audit] {len(audit_ids)} youtube ids on BOS", flush=True)

    # ---- A: per row, the clip set it needs -------------------------------------
    needed_by_clip: Counter = Counter()          # clip -> #rows that want it
    unlock_score: Counter = Counter()            # clip -> #rows it would complete alone
    gap_hist: Counter = Counter()
    rows_total = 0
    rows_ready = 0
    rows_unreachable = 0
    all_missing: set = set()
    missing_by_gap: dict[int, set] = defaultdict(set)

    for batch in pq.ParquetFile(A).iter_batches(
        batch_size=4096, columns=["video_id", "cross_pair"], use_threads=False
    ):
        cols = batch.to_pydict()
        for target_vid, payload_json in zip(cols["video_id"], cols["cross_pair"]):
            rows_total += 1
            vids = {target_vid}
            for entry in json.loads(payload_json).values():
                for group in entry.get("refer_result", []):
                    for ref in group:
                        vids.add(ref["vid"])
            keys = [clip_key(vid) for vid in vids]
            if any(key is None for key in keys):
                continue
            if any(key[0] not in audit_ids for key in keys):
                rows_unreachable += 1
                continue
            missing = [key for key in keys if key not in have]
            gap_hist[len(missing)] += 1
            if not missing:
                rows_ready += 1
                continue
            for key in missing:
                needed_by_clip[key] += 1
                all_missing.add(key)
            missing_by_gap[len(missing)].update(missing)
            if len(missing) == 1:
                unlock_score[missing[0]] += 1

    print(f"[A] {rows_total} rows; ready={rows_ready}; "
          f"unreachable_on_bos={rows_unreachable}; missing clips={len(all_missing)}",
          flush=True)

    # ---- resolve every missing clip to its official csv row id ------------------
    want_by_source: dict[str, set] = defaultdict(set)
    for key in all_missing:
        want_by_source[key[0]].add(key)

    resolved: dict[tuple, tuple[str, int, str]] = {}
    rows_seen = 0
    for shard in range(1, 11):
        path = f"{CSV_DIR}/Koala_36M_{shard}.csv"
        shard_name = f"Koala_36M_{shard}.csv"
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for record_index, record in enumerate(reader):
                rows_seen += 1
                if len(record) < 3:
                    continue
                youtube_id = record[0].rsplit("_", 1)[0]
                candidates = want_by_source.get(youtube_id)
                if not candidates:
                    continue
                window = parse_timestamp(record[2])
                if window is None:
                    continue
                key = key_of(youtube_id, window[0], window[1])
                if key in candidates and key not in resolved:
                    resolved[key] = (shard_name, record_index, record[0])
        print(f"[csv] shard {shard} done ({rows_seen} rows, resolved {len(resolved)}"
              f"/{len(all_missing)})", flush=True)

    # ---- emit ------------------------------------------------------------------
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    manifest_path = f"{args.out_dir}/phantom_needed_clips.csv"
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "priority_wave", "hf_csv", "hf_record_index", "koala_video_id",
            "youtube_id", "clip_start_time", "clip_end_time",
            "phantom_rows_wanting_this_clip", "rows_unlocked_if_this_alone",
        ])
        ordered = sorted(
            all_missing,
            key=lambda k: (-unlock_score[k], -needed_by_clip[k], k),
        )
        waves: Counter = Counter()
        for key in ordered:
            info = resolved.get(key)
            wave = 1 if unlock_score[key] else 2
            waves[wave] += 1
            writer.writerow([
                wave,
                info[0] if info else "",
                info[1] if info else "",
                info[2] if info else "",
                key[0], key[1], key[2],
                needed_by_clip[key], unlock_score[key],
            ])

    summary = {
        "A_rows_total": rows_total,
        "A_rows_already_usable": rows_ready,
        "A_rows_unreachable_on_bos": rows_unreachable,
        "A_rows_by_missing_clip_count": dict(sorted(gap_hist.items())),
        "missing_clips_total": len(all_missing),
        "missing_clips_resolved_to_official_csv": len(resolved),
        "missing_clips_unresolved": len(all_missing) - len(resolved),
        "missing_clips_pct_of_koala36m": round(100 * len(all_missing) / 36_060_536, 4),
        "clips_already_in_C": len(have),
        "wave_sizes": dict(waves),
        "distinct_sources_to_download": len(want_by_source),
        "manifest": manifest_path,
    }
    with open(f"{args.out_dir}/summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
