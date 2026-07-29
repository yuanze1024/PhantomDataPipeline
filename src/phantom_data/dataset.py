"""Pure, network-free logic for aligning Phantom-Data annotations to Koala videos.

Nothing here touches BOS, decord, or streamlit. The heavy scans (meta_info, audit
csv) run once inside ``PhantomIndex`` and are cached on the instance.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from . import canvas as canvas_module

DATA_DIR = "/mnt/pfs/users/yuanze/datasets/phantom_data_koala36m"
FILTERED_PARQUET = f"{DATA_DIR}/koala36M_multi_ref_merged_filtered.parquet"
META_PARQUET = f"{DATA_DIR}/koala36M_multi_ref_meta_info_merged.parquet"
AUDIT_CSV = "/mnt/pfs/share/pengchunli/koala36m_thordata_audit/thordata_bos_videos.csv"
BOS_BUCKET = "external-data"
#: Re-exported from :mod:`phantom_data.canvas` (the single source of truth) because
#: existing call sites import ``CANVAS`` from here.
CANVAS = canvas_module.CANVAS

_YOUTUBE_RE = re.compile(r"v=([^&]+)")


def parse_vid(vid: str) -> tuple[str, float, float]:
    """Split ``<uuid>_<start>_<end>`` into (uuid, start_seconds, end_seconds)."""
    uuid, start_str, end_str = vid.rsplit("_", 2)
    return uuid, float(start_str), float(end_str)


def youtube_id_from_url(url: str) -> str | None:
    """Extract the 11-char youtube id from a watch url (``v=<id>``)."""
    match = _YOUTUBE_RE.search(url)
    return match.group(1) if match else None


def frame_time(start: float, end: float, frame_idx_norm: str | float) -> float:
    """Absolute time (seconds) of a normalized-in-[0,1] frame position within a clip."""
    return start + float(frame_idx_norm) * (end - start)


def scale_bbox(box: list[float], W: int, H: int, canvas: float = CANVAS) -> list[float]:
    """Map a bbox from the long-edge=``canvas`` annotation frame onto a (W, H) frame.

    .. warning::
       This convention is **unverified and under active calibration**. Measurement over
       ~41k boxes shows the y axis obeys the 768 long-edge fit exactly while x overshoots
       and clamps at 768/798/800/832 (max observed 981), so a single isotropic canvas
       cannot be right. The competing hypotheses live in :mod:`phantom_data.canvas`.

    Kept as the isotropic long-edge mapping (hypothesis ``H_768_long``) because the
    already-built pilot dataset was produced with it; this is a thin wrapper so there is
    exactly one implementation of the scale math.
    """
    hypothesis = (
        canvas_module.H_768_long if float(canvas) == canvas_module.CANVAS
        else canvas_module.long_edge("H_custom_long", f"long edge = {canvas:g}", float(canvas))
    )
    return canvas_module.map_box(box, W, H, hypothesis)


def bos_key_for(youtube_id: str, ext: str) -> str:
    """BOS object key for a source video given its youtube id and file extension."""
    return f"Thordata/{youtube_id}/{youtube_id}{ext}"


class PhantomIndex:
    """Eagerly reads the filtered table; builds+caches the vid/uuid/ext lookups on demand."""

    def __init__(
        self,
        filtered_parquet: str | Path = FILTERED_PARQUET,
        meta_parquet: str | Path = META_PARQUET,
        audit_csv: str | Path = AUDIT_CSV,
    ) -> None:
        self.filtered_parquet = str(filtered_parquet)
        self.meta_parquet = str(meta_parquet)
        self.audit_csv = str(audit_csv)
        self._filtered_table = None
        self._vid2meta: dict[str, dict] | None = None
        self._uuid2youtube_id: dict[str, str] | None = None
        self._ext_of: dict[str, str] | None = None

    # ----- filtered table -----
    @property
    def _filtered(self):
        """Lazily read; the build pipeline only needs the vid/ext lookups, not the table."""
        if self._filtered_table is None:
            self._filtered_table = pq.read_table(
                self.filtered_parquet, columns=["video_id", "video_caption", "cross_pair"]
            )
        return self._filtered_table

    @property
    def num_rows(self) -> int:
        return self._filtered.num_rows

    def get_row(self, i: int) -> dict:
        return {
            "video_id": self._filtered["video_id"][i].as_py(),
            "video_caption": self._filtered["video_caption"][i].as_py(),
            "cross_pair": self._filtered["cross_pair"][i].as_py(),
        }

    # ----- meta_info: vid -> meta, uuid -> youtube_id -----
    def _build_vid2meta(self) -> None:
        columns = ["vid", "youtube_url", "timestamp", "width", "height", "fps"]
        table = pq.read_table(self.meta_parquet, columns=columns)
        vids = table["vid"].to_pylist()
        urls = table["youtube_url"].to_pylist()
        timestamps = table["timestamp"].to_pylist()
        widths = table["width"].to_pylist()
        heights = table["height"].to_pylist()
        fpss = table["fps"].to_pylist()
        vid2meta: dict[str, dict] = {}
        uuid2youtube_id: dict[str, str] = {}
        for vid, url, timestamp, width, height, fps in zip(
            vids, urls, timestamps, widths, heights, fpss
        ):
            youtube_id = youtube_id_from_url(url) if url else None
            vid2meta[vid] = {
                "youtube_url": url,
                "youtube_id": youtube_id,
                "timestamp": timestamp,
                "width": width,
                "height": height,
                "fps": fps,
            }
            uuid = vid.rsplit("_", 2)[0]
            if youtube_id is not None:
                uuid2youtube_id[uuid] = youtube_id
        self._vid2meta = vid2meta
        self._uuid2youtube_id = uuid2youtube_id

    @property
    def vid2meta(self) -> dict[str, dict]:
        if self._vid2meta is None:
            self._build_vid2meta()
        assert self._vid2meta is not None
        return self._vid2meta

    @property
    def uuid2youtube_id(self) -> dict[str, str]:
        if self._uuid2youtube_id is None:
            self._build_vid2meta()
        assert self._uuid2youtube_id is not None
        return self._uuid2youtube_id

    # ----- audit csv: youtube_id -> ext -----
    @property
    def ext_of(self) -> dict[str, str]:
        if self._ext_of is None:
            ext_of: dict[str, str] = {}
            with open(self.audit_csv, newline="") as handle:
                reader = csv.reader(handle)
                next(reader)  # header: video_id,ext,size,last_modified_utc,bos_key
                for record in reader:
                    ext_of[record[0]] = record[1]
            self._ext_of = ext_of
        return self._ext_of

    # ----- resolution -----
    def youtube_id_for_vid(self, vid: str) -> str | None:
        """Resolve a vid to a youtube id via its own meta row, falling back to uuid map."""
        meta = self.vid2meta.get(vid)
        if meta is not None and meta["youtube_id"] is not None:
            return meta["youtube_id"]
        uuid = vid.rsplit("_", 2)[0]
        return self.uuid2youtube_id.get(uuid)

    def resolve_bos_key(self, vid: str) -> tuple[str, str] | None:
        """Return (bucket, key) for a vid, or None if its youtube id has no known ext."""
        youtube_id = self.youtube_id_for_vid(vid)
        if youtube_id is None:
            return None
        ext = self.ext_of.get(youtube_id)
        if ext is None:
            return None
        return BOS_BUCKET, bos_key_for(youtube_id, ext)

    def _resolve_clip(self, vid: str, frame_idx_norm: str | float) -> dict:
        """Common per-clip resolution: parse the vid, compute abs time, resolve BOS key."""
        _, start, end = parse_vid(vid)
        resolved = self.resolve_bos_key(vid)
        return {
            "vid": vid,
            "start": start,
            "end": end,
            "frame_idx": frame_idx_norm,
            "abs_time": frame_time(start, end, frame_idx_norm),
            "bos": None if resolved is None else {"bucket": resolved[0], "key": resolved[1]},
        }

    def build_sample(self, row_idx: int) -> dict:
        """Assemble the full render view of one filtered row.

        Each noun phrase yields a target (from ``obj_from_tgt_video``) and a flat list
        of references (from the nested ``refer_result`` groups). Reference clips are
        resolved against their OWN vid's start/end, not the target's.
        """
        row = self.get_row(row_idx)
        cross_pair = json.loads(row["cross_pair"])
        phrases = []
        for phrase, payload in cross_pair.items():
            targets = []
            for target in payload.get("obj_from_tgt_video", []):
                clip = self._resolve_clip(target["vid"], target["frame_idx"])
                clip["bbox_loc"] = target["bbox_loc"]  # flat [x0,y0,x1,y1]
                clip["bbox_cls"] = target.get("bbox_cls")
                targets.append(clip)
            references = []
            for group in payload.get("refer_result", []):
                for ref in group:
                    clip = self._resolve_clip(ref["vid"], ref["frame_idx"])
                    clip["bbox_loc"] = ref["bbox_loc"][0]  # nested [[x0,y0,x1,y1]]
                    clip["bbox_cls"] = ref.get("bbox_cls")
                    clip["score"] = ref.get("scores")
                    references.append(clip)
            phrases.append({"phrase": phrase, "targets": targets, "references": references})

        target_vid = row["video_id"]
        meta = self.vid2meta.get(target_vid, {})
        resolved = self.resolve_bos_key(target_vid)
        _, start, end = parse_vid(target_vid)
        return {
            "row_idx": row_idx,
            "video_id": target_vid,
            "caption": row["video_caption"],
            "start": start,
            "end": end,
            "youtube_url": meta.get("youtube_url"),
            "youtube_id": self.youtube_id_for_vid(target_vid),
            "meta_width": meta.get("width"),
            "meta_height": meta.get("height"),
            "fps": meta.get("fps"),
            "bos": None if resolved is None else {"bucket": resolved[0], "key": resolved[1]},
            "phrases": phrases,
            "cross_pair": cross_pair,
        }
