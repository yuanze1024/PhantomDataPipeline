"""Streaming join of bbox annotations (table A) onto true source resolutions (table B).

Both parquet files are pathological for eager readers:

* A ``koala36M_multi_ref_merged_filtered.parquet`` -- 651,031 rows in ONE row group
  (1.09 GB uncompressed). ``read_table`` / ``read_row_group(0)`` is a full
  materialization and will blow memory.
* B ``koala36M_multi_ref_meta_info_merged.parquet`` -- 1,129,029 rows, 2 huge row
  groups.

So everything here goes through ``pq.ParquetFile(...).iter_batches(...)`` with
``use_threads=False`` (the filesystem is a shared juicefs mount; this tool is not
allowed to hog it). Deliberately does NOT reuse ``phantom_data.dataset.PhantomIndex``
or ``phantom_data.build.plan.read_filtered_rows`` -- both eagerly read whole tables.
"""
from __future__ import annotations

import json
from typing import Iterator

import pyarrow.parquet as pq

DATA_DIR = "/mnt/pfs/users/yuanze/datasets/phantom_data_koala36m"
FILTERED_PARQUET = f"{DATA_DIR}/koala36M_multi_ref_merged_filtered.parquet"
META_PARQUET = f"{DATA_DIR}/koala36M_multi_ref_meta_info_merged.parquet"

# ---------------------------------------------------------------------------
# aspect buckets
# ---------------------------------------------------------------------------

BUCKET_16_9 = "16:9"
BUCKET_4_3 = "4:3"
BUCKET_1_1 = "1:1"
BUCKET_19_1 = "1.9:1"
BUCKET_WIDE = ">=2.2:1"

#: Buckets that are rare in the data and drive the sampling early-stop.
RARE_BUCKETS = (BUCKET_1_1, BUCKET_4_3, BUCKET_WIDE)


def aspect_ratio(w: int, h: int) -> float:
    """``w/h``, or ``0.0`` for a degenerate height (never raises)."""
    if not h:
        return 0.0
    return float(w) / float(h)


def long_edge_needed(box, w: int, h: int) -> float:
    """Smallest isotropic canvas long edge that contains ``box`` for a ``w x h`` source.

    An isotropic canvas with long edge ``L`` scales the source by ``L / max(w, h)``, so
    it measures ``L x L*h/w`` for a landscape source and ``L*w/h x L`` for a portrait
    one. A box ``[x1, y1, x2, y2]`` fits iff ``x2 <= canvas_w`` and ``y2 <= canvas_h``,
    which inverts to:

    * landscape (``w >= h``): ``L >= x2``      and ``L >= y2 * w / h``
    * portrait  (``h > w``) : ``L >= x2 * h / w`` and ``L >= y2``

    Both collapse into one expression, using ``m = max(w, h)``::

        L_needed = max(x2 * m / w, y2 * m / h)

    (landscape has ``m == w`` so the first term is ``x2``; portrait has ``m == h`` so the
    second term is ``y2``.)

    Returns ``0.0`` for a degenerate source resolution rather than raising.
    """
    if w <= 0 or h <= 0:
        return 0.0
    _x1, _y1, x2, y2 = box
    m = float(max(w, h))
    return max(float(x2) * m / float(w), float(y2) * m / float(h))


def aspect_bucket(w: int, h: int) -> str:
    """Coarse aspect-ratio bucket label for a source resolution.

    Tolerances are absolute on the ratio: 16:9 and 4:3 and 1:1 get +/-0.02, 1.9:1 is
    the open band 1.85-1.95, anything >= 2.2 collapses into one wide bucket. Ratios
    that match nothing get ``"other:<ratio rounded to 2dp>"`` so the report can still
    show what showed up rather than dropping it on the floor.
    """
    ratio = aspect_ratio(w, h)
    if ratio <= 0.0:
        return "other:0.0"
    if abs(ratio - 1.0) <= 0.02:
        return BUCKET_1_1
    if abs(ratio - (4.0 / 3.0)) <= 0.02:
        return BUCKET_4_3
    if abs(ratio - (16.0 / 9.0)) <= 0.02:
        return BUCKET_16_9
    if 1.85 <= ratio <= 1.95:
        return BUCKET_19_1
    if ratio >= 2.2:
        return BUCKET_WIDE
    return f"other:{round(ratio, 2)}"


# ---------------------------------------------------------------------------
# table B: vid -> (width, height)
# ---------------------------------------------------------------------------


def build_vid_wh(
    meta_parquet: str = META_PARQUET,
    batch_size: int = 131072,
    progress_every: int = 0,
    log=None,
) -> dict[str, tuple[int, int]]:
    """Stream table B and return ``vid -> (width, height)``.

    ~1.13M entries. Footprint is roughly 150-250 MB in CPython (dict slots + one
    str + one 2-tuple of small ints per row), which is acceptable and far cheaper
    than the dict-of-dicts that ``PhantomIndex._build_vid2meta`` builds. Only three
    columns are touched, so the ``youtube_url`` / ``timestamp`` / ``fps`` payload is
    never decoded.
    """
    parquet = pq.ParquetFile(meta_parquet)
    out: dict[str, tuple[int, int]] = {}
    seen = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["vid", "width", "height"], use_threads=False
    ):
        vids = batch.column(0).to_pylist()
        widths = batch.column(1).to_pylist()
        heights = batch.column(2).to_pylist()
        for vid, width, height in zip(vids, widths, heights):
            if vid is None or width is None or height is None:
                continue
            out[vid] = (int(width), int(height))
        seen += batch.num_rows
        if progress_every and log is not None and seen % progress_every < batch_size:
            log(f"[build_vid_wh] rows={seen} entries={len(out)}")
    return out


# ---------------------------------------------------------------------------
# bbox_loc unwrapping
# ---------------------------------------------------------------------------


def unwrap_bbox(raw) -> list[float] | None:
    """Normalize a ``bbox_loc`` to a flat ``[x1, y1, x2, y2]``.

    Targets store it flat, refs store it one level deeper (``[[x1,y1,x2,y2]]``), and
    nobody has promised those two shapes are the only ones. Peel list nesting until
    the first element is a scalar, then sanity-check the arity and the numeric type.
    Returns ``None`` for anything malformed -- callers count, they don't crash.
    """
    box = raw
    depth = 0
    while isinstance(box, (list, tuple)) and box and isinstance(box[0], (list, tuple)):
        box = box[0]
        depth += 1
        if depth > 8:  # pathological nesting; give up rather than spin
            return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    out: list[float] = []
    for coord in box:
        if isinstance(coord, bool) or not isinstance(coord, (int, float)):
            return None
        out.append(float(coord))
    return out


# ---------------------------------------------------------------------------
# table A: stream boxes joined to source W/H
# ---------------------------------------------------------------------------


class JoinStats:
    """Mutable counters filled in by :func:`iter_boxes` while it streams."""

    def __init__(self) -> None:
        self.rows_read = 0
        self.rows_bad_json = 0
        self.boxes_emitted = 0
        self.boxes_malformed = 0
        self.boxes_no_vid = 0
        self.boxes_vid_unknown = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "rows_read": self.rows_read,
            "rows_bad_json": self.rows_bad_json,
            "boxes_emitted": self.boxes_emitted,
            "boxes_malformed": self.boxes_malformed,
            "boxes_no_vid": self.boxes_no_vid,
            "boxes_vid_unknown": self.boxes_vid_unknown,
        }


def iter_row_boxes(cross_pair_json: str, stats: JoinStats) -> Iterator[tuple]:
    """Yield ``(kind, box, vid, phrase, bbox_cls)`` for one row's ``cross_pair``.

    Pure (no I/O, no resolution lookup) so it is unit-testable. ``kind`` is
    ``"target"`` or ``"ref"``. Every box carries its OWN ``vid``: a ref box comes
    from a different clip of the same source video than the target, so the row's
    ``video_id`` is the wrong key for the resolution lookup.
    """
    try:
        cross_pair = json.loads(cross_pair_json)
    except (TypeError, ValueError):
        stats.rows_bad_json += 1
        return
    if not isinstance(cross_pair, dict):
        stats.rows_bad_json += 1
        return

    for phrase, payload in cross_pair.items():
        if not isinstance(payload, dict):
            stats.rows_bad_json += 1
            continue

        for entry in payload.get("obj_from_tgt_video") or ():
            if not isinstance(entry, dict):
                stats.boxes_malformed += 1
                continue
            box = unwrap_bbox(entry.get("bbox_loc"))
            if box is None:
                stats.boxes_malformed += 1
                continue
            yield "target", box, entry.get("vid"), phrase, entry.get("bbox_cls")

        # refer_result is a list of groups, each group a list of ref dicts. Measured
        # shape is 1 group of 1 box, but tolerate both extra groups and a flattened
        # single-level list.
        for group in payload.get("refer_result") or ():
            entries = group if isinstance(group, (list, tuple)) else (group,)
            for entry in entries:
                if not isinstance(entry, dict):
                    stats.boxes_malformed += 1
                    continue
                box = unwrap_bbox(entry.get("bbox_loc"))
                if box is None:
                    stats.boxes_malformed += 1
                    continue
                yield "ref", box, entry.get("vid"), phrase, entry.get("bbox_cls")


def iter_boxes(
    filtered_parquet: str = FILTERED_PARQUET,
    vid_wh: dict[str, tuple[int, int]] | None = None,
    batch_size: int = 2000,
    max_rows: int | None = None,
    stats: JoinStats | None = None,
) -> Iterator[tuple[str, list[float], int, int, str, str, object]]:
    """Stream table A, yielding one record per box joined to its source resolution.

    Yields ``(kind, box, src_w, src_h, vid, phrase, bbox_cls)``. Boxes whose ``vid``
    is absent from ``vid_wh`` are skipped and counted (``boxes_vid_unknown``); so are
    boxes with no ``vid`` at all (``boxes_no_vid``).

    The generator is lazy on purpose: the caller decides when to stop (rare-bucket
    saturation), and abandoning the generator releases the parquet reader.
    """
    if vid_wh is None:
        raise ValueError("vid_wh is required; build it with build_vid_wh()")
    if stats is None:
        stats = JoinStats()

    parquet = pq.ParquetFile(filtered_parquet)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["video_id", "cross_pair"], use_threads=False
    ):
        payloads = batch.column(1).to_pylist()
        for cross_pair_json in payloads:
            if max_rows is not None and stats.rows_read >= max_rows:
                return
            stats.rows_read += 1
            if cross_pair_json is None:
                stats.rows_bad_json += 1
                continue
            for kind, box, vid, phrase, bbox_cls in iter_row_boxes(cross_pair_json, stats):
                if not vid:
                    stats.boxes_no_vid += 1
                    continue
                wh = vid_wh.get(vid)
                if wh is None:
                    stats.boxes_vid_unknown += 1
                    continue
                stats.boxes_emitted += 1
                yield kind, box, wh[0], wh[1], vid, phrase, bbox_cls
