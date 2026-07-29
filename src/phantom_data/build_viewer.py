"""Streamlit browser for a *built* Phantom-Koala dataset (stage B/C/D output on disk).

Distinct from :mod:`phantom_data.viewer`, which browses raw Phantom annotations by
decoding frames from BOS online. This one reads only local files: the 81-frame clip, the
stage C contact sheet, the reference cutouts, and the funnel decisions an index already
computed. Nothing here re-derives a quality verdict -- ``indexes/<name>/
quality_decisions.jsonl`` is the source of truth, so the browser never opens a masklet
npz and never disagrees with the index the trainer will consume.

One sample per page. Sidebar filters exist to answer one specific question: of the clips
the UltraVid ``ref_clip`` threshold rejects, how many have a genuinely bad cutout versus
merely a terse noun-phrase prompt.

Streamlit compatibility: the deployment runs 1.23.1, so this module uses
``use_column_width`` (not ``use_container_width``) on images and
``st.experimental_rerun`` (not ``st.rerun``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "/mnt/pfs/data/yuanze/phantom_koala_bboxref_v1"
DEFAULT_INDEX = "phantom_pilot_v1"
SELFCHECK_SUBDIR = "segment_real"

FILTER_ALL = "all"
FILTER_PASSED = "passed the funnel"
FILTER_REJECTED = "rejected by the funnel"
FILTER_MULTI = "multi-subject only"


# --------------------------------------------------------------------------------------
# data loading (pure, unit tested)
# --------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_records(dataset: Path, index_name: str) -> list[dict[str, Any]]:
    """Join stage C rows with stage B provenance and the index's funnel verdict.

    Every ``segmented.jsonl`` row appears, whether the index kept it or not: a rejected
    clip is exactly what the user needs to inspect. Samples with no decision recorded
    (index built from a different manifest) get ``decision=None``.
    """
    segmented = read_jsonl(dataset / "segmented.jsonl")
    extracted = {row["sample_id"]: row for row in read_jsonl(dataset / "extracted.jsonl")}
    decisions = {
        row["sample_id"]: row
        for row in read_jsonl(dataset / "indexes" / index_name / "quality_decisions.jsonl")
    }
    records = []
    for sample in segmented:
        sample_id = sample["sample_id"]
        records.append({
            "sample_id": sample_id,
            "sample": sample,
            "extracted": extracted.get(sample_id) or {},
            "decision": decisions.get(sample_id),
        })
    return records


def record_codes(record: dict[str, Any]) -> list[str]:
    decision = record.get("decision") or {}
    return list(decision.get("codes") or [])


def record_passed(record: dict[str, Any]) -> bool | None:
    decision = record.get("decision")
    return None if decision is None else bool(decision.get("passed"))


def subject_reasons(record: dict[str, Any], subject_id: int) -> list[dict[str, Any]]:
    """Reason entries that name this specific subject."""
    decision = record.get("decision") or {}
    return [
        reason for reason in decision.get("reasons") or []
        if int(reason.get("subject_id", -1)) == int(subject_id)
    ]


def available_codes(records: list[dict[str, Any]]) -> list[str]:
    codes: set[str] = set()
    for record in records:
        codes.update(record_codes(record))
    return sorted(codes)


def filter_records(records: list[dict[str, Any]], mode: str,
                   codes: list[str] | None = None) -> list[dict[str, Any]]:
    """Sidebar filter. ``codes`` narrows the rejected view to specific reason codes."""
    if mode == FILTER_PASSED:
        return [r for r in records if record_passed(r) is True]
    if mode == FILTER_REJECTED:
        rejected = [r for r in records if record_passed(r) is False]
        if codes:
            wanted = set(codes)
            rejected = [r for r in rejected if wanted & set(record_codes(r))]
        return rejected
    if mode == FILTER_MULTI:
        return [r for r in records if len(r["sample"].get("subjects") or []) > 1]
    return list(records)


def window_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Flat metadata block for the page header."""
    sample = record["sample"]
    extracted = record["extracted"]
    source = extracted.get("source") or {}
    subjects = sample.get("subjects") or []
    return {
        "sample_id": sample["sample_id"],
        "resolution": f"{sample.get('width')}x{sample.get('height')}",
        "frame_count": sample.get("frame_count"),
        "fps": extracted.get("fps") or source.get("fps"),
        "num_subjects": len(subjects),
        "seed_frames": [s.get("seed_frame_index") for s in subjects],
        "phantom_video_id": extracted.get("phantom_video_id"),
        "koala_video_id": sample.get("video_id"),
        "window_sec": (
            None if source.get("window_start_sec") is None
            else f"{source['window_start_sec']:.3f} -> {source.get('window_end_sec', 0):.3f}"
        ),
        "clip_sec": (
            None if source.get("clip_start_sec") is None
            else f"{source['clip_start_sec']:.3f} -> {source.get('clip_end_sec', 0):.3f}"
        ),
        "bos_key": None if not source else f"{source.get('bucket')}/{source.get('key')}",
        "source_fps": source.get("fps_source"),
    }


def label_for(record: dict[str, Any]) -> str:
    """Short page label used in the sample picker."""
    passed = record_passed(record)
    mark = "?" if passed is None else ("ok " if passed else "REJ")
    codes = ",".join(record_codes(record))
    suffix = f"  [{codes}]" if codes else ""
    return f"{mark}  {record['sample_id']}{suffix}"


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------


def render_subject(st, dataset: Path, record: dict[str, Any], subject: dict[str, Any],
                   thresholds: dict[str, Any]) -> None:
    reasons = subject_reasons(record, int(subject["subject_id"]))
    columns = st.columns([1, 2])
    reference = dataset / str(subject.get("object_reference") or "")
    with columns[0]:
        if reference.is_file():
            st.image(str(reference), use_column_width=True)
        else:
            st.warning(f"missing cutout: {subject.get('object_reference')}")
    with columns[1]:
        st.markdown(f"**subj{int(subject['subject_id']):02d} — `{subject.get('prompt')}`**")
        score = subject.get("ref_clip_score")
        visible = subject.get("visible_frame_count")
        total = record["sample"].get("frame_count")
        st.text(
            f"ref_clip_score      {score}   (threshold {thresholds.get('min_ref_clip_score')})\n"
            f"visible_frame_count {visible}/{total}   "
            f"(threshold {thresholds.get('min_visible_frames')})\n"
            f"max_mask_area_ratio {subject.get('max_mask_area_ratio')}\n"
            f"ref_mask_coverage   {subject.get('ref_mask_coverage')}\n"
            f"seed_frame_index    {subject.get('seed_frame_index')}"
        )
        if reasons:
            for reason in reasons:
                st.error(f"blocked by `{reason['code']}` (value {reason.get('value')})")
        else:
            st.success("passes every threshold")
        ref = (record["extracted"].get("subjects") or [{}])
        pointer = next(
            (item.get("ref") for item in ref
             if int(item.get("subject_id", -1)) == int(subject["subject_id"])),
            None,
        )
        if pointer:
            st.caption(
                f"ref phrase: `{pointer.get('bbox_cls')}` | ref t={pointer.get('abs_time')}s "
                f"| ref clip `{pointer.get('phantom_vid')}`"
            )


def render_sample(st, dataset: Path, record: dict[str, Any],
                  thresholds: dict[str, Any]) -> None:
    """Render one sample page. Kept free of widget state so it can be smoke-tested."""
    sample = record["sample"]
    meta = window_summary(record)
    passed = record_passed(record)
    heading = f"### {sample['sample_id']}"
    if passed is True:
        heading += "  —  in index"
    elif passed is False:
        heading += f"  —  REJECTED ({', '.join(record_codes(record))})"
    st.markdown(heading)

    columns = st.columns([3, 2])
    with columns[0]:
        clip = dataset / str(sample.get("video") or "")
        if clip.is_file():
            st.video(str(clip))
        else:
            st.warning(f"missing clip: {sample.get('video')}")
    with columns[1]:
        st.text(
            f"resolution   {meta['resolution']}\n"
            f"frames       {meta['frame_count']} @ fps {meta['fps']} "
            f"(source fps {meta['source_fps']})\n"
            f"subjects     {meta['num_subjects']}  seed frames {meta['seed_frames']}\n"
            f"window       {meta['window_sec']}\n"
            f"phantom clip {meta['clip_sec']}\n"
            f"koala id     {meta['koala_video_id']}\n"
            f"phantom id   {meta['phantom_video_id']}\n"
            f"BOS          {meta['bos_key']}"
        )
    st.markdown("**caption (training `prompt`)**")
    st.write(sample.get("clip_prompt") or "(empty)")

    sheet = dataset / "_selfcheck" / SELFCHECK_SUBDIR / f"{sample['sample_id']}.jpg"
    if sheet.is_file():
        st.markdown("**stage C contact sheet** (mask outline + box; seed frame boxed yellow)")
        st.image(str(sheet), use_column_width=True)
    else:
        st.info(f"no contact sheet at {sheet}")

    st.markdown("**subjects**")
    for subject in sample.get("subjects") or []:
        render_subject(st, dataset, record, subject, thresholds)
        st.markdown("---")

    dropped = sample.get("dropped_subjects") or []
    if dropped:
        st.markdown("**dropped during stage C**")
        st.json(dropped)
    with st.expander("raw stage C row"):
        st.json(sample)


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="Phantom-Koala built dataset", layout="wide")
    st.title("Phantom-Koala built dataset browser")
    st.caption(
        "Local files only: 81-frame clips, stage C contact sheets, reference cutouts, and "
        "the funnel verdicts an index already computed. Quality decisions are read from "
        "quality_decisions.jsonl, never recomputed here."
    )

    dataset = Path(st.sidebar.text_input(
        "dataset root", value=os.getenv("PHANTOM_BUILD_DATASET", DEFAULT_DATASET)))
    indexes = sorted(p.name for p in (dataset / "indexes").glob("*") if p.is_dir())
    default_index = os.getenv("PHANTOM_BUILD_INDEX", DEFAULT_INDEX)
    index_name = st.sidebar.selectbox(
        "index (supplies the funnel verdict)", indexes,
        index=indexes.index(default_index) if default_index in indexes else 0,
    )
    funnel = json.loads(
        (dataset / "indexes" / index_name / "funnel.json").read_text(encoding="utf-8"))
    thresholds = funnel["thresholds"]
    records = load_records(dataset, index_name)

    counts = funnel["counts"]
    st.sidebar.markdown(
        f"**funnel** `{index_name}`\n\n"
        f"- built: {counts['built']}\n"
        f"- passed: {counts['quality_passed']}\n"
        f"- removed: {counts['quality_removed']}\n"
        f"- train/eval: {counts['train']} / {counts['eval']}"
    )
    if funnel.get("threshold_deltas"):
        st.sidebar.warning(
            "relaxed thresholds: "
            + ", ".join(f"{k}={v['this_index']} (UltraVid {v['ultravid']})"
                        for k, v in funnel["threshold_deltas"].items())
        )

    mode = st.sidebar.radio("show", [FILTER_ALL, FILTER_PASSED, FILTER_REJECTED, FILTER_MULTI])
    codes: list[str] = []
    if mode == FILTER_REJECTED:
        codes = st.sidebar.multiselect("rejection reason", available_codes(records))
    visible = filter_records(records, mode, codes)
    st.sidebar.markdown(f"**{len(visible)} / {len(records)} samples match**")
    if not visible:
        st.warning("no samples match this filter")
        return

    if st.session_state.get("_filter_key") != (mode, tuple(codes), index_name):
        st.session_state["_filter_key"] = (mode, tuple(codes), index_name)
        st.session_state["_page"] = 0
    page = int(st.session_state.get("_page", 0)) % len(visible)

    navigation = st.columns([1, 1, 6])
    if navigation[0].button("prev", use_container_width=True):
        st.session_state["_page"] = page - 1
        st.experimental_rerun()
    if navigation[1].button("next", use_container_width=True):
        st.session_state["_page"] = page + 1
        st.experimental_rerun()
    picked = navigation[2].selectbox(
        f"sample ({page + 1} / {len(visible)})",
        list(range(len(visible))), index=page,
        format_func=lambda i: label_for(visible[i]),
    )
    if picked != page:
        st.session_state["_page"] = picked
        st.experimental_rerun()

    render_sample(st, dataset, visible[page], thresholds)


# ``streamlit run`` executes this file with ``__name__ == "__main__"`` on every rerun.
if __name__ == "__main__":
    render()
