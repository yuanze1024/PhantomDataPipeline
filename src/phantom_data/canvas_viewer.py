"""Streamlit browser for the canvas-hypothesis panels rendered by ``tools/canvas_panels.py``.

Purpose is a single human decision: which coordinate protocol puts the annotation box on the
object. Numbers cannot settle it (several hypotheses produce in-frame boxes for any given
annotation), so this page puts every candidate on the same frame and gets out of the way.

Reads only the pre-rendered PNGs and ``canvas_report.json``. No BOS, no GPU, no re-derivation
of any coordinate shown -- the sheet and the table come from the same render pass.

Streamlit compatibility: the deployment runs 1.23.1, so ``use_column_width`` (not
``use_container_width``) on images and ``st.experimental_rerun`` (not ``st.rerun``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "/mnt/pfs/data/yuanze/phantom_koala_inspect100_v1"
DEFAULT_OUT_ROOT = "_canvas"

#: The protocol every existing artifact was built with, called out so the page never
#: silently implies the current default is the right answer.
CURRENT_DEFAULT = "H_768_long"


def load_report(dataset: Path, out_root: str = DEFAULT_OUT_ROOT) -> dict[str, Any]:
    path = dataset / out_root / "canvas_report.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}


def sheet_paths(dataset: Path, out_root: str, sample_id: str) -> list[Path]:
    directory = dataset / out_root / sample_id
    return sorted(directory.glob("canvas_subj*.png"))


def resolution_label(record: dict[str, Any]) -> str:
    width, height = (record.get("resolution") or [0, 0])[:2]
    if not height:
        return "?"
    ratio = width / height
    name = {round(16 / 9, 2): "16:9", round(4 / 3, 2): "4:3", 1.0: "1:1"}.get(
        round(ratio, 2), f"{ratio:.2f}")
    return f"{width}x{height} ({name})"


def label_for(record: dict[str, Any]) -> str:
    phrases = ", ".join(
        str(subject.get("phrase") or "?") for subject in record.get("subjects") or [])
    return f"{resolution_label(record):22} {record['sample_id'][:14]}  {phrases[:40]}"


def sort_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by resolution so a protocol that only works for 16:9 is obvious."""
    return sorted(records, key=lambda r: (
        -(r.get("resolution") or [0])[0], r["sample_id"]))


def candidate_table(subject: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in subject.get("candidates") or []:
        mapped = candidate.get("mapped") or []
        scales = candidate.get("scales") or [0, 0]
        rows.append({
            "hypothesis": candidate["hypothesis"],
            "in frame": "yes" if candidate.get("in_frame") else "NO",
            "sx": scales[0],
            "sy": scales[1] if len(scales) > 1 else scales[0],
            "mapped xyxy": ", ".join(f"{v:.0f}" for v in mapped),
            "formula": candidate.get("formula", ""),
        })
    return rows


def render_sample(st, dataset: Path, out_root: str, record: dict[str, Any]) -> None:
    sample_id = record["sample_id"]
    st.markdown(f"### {sample_id}")
    st.text(f"resolution   {resolution_label(record)}")

    sheets = sheet_paths(dataset, out_root, sample_id)
    if not sheets:
        st.warning(f"no rendered sheet under {dataset / out_root / sample_id}")
        return

    for subject, sheet in zip(record.get("subjects") or [], sheets):
        raw = subject.get("raw_annotation") or []
        st.markdown(
            f"**subj{int(subject['subject_id']):02d} — `{subject.get('phrase')}`**  ·  "
            f"seed frame {subject.get('seed_frame_index')}  ·  "
            f"raw annotation `[{', '.join(f'{v:.0f}' for v in raw)}]`")
        st.image(str(sheet), use_column_width=True)
        st.caption(
            "Yellow = the box that protocol implies. Red = the box falls outside the frame. "
            f"The pipeline currently uses **{CURRENT_DEFAULT}**; a correct protocol is the "
            "one whose box sits on the object named above."
        )
        st.table(candidate_table(subject))


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="bbox canvas protocol", layout="wide")
    st.title("Which bbox protocol is right?")
    st.caption(
        "One frame, one panel per candidate coordinate protocol. Pick the protocol whose "
        "box lands on the named object — that decision cannot be made from numbers, because "
        "several protocols yield in-frame boxes for any given annotation."
    )

    dataset = Path(st.sidebar.text_input(
        "dataset root", value=os.getenv("PHANTOM_CANVAS_DATASET", DEFAULT_DATASET)))
    out_root = st.sidebar.text_input(
        "render dir", value=os.getenv("PHANTOM_CANVAS_OUT_ROOT", DEFAULT_OUT_ROOT))

    report = load_report(dataset, out_root)
    records = sort_samples(report.get("samples") or [])
    if not records:
        st.warning(
            f"no canvas_report.json under {dataset / out_root}. Render some first:\n\n"
            f"`python tools/canvas_panels.py --dataset {dataset} --sample <sample_id>`")
        return

    st.sidebar.markdown(
        f"**{len(records)} samples**\n\n"
        f"candidates: {len(report.get('hypotheses') or [])}\n\n"
        f"current default: `{CURRENT_DEFAULT}`"
    )
    if report.get("failures"):
        st.sidebar.warning(f"{len(report['failures'])} sample(s) failed to render")

    if st.session_state.get("_canvas_key") != (str(dataset), out_root):
        st.session_state["_canvas_key"] = (str(dataset), out_root)
        st.session_state["_canvas_page"] = 0
    page = int(st.session_state.get("_canvas_page", 0)) % len(records)

    navigation = st.columns([1, 1, 6])
    if navigation[0].button("prev", use_container_width=True):
        st.session_state["_canvas_page"] = page - 1
        st.experimental_rerun()
    if navigation[1].button("next", use_container_width=True):
        st.session_state["_canvas_page"] = page + 1
        st.experimental_rerun()
    picked = navigation[2].selectbox(
        f"sample ({page + 1} / {len(records)})", list(range(len(records))), index=page,
        format_func=lambda i: label_for(records[i]))
    if picked != page:
        st.session_state["_canvas_page"] = picked
        st.experimental_rerun()

    render_sample(st, dataset, out_root, records[page])


if __name__ == "__main__":
    render()
