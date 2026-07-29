"""Streamlit browser for the inspection sheets rendered by :mod:`phantom_data.inspect`.

One sample per page, three questions per page, in the order they need answering:

    is the box right?  ->  target.png    (boxes only: yellow annotation, cyan SAM2)
    is the mask clean? ->  mask.png      (mask alone, holes in red)
    is the ref right?  ->  reference.png (full ref frame + box, beside the cutout)

Reads only local PNGs and the ``metrics.json`` the renderer already wrote -- zero compute,
zero BOS access, and no re-derivation of any number shown. Distinct from
:mod:`phantom_data.build_viewer`, which needs a stage D index to supply funnel verdicts;
this one deliberately shows no pass/reject verdict because a 10-sample set cannot calibrate
a threshold.

Streamlit compatibility: the deployment runs 1.23.1, so ``use_column_width`` (not
``use_container_width``) on images and ``st.experimental_rerun`` (not ``st.rerun``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "/mnt/pfs/data/yuanze/phantom_koala_inspect_v1"
DEFAULT_OUT_ROOT = "_inspect"

#: A hole ratio at or above this is called out in red. Not a filter and not a fix -- the
#: point of this round is to decide what the threshold should be, so it only colours text.
HOLE_ATTENTION = 0.02

SORT_ID = "sample id"
SORT_VIDEO_HOLE = "worst video hole (desc)"
SORT_REF_HOLE = "worst ref hole (desc)"
SORT_CLAMP = "box clamp loss (desc)"


# --------------------------------------------------------------------------------------
# loading + ordering (pure, unit tested)
# --------------------------------------------------------------------------------------


def load_metrics(dataset: Path, out_root: str = DEFAULT_OUT_ROOT,
                 damaged: list[str] | None = None) -> list[dict[str, Any]]:
    """Every readable ``<out_root>/<sample_id>/metrics.json``, sorted by sample id.

    Unreadable files are skipped rather than raised: one truncated json used to take the
    whole page down with a ``JSONDecodeError``, which is a bad trade when the other 99
    samples are fine. ``damaged`` collects the sample ids so the page can report them.
    """
    root = dataset / out_root
    records = []
    for path in sorted(root.glob("*/metrics.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            # Typically a 0-byte file left by a full disk mid-render.
            if damaged is not None:
                damaged.append(path.parent.name)
            continue
        record["_dir"] = str(path.parent)
        records.append(record)
    return records


def worst_video_hole(record: dict[str, Any]) -> float:
    return max((subject["holes_video"]["worst"]
                for subject in record.get("subjects") or []), default=0.0)


def worst_ref_hole(record: dict[str, Any]) -> float:
    return max((subject["holes_ref_alpha"]
                for subject in record.get("subjects") or []), default=0.0)


def worst_clamp_loss(record: dict[str, Any]) -> float:
    """Largest ``area_lost_pct`` over both box kinds of every subject."""
    values = []
    for subject in record.get("subjects") or []:
        for key in ("seed_box_clamp", "ref_box_clamp"):
            report = subject.get(key)
            if report:
                values.append(float(report.get("area_lost_pct") or 0.0))
    return max(values, default=0.0)


def sort_records(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == SORT_VIDEO_HOLE:
        return sorted(records, key=worst_video_hole, reverse=True)
    if mode == SORT_REF_HOLE:
        return sorted(records, key=worst_ref_hole, reverse=True)
    if mode == SORT_CLAMP:
        return sorted(records, key=worst_clamp_loss, reverse=True)
    return sorted(records, key=lambda record: record["sample_id"])


def label_for(record: dict[str, Any]) -> str:
    return (f"{record['sample_id']}   "
            f"vid_hole {100 * worst_video_hole(record):.1f}%  "
            f"ref_hole {100 * worst_ref_hole(record):.1f}%  "
            f"clamp {worst_clamp_loss(record):.1f}%")


def clamp_line(report: dict[str, Any] | None) -> str:
    """One-line human reading of a clamp report, or why there is none."""
    if not report:
        return "(no raw annotation box recorded)"
    overflow = report["overflow_px"]
    sides = ", ".join(f"{side} {value:g}px"
                      for side, value in overflow.items() if value > 0.01)
    verdict = f"CLAMPED ({sides}), lost {report['area_lost_pct']:g}% of box area" \
        if report["clamped_any"] else "in bounds"
    return (f"raw {report['raw']} -> mapped {report['mapped']} "
            f"-> used {report['clamped']}  on {report['frame'][0]}x{report['frame'][1]}  |  "
            f"{verdict}")


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------


def render_subject_metrics(st, subject: dict[str, Any]) -> None:
    video = subject["holes_video"]
    ref_hole = subject["holes_ref_alpha"]
    st.markdown(f"**subj{int(subject['subject_id']):02d} — `{subject.get('prompt')}`**")
    st.text(
        f"seed frame          {subject.get('seed_frame_index')}\n"
        f"visible frames      {subject.get('visible_frame_count')}\n"
        f"ref_clip_score      {subject.get('ref_clip_score')}\n"
        f"ref_mask_coverage   {subject.get('ref_mask_coverage')}   "
        f"components {subject.get('ref_mask_components')}\n"
        f"max_mask_area_ratio {subject.get('max_mask_area_ratio')}"
    )
    holes = (f"video mask holes    median {100 * video['median']:.3f}%   "
             f"p90 {100 * video['p90']:.3f}%   "
             f"worst {100 * video['worst']:.2f}% @ frame {video['worst_frame']}\n"
             f"ref cutout holes    {100 * ref_hole:.2f}%")
    if video["worst"] >= HOLE_ATTENTION or ref_hole >= HOLE_ATTENTION:
        st.error(holes)
    else:
        st.text(holes)
    st.caption("seed box:  " + clamp_line(subject.get("seed_box_clamp")))
    st.caption("ref box:   " + clamp_line(subject.get("ref_box_clamp")))
    if subject.get("ref_time_sec") is not None:
        st.caption(f"ref frame taken at t={subject['ref_time_sec']}s "
                   f"from clip `{subject.get('ref_phantom_vid')}`")


def render_sample(st, record: dict[str, Any]) -> None:
    """Render one sample page. Free of widget state so it can be smoke-tested."""
    directory = Path(record["_dir"])
    st.markdown(f"### {record['sample_id']}")
    width, height = record.get("resolution") or [None, None]
    source = record.get("source") or {}
    st.text(
        f"resolution   {width}x{height}\n"
        f"frames       {record.get('frame_count')} @ fps {source.get('fps')} "
        f"(source fps {source.get('fps_source')})\n"
        f"subjects     {len(record.get('subjects') or [])}\n"
        f"rendered     frames {record.get('frames_rendered')}\n"
        f"phantom id   {record.get('phantom_video_id')}\n"
        f"BOS          {source.get('bucket')}/{source.get('key')}"
    )
    st.markdown("**caption (training `prompt`)**")
    st.write(record.get("caption") or "(empty)")

    panels = [
        ("target.png",
         "**1 · does the track hold?**  cyan = bbox of SAM2's mask on that frame. The seed "
         "frame is outlined yellow. No mask drawn, so a drifting box cannot hide behind an "
         "outline. (Re-render with `--show-annotation-box` to overlay the Phantom "
         "annotation box in yellow.)"),
        ("mask.png",
         "**2 · is the mask clean?**  same frames, same order as above. white = mask, "
         "**red = interior holes** `binary_fill_holes` would close, grey = background. "
         "Last tiles are the reference cutout alphas."),
        ("reference.png",
         "**3 · is the reference right?**  the FULL reference frame with the mapped box, "
         "beside the white-matte cutout that box produced."),
    ]
    for name, blurb in panels:
        path = directory / name
        st.markdown(blurb)
        if path.is_file():
            st.image(str(path), use_column_width=True)
        else:
            st.warning(f"missing {name}")

    st.markdown("**per-subject numbers**")
    for subject in record.get("subjects") or []:
        render_subject_metrics(st, subject)
        st.markdown("---")
    with st.expander("raw metrics.json"):
        st.json({k: v for k, v in record.items() if k != "_dir"})


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="Phantom-Koala inspection", layout="wide")
    st.title("Phantom-Koala sample inspection")
    st.caption(
        "Local PNGs only, rendered by phantom_data.inspect. No quality verdict is shown: "
        "this set is too small to calibrate a threshold, and hole filling is a decision to "
        "take after looking, not before."
    )

    dataset = Path(st.sidebar.text_input(
        "dataset root", value=os.getenv("PHANTOM_INSPECT_DATASET", DEFAULT_DATASET)))
    out_root = st.sidebar.text_input(
        "render dir", value=os.getenv("PHANTOM_INSPECT_OUT_ROOT", DEFAULT_OUT_ROOT))
    damaged: list[str] = []
    records = load_metrics(dataset, out_root, damaged=damaged)
    if damaged:
        st.warning(
            f"{len(damaged)} sample(s) have an unreadable metrics.json and were skipped "
            f"(usually a render interrupted by a full disk). Re-render them with "
            f"`python -m phantom_data.inspect --dataset {dataset} --only "
            f"{' '.join(damaged[:3])}{' ...' if len(damaged) > 3 else ''}`"
        )
    if not records:
        st.warning(f"no metrics.json under {dataset / out_root}; run "
                   f"`python -m phantom_data.inspect --dataset {dataset}` first")
        return

    mode = st.sidebar.radio("order by",
                            [SORT_ID, SORT_VIDEO_HOLE, SORT_REF_HOLE, SORT_CLAMP])
    visible = sort_records(records, mode)

    flagged = sum(1 for record in visible
                  if max(worst_video_hole(record), worst_ref_hole(record)) >= HOLE_ATTENTION)
    clamped = sum(1 for record in visible if worst_clamp_loss(record) > 0)
    st.sidebar.markdown(
        f"**{len(visible)} samples**\n\n"
        f"- holes ≥ {100 * HOLE_ATTENTION:g}%: {flagged}\n"
        f"- any box clamped: {clamped}"
    )

    if st.session_state.get("_sort_key") != (mode, out_root, str(dataset)):
        st.session_state["_sort_key"] = (mode, out_root, str(dataset))
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

    render_sample(st, visible[page])


# ``streamlit run`` executes this file with ``__name__ == "__main__"`` on every rerun.
if __name__ == "__main__":
    render()
