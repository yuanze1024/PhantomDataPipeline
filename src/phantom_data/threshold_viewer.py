"""Pick the identity threshold by looking at what each score actually means.

``identity_min = 0.5`` currently decides which 14% of the pilot is discarded, and it was chosen by
guess. The honest fix is human labels, but a threshold can also be *read off* the data if the page
makes the right comparison easy: not "is this subject good" one at a time, but "where along the
score axis does good turn into bad".

So subjects are ordered by score and the page shows a **window** around a chosen cut -- a few
subjects just above it and a few just below -- side by side. That is the comparison a threshold is:
everything above ships, everything below is thrown away, and the question is only whether the
boundary falls in the right place. Scanning subjects in isolation cannot answer it, because a
mediocre subject looks mediocre whether the cut is at 0.4 or 0.6.

Reads the same ``gate_report.json`` the gate does, and computes nothing: the scores shown are the
ones the pipeline produced.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "/mnt/pfs/data/yuanze/phantom_koala_newbox_v1"
DEFAULT_OUT_ROOT = "_cand_v3"

#: Subjects shown on each side of the cut. Enough to see a trend, few enough to fit on a screen
#: without scrolling -- the comparison only works if both sides are visible at once.
WINDOW = 4

#: Histogram resolution for the score distribution. 20 bins over [0, 1] puts each bin at 0.05,
#: which matches the granularity anyone would actually set a threshold to.
BINS = 20


def load_report(dataset: Path, out_root: str) -> dict[str, Any]:
    import json

    path = dataset / out_root / "gate_report.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def scored_subjects(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Subjects that have an identity score, worst first.

    Ascending because the interesting end is the bottom: the threshold's job is to cut off a tail,
    and a page that opens on the best subjects shows nothing about where that tail begins.
    """
    subjects = [s for s in (report.get("subjects") or [])
                if s.get("rule_identity") is not None]
    return sorted(subjects, key=lambda s: float(s["rule_identity"]))


def histogram(subjects: list[dict[str, Any]], bins: int = BINS) -> list[dict[str, Any]]:
    """Score distribution as bin counts, with a cumulative "discarded below this" column.

    The cumulative column is the one that matters for a threshold decision: it converts a bin
    count into the concrete cost of cutting there, which is the number the decision is about.
    """
    rows: list[dict[str, Any]] = []
    values = [float(s["rule_identity"]) for s in subjects]
    total = len(values)
    cumulative = 0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        # Last bin closes on the right so a perfect 1.0 is not dropped from the table.
        count = sum(1 for v in values
                    if low <= v < high or (index == bins - 1 and v == high))
        cumulative += count
        rows.append({
            "score range": f"{low:.2f} – {high:.2f}",
            "subjects": count,
            "cut here → discarded": f"{cumulative - count} ({(cumulative - count) / max(1, total):.0%})",
        })
    return rows


def window_around(subjects: list[dict[str, Any]], cut: float,
                  window: int = WINDOW) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(just below the cut, just above the cut)``, nearest the boundary first.

    "Nearest first" so the two lists read outward from the boundary in both directions -- the
    subjects that decide whether the cut is right are the ones closest to it.
    """
    below = [s for s in subjects if float(s["rule_identity"]) < cut]
    above = [s for s in subjects if float(s["rule_identity"]) >= cut]
    return below[-window:][::-1], above[:window]


def crop_box(frame, box):
    """Plain crop, or None. Local copy so the viewer needs no pipeline import."""
    import numpy as np

    if not box:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray(frame)[y1:y2, x1:x2]


def render_card(st, dataset: Path, out_root: str, subject: dict[str, Any],
                cut: float) -> None:
    """One subject as the pair of crops the identity score was computed from.

    The crops, not the full frames: the score is a comparison of these two images, so showing
    anything else invites judging the subject on evidence the number never saw.
    """
    import numpy as np
    from PIL import Image

    score = float(subject["rule_identity"])
    kept = score >= cut
    root = dataset / out_root
    columns = st.columns([1, 1])
    for column, (side, frame_key) in zip(columns, (("ref", "ref_frame"),
                                                   ("seed", "seed_frame"))):
        path = root / str(subject.get(frame_key) or "")
        box = subject.get(f"chosen_box_{side}")
        if not path.is_file() or not box:
            column.caption(f"{side}: missing")
            continue
        crop = crop_box(np.asarray(Image.open(path).convert("RGB")), box)
        if crop is None:
            column.caption(f"{side}: empty crop")
            continue
        column.image(crop, use_column_width=True,
                     caption="reference" if side == "ref" else "target")
    marker = "KEEP" if kept else "DROP"
    ranking = subject.get("ranking") or {}
    st.markdown(
        f"`{score:.3f}` **{marker}**  ·  {subject.get('sample_id', '')[:12]} "
        f"subj{int(subject.get('subject_id', 0)):02d}  ·  `{subject.get('dis')}`  ·  "
        f"margin {ranking.get('margin')}  ·  "
        f"detector won: {ranking.get('used_detector')}")


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="identity threshold", layout="wide")
    st.title("Where should the identity threshold go?")

    dataset = Path(st.sidebar.text_input(
        "dataset root", value=os.getenv("PHANTOM_THRESH_DATASET", DEFAULT_DATASET)))
    out_root = st.sidebar.text_input(
        "report dir", value=os.getenv("PHANTOM_THRESH_OUT_ROOT", DEFAULT_OUT_ROOT))

    report = load_report(dataset, out_root)
    subjects = scored_subjects(report)
    if not subjects:
        st.warning(f"no scored subjects in {dataset / out_root}/gate_report.json")
        return

    cut = st.sidebar.slider("identity_min (the cut)", 0.0, 1.0, 0.5, 0.01)
    window = st.sidebar.slider("subjects shown each side", 1, 8, WINDOW, 1)

    dropped = sum(1 for s in subjects if float(s["rule_identity"]) < cut)
    total = len(subjects)
    st.markdown(
        f"### cut at `{cut:.2f}` → keep **{total - dropped}**, discard **{dropped}** "
        f"({dropped / total:.0%}) of {total}")
    st.caption(
        "The pairs below are what the score compared. Read down the two columns: if the DROP side "
        "looks as good as the KEEP side, the cut is too high; if DROP still contains obviously "
        "matching pairs, it is too low. Judging subjects one at a time cannot settle this — a "
        "mediocre pair looks mediocre wherever the cut happens to be."
    )

    below, above = window_around(subjects, cut, window)
    left, right = st.columns(2)
    with left:
        st.markdown(f"#### just below — discarded ({len(below)} shown)")
        st.caption("nearest the cut first; these are the pairs the threshold throws away")
        for subject in below:
            render_card(st, dataset, out_root, subject, cut)
            st.markdown("---")
    with right:
        st.markdown(f"#### just above — kept ({len(above)} shown)")
        st.caption("nearest the cut first; these are the worst pairs that still ship")
        for subject in above:
            render_card(st, dataset, out_root, subject, cut)
            st.markdown("---")

    with st.expander("score distribution and the cost of each cut", expanded=True):
        st.table(histogram(subjects))

    with st.expander("every subject, worst first"):
        st.table([
            {"identity": round(float(s["rule_identity"]), 4),
             "verdict": "KEEP" if float(s["rule_identity"]) >= cut else "DROP",
             "phrase": s.get("dis"),
             "margin": (s.get("ranking") or {}).get("margin"),
             "detector won": (s.get("ranking") or {}).get("used_detector"),
             "sample": f"{str(s.get('sample_id'))[:12]} subj{int(s.get('subject_id', 0)):02d}"}
            for s in subjects
        ])


if __name__ == "__main__":
    render()
