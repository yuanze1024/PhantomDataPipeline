"""Streamlit browser for Phantom-Data annotations aligned to Koala videos on BOS.

One sample per page. Everything reads online from BOS; nothing is downloaded to disk.
No error handling by design — bugs should surface loudly.
"""
from __future__ import annotations

import os
import random

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from phantom_data.bos import FrameGrabber, load_aksk, make_client
from phantom_data.dataset import PhantomIndex, scale_bbox

TARGET_COLOR = "#22c55e"  # green
REF_COLOR = "#22d3ee"  # cyan


@st.cache_resource(show_spinner="Building Phantom index (scans meta_info + audit csv once)...")
def get_index(data_dir: str, audit_csv: str) -> PhantomIndex:
    return PhantomIndex(
        filtered_parquet=f"{data_dir}/koala36M_multi_ref_merged_filtered.parquet",
        meta_parquet=f"{data_dir}/koala36M_multi_ref_meta_info_merged.parquet",
        audit_csv=audit_csv,
    )


@st.cache_resource(show_spinner="Connecting to BOS...")
def get_grabber() -> FrameGrabber:
    ak, sk = load_aksk()
    return FrameGrabber(make_client(ak, sk))


def draw_boxes(frame: np.ndarray, boxes: list[tuple[list[float], str, str]]) -> Image.Image:
    """Draw scaled boxes + labels on a frame. Each entry is (scaled_box, color, label)."""
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box, color, label in boxes:
        draw.rectangle(tuple(box), outline=color, width=6)
        text_xy = (box[0] + 4, max(0.0, box[1] - 18))
        draw.text(text_xy, label, fill=color)
    return image


def render_clip(grabber: FrameGrabber, clip: dict, color: str, label: str, caption: str) -> None:
    """Grab one clip's frame, scale its box against the real decoded dims, draw, show."""
    if clip["bos"] is None:
        st.error(f"unresolved BOS key for vid {clip['vid']}")
        return
    bucket, key = clip["bos"]["bucket"], clip["bos"]["key"]
    W, H = grabber.video_dims(bucket, key)
    frame = grabber.grab(bucket, key, clip["abs_time"])
    scaled = scale_bbox(clip["bbox_loc"], W, H)
    image = draw_boxes(frame, [(scaled, color, label)])
    st.image(image, caption=caption, use_column_width=True)


def render() -> None:
    st.set_page_config(page_title="Phantom-Data browser", layout="wide")
    st.title("Phantom-Data annotation browser")
    st.caption(
        "Phantom-Data cross_pair annotations aligned to Koala source videos on Baidu BOS. "
        "Frames are decoded online over presigned URLs; nothing is downloaded to disk."
    )

    data_dir = st.sidebar.text_input(
        "Phantom data dir",
        value=os.getenv("PHANTOM_DATA_DIR", "/mnt/pfs/users/yuanze/datasets/phantom_data_koala36m"),
    )
    audit_csv = st.sidebar.text_input(
        "BOS audit csv",
        value=os.getenv(
            "PHANTOM_AUDIT_CSV",
            "/mnt/pfs/share/pengchunli/koala36m_thordata_audit/thordata_bos_videos.csv",
        ),
    )
    index = get_index(data_dir, audit_csv)
    grabber = get_grabber()

    if "row_idx" not in st.session_state:
        st.session_state.row_idx = 0

    st.sidebar.markdown(f"**{index.num_rows:,}** samples")
    row_idx = st.sidebar.number_input(
        "Sample index", min_value=0, max_value=index.num_rows - 1,
        value=st.session_state.row_idx, step=1,
    )
    st.session_state.row_idx = int(row_idx)
    columns = st.sidebar.columns(3)
    if columns[0].button("Prev", use_container_width=True):
        st.session_state.row_idx = max(0, st.session_state.row_idx - 1)
        st.experimental_rerun()
    if columns[1].button("Next", use_container_width=True):
        st.session_state.row_idx = min(index.num_rows - 1, st.session_state.row_idx + 1)
        st.experimental_rerun()
    if columns[2].button("Random", use_container_width=True):
        st.session_state.row_idx = random.randint(0, index.num_rows - 1)
        st.experimental_rerun()

    sample = index.build_sample(st.session_state.row_idx)

    st.write(
        {
            "video_id": sample["video_id"],
            "youtube_url": sample["youtube_url"],
            "bos_key": None if sample["bos"] is None else sample["bos"]["key"],
            "start": sample["start"],
            "end": sample["end"],
            "meta_dims": [sample["meta_width"], sample["meta_height"]],
            "fps": sample["fps"],
        }
    )
    st.subheader(sample["caption"])

    for phrase in sample["phrases"]:
        st.markdown(f"### {phrase['phrase']}")
        left, right = st.columns(2)
        with left:
            st.caption("target (obj_from_tgt_video)")
            for target in phrase["targets"]:
                render_clip(
                    grabber, target, TARGET_COLOR, phrase["phrase"],
                    caption=f"{target['bbox_cls']} @ t={target['abs_time']:.2f}s",
                )
        with right:
            st.caption("references (refer_result)")
            for ref in phrase["references"]:
                score = ref.get("score")
                label = ref["bbox_cls"] if score is None else f"{ref['bbox_cls']} {score:.2f}"
                render_clip(
                    grabber, ref, REF_COLOR, label,
                    caption=f"{label} @ t={ref['abs_time']:.2f}s  ({ref['vid']})",
                )

    st.divider()
    st.subheader("Raw cross_pair")
    st.json(sample["cross_pair"])


if __name__ == "__main__":
    render()
