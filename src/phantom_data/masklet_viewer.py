"""Browse stage C masklets: the 81-frame mask overlaid on the clip, plus per-frame geometry.

The gate viewer shows two still frames, which is the right tool for judging a *box*. It cannot
show whether a **masklet** is good, because the failures that matter in an 81-frame propagation
are temporal: the mask latches onto a different object partway through, dissolves when the subject
turns, or slides off during fast motion. A still frame from the middle of a bad masklet often
looks fine.

So this renders every frame with the mask outlined and the derived tight box drawn, as a strip you
can scrub, plus the per-frame area and box series underneath. Area collapsing to near zero and
recovering is the signature of a dropped-and-reacquired track; area doubling is a leak onto a
neighbouring object; a box that drifts monotonically while the area stays flat is the mask sliding
off the subject.

Reads ``segmented*.jsonl`` and the packed masklet npz directly, not stage D's index -- the point
is to look at stage C's output *before* deciding what the index should keep.

Nothing here recomputes a mask: every pixel comes from the npz the trainer will consume, so the
page cannot show something the data does not contain.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from phantom_data.inspect import decode_frames, read_jsonl

DEFAULT_DATASET = "/mnt/pfs/data/yuanze/phantom_koala_newbox_v1"
DEFAULT_MANIFEST = "segmented_cand_v3_30.jsonl"

#: Mask veil and box colours. A translucent fill plus a solid outline, because a 1px outline on a
#: busy 1080p frame is invisible at strip scale, and the fill is what makes "which pixels does the
#: mask claim" readable at a glance.
#:
#: The veil is a desaturating *tint*, not neutral grey. A pure grey veil is mathematically
#: invisible wherever the frame is already mid-grey -- blending 128 towards 128 is a no-op, and
#: measured on a mid-grey test frame the overlay changed the pixel by exactly 0. Since a great many
#: of these frames are mid-tone indoor scenes, that failure would hit precisely the subjects that
#: most need looking at. A cool blue-grey shifts hue as well as luminance, so the veil survives at
#: every brightness.
MASK_FILL = (96, 128, 190)
MASK_OUTLINE = (255, 255, 255)
BOX_COLOUR = (255, 200, 60)

#: How opaque the fill is. 0.45 keeps the subject's own detail visible underneath -- the whole
#: judgement is whether the mask edge follows the object's edge, which needs both visible.
MASK_ALPHA = 0.45

#: Frames shown in the strip. The full 81 would not fit legibly; these are evenly spaced with the
#: seed frame always included, since the seed is where the box came from.
STRIP_FRAMES = 8


def load_manifest(dataset: Path, manifest: str) -> list[dict[str, Any]]:
    path = dataset / manifest
    if not path.is_file():
        return []
    return read_jsonl(path)


def unpack_masklet(dataset: Path, sample_id: str) -> tuple[np.ndarray | None, list[int]]:
    """The unpacked ``(subjects, frames, H, W)`` bool masklet and its subject ids.

    Bit-packed on disk along the width axis -- 81 frames of 1080p bool would be 170 MB raw
    against a few hundred KB packed -- so unpacking has to trim back to ``mask_width``, which the
    npz stores precisely because the packed width rounds up to a byte boundary.
    """
    path = dataset / "masklets" / f"{sample_id}.npz"
    if not path.is_file():
        return None, []
    data = np.load(path)
    packed = data["subject_masks_packed"]
    width = int(data["mask_width"])
    masks = np.unpackbits(packed, axis=-1)[..., :width].astype(bool)
    ids = [int(v) for v in data["source_subject_ids"]]
    return masks, ids


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask, by 1-pixel erosion difference.

    An outline rather than a translucent fill: a fill hides the subject underneath, and the
    judgement being made is whether the mask edge follows the object's edge.
    """
    from scipy import ndimage

    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, iterations=2)
    return mask & ~eroded


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def draw_overlay(frame: np.ndarray, mask: np.ndarray, box: list[int] | None,
                 thickness: int = 3, alpha: float = MASK_ALPHA) -> np.ndarray:
    """One frame with the mask veiled in translucent grey, outlined, and its tight box drawn.

    The fill is blended in float and rounded once, rather than composited in uint8, so a partly
    transparent veil cannot quantise away on dark subjects -- a mask over near-black pixels is
    exactly where a too-subtle overlay becomes invisible.
    """
    out = np.array(frame, dtype=np.uint8, copy=True)
    if mask.any():
        region = out[mask].astype(np.float32)
        veil = np.asarray(MASK_FILL, dtype=np.float32)
        out[mask] = np.round(region * (1.0 - alpha) + veil * alpha).astype(np.uint8)
        # A solid outline on top of the fill: the fill answers "which pixels", the outline makes
        # the boundary crisp enough to compare against the object's edge at strip scale.
        out[mask_outline(mask)] = MASK_OUTLINE
    if box:
        x1, y1, x2, y2 = box
        x2 = min(x2, out.shape[1]); y2 = min(y2, out.shape[0])
        for offset in range(thickness):
            if y1 + offset < out.shape[0]:
                out[y1 + offset, x1:x2] = BOX_COLOUR
            if y2 - 1 - offset >= 0:
                out[y2 - 1 - offset, x1:x2] = BOX_COLOUR
            if x1 + offset < out.shape[1]:
                out[y1:y2, x1 + offset] = BOX_COLOUR
            if x2 - 1 - offset >= 0:
                out[y1:y2, x2 - 1 - offset] = BOX_COLOUR
    return out


def frame_series(masks: np.ndarray) -> dict[str, Any]:
    """Per-frame geometry of one subject's masklet: the numbers that expose temporal failures."""
    areas = [int(m.sum()) for m in masks]
    boxes = [bbox_from_mask(m) for m in masks]
    present = [i for i, a in enumerate(areas) if a > 0]
    positive = [a for a in areas if a > 0]
    median = float(np.median(positive)) if positive else 0.0
    return {
        "areas": areas,
        "boxes": boxes,
        "frames_with_mask": len(present),
        "total_frames": len(areas),
        "first_frame": present[0] if present else None,
        "last_frame": present[-1] if present else None,
        # Gaps inside the visible span mean the track dropped and was reacquired -- a different
        # and worse thing than a subject that simply leaves the frame at the end.
        "interior_gaps": [i for i in range(present[0], present[-1] + 1)
                          if areas[i] == 0] if present else [],
        "area_median": median,
        # Relative to the median rather than to the max: a single leaked frame would inflate the
        # max and make the whole series look stable by comparison.
        "area_min_ratio": round(min(positive) / median, 4) if positive else 0.0,
        "area_max_ratio": round(max(positive) / median, 4) if positive else 0.0,
        "union_box": bbox_from_mask(masks.any(axis=0)) if masks.size else None,
    }


def strip_indices(total: int, seed: int | None, count: int = STRIP_FRAMES) -> list[int]:
    """Evenly spaced frame indices, always including the seed frame."""
    if total <= count:
        return list(range(total))
    picks = {int(round(i * (total - 1) / (count - 1))) for i in range(count)}
    if seed is not None and 0 <= seed < total:
        picks.add(int(seed))
    return sorted(picks)


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="masklets", layout="wide")
    st.title("Stage C masklets: does the mask hold across 81 frames?")

    dataset = Path(st.sidebar.text_input(
        "dataset root", value=os.getenv("PHANTOM_MASKLET_DATASET", DEFAULT_DATASET)))
    manifest = st.sidebar.text_input(
        "segmented manifest", value=os.getenv("PHANTOM_MASKLET_MANIFEST", DEFAULT_MANIFEST))

    rows = load_manifest(dataset, manifest)
    if not rows:
        st.warning(f"no rows in {dataset / manifest}. Run stage C first:\n\n"
                   f"`python -m phantom_data.build.segment --dataset {dataset} "
                   f"--input gated_cand_v3.jsonl --limit 30`")
        return

    flat: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        for position, _subject in enumerate(row.get("subjects") or []):
            flat.append((index, position))
    if not flat:
        st.warning("the manifest has no subjects")
        return

    st.sidebar.markdown(f"**{len(rows)} samples · {len(flat)} subjects**")
    show_mask = st.sidebar.checkbox("veil the mask (translucent grey)", value=True)
    show_box = st.sidebar.checkbox("draw the mask's tight box (yellow)", value=True)
    alpha = st.sidebar.slider(
        "mask opacity", 0.0, 1.0, MASK_ALPHA, 0.05,
        help="At 1.0 the mask becomes a solid silhouette, which shows its shape but hides "
             "whether the edge follows the subject. Around 0.45 shows both.")
    frame_count = st.sidebar.slider("frames in the strip", 4, 16, STRIP_FRAMES, 1)

    page = int(st.session_state.get("_masklet_page", 0)) % len(flat)
    columns = st.columns([1, 1, 6])
    if columns[0].button("prev", use_container_width=True):
        page = (page - 1) % len(flat)
    if columns[1].button("next", use_container_width=True):
        page = (page + 1) % len(flat)
    st.session_state["_masklet_page"] = page
    columns[2].markdown(f"**subject {page + 1} / {len(flat)}**")

    row_index, subject_position = flat[page]
    row = rows[row_index]
    subject = (row.get("subjects") or [])[subject_position]
    sample_id = str(row["sample_id"])

    masks, ids = unpack_masklet(dataset, sample_id)
    if masks is None:
        st.error(f"no masklet npz for {sample_id}")
        return
    subject_id = int(subject.get("subject_id", 0))
    plane = ids.index(subject_id) if subject_id in ids else subject_position
    subject_masks = masks[plane]

    series = frame_series(subject_masks)
    seed = subject.get("seed_frame_index")

    st.markdown(f"### {sample_id} · subj{subject_id:02d}")
    st.markdown(f"**phrase:** `{subject.get('phrase')}`")

    healthy = (series["frames_with_mask"] == series["total_frames"]
               and not series["interior_gaps"]
               and 0.4 <= series["area_min_ratio"] and series["area_max_ratio"] <= 2.5)
    (st.success if healthy else st.warning)(
        f"mask present on {series['frames_with_mask']}/{series['total_frames']} frames  ·  "
        f"area vs median: min {series['area_min_ratio']:.2f}x, max "
        f"{series['area_max_ratio']:.2f}x  ·  interior gaps: "
        f"{series['interior_gaps'] or 'none'}")
    st.caption(
        "Area far below the median on some frames means the mask partly dissolved; far above "
        "means it leaked onto something else; an interior gap means the track dropped and was "
        "reacquired. A still frame from a bad masklet usually looks fine, which is why these "
        "numbers are here."
    )

    video = dataset / str(row["video"])
    if not video.is_file():
        st.error(f"clip missing: {video}")
        return
    frames = decode_frames(video)

    indices = strip_indices(len(subject_masks), seed, frame_count)
    grid = st.columns(min(4, len(indices)))
    for position, frame_index in enumerate(indices):
        if frame_index >= len(frames):
            continue
        mask = subject_masks[frame_index]
        box = series["boxes"][frame_index]
        image = draw_overlay(frames[frame_index],
                            mask if show_mask else np.zeros_like(mask),
                            box if show_box else None, alpha=alpha)
        label = f"frame {frame_index}"
        if seed is not None and frame_index == int(seed):
            label += "  ← seed (the box came from here)"
        area = series["areas"][frame_index]
        ratio = area / series["area_median"] if series["area_median"] else 0
        grid[position % len(grid)].image(
            image, use_column_width=True,
            caption=f"{label}  ·  area {area} ({ratio:.2f}x median)")

    with st.expander("per-frame area and box"):
        st.table([
            {"frame": i, "area": series["areas"][i],
             "vs median": (f"{series['areas'][i] / series['area_median']:.2f}x"
                           if series["area_median"] else "-"),
             "box": series["boxes"][i]}
            for i in range(series["total_frames"])
        ])

    with st.expander("stage C's own record for this subject"):
        st.table([{"field": key, "value": str(value)}
                  for key, value in sorted(subject.items())
                  if not isinstance(value, (dict, list))])


if __name__ == "__main__":
    render()
