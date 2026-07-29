"""Measure whether the DINOv2 crop policy changes the identity score enough to matter.

Nothing here changes the pipeline. ``redetect.Models.dino_embedding`` feeds the box crop to
the ``facebook/dinov2-base`` processor with its shipped configuration -- ``shortest_edge=256``
then ``center_crop=224`` -- so a tall crop (a standing person) has its *short* edge scaled to
256, which pushes the long edge past 224, and the centre crop then throws the head and the
feet away. That makes a low identity cosine ambiguous: it can mean "a different object" or
"the discriminative parts were cut off", and the number alone cannot say which.

So this tool recomputes the same cosine under several preprocessing policies and reports how
far the scores, the ranking, and -- the part that decides whether any of this matters -- the
keep/drop calls at ``IDENTITY_MIN`` move.

Policies:

``baseline``
    Literally :meth:`redetect.Models.dino_embedding`, called on the same frames and boxes.
    It is here to be *checked against the stored numbers*: if it does not reproduce
    ``dino_cos_chosen`` from ``gate_report.json``, every other column is untrustworthy and
    that mismatch is the finding, not the policy comparison.

``expand224`` / ``expand336`` / ``expand448``
    Expand the box to square **using the real surrounding pixels** (clamped at the frame
    edge, mean-grey padding only where the frame runs out), then resize the whole thing. No
    centre crop, so nothing of the subject is discarded. 224/336/448 are all multiples of 14:
    DINOv2 is a patch-14 ViT with interpolatable position embeddings, so it accepts them --
    the higher sizes are here to check that empirically rather than by assertion.

``letterbox224``
    The box crop padded to square with mean grey, then resized. This separates the two
    mechanisms that ``expand224`` mixes: it also stops cutting the subject, but it adds *no*
    real context. If ``letterbox224`` moves the score as much as ``expand224``, the gain is
    "stop cutting", not "more context"; if only ``expand224`` moves, it is the context.

**Which pixels the frames come from is not a detail.** ``gate_report.json``'s ``ref_frame`` /
``seed_frame`` are *display copies* -- stage 2 saves them as quality-92 JPEGs -- while the
stored cosine was computed on the originals (stage B's reference jpg and the clip frame
decoded with decord). Measured here: reading the display copies reproduces the stored numbers
only to a median 0.004 / max 0.043, and reading the originals reproduces them to **0.000000**.
Since a policy delta of interest is ~0.07, a 0.043 provenance error is not a rounding
difference, so ``--frame-source original`` is the default and ``saved`` is kept only to
re-measure that JPEG noise floor.

Outputs, all under ``--out`` (default ``$ROOT/outputs/dino_crop_comparison``):

  ``dino_crop_comparison.json``   per-subject scores, per-policy stats, churn, correlations
  ``panels/<rank>_<sample>.png``  the crops the model actually sees, per policy, per side

Usage (needs a GPU pod and the HF cache; ~1400 forward passes for the pilot's 140 subjects):
  python tools/compare_dino_crop.py --dataset /mnt/pfs/data/yuanze/phantom_koala_inspect100_v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from phantom_data.boxes import clamp_box, crop_box, is_box
from phantom_data.inspect import decode_frames, read_jsonl
from phantom_data.redetect import IDENTITY_MIN, Models

#: Padding colour for the region a policy asks for and the frame cannot supply. ImageNet's
#: mean in 8-bit, so the pad normalises to ~0 instead of injecting a black frame that the
#: patch embedding would read as real structure.
PAD_RGB = (124, 116, 104)

#: Panels drawn: the biggest movers, plus a few of the least-moved as a control. Without the
#: control the panels only ever show the policy changing things, which is not the question.
TOP_MOVERS = 10
CONTROLS = 3


# --------------------------------------------------------------------------------------
# pure geometry
# --------------------------------------------------------------------------------------


def expand_to_aspect(box: Sequence[float] | None, width: int, height: int,
                     aspect: float = 1.0) -> tuple[list[int], tuple[int, int, int, int]] | None:
    """Grow ``box`` to ``aspect`` (w/h) with real pixels first, padding only as a last resort.

    Returns ``(frame_box, (pad_left, pad_top, pad_right, pad_bottom))`` where ``frame_box``
    is inside the frame and the padded result has the requested aspect. None if ``box`` is
    not a box.

    The order matters and is the whole point of the policy: the deficit is taken from the
    frame around the box wherever the frame has it, the box is *shifted* rather than padded
    when it sits against an edge, and padding appears only for the part no frame pixel can
    cover. That is what makes this different from :func:`letterbox`, which pads everything.

    Only ever grows. The other way to hit a target aspect is to shrink the long side, which
    is the centre crop this exists to replace.
    """
    clamped = clamp_box(box, width, height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    box_w, box_h = x2 - x1, y2 - y1

    want_w, want_h = box_w, box_h
    if box_w < box_h * aspect:
        want_w = box_h * aspect
    else:
        want_h = box_w / aspect

    x1, x2, pad_left, pad_right = _grow_axis(x1, x2, want_w, width)
    y1, y2, pad_top, pad_bottom = _grow_axis(y1, y2, want_h, height)
    return [x1, y1, x2, y2], (pad_left, pad_top, pad_right, pad_bottom)


def _grow_axis(low: int, high: int, want: float, limit: int) -> tuple[int, int, int, int]:
    """Grow ``[low, high)`` to ``want`` within ``[0, limit]``; the shortfall becomes padding.

    Centred on the current span, then slid inwards when one side hits the frame edge -- an
    edge is a reason to take the pixels from the other side, not a reason to pad.
    """
    want_int = max(high - low, int(round(want)))
    deficit = want_int - (high - low)
    if deficit <= 0:
        return low, high, 0, 0

    new_low = low - deficit // 2
    new_high = new_low + want_int
    if new_low < 0:
        new_high += -new_low
        new_low = 0
    if new_high > limit:
        new_low -= new_high - limit
        new_high = limit
    if new_low < 0:  # the frame itself is smaller than the target: pad the remainder
        missing = -new_low
        new_low = 0
        return new_low, new_high, missing // 2, missing - missing // 2
    return new_low, new_high, 0, 0


def letterbox(crop_h: int, crop_w: int, aspect: float = 1.0) -> tuple[int, int, int, int]:
    """Symmetric padding that brings a ``crop_h`` x ``crop_w`` crop to ``aspect`` (w/h).

    Returns ``(pad_left, pad_top, pad_right, pad_bottom)``. Odd deficits put the extra pixel
    on the right/bottom, so the result is deterministic rather than centred to a half pixel.
    """
    want_w, want_h = crop_w, crop_h
    if crop_w < crop_h * aspect:
        want_w = int(round(crop_h * aspect))
    else:
        want_h = int(round(crop_w / aspect))
    dx, dy = max(0, want_w - crop_w), max(0, want_h - crop_h)
    return dx // 2, dy // 2, dx - dx // 2, dy - dy // 2


def elongation(box: Sequence[float] | None) -> float | None:
    """How far from square a box is: ``max(w/h, h/w)``, so 1.0 is square and never below."""
    if not is_box(box):
        return None
    w, h = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
    if w <= 0 or h <= 0:
        return None
    return round(max(w / h, h / w), 4)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson r, or None when either side is constant (r is undefined, not zero)."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation, via Pearson on the ranks. Ties get their average rank."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    return pearson(_ranks(xs), _ranks(ys))


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


# --------------------------------------------------------------------------------------
# policies: frame + box -> the uint8 image the model is shown
# --------------------------------------------------------------------------------------


def pad_image(image: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    """``image`` with ``(left, top, right, bottom)`` bands of :data:`PAD_RGB` around it."""
    left, top, right, bottom = pads
    if not any(pads):
        return image
    h, w = image.shape[:2]
    out = np.empty((h + top + bottom, w + left + right, 3), dtype=np.uint8)
    out[:, :] = PAD_RGB
    out[top:top + h, left:left + w] = image
    return out


def resize(image: np.ndarray, size: int) -> np.ndarray:
    """Bicubic resize to ``size`` x ``size`` -- the resample DINOv2's processor is configured with."""
    from PIL import Image

    pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    return np.asarray(pil.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def view_expand(frame: np.ndarray, box: Sequence[float] | None, size: int = 224,
                aspect: float = 1.0) -> np.ndarray | None:
    """The ``expand`` policy's model input: box grown to ``aspect`` with real pixels, resized."""
    height, width = frame.shape[:2]
    grown = expand_to_aspect(box, width, height, aspect)
    if grown is None:
        return None
    (x1, y1, x2, y2), pads = grown
    patch = np.asarray(frame[y1:y2, x1:x2], dtype=np.uint8)
    return resize(pad_image(patch, pads), size)


def view_letterbox(frame: np.ndarray, box: Sequence[float] | None, size: int = 224,
                   aspect: float = 1.0) -> np.ndarray | None:
    """The ``letterbox`` policy's model input: the plain crop padded to ``aspect``, resized."""
    patch = crop_box(frame, box)
    if patch is None:
        return None
    h, w = patch.shape[:2]
    return resize(pad_image(patch, letterbox(h, w, aspect)), size)


def view_baseline(models: Models, frame: np.ndarray,
                  box: Sequence[float] | None) -> np.ndarray | None:
    """What today's path shows the model, for the panels only.

    Runs the real processor with normalisation and rescaling switched off, so the geometry
    (shortest edge 256, centre crop 224) comes from the shipped config rather than from a
    re-implementation here that could drift from it.
    """
    from PIL import Image

    patch = crop_box(frame, box)
    if patch is None:
        return None
    processor, _model, _torch = models.dinov2
    pixels = processor(images=[Image.fromarray(patch)], do_rescale=False, do_normalize=False,
                       return_tensors="np").pixel_values[0]
    return np.asarray(np.clip(pixels.transpose(1, 2, 0), 0, 255), dtype=np.uint8)


def embed_view(models: Models, view: np.ndarray | None):
    """Normalised DINOv2 pooled embedding of an already-prepared square image.

    ``do_resize``/``do_center_crop`` off: the policy decided the geometry, and letting the
    processor resize again would silently reinstate the very step being measured. Rescale and
    normalise stay on so the tensor statistics match :meth:`Models.dino_embedding` exactly.
    """
    from PIL import Image

    if view is None:
        return None
    processor, model, torch = models.dinov2
    inputs = processor(images=[Image.fromarray(view)], do_resize=False, do_center_crop=False,
                       return_tensors="pt").to(models.device)
    with torch.inference_mode():
        pooled = model(**inputs).pooler_output
    return pooled / pooled.norm(dim=-1, keepdim=True)


def cosine(left, right) -> float | None:
    if left is None or right is None:
        return None
    return round(float((left * right).sum().item()), 6)


#: name -> (view function, whether the view needs the model's own processor)
POLICIES: dict[str, Any] = {
    "baseline": None,  # special-cased: calls Models.dino_embedding untouched
    "expand224": lambda models, frame, box: view_expand(frame, box, 224),
    "expand336": lambda models, frame, box: view_expand(frame, box, 336),
    "expand448": lambda models, frame, box: view_expand(frame, box, 448),
    "letterbox224": lambda models, frame, box: view_letterbox(frame, box, 224),
}


# --------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------


def score_subject(models: Models, ref_frame: np.ndarray, ref_box: Any,
                  seed_frame: np.ndarray, seed_box: Any,
                  policies: Sequence[str]) -> dict[str, dict[str, Any]]:
    """One subject's cosine and embedding norms under each policy, with timings."""
    out: dict[str, dict[str, Any]] = {}
    for name in policies:
        started = time.time()
        if name == "baseline":
            left = models.dino_embedding(ref_frame, ref_box)
            right = models.dino_embedding(seed_frame, seed_box)
        else:
            view = POLICIES[name]
            left = embed_view(models, view(models, ref_frame, ref_box))
            right = embed_view(models, view(models, seed_frame, seed_box))
        finite = all(bool(e.isfinite().all().item()) for e in (left, right) if e is not None)
        out[name] = {
            "cos": cosine(left, right),
            "finite": finite,
            "seconds": round(time.time() - started, 4),
        }
    return out


def summarise(rows: list[dict[str, Any]], policies: Sequence[str],
              identity_min: float = IDENTITY_MIN) -> dict[str, Any]:
    """Distributions, deltas against baseline, and -- the deciding number -- decision churn."""
    stats: dict[str, Any] = {}
    base = [r["scores"]["baseline"]["cos"] for r in rows]
    for name in policies:
        values = [r["scores"][name]["cos"] for r in rows]
        pairs = [(b, v) for b, v in zip(base, values) if b is not None and v is not None]
        deltas = [v - b for b, v in pairs]
        arr = np.asarray([v for v in values if v is not None], dtype=float)
        crossed_up = [r["sample_key"] for r, b, v in zip(rows, base, values)
                      if b is not None and v is not None
                      and b < identity_min <= v]
        crossed_down = [r["sample_key"] for r, b, v in zip(rows, base, values)
                        if b is not None and v is not None
                        and v < identity_min <= b]
        elong = [r["elongation_max"] for r, b, v in zip(rows, base, values)
                 if b is not None and v is not None and r["elongation_max"] is not None]
        elong_deltas = [v - b for r, b, v in zip(rows, base, values)
                        if b is not None and v is not None and r["elongation_max"] is not None]
        stats[name] = {
            "n": int(arr.size),
            "median": round(float(np.median(arr)), 4) if arr.size else None,
            "mean": round(float(arr.mean()), 4) if arr.size else None,
            "p05": round(float(np.percentile(arr, 5)), 4) if arr.size else None,
            "p95": round(float(np.percentile(arr, 95)), 4) if arr.size else None,
            "kept_at_threshold": int((arr >= identity_min).sum()) if arr.size else None,
            "delta_median": round(float(np.median(deltas)), 4) if deltas else None,
            "delta_mean": round(float(np.mean(deltas)), 4) if deltas else None,
            "abs_delta_median": round(float(np.median(np.abs(deltas))), 4) if deltas else None,
            "abs_delta_p95": (round(float(np.percentile(np.abs(deltas), 95)), 4)
                              if deltas else None),
            "abs_delta_max": round(float(np.max(np.abs(deltas))), 4) if deltas else None,
            "crossed_up": crossed_up,
            "crossed_down": crossed_down,
            "churn": len(crossed_up) + len(crossed_down),
            "spearman_vs_baseline": spearman([b for b, _ in pairs], [v for _, v in pairs]),
            "r_delta_vs_elongation": pearson(elong, elong_deltas),
            "r_abs_delta_vs_elongation": pearson(elong, [abs(d) for d in elong_deltas]),
            "seconds_per_subject": round(float(np.mean(
                [r["scores"][name]["seconds"] for r in rows])), 4) if rows else None,
            "all_finite": all(r["scores"][name]["finite"] for r in rows),
        }
    return stats


def reproduction_check(rows: list[dict[str, Any]], tolerance: float = 1e-3) -> dict[str, Any]:
    """Does ``baseline`` reproduce the stored ``dino_cos_chosen``?

    Checked before anything else is believed. The stored number came from a different process
    on possibly different hardware, so an exact match is not required -- but a drift beyond
    ``tolerance`` means the two are not measuring the same thing and the comparison below is
    about something other than the crop policy.
    """
    diffs = [(r["sample_key"], round(r["scores"]["baseline"]["cos"] - r["stored_dino_cos"], 6))
             for r in rows
             if r["scores"]["baseline"]["cos"] is not None and r["stored_dino_cos"] is not None]
    worst = sorted(diffs, key=lambda d: -abs(d[1]))[:5]
    absolute = [abs(d) for _, d in diffs]
    return {
        "compared": len(diffs),
        "max_abs_diff": round(max(absolute), 6) if absolute else None,
        "median_abs_diff": round(float(np.median(absolute)), 6) if absolute else None,
        "within_tolerance": bool(absolute and max(absolute) <= tolerance),
        "tolerance": tolerance,
        "worst": [{"sample_key": k, "diff": d} for k, d in worst],
    }


# --------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------


def panel(models: Models, row: dict[str, Any], ref_frame: np.ndarray, seed_frame: np.ndarray,
          policies: Sequence[str], tile: int = 224):
    """One PNG: a row per policy, reference and target as the model sees them, labelled."""
    from PIL import Image, ImageDraw

    gutter, header, label_h = 12, 48, 20
    width = gutter + 2 * (tile + gutter)
    height = header + len(policies) * (tile + label_h + gutter)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    # Three short lines rather than two long ones: at the default bitmap font the canvas fits
    # ~78 characters, and a truncated sample_id makes a panel impossible to look up.
    draw.text((gutter, 4), row["sample_key"], fill=(255, 255, 255))
    draw.text((gutter, 18), row["phrase"][:74], fill=(150, 150, 150))
    draw.text((gutter, 32),
              f"elongation ref {row['elongation_ref']} target {row['elongation_seed']}"
              f"   left = reference, right = target", fill=(170, 170, 170))

    for index, name in enumerate(policies):
        top = header + index * (tile + label_h + gutter)
        score = row["scores"][name]["cos"]
        base = row["scores"]["baseline"]["cos"]
        delta = None if score is None or base is None else round(score - base, 4)
        keep = score is not None and score >= IDENTITY_MIN
        colour = (120, 230, 140) if keep else (240, 120, 120)
        draw.text((gutter, top),
                  f"{name}   cos {score}   delta {delta}   {'KEEP' if keep else 'DROP'}",
                  fill=colour)
        sides = [(ref_frame, row["box_ref"]), (seed_frame, row["box_seed"])]
        for side, (frame, box) in enumerate(sides):
            view = (view_baseline(models, frame, box) if name == "baseline"
                    else POLICIES[name](models, frame, box))
            if view is None:
                continue
            image = Image.fromarray(view).resize((tile, tile), Image.NEAREST)
            canvas.paste(image, (gutter + side * (tile + gutter), top + label_h))
    return canvas


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def load_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Subjects with both a stored cosine and both chosen boxes, in report order."""
    rows = []
    for subject in report.get("subjects", []):
        ref_box, seed_box = subject.get("chosen_box_ref"), subject.get("chosen_box_seed")
        if not is_box(ref_box) or not is_box(seed_box):
            continue
        rows.append({
            "sample_key": f"{subject['sample_id']}#subj{int(subject.get('subject_id', 0)):02d}",
            "sample_id": subject["sample_id"],
            "subject_id": subject.get("subject_id"),
            "phrase": str(subject.get("dis") or subject.get("phrase") or ""),
            "ref_frame": subject["ref_frame"],
            "seed_frame": subject["seed_frame"],
            "box_ref": ref_box,
            "box_seed": seed_box,
            "elongation_ref": elongation(ref_box),
            "elongation_seed": elongation(seed_box),
            "elongation_max": max(elongation(ref_box) or 1.0, elongation(seed_box) or 1.0),
            "stored_dino_cos": subject.get("dino_cos_chosen"),
            "stored_verdict": subject.get("verdict"),
        })
    return rows


def read_frame(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


class FrameSource:
    """The two frames of a subject, from either provenance, with the clip decoded at most once.

    ``original`` is stage B's own pixels -- the reference jpg it wrote and the clip frame
    decoded from the mp4 -- which is what the stored cosine saw. ``saved`` is stage 2's
    display copies. Decoding a clip is by far the slow part, so consecutive subjects of the
    same sample reuse one decode; the report lists subjects grouped by sample already.
    """

    def __init__(self, dataset: Path, report_root: str, kind: str = "original") -> None:
        self.dataset, self.kind = dataset, kind
        self.frames_root = dataset / report_root
        self.extracted: dict[str, dict[str, Any]] = {}
        if kind == "original":
            for row in read_jsonl(dataset / "extracted.jsonl"):
                self.extracted[str(row["sample_id"])] = row
        self._clip: tuple[str, list[np.ndarray]] | None = None

    def _clip_frames(self, sample_id: str) -> list[np.ndarray]:
        if self._clip is None or self._clip[0] != sample_id:
            row = self.extracted[sample_id]
            self._clip = (sample_id, decode_frames(self.dataset / row["video"]))
        return self._clip[1]

    def get(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "saved":
            return (read_frame(self.frames_root / row["ref_frame"]),
                    read_frame(self.frames_root / row["seed_frame"]))
        sample_id = str(row["sample_id"])
        record = self.extracted[sample_id]
        subject = next(s for s in record["subjects"]
                       if int(s["subject_id"]) == int(row["subject_id"]))
        frames = self._clip_frames(sample_id)
        index = min(int(subject["seed_frame_index"]), len(frames) - 1)
        return read_frame(self.dataset / subject["ref"]["frame"]), frames[index]


def run(dataset: Path, out_dir: Path, report_root: str = "_redetect100",
        policies: Sequence[str] = tuple(POLICIES), limit: int | None = None,
        panels: bool = True, device: str | None = None,
        frame_source: str = "original") -> dict[str, Any]:
    report_path = dataset / report_root / "gate_report.json"
    report = json.loads(report_path.read_text())
    rows = load_rows(report)
    if limit:
        rows = rows[:limit]
    models = Models(device=device) if device else Models()
    source = FrameSource(dataset, report_root, frame_source)

    for index, row in enumerate(rows, start=1):
        ref_frame, seed_frame = source.get(row)
        row["scores"] = score_subject(models, ref_frame, row["box_ref"], seed_frame,
                                      row["box_seed"], policies)
        if index % 20 == 0 or index == len(rows):
            print(f"[compare] {index}/{len(rows)}", flush=True)

    stats = summarise(rows, policies)
    check = reproduction_check(rows)

    panel_paths: dict[str, str] = {}
    if panels:
        primary = "expand224" if "expand224" in policies else policies[-1]
        scored = [r for r in rows if r["scores"][primary]["cos"] is not None
                  and r["scores"]["baseline"]["cos"] is not None]
        ordered = sorted(scored, key=lambda r: -abs(r["scores"][primary]["cos"]
                                                    - r["scores"]["baseline"]["cos"]))
        chosen = ordered[:TOP_MOVERS] + ordered[-CONTROLS:]
        panel_dir = out_dir / "panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(chosen, start=1):
            kind = "mover" if rank <= TOP_MOVERS else "control"
            ref_frame, seed_frame = source.get(row)
            image = panel(models, row, ref_frame, seed_frame, policies)
            name = f"{rank:02d}_{kind}_{row['sample_id'][:12]}_subj{row['subject_id']}.png"
            image.save(panel_dir / name)
            panel_paths[row["sample_key"]] = str(panel_dir / name)
            print(f"[panel] {name}", flush=True)

    payload = {
        "dataset": str(dataset),
        "report": str(report_path),
        "identity_min": IDENTITY_MIN,
        "frame_source": frame_source,
        "policies": list(policies),
        "subjects": len(rows),
        "baseline_reproduction": check,
        "stats": stats,
        "panels": panel_paths,
        "rows": [{k: v for k, v in row.items() if k not in {"ref_frame", "seed_frame"}}
                 for row in rows],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dino_crop_comparison.json").write_text(json.dumps(payload, indent=1))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report-root", default="_redetect100")
    parser.add_argument("--out", type=Path,
                        default=Path("/mnt/pfs/users/yuanze/projects/2026/BboxCondition/"
                                     "outputs/dino_crop_comparison"))
    parser.add_argument("--policies", nargs="*", default=list(POLICIES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--frame-source", choices=("original", "saved"), default="original",
                       help="'original' = stage B's ref jpg + the decoded clip frame, which is "
                            "what the stored cosine saw; 'saved' = stage 2's display JPEGs, "
                            "which add ~0.004 median / 0.043 max of re-encode noise")
    args = parser.parse_args(argv)

    payload = run(args.dataset, args.out, report_root=args.report_root,
                  policies=args.policies, limit=args.limit, panels=not args.no_panels,
                  device=args.device, frame_source=args.frame_source)
    check = payload["baseline_reproduction"]
    print(json.dumps({"baseline_reproduction": check,
                      "churn": {k: v["churn"] for k, v in payload["stats"].items()},
                      "median": {k: v["median"] for k, v in payload["stats"].items()}},
                     indent=1))
    return 0 if check["within_tolerance"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
