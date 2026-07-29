"""Why does the tight box cut off limbs? Attribute it to despeckling, to SAM2, or to the prompt.

The reviewer's report: the SAM2 tight boxes beat Phantom's, but a part of the body is often
outside the box. Three candidate causes, and they call for different fixes, so guessing is
expensive:

1. **Despeckling ate it.** ``segment_reference`` runs ``largest_components`` before anything
   reads the mask, and ``bbox_from_mask`` is computed on the *filtered* mask. A limb that SAM2
   split into its own connected component and that holds less than 5% of the torso's area is
   deleted as speckle, and the box shrinks to the torso. Fix: compute the box from the raw mask,
   or raise the threshold. Cheap and local.
2. **SAM2 missed it.** The mask itself never covered the limb -- an occlusion, motion blur, or a
   thin extremity the single box prompt did not disambiguate. Fix: better prompting (multimask,
   point prompts, or the video predictor's temporal context). Expensive.
3. **The prompt box clipped it.** Phantom's box already excluded the limb, and SAM2 respects a
   box prompt fairly tightly, so the mask cannot recover pixels far outside it. Fix: dilate the
   prompt before segmenting. Cheap, but risks pulling in neighbouring objects.

These are separable by measurement. For each subject this probe recomputes the mask three ways
-- raw, despeckled (what shipped), and from a dilated prompt -- and reports the bounding box of
each. If (1) is the cause, ``raw`` is materially larger than ``despeckled``. If (3) is, the
dilated-prompt box is larger than both. If neither, it is (2) and the fix is in prompting.

Restricted to subjects the reviewer has labelled, and reported split by their verdict: the
question is not "do the boxes differ" but "do they differ on the ones a human called bad".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from phantom_data import labels as labelling
from phantom_data.boxes import clamp_box, iou
from phantom_data.build import segment
from phantom_data.build.segment import bbox_from_mask, largest_components


def raw_mask(models, image_rgb: np.ndarray, box: list[float],
             device: str = "cuda") -> np.ndarray | None:
    """SAM2's mask for a box prompt with **no** component filtering.

    Deliberately duplicates ``segment_reference``'s predictor call rather than calling it,
    because the whole point is to observe the mask *before* the despeckling that function
    applies unconditionally.
    """
    predictor = models.image
    with segment._autocast(device):
        predictor.set_image(image_rgb)
        masks, _scores, _low = predictor.predict(
            box=np.asarray(box, dtype=np.float32)[None, :], multimask_output=False)
    return np.asarray(masks).reshape(-1, *image_rgb.shape[:2])[0] > 0


def dilate_box(box: list[float], width: int, height: int,
               margin: float = 0.15) -> list[int] | None:
    """Grow a box by ``margin`` of its own size on every side, clamped to the frame.

    Tests cause (3): if the prompt was clipping the subject, a looser prompt lets SAM2 reach the
    limb. The margin is relative rather than absolute so it scales with the subject.
    """
    clamped = clamp_box(box, width, height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return clamp_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], width, height)


def area(box: list[int] | None) -> int:
    if not box:
        return 0
    return max(0, (box[2] - box[0])) * max(0, (box[3] - box[1]))


def probe_side(models, frame: np.ndarray, prompt: Any, device: str) -> dict[str, Any]:
    """All three box variants for one side of one subject."""
    height, width = frame.shape[:2]
    clamped = clamp_box(prompt, width, height)
    if clamped is None:
        return {"error": "degenerate prompt"}

    mask = raw_mask(models, frame, clamped, device=device)
    if mask is None or not mask.any():
        return {"error": "empty mask"}
    filtered = largest_components(mask)

    box_raw = bbox_from_mask(mask)
    box_filtered = bbox_from_mask(filtered)

    grown = dilate_box(clamped, width, height)
    box_dilated = None
    if grown is not None:
        dmask = raw_mask(models, frame, grown, device=device)
        if dmask is not None and dmask.any():
            box_dilated = bbox_from_mask(largest_components(dmask))

    from scipy import ndimage
    _, n_raw = ndimage.label(mask)
    _, n_filtered = ndimage.label(filtered)

    return {
        "prompt_box": list(clamped),
        "box_raw": box_raw,
        "box_despeckled": box_filtered,
        "box_dilated_prompt": box_dilated,
        # The three attributions, as area ratios against the box that shipped.
        "raw_vs_shipped_area": round(area(box_raw) / max(1, area(box_filtered)), 4),
        "dilated_vs_shipped_area": (round(area(box_dilated) / max(1, area(box_filtered)), 4)
                                    if box_dilated else None),
        "shipped_vs_prompt_area": round(area(box_filtered) / max(1, area(clamped)), 4),
        "iou_raw_vs_shipped": (lambda v: None if v is None else round(v, 4))(
            iou(box_raw, box_filtered)),
        "components_raw": int(n_raw),
        "components_after_despeckle": int(n_filtered),
        "pixels_raw": int(mask.sum()),
        "pixels_despeckled": int(filtered.sum()),
        "pixels_lost_to_despeckle": int(mask.sum() - filtered.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report-root", default="_tighten_v1")
    parser.add_argument("--label-dir", default="_labels_tighten_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sam2-config", default=segment.DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", default=segment.DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from PIL import Image

    root = args.dataset / args.report_root
    report = json.loads((root / "gate_report.json").read_text())
    store = labelling.load_labels(args.dataset / args.label_dir)
    print(f"{len(store)} labels on disk", flush=True)

    models = segment.Models(args.sam2_config, args.sam2_checkpoint,
                            segment.DEFAULT_CLIP_MODEL, device=args.device)

    results: list[dict[str, Any]] = []
    for subject in report["subjects"]:
        key = (str(subject["sample_id"]), int(subject["subject_id"]))
        label = store.get(key)
        if label is None:
            continue
        row: dict[str, Any] = {"sample_id": key[0], "subject_id": key[1],
                               "verdict": label["verdict"], "dis": subject.get("dis"),
                               "identity": subject.get("rule_identity")}
        for side, frame_key in (("ref", "ref_frame"), ("seed", "seed_frame")):
            path = root / str(subject[frame_key])
            if not path.is_file():
                row[side] = {"error": "missing frame"}
                continue
            frame = np.asarray(Image.open(path).convert("RGB"))
            row[side] = probe_side(models, frame, subject[f"box_{side}_phantom"], args.device)
        results.append(row)
        print(f"[{len(results)}] {row['verdict']:4s} {str(row['dis'])[:26]:26s} "
              f"raw/shipped ref={row['ref'].get('raw_vs_shipped_area')} "
              f"seed={row['seed'].get('raw_vs_shipped_area')}  "
              f"lost_px ref={row['ref'].get('pixels_lost_to_despeckle')}", flush=True)
        if args.limit and len(results) >= args.limit:
            break

    def summarise(rows: list[dict[str, Any]], name: str) -> None:
        if not rows:
            print(f"  {name}: none")
            return
        vals: list[float] = []
        dil: list[float] = []
        lost: list[int] = []
        for row in rows:
            for side in ("ref", "seed"):
                d = row.get(side) or {}
                if d.get("raw_vs_shipped_area"):
                    vals.append(d["raw_vs_shipped_area"])
                if d.get("dilated_vs_shipped_area"):
                    dil.append(d["dilated_vs_shipped_area"])
                if d.get("pixels_lost_to_despeckle") is not None:
                    lost.append(d["pixels_lost_to_despeckle"])
        med = lambda v: round(float(np.median(v)), 4) if v else None
        print(f"  {name} (n={len(rows)} subjects, {len(vals)} sides):")
        print(f"      raw box / shipped box area   median={med(vals)}  max={max(vals) if vals else None}")
        print(f"      dilated-prompt / shipped     median={med(dil)}  max={max(dil) if dil else None}")
        print(f"      mask pixels lost to despeckle median={med(lost)}  max={max(lost) if lost else None}")

    print()
    print("If despeckling is the cause, 'raw / shipped' is >1 on the failures.")
    print("If the prompt was clipping, 'dilated / shipped' is >1 instead.")
    print("If both are ~1.0, SAM2 never saw the limb and the fix is in prompting.")
    summarise([r for r in results if r["verdict"] == labelling.FAIL], "FAILED by reviewer")
    summarise([r for r in results if r["verdict"] == labelling.PASS], "PASSED by reviewer")

    out = args.out or (args.dataset / args.report_root / "mask_completeness_probe.json")
    from phantom_data.inspect import atomic_write_bytes
    atomic_write_bytes(out, (json.dumps({"results": results}, ensure_ascii=False, indent=2)
                             + "\n").encode("utf-8"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
