"""Stage 2', text-free: tighten Phantom's boxes with SAM2 and write a viewer-readable report.

Replaces ``redetect_run.py`` in the box-correction slot. Same inputs (stage B's
``extracted.jsonl``), same output *shape* (a ``gate_report.json`` plus unannotated frame jpgs
that ``gate_viewer`` renders boxes onto), but no Grounding DINO, no CLIP text score, and no LLM
phrase -- see :mod:`phantom_data.tighten` for why the text-driven chain was abandoned.

    A plan -> B extract -> 2' tighten -> [label] -> 3 gate -> C segment -> D index

The report keeps ``redetect_run``'s field names where the meaning survives, because
``gate_viewer`` and ``gate_apply`` read those names:

* ``box_{side}_phantom`` -- Phantom's box mapped to frame pixels. Drawn red by the viewer.
* ``box_{side}_dis`` -- the SAM2 tight box. Drawn blue. The name is a misnomer inherited from the
  detector era ("dis" was the LLM phrase); it is kept so the viewer needs no changes, and it is
  what ``chosen_box_*`` points at.
* ``chosen_box_{ref,seed}`` -- what ships to stage C. Always the tight box here: there is no
  competing candidate to choose between any more, which is the simplification the text-free
  chain buys.

Deliberately **not** written: ``rule_clip``, ``crop_clip_*``, ``detector_score_*``,
``candidates_*``, ``iou_dis_vs_phantom``. Those either measured text-image agreement or described
a detector, and emitting them as nulls would leave the viewer's tables showing empty columns for
judges that no longer exist. ``rule_identity`` *is* written when ID-Sim is enabled, so the
identity column keeps working -- but it holds an ID-Sim similarity (1 - distance), not a DINOv2
cosine, and ``identity_metric`` records which.

Usage::

    python tools/tighten_run.py --dataset <root> --out-root _tighten_v1 --limit 6   # smoke
    python tools/tighten_run.py --dataset <root> --out-root _tighten_v1             # all
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from phantom_data import candidates, tighten
from phantom_data.build import segment
from phantom_data.inspect import atomic_write_bytes, decode_frames, read_jsonl

JPEG_QUALITY = 92
DEFAULT_OUT_ROOT = "_tighten_v1"
DEFAULT_INPUT = "extracted.jsonl"

#: Abort a run that has failed this many times without producing a single subject. A missing
#: dependency or a wrong path fails identically on every sample, and reporting it 138 times
#: costs the whole point of a smoke run.
FAILURE_STREAK_ABORT = 3

#: ID-Sim lives outside this package (third_party/id_sim) and needs its own weights cache.
IDSIM_REPO = "/mnt/pfs/users/yuanze/projects/2026/BboxCondition/third_party/id_sim"
IDSIM_CACHE = "/mnt/pfs/users/yuanze/models/id_sim_checkpoint"
IDSIM_TYPE = os.environ.get("IDSIM_TYPE", "dinov2_vitl14_cls_patch")


def write_frame(path: Path, frame: np.ndarray) -> None:
    """Store one unannotated frame. Boxes are drawn by the viewer, not baked in."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB").save(
        buffer, format="JPEG", quality=JPEG_QUALITY)
    atomic_write_bytes(path, buffer.getvalue())


class IdSim:
    """ID-Sim similarity between two crops, loaded lazily.

    Returns a *similarity* (1 - distance) rather than the raw distance, so the number is
    higher-is-better like the DINOv2 cosine it replaces. The viewer's identity slider and
    ``redetect.decide``'s ``identity >= threshold`` comparison both assume that direction, and
    flipping it here is cheaper than teaching every consumer about a second convention.
    """

    def __init__(self, device: str = "cuda", enabled: bool = True) -> None:
        self.device = device
        self.enabled = enabled
        self._model = None
        self._preprocess = None

    @property
    def loaded(self):
        if self._model is None:
            import torch

            # DINOv2's backbone weights come from torch.hub's own cache layout, so the hub dir
            # is the cache root -- not <cache>/checkpoints, which is where hub then puts them.
            torch.hub.set_dir(IDSIM_CACHE + "/hub")
            if IDSIM_REPO not in sys.path:
                sys.path.insert(0, IDSIM_REPO)
            from id_sim import id_sim

            self._model, self._preprocess = id_sim(
                pretrained=True, device=self.device, cache_dir=IDSIM_CACHE,
                id_sim_type=IDSIM_TYPE)
        return self._model, self._preprocess

    def similarity(self, left_crop: np.ndarray | None,
                   right_crop: np.ndarray | None) -> float | None:
        if not self.enabled or left_crop is None or right_crop is None:
            return None
        if min(left_crop.shape[:2]) < 2 or min(right_crop.shape[:2]) < 2:
            return None
        import torch
        from PIL import Image

        model, preprocess = self.loaded
        left = preprocess(Image.fromarray(left_crop)).to(self.device)
        right = preprocess(Image.fromarray(right_crop)).to(self.device)
        with torch.inference_mode():
            distance = float(model(left, right, mode="cls"))
        return round(1.0 - distance, 6)

    # ----- candidate ranking -------------------------------------------------------------
    # Split into embed + compare because that is where the ranking design's economics live:
    # one embedding costs 33.6 ms and one comparison of cached embeddings costs 0.202 ms, so an
    # exhaustive N x M pairing is essentially free once each candidate is embedded once.

    def embed(self, crop: np.ndarray | None):
        """Cached ID-Sim embedding of one crop, or None when there is nothing to embed."""
        if not self.enabled or crop is None or min(crop.shape[:2]) < 2:
            return None
        import torch
        from PIL import Image

        model, preprocess = self.loaded
        with torch.inference_mode():
            return model.embed(preprocess(Image.fromarray(crop)).to(self.device), mode="cls")

    def compare(self, left, right) -> float | None:
        """Similarity (1 - distance) between two cached embeddings.

        Reaches through ``model.model``: the public ``IDSimModel`` wrapper exposes only
        ``forward``/``embed``, and the distance-from-embeddings entry point lives on the inner
        ``PerceptualModel``. Without it every pair would re-run the backbone on the reference.
        """
        if left is None or right is None:
            return None
        import torch

        model, _preprocess = self.loaded
        with torch.inference_mode():
            # Returns a *dict* of per-feature-type distances ({"cls": tensor}, plus "patch" when
            # the policy asks for it), not a scalar. Indexing "cls" rather than float()-ing the
            # dict, and asserting the key rather than falling back, because a silent switch to a
            # different feature type would change every score in the report without a trace.
            distances = model.model.compute_distance_from_embeddings(
                {"cls_embed": left["cls"]}, {"cls_embed": right["cls"]})
            distance = float(distances["cls"].reshape(-1)[0])
        return round(1.0 - distance, 6)


def matted_crop(frame: np.ndarray, mask: np.ndarray | None,
                box: list[int] | None) -> np.ndarray | None:
    """The tight-box crop with the background replaced by white.

    This is the crop the identity judge should see, and having the mask is the reason the
    text-free chain can produce it. The incumbent judge scored a plain rectangle, so two frames
    a median 83 seconds apart contributed their *backgrounds* to the similarity -- measured
    consequence: changing only the crop policy flipped the keep/drop verdict on 26-30 of 140
    subjects, with per-subject cosine swings up to 0.35. Matting removes that channel entirely.
    """
    if mask is None or box is None:
        return None
    x1, y1, x2, y2 = box
    rgb = frame[y1:y2, x1:x2]
    if rgb.size == 0:
        return None
    local = mask[y1:y2, x1:x2]
    return np.where(local[..., None], rgb, 255).astype(np.uint8)


def process_subject_candidates(dataset: Path, sample_id: str, subject: dict[str, Any],
                               seed_frame: np.ndarray, clip_size: tuple[int, int],
                               models: segment.Models, idsim: IdSim, detector: Any,
                               out_root: Path, top_k: int) -> dict[str, Any]:
    """Propose several boxes per side and let ID-Sim pick the best cross-side pair.

    The report row keeps the same viewer field names as the single-box path, so ``gate_viewer``
    and ``gate_apply`` need no changes: ``box_*_dis`` / ``chosen_box_*`` hold the *winning*
    candidate. What is added is the evidence for that choice -- every candidate, every pair score,
    the margin over the runner-up, and which source won -- because "ID-Sim picked box 2 of 4" is
    only trustworthy if the losers are on the record too.
    """
    from PIL import Image

    subject_id = int(subject["subject_id"])
    ref_relative = (subject.get("ref") or {})["frame"]
    ref_frame = np.asarray(Image.open(dataset / ref_relative).convert("RGB"))
    ref_height, ref_width = ref_frame.shape[:2]
    clip_width, clip_height = clip_size

    ref_box = segment.scale_bbox_to_frame(
        (subject.get("ref") or {})["bbox_768"], ref_width, ref_height)
    seed_box = segment.scale_bbox_to_frame(
        subject["seed_bbox_768"], clip_width, clip_height)

    # The head noun only. Passing the full phrase let part words win their own boxes -- the
    # detector scores a query by max over its tokens, so "glasses" can carry a box on its own.
    query = candidates.subject_noun(subject.get("phrase"))

    ref_pool = candidates.side_candidates(models, detector, ref_frame, ref_box, query,
                                          device=models.device, top_k=top_k)
    seed_pool = candidates.side_candidates(models, detector, seed_frame, seed_box, query,
                                           device=models.device, top_k=top_k)
    ranked = candidates.best_pair(idsim.embed, idsim.compare, ref_pool, seed_pool,
                                  ref_frame, seed_frame)

    chosen = ranked.get("chosen")
    ref_pick = ref_pool[chosen["ref_index"]] if chosen else None
    seed_pick = seed_pool[chosen["seed_index"]] if chosen else None

    # The winner already carries the mask and tight box that `side_candidates` derived for it, so
    # there is nothing further to compute: the box that ships is the one that was ranked.
    #
    # A mask-feedback growth step used to sit here, re-segmenting the winner with the box prompt
    # removed so the instance could complete past the box that seeded it. It was measured and
    # dropped: 44% of sides did escape their seed box, but the final-to-seed box area ratio came
    # out at a median of 0.999 and a p90 of 1.020 -- adjustment, not completion -- for ~0.3 s per
    # subject. The candidate pool is what actually addresses a clipped box, because detector
    # proposals are not bounded by Phantom's rectangle to begin with.
    ref_mask = ref_pick.get("mask") if ref_pick else None
    seed_mask = seed_pick.get("mask") if seed_pick else None

    ref_rel = f"frames/{sample_id}_subj{subject_id:02d}_ref.jpg"
    seed_rel = f"frames/{sample_id}_subj{subject_id:02d}_seed.jpg"
    write_frame(out_root / ref_rel, ref_frame)
    write_frame(out_root / seed_rel, seed_frame)

    def strip(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Candidates without their masks -- arrays do not belong in a json report."""
        return [{k: v for k, v in c.items() if k != "mask"} for c in pool]

    identity = ranked.get("identity")
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "phrase": subject.get("phrase"),
        "dis": subject.get("phrase"),
        "seed_frame_index": int(subject["seed_frame_index"]),
        "ref_frame": ref_rel,
        "seed_frame": seed_rel,
        "box_ref_phantom": list(ref_box) if ref_box else None,
        "box_seed_phantom": list(seed_box) if seed_box else None,
        "box_ref_dis": ref_pick["box"] if ref_pick else None,
        "box_seed_dis": seed_pick["box"] if seed_pick else None,
        "chosen_box_ref": ref_pick["box"] if ref_pick else None,
        "chosen_box_seed": seed_pick["box"] if seed_pick else None,
        "pick_ref": "dis" if ref_pick else "no_box",
        "pick_seed": "dis" if seed_pick else "no_box",
        "pick_ref_reason": (f"{ref_pick['source']}, ranked 1 of {ranked['ref_candidates']} "
                            f"by id-sim" if ref_pick else "no candidate produced a mask"),
        "pick_seed_reason": (f"{seed_pick['source']}, ranked 1 of {ranked['seed_candidates']} "
                             f"by id-sim" if seed_pick else "no candidate produced a mask"),
        "both_tightened": bool(ref_pick and seed_pick),
        "dino_cos_chosen": identity,
        "rule_identity": identity,
        "identity_metric": (f"id_sim:{IDSIM_TYPE}:matted_tight_crop:best_of_pairs"
                            if identity is not None else None),
        # The evidence for the choice.
        "ranking": {k: v for k, v in ranked.items() if k != "all_pairs"},
        "pairs": ranked.get("all_pairs"),
        "ref_pool": strip(ref_pool),
        "seed_pool": strip(seed_pool),
        "tighten_ref": tighten.tighten_diagnostics(
            ref_mask, ref_pick.get("box") if ref_pick else None, ref_box,
            ref_frame.shape[:2]),
        "tighten_seed": tighten.tighten_diagnostics(
            seed_mask, seed_pick.get("box") if seed_pick else None, seed_box,
            seed_frame.shape[:2]),
    }
    for side in ("ref", "seed"):
        row[f"tighten_{side}"].pop("dilation", None)
    return row


def process_subject(dataset: Path, sample_id: str, subject: dict[str, Any],
                    seed_frame: np.ndarray, clip_size: tuple[int, int],
                    models: segment.Models, idsim: IdSim, out_root: Path) -> dict[str, Any]:
    """Tighten both of one subject's boxes and build its report row."""
    from PIL import Image

    subject_id = int(subject["subject_id"])
    ref_relative = (subject.get("ref") or {})["frame"]
    ref_frame = np.asarray(Image.open(dataset / ref_relative).convert("RGB"))
    ref_height, ref_width = ref_frame.shape[:2]
    clip_width, clip_height = clip_size

    # The coordinate trap, same as redetect_run's: stage B stores *annotation canvas* boxes, and
    # the reference jpg is a frame of a different video at a different resolution from the clip.
    # Mapping the reference box against the clip's dimensions shifts it by the ratio of the two,
    # so the two sides are mapped separately against their own frames.
    ref_box = segment.scale_bbox_to_frame(
        (subject.get("ref") or {})["bbox_768"], ref_width, ref_height)
    seed_box = segment.scale_bbox_to_frame(
        subject["seed_bbox_768"], clip_width, clip_height)

    result = tighten.tighten_subject(models, ref_frame, ref_box, seed_frame, seed_box,
                                     device=models.device)
    ref_diag, seed_diag = result["ref"], result["seed"]

    identity = idsim.similarity(
        matted_crop(ref_frame, result["_ref_mask"], ref_diag.get("tight_box")),
        matted_crop(seed_frame, result["_seed_mask"], seed_diag.get("tight_box")))

    ref_rel = f"frames/{sample_id}_subj{subject_id:02d}_ref.jpg"
    seed_rel = f"frames/{sample_id}_subj{subject_id:02d}_seed.jpg"
    write_frame(out_root / ref_rel, ref_frame)
    write_frame(out_root / seed_rel, seed_frame)

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "phrase": subject.get("phrase"),
        # ``dis`` is what the viewer prints as the phrase. Phantom's own phrase, unenriched:
        # no LLM text is consulted anywhere in this chain, and leaving the field absent would
        # make the viewer show "None" where the object's name belongs.
        "dis": subject.get("phrase"),
        "seed_frame_index": int(subject["seed_frame_index"]),
        "ref_frame": ref_rel,
        "seed_frame": seed_rel,
        # Viewer field names: red box = phantom, blue box = the refined one.
        "box_ref_phantom": list(ref_diag["prompt_box"] or []) or None,
        "box_seed_phantom": list(seed_diag["prompt_box"] or []) or None,
        "box_ref_dis": ref_diag["tight_box"],
        "box_seed_dis": seed_diag["tight_box"],
        "pick_ref": "dis" if ref_diag["tightened"] else "no_box",
        "pick_seed": "dis" if seed_diag["tightened"] else "no_box",
        "pick_ref_reason": "sam2 tight box from phantom prompt",
        "pick_seed_reason": "sam2 tight box from phantom prompt",
        "chosen_box_ref": ref_diag["tight_box"],
        "chosen_box_seed": seed_diag["tight_box"],
        "both_tightened": result["both_tightened"],
        # ``dino_cos_chosen`` is the name ``redetect.decide`` reads the identity score from -- a
        # contract shared with the four detector-era rules, so the value goes under that key even
        # though it is an ID-Sim similarity rather than a DINOv2 cosine. ``identity_metric``
        # records which metric actually produced it; renaming the key instead would make every
        # rule silently see "no identity score" and drop all 140 subjects.
        "dino_cos_chosen": identity,
        "rule_identity": identity,
        "identity_metric": (f"id_sim:{IDSIM_TYPE}:matted_tight_crop" if identity is not None
                            else None),
        "tighten_ref": {k: v for k, v in ref_diag.items() if k != "prompt_box"},
        "tighten_seed": {k: v for k, v in seed_diag.items() if k != "prompt_box"},
    }
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N subjects (smoke runs); 0 = all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-idsim", action="store_true",
                        help="skip the identity score; boxes only, for a fast geometry check")
    parser.add_argument("--candidates", action="store_true",
                        help="propose detector boxes alongside phantom's and let ID-Sim rank "
                             "every cross-side pair (see phantom_data.candidates)")
    parser.add_argument("--detector-top-k", type=int, default=candidates.DETECTOR_TOP_K)
    parser.add_argument("--sam2-config", default=os.environ.get(
        "SAM2_CONFIG", segment.DEFAULT_SAM2_CONFIG))
    parser.add_argument("--sam2-checkpoint", default=os.environ.get(
        "SAM2_CHECKPOINT", segment.DEFAULT_SAM2_CHECKPOINT))
    args = parser.parse_args(argv)

    root = args.dataset / args.out_root
    rows = read_jsonl(args.dataset / args.input)
    print(f"{len(rows)} samples in {args.input}", flush=True)

    models = segment.Models(args.sam2_config, args.sam2_checkpoint,
                            segment.DEFAULT_CLIP_MODEL, device=args.device)
    idsim = IdSim(device=args.device, enabled=not args.no_idsim)
    detector = None
    if args.candidates:
        if args.no_idsim:
            parser.error("--candidates needs ID-Sim to rank the candidates")
        from phantom_data import redetect
        detector = redetect.Models(device=args.device)

    subjects_out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    for sample in rows:
        sample_id = str(sample["sample_id"])
        try:
            frames = decode_frames(args.dataset / sample["video"])
            if not frames:
                raise ValueError("decoded 0 frames")
            height, width = frames[0].shape[:2]
            for subject in sample.get("subjects") or []:
                index = min(int(subject["seed_frame_index"]), len(frames) - 1)
                if args.candidates:
                    subjects_out.append(process_subject_candidates(
                        args.dataset, sample_id, subject, frames[index], (width, height),
                        models, idsim, detector, root, args.detector_top_k))
                    last = subjects_out[-1]
                    rank = last["ranking"]
                    print(f"[{len(subjects_out)}] {sample_id} subj{last['subject_id']:02d}  "
                          f"pool={rank['ref_candidates']}x{rank['seed_candidates']}  "
                          f"identity={last['rule_identity']}  "
                          f"margin={rank.get('margin')}  "
                          f"detector_won={rank.get('used_detector')}", flush=True)
                else:
                    subjects_out.append(process_subject(
                        args.dataset, sample_id, subject, frames[index], (width, height),
                        models, idsim, root))
                    last = subjects_out[-1]
                    print(f"[{len(subjects_out)}] {sample_id} subj{last['subject_id']:02d}  "
                          f"tightened={last['both_tightened']}  "
                          f"area_ratio ref={last['tighten_ref'].get('area_ratio')} "
                          f"seed={last['tighten_seed'].get('area_ratio')}  "
                          f"identity={last['rule_identity']}", flush=True)
                if args.limit and len(subjects_out) >= args.limit:
                    break
            if args.limit and len(subjects_out) >= args.limit:
                break
        except Exception as error:  # noqa: BLE001 -- one bad clip must not end the run
            failures.append({"sample_id": sample_id, "error": repr(error)})
            print(f"!! {sample_id}: {error!r}", flush=True)
            # Stop after a run of failures with nothing produced: that is an environment or
            # coordinate bug, not a bad clip, and grinding through 138 samples to report the
            # same repr 138 times wastes the smoke run's whole purpose.
            if len(failures) >= FAILURE_STREAK_ABORT and not subjects_out:
                print(f"!! aborting: {len(failures)} consecutive failures, 0 subjects produced",
                      flush=True)
                break

    elapsed = time.time() - started
    tightened = sum(1 for s in subjects_out if s["both_tightened"])
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chain": ("candidate_pool_idsim_ranked" if args.candidates
                  else "text_free_sam2_tighten"),
        "source": {"dataset": str(args.dataset), "input": args.input},
        "models": {"sam2_config": args.sam2_config,
                   "sam2_checkpoint": args.sam2_checkpoint,
                   "identity": IDSIM_TYPE if idsim.enabled else None},
        # No thresholds are asserted: the gate for this chain is undecided pending labels, and
        # writing a placeholder rule would let the viewer imply one had been chosen.
        "rule": {"identity_min": None, "note": "undecided; awaiting human labels"},
        "counts": {"subjects": len(subjects_out), "both_tightened": tightened,
                   "failures": len(failures),
                   "detector_won": sum(1 for s in subjects_out
                                       if (s.get("ranking") or {}).get("used_detector"))},
        "timing": {"wall_sec": round(elapsed, 2),
                   "sec_per_subject": round(elapsed / max(1, len(subjects_out)), 3)},
        "subjects": subjects_out,
        "failures": failures,
    }
    atomic_write_bytes(root / "gate_report.json",
                       (json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                       .encode("utf-8"))
    print(json.dumps({k: report[k] for k in ("counts", "timing")}, indent=2))
    print(f"wrote {root / 'gate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
