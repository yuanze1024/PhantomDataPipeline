"""Propose several boxes per side, then let ID-Sim pick the pair that is the same instance.

The design each earlier attempt argued its way into. Two box sources with **non-overlapping
failure modes**, and a judge that is good at exactly the thing neither source can do:

* **Grounding DINO** boxes are tight -- box regression is its training objective -- but it picks
  the wrong *instance*. Measured: on ``dog doll`` the frame held two dolls, the detector scored
  the left one 0.67 and the right one 0.29, and Phantom had annotated the right one. Its boxes
  are also free of Phantom's clipping, so a limb Phantom cut off is often inside a detector box.
* **SAM2 from Phantom's box** always segments the instance Phantom meant -- that is what the box
  prompt guarantees -- but inherits Phantom's geometry, so a limb outside Phantom's box is learned
  as background. Growing the prompt to fix that made things worse, not better: prompt dilation
  enlarged 81% of boxes, left the reviewer's four failures at a median 1.000x, and dropped median
  ID-Sim identity from 0.659 to 0.408 by swallowing background.
* **ID-Sim** cannot produce a box at all, but separates same-instance from different-instance on
  this data with AUROC 0.998, and comparing two cached embeddings costs 0.202 ms against 33.6 ms
  to compute one. So ranking many candidates is nearly free once they are embedded.

Put together: each source proposes, ID-Sim decides. A wrong-instance detector box loses to the
right-instance SAM2 box; a clipped SAM2 box loses to a complete detector box.

**Both sides are ranked against each other**, not one against a trusted reference. Every
candidate on the reference side is scored against every candidate on the target side and the best
pair wins. The reviewer's argument for this: two independent sides both going wrong *in the same
way* is unlikely, so requiring agreement filters what a single trusted side cannot. The residual
risk is real and named in :func:`best_pair`'s docstring -- a distractor identity present in both
clips can be picked twice -- which is why the pair margin is recorded rather than discarded.

Text enters only as a **nomination** mechanism, never as a verdict. Grounding DINO needs a query,
so it gets Phantom's bare phrase; the probe in ``tools/probe_dino_tokens.py`` showed short queries
score cleanly (``French Bulldog``: 0.734 / 0.746 per word, ``min`` and ``max`` agreeing on the
same box) while long LLM phrases let part words hijack the ranking (a face box and an eye box
scoring 0.31 each on their own tokens). Phantom's phrases run a median of 2 words, which is the
regime where the detector behaves. Nothing downstream reads a text score.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from phantom_data.boxes import clamp_box, iou

#: Candidate boxes whose IoU exceeds this are the same proposal from two sources; keep one.
#: Deduplication matters for the margin: two near-identical candidates would otherwise split the
#: ranking and make an unambiguous choice look like a close call.
DEDUPE_IOU = 0.9

#: Detector proposals to request per side. Beyond this the pool is mostly background boxes, and
#: the pilot's ambiguity is concentrated at 2-3 candidates (39% of subjects had 2+).
DETECTOR_TOP_K = 3

#: Sources, in the order they are proposed. Recorded per candidate so a shipped box can always be
#: traced to what produced it -- the question "why is this box bigger than Phantom's?" must be
#: answerable from the report alone.
SOURCE_PHANTOM_BOX = "phantom_box"
SOURCE_DETECTOR_BOX = "detector_box"

# Retained for reading reports written by the pre-rank-first runs (_cand_v1).
SOURCE_PHANTOM_SAM2 = "phantom_sam2"
SOURCE_DETECTOR_SAM2 = "detector_sam2"

#: Detector boxes below this share of Phantom's box area are parts, not instances.
#: See :func:`plausible_instance`; measured part boxes ran 0.02-0.23.
MIN_CANDIDATE_AREA_SHARE = 0.30


def dedupe_candidates(candidates: list[dict[str, Any]],
                      threshold: float = DEDUPE_IOU) -> list[dict[str, Any]]:
    """Drop candidates that duplicate an earlier one geometrically.

    Order matters and is deliberate: :data:`SOURCE_PHANTOM_SAM2` is proposed first, so when the
    detector independently finds the same box the surviving record credits the source whose
    semantics are guaranteed rather than the one that merely agreed.
    """
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        box = candidate.get("box")
        if not box:
            continue
        duplicate = False
        for existing in kept:
            overlap = iou(box, existing["box"])
            if overlap is not None and overlap >= threshold:
                existing.setdefault("also_proposed_by", []).append(candidate["source"])
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def subject_noun(phrase: str | None) -> str:
    """The head noun phrase of a Phantom phrase, for use as a detector query.

    Grounding DINO reports confidence as a **max over text tokens**, so every word in the query
    can win a box on its own. Measured: ``man with short dark hair and glasses`` yields boxes for
    the hair and the glasses scoring ~0.31 on their own tokens, and after ID-Sim ranking those
    part boxes shipped -- phrases containing a part word were 5x more likely to ship a box under
    half Phantom's area (20% of sides vs 4%).

    Truncating at the first prepositional or participial joiner keeps the subject and discards the
    attributes. No LLM: the attributes are exactly what must be removed, so there is nothing for a
    model to add, and a rule is deterministic and free where an ``_enrich`` pass was neither.
    """
    words = str(phrase or "").replace(",", " ").split()
    cut = ("with", "in", "wearing", "on", "at", "holding", "and", "that", "which",
           "having", "carrying", "next", "near", "behind", "beside")
    kept: list[str] = []
    for word in words:
        if word.lower().strip(".") in cut:
            break
        kept.append(word)
    # Never return empty: a phrase that is nothing but joiners is degenerate, and an empty query
    # would silently disable the detector for that subject rather than failing visibly.
    return " ".join(kept) if kept else str(phrase or "").strip()


def plausible_instance(box: list[int], reference_box: Any, min_area_share: float) -> bool:
    """Is ``box`` big enough, relative to Phantom's, to be the instance rather than a part of it?

    Phantom's box is drawn around the subject, so the subject cannot occupy a small fraction of
    it. The part boxes measured on the pilot ran 0.02-0.23 of Phantom's area, well clear of any
    genuine instance. A second guard alongside :func:`subject_noun` because part boxes are not
    purely a text artefact -- ``cartoon character`` is two words and still produced a 0.12 box.
    """
    if reference_box is None:
        return True
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    ref_area = max(1, (reference_box[2] - reference_box[0])
                   * (reference_box[3] - reference_box[1]))
    return area >= ref_area * min_area_share


def side_candidates(sam_models, detector, frame_rgb: np.ndarray, phantom_box: Any,
                    query: str | None, device: str = "cuda",
                    top_k: int = DETECTOR_TOP_K,
                    min_area_share: float = MIN_CANDIDATE_AREA_SHARE
                    ) -> list[dict[str, Any]]:
    """Every candidate tight box for one frame, each with the mask that produced it.

    **Every candidate is segmented and matted before ranking**, and the measurement says this is
    worth its 0.15 s. A version that ranked plain rectangle crops instead -- on the reasonable
    theory that ID-Sim is trained for exactly this invariance -- lost identity on 43 of 140
    subjects against 16 gained, dropping p10 from 0.524 to 0.208 and collapsing the median
    decision margin from 0.090 to 0.010. On the four subjects the reviewer had failed it fell from
    0.804 back to 0.469, undoing the fix. The reason the invariance is not enough here: the two
    frames come from *different clips* a median 83 seconds apart, so their backgrounds are
    unrelated rather than merely shifted, and a rectangle crop asks the judge to compare two
    unrelated scenes that happen to contain the subject.

    Candidates are filtered by :func:`plausible_instance` before segmentation, so a part box costs
    nothing and cannot win on a technicality.
    """
    from phantom_data import tighten
    proposals: list[dict[str, Any]] = []
    height, width = frame_rgb.shape[:2]

    prompt = clamp_box(phantom_box, width, height)
    if prompt is not None:
        mask, tight, _trace = tighten.tighten_box(sam_models, frame_rgb, prompt,
                                                  device=device, max_rounds=0)
        if tight is not None:
            proposals.append({"source": SOURCE_PHANTOM_SAM2, "box": tight, "mask": mask,
                              "prompt_box": list(prompt), "detector_score": None})

    if detector is not None and str(query or "").strip():
        for rank, found in enumerate(detector.detect(frame_rgb, str(query), top_k=top_k)):
            box = clamp_box(found["box"], width, height)
            if box is None:
                continue
            if not plausible_instance(box, prompt, min_area_share):
                # A part of the subject, not the subject. Dropped before segmentation, because
                # ID-Sim compares a part against a part quite happily -- two crops of the same
                # pair of glasses really are the same glasses -- so the filter has to come first.
                continue
            mask, tight, _trace = tighten.tighten_box(sam_models, frame_rgb, box,
                                                      device=device, max_rounds=0)
            if tight is None:
                continue
            proposals.append({"source": SOURCE_DETECTOR_SAM2, "box": tight, "mask": mask,
                              "prompt_box": list(box), "detector_rank": rank,
                              "detector_score": found.get("detector_score")})

    return dedupe_candidates(proposals)


def matted_crop(frame_rgb: np.ndarray, mask: np.ndarray | None,
                box: list[int] | None) -> np.ndarray | None:
    """The tight-box crop with background replaced by white.

    Matting rather than a plain rectangle because the two frames are a median 83 seconds apart in
    *different clips*, so their backgrounds are unrelated and a plain crop makes the identity
    judge partly a background comparison. Measured on the pilot: changing only the crop policy
    flipped the keep/drop verdict on 26-30 of 140 subjects, with per-subject cosine swings up to
    0.35. Having the mask for free is the reason this chain can afford to do it.
    """
    if mask is None or box is None:
        return None
    x1, y1, x2, y2 = box
    rgb = frame_rgb[y1:y2, x1:x2]
    if rgb.size == 0:
        return None
    return np.where(mask[y1:y2, x1:x2][..., None], rgb, 255).astype(np.uint8)


def crop_for(frame_rgb: np.ndarray, candidate: dict[str, Any]) -> np.ndarray | None:
    """The crop ID-Sim scores for one candidate: matted when a mask exists, else the plain box.

    Rank-first proposals have no mask, and that is deliberate -- ID-Sim was trained to be
    invariant to context and lighting, which is the job matting would be doing. The matted branch
    remains for reading older reports and for any caller that already holds a mask.
    """
    box = candidate.get("box")
    if box is None:
        return None
    mask = candidate.get("mask")
    if mask is not None:
        return matted_crop(frame_rgb, mask, box)
    x1, y1, x2, y2 = box
    crop = frame_rgb[y1:y2, x1:x2]
    return None if crop.size == 0 else crop


def best_pair(embed_fn, compare_fn, ref_candidates: list[dict[str, Any]],
              seed_candidates: list[dict[str, Any]], ref_frame: np.ndarray,
              seed_frame: np.ndarray) -> dict[str, Any]:
    """Score every (reference, target) candidate pair and return the best, with its margin.

    Embeddings are computed once per candidate and reused across the pairing, which is what makes
    an exhaustive N x M comparison affordable: 3 candidates a side is 6 embeddings at 33.6 ms and
    9 comparisons at 0.202 ms, so the pairing itself is 0.3% of the cost.

    **The margin is the point of returning more than a winner.** Mutual ranking can fail in one
    specific way the reviewer's independence argument does not cover: a distractor identity that
    appears in *both* clips -- the second person in a two-hander, a matching pair of props -- can
    be selected on both sides, and the pair then scores high while describing the wrong subject.
    That failure is invisible in the winning score alone, but it tends to leave a *small margin*
    over the runner-up, because the true subject is also in both pools. So ``margin`` and
    ``runner_up`` are recorded for review, and no threshold is asserted on them here: which
    margin is too small is a question for the human labels, and guessing it now would repeat the
    mistake this whole chain was built to undo.
    """
    ref_embeds, seed_embeds = [], []
    for candidate in ref_candidates:
        ref_embeds.append(embed_fn(crop_for(ref_frame, candidate)))
    for candidate in seed_candidates:
        seed_embeds.append(embed_fn(crop_for(seed_frame, candidate)))

    scored: list[dict[str, Any]] = []
    for i, left in enumerate(ref_embeds):
        for j, right in enumerate(seed_embeds):
            if left is None or right is None:
                continue
            similarity = compare_fn(left, right)
            if similarity is None:
                continue
            scored.append({"ref_index": i, "seed_index": j,
                           "ref_source": ref_candidates[i]["source"],
                           "seed_source": seed_candidates[j]["source"],
                           "similarity": round(float(similarity), 6)})

    if not scored:
        return {"chosen": None, "reason": "no comparable candidate pair",
                "pairs_scored": 0,
                "ref_candidates": len(ref_candidates),
                "seed_candidates": len(seed_candidates)}

    scored.sort(key=lambda row: -row["similarity"])
    winner = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None
    return {
        "chosen": winner,
        "runner_up": runner_up,
        # None rather than 0.0 when there was no alternative: an unopposed winner has no margin,
        # which is a different statement from "the margin was zero".
        "margin": (round(winner["similarity"] - runner_up["similarity"], 6)
                   if runner_up else None),
        "identity": winner["similarity"],
        "pairs_scored": len(scored),
        "ref_candidates": len(ref_candidates),
        "seed_candidates": len(seed_candidates),
        # Whether the winning pair used a detector proposal on either side. The headline question
        # about this design -- does adding the detector actually change the answer? -- is settled
        # by counting these, not by inspecting boxes.
        "used_detector": bool(
            winner["ref_source"] in (SOURCE_DETECTOR_BOX, SOURCE_DETECTOR_SAM2)
            or winner["seed_source"] in (SOURCE_DETECTOR_BOX, SOURCE_DETECTOR_SAM2)),
        "all_pairs": scored,
    }
