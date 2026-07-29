"""Does Grounding DINO's confidence score respect every word of a phrase, or just one?

The suspicion this probe answers: on a phrase like ``dog doll``, the detector appears to score
any old doll highly, ignoring ``dog``. If true, ``best_by_detector``
(:func:`phantom_data.redetect.best_by_detector`) is picking boxes on a number that cannot
express "all the concepts matter", and no amount of reordering candidates would fix it.

The mechanism is in ``GroundingDinoProcessor.post_process_grounded_object_detection``::

    batch_probs  = torch.sigmoid(batch_logits)        # (queries, 256) -- one slot per text token
    batch_scores = torch.max(batch_probs, dim=-1)[0]  # (queries,)     -- MAX over tokens

So the reported score is *the single best token match*. A box that nails ``doll`` and misses
``dog`` scores identically to one that nails both: the max hides the miss. That is a property of
the aggregation, not of the model -- the per-token probabilities are right there in
``outputs.logits``, and this probe reads them instead of the max.

For each query it reports, per candidate box: the max score (what the pipeline uses today), and
then the per-content-word score, where a word's score is the max over the *sub-tokens it
tokenizes into* (``bulldog`` becomes ``bull``/``##dog``, and a word-level number has to
re-aggregate those or it under-reports every long word). From those it computes two alternative
aggregations:

* ``min`` over content words -- the strict reading of "every concept must be present". This is
  the candidate for replacing the max.
* ``mean`` over content words -- softer, and the useful diagnostic when ``min`` is dragged down
  by one word the detector cannot ground at all (colours and materials are often like this).

The deciding output is the **ranking**: how often ``min`` and ``max`` disagree about which box
is best. If they always agree, the aggregation is a non-issue on this data and the effort belongs
elsewhere. If they disagree, the boxes each picks are printed side by side so the disagreement
can be judged by eye rather than by the metric that is under suspicion.

Runs on CPU (slowly) so it needs no GPU pod. Usage::

    python tools/probe_dino_tokens.py --dataset <root> --report-root _redetect_trust --limit 12
    python tools/probe_dino_tokens.py --queries "dog doll" "doll" --image <path.jpg>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from phantom_data import redetect
from phantom_data.boxes import iou
from phantom_data.inspect import atomic_write_bytes

#: Words that carry no grounding content. A phrase's ``min`` aggregation must skip these or it
#: measures how well the detector grounds the word "with", which is not a question anyone asked.
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "with", "in", "on", "at", "to", "for", "from",
    "is", "are", "was", "were", "be", "being", "been", "very", "its", "it", "his", "her",
    "their", "that", "this", "these", "those", "wearing", "has", "have",
})


def content_words(phrase: str) -> list[str]:
    """The words a phrase is actually asking to be grounded, in order, deduplicated.

    ``wearing`` and ``has`` are treated as stop words even though they are verbs with meaning:
    Grounding DINO grounds them to the garment, not to the act, so scoring them separately
    double-counts the noun that follows.
    """
    seen: list[str] = []
    for raw in str(phrase or "").lower().replace(",", " ").replace(".", " ").split():
        word = raw.strip("'\"()-")
        if not word or word in STOP_WORDS or word in seen:
            continue
        seen.append(word)
    return seen


def word_token_spans(tokenizer, prompt: str, words: list[str]) -> dict[str, list[int]]:
    """Map each content word to the token positions it occupies in the encoded prompt.

    Needed because the score tensor is indexed by *token*, not by word: BERT's wordpiece
    tokenizer splits ``bulldog`` into ``bull`` + ``##dog``, and taking a single token's
    probability as the word's score would systematically under-report multi-piece words --
    exactly the long descriptive words this pipeline generates.
    """
    encoded = tokenizer(prompt, return_tensors="pt")
    token_ids = encoded["input_ids"][0].tolist()
    pieces = tokenizer.convert_ids_to_tokens(token_ids)

    spans: dict[str, list[int]] = {}
    for word in words:
        wanted = tokenizer.tokenize(word)
        if not wanted:
            continue
        positions: list[int] = []
        # Scan for the contiguous run of pieces matching this word's tokenization. First match
        # wins; a word repeated in the prompt was deduplicated by content_words already.
        for start in range(len(pieces) - len(wanted) + 1):
            if pieces[start:start + len(wanted)] == wanted:
                positions = list(range(start, start + len(wanted)))
                break
        if positions:
            spans[word] = positions
    return spans


def probe_frame(models: redetect.Models, frame: np.ndarray, query: str,
                threshold: float = redetect.BOX_THRESHOLD,
                text_threshold: float = redetect.TEXT_THRESHOLD,
                top_k: int = redetect.TOP_K) -> dict[str, Any]:
    """Per-token scores for every candidate box on one frame. No aggregation is privileged."""
    from PIL import Image

    processor, model, torch = models.dino
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    # Same prompt normalisation as redetect.Models.detect, so the numbers are comparable to
    # what the pipeline stored rather than to a differently-punctuated query.
    prompt = str(query).strip().lower().rstrip(".") + "."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(models.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0]          # (queries, 256)
    scores = probs.max(dim=-1)[0]                     # what the pipeline uses
    boxes = outputs.pred_boxes[0]

    height, width = frame.shape[:2]
    words = content_words(query)
    spans = word_token_spans(processor.tokenizer, prompt, words)

    keep = (scores > threshold).nonzero().flatten().tolist()
    keep.sort(key=lambda i: -float(scores[i]))
    candidates: list[dict[str, Any]] = []
    for index in keep[:top_k]:
        cx, cy, bw, bh = [float(v) for v in boxes[index].tolist()]
        box = [round((cx - bw / 2) * width, 1), round((cy - bh / 2) * height, 1),
               round((cx + bw / 2) * width, 1), round((cy + bh / 2) * height, 1)]
        per_word = {word: round(float(probs[index, positions].max()), 4)
                    for word, positions in spans.items()}
        values = list(per_word.values())
        candidates.append({
            "box": box,
            "max": round(float(scores[index]), 4),
            "per_word": per_word,
            "min": round(min(values), 4) if values else None,
            "mean": round(sum(values) / len(values), 4) if values else None,
        })
    return {"query": query, "prompt": prompt, "content_words": words,
            "candidates": candidates}


def rank_disagreement(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Do ``max`` and ``min`` pick the same box? Returns both picks and their IoU.

    IoU between the two picks is the number that matters: two rankings that disagree on the
    ordering but pick overlapping boxes are a curiosity, whereas picks that do not overlap mean
    the aggregation decides *which object* gets segmented.
    """
    if not candidates:
        return {"agree": None}
    by_max = max(candidates, key=lambda c: c["max"])
    scored = [c for c in candidates if c.get("min") is not None]
    if not scored:
        return {"agree": None}
    by_min = max(scored, key=lambda c: c["min"])
    overlap = iou(by_max["box"], by_min["box"])
    return {
        "agree": by_max["box"] == by_min["box"],
        "box_by_max": by_max["box"], "box_by_min": by_min["box"],
        "iou_between_picks": None if overlap is None else round(overlap, 4),
        "max_pick_min_score": by_max.get("min"),
        "min_pick_max_score": by_min.get("max"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--report-root", default="_redetect_trust")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--only-multiword", action="store_true",
                        help="skip subjects whose phrase has one content word: the aggregation "
                             "question is meaningless there (min == max by definition)")
    parser.add_argument("--queries", nargs="*", default=None,
                        help="ad-hoc phrases to probe against --image instead of a dataset")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/dino_token_probe"))
    args = parser.parse_args(argv)

    models = redetect.Models(device=args.device)
    results: list[dict[str, Any]] = []

    if args.queries:
        if not args.image:
            parser.error("--queries needs --image")
        from PIL import Image
        frame = np.asarray(Image.open(args.image).convert("RGB"))
        for query in args.queries:
            probe = probe_frame(models, frame, query)
            probe["ranking"] = rank_disagreement(probe["candidates"])
            probe["image"] = str(args.image)
            results.append(probe)
    else:
        if not args.dataset:
            parser.error("need --dataset or (--queries and --image)")
        report = json.loads((args.dataset / args.report_root / "gate_report.json")
                            .read_text(encoding="utf-8"))
        from PIL import Image
        for subject in report.get("subjects") or []:
            phrase = subject.get("dis") or subject.get("phrase") or ""
            if args.only_multiword and len(content_words(phrase)) < 2:
                continue
            frame_path = args.dataset / args.report_root / str(subject.get("ref_frame") or "")
            if not frame_path.is_file():
                continue
            frame = np.asarray(Image.open(frame_path).convert("RGB"))
            probe = probe_frame(models, frame, phrase)
            probe["ranking"] = rank_disagreement(probe["candidates"])
            probe.update({"sample_id": subject["sample_id"],
                          "subject_id": subject["subject_id"],
                          "stored_detector_score": subject.get("detector_score_ref_dis"),
                          "stored_box": subject.get("box_ref_dis"),
                          "phantom_box": subject.get("box_ref_phantom")})
            results.append(probe)
            print(f"[{len(results)}/{args.limit}] {phrase[:52]!r} "
                  f"cand={len(probe['candidates'])} "
                  f"agree={probe['ranking'].get('agree')}", flush=True)
            if len(results) >= args.limit:
                break

    decided = [r for r in results if r["ranking"].get("agree") is not None]
    disagreed = [r for r in decided if not r["ranking"]["agree"]]
    summary = {
        "probed": len(results),
        "multiword_decided": len(decided),
        "max_and_min_disagree": len(disagreed),
        "disagreement_rate": (round(len(disagreed) / len(decided), 4) if decided else None),
        # A disagreement where the two picks barely overlap means the aggregation chooses the
        # object, not just the box. That is the severe case and is worth counting separately.
        "disagree_with_iou_below_0.5": sum(
            1 for r in disagreed
            if (r["ranking"].get("iou_between_picks") or 0) < 0.5),
    }
    payload = {"summary": summary, "results": results}
    atomic_write_bytes(args.out / "dino_token_probe.json",
                       (json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                       .encode("utf-8"))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
