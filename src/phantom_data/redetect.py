"""The object detector that proposes candidate boxes, and the keep/drop rule applied to pairs.

Two things live here, both consumed by the text-free box chain:

* :class:`Models` wraps Grounding DINO. :mod:`phantom_data.candidates` asks it for several boxes
  per frame, which -- alongside Phantom's own box -- form the candidate pool that ID-Sim then
  ranks. The detector *proposes*; nothing here decides.
* :func:`decide` applies the keep/drop rule to a subject's scores, and is called by both
  ``tools/gate_apply.py`` (which writes stage C's input) and ``tools/tighten_run.py``.

**The query is a bare noun phrase, never a long description.** Grounding DINO reports confidence
as a max over the prompt's text tokens, so every word can win a box on its own: measured on the
pilot, ``man with short dark hair and glasses`` produced separate boxes for the hair and the
glasses scoring ~0.31 each on their own tokens, and after ranking those part boxes shipped --
phrases containing a part word were 5x more likely to ship a box under half Phantom's area (20% of
sides against 4%). :func:`candidates.subject_noun` truncates the phrase before it gets here, and
:func:`candidates.plausible_instance` filters part-sized boxes as a second guard.

**One rule, because there is only one judge.** Four earlier rules ANDed a CLIP text score and an
IoU against Phantom's annotation with the identity check. Both were removed from the chain: the
CLIP score answered "is this crop the thing the phrase names", which a box offset by 40% still
passes, and the IoU compared two fallible annotations without being able to say which was wrong
(identity and IoU were statistically independent on the pilot, Spearman +0.008). What remains is
:data:`RULE_IDENTITY_ONLY` -- is the reference the same instance as the target -- because the box
now comes from SAM2 segmenting what a proposal pointed at, so "is the box on the intended object"
is answered by construction rather than by a confirming judge.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------

GROUNDING_DINO = ("/mnt/pfs/share/pretrained_model/.cache/huggingface/hub/"
                  "models--IDEA-Research--grounding-dino-base")

#: Detector confidence floor. Below this a proposal is noise; the pool is better served by
#: Phantom's own box than by a box the detector barely believes in.
BOX_THRESHOLD = 0.25

#: Token-level floor inside ``post_process_grounded_object_detection``.
TEXT_THRESHOLD = 0.20

#: Proposals returned per frame, highest confidence first. The caller asks for fewer (three) --
#: this is the ceiling on what the detector is allowed to volunteer.
TOP_K = 6

#: Which box a side ended up using, as recorded in the report and read by :func:`decide`.
#: ``FROM_DIS`` is a name inherited from the text-driven era ("dis" was the LLM phrase); it now
#: means "a box the chain derived", as opposed to Phantom's raw annotation.
PHANTOM = "phantom"
FROM_DIS = "dis"
NO_BOX = "no_box"

# --------------------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------------------

KEEP = "keep"
DROP = "drop"

#: Identity floor: ID-Sim similarity (1 - distance) between the two chosen crops, below which the
#: reference and the target are taken to be different instances.
#:
#: 0.6 was the DINOv2-cosine era's value and is kept as the module default only so callers that
#: never pass a threshold behave as before. The chain runs at **0.2**, chosen by reading the score
#: distribution against the imagery: it discards 5 of the 140 pilot subjects (4%). The scores are
#: not interchangeable between the two metrics -- ID-Sim's distribution is far better separated
#: (same-vs-different-instance AUROC 0.998 on this data) -- so a threshold from one does not
#: transfer to the other.
IDENTITY_MIN = 0.6

#: Identity alone. The only rule the text-free chain offers, and the reason it is a named constant
#: rather than an implicit default: a missing judge must never read as a passing one. On a
#: text-free report the four withdrawn rules each correctly returned keep=0/140, because each
#: ANDed a clip score that does not exist. Treating absent as satisfied is the failure this avoids.
RULE_IDENTITY_ONLY = "identity_only"

RULES = (RULE_IDENTITY_ONLY,)
DEFAULT_RULE = RULE_IDENTITY_ONLY


def resolve_snapshot(path: str) -> str:
    """The newest snapshot directory inside a HuggingFace hub cache entry, or ``path`` itself.

    Hub caches store weights under ``snapshots/<revision>/``; pointing ``from_pretrained`` at the
    cache root fails. Falls through unchanged for a plain directory so an explicit checkpoint path
    still works.
    """
    root = Path(path)
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return str(root)
    revisions = sorted((p for p in snapshots.iterdir() if p.is_dir()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    return str(revisions[0]) if revisions else str(root)


class Models:
    """Grounding DINO, loaded once and lazily.

    Lazy because the pipeline's stages are separate processes and only the box-proposal stage
    needs a detector; importing this module must not cost 700 MB of weights.
    """

    def __init__(self, dino_path: str = GROUNDING_DINO, device: str = "cuda") -> None:
        self.dino_path = dino_path
        self.device = device
        self._dino = None

    @property
    def dino(self):
        if self._dino is None:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            snapshot = resolve_snapshot(self.dino_path)
            processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                snapshot, local_files_only=True).to(self.device).eval()
            self._dino = (processor, model, torch)
        return self._dino

    def detect(self, frame: np.ndarray, query: str, threshold: float = BOX_THRESHOLD,
               text_threshold: float = TEXT_THRESHOLD,
               top_k: int = TOP_K) -> list[dict[str, Any]]:
        """Grounding DINO boxes for ``query``, highest confidence first.

        ``detector_score`` is what the processor reports, which is a **max over the prompt's text
        tokens** -- the single best token match, not agreement with the whole phrase. That is why
        the caller passes a bare noun: on a longer prompt this number cannot distinguish a box
        that matches every word from one that matches only the last.
        """
        from PIL import Image

        if not str(query or "").strip():
            return []
        processor, model, torch = self.dino
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        prompt = str(query).strip().lower().rstrip(".") + "."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]])[0]
        boxes = [[round(float(v), 1) for v in box] for box in result["boxes"].tolist()]
        scores = [round(float(s), 4) for s in result["scores"].tolist()]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [{"box": boxes[i], "detector_score": scores[i]} for i in order]


# --------------------------------------------------------------------------------------
# keep / drop
# --------------------------------------------------------------------------------------

def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def decide(scores: dict[str, Any], rule: str = DEFAULT_RULE,
           identity_min: float = IDENTITY_MIN, clip_min: float | None = None,
           iou_min: float | None = None,
           iou_floor: float | None = None) -> dict[str, Any]:
    """Keep or drop one subject, on identity alone.

    ``clip_min`` / ``iou_min`` / ``iou_floor`` are accepted and ignored. They are kept in the
    signature because ``gate_apply.py`` records every threshold it was invoked with into the
    manifest's provenance block, and dropping the parameters would change that block's shape for
    reports that are already on disk. The returned dict likewise keeps ``clip`` and ``iou`` keys
    holding ``None``: stage C carries this dict through as provenance without reading it, so the
    shape is part of the on-disk contract even though the values are gone.

    A subject with no identity score at all is dropped. An unmeasured pair is not an endorsed one.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")

    identity = scores.get("dino_cos_chosen")
    identity_ok = identity is not None and identity >= identity_min
    picks = {side: scores.get(f"pick_{side}") for side in ("ref", "seed")}
    box_set = ("phantom" if set(picks.values()) == {PHANTOM}
               else FROM_DIS if set(picks.values()) == {FROM_DIS}
               else f"ref={picks['ref']}, target={picks['seed']}")
    box_reason = "; ".join(f"{side}: {scores.get(f'pick_{side}_reason')}"
                           for side in ("ref", "seed"))

    def result(verdict: str, reason: str, **extra: Any) -> dict[str, Any]:
        return {
            "verdict": verdict,
            "rule": rule,
            "box_set": box_set,
            "box_reason": box_reason,
            "reason": reason,
            "identity": identity,
            "identity_ok": identity_ok,
            # Withdrawn judges, still present as nulls -- see the docstring.
            "clip": None,
            "clip_ok": None,
            "iou": None,
            "iou_ok": None,
            **extra,
        }

    # A side with no usable box is dropped before the identity check, and on the *pick* rather
    # than on the scores: a subject missing only its reference box would otherwise still carry a
    # perfectly good identity score and ship as a pair with one box missing.
    missing = [side for side in ("ref", "seed") if picks[side] == NO_BOX]
    if missing:
        return result(
            DROP,
            f"no usable box on the {' and '.join(missing)} side",
            no_box_sides=missing)

    if identity is None:
        return result(DROP, "no identity score — nothing judged this pair")
    if not identity_ok:
        return result(
            DROP,
            f"identity {fmt(identity)} < {identity_min} — the reference and the target are "
            f"probably not the same object")
    return result(
        KEEP,
        f"identity {fmt(identity)} >= {identity_min}; the box is SAM2's mask around what a "
        f"proposal pointed at, so no separate judge confirms it")
