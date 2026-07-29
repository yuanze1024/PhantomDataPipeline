"""Re-detect every subject's boxes, keep the better one per side, then filter the pairs.

Phantom's boxes are frequently skewed, and -- this is the part that shaped the design -- a
skewed box does not reliably score badly. An earlier version only re-detected subjects whose
CLIP score fell below a gate, which meant it skipped most of the boxes worth refining. So
every subject is re-detected now, and the CLIP score is used to decide *how much to trust the
annotation*, not whether to look at it.

Per subject, per side (reference and target):

1. Crop Phantom's box, score it against the enriched phrase (``dis``) with CLIP.
2. Run Grounding DINO with the same phrase; keep its highest-confidence box.
3. :func:`pick_side` chooses between them. Where Phantom's crop scored well the detector may
   only refine (``IoU >= IOU_MIN``); where it scored badly the detector's box is taken.
4. Score the chosen box, then take one DINOv2 cosine between the two chosen crops.

Then :func:`decide` filters whole pairs: identity is mandatory, and either semantics or
agreement with Phantom confirms the box.

Decisions here that came from measurement, and are worth not undoing:

**Boxes are chosen by detector confidence, never by CLIP.** Choosing among detector
candidates by CLIP was tried and is actively harmful: the CLIP-best candidate differed from
the detector-best in 9/12 cases, ``r(log box area, clip score) = +0.34``, and two cases were
confirmed where CLIP preferred a picture-in-picture face (3% of the correct area) and a patch
of fur on a dog. CLIP measures here, it does not select.

**One text everywhere.** Every CLIP score uses ``dis`` and the same plain-crop treatment,
which is what makes ``new - old`` a meaningful subtraction rather than a comparison of two
different questions.

**These CLIP scores are not comparable to the shipped ``ref_clip_score``.** That one sees
SAM2's white-matte cutout; these see the plain crop, background included, which depresses
them across the board -- hence ``CLIP_MIN = 0.21`` rather than the 0.23 calibrated on
cutouts. The ``crop_clip_`` prefix keeps the two apart.

**The renderer draws no boxes.** It writes the two frames unannotated and puts every
coordinate in the report, so the viewer overlays them itself. Colours, widths, which boxes to
show, and crop sizes are then page-level decisions rather than reasons to re-run the GPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .boxes import box_fraction, crop_box, iou

GROUNDING_DINO = ("/mnt/pfs/share/pretrained_model/.cache/huggingface/hub/"
                  "models--IDEA-Research--grounding-dino-base/snapshots")
CLIP_MODEL = ("/mnt/pfs/share/pretrained_model/.cache/huggingface/hub/"
              "models--openai--clip-vit-base-patch32/snapshots/"
              "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
DINOV2_MODEL = ("/mnt/pfs/share/pretrained_model/.cache/huggingface/hub/"
                "models--facebook--dinov2-base/snapshots")

#: Same thresholds UltraVid grounds with, kept identical so the two pipelines' detections
#: stay comparable.
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
TOP_K = 6

#: The two box sets carried through the report: Phantom's shipped box and the detector's.
#:
#: Querying the detector with a bare class name instead of the full phrase was tried and
#: dropped: aggregate quality was the same, but a bare noun can silently match the wrong
#: individual of the same class where a description cannot.
PHANTOM, FROM_DIS = "phantom", "dis"
BOX_SETS = (PHANTOM, FROM_DIS)

#: Alias kept for readability where the detector's box set is meant as "the new box".
CHOSEN_QUERY = FROM_DIS

#: A third possible *outcome* of the per-side choice, not a third box set: "there is no usable
#: box on this side". Only :func:`pick_side` with ``trust_detector=True`` can produce it, and
#: :func:`decide` turns it into an unconditional drop.
#:
#: It exists because the alternative -- reusing ``PHANTOM`` as the fallback -- makes a
#: filtered-out subject indistinguishable in the report from one where Phantom's box was
#: actively preferred. Both would read ``pick_ref = "phantom"``, and the funnel could not count
#: them apart. This is a distinct value so "the detector abstained" is visible.
NO_BOX = "no_box"

# --------------------------------------------------------------------------------------
# the keep/drop rule
# --------------------------------------------------------------------------------------

#: Identity gate, mandatory. A reference image is only usable if it shows the *same object*
#: as the target, and 0.6 was picked by looking at where the false negatives start: it does
#: throw away some genuine same-object pairs, but its false-*positive* rate is low, and a
#: wrong reference is worse than a missing one.
IDENTITY_MIN = 0.6

#: Semantic gate (one of two ways to pass). Lower than the 0.23 used against SAM2 cutouts
#: elsewhere in this repo, on purpose: these crops keep their background, which depresses CLIP
#: scores across the board, so 0.23 would be cutting on an artefact of the crop rather than on
#: the box being wrong.
CLIP_MIN = 0.21

#: Agreement gate (the other way to pass), also the box-selection threshold. At or above
#: this, Grounding DINO and Phantom's annotator agree about which object this is, which is
#: taken as confirmation of the human label rather than as a reason to replace it.
IOU_MIN = 0.75

#: Floor for :data:`RULE_IOU_FLOOR_PEAK`: the IoU *both* sides must clear. Separate from
#: :data:`IOU_MIN`, which that rule reuses as the peak one side must additionally reach.
#:
#: 0.5 rather than 0.75 because the premise is that Phantom's annotation is semantically right
#: and merely offset -- the detector is a normaliser, not a second opinion on *which* object.
#: Under that premise a middling IoU means "same object, shifted box", which is exactly what
#: should survive; only a *near-zero* IoU means the detector left the object. Measured on the
#: 140-subject pilot the two are cleanly separated: all 11 detector-wandered subjects have their
#: weaker side at or below 0.06, so anything from 0.3 to 0.5 catches the same 11 and the choice
#: within that band only trades tolerance.
IOU_FLOOR_MIN = 0.5

KEEP, DROP = "keep", "drop"

#: The two candidate rules, kept side by side so the choice can be made by looking rather than
#: argued. Both read the same three numbers; they differ only in whether identity can be
#: substituted for.
#:
#: ``identity AND (clip OR IoU)`` -- identity is non-negotiable.
RULE_IDENTITY_REQUIRED = "identity_required"
#: ``(clip AND identity) OR IoU`` -- a high IoU alone is enough.
RULE_IOU_STANDS = "iou_stands"
#: ``identity AND clip AND (both sides' IoU >= iou_min)`` -- the only rule that reads the two
#: IoUs separately instead of taking their ``max``.
#:
#: Added because under ``trust_detector`` the detector's box ships unconditionally, and the
#: ``max``-of-sides IoU cannot see a detector that wandered off: measured on the 140-subject
#: pilot, 11 subjects (8%) had one side at IoU < 0.3 *with Phantom's crop scoring higher* --
#: the detector had found a different object -- and 10 of them were kept, because a high
#: identity (the detector wandering to the *same* wrong object on both sides scores well) plus
#: a passing clip satisfied ``iou_stands`` on its own. Requiring both sides to agree with the
#: annotation catches all 11.
RULE_IOU_BOTH_SIDES = "iou_both_sides"
#: ``identity AND clip AND (both sides >= iou_floor) AND (one side >= iou_min)`` -- two-sided,
#: but with the two sides held to different bars.
#:
#: The premise this encodes, which the other rules do not: Phantom's annotation is taken to be
#: *semantically* correct and merely offset, so Grounding DINO is a normaliser rather than a
#: second opinion about which object the phrase names. Under that premise the two failure modes
#: are worth separating, and a single threshold cannot:
#:
#: * **the detector left the object** -- near-zero IoU on one side. The floor catches this. On
#:   the pilot all 11 such subjects have their weaker side at or below 0.06.
#: * **the box is merely offset** -- middling IoU on both sides. This should survive, and does:
#:   the IoU gate alone admits 109/140 here versus 71/140 when both sides must reach 0.75.
#:
#: The peak then asks that the normalisation be anchored: at least one side has to land where
#: detector and annotator genuinely agree. It is what separates "offset" from "both boxes are
#: vaguely in the area" -- 6 pilot subjects clear the floor with neither side reaching the peak
#: (e.g. ref 0.544 / target 0.587), and nothing in those establishes the box.
RULE_IOU_FLOOR_PEAK = "iou_floor_peak"

#: Identity alone. The rule for the **text-free chain** (:mod:`phantom_data.tighten`), where
#: there is no clip score and no detector-vs-annotation IoU to AND with -- the box comes from
#: SAM2 segmenting what Phantom's box points at, so "is the box on the intended object" is
#: answered by construction rather than by a confirming judge.
#:
#: Kept separate rather than letting the other four degrade gracefully on missing scores,
#: because a missing judge must never read as a passing one: on a text-free report all four of
#: the older rules correctly return keep=0/140, since each ANDs a clip score that does not
#: exist. Silently treating absent as satisfied is the failure mode this constant avoids.
RULE_IDENTITY_ONLY = "identity_only"

RULES = (RULE_IDENTITY_REQUIRED, RULE_IOU_STANDS, RULE_IOU_BOTH_SIDES, RULE_IOU_FLOOR_PEAK,
         RULE_IDENTITY_ONLY)
DEFAULT_RULE = RULE_IOU_STANDS


def resolve_snapshot(path: str) -> str:
    """A hub cache dir holds its weights one snapshot level down; a plain dir does not."""
    candidate = Path(path)
    if candidate.name == "snapshots":
        children = sorted(child for child in candidate.iterdir() if child.is_dir())
        if not children:
            raise FileNotFoundError(f"no snapshot under {candidate}")
        return str(children[0])
    return str(candidate)


class Models:
    """Grounding DINO + CLIP + DINOv2, each loaded on first use, all local-files-only.

    Lazy so that constructing ``Models`` costs nothing: the tests build one and never touch a
    weight, and a run that fails while reading the dataset should not first spend a minute
    loading three networks.
    """

    def __init__(self, dino_path: str = GROUNDING_DINO, clip_path: str = CLIP_MODEL,
                 dinov2_path: str = DINOV2_MODEL, device: str = "cuda") -> None:
        self.dino_path = dino_path
        self.clip_path = clip_path
        self.dinov2_path = dinov2_path
        self.device = device
        self._dino = None
        self._clip = None
        self._dinov2 = None

    @property
    def dino(self):
        if self._dino is None:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            path = resolve_snapshot(self.dino_path)
            processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                path, local_files_only=True).eval().to(self.device)
            self._dino = (processor, model, torch)
        return self._dino

    @property
    def clip(self):
        if self._clip is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            path = resolve_snapshot(self.clip_path)
            processor = CLIPProcessor.from_pretrained(path, local_files_only=True)
            model = CLIPModel.from_pretrained(path, local_files_only=True)
            self._clip = (processor, model.eval().to(self.device), torch)
        return self._clip

    @property
    def dinov2(self):
        if self._dinov2 is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            path = resolve_snapshot(self.dinov2_path)
            processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
            model = AutoModel.from_pretrained(path, local_files_only=True)
            self._dinov2 = (processor, model.eval().to(self.device), torch)
        return self._dinov2

    # ----- inference ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, query: str, threshold: float = BOX_THRESHOLD,
               text_threshold: float = TEXT_THRESHOLD,
               top_k: int = TOP_K) -> list[dict[str, Any]]:
        """Grounding DINO boxes for ``query``, highest confidence first.

        ``query`` is the full phrase, not a bare class name. Grounding DINO handles referring
        expressions natively (its training includes RefCOCO-style comprehension), so no
        separate code path is needed to feed it more than a noun.
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

    def crop_clip_score(self, frame: np.ndarray, box: Any, text: str) -> float | None:
        """CLIP cosine between the plain box crop and ``text`` (always the ``dis`` phrase)."""
        from PIL import Image

        crop = crop_box(frame, box)
        if crop is None or not str(text or "").strip():
            return None
        processor, model, torch = self.clip
        inputs = processor(text=[str(text)], images=[Image.fromarray(crop)],
                           return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            image_features = model.get_image_features(pixel_values=inputs.pixel_values)
            text_features = model.get_text_features(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return round(float((image_features * text_features).sum().item()), 6)

    def dino_embedding(self, frame: np.ndarray, box: Any):
        """Normalised DINOv2 pooled embedding of a box crop, or None for a missing box."""
        from PIL import Image

        crop = crop_box(frame, box)
        if crop is None:
            return None
        processor, model, torch = self.dinov2
        inputs = processor(images=[Image.fromarray(crop)], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            pooled = model(**inputs).pooler_output
        return pooled / pooled.norm(dim=-1, keepdim=True)

    def identity_cosine(self, ref_frame: np.ndarray, ref_box: Any, seed_frame: np.ndarray,
                        seed_box: Any) -> float | None:
        """DINOv2 cosine between the reference crop and the target crop.

        DINOv2 rather than CLIP because the question is whether these are the *same
        individual*, and CLIP's image-text training makes it answer "both are a man" with a
        high score. DINOv2 is self-supervised and appearance-sensitive, which is why it is
        the "DINO score" of subject-driven generation.
        """
        left = self.dino_embedding(ref_frame, ref_box)
        right = self.dino_embedding(seed_frame, seed_box)
        if left is None or right is None:
            return None
        return round(float((left * right).sum().item()), 6)


# --------------------------------------------------------------------------------------
# pure logic (unit tested against a stub Models)
# --------------------------------------------------------------------------------------


def best_by_detector(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The highest-confidence candidate.

    Deliberately not "the highest CLIP score": that variant was measured to disagree with
    the detector in 9/12 cases and to prefer tiny high-contrast fragments, because CLIP
    scores rise with crop tightness (r = +0.34 against log area).
    """
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.get("detector_score") or 0.0)


def pick_side(phantom_clip: float | None, box_iou: float | None, has_new_box: bool,
              clip_min: float = CLIP_MIN, iou_min: float = IOU_MIN,
              trust_detector: bool = True) -> tuple[str, str]:
    """Which box to use on one side, and why. Returns ``(box set, reason)``.

    Two modes, and which one is right depends on **where this runs in the pipeline**.

    ``trust_detector=True`` (the default) -- *the detector's box wins whenever it found one.*

    The reason is a coordinate-system argument, not a quality argument. Grounding DINO returns
    **real frame pixel coordinates**: it looked at the decoded frame, so its box needs no
    mapping and cannot be misplaced by a wrong one. Phantom's raw coordinates have to be
    projected through an annotation canvas that is *not resolved* -- ``H_768_long`` is the
    working hypothesis, and it is measurably wrong on the x axis (14% of all boxes overflow the
    canvas; 17-22% for 4:3 and 29% for 1:1 -- see the calibration section in the README).
    So the two boxes are not two opinions of equal standing: one is in the frame's own
    coordinates and the other is a guess about a projection. Phantom's box demotes to a prior,
    and its role shrinks to what it is still good for -- the IoU agreement number, and the
    ``dis`` phrase that found the object at all.

    That makes the no-box case a **filter-out** rather than a fallback. Returning
    :data:`NO_BOX`, not ``PHANTOM``: if the detector found nothing, the only box left is one we
    have no frame-space coordinates for, and shipping it is exactly the thing this mode exists
    to stop. :func:`decide` drops these unconditionally.

    ``trust_detector=False`` -- *the historical rule, preserved byte-for-byte.* This is what
    produced the pilot's ``gate_report.json``, so it stays reproducible as the control.
    It splits on whether Phantom's box was already good enough to protect:

    **Phantom's crop scored well** (``>= clip_min``) -- the annotation is basically right, and
    the detector is only allowed to *refine* it. ``IoU >= iou_min`` means the new box is a
    small correction, so it is taken; a lower IoU means the detector wandered off the object
    the annotator marked, so Phantom's box stands. The net effect is deliberate: where the
    annotation is sound, only small adjustments are accepted.

    **Phantom's crop scored badly** (``< clip_min``) -- the annotation is the thing in doubt,
    so there is nothing to protect. The detector's box is taken whenever it found one, and the
    keep/drop rule then decides whether the result is usable at all.

    Why the modes matter *now*: box correction moved to **before** SAM2 (the order is
    ``B extract -> enrich -> redetect -> gate -> C segment``). Under the old order the corrected
    box was a report annotation and keeping Phantom's box cost nothing; under the new one the
    chosen box is what SAM2 cuts from and what ships as the training condition, so
    "keep a box we already believe is skewed" became an active decision to poison the product.
    Measured on the pilot, the old rule kept Phantom's box on 47/140 reference sides and 33/140
    target sides purely because IoU fell below 0.75.
    """
    if not has_new_box:
        if trust_detector:
            return NO_BOX, ("detector found nothing and phantom's box has no trusted "
                            "frame-space coordinates, so this side is filtered out")
        return PHANTOM, "detector found nothing"
    if trust_detector:
        return FROM_DIS, ("detector box is in real frame coordinates, so it is preferred over "
                          "phantom's canvas-mapped box")
    if phantom_clip is not None and phantom_clip >= clip_min:
        if box_iou is not None and box_iou >= iou_min:
            return FROM_DIS, f"phantom box was fine; new box refines it (IoU >= {iou_min})"
        return PHANTOM, (f"phantom box was fine and the new box moved too far "
                         f"(IoU < {iou_min})")
    return FROM_DIS, f"phantom crop scored below {clip_min}, so the new box is used"


def score_side(models: Models, frame: np.ndarray, box: Any, dis: str) -> dict[str, Any]:
    """CLIP score plus geometry for one box on one side."""
    return {
        "box": list(box) if box else None,
        "crop_clip": models.crop_clip_score(frame, box, dis),
        "box_fraction": box_fraction(box, frame),
    }


def redetect_side(models: Models, frame: np.ndarray, dis: str) -> dict[str, Any]:
    """Detect on one frame, keeping the best box and how many candidates it beat.

    No CLIP here. The crop score is only computed for whichever box :func:`pick_side` ends up
    selecting, so scoring both would pay for a number that is usually thrown away -- and the
    selection reads Phantom's score, never the new box's, so it does not need it.

    One detector pass per side, with the full phrase as the query.
    """
    candidates = models.detect(frame, dis)
    best = best_by_detector(candidates)
    return {
        CHOSEN_QUERY: {
            "query": dis,
            "candidate_count": len(candidates),
            "box": (best or {}).get("box"),
            "detector_score": (best or {}).get("detector_score"),
            # Present but unset: the caller fills it in only if this box gets selected. The
            # key always exists so "not scored" and "no such field" cannot be confused.
            "crop_clip": None,
            "box_fraction": box_fraction((best or {}).get("box"), frame),
        }
    }


def analyse_subject(models: Models, ref_frame: np.ndarray, seed_frame: np.ndarray,
                    ref_box: Any, seed_box: Any, dis: str,
                    clip_min: float = CLIP_MIN,
                    iou_min: float = IOU_MIN,
                    trust_detector: bool = True) -> dict[str, Any]:
    """The whole cascade for one subject: detect, choose per side, then score the choice.

    Every subject is re-detected. An earlier version gated detection on Phantom's CLIP score
    to save GPU time, but Phantom's boxes are frequently skewed *without* scoring badly, so
    the gate was skipping exactly the boxes worth refining.

    Order matters here and is what keeps the cost down: the per-side choice
    (:func:`pick_side`) reads only Phantom's crop score and the two boxes' IoU, so the
    expensive identity cosine and the new box's crop score are computed once, on whichever box
    was actually selected. Nothing is discarded -- :func:`decide` does that afterwards, on the
    flattened numbers, so its thresholds can move without touching the GPU.

    ``trust_detector`` is passed straight to :func:`pick_side`; see there for why the default
    prefers the detector. A side picked :data:`NO_BOX` has no box to score, so its chosen CLIP
    score and the identity cosine come out None, and :func:`decide` reads that as a drop.
    Phantom's own numbers are still computed and reported for that side -- the point of the
    filter is that its box is untrustworthy, not that measuring it is uninteresting.
    """
    phantom_ref = score_side(models, ref_frame, ref_box, dis)
    phantom_seed = score_side(models, seed_frame, seed_box, dis)

    ref_sides: dict[str, Any] = {PHANTOM: phantom_ref, **redetect_side(models, ref_frame, dis)}
    seed_sides: dict[str, Any] = {PHANTOM: phantom_seed,
                                  **redetect_side(models, seed_frame, dis)}

    ious = {
        "dis_vs_phantom": iou(ref_sides[FROM_DIS]["box"], phantom_ref["box"]),
        "seed_dis_vs_phantom": iou(seed_sides[FROM_DIS]["box"], phantom_seed["box"]),
    }
    ref_pick, ref_reason = pick_side(phantom_ref["crop_clip"], ious["dis_vs_phantom"],
                                    bool(ref_sides[FROM_DIS]["box"]), clip_min, iou_min,
                                    trust_detector)
    seed_pick, seed_reason = pick_side(phantom_seed["crop_clip"], ious["seed_dis_vs_phantom"],
                                       bool(seed_sides[FROM_DIS]["box"]), clip_min, iou_min,
                                       trust_detector)

    # The chosen box's CLIP score, computed only where it was not already known. Phantom's is
    # always known; the new box's was skipped in redetect_side precisely to land here.
    # ``NO_BOX`` is not a key in ``sides`` -- there is no box, so there is nothing to score.
    for sides, frame_, pick in ((ref_sides, ref_frame, ref_pick),
                                (seed_sides, seed_frame, seed_pick)):
        entry = sides.get(pick) or {}
        if entry.get("crop_clip") is None and entry.get("box"):
            entry["crop_clip"] = models.crop_clip_score(frame_, entry["box"], dis)

    # One identity cosine, on the two boxes actually selected -- the pair that will ship.
    # A ``NO_BOX`` side contributes no box, so ``identity_cosine`` returns None and the pair
    # fails the mandatory identity gate. That is the intended route to the drop.
    chosen_cos = models.identity_cosine(ref_frame, (ref_sides.get(ref_pick) or {}).get("box"),
                                        seed_frame,
                                        (seed_sides.get(seed_pick) or {}).get("box"))
    return {
        "dis": dis,
        "clip_min": clip_min,
        "iou_min": iou_min,
        "trust_detector": trust_detector,
        "ref": ref_sides,
        "seed": seed_sides,
        "ref_pick": ref_pick,
        "ref_pick_reason": ref_reason,
        "seed_pick": seed_pick,
        "seed_pick_reason": seed_reason,
        "dino_cos_chosen": chosen_cos,
        "iou": ious,
        # ``FROM_DIS`` only: a ``NO_BOX`` side replaced nothing, it filtered the subject out.
        # Counting it as a replacement would inflate the "boxes corrected" figure with
        # subjects that never ship.
        "replaced": [side for side, pick in (("ref", ref_pick), ("seed", seed_pick))
                     if pick == FROM_DIS],
    }


# --------------------------------------------------------------------------------------
# report flattening (what the viewer reads)
# --------------------------------------------------------------------------------------


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return format(float(value), f".{digits}f")


def delta(new: Any, old: Any) -> str:
    if new is None or old is None:
        return "n/a"
    return f"{float(new) - float(old):+.3f}"


def flat_scores(analysis: dict[str, Any]) -> dict[str, Any]:
    """The six CLIP scores and three cosines as flat keys the viewer can sort on.

    All six share one text (``dis``) and one crop treatment, so any pair of them can be
    subtracted directly.
    """
    flat: dict[str, Any] = {}
    for name in BOX_SETS:
        for side in ("ref", "seed"):
            entry = analysis[side].get(name) or {}
            flat[f"crop_clip_{side}_{name}"] = entry.get("crop_clip")
            flat[f"box_{side}_{name}"] = entry.get("box")
            if name != PHANTOM:
                flat[f"detector_score_{side}_{name}"] = entry.get("detector_score")
                flat[f"candidates_{side}_{name}"] = entry.get("candidate_count")
    for key, value in analysis["iou"].items():
        flat[f"iou_{key}"] = value
    # Which box each side settled on, and the numbers for that choice. The viewer overlays
    # boxes from the coordinates above, so it needs to know which pair is the live one.
    #
    # ``NO_BOX`` is not one of ``BOX_SETS``, so it has no ``box_<side>_<pick>`` key to copy
    # from -- the chosen box and score are None, which is the honest reading of "the detector
    # abstained and phantom's box is not trusted here".
    for side in ("ref", "seed"):
        pick = analysis[f"{side}_pick"]
        flat[f"pick_{side}"] = pick
        flat[f"pick_{side}_reason"] = analysis[f"{side}_pick_reason"]
        flat[f"chosen_box_{side}"] = flat.get(f"box_{side}_{pick}")
        flat[f"chosen_clip_{side}"] = flat.get(f"crop_clip_{side}_{pick}")
    flat["dino_cos_chosen"] = analysis["dino_cos_chosen"]
    flat["replaced"] = analysis["replaced"]
    return flat


def _best(values: list[Any]) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def decide(scores: dict[str, Any], rule: str = DEFAULT_RULE,
           identity_min: float = IDENTITY_MIN, clip_min: float = CLIP_MIN,
           iou_min: float = IOU_MIN,
           iou_floor: float = IOU_FLOOR_MIN) -> dict[str, Any]:
    """Apply one of the four keep/drop rules to a subject's flat scores.

    Runs on the boxes :func:`pick_side` already selected -- box selection and keep/drop are
    separate questions, and doing them together made it possible to judge one box set on
    another's numbers.

    The three measurements, each ``max`` over the two sides rather than ``min``: one confirmed
    side plus a passing identity check already pins both.

    * **identity** -- ``dino_cos`` between the two chosen crops. Whether they show the same
      object.
    * **clip** -- the better crop's score against the phrase. Whether a box is on the thing
      the phrase names.
    * **IoU** -- the better side's agreement between the new box and Phantom's. Whether the
      detector and the annotator picked the same region.

    The first two rules require at least one of clip or IoU, and differ **only in whether IoU can
    substitute for identity**, which is easier to read as a table than as two branches:

    =========================  =====================  ================
    identity / clip / IoU      ``identity_required``  ``iou_stands``
    =========================  =====================  ================
    pass / pass / any          keep                   keep
    pass / fail / pass         keep                   keep
    pass / fail / fail         drop                   drop
    fail / any  / pass         **drop**               **keep**
    fail / any  / fail         drop                   drop
    =========================  =====================  ================

    The other two demand identity **and** clip, and differ in how they read the two IoUs -- both
    reject the one-sided pass that ``max`` hides, but with different tolerance for an offset box:

    ==============================  ===================  ==================
    ref IoU / target IoU            ``iou_both_sides``   ``iou_floor_peak``
    ==============================  ===================  ==================
    0.90 / 0.85                     keep                 keep
    0.90 / 0.60  (offset one side)  **drop**             **keep**
    0.60 / 0.55  (both loose)       drop                 drop
    0.90 / 0.02  (detector left)    drop                 drop
    ==============================  ===================  ==================

    Measured on the 140-subject pilot at ``identity>=0.6, clip>=0.21, iou>=0.75``:
    ``iou_stands`` 131, ``identity_required`` 70, ``iou_floor_peak`` 53 (floor 0.5),
    ``iou_both_sides`` 36. The two-sided pair catch the same 11 detector-wandered subjects;
    ``iou_floor_peak`` keeps 17 more, all of them boxes that are offset rather than wrong.

    :data:`RULE_IDENTITY_REQUIRED` treats identity as non-negotiable: a reference that is not
    the same object as the target is unusable however good the rest looks, so clip and IoU only
    compete to confirm the box.

    :data:`RULE_IOU_STANDS` lets a high IoU carry a subject on its own. Worth being explicit
    about the tradeoff, because it is the whole reason to compare the two: IoU is computed
    within each frame against Phantom's box for that same frame, so it says nothing about
    whether the reference and target frames -- median 83 seconds apart -- show the same
    individual. Measured on this dataset it keeps 56 subjects the other rule drops -- and none
    the other rule keeps -- every one of them rejected by identity, with a median identity of
    0.485 and 15 below 0.4.

    ``rescued_by_iou`` in the result flags the subjects that passed *because* of the IoU branch.
    That count runs slightly higher than 56 (57 here) because a subject whose identity passes
    but whose clip narrowly misses also takes this branch; the net difference between the rules
    is the number to compare.
    """
    identity = scores.get("dino_cos_chosen")
    clip = _best([scores.get(f"chosen_clip_{side}") for side in ("ref", "seed")])
    iou = _best([scores.get("iou_dis_vs_phantom"),
                 scores.get("iou_seed_dis_vs_phantom")])
    picks = {side: scores.get(f"pick_{side}") for side in ("ref", "seed")}
    box_set = ("phantom" if set(picks.values()) == {PHANTOM}
               else FROM_DIS if set(picks.values()) == {FROM_DIS}
               else f"ref={picks['ref']}, target={picks['seed']}")
    box_reason = "; ".join(f"{side}: {scores.get(f'pick_{side}_reason')}"
                           for side in ("ref", "seed"))

    identity_ok = identity is not None and identity >= identity_min
    clip_ok = clip is not None and clip >= clip_min
    iou_ok = iou is not None and iou >= iou_min

    # Per-side IoUs, for the rule that will not accept one side standing in for the other. A
    # missing side is a failure, not a skip: "we never measured it" must not read as "it
    # agreed".
    iou_ref = scores.get("iou_dis_vs_phantom")
    iou_seed = scores.get("iou_seed_dis_vs_phantom")
    iou_both_ok = (iou_ref is not None and iou_seed is not None
                   and iou_ref >= iou_min and iou_seed >= iou_min)

    # The floor/peak split. ``both_measured`` is factored out because a missing side must fail
    # the floor *and* the peak: with only one side measured, ``max`` of the present values would
    # otherwise satisfy the peak on its own, which is the exact one-sided pass this rule exists
    # to refuse.
    both_measured = iou_ref is not None and iou_seed is not None
    iou_floor_ok = both_measured and min(iou_ref, iou_seed) >= iou_floor
    iou_peak_ok = both_measured and max(iou_ref, iou_seed) >= iou_min
    iou_floor_peak_ok = iou_floor_ok and iou_peak_ok

    # A side with no usable box short-circuits both rules. This guard is not redundant: the
    # three numbers are each a ``max`` over the two sides, so a subject missing only its
    # *reference* box still carries the target side's IoU, and ``iou_stands`` would keep it on
    # that number alone -- shipping a pair with one box missing. The gate must read the pick,
    # not just the scores.
    missing = [side for side in ("ref", "seed") if picks[side] == NO_BOX]
    if missing:
        return {
            "verdict": DROP,
            "rule": rule,
            "box_set": box_set,
            "box_reason": box_reason,
            "reason": (f"no usable box on the {' and '.join(missing)} side "
                       f"(detector found nothing and phantom's box is not trusted)"),
            "identity": identity,
            "identity_ok": identity_ok,
            "clip": clip,
            "clip_ok": clip_ok,
            "iou": iou,
            "iou_ok": iou_ok,
            "iou_ref": iou_ref,
            "iou_seed": iou_seed,
            "iou_both_ok": iou_both_ok,
            "iou_floor_ok": iou_floor_ok,
            "iou_peak_ok": iou_peak_ok,
            "iou_floor_peak_ok": iou_floor_peak_ok,
            "rescued_by_iou": False,
            "no_box_sides": missing,
        }

    if rule == RULE_IOU_STANDS:
        verdict = KEEP if ((clip_ok and identity_ok) or iou_ok) else DROP
        if clip_ok and identity_ok:
            reason = f"clip {fmt(clip)} and identity {fmt(identity)} both pass"
        elif iou_ok:
            # The branch worth watching: it admits a pair the identity gate rejected, on the
            # strength of the boxes agreeing. Agreement is measured within each frame
            # separately, so it carries no information about whether the two frames -- median
            # 83 seconds apart -- show the same individual.
            reason = (f"IoU {fmt(iou)} >= {iou_min} alone; identity {fmt(identity)} "
                      f"{'passes' if identity_ok else f'is below {identity_min}'} and clip "
                      f"{fmt(clip)} {'passes' if clip_ok else f'is below {clip_min}'}")
        elif not identity_ok:
            reason = (f"identity {fmt(identity)} < {identity_min} and IoU {fmt(iou)} < "
                      f"{iou_min} — nothing establishes the pair")
        else:
            reason = (f"clip {fmt(clip)} < {clip_min} and IoU {fmt(iou)} < {iou_min} — "
                      f"nothing confirms the box is on the intended object")
    elif rule == RULE_IOU_BOTH_SIDES:
        verdict = KEEP if (identity_ok and clip_ok and iou_both_ok) else DROP
        sides = f"ref {fmt(iou_ref)} / target {fmt(iou_seed)}"
        if not iou_both_ok:
            # Named first because it is the failure this rule exists to catch, and it is the
            # one a high identity would otherwise hide.
            reason = (f"IoU must pass on both sides (>= {iou_min}) and did not: {sides} — the "
                      f"detector disagrees with the annotation on at least one side")
        elif not identity_ok:
            reason = (f"both IoUs pass ({sides}) but identity {fmt(identity)} < "
                      f"{identity_min} — probably not the same object")
        elif not clip_ok:
            reason = (f"both IoUs pass ({sides}) but clip {fmt(clip)} < {clip_min} — nothing "
                      f"confirms the box is on the intended object")
        else:
            reason = (f"identity {fmt(identity)}, clip {fmt(clip)} and both IoUs ({sides}) "
                      f"all pass")
    elif rule == RULE_IOU_FLOOR_PEAK:
        verdict = KEEP if (identity_ok and clip_ok and iou_floor_peak_ok) else DROP
        sides = f"ref {fmt(iou_ref)} / target {fmt(iou_seed)}"
        if not iou_floor_ok:
            # The failure the floor exists for, and it is named first because it is the only one
            # here that means the box is on the *wrong object* rather than merely loose.
            reason = (f"one side's IoU is below the floor {iou_floor} ({sides}) — the detector "
                      f"probably left the annotated object on that side")
        elif not iou_peak_ok:
            reason = (f"both sides clear the floor {iou_floor} ({sides}) but neither reaches "
                      f"{iou_min} — the boxes are in the area but nothing anchors them")
        elif not identity_ok:
            reason = (f"the IoUs pass ({sides}) but identity {fmt(identity)} < {identity_min} — "
                      f"probably not the same object")
        elif not clip_ok:
            reason = (f"the IoUs pass ({sides}) but clip {fmt(clip)} < {clip_min} — nothing "
                      f"confirms the box is on the intended object")
        else:
            reason = (f"identity {fmt(identity)}, clip {fmt(clip)}, both sides >= {iou_floor} "
                      f"and one side >= {iou_min} ({sides})")
    elif rule == RULE_IDENTITY_ONLY:
        # No clip, no detector IoU: the text-free chain has one judge. A subject with no
        # identity score at all is dropped rather than kept -- an unmeasured pair is not an
        # endorsed one, and the missing-box guard above has already handled the no-box case.
        verdict = KEEP if identity_ok else DROP
        if identity is None:
            reason = "no identity score — nothing judged this pair"
        elif identity_ok:
            reason = (f"identity {fmt(identity)} >= {identity_min}; the box is SAM2's mask "
                      f"around what phantom's box pointed at, so no separate judge confirms it")
        else:
            reason = (f"identity {fmt(identity)} < {identity_min} — the reference and the "
                      f"target are probably not the same object")
    else:
        verdict = KEEP if (identity_ok and (clip_ok or iou_ok)) else DROP
        if not identity_ok:
            reason = (f"identity {fmt(identity)} < {identity_min} — the reference and the "
                      f"target are probably not the same object")
        elif clip_ok and iou_ok:
            reason = f"identity ok; both clip {fmt(clip)} and IoU {fmt(iou)} pass"
        elif clip_ok:
            reason = f"identity ok; clip {fmt(clip)} >= {clip_min}"
        elif iou_ok:
            reason = (f"identity ok; IoU {fmt(iou)} >= {iou_min} — clip {fmt(clip)} is below "
                      f"{clip_min} but the box agrees with Phantom's")
        else:
            reason = (f"identity ok, but clip {fmt(clip)} < {clip_min} and IoU {fmt(iou)} < "
                      f"{iou_min} — nothing confirms the box is on the intended object")
    return {
        "verdict": verdict,
        "rule": rule,
        "box_set": box_set,
        "box_reason": box_reason,
        "reason": reason,
        "identity": identity,
        "identity_ok": identity_ok,
        "clip": clip,
        "clip_ok": clip_ok,
        "iou": iou,
        "iou_ok": iou_ok,
        # Per-side IoUs and their conjunction, reported under every rule so the two-sided
        # figure can be inspected (and a threshold re-tuned) without re-running the GPU pass.
        "iou_ref": iou_ref,
        "iou_seed": iou_seed,
        "iou_both_ok": iou_both_ok,
        # The floor/peak pair reported separately, so the two failure modes stay countable: a
        # floor failure means the detector left the object, a peak failure means both boxes are
        # merely loose. Collapsing them into one flag loses the distinction the rule is built on.
        "iou_floor_ok": iou_floor_ok,
        "iou_peak_ok": iou_peak_ok,
        "iou_floor_peak_ok": iou_floor_peak_ok,
        # True when this subject only survives because IoU stood in for identity. Counted
        # separately because that is the substitution the rule choice is really about.
        "rescued_by_iou": (rule == RULE_IOU_STANDS and verdict == KEEP
                           and iou_ok and not (clip_ok and identity_ok)),
        # Always present so callers can read it without a ``.get`` default; empty on this
        # path by construction (the missing-box case returned above).
        "no_box_sides": [],
    }


def detail_extra(analysis: dict[str, Any]) -> dict[str, Any]:
    """The field/value table the viewer shows under each sheet.

    Ordered so the verdict and the three numbers it turns on come first, and everything else
    is context for those. The three gates are marked pass/fail individually, because "why was
    this dropped" is the question the table exists to answer.
    """
    scores = flat_scores(analysis)
    ruling = decide(scores)
    mark = lambda ok: "PASS" if ok else "fail"  # noqa: E731
    rows: dict[str, Any] = {
        "verdict": ruling["verdict"].upper(),
        "why": ruling["reason"],
        "phrase": analysis["dis"],
        "boxes used": ruling["box_set"],
        f"identity dino_cos >= {IDENTITY_MIN}":
            f"{fmt(ruling['identity'], 4)}  {mark(ruling['identity_ok'])}  (mandatory)",
        f"crop clip >= {CLIP_MIN}":
            f"{fmt(ruling['clip'])}  {mark(ruling['clip_ok'])}  (best of the two sides)",
        f"IoU vs phantom >= {IOU_MIN}":
            f"{fmt(ruling['iou'], 4)}  {mark(ruling['iou_ok'])}  (best of the two sides)",
    }
    for side, label in (("ref", "reference"), ("seed", "target")):
        rows[f"{label}: box chosen"] = (f"{scores.get(f'pick_{side}')}  "
                                        f"({scores.get(f'pick_{side}_reason')})")
        phantom_clip = scores.get(f"crop_clip_{side}_{PHANTOM}")
        new_clip = scores.get(f"crop_clip_{side}_{FROM_DIS}")
        line = f"phantom {fmt(phantom_clip)}"
        if new_clip is not None:
            line += f"   new {fmt(new_clip)} ({delta(new_clip, phantom_clip)})"
        rows[f"{label}: crop clip"] = line
        # "conf" spelled out rather than abbreviated: this is the detector's confidence, not
        # anything to do with the phrase.
        rows[f"{label}: detector"] = (
            f"conf {fmt(scores.get(f'detector_score_{side}_{FROM_DIS}'))}"
            f"  cand {scores.get(f'candidates_{side}_{FROM_DIS}')}")
    for key, label in (("dis_vs_phantom", "IoU reference: new box vs phantom"),
                       ("seed_dis_vs_phantom", "IoU target: new box vs phantom")):
        rows[label] = fmt(scores.get(f"iou_{key}"), 4)
    return rows


def subject_record(sample_id: str, subject_id: int, analysis: dict[str, Any],
                   frames: dict[str, str], entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """One report row.

    ``frames`` carries the paths of the two *unannotated* frames. The viewer draws the boxes
    itself from the coordinates in this record, so changing a colour, a line width, or which
    boxes are shown costs a page reload instead of a GPU pass over the dataset.

    ``entry`` is the subject row this record was built from, read only for provenance fields.
    Two of them are **absent under the new pipeline order** and that is expected, not a bug:
    ``ref_clip_score`` and ``ref_mask_coverage`` are SAM2 numbers, and SAM2 now runs *after*
    this pass. They stay in the schema (as None) rather than being deleted, because
    ``trust_detector=False`` against a stage C manifest still fills them and the pilot's report
    is meant to keep validating against this same shape. Neither was ever an input to a
    decision -- ``decide`` reads none of them -- so losing them costs display context only, and
    ``ref_clip_score`` was never comparable to the ``crop_clip_*`` fields anyway: that one sees
    SAM2's white-matte cutout, these see the plain crop, and the same box can differ by 0.05.
    """
    entry = entry or {}
    scores = flat_scores(analysis)
    ruling = decide(scores)
    return {
        "sample_id": sample_id,
        "subject_id": int(subject_id),
        "phrase": analysis["dis"],
        "phrase_words": len(str(analysis["dis"]).split()),
        "ref_clip_score": entry.get("ref_clip_score"),
        "ref_mask_coverage": entry.get("ref_mask_coverage"),
        "text_source": analysis.get("text_source"),
        "ref_frame": frames["ref"],
        "seed_frame": frames["seed"],
        "seed_frame_index": entry.get("seed_frame_index"),
        "extra": detail_extra(analysis),
        **scores,
        "dis": analysis["dis"],
        # The ruling is stored rather than recomputed in the viewer so the page and the sheet
        # cannot disagree. The thresholds live in this module, so changing one is a re-run of
        # this cheap flattening pass, not of the GPU work.
        "verdict": ruling["verdict"],
        "verdict_reason": ruling["reason"],
        "verdict_box_set": ruling["box_set"],
        "verdict_box_reason": ruling["box_reason"],
        "rule_identity": ruling["identity"],
        "rule_identity_ok": ruling["identity_ok"],
        "rule_clip": ruling["clip"],
        "rule_clip_ok": ruling["clip_ok"],
        "rule_iou": ruling["iou"],
        "rule_iou_ok": ruling["iou_ok"],
    }


# --------------------------------------------------------------------------------------
# summary (printed after a run, and the basis of the tables to be judged)
# --------------------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    ordered = sorted(v for v in values if v is not None)
    return ordered[len(ordered) // 2] if ordered else None


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """What the rule did, and the medians behind it.

    The keep/drop counts lead, then a breakdown of *which gate* did the dropping -- the two
    reasons are actionable in different ways. A subject failing identity may be a genuinely
    mismatched reference or an over-strict 0.6; a subject passing identity but failing both
    alternatives is a box nothing could confirm.
    """
    total = len(records)
    kept = [r for r in records if r.get("verdict") == KEEP]
    dropped = [r for r in records if r.get("verdict") == DROP]

    def column(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [r[key] for r in rows if r.get(key) is not None]

    def replaced(rows: list[dict[str, Any]], side: str) -> int:
        return sum(1 for r in rows if r.get(f"pick_{side}") == FROM_DIS)

    summary: dict[str, Any] = {
        "subjects": total,
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_on_identity": sum(1 for r in dropped if not r.get("rule_identity_ok")),
        "dropped_on_clip_and_iou": sum(1 for r in dropped if r.get("rule_identity_ok")),
        "kept_by_clip_only": sum(1 for r in kept
                                 if r.get("rule_clip_ok") and not r.get("rule_iou_ok")),
        "kept_by_iou_only": sum(1 for r in kept
                                if r.get("rule_iou_ok") and not r.get("rule_clip_ok")),
        "kept_by_both": sum(1 for r in kept
                            if r.get("rule_clip_ok") and r.get("rule_iou_ok")),
        # Per side, because the choice is made per side: a subject can keep Phantom's
        # reference box and take the detector's target box.
        "ref_box_replaced": replaced(records, "ref"),
        "seed_box_replaced": replaced(records, "seed"),
        "both_boxes_replaced": sum(1 for r in records
                                   if r.get("pick_ref") == FROM_DIS
                                   and r.get("pick_seed") == FROM_DIS),
        "no_box_found_ref": sum(1 for r in records if not r.get(f"box_ref_{FROM_DIS}")),
        "no_box_found_seed": sum(1 for r in records if not r.get(f"box_seed_{FROM_DIS}")),
        # Under ``trust_detector=True`` a side with no detected box is filtered out rather
        # than falling back to Phantom's. Counted separately from the two lines above: those
        # say "the detector found nothing", this says "and that decided the subject's fate".
        # Always 0 under ``trust_detector=False``, where the fallback absorbs the case.
        "filtered_no_box_ref": sum(1 for r in records if r.get("pick_ref") == NO_BOX),
        "filtered_no_box_seed": sum(1 for r in records if r.get("pick_seed") == NO_BOX),
        "fell_back_to_phantom_text": sum(1 for r in records
                                         if r.get("text_source") == "phantom_fallback"),
    }
    for name in BOX_SETS:
        summary[f"median_clip_ref_{name}"] = _median(column(records, f"crop_clip_ref_{name}"))
        summary[f"median_clip_seed_{name}"] = _median(
            column(records, f"crop_clip_seed_{name}"))
    summary["median_dino_cos_chosen"] = _median(column(records, "dino_cos_chosen"))
    summary["median_detector_conf_ref"] = _median(
        column(records, f"detector_score_ref_{FROM_DIS}"))
    summary["median_iou_ref_vs_phantom"] = _median(column(records, "iou_dis_vs_phantom"))
    summary["median_iou_seed_vs_phantom"] = _median(
        column(records, "iou_seed_dis_vs_phantom"))
    return summary
