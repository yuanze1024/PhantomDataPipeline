"""Streamlit browser for the keep/drop rule: what it filtered, and why.

Boxes are drawn **here**, not baked into the images. The render pass writes each subject's two
frames unannotated and puts every coordinate in ``gate_report.json``, so a colour, a line
width, or hiding one box is a page reload rather than a GPU pass over the dataset. Both boxes
are always available: judging whether an IoU of 0.6 means "small nudge" or "different object"
requires seeing the overlap, not just the number.

No *measurement* happens here -- every score is read from the report as the run produced it.
The keep/drop **verdict**, on the other hand, is computed live from those scores, which is what
lets the rule selector and the three threshold sliders take effect immediately. It calls
:func:`phantom_data.redetect.decide`, the same function the pipeline uses, so the page cannot
apply a rule the pipeline would not. The report's own stored verdict fields are deliberately
ignored: they were computed under whatever thresholds that run used, and showing them next to
a live verdict that disagrees is worse than not showing them.

Four rules are offered because the choice between them is a judgement about this dataset that
is best made by looking (see :func:`phantom_data.redetect.decide` for the reasoning):

* ``identity AND (clip OR IoU)`` -- identity is non-negotiable.
* ``(clip AND identity) OR IoU`` -- a high IoU alone suffices. At the default thresholds this
  keeps 56 subjects the first rule drops, and none that it keeps; every one of the 56 was
  rejected by identity.
* ``identity AND clip AND (both sides' IoU)`` -- the box has to agree with the annotation on
  the reference *and* the target. The first two take the ``max`` of the two IoUs, which cannot
  see a detector that wandered onto a different object on one side.
* ``identity AND clip AND (both sides >= floor, one side >= peak)`` -- also two-sided, but it
  separates "the box is offset" from "the detector left the object" by holding the two sides to
  different bars.

The last two differ only in tolerance for an offset box, and that difference is the whole
question, so the page reports it three ways rather than as one keep number: the keep count swept
across the peak **and** across the floor, each gate's pass rate in isolation (the keep total is
close to the product of near-independent gates, so it reads far lower than any one of them), and
**the individual subjects the two rules disagree about**, listed with their two IoUs. A count
tells you the size of a decision; the list is what lets you check it.

Which box each side uses is a separate decision made at render time, per side: where Phantom's
crop already scored well the detector may only refine it (high IoU), and where it scored badly
the detector's box is taken outright. No slider changes that.

Streamlit compatibility: the deployment runs 1.23.1 -- ``use_column_width`` (not
``use_container_width``) on images, ``st.experimental_rerun`` (not ``st.rerun``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Absolute, not relative: ``streamlit run`` executes this file as a top-level script, so
# ``from .boxes import ...`` raises "attempted relative import with no known parent package"
# at page-render time -- which the headless smoke does not catch, because it imports the
# module as part of the package.
from phantom_data import labels as labelling
from phantom_data import redetect
from phantom_data.boxes import clamp_box

DEFAULT_DATASET = "/mnt/pfs/data/yuanze/phantom_koala_inspect100_v1"
DEFAULT_OUT_ROOT = "_redetect100"

KEEP, DROP = "keep", "drop"
PHANTOM, NEW = "phantom", "dis"

#: The two rules, labelled for the radio. Verdicts are computed here from the stored scores,
#: so moving a threshold updates the counts immediately -- the report's own verdict fields are
#: ignored by this page precisely so the two cannot drift apart in the reader's head.
RULE_LABELS = {
    "identity must pass, then clip or IoU confirms it": redetect.RULE_IDENTITY_REQUIRED,
    "(clip and identity) or IoU on its own": redetect.RULE_IOU_STANDS,
    "identity and clip and IoU on BOTH sides": redetect.RULE_IOU_BOTH_SIDES,
    "identity and clip and (both sides >= floor, one side >= peak)":
        redetect.RULE_IOU_FLOOR_PEAK,
    "identity alone (text-free chain: no clip, no detector IoU)":
        redetect.RULE_IDENTITY_ONLY,
}

#: Formula strings for the caption, keyed the same way ``decide`` branches. A dict rather than a
#: conditional expression so adding a fourth rule cannot leave the caption silently describing
#: the wrong one.
RULE_FORMULAS = {
    redetect.RULE_IDENTITY_REQUIRED: "identity AND (clip OR IoU)",
    redetect.RULE_IOU_STANDS: "(clip AND identity) OR IoU",
    redetect.RULE_IOU_BOTH_SIDES: "identity AND clip AND (IoU_ref AND IoU_target)",
    redetect.RULE_IOU_FLOOR_PEAK: "identity AND clip AND min(IoU) >= floor AND max(IoU) >= peak",
    redetect.RULE_IDENTITY_ONLY: "identity",
}

#: The IoU values the sweep evaluates. Coarse on purpose: the point is the shape of the curve
#: and where it falls off a cliff, and 21 points over the full range renders instantly while
#: 0.01 steps would judge 140 subjects 101 times on every page view.
SWEEP_STEPS = tuple(round(0.05 * i, 2) for i in range(21))

#: Box colours, also used for the crop labels so a crop traces back to its rectangle by
#: colour alone.
PHANTOM_COLOUR = (255, 96, 96)      # red
NEW_COLOUR = (96, 176, 255)         # blue
CHOSEN_OUTLINE = (255, 255, 255)    # white tick beside whichever box was selected

FRAME_WIDTH = 620
CROP_WIDTH = 320

SHOW_ALL = "all"
SHOW_KEPT = "kept"
SHOW_DROPPED = "dropped"
SHOW_DROPPED_IDENTITY = "dropped: not the same object (identity gate)"
SHOW_DROPPED_UNCONFIRMED = "dropped: nothing confirmed the box (clip and IoU both failed)"
SHOW_KEPT_BY_IOU = "kept only because the box agrees with Phantom's"
SHOW_ONE_SIDED = "one side's IoU passes, the other does not"
SHOW_FLOOR_PEAK_ONLY = "the two two-sided rules disagree (offset, not wrong object)"
SHOW_BOX_REPLACED = "at least one box replaced by the detector"
SHOW_BOX_KEPT = "both boxes kept as Phantom drew them"

SHOW_FILTERS = (SHOW_ALL, SHOW_KEPT, SHOW_DROPPED, SHOW_DROPPED_IDENTITY,
                SHOW_DROPPED_UNCONFIRMED, SHOW_KEPT_BY_IOU, SHOW_ONE_SIDED,
                SHOW_FLOOR_PEAK_ONLY, SHOW_BOX_REPLACED, SHOW_BOX_KEPT)

SORT_IDENTITY = "identity dino_cos, ascending"
SORT_CLIP = "crop clip, ascending"
SORT_IOU_REF = "IoU reference (new vs phantom), ascending"
SORT_IOU_SEED = "IoU target (new vs phantom), ascending"
SORT_IOU_WEAK = "weaker side's IoU, ascending"
SORT_SAMPLE = "sample id"

SORT_ORDERS = (SORT_IDENTITY, SORT_CLIP, SORT_IOU_REF, SORT_IOU_SEED, SORT_IOU_WEAK,
               SORT_SAMPLE)


def load_report(dataset: Path, out_root: str = DEFAULT_OUT_ROOT) -> dict[str, Any]:
    path = dataset / out_root / "gate_report.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}


def replaced_sides(subject: dict[str, Any]) -> list[str]:
    """Which sides ended up on the detector's box. Per side, because the choice is per side."""
    return [side for side in ("ref", "seed") if subject.get(f"pick_{side}") == NEW]


def judge(subject: dict[str, Any], rule: str, identity_min: float, clip_min: float,
          iou_min: float, iou_floor: float = redetect.IOU_FLOOR_MIN) -> dict[str, Any]:
    """The live verdict for one subject under the selected rule and thresholds.

    Delegates to :func:`redetect.decide` rather than reimplementing the comparisons, so the
    page cannot drift from the rule the pipeline would apply.
    """
    return redetect.decide(subject, rule, identity_min, clip_min, iou_min, iou_floor)


def annotate(subjects: list[dict[str, Any]], rule: str, identity_min: float, clip_min: float,
             iou_min: float, iou_floor: float = redetect.IOU_FLOOR_MIN) -> list[dict[str, Any]]:
    """Attach the live verdict to each subject under ``_live``.

    Done once per page render and threaded through the filters, ordering, counts and labels:
    re-judging inside each of those would evaluate the same rule four times and, worse, let
    them disagree if one were ever updated and another not.
    """
    return [{**s, "_live": judge(s, rule, identity_min, clip_min, iou_min, iou_floor)}
            for s in subjects]


def _live(subject: dict[str, Any]) -> dict[str, Any]:
    return subject.get("_live") or {}


def filter_subjects(subjects: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Subjects matching one filter, judged by the live verdict in ``_live``.

    The two drop filters split the drops by cause, because they mean different things: an
    identity failure is either a genuinely mismatched pair or an over-strict threshold, while a
    subject that cleared identity and failed both alternatives is a box nothing could confirm.
    """
    if mode == SHOW_KEPT:
        return [s for s in subjects if _live(s).get("verdict") == KEEP]
    if mode == SHOW_DROPPED:
        return [s for s in subjects if _live(s).get("verdict") == DROP]
    if mode == SHOW_DROPPED_IDENTITY:
        return [s for s in subjects if _live(s).get("verdict") == DROP
                and not _live(s).get("identity_ok")]
    if mode == SHOW_DROPPED_UNCONFIRMED:
        return [s for s in subjects if _live(s).get("verdict") == DROP
                and _live(s).get("identity_ok")]
    if mode == SHOW_KEPT_BY_IOU:
        # Under the identity-required rule this means "clip failed, IoU confirmed the box";
        # under the IoU-stands rule it means "IoU substituted for identity". Both are the
        # subjects that would be lost without the IoU route, which is what makes them the set
        # worth looking at when choosing between the rules.
        return [s for s in subjects if _live(s).get("verdict") == KEEP
                and _live(s).get("iou_ok") and not (_live(s).get("clip_ok")
                                                    and _live(s).get("identity_ok"))]
    if mode == SHOW_ONE_SIDED:
        # Exactly the disagreement the ``max``-of-sides IoU cannot express: one side confirmed,
        # one side not. Under the two rules that take the max these all read as "IoU passes";
        # this is the set to look at before choosing a two-sided tolerance, because it holds
        # both the nudged boxes and the wandered ones.
        return [s for s in subjects
                if _live(s).get("iou_ok") and not _live(s).get("iou_both_ok")]
    if mode == SHOW_FLOOR_PEAK_ONLY:
        # The subjects the floor/peak split admits and the strict two-sided rule refuses -- i.e.
        # exactly what the looser floor buys. Judged on the IoU gates alone, not on the full
        # verdict: a subject failing identity would otherwise vanish from this list even though
        # it is one of the boxes the comparison is about.
        return [s for s in subjects
                if _live(s).get("iou_floor_peak_ok") and not _live(s).get("iou_both_ok")]
    if mode == SHOW_BOX_REPLACED:
        return [s for s in subjects if replaced_sides(s)]
    if mode == SHOW_BOX_KEPT:
        return [s for s in subjects if not replaced_sides(s)]
    return list(subjects)


def weaker_iou(subject: dict[str, Any]) -> float | None:
    """``min`` of the two per-side IoUs -- the number the two-sided rule thresholds against.

    ``None`` when either side was never measured, matching :func:`redetect.decide`'s treatment
    of a missing side as a failure rather than a skip. Returning 0.0 instead would read as
    "measured, and it disagreed", which is a different fact.
    """
    values = [subject.get("iou_dis_vs_phantom"), subject.get("iou_seed_dis_vs_phantom")]
    if any(value is None for value in values):
        return None
    return min(float(value) for value in values)


def sort_subjects(subjects: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Ascending on the chosen metric, with unmeasured subjects last.

    Unmeasured sorts last rather than first: a missing value means the number does not exist
    for that subject, and sorting it as 0.0 would bury the subjects the metric is being
    inspected for.
    """
    # identity and clip come from the live verdict (they are the max-over-sides values the
    # rule actually compares); the two IoUs are per-side and read straight off the record.
    live_keys = {SORT_IDENTITY: "identity", SORT_CLIP: "clip"}
    record_keys = {SORT_IOU_REF: "iou_dis_vs_phantom",
                   SORT_IOU_SEED: "iou_seed_dis_vs_phantom"}
    if mode == SORT_IOU_WEAK:
        value = weaker_iou
    elif mode in live_keys:
        def value(subject):
            return _live(subject).get(live_keys[mode])
    elif mode in record_keys:
        def value(subject):
            return subject.get(record_keys[mode])
    else:
        return sorted(subjects, key=lambda s: (s.get("sample_id", ""), s.get("subject_id", 0)))
    return sorted(subjects, key=lambda s: (value(s) is None,
                                           value(s) if value(s) is not None else 0.0))


def counts(subjects: list[dict[str, Any]]) -> dict[str, int]:
    """The headline numbers under the live verdict: kept, what each gate cost, replacements."""
    kept = [s for s in subjects if _live(s).get("verdict") == KEEP]
    dropped = [s for s in subjects if _live(s).get("verdict") == DROP]
    return {
        "subjects": len(subjects),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_on_identity": sum(1 for s in dropped if not _live(s).get("identity_ok")),
        "dropped_unconfirmed": sum(1 for s in dropped if _live(s).get("identity_ok")),
        "kept_by_clip_only": sum(1 for s in kept if _live(s).get("clip_ok")
                                 and not _live(s).get("iou_ok")),
        "kept_by_iou_only": sum(1 for s in kept if _live(s).get("iou_ok")
                                and not _live(s).get("clip_ok")),
        "kept_by_both": sum(1 for s in kept if _live(s).get("clip_ok")
                            and _live(s).get("iou_ok")),
        # Only the IoU-stands rule can produce these: kept despite identity failing. This is
        # the number the rule choice turns on, so it gets its own line rather than being
        # folded into the keep total.
        "rescued_by_iou": sum(1 for s in kept if _live(s).get("rescued_by_iou")),
        "identity_failed": sum(1 for s in subjects if not _live(s).get("identity_ok")),
        "clip_failed": sum(1 for s in subjects if not _live(s).get("clip_ok")),
        # The two-sided IoU gate on its own, and the disagreement the ``max`` hides. Counted for
        # every rule, not just the two-sided one, so switching rules shows what it would cost
        # before it is switched to.
        "iou_both_ok": sum(1 for s in subjects if _live(s).get("iou_both_ok")),
        "iou_one_sided": sum(1 for s in subjects if _live(s).get("iou_ok")
                             and not _live(s).get("iou_both_ok")),
        # The floor/peak gates, and the gap between the two two-sided rules. Counted on the IoU
        # gates alone (identity and clip excluded) so this reads as a property of the boxes.
        "iou_floor_peak_ok": sum(1 for s in subjects if _live(s).get("iou_floor_peak_ok")),
        "iou_floor_failed": sum(1 for s in subjects if _live(s).get("iou_floor_ok") is False),
        "iou_peak_failed": sum(1 for s in subjects if _live(s).get("iou_floor_ok")
                               and not _live(s).get("iou_peak_ok")),
        "floor_peak_gain": sum(1 for s in subjects if _live(s).get("iou_floor_peak_ok")
                               and not _live(s).get("iou_both_ok")),
        "ref_replaced": sum(1 for s in subjects if "ref" in replaced_sides(s)),
        "seed_replaced": sum(1 for s in subjects if "seed" in replaced_sides(s)),
        "both_replaced": sum(1 for s in subjects if len(replaced_sides(s)) == 2),
    }


def sweep_iou(subjects: list[dict[str, Any]], rule: str, identity_min: float, clip_min: float,
              iou_floor: float = redetect.IOU_FLOOR_MIN,
              steps: tuple[float, ...] = SWEEP_STEPS) -> list[dict[str, Any]]:
    """Keep count at every IoU tolerance in ``steps``, holding the other two thresholds fixed.

    One slider position answers one point on this curve. Choosing a tolerance needs the whole
    curve -- specifically where it stops falling gently and drops off a cliff, which is the
    boundary between "the box is nudged" and "the detector found a different object".

    ``subjects`` must be the *raw* records, not annotated ones: this re-judges each subject at
    each step, and a stale ``_live`` from a previous threshold would be ignored anyway but
    invites the reader to assume the numbers relate.
    """
    rows = []
    total = max(1, len(subjects))
    for iou_min in steps:
        # The floor is clamped to the peak rather than left above it: a floor over the peak makes
        # the peak vacuous, so the low end of the sweep would silently be measuring
        # "both sides >= floor" and the curve would flatten for a reason the reader cannot see.
        floor = min(iou_floor, iou_min)
        kept = sum(1 for subject in subjects
                   if redetect.decide(subject, rule, identity_min, clip_min, iou_min,
                                      floor).get("verdict") == KEEP)
        label = f"{iou_min:.2f}"
        if rule == redetect.RULE_IOU_FLOOR_PEAK:
            label = f"{floor:.2f} / {iou_min:.2f}"
        rows.append({("floor / peak" if rule == redetect.RULE_IOU_FLOOR_PEAK else "IoU >="): label,
                     "kept": kept, "%": f"{100 * kept / total:.0f}%",
                     "dropped": len(subjects) - kept})
    return rows


def sweep_floor(subjects: list[dict[str, Any]], identity_min: float, clip_min: float,
                iou_min: float, steps: tuple[float, ...] = SWEEP_STEPS) -> list[dict[str, Any]]:
    """Keep count as the *floor* moves, with the peak held at ``iou_min``.

    The other sweep moves the peak; this one moves the bar both sides must clear, which is the
    knob that decides how much offset is tolerated. Steps above ``iou_min`` are skipped rather
    than clamped: there the floor would imply the peak and the rule degenerates into
    ``iou_both_sides``, so reporting those rows under a floor label would misattribute the count.
    """
    rows = []
    total = max(1, len(subjects))
    for floor in steps:
        if floor > iou_min:
            continue
        kept = sum(1 for subject in subjects
                   if redetect.decide(subject, redetect.RULE_IOU_FLOOR_PEAK, identity_min,
                                      clip_min, iou_min, floor).get("verdict") == KEEP)
        rows.append({"floor >=": f"{floor:.2f}", "kept": kept,
                     "%": f"{100 * kept / total:.0f}%"})
    return rows


def compare_rules(subjects: list[dict[str, Any]], identity_min: float, clip_min: float,
                  iou_min: float, iou_floor: float = redetect.IOU_FLOOR_MIN
                  ) -> list[dict[str, Any]]:
    """All three rules' keep counts at the current thresholds, for one glance.

    Shown because the rules are not nested: ``iou_both_sides`` can drop a subject that
    ``identity_required`` keeps and vice versa, so a single keep number gives no sense of what
    switching would cost.
    """
    rows = []
    total = max(1, len(subjects))
    for rule in RULE_LABELS.values():
        kept = sum(1 for subject in subjects
                   if redetect.decide(subject, rule, identity_min, clip_min, iou_min,
                                      iou_floor).get("verdict") == KEEP)
        rows.append({"rule": rule, "kept": kept, "%": f"{100 * kept / total:.0f}%"})
    return rows


def gate_breakdown(subjects: list[dict[str, Any]], identity_min: float, clip_min: float,
                   iou_min: float, iou_floor: float) -> list[dict[str, Any]]:
    """Each gate's pass count *in isolation*, so a low keep total can be attributed.

    The gates are near-independent on this dataset (measured Pearson r = -0.012 between clip and
    dino_cos), so the keep total is close to their product and reads as much lower than any one
    of them. Without this table the natural conclusion is "the IoU rule is too strict" when the
    binding constraint is usually identity.
    """
    total = max(1, len(subjects))
    rulings = [redetect.decide(subject, redetect.RULE_IOU_FLOOR_PEAK, identity_min, clip_min,
                               iou_min, iou_floor) for subject in subjects]

    def row(name: str, predicate) -> dict[str, Any]:
        passing = sum(1 for ruling in rulings if predicate(ruling))
        return {"gate": name, "passes": passing, "%": f"{100 * passing / total:.0f}%"}

    return [
        row(f"identity >= {identity_min:.2f}", lambda r: r.get("identity_ok")),
        row(f"clip >= {clip_min:.3f}", lambda r: r.get("clip_ok")),
        row(f"both sides >= {iou_floor:.2f} (floor)", lambda r: r.get("iou_floor_ok")),
        row(f"one side >= {iou_min:.2f} (peak)", lambda r: r.get("iou_peak_ok")),
        row("floor AND peak", lambda r: r.get("iou_floor_peak_ok")),
        row(f"both sides >= {iou_min:.2f} (strict)", lambda r: r.get("iou_both_ok")),
    ]


def floor_peak_delta(subjects: list[dict[str, Any]], identity_min: float, clip_min: float,
                     iou_min: float, iou_floor: float) -> dict[str, Any]:
    """What the floor/peak split changes relative to the strict two-sided rule.

    Reported as three disjoint sets rather than two keep counts, because a difference of "+17"
    could mean 17 gained or 25 gained and 8 lost, and those call for different decisions. Each
    entry lists the subjects so the claim can be checked by looking at them.
    """
    gained, lost, wandered = [], [], []
    for subject in subjects:
        strict = redetect.decide(subject, redetect.RULE_IOU_BOTH_SIDES, identity_min, clip_min,
                                 iou_min)
        split = redetect.decide(subject, redetect.RULE_IOU_FLOOR_PEAK, identity_min, clip_min,
                                iou_min, iou_floor)
        entry = {
            "sample_id": subject.get("sample_id"),
            "subj": int(subject.get("subject_id") or 0),
            "phrase": str(subject.get("dis") or subject.get("phrase") or "")[:36],
            "ref IoU": _fmt(split.get("iou_ref")),
            "target IoU": _fmt(split.get("iou_seed")),
            "identity": _fmt(split.get("identity")),
            "clip": _fmt(split.get("clip")),
        }
        if split["verdict"] == KEEP and strict["verdict"] == DROP:
            gained.append(entry)
        elif split["verdict"] == DROP and strict["verdict"] == KEEP:
            # Should be empty by construction (the split's IoU gate is strictly weaker at
            # floor <= peak, and the other two gates are identical), so a non-empty list here is
            # a bug worth surfacing rather than a finding.
            lost.append(entry)
        # Independent of the verdicts: the boxes the floor rejects. Listed because it is the
        # claim the floor is there to make good on, and it should hold at any floor in the band.
        if split.get("iou_floor_ok") is False:
            wandered.append(entry)
    return {"gained": gained, "lost": lost, "floor_rejected": wandered}


def weak_side_distribution(subjects: list[dict[str, Any]],
                           steps: tuple[float, ...] = SWEEP_STEPS) -> list[dict[str, Any]]:
    """How many subjects have their *weaker* side at or above each threshold.

    This is the IoU gate of the two-sided rule in isolation -- identity and clip removed -- so a
    keep count that looks low can be attributed to the right gate. Without it, the two-sided
    rule's drops are indistinguishable from identity's, and the two are nearly independent on
    this dataset (measured Pearson r = -0.012 between clip and dino_cos).
    """
    values = [weaker_iou(subject) for subject in subjects]
    measured = [value for value in values if value is not None]
    total = max(1, len(subjects))
    rows = []
    for threshold in steps:
        passing = sum(1 for value in measured if value >= threshold)
        rows.append({"weaker side >=": f"{threshold:.2f}", "subjects": passing,
                     "%": f"{100 * passing / total:.0f}%"})
    return rows


def label_for(subject: dict[str, Any]) -> str:
    """One selectbox line: live verdict, the three rule numbers, and the phrase."""
    live = _live(subject)

    def num(key: str, digits: int = 3) -> str:
        value = live.get(key)
        return "-" if value is None else format(value, f".{digits}f")

    flag = "KEEP" if live.get("verdict") == KEEP else "drop"
    # Marked because these are the subjects the rule choice is about -- kept on IoU alone,
    # with identity having said no.
    if live.get("rescued_by_iou"):
        flag = "KEEP*"
    swapped = replaced_sides(subject)
    tag = f"  [new: {'+'.join(swapped)}]" if swapped else ""
    # Both IoUs, not their max: the max hides exactly the case the two-sided rule exists for,
    # and the selectbox is where a reader scans for it.
    return (f"{flag}  id {num('identity')}  clip {num('clip')}  "
            f"iou {num('iou_ref')}/{num('iou_seed')}{tag}  "
            f"{str(subject.get('sample_id', ''))[:10]} "
            f"'{str(subject.get('dis') or subject.get('phrase'))[:30]}'")


# --------------------------------------------------------------------------------------
# drawing (the part that used to be baked into the PNGs)
# --------------------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else format(float(value), f".{digits}f")


def _scaled(image, width: int):
    scale = width / image.width
    return image.resize((width, max(1, int(round(image.height * scale)))))


def draw_boxes(image, boxes: list[tuple[Any, tuple[int, int, int]]], width: int = FRAME_WIDTH):
    """Overlay boxes on a copy of ``image``, then scale for display.

    Drawn before scaling so the coordinates are used as reported -- scaling first would make
    every box need the same transform applied by hand. Line width scales with the frame so a
    box stays visible on a 1920-wide source and does not swamp a small one.
    """
    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    thickness = max(3, canvas.width // 320)
    for box, colour in boxes:
        if box:
            draw.rectangle(tuple(round(float(v)) for v in box), outline=colour,
                           width=thickness)
    return _scaled(canvas, width)


def crop_to(image, box: Any, width: int = CROP_WIDTH):
    """What one box contains, scaled up. None (or a degenerate box) yields None."""
    clamped = clamp_box(box, image.width, image.height)
    if clamped is None:
        return None
    return _scaled(image.crop(tuple(clamped)), width)


def detail_rows(subject: dict[str, Any]) -> list[dict[str, Any]]:
    """The field/value table under the frames.

    The three gate rows come from the *live* verdict, and the rows that follow are the
    render-time facts (box choices, per-side scores, IoUs) which no slider can change. The
    report's own stored verdict is deliberately not shown: it was computed under whatever
    thresholds the run used, and displaying it beside a live one that disagrees is worse than
    not showing it.

    Values are stringified because the column is mixed -- one bool among the strings makes
    Streamlit's Arrow conversion fall back and log a warning on every page view.
    """
    live = _live(subject)
    mark = lambda ok: "PASS" if ok else "fail"  # noqa: E731
    rows: dict[str, Any] = {}
    if live:
        rows = {
            "verdict": str(live.get("verdict", "")).upper(),
            "why": live.get("reason"),
            "identity  dino_cos": f"{_fmt(live.get('identity'), 4)}  "
                                  f"{mark(live.get('identity_ok'))}",
            "semantic  crop clip": f"{_fmt(live.get('clip'))}  {mark(live.get('clip_ok'))}",
            "agreement IoU (better side)": f"{_fmt(live.get('iou'), 4)}  "
                                           f"{mark(live.get('iou_ok'))}",
            # Spelled out per side as well: the row above is a ``max``, so on its own it cannot
            # distinguish "both boxes agree" from "one agrees and one is on another object".
            "agreement IoU (ref / target)": f"{_fmt(live.get('iou_ref'), 4)} / "
                                            f"{_fmt(live.get('iou_seed'), 4)}  "
                                            f"{mark(live.get('iou_both_ok'))} both",
            # The floor and peak as separate rows: they answer different questions ("did the
            # detector stay on the object" vs "is the normalisation anchored"), and one combined
            # PASS/fail would not say which of the two a drop came from.
            "  floor  (both sides)": mark(live.get("iou_floor_ok")),
            "  peak   (one side)": mark(live.get("iou_peak_ok")),
        }
    for key, value in (subject.get("extra") or {}).items():
        # Skip the stored verdict fields; the live ones above supersede them.
        if key in ("verdict", "why") or key.startswith(("identity ", "crop clip ",
                                                        "IoU vs phantom ")):
            continue
        rows[key] = value
    return [{"field": key, "value": "-" if value is None else str(value)}
            for key, value in rows.items()]


#: Which drawn box the pipeline actually ships, per side, in the reviewer's colour vocabulary.
#: The label is a verdict *on a specific box*, so this cannot be hidden by blind mode -- see
#: :func:`render_subject`. Under ``trust_detector`` the answer is the detector's box on both
#: sides for all 140 pilot subjects; under the historical rule it varies per side, which is
#: exactly why this is derived per subject rather than stated once in the caption.
BOX_COLOUR_NAMES = {PHANTOM: "red (phantom)", NEW: "blue (grounding dino)"}


def judged_boxes(subject: dict[str, Any]) -> list[str]:
    """Human-readable description of which box each side ships, e.g. ``["REFERENCE: blue …"]``.

    Returns one entry per side so a subject whose two sides picked different boxes reads
    correctly; sides with no box at all are reported as such rather than omitted, because a
    missing target box is a thing the reviewer should fail rather than silently not see.
    """
    out: list[str] = []
    for side, title in (("ref", "REFERENCE"), ("seed", "TARGET")):
        pick = subject.get(f"pick_{side}")
        if pick in BOX_COLOUR_NAMES:
            out.append(f"{title}: {BOX_COLOUR_NAMES[pick]}")
        else:
            out.append(f"{title}: no box ({pick})")
    return out


def label_widget(st, label_dir: Path, subject: dict[str, Any], blind: bool) -> None:
    """The pass/fail control. Writes one file per subject; see :mod:`phantom_data.labels`.

    Placed at the top of the subject panel and rendered *before* any verdict or score, so that
    in blind mode the reviewer's first read of the page is the imagery. Ordering is the whole
    mechanism here: a KEEP banner above this widget would anchor the label to the machine, and
    grading the machine is what these labels are for.
    """
    sample_id, subject_id = str(subject["sample_id"]), int(subject["subject_id"])
    existing = labelling.read_label(label_dir, sample_id, subject_id)

    columns = st.columns([1, 1, 1, 3])
    for column, verdict in zip(columns, labelling.VERDICTS):
        chosen = existing is not None and existing["verdict"] == verdict
        if column.button(("● " if chosen else "") + labelling.VERDICT_LABELS[verdict],
                         key=f"label_{verdict}_{sample_id}_{subject_id}",
                         use_container_width=True):
            labelling.write_label(label_dir, labelling.make_record(
                sample_id, subject_id, verdict, blind=blind,
                scores=labelling.scores_of(subject)))
            # Advance on label: the reviewer's next action is almost always the next subject,
            # and 200 labels means 200 saved clicks.
            st.session_state["_gate_page"] = int(st.session_state.get("_gate_page", 0)) + 1
            st.experimental_rerun()
    if existing is not None:
        if columns[2].button("clear", key=f"label_clear_{sample_id}_{subject_id}",
                             use_container_width=True):
            labelling.delete_label(label_dir, sample_id, subject_id)
            st.experimental_rerun()
        seen = "blind" if existing.get("blind") else "saw the verdict"
        columns[3].caption(f"labelled **{labelling.VERDICT_LABELS[existing['verdict']]}** "
                           f"({seen}) → `{labelling.label_name(sample_id, subject_id)}`")
    else:
        columns[3].caption("unlabelled" + ("  ·  blind: the verdict and scores are hidden below"
                                           if blind else ""))


def render_subject(st, dataset: Path, out_root: str, subject: dict[str, Any],
                   show_phantom: bool = True, show_new: bool = True,
                   iou_min: float | None = None, iou_floor: float | None = None,
                   floor_peak: bool = False, label_dir: Path | None = None,
                   blind: bool = False) -> None:
    from PIL import Image

    live = _live(subject)
    kept = live.get("verdict") == KEEP
    st.markdown(f"### {subject['sample_id']}  ·  subj{int(subject['subject_id']):02d}")
    if label_dir is not None:
        label_widget(st, label_dir, subject, blind)
    # Under blind mode every machine-derived judgement is withheld: the verdict banner, the
    # three scores, the per-side IoU marks, the clip figure on each crop, and the detail table.
    # Withholding only the banner would not help -- the scores alone are enough to reconstruct
    # the verdict, and a reviewer who can see identity 0.31 has been told the answer.
    if not blind:
        (st.success if kept else st.error)(
            f"**{'KEEP' if kept else 'DROP'}** — {live.get('reason')}")
    if live.get("rescued_by_iou") and not blind:
        st.warning(
            "Kept on IoU alone, with identity below its threshold. IoU is measured inside each "
            "frame against Phantom's box for that frame, so it does not speak to whether these "
            "two frames — median 83 seconds apart — show the same individual. The crops below "
            "are what settles it."
        )
    if blind:
        # The phrase stays: it is the annotation being checked, not a machine score. Judging
        # "is this box on the right object" requires knowing which object was asked for.
        #
        # And so does *which box is the product*. Hiding that was an error: the reviewer was
        # left guessing whether the verdict applies to the red box or the blue one, and a label
        # against the wrong box is worse than no label. What blind mode must withhold is the
        # machine's *judgement* (verdict, scores) -- not the pipeline's factual output.
        st.markdown(
            f"**phrase:** `{subject.get('dis') or subject.get('phrase')}`  ·  "
            f"judge the **{'; '.join(judged_boxes(subject)) or 'boxes'}**")
    else:
        st.markdown(
            f"**phrase:** `{subject.get('dis') or subject.get('phrase')}`  ·  "
            f"identity `{_fmt(live.get('identity'), 4)}`  ·  "
            f"clip `{_fmt(live.get('clip'))}`  ·  "
            f"IoU ref `{_fmt(live.get('iou_ref'), 4)}` / target "
            f"`{_fmt(live.get('iou_seed'), 4)}`")
    if live.get("iou_ok") and not live.get("iou_both_ok") and not blind:
        # Split by which side of the floor the weak side falls on, because that is exactly the
        # distinction the floor/peak rule draws and the one the reader is being asked to check.
        if live.get("iou_floor_peak_ok"):
            st.info(
                "One side reaches the peak and the other only clears the floor — under "
                "`iou_floor_peak` this counts as an **offset** box, not a wrong one, and is "
                "kept. `iou_both_sides` would drop it. The crops below are what settles which "
                "reading is right."
            )
        else:
            st.warning(
                "One side's box agrees with the annotation and the other's does not, with the "
                "weaker side **below the floor** — the detector probably left the annotated "
                "object on that side. Both two-sided rules drop this; the rules that take the "
                "max of the two IoUs would keep it."
            )

    root = dataset / out_root
    for side, title in (("ref", "REFERENCE"), ("seed", "TARGET")):
        frame_path = root / str(subject.get(f"{side}_frame") or "")
        if not frame_path.is_file():
            st.warning(f"missing frame {frame_path}")
            continue
        image = Image.open(frame_path).convert("RGB")
        phantom_box = subject.get(f"box_{side}_{PHANTOM}")
        new_box = subject.get(f"box_{side}_{NEW}")
        pick = subject.get(f"pick_{side}")
        iou_value = subject.get("iou_dis_vs_phantom" if side == "ref"
                                else "iou_seed_dis_vs_phantom")

        overlay: list[tuple[Any, tuple[int, int, int]]] = []
        if show_phantom:
            overlay.append((phantom_box, PHANTOM_COLOUR))
        if show_new:
            overlay.append((new_box, NEW_COLOUR))

        # Per-side pass/fail against the live tolerance, because this side's IoU is what the
        # two-sided rules read -- the header number alone leaves the reader comparing by eye.
        # Under floor/peak the bar *this side* must clear is the floor, so marking it against the
        # peak would flag a side the rule is happy with.
        bar = iou_floor if (floor_peak and iou_floor is not None) else iou_min
        side_ok = ""
        if iou_value is not None and bar is not None and not blind:
            side_ok = ("  ·  **PASS**" if iou_value >= bar
                       else f"  ·  **below the {'floor' if floor_peak else 'tolerance'} "
                            f"{bar:.2f}**")
            if floor_peak and iou_value >= bar and iou_min is not None and iou_value >= iou_min:
                side_ok += " (and reaches the peak)"
        if blind:
            # Which box ships is stated per side, not only once at the top: the reviewer is
            # scrolling between two frames and four crops, and the sentence above scrolls away.
            # The *reason* it was picked stays hidden -- that is the machine's argument.
            st.markdown(f"**{title}** — judge the "
                        f"**{BOX_COLOUR_NAMES.get(pick, f'({pick}: no box)')}** box")
        else:
            st.markdown(
                f"**{title}** — using the **{pick}** box "
                f"({subject.get(f'pick_{side}_reason')})  ·  "
                f"IoU between them: `{'-' if iou_value is None else round(iou_value, 4)}`"
                f"{side_ok}")
        columns = st.columns([3, 1, 1])
        columns[0].image(draw_boxes(image, overlay), use_column_width=True,
                         caption="red = phantom    blue = grounding dino")
        for column, (name, box, colour) in zip(
                columns[1:], ((PHANTOM, phantom_box, PHANTOM_COLOUR),
                              (NEW, new_box, NEW_COLOUR))):
            crop = crop_to(image, box)
            clip = subject.get(f"crop_clip_{side}_{name}")
            # ``← used`` is withheld under blind too: which box the pipeline picked is itself a
            # machine decision, and it points at the crop the reviewer is meant to judge cold.
            # ``← judge this`` survives blind mode; the clip score does not. Which crop is the
            # product is a fact the reviewer needs; what CLIP thought of it is the opinion under
            # test. (Earlier this marker was hidden too, which left the reviewer unable to tell
            # whether their verdict applied to the red box or the blue one.)
            mark = " ← judge this" if name == pick else ""
            caption = (f"{name}{mark}" if blind else
                       f"{name}{mark}  ·  clip "
                       f"{'-' if clip is None else format(clip, '.3f')}")
            if crop is None:
                column.info(f"{name}: no box")
            else:
                column.image(crop, use_column_width=True, caption=caption)

    if blind:
        st.caption(
            "Blind labelling: the verdict and every score are hidden, **but which box ships is "
            "not** — it is marked `← judge this` on the crop and named per side above. Judge "
            "only that box: is it on the object the phrase names and tight to it, and is the "
            "reference the **same individual** as the target? Either failure is 不合格. Red = "
            "phantom's annotation, blue = grounding dino; the other box is drawn for contrast "
            "so the offset is visible, and is not what you are grading."
        )
        return
    st.caption(
        "Boxes are drawn by this page from the coordinates in the report — the stored frames "
        "are unannotated. The crops show what each box actually contains, which is what "
        "settles whether a box is on the right object; the clip scores are on the plain crop "
        "(background included, no SAM2 matte), which is why the threshold is 0.21 rather than "
        "the 0.23 calibrated against cutouts."
    )
    st.table(detail_rows(subject))


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="keep / drop", layout="wide")
    st.title("What the filter kept, and what it dropped")

    dataset = Path(st.sidebar.text_input(
        "dataset root", value=os.getenv("PHANTOM_GATE_DATASET", DEFAULT_DATASET)))
    out_root = st.sidebar.text_input(
        "render dir", value=os.getenv("PHANTOM_GATE_OUT_ROOT", DEFAULT_OUT_ROOT))

    report = load_report(dataset, out_root)
    subjects = report.get("subjects") or []
    if not subjects:
        st.warning(f"no gate_report.json under {dataset / out_root}. Render first:\n\n"
                   f"`python tools/redetect_run.py --dataset {dataset}`")
        return

    stored = report.get("rule") or {}

    def threshold(name: str, fallback: float) -> float:
        """A stored threshold, or ``fallback`` when the report declines to assert one.

        ``dict.get(name, fallback)`` is wrong here: the text-free chain writes
        ``{"identity_min": null}`` on purpose -- the gate for that chain is undecided pending
        human labels, and emitting a number would imply one had been chosen. The key is
        present, so ``get`` returns None rather than the default and ``float(None)`` raises.
        """
        value = stored.get(name)
        return fallback if value is None else float(value)

    rule_label = st.sidebar.radio("rule", list(RULE_LABELS),
                                 index=list(RULE_LABELS.values()).index(redetect.DEFAULT_RULE))
    rule = RULE_LABELS[rule_label]
    identity_min = st.sidebar.slider("identity: dino_cos >=", 0.0, 1.0,
                                     threshold("identity_min", 0.6), 0.01)
    clip_min = st.sidebar.slider("semantic: crop clip >=", 0.10, 0.40,
                                 threshold("clip_min", 0.21), 0.005)
    iou_min = st.sidebar.slider(
        "bbox tolerance: IoU vs phantom >= (peak)", 0.0, 1.0,
        threshold("iou_min", 0.75), 0.01,
        help="Under iou_both_sides this is required of the reference AND the target separately; "
             "under iou_floor_peak it is the higher bar one side must reach; under the other two "
             "only the better side has to clear it.")
    iou_floor = st.sidebar.slider(
        "bbox tolerance: floor both sides must clear", 0.0, 1.0,
        threshold("iou_floor", redetect.IOU_FLOOR_MIN), 0.01,
        help="Only read by iou_floor_peak. This is the knob that decides how much offset is "
             "tolerated: a middling IoU means the box is shifted, a near-zero one means the "
             "detector left the object.")
    if iou_floor > iou_min:
        # Clamped rather than merely warned about, because above the peak the floor makes the
        # peak vacuous and the rule silently becomes iou_both_sides -- while the caption would
        # still describe a two-stage gate.
        st.sidebar.warning(
            f"floor {iou_floor:.2f} is above the peak {iou_min:.2f}; clamped to the peak, where "
            f"the rule is identical to `iou_both_sides`."
        )
        iou_floor = iou_min

    # Kept unannotated for the sweep: those helpers re-judge at each step, and handing them
    # records carrying a ``_live`` from one particular threshold invites reading the two as
    # related when they are not.
    raw = list(subjects)
    subjects = annotate(subjects, rule, identity_min, clip_min, iou_min, iou_floor)
    tally = counts(subjects)
    total = max(1, tally["subjects"])
    floor_peak = rule == redetect.RULE_IOU_FLOOR_PEAK

    formula = RULE_FORMULAS[rule]
    thresholds = (f"identity >= {identity_min:.2f}, clip >= {clip_min:.3f}, "
                  f"floor >= {iou_floor:.2f}, peak >= {iou_min:.2f}" if floor_peak
                  else f"identity >= {identity_min:.2f}, clip >= {clip_min:.3f}, "
                       f"IoU >= {iou_min:.2f}")
    st.caption(
        f"**{tally['kept']} of {tally['subjects']} kept** "
        f"({100 * tally['kept'] / total:.0f}%) — `{formula}`, with {thresholds}. "
        f"Verdicts are computed live from the stored scores; move a slider and these numbers "
        f"move with it."
    )

    with st.expander(
            f"bbox tolerance sweep — keep count at every IoU threshold "
            f"(rule `{rule}`, identity >= {identity_min:.2f}, clip >= {clip_min:.3f})",
            expanded=True):
        columns = st.columns(3)
        columns[0].markdown("**keep count vs the peak**" if floor_peak
                            else "**keep count vs IoU tolerance**")
        columns[0].caption(
            f"The whole curve for the rule currently selected. Everything except the "
            f"{'peak' if floor_peak else 'IoU threshold'} is held at the sidebar's values"
            + (f", with the floor at {iou_floor:.2f} (clamped down where it would exceed the "
               f"peak)." if floor_peak else ", so this isolates the tolerance.")
        )
        columns[0].table(sweep_iou(raw, rule, identity_min, clip_min, iou_floor))

        if floor_peak:
            columns[1].markdown("**keep count vs the floor**")
            columns[1].caption(
                f"The knob that decides how much offset is tolerated, with the peak held at "
                f"{iou_min:.2f}. Rows above the peak are omitted: there the floor implies the "
                f"peak and the rule degenerates into `iou_both_sides`."
            )
            columns[1].table(sweep_floor(raw, identity_min, clip_min, iou_min))
        else:
            columns[1].markdown("**the IoU gate alone**")
            columns[1].caption(
                "How many subjects have their *weaker* side at or above each threshold, with "
                "identity and clip removed. Compare against the left column to see which gate "
                "is actually costing the keeps — the two are near-independent here."
            )
            columns[1].table(weak_side_distribution(raw))

        columns[2].markdown("**all four rules, at these thresholds**")
        columns[2].caption(
            "The rules are not nested: each can drop a subject another keeps, so switching is "
            "a trade rather than a tightening."
        )
        columns[2].table(compare_rules(raw, identity_min, clip_min, iou_min, iou_floor))

        st.markdown("**each gate on its own** — which one is actually costing the keeps")
        st.caption(
            "The gates are near-independent on this dataset, so the keep total is close to their "
            "product and reads far lower than any single gate. Reading these in isolation is how "
            "you tell 'the IoU rule is too strict' from 'identity is the binding constraint'."
        )
        st.table(gate_breakdown(raw, identity_min, clip_min, iou_min, iou_floor))

    with st.expander(
            f"what floor/peak changes vs both-sides-{iou_min:.2f} — the subjects themselves",
            expanded=floor_peak):
        delta = floor_peak_delta(raw, identity_min, clip_min, iou_min, iou_floor)
        st.markdown(
            f"**{len(delta['gained'])} subjects kept by `iou_floor_peak` "
            f"(floor {iou_floor:.2f} / peak {iou_min:.2f}) that `iou_both_sides` "
            f"({iou_min:.2f} on both) drops.** These are the offset boxes the looser floor buys. "
            f"Every one has one side at or above {iou_min:.2f}, so the annotation is anchored on "
            f"that side; the other side is between {iou_floor:.2f} and {iou_min:.2f}."
        )
        st.table(delta["gained"] or [{"(none)": "no subject differs at these thresholds"}])
        if delta["lost"]:
            # Empty by construction while floor <= peak. If it ever fills, the rules are not
            # ordered the way the comparison above assumes and the table is the evidence.
            st.error(
                f"{len(delta['lost'])} subjects go the other way — kept by `iou_both_sides` and "
                f"dropped by `iou_floor_peak`. That should be impossible while the floor is at "
                f"or below the peak; treat it as a bug, not a finding."
            )
            st.table(delta["lost"])
        st.markdown(
            f"**{len(delta['floor_rejected'])} subjects the floor rejects** — at least one side "
            f"below {iou_floor:.2f}, i.e. the detector left the annotated object on that side. "
            f"This is the set the floor exists to catch, and the claim worth spot-checking in "
            f"the crops."
        )
        st.table(delta["floor_rejected"]
                 or [{"(none)": "no subject has a side below the floor"}])

    st.caption(
        f"Independently at the current settings: identity passes for "
        f"{tally['subjects'] - tally['identity_failed']}, clip for "
        f"{tally['subjects'] - tally['clip_failed']}, both-side IoU for "
        f"{tally['iou_both_ok']}, floor+peak for {tally['iou_floor_peak_ok']}. "
        f"**{tally['iou_one_sided']} subjects have one side passing and the other failing** — "
        f"invisible to the two rules that take the max of the two sides. The "
        f"`{SHOW_ONE_SIDED}` and `{SHOW_FLOOR_PEAK_ONLY}` filters below show them."
    )
    lines = [
        f"**{tally['subjects']} subjects**", "",
        f"- kept: **{tally['kept']}** ({100 * tally['kept'] / total:.0f}%)",
        f"- dropped: **{tally['dropped']}**",
        f"  - identity failed: {tally['dropped_on_identity']}",
        f"  - box unconfirmed: {tally['dropped_unconfirmed']}",
        "",
        "of those kept:",
        f"- clip only: {tally['kept_by_clip_only']}",
        f"- IoU only: {tally['kept_by_iou_only']}",
        f"- both: {tally['kept_by_both']}",
    ]
    if rule == redetect.RULE_IOU_STANDS:
        # The comparison the rule choice rests on, stated as a number rather than left to be
        # inferred from the keep total moving.
        lines += [
            "",
            f"**kept on IoU despite identity failing: {tally['rescued_by_iou']}**",
            f"(identity fails for {tally['identity_failed']} subjects in total)",
        ]
    lines += [
        "",
        f"bbox tolerance (floor {iou_floor:.2f} / peak {iou_min:.2f}):",
        f"- both sides >= peak: {tally['iou_both_ok']}",
        f"- floor+peak: {tally['iou_floor_peak_ok']}",
        f"  - of which offset-only: {tally['floor_peak_gain']}",
        f"- **one side only: {tally['iou_one_sided']}**",
        f"- below the floor: {tally['iou_floor_failed']}",
        f"- floor ok, no side at peak: {tally['iou_peak_failed']}",
    ]
    lines += [
        "",
        "boxes replaced by the detector:",
        f"- reference: {tally['ref_replaced']}",
        f"- target: {tally['seed_replaced']}",
        f"- both sides: {tally['both_replaced']}",
    ]
    st.sidebar.markdown("\n".join(lines))
    if report.get("failures"):
        st.sidebar.warning(f"{len(report['failures'])} sample(s) failed to render")

    show = st.sidebar.radio("show", list(SHOW_FILTERS))
    order = st.sidebar.radio("order by", list(SORT_ORDERS))
    show_phantom = st.sidebar.checkbox("draw phantom box (red)", value=True)
    show_new = st.sidebar.checkbox("draw grounding dino box (blue)", value=True)

    # ---- labelling -------------------------------------------------------------------
    st.sidebar.markdown("---")
    labelling_on = st.sidebar.checkbox(
        "label pass / fail", value=bool(os.getenv("PHANTOM_LABEL", "")),
        help="Write one 合格/不合格 verdict per subject to <dataset>/_labels/. These are the "
             "only ground truth there is: without them no threshold can be validated and no "
             "identity model can be compared.")
    label_dir = dataset / os.getenv("PHANTOM_LABEL_DIR", labelling.DEFAULT_LABEL_DIR)
    blind = False
    if labelling_on:
        blind = st.sidebar.checkbox(
            "blind (hide verdict and scores)", value=True,
            help="Hides the KEEP/DROP banner, all three scores, the picked-box marker and the "
                 "detail table. The page is otherwise an anchoring machine: a reviewer who "
                 "reads 'identity 0.31' before deciding will regress toward the rule these "
                 "labels exist to grade.")
        existing = labelling.load_labels(label_dir)
        progress = labelling.label_summary(existing, total=len(subjects))
        st.sidebar.markdown(
            f"**{progress['labelled']} / {progress['total']} labelled**  ·  "
            f"{progress['pass']} pass / {progress['fail']} fail"
            + (f"  ·  pass rate {progress['pass_rate']:.0%}" if progress.get('pass_rate')
               else "")
            + f"  \n{progress['blind']} labelled blind  ·  `{label_dir}`")
        if st.sidebar.button("jump to next unlabelled", use_container_width=True):
            # Resolved against the *visible* ordering, not the report order, so the jump lands
            # on a subject the current filter actually shows.
            ordered = sort_subjects(filter_subjects(subjects, show), order)
            target = labelling.next_unlabelled(
                ordered, existing, int(st.session_state.get("_gate_page", 0)) + 1)
            if target is None:
                st.sidebar.success("every visible subject is labelled")
            else:
                st.session_state["_gate_page"] = target
                st.experimental_rerun()

    visible = sort_subjects(filter_subjects(subjects, show), order)
    if not visible:
        st.warning("no subjects match this filter")
        return

    # The rule and thresholds belong in the key: they change which subjects are visible and in
    # what order, so a stale page index would land on a different subject than the one the
    # reader was looking at.
    key = (str(dataset), out_root, show, order, rule,
           round(identity_min, 4), round(clip_min, 4), round(iou_min, 4), round(iou_floor, 4))
    if st.session_state.get("_gate_key") != key:
        st.session_state["_gate_key"] = key
        st.session_state["_gate_page"] = 0
    page = int(st.session_state.get("_gate_page", 0)) % len(visible)

    navigation = st.columns([1, 1, 6])
    if navigation[0].button("prev", use_container_width=True):
        st.session_state["_gate_page"] = page - 1
        st.experimental_rerun()
    if navigation[1].button("next", use_container_width=True):
        st.session_state["_gate_page"] = page + 1
        st.experimental_rerun()
    picked = navigation[2].selectbox(
        f"subject ({page + 1} / {len(visible)})", list(range(len(visible))), index=page,
        format_func=lambda i: label_for(visible[i]))
    if picked != page:
        st.session_state["_gate_page"] = picked
        st.experimental_rerun()

    render_subject(st, dataset, out_root, visible[page], show_phantom, show_new, iou_min,
                   iou_floor, floor_peak,
                   label_dir=label_dir if labelling_on else None, blind=blind)


if __name__ == "__main__":
    render()
