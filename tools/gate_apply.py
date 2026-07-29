"""Stage 3 of the box pipeline: apply the gate and write stage C's input.

This is the step that was missing. ``redetect_run.py`` produced a report and ``gate_viewer``
computed keep/drop live in the browser, but **nothing wrote the corrected boxes back**, so stage
C kept cutting masklets from Phantom's original boxes. On the pilot that meant 93/140 reference
boxes and 107/140 target boxes were known to be wrong and used anyway -- and that box is the
training conditioning signal, not a diagnostic.

    A plan -> B extract -> 1 enrich -> 2 redetect -> 3 gate(apply) -> C segment -> D index

Reads ``gate_report.json`` plus the stage B manifest it was computed from, applies
:func:`redetect.decide`, and writes ``gated.jsonl``: the same row shape stage C's
``parse_sample`` already accepts, with the corrected boxes substituted and the dropped subjects
removed. Stage C then runs on 56% of the pilot's subjects instead of all of them, from boxes
that are right.

**The coordinate-space trap, which is the one thing in this file that can silently corrupt
data.** The report's ``chosen_box_*`` are already **real frame pixel coordinates** -- Grounding
DINO looked at the decoded frame, and Phantom's boxes were put into frame space before being
compared to it. But stage B's ``seed_bbox_768`` are *raw annotation* coordinates, and stage C
maps whatever it reads through the annotation canvas. Mapping an already-mapped box rescales it
by ``max(W, H) / 768`` a second time -- 2.5x on a 1920x1080 clip -- which raises nothing and
just puts every box somewhere else. So every row written here carries ``box_space: "frame"`` and
stage C dispatches on it (see :data:`segment.BOX_SPACE_FRAME`). The tag travels with the data
rather than being a CLI flag, because a flag can be pointed at the wrong file.

Also writes ``gated_drops.json``: every dropped subject with which gate rejected it, so the
funnel is countable without re-deriving it from the report.

Usage: python tools/gate_apply.py --dataset <root> [--rule iou_stands]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phantom_data import redetect
from phantom_data.build import segment
from phantom_data.io import atomic_write_bytes, read_jsonl

DEFAULT_OUT_ROOT = "_redetect100"
DEFAULT_INPUT = "extracted.jsonl"
DEFAULT_OUTPUT = "gated.jsonl"
DEFAULT_DROPS = "gated_drops.json"


def gate_reason(ruling: dict[str, Any]) -> str:
    """Which gate rejected a subject, as one word for counting.

    Ordered by how actionable the answer is rather than by the order the rule evaluates them.
    ``no_box`` first because it is a detector abstention, not a measurement; then ``identity``,
    which is the gate that does nearly all the dropping (62/62 of the pilot's drops) and means
    the reference is probably a different object.

    ``no_box`` first because it is an abstention rather than a measurement: no proposal produced a
    mask on that side, so there was nothing for the identity judge to compare. Everything else the
    single-judge rule can reject for is an identity failure.
    """
    if ruling.get("no_box_sides"):
        return "no_box"
    if not ruling.get("identity_ok"):
        return "identity"
    return "unknown"


def corrected_subject(subject: dict[str, Any], record: dict[str, Any],
                      ruling: dict[str, Any] | None = None) -> dict[str, Any]:
    """One stage B subject with its two boxes replaced by the gate's chosen boxes.

    Pure, and the reason this is a function rather than three lines inside a loop: it is the
    only place where the corrected coordinates are written into the shape stage C parses, so it
    is the thing worth testing without a GPU or a report on disk.

    The boxes keep stage B's field *names* (``seed_bbox_768`` / ``ref.bbox_768``) while holding
    values in a different coordinate space. That reads wrong and is deliberate: those names are
    stage C's input contract (``parse_sample`` requires them), the ``_768`` suffix is already a
    documented misnomer meaning "the box field", and the row's ``box_space`` tag is what states
    the space. Inventing parallel ``*_frame`` fields would mean teaching ``parse_sample`` a
    second schema, and then two code paths could disagree about which one wins.

    Everything else on the subject is carried through untouched -- ``seed_frame_index``,
    ``phrase``, and the ref pointer's ``frame`` are what stage C needs and are not ours to
    change. The gate's own numbers are attached under ``gate`` for provenance, so a built sample
    can be traced to the decision that let it through.

    ``ruling`` is only passed under ``--keep-all``, where a row may carry a subject the gate
    would have dropped; it records the verdict on the subject so a reviewer can tell "the mask
    is bad" apart from "the gate had already rejected this pair".
    """
    ref = dict(subject.get("ref") or {})
    ref["bbox_768"] = list(record["chosen_box_ref"])
    gate = {
        "pick_ref": record.get("pick_ref"),
        "pick_seed": record.get("pick_seed"),
        "identity": record.get("rule_identity"),
        "clip": record.get("rule_clip"),
        "iou": record.get("rule_iou"),
        "dis": record.get("dis"),
    }
    if ruling is not None:
        # Only written under --keep-all, where the row can carry a subject the gate would have
        # dropped. Downstream (stage C copies ``extra`` into the bbox json, the inspector reads
        # it) this is the only place the verdict survives, so a reviewer can tell "the mask is
        # bad" from "the gate had already rejected this pair".
        gate["verdict"] = ruling["verdict"]
        gate["verdict_reason"] = ruling["reason"]
        gate["gate_reason"] = None if ruling["verdict"] == redetect.KEEP else gate_reason(ruling)
    return {
        **subject,
        "seed_bbox_768": list(record["chosen_box_seed"]),
        "ref": ref,
        "gate": gate,
    }


def gated_row(row: dict[str, Any], records: list[dict[str, Any]],
              rule: str = redetect.DEFAULT_RULE,
              identity_min: float = redetect.IDENTITY_MIN,
              clip_min: float | None = None,
              iou_min: float | None = None,
              iou_floor: float | None = None,
              keep_all: bool = False) -> tuple[dict[str, Any] | None,
                                               list[dict[str, Any]]]:
    """Gate one stage B sample. Returns ``(row for stage C or None, dropped subjects)``.

    ``None`` when no subject survived: stage C rejects a subject-less sample anyway
    (``parse_sample`` raises on empty ``subjects``), so emitting one would turn a clean filter
    result into a stage C failure.

    A subject with no report record is dropped with reason ``no_record``, not passed through.
    Passing it through is the tempting default and it is the wrong one -- it would ship exactly
    the uncorrected Phantom box this pipeline exists to remove, and it would do it silently.

    ``keep_all`` emits every subject the report has corrected boxes for, verdict included, and
    still records what the gate would have said (in the row's ``gate.verdict`` and in the drop
    list). It exists for **inspection**, where filtering first hides the evidence: a low identity
    score can mean "genuinely a different object" or "the crop was bad", and you can only tell
    those apart by looking at the mask the box produced. It never relaxes the two structural
    drops -- ``no_record`` and a missing chosen box -- because neither has a corrected box to
    segment from, so keeping them would ship an uncorrected Phantom box under a ``frame`` tag.
    Not for training data.
    """
    by_id = {int(r["subject_id"]): r for r in records}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for subject in row.get("subjects") or []:
        subject_id = int(subject["subject_id"])
        record = by_id.get(subject_id)
        if record is None:
            dropped.append({"sample_id": row["sample_id"], "subject_id": subject_id,
                            "gate": "no_record",
                            "reason": "no subject record in gate_report.json"})
            continue
        ruling = redetect.decide(record, rule, identity_min, clip_min, iou_min, iou_floor)
        if ruling["verdict"] != redetect.KEEP:
            record_of_drop = {
                "sample_id": row["sample_id"], "subject_id": subject_id,
                "gate": gate_reason(ruling), "reason": ruling["reason"],
                "identity": ruling["identity"], "clip": ruling["clip"], "iou": ruling["iou"],
            }
            # ``kept_anyway`` must not lie: a subject whose chosen box is missing cannot be
            # kept even under ``keep_all``, so it falls through to the drop below and is
            # counted under its rule gate here, exactly as in the default path.
            has_boxes = bool(record.get("chosen_box_ref")) and bool(record.get("chosen_box_seed"))
            if not (keep_all and has_boxes):
                dropped.append(record_of_drop)
                continue
            dropped.append({**record_of_drop, "kept_anyway": True})
        # Both boxes must exist to ship. Under trust_detector a filtered side already has a
        # None box and ``decide`` dropped it above; this is the belt-and-braces check that a
        # row with a missing coordinate can never reach SAM2. Unconditional: ``keep_all`` has
        # nothing to substitute here, and falling back to the uncorrected Phantom box would
        # ship it under a row claiming ``box_space: frame``.
        if not record.get("chosen_box_ref") or not record.get("chosen_box_seed"):
            dropped.append({"sample_id": row["sample_id"], "subject_id": subject_id,
                            "gate": "no_box",
                            "reason": "kept by the rule but a chosen box is missing"})
            continue
        kept.append(corrected_subject(subject, record, ruling if keep_all else None))
    if not kept:
        return None, dropped
    return {
        **row,
        "subjects": kept,
        # The tag that stops stage C re-mapping these coordinates. See the module docstring.
        "box_space": segment.BOX_SPACE_FRAME,
        "dropped_subjects": [*(row.get("dropped_subjects") or []), *dropped],
    }, dropped


def apply_gate(rows: list[dict[str, Any]], subjects: list[dict[str, Any]],
               rule: str = redetect.DEFAULT_RULE,
               identity_min: float = redetect.IDENTITY_MIN,
               clip_min: float | None = None,
               iou_min: float | None = None,
               iou_floor: float | None = None,
               keep_all: bool = False) -> dict[str, Any]:
    """Gate a whole manifest. Pure: takes the parsed report rows, returns rows and counts."""
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in subjects:
        by_sample.setdefault(str(record["sample_id"]), []).append(record)

    kept_rows: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for row in rows:
        gated, dropped = gated_row(row, by_sample.get(str(row["sample_id"])) or [],
                                   rule, identity_min, clip_min, iou_min, iou_floor,
                                   keep_all=keep_all)
        drops.extend(dropped)
        if gated is not None:
            kept_rows.append(gated)

    by_gate: dict[str, int] = {}
    for item in drops:
        by_gate[item["gate"]] = by_gate.get(item["gate"], 0) + 1
    return {
        "rows": kept_rows,
        "drops": drops,
        "summary": {
            "samples_in": len(rows),
            "samples_out": len(kept_rows),
            "samples_emptied": len(rows) - len(kept_rows),
            "subjects_in": sum(len(row.get("subjects") or []) for row in rows),
            "subjects_out": sum(len(row["subjects"]) for row in kept_rows),
            "subjects_dropped": len(drops),
            # Under ``keep_all`` these are subjects the rule rejected that were emitted anyway,
            # so ``subjects_dropped`` is a count of *verdicts*, not of subjects withheld.
            "subjects_kept_anyway": sum(1 for item in drops if item.get("kept_anyway")),
            "keep_all": keep_all,
            "dropped_by_gate": by_gate,
            "rule": rule,
            "identity_min": identity_min, "clip_min": clip_min, "iou_min": iou_min,
            # Recorded for every rule even though only ``iou_floor_peak`` reads it: a manifest
            # whose summary omits a threshold cannot be told apart from one produced before that
            # threshold existed, and this file is what the funnel and any rerun read back.
            "iou_floor": iou_floor,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                        help="where gate_report.json lives, relative to --dataset")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="stage B manifest the report was computed from")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="stage C input manifest, written under --dataset")
    parser.add_argument("--rule", default=redetect.DEFAULT_RULE, choices=redetect.RULES)
    parser.add_argument("--identity-min", type=float, default=redetect.IDENTITY_MIN)
    # Accepted and recorded, but no longer read by any rule: the CLIP and IoU judges were
    # withdrawn from the chain. Kept so the manifest's provenance block keeps its shape for
    # reports already on disk, and so an old command line does not become an error.
    parser.add_argument("--clip-min", type=float, default=None,
                        help="withdrawn judge; recorded for provenance, not applied")
    parser.add_argument("--iou-min", type=float, default=None,
                        help="the IoU one side must reach. Under --rule iou_both_sides both "
                             "sides must reach it; under iou_floor_peak it is the peak.")
    parser.add_argument("--iou-floor", type=float, default=None,
                        help="only used by --rule iou_floor_peak: the IoU *both* sides must "
                             "clear, separating an offset box from a detector that left the "
                             "annotated object")
    parser.add_argument("--keep-all", action="store_true",
                        help="emit every subject that has corrected boxes, including the ones "
                             "the rule rejects, recording the verdict on each row "
                             "(gate.verdict / gate.gate_reason). For inspection only -- "
                             "filtering first hides whether a low identity score means "
                             "'different object' or 'bad crop'. Not for training data.")
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    report_path = dataset / args.out_root / "gate_report.json"
    if not report_path.is_file():
        parser.error(f"no gate_report.json at {report_path} — run tools/tighten_run.py first")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Both default to None now that the IoU judge is withdrawn, so the ordering check only runs
    # when a caller passes both explicitly. It is kept rather than deleted because the flags are
    # still accepted for provenance, and a command line that sets them to a contradictory pair
    # should still be told so rather than have the values silently recorded as if meaningful.
    if args.iou_floor is not None and args.iou_min is not None \
            and args.iou_floor > args.iou_min:
        parser.error(f"--iou-floor ({args.iou_floor}) must not exceed --iou-min "
                     f"({args.iou_min})")

    result = apply_gate(read_jsonl(dataset / args.input), report.get("subjects") or [],
                        args.rule, args.identity_min, args.clip_min, args.iou_min,
                        args.iou_floor, keep_all=args.keep_all)

    atomic_write_bytes(dataset / args.output, "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in result["rows"]).encode("utf-8"))
    atomic_write_bytes(dataset / args.out_root / DEFAULT_DROPS, (json.dumps(
        {"summary": result["summary"], "trust_detector": report.get("trust_detector"),
         "drops": result["drops"]}, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {dataset / args.output} "
          f"({result['summary']['subjects_out']} subjects, box_space=frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
