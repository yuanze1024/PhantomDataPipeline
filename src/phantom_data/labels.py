"""Human pass/fail labels for gated subjects: the ground truth the thresholds never had.

Every threshold in :mod:`phantom_data.redetect` was set by looking at a handful of samples.
Nothing in the repo could say whether ``IDENTITY_MIN = 0.6`` is right, whether a different
identity model would be better, or how much of the pilot is actually bad -- there were **zero**
human labels on disk, so every judge swap and every threshold move was unfalsifiable. On the
140-subject pilot the identity median lands at 0.609 against that 0.6 threshold, which makes the
keep rate maximally sensitive to noise; and the three judges are near-independent (identity vs
IoU Spearman +0.008), so their AND drops far more than any one of them suggests. Neither fact
can be acted on without labels. This module is the write path.

**One file per subject**, ``<dataset>/_labels/<sample_id>_subj<NN>.json``. Not one appended
jsonl, for two reasons that both bite on this filesystem: ``flock`` is broken on juicefs, so
concurrent appends from two browser tabs cannot be made safe; and a per-subject file makes
relabelling idempotent (the second write replaces the first) and resume free (the loader just
reads what is there). Each write goes through :func:`phantom_data.inspect.atomic_write_bytes`,
so a full disk leaves a missing file rather than a truncated one that takes down the page.

**The label is deliberately one bit.** Not because failure modes do not matter -- which stage to
fix is exactly the open question -- but because the *attribution* does not have to come from the
human. Every judge's score is already in ``gate_report.json`` for all 140 subjects, so a single
binary label per subject is enough to compute each judge's AUROC separately and see which one
tracks the human. Asking for a reason code would triple the click cost to recover information
already recoverable from the scores.

**Blind mode exists because the page is an anchoring machine.** ``render_subject`` opens with a
KEEP/DROP banner and prints all three scores; a human reading that before deciding will regress
toward the machine, and these labels exist precisely to grade the machine. So the label widget
can hide the verdict and the scores, and every record notes whether it was written blind
(:data:`SCHEMA_VERSION` 1 records ``blind``). Non-blind labels are still usable -- for triage,
or where the reviewer overrode a verdict they could see -- but a judge comparison should say
which subset it ran on.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from phantom_data.inspect import atomic_write_bytes

#: Bumped when a field changes meaning. Readers should skip records they do not understand
#: rather than guess: a label whose semantics are unknown is worse than a missing one.
SCHEMA_VERSION = 1

DEFAULT_LABEL_DIR = "_labels"

#: The two verdicts, in the reviewer's own words. Stored as ASCII keys rather than the Chinese
#: labels so the on-disk file is greppable and diffable from any locale, with the display text
#: kept in :data:`VERDICT_LABELS`.
PASS, FAIL = "pass", "fail"
VERDICTS = (PASS, FAIL)

VERDICT_LABELS = {PASS: "合格 (pass)", FAIL: "不合格 (fail)"}

#: ``sample_id`` is a 32-hex uuid plus ``_w`` and 9 digits; ``subject_id`` is small. Validated
#: on the way in because these two strings become a filename, and a stray ``/`` or ``..`` in a
#: manifest would write outside the label directory.
_SAMPLE_ID = re.compile(r"^[0-9a-zA-Z_-]{1,128}$")


def label_name(sample_id: str, subject_id: int | str) -> str:
    """Filename for one subject's label. Mirrors ``enrich.cache_name``'s convention."""
    if not _SAMPLE_ID.match(str(sample_id)):
        raise ValueError(f"unsafe sample_id for a filename: {sample_id!r}")
    return f"{sample_id}_subj{int(subject_id):02d}.json"


def label_path(label_dir: Path, sample_id: str, subject_id: int | str) -> Path:
    return Path(label_dir) / label_name(sample_id, subject_id)


def make_record(sample_id: str, subject_id: int | str, verdict: str, *,
                blind: bool = False, note: str = "",
                scores: dict[str, Any] | None = None,
                labelled_at: str | None = None) -> dict[str, Any]:
    """Build one label record. Pure, so the schema can be tested without a filesystem.

    ``scores`` snapshots the judge numbers the subject had **when it was labelled**. That is
    redundant with ``gate_report.json`` right up until the report is regenerated -- and it is
    about to be, because the pilot's ``dis`` text was silently degraded (``text_source:
    phantom_fallback`` on 140/140, median 2 words) and rerunning redetect will move every clip
    score and some boxes. Without this snapshot there would be no way to tell whether a label
    was written against the old numbers or the new ones.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    record = {
        "schema": SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "subject_id": int(subject_id),
        "verdict": verdict,
        # Whether the reviewer could see the machine's verdict and scores. See module docstring.
        "blind": bool(blind),
        "note": str(note or ""),
    }
    if labelled_at:
        record["labelled_at"] = str(labelled_at)
    if scores:
        record["scores_at_label"] = scores
    return record


def write_label(label_dir: Path, record: dict[str, Any]) -> Path:
    """Persist one record, replacing any earlier label for the same subject."""
    path = label_path(label_dir, record["sample_id"], record["subject_id"])
    payload = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)
    return path


def delete_label(label_dir: Path, sample_id: str, subject_id: int | str) -> bool:
    """Remove a label so a subject returns to unlabelled. True if one was there.

    Needed because the alternative -- a third "unsure" verdict -- would quietly become a
    dumping ground and leave the label set neither complete nor clean.
    """
    path = label_path(label_dir, sample_id, subject_id)
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return False


def read_label(label_dir: Path, sample_id: str, subject_id: int | str) -> dict[str, Any] | None:
    """One label, or None. A corrupt or unknown-schema file reads as None, not as an error:
    the labelling page must stay usable when one file is bad."""
    path = label_path(label_dir, sample_id, subject_id)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA_VERSION:
        return None
    if record.get("verdict") not in VERDICTS:
        return None
    return record


def load_labels(label_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Every label on disk, keyed by ``(sample_id, subject_id)``.

    Reads the directory rather than an index file: with one file per subject the directory
    *is* the index, and a separate index could disagree with it.
    """
    directory = Path(label_dir)
    if not directory.is_dir():
        return {}
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (not isinstance(record, dict) or record.get("schema") != SCHEMA_VERSION
                or record.get("verdict") not in VERDICTS):
            continue
        try:
            key = (str(record["sample_id"]), int(record["subject_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = record
    return out


def label_summary(labels: dict[tuple[str, int], dict[str, Any]],
                  total: int | None = None) -> dict[str, Any]:
    """Counts for the sidebar: how many labelled, the pass rate, how many blind."""
    values = list(labels.values())
    passed = sum(1 for record in values if record["verdict"] == PASS)
    summary: dict[str, Any] = {
        "labelled": len(values),
        "pass": passed,
        "fail": len(values) - passed,
        "blind": sum(1 for record in values if record.get("blind")),
    }
    if total is not None:
        summary["total"] = int(total)
        summary["remaining"] = max(0, int(total) - len(values))
    if values:
        summary["pass_rate"] = round(passed / len(values), 4)
    return summary


def next_unlabelled(subjects: list[dict[str, Any]],
                    labels: dict[tuple[str, int], dict[str, Any]],
                    start: int = 0) -> int | None:
    """Index of the next subject with no label, scanning from ``start`` and wrapping once.

    Wrapping matters for a session resumed after a break: the reviewer lands wherever the page
    was, and the unlabelled remainder is usually *behind* that point.
    """
    if not subjects:
        return None
    count = len(subjects)
    for offset in range(count):
        index = (start + offset) % count
        subject = subjects[index]
        key = (str(subject.get("sample_id")), int(subject.get("subject_id", 0)))
        if key not in labels:
            return index
    return None


def scores_of(subject: dict[str, Any]) -> dict[str, Any]:
    """The judge numbers worth snapshotting alongside a label. Report field names, verbatim."""
    fields = ("rule_identity", "rule_clip", "chosen_clip_ref", "chosen_clip_seed",
              "iou_dis_vs_phantom", "iou_seed_dis_vs_phantom",
              "detector_score_ref_dis", "detector_score_seed_dis",
              "candidates_ref_dis", "candidates_seed_dis", "text_source")
    return {name: subject.get(name) for name in fields if subject.get(name) is not None}
