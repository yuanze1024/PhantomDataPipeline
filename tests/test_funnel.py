"""Tests for the full-chain funnel in :mod:`phantom_data.build.funnel`.

The module's job is accounting, so the tests are organised around the three properties it
promises rather than around its functions:

* **Reconciliation is the product.** A chain where a stage silently loses samples must be
  *reported*, not smoothed over. That is what :func:`test_reconciliation_reports_a_stage_that
  _lost_samples` and the deliberately-broken fixture tree exist for -- a funnel that quietly
  balanced a broken pipeline would be worse than no funnel.
* **Missing is not zero.** A partially-run dataset must still produce a report, and a stage
  that never ran must not be credited with a clean hand-off. ``dropped=None`` and
  ``dropped=0`` are different answers and both are asserted.
* **Clips and subjects are counted separately.** They diverge (the real pilot has 140
  subjects across 138 clips), so the fixtures below reproduce that shape rather than using
  one subject per clip, which would let an axis mix-up pass.

No network, no GPU, no real dataset: every fixture is a tmp_path tree of small JSONs, built
to the same on-disk contract the real stages write.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phantom_data.build import funnel


# --------------------------------------------------------------------------------------
# fixture builders -- write the artifacts each stage really writes
# --------------------------------------------------------------------------------------


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def plan_stats(**overrides):
    """Stage A stats in the real shape (numbers from the pilot's specs_inspect160)."""
    base = {
        "seed": 20260728,
        "filtered_parquet": "/x/filtered.parquet",
        "source_rows": 651031,
        "unique_sources": 183409,
        "sources_considered": 12,
        "requested_sources": 10,
        "samples": 10,
        "rejected": {"clip_too_short": 1, "bos_unresolved": 1},
        "subjects_total": 11,
        "subject_count_histogram": {"1": 9, "2": 1},
        "dropped_subject_reasons": {"seed_outside_window": 1},
        "window": {"num_frames": 81, "fps": 16, "seconds": 5.0625},
    }
    base.update(overrides)
    return base


def build_full_dataset(root: Path) -> Path:
    """A dataset where every stage ran and the chain balances end to end.

    Shape chosen to make the two axes disagree the way the real data does: 10 clips carry 11
    subjects (one clip has two), stage B loses 1 clip and its 1 subject, and stage 3 drops 2
    subjects but only 1 clip -- because the 2-subject clip keeps a survivor.
    """
    specs = [{"sample_id": f"s{i:02d}", "subjects": [{"subject_id": 1}]} for i in range(10)]
    specs[0]["subjects"].append({"subject_id": 2})  # the 2-subject clip
    write_jsonl(root / "specs.jsonl", specs)
    write_json(root / "specs.jsonl.stats.json", plan_stats())

    # Stage B: s00..s08 pass (s00 with 2 subjects), s09 fails on a decode error.
    for spec in specs[:9]:
        write_json(root / "_stages" / "extract" / f"{spec['sample_id']}.json",
                   {"status": "passed", "subjects": len(spec["subjects"]),
                    "frame_count": 81})
    write_json(root / "_stages" / "extract" / "s09.json",
               {"status": "failed", "error": "DECORDError: could not read /x.mp4"})
    write_json(root / "_stages" / "extract.summary.json",
               {"specs": 10, "attempted": 10, "passed": 9, "failed": 1, "failures": []})

    surviving = [{"sample_id": s["sample_id"], "subjects": s["subjects"]} for s in specs[:9]]
    write_jsonl(root / "extracted.jsonl", surviving)

    # Stage 1: a summary covering all 10 surviving subjects, one on the thin fallback.
    write_json(root / "_stages" / "enrich.summary.json",
               {"subjects_in_manifest": 10, "attempted": 10, "coverage_gap": 0,
                "cache_hits": 4, "fresh": 6, "llm_ok": 9, "fallback": 1,
                "fallback_errors": {"URLError": 1}, "dropped": 0})

    # Stage 2 + the inline verdicts stage 3 reads: 10 subjects, 2 dropped on identity, both
    # drops arranged so only ONE clip dies (s00 keeps subject 2).
    subjects = []
    for spec in specs[:9]:
        for subject in spec["subjects"]:
            sid, subj = spec["sample_id"], subject["subject_id"]
            drop = (sid == "s00" and subj == 1) or sid == "s01"
            subjects.append({
                "sample_id": sid, "subject_id": subj,
                "verdict": "drop" if drop else "keep",
                "rule_identity_ok": not drop, "rule_clip_ok": True, "rule_iou_ok": True,
                "verdict_reason": "identity 0.41 < 0.6" if drop else "identity ok",
            })
    write_json(root / "_redetect100" / "gate_report.json", {
        "rule": {"identity_min": 0.6, "clip_min": 0.21, "iou_min": 0.75},
        "summary": {"subjects": 10, "kept": 8, "dropped": 2, "dropped_on_identity": 2,
                    "no_box_found_ref": 0, "no_box_found_seed": 0,
                    "ref_box_replaced": 7, "seed_box_replaced": 8},
        "subjects": subjects,
        "failures": [],
    })

    # Stage 3's own artifacts, in tools/gate_apply.py's real shapes: ``gated.jsonl`` is one
    # row per surviving CLIP (stage C's input shape), while the drop payload is a JSON object
    # whose ``drops`` list is one record per dropped SUBJECT. Counting clips as subjects here
    # is the trap the funnel has to avoid, so the fixture keeps them genuinely different.
    kept = [s for s in subjects if s["verdict"] == "keep"]
    drops = [s for s in subjects if s["verdict"] == "drop"]
    kept_by_clip: dict[str, list[dict]] = {}
    for subject in kept:
        kept_by_clip.setdefault(subject["sample_id"], []).append(
            {"subject_id": subject["subject_id"]})
    write_jsonl(root / "gated.jsonl",
                [{"sample_id": sample_id, "subjects": subs, "box_space": "frame"}
                 for sample_id, subs in sorted(kept_by_clip.items())])
    write_json(root / "_redetect100" / "gated_drops.json", {
        "summary": {
            "samples_in": 9, "samples_out": len(kept_by_clip),
            "samples_emptied": 9 - len(kept_by_clip),
            "subjects_in": 10, "subjects_out": len(kept),
            "subjects_dropped": len(drops),
            "dropped_by_gate": {"identity": len(drops)},
            "rule": "identity_required",
            "identity_min": 0.6, "clip_min": 0.21, "iou_min": 0.75,
        },
        "drops": [{"sample_id": s["sample_id"], "subject_id": s["subject_id"],
                   "gate": "identity", "reason": s["verdict_reason"]} for s in drops],
    })

    # Stage C: the 8 gated clips, all passing, carrying 8 subjects.
    for sample_id in sorted({s["sample_id"] for s in kept}):
        write_json(root / "_stages" / "segment" / f"{sample_id}.json",
                   {"status": "passed",
                    "sample": {"status": "built", "sample_id": sample_id,
                               "subjects": [{"subject_id": 1}]}})
    return root


def build_partial_dataset(root: Path) -> Path:
    """Only stage A has run. Everything after it must report ``missing``, not zero."""
    write_json(root / "specs.jsonl.stats.json", plan_stats())
    write_jsonl(root / "specs.jsonl",
                [{"sample_id": f"s{i:02d}", "subjects": [{"subject_id": 1}]}
                 for i in range(10)])
    return root


def build_broken_dataset(root: Path) -> Path:
    """A chain that does NOT reconcile, and must be reported rather than swallowed.

    The break is the realistic one: 10 specs were planned but stage B wrote only 5 markers
    (4 passed, 1 failed with a reason), so 5 clips are neither in the output nor explained by
    any artifact. ``in 10 != out 4 + dropped 1`` leaves 5 unaccounted -- exactly the case the
    funnel must refuse to balance.
    """
    write_jsonl(root / "specs.jsonl",
                [{"sample_id": f"s{i:02d}", "subjects": [{"subject_id": 1}]}
                 for i in range(10)])
    write_json(root / "specs.jsonl.stats.json",
               plan_stats(samples=10, subjects_total=10, dropped_subject_reasons={}))
    for index in range(4):
        write_json(root / "_stages" / "extract" / f"s{index:02d}.json",
                   {"status": "passed", "subjects": 1})
    write_json(root / "_stages" / "extract" / "s04.json",
               {"status": "failed", "error": "DECORDError: nope"})
    write_jsonl(root / "extracted.jsonl",
                [{"sample_id": f"s{i:02d}", "subjects": [{"subject_id": 1}]}
                 for i in range(4)])
    return root


# --------------------------------------------------------------------------------------
# fully populated tree
# --------------------------------------------------------------------------------------


def test_full_dataset_reconciles_end_to_end(tmp_path):
    """Every stage accounts for its own input, and each boundary hands off cleanly."""
    build_full_dataset(tmp_path)
    summary = funnel.build_funnel(tmp_path, specs="specs.jsonl")
    check = summary["reconciliation"]
    assert check["balanced"], check["problems"]
    assert check["checks_performed"] > 0, "a balanced verdict with zero checks is vacuous"


def test_full_dataset_has_a_row_for_every_stage(tmp_path):
    build_full_dataset(tmp_path)
    summary = funnel.build_funnel(tmp_path, specs="specs.jsonl")
    seen = [row["stage"] for row in summary["stages"]]
    for stage in ("plan", "extract", "enrich", "redetect", "gate", "segment"):
        assert stage in seen
    # Stage D is absent from this fixture, so it reports missing rather than being omitted.
    assert any(row["stage"] == "index" and row["state"] == funnel.STATE_MISSING
               for row in summary["stages"])


def rows_by_stage(summary):
    return {row["stage"]: row for row in summary["stages"]}


def test_plan_row_separates_rejections_from_the_unwalked_remainder(tmp_path):
    """``--num-sources`` stopping the walk is not a drop and must not be tallied as one."""
    build_full_dataset(tmp_path)
    plan = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))["plan"]
    assert plan["clips"]["in"] == 12  # sources_considered, not the 183409 pool
    assert plan["clips"]["out"] == 10
    assert plan["clips"]["dropped"] == 2
    assert plan["drop_reasons"]["clip_too_short"] == 1
    remainder = [k for k in plan["drop_reasons"] if "not_reached" in k]
    assert remainder, "the un-walked remainder must be named, not silently omitted"
    assert plan["drop_reasons"][remainder[0]] == 183409 - 12


def test_plan_row_counts_subjects_separately_from_clips(tmp_path):
    """A subject dropped to seed_outside_window does not drop its clip."""
    build_full_dataset(tmp_path)
    plan = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))["plan"]
    assert plan["subjects"]["out"] == 11
    assert plan["subjects"]["dropped"] == 1
    assert plan["subject_drop_reasons"] == {"seed_outside_window": 1}
    # The clip axis is untouched by that subject drop.
    assert plan["clips"]["dropped"] == 2


def test_extract_row_takes_a_failed_clips_subjects_with_it(tmp_path):
    build_full_dataset(tmp_path)
    extract = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))["extract"]
    assert extract["clips"]["in"] == 10
    assert extract["clips"]["out"] == 9
    assert extract["drop_reasons"] == {"decode_error": 1}
    assert extract["subjects"]["in"] == 11
    assert extract["subjects"]["out"] == 10
    assert extract["subject_drop_reasons"] == {"clip_dropped_with_its_subjects": 1}


def test_gate_row_drops_more_subjects_than_clips(tmp_path):
    """The divergence a single number would hide: 2 subjects die, only 1 clip does."""
    build_full_dataset(tmp_path)
    gate = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))["gate"]
    assert gate["subjects"]["dropped"] == 2
    assert gate["clips"]["dropped"] == 1
    assert gate["subject_drop_reasons"] == {"identity": 2}


def test_gate_row_does_not_read_clip_rows_as_subjects(tmp_path):
    """``gated.jsonl`` is one row per clip; ``drops`` is one per subject. Do not conflate.

    The 8 surviving subjects live in 8 clip rows here only because the one 2-subject clip lost
    a subject, so ``len(gated.jsonl)`` happens to be 8 too. The assertion that matters is that
    the subject total comes from the summary, so a multi-subject clip cannot be undercounted.
    """
    build_full_dataset(tmp_path)
    gate = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))["gate"]
    assert gate["subjects"]["in"] == 10 and gate["subjects"]["out"] == 8
    assert gate["clips"]["in"] == 9 and gate["clips"]["out"] == 8
    # Sourced from gate_apply's own artifacts, not recomputed from the inline verdicts.
    assert "gated_drops.json" in gate["source"]
    assert gate["extra"]["rule"] == "identity_required"


def test_gate_row_falls_back_to_the_pilots_inline_verdicts(tmp_path):
    """Datasets predating gate_apply keep their funnel row, recomputed from gate_report.json.

    Both artifacts must be absent for the fallback to engage, and the counts have to match
    what the explicit artifacts reported -- they are two readings of the same decision.
    """
    build_full_dataset(tmp_path)
    explicit = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))["gate"]
    (tmp_path / "gated.jsonl").unlink()
    (tmp_path / "_redetect100" / "gated_drops.json").unlink()
    fallback = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))["gate"]
    assert fallback["state"] == funnel.STATE_OK
    assert fallback["subjects"]["dropped"] == explicit["subjects"]["dropped"] == 2
    assert fallback["clips"]["dropped"] == explicit["clips"]["dropped"] == 1
    assert fallback["subject_drop_reasons"] == {"identity": 2}
    assert "gate_report.json" in fallback["extra"]["verdicts_from"]


def test_enrich_row_is_pass_through_and_says_so(tmp_path):
    """Stage 1 degrades subjects but drops none; the distinction is explicit."""
    build_full_dataset(tmp_path)
    row = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))["enrich"]
    assert row["subjects"]["dropped"] == 0
    assert row["subjects"]["in"] == row["subjects"]["out"] == 10
    assert row["extra"]["fallback"] == 1
    assert row["extra"]["degradation_not_a_drop"] is True
    # Stage 1 does not act on clips at all, so that axis must stay unmeasured rather than
    # claiming a pass-through it never checked.
    assert row["clips"] == {"in": None, "out": None, "dropped": None}


def test_index_rows_are_chained_from_stage_ds_own_funnel(tmp_path):
    """Stage D's numbers are read verbatim, split into its two independent filters."""
    build_full_dataset(tmp_path)
    write_json(tmp_path / "indexes" / "pilot_v1" / "funnel.json", {
        "index_name": "pilot_v1",
        "counts": {"source": 8, "built": 8, "quality_passed": 6, "quality_removed": 2,
                   "dedup_changed": 1, "train": 5, "eval": 1},
        "quality_rejection_clips": {"ref_clip_score": 2},
        "final_subjects": 6,
        "thresholds": {"min_ref_clip_score": 0.23},
        "threshold_deltas": {},
        "split": {"train_clips": 5, "eval_clips": 1},
    })
    summary = funnel.build_funnel(tmp_path, specs="specs.jsonl", index_name="pilot_v1")
    rows = rows_by_stage(summary)
    assert rows["index_quality"]["clips"]["in"] == 8
    assert rows["index_quality"]["clips"]["out"] == 6
    assert rows["index_quality"]["drop_reasons"]["quality:ref_clip_score"] == 2
    assert rows["index_split"]["clips"]["out"] == 6
    assert summary["headline"]["final_clips"] == 6
    assert summary["reconciliation"]["balanced"], summary["reconciliation"]["problems"]


# --------------------------------------------------------------------------------------
# partially-run tree -- missing must not read as zero
# --------------------------------------------------------------------------------------


def test_partial_dataset_still_produces_a_report(tmp_path):
    build_partial_dataset(tmp_path)
    summary = funnel.build_funnel(tmp_path, specs="specs.jsonl")
    assert rows_by_stage(summary)["plan"]["state"] == funnel.STATE_OK
    assert summary["headline"]["last_stage_run"] == "plan"
    # And it renders, which is the other half of "still produces a report".
    assert "Phantom-Koala Full-Chain Funnel" in funnel.format_markdown(summary)


def test_missing_stage_reports_none_not_zero(tmp_path):
    """The distinction the whole ``state`` field exists for."""
    build_partial_dataset(tmp_path)
    rows = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))
    for stage in ("extract", "redetect", "gate", "segment"):
        assert rows[stage]["state"] == funnel.STATE_MISSING, stage
        assert rows[stage]["clips"]["dropped"] is None, (
            f"{stage} never ran, so its drop count is unknown, not 0")
    # Contrast: enrich in the full tree *did* run and dropped nothing, which is a real 0.
    other = tmp_path / "full"
    build_full_dataset(other)
    assert rows_by_stage(funnel.build_funnel(other, specs="specs.jsonl"))["enrich"][
        "subjects"]["dropped"] == 0


def test_missing_stages_are_excluded_from_reconciliation(tmp_path):
    """A stage that never ran must not be credited with a clean hand-off."""
    build_partial_dataset(tmp_path)
    check = funnel.build_funnel(tmp_path, specs="specs.jsonl")["reconciliation"]
    assert check["balanced"], check["problems"]
    skipped = [item["stage"] for item in check["skipped"] if item.get("why") == "stage not run"]
    for stage in ("extract", "redetect", "gate", "segment"):
        assert stage in skipped


def test_partial_markdown_names_the_stages_that_did_not_run(tmp_path):
    build_partial_dataset(tmp_path)
    text = funnel.format_markdown(funnel.build_funnel(tmp_path, specs="specs.jsonl"))
    assert "## Stages Not Run" in text
    assert "`segment`" in text


def test_enrich_falls_back_to_the_cache_and_flags_the_ambiguity(tmp_path):
    """No summary: the cache is usable but coarser, and the report says why.

    Failures are deliberately not cached, so a missing cache file cannot be told apart from
    "not enriched yet" -- the row must admit that rather than pick one.
    """
    build_full_dataset(tmp_path)
    (tmp_path / "_stages" / "enrich.summary.json").unlink()
    for index in range(8):  # 8 cache files for 10 subjects
        write_json(tmp_path / "_enrich" / f"s{index:02d}_subj01.json",
                   {"dis": "a woman in a red coat", "text_source": "llm"})
    row = rows_by_stage(funnel.build_funnel(tmp_path, specs="specs.jsonl"))["enrich"]
    assert row["state"] == funnel.STATE_PARTIAL
    assert row["subjects"]["dropped"] == 0, "enrich never drops, even when incomplete"
    assert row["extra"]["fallback_or_pending"] == 2
    assert "never succeeded" in row["extra"]["why_ambiguous"]


# --------------------------------------------------------------------------------------
# deliberately broken tree -- the check that matters most
# --------------------------------------------------------------------------------------


def test_reconciliation_reports_a_stage_that_lost_samples(tmp_path):
    """5 clips vanished with no recorded reason. The funnel must say so, loudly.

    The number is 5, not 4: of 10 planned specs only 5 have markers (4 passed + 1 explained
    failure), so ``in 10 != out 4 + dropped 1``. The 5 without markers are unaccounted
    precisely because a missing marker is not a drop -- see ``_stage_extract``.
    """
    build_broken_dataset(tmp_path)
    summary = funnel.build_funnel(tmp_path, specs="specs.jsonl")
    check = summary["reconciliation"]
    assert not check["balanced"], "a chain missing 5 clips must not be reported as balanced"
    imbalances = [p for p in check["problems"]
                  if p["kind"] == "stage_imbalance" and p["stage"] == "extract"
                  and p["axis"] == "clips"]
    assert imbalances, check["problems"]
    assert imbalances[0]["unaccounted"] == 5
    assert imbalances[0]["in"] == 10 and imbalances[0]["out"] == 4


def test_broken_chain_is_loud_in_the_markdown(tmp_path):
    """Rendered before the table, not buried under it."""
    build_broken_dataset(tmp_path)
    text = funnel.format_markdown(funnel.build_funnel(tmp_path, specs="specs.jsonl"))
    assert "NOT RECONCILED" in text
    assert text.index("NOT RECONCILED") < text.index("## Chain"), (
        "a discrepancy rendered after the table is a discrepancy nobody reads")


def test_broken_chain_exits_non_zero(tmp_path, capsys):
    """A scale-up loop must be able to notice without a human reading the table."""
    build_broken_dataset(tmp_path)
    code = funnel.main(["--dataset", str(tmp_path), "--specs", "specs.jsonl", "--no-write"])
    assert code == 2
    assert "NOT RECONCILED" in capsys.readouterr().out


def test_full_chain_exits_zero_and_writes_both_files(tmp_path):
    build_full_dataset(tmp_path)
    assert funnel.main(["--dataset", str(tmp_path), "--specs", "specs.jsonl"]) == 0
    written = json.loads((tmp_path / "funnel_full.json").read_text(encoding="utf-8"))
    assert written["reconciliation"]["balanced"]
    assert "## Chain" in (tmp_path / "FUNNEL.md").read_text(encoding="utf-8")


def test_boundary_gap_between_stages_is_reported(tmp_path):
    """A sample lost *between* stages, where no single stage's own row looks wrong."""
    stages = [
        funnel.stage_row("plan", funnel.STATE_OK, clips_in=10, clips_out=10, clips_dropped=0),
        # Internally consistent (7 == 7 + 0) yet 3 clips never arrived from plan.
        funnel.stage_row("extract", funnel.STATE_OK, clips_in=7, clips_out=7, clips_dropped=0),
    ]
    check = funnel.reconcile(stages)
    assert not check["balanced"]
    gaps = [p for p in check["problems"] if p["kind"] == "boundary_gap"]
    assert gaps and gaps[0]["unaccounted"] == 3
    assert gaps[0]["from"] == "plan" and gaps[0]["to"] == "extract"


def test_reconcile_never_treats_none_as_zero(tmp_path):
    """Coercing an unmeasured axis to 0 would manufacture a false balance."""
    stages = [funnel.stage_row("plan", funnel.STATE_OK, clips_in=10, clips_out=8,
                               clips_dropped=None)]
    check = funnel.reconcile(stages)
    # Not a problem (nothing to check) and not a pass either: it is recorded as skipped.
    assert check["balanced"]
    assert any(item.get("axis") == "clips" for item in check["skipped"])
    # Whereas a real 0 there is checked, and fails.
    stages[0]["clips"]["dropped"] = 0
    assert not funnel.reconcile(stages)["balanced"]


def test_reconcile_bridges_over_a_skipped_stage(tmp_path):
    """A mid-flight dataset still gets its surviving boundaries verified."""
    stages = [
        funnel.stage_row("plan", funnel.STATE_OK, clips_in=10, clips_out=10, clips_dropped=0),
        funnel.stage_row("enrich", funnel.STATE_MISSING),
        funnel.stage_row("segment", funnel.STATE_OK, clips_in=4, clips_out=4, clips_dropped=0),
    ]
    problems = funnel.reconcile(stages)["problems"]
    assert [p["kind"] for p in problems] == ["boundary_gap"]
    assert problems[0]["from"] == "plan" and problems[0]["to"] == "segment"


# --------------------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("error,expected", [
    ("DECORDError: [11:22:33] could not open /x.mp4", "decode_error"),
    ("TimeoutError: exceeded 900s (worker wedged, abandoned)",
     "decode_timeout_worker_wedged"),
    ("OSError: [Errno 5] Input/output error", "OSError"),
    ("", "unknown_error"),
    ("BrandNewFailure: never seen before", "BrandNewFailure"),
])
def test_extract_error_classification(error, expected):
    """The wedge case is split from a plain decode error because they differ operationally.

    A DECORDError costs one failed read; a wedge held a worker 65 minutes and ended with the
    pod OOM-killed. An unknown failure keeps its own type name rather than being folded into
    ``other``, so a new mode shows up as itself the first time it happens.
    """
    assert funnel.classify_extract_error(error) == expected


def test_gate_tally_counts_a_clip_dead_only_when_all_its_subjects_die():
    subjects = [
        {"sample_id": "a", "subject_id": 1, "verdict": "drop", "rule_identity_ok": False},
        {"sample_id": "a", "subject_id": 2, "verdict": "keep", "rule_identity_ok": True},
        {"sample_id": "b", "subject_id": 1, "verdict": "drop", "rule_identity_ok": False},
    ]
    tally = funnel.gate_reasons_from_subjects(subjects)
    assert tally["subjects_dropped"] == 2
    assert tally["clips_dropped"] == 1, "clip 'a' survives on its second subject"
    assert tally["subject_reasons"] == {"identity": 2}


def test_gate_tally_reads_the_pilots_nested_verdict_shape():
    """The pilot's gate_report puts the verdict under ``extra``; both shapes are accepted."""
    subjects = [{"sample_id": "a", "subject_id": 1, "extra": {"verdict": "KEEP"}},
                {"sample_id": "b", "subject_id": 1, "extra": {"verdict": "DROP"},
                 "verdict_reason": "identity 0.42 < 0.6"}]
    tally = funnel.gate_reasons_from_subjects(subjects)
    assert tally["subjects_kept"] == 1 and tally["subjects_dropped"] == 1
    assert tally["subject_reasons"] == {"identity": 1}


def test_redetect_counts_a_missing_box_as_a_drop():
    """Under the new order a no-box subject is filtered out, not fallen back."""
    report = {"summary": {"subjects": 10, "no_box_found_ref": 2, "no_box_found_seed": 1,
                          "ref_box_replaced": 5}, "failures": []}
    row = funnel._stage_redetect(report, 10, "x")
    assert row["subjects"]["dropped"] == 3
    assert row["subject_drop_reasons"] == {"no_box_found_ref": 2, "no_box_found_seed": 1}
    assert row["subjects"]["out"] == 7
    # Replacing a box is not dropping a subject.
    assert row["extra"]["replacement_is_not_a_drop"] is True


def test_segment_separates_a_data_rejection_from_a_code_failure():
    """``rejected`` is terminal and about the data; ``failed`` is retryable and about code."""
    markers = {
        "a": {"status": "passed", "sample": {"subjects": [{"subject_id": 1}]}},
        "b": {"status": "rejected", "reasons": [{"code": "empty_masklet"}]},
        "c": {"status": "failed", "error": "RuntimeError: CUDA oom"},
    }
    row = funnel._stage_segment(markers, 3, "x")
    assert row["clips"]["out"] == 1 and row["clips"]["dropped"] == 2
    assert row["drop_reasons"] == {"rejected:empty_masklet": 1, "failed:RuntimeError": 1}


def test_merge_plan_stats_sums_counters_but_not_the_global_pool():
    """Summing ``unique_sources`` across shards would multiply the pool by the shard count."""
    shards = [
        plan_stats(samples=10, sources_considered=12, subjects_total=11,
                   rejected={"clip_too_short": 1, "bos_unresolved": 1},
                   shard={"shard_id": 0, "num_shards": 2}),
        plan_stats(samples=10, sources_considered=13, subjects_total=10,
                   rejected={"clip_too_short": 3},
                   shard={"shard_id": 1, "num_shards": 2}),
    ]
    merged = funnel.merge_plan_stats(shards)
    assert merged["samples"] == 20
    assert merged["sources_considered"] == 25
    assert merged["subjects_total"] == 21
    assert merged["rejected"] == {"clip_too_short": 4, "bos_unresolved": 1}
    assert merged["unique_sources"] == 183409, "the pool is global, not per shard"
    assert merged["source_rows"] == 651031
    assert merged["shards"]["count"] == 2


def test_merge_plan_stats_is_identity_for_a_single_run():
    stats = plan_stats()
    assert funnel.merge_plan_stats([stats]) == stats
    assert funnel.merge_plan_stats([]) == {}


def test_cell_distinguishes_unmeasured_from_zero():
    """`-` vs `0` at a glance is the reason ``None`` is carried through the row builders."""
    assert funnel._cell(None) == "-"
    assert funnel._cell(0) == "0"
    assert funnel._cell(183409) == "183,409"


def test_read_helpers_tolerate_absent_and_torn_files(tmp_path):
    """A run killed mid-write leaves a torn final line; it must not crash the report."""
    assert funnel.read_json(tmp_path / "nope.json") is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert funnel.read_json(tmp_path / "bad.json") is None
    (tmp_path / "torn.jsonl").write_text('{"a": 1}\n\n{"b": 2}\n{"c":', encoding="utf-8")
    assert funnel.read_jsonl_rows(tmp_path / "torn.jsonl") == [{"a": 1}, {"b": 2}]
    assert funnel.read_markers(tmp_path / "nodir") == {}


def test_artifact_gaps_are_published_in_the_json(tmp_path):
    """The limits of the accounting ship with its results, not just in a docstring."""
    build_partial_dataset(tmp_path)
    gaps = funnel.build_funnel(tmp_path, specs="specs.jsonl")["artifact_gaps"]
    assert {gap["stage"] for gap in gaps} >= {"plan", "extract", "enrich", "redetect"}
    assert all(gap["consequence"] and gap["fix"] for gap in gaps)
