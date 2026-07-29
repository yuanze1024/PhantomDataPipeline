"""Stage 1 of the box pipeline: fill the phrase cache for every subject in a built dataset.

Split out from stage 2 on purpose. This stage is network-bound and costs money per call,
whereas stage 2 is GPU-bound and free to repeat; keeping them apart means the detector pass can
be re-run any number of times against a cache that was paid for once.

Concurrency is threads, not processes: every worker is blocked on a socket, and a shared
cache directory across processes would race on the same file names.

Usage: python tools/enrich_subjects.py --dataset <root> [--workers 8] [--limit N]

Writes ``<root>/_stages/enrich.summary.json`` so this stage has a funnel row that does not
require walking the cache. It is a *summary*, not a marker store: the cache directory is
already the resume state, and :func:`phantom_data.enrich.cached_enrich` deliberately does
not cache failures (a dead gateway is exactly when a retry is wanted), so a missing cache
file means "never succeeded" and the summary says so rather than implying a drop. Nothing
is filtered out here -- every subject leaves with a ``dis``, just possibly Phantom's own
thin phrase -- so the funnel row is pass-through with a degradation count.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

from phantom_data import enrich
from phantom_data.inspect import read_jsonl

DEFAULT_CACHE = "_enrich"
STAGE = "enrich"


def subject_jobs(dataset: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """One job per subject, carrying both Phantom texts and the clip caption.

    The reference-side ``bbox_cls`` is a different string from the target's for ~65% of
    subjects, and only ``extracted.jsonl`` has it -- the bbox JSONs carry the target's
    alone -- so the fallback text has to come from here.
    """
    jobs: list[dict[str, Any]] = []
    for row in read_jsonl(dataset / "extracted.jsonl"):
        for subject in row.get("subjects") or []:
            jobs.append({
                "sample_id": row["sample_id"],
                "subject_id": int(subject["subject_id"]),
                "caption": str(row.get("caption") or ""),
                "phrase": str(subject.get("phrase") or subject.get("bbox_cls") or ""),
                "ref_phrase": str((subject.get("ref") or {}).get("bbox_cls") or ""),
            })
    return jobs[:limit] if limit else jobs


def summarize(results: list[dict[str, Any]], *, subjects_in_manifest: int,
              elapsed_sec: float = 0.0, model: str = enrich.MODEL,
              cache_dir: str = "") -> dict[str, Any]:
    """Funnel row for stage 1, from the per-subject results this run collected.

    Pure so it is testable without a gateway. Field meanings, because two of them are easy
    to misread:

    * ``attempted`` is what this *run* looked at (``--limit`` truncates it), while
      ``subjects_in_manifest`` is the denominator the funnel actually needs -- with
      ``--limit 3`` on 140 subjects, ``attempted=3`` and the other 137 are simply not done.
      ``coverage_gap`` names that difference so a partial run cannot read as a complete one.
    * ``llm_ok`` counts subjects now holding an LLM phrase, whether this run paid for it
      (``fresh``) or read it from disk (``cache_hits``). ``fallback`` counts subjects left on
      Phantom's own phrase -- a *degradation*, not a drop: they continue down the pipeline.

    ``dropped`` is fixed at 0 and stated explicitly rather than omitted, because the funnel
    reconciler must be able to tell "this stage drops nothing by design" from "this stage's
    drop count is unknown".
    """
    total = len(results)
    cache_hits = sum(1 for r in results if r.get("cache_hit"))
    fallbacks = [r for r in results if r.get("text_source") == enrich.SOURCE_FALLBACK]
    llm_ok = sum(1 for r in results if r.get("text_source") == enrich.SOURCE_LLM)
    errors: dict[str, int] = {}
    for result in fallbacks:
        key = str(result.get("error") or "unknown")
        errors[key] = errors.get(key, 0) + 1
    words = sorted(len((r.get("dis") or "").split()) for r in results)
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model,
        "cache_dir": cache_dir,
        "subjects_in_manifest": int(subjects_in_manifest),
        "attempted": total,
        # A run that did not look at every subject in the manifest. Non-zero means the
        # funnel's enrich row covers only part of the dataset.
        "coverage_gap": max(0, int(subjects_in_manifest) - total),
        "cache_hits": cache_hits,
        "fresh": total - cache_hits,
        "llm_ok": llm_ok,
        "fallback": len(fallbacks),
        # Failures are never cached (see the module docstring), so on a re-run these
        # subjects are retried; a persistent count here means a persistently dead gateway.
        "fallback_errors": dict(sorted(errors.items(), key=lambda kv: (-kv[1], kv[0]))),
        "dropped": 0,
        "drops_nothing_by_design": True,
        "dis_words_median": words[len(words) // 2] if words else 0,
        "dis_words_max": words[-1] if words else 0,
        "thin_phantom_phrases": sum(1 for r in results
                                    if len((r.get("phrase") or "").split()) <= 2),
        "elapsed_sec": round(float(elapsed_sec), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=enrich.MODEL)
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    cache_dir = dataset / args.cache_dir
    # Full job list before --limit, so the summary's denominator is the dataset and not the
    # slice this invocation happened to run.
    all_jobs = subject_jobs(dataset)
    jobs = all_jobs[: args.limit] if args.limit else all_jobs
    key = enrich.read_key()
    print(f"{len(jobs)} subjects, cache {cache_dir}, model {args.model}", flush=True)

    started = time.time()
    results: list[dict[str, Any]] = []

    def run(job: dict[str, Any]) -> dict[str, Any]:
        result = enrich.cached_enrich(
            cache_dir, job["sample_id"], job["subject_id"], job["caption"], job["phrase"],
            job["ref_phrase"], key=key, model=args.model)
        return {**job, **result}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(run, jobs), 1):
            results.append(result)
            if index % 10 == 0 or index == len(jobs):
                print(f"  [{index}/{len(jobs)}] {time.time() - started:.0f}s", flush=True)

    hits = sum(1 for r in results if r.get("cache_hit"))
    fallbacks = [r for r in results if r["text_source"] == enrich.SOURCE_FALLBACK]
    words = sorted(len(r["dis"].split()) for r in results)
    thin = sum(1 for r in results if len(r["phrase"].split()) <= 2)
    print(f"\n{len(results)} subjects in {time.time() - started:.0f}s "
          f"({hits} from cache, {len(fallbacks)} fell back to the Phantom phrase)")
    print(f"dis words: median {words[len(words) // 2]}, max {words[-1]}, "
          f">8 words {sum(1 for w in words if w > 8)}")
    print(f"subjects whose Phantom phrase was <=2 words: {thin} "
          f"({100 * thin / max(1, len(results)):.0f}%) -- these are what this stage is for")
    for result in fallbacks[:10]:
        print(f"  FALLBACK {result['sample_id']} subj{result['subject_id']:02d} "
              f"({result.get('error')})")
    for result in results[:12]:
        print(f"  '{result['phrase'][:28]:28s}' -> '{result['dis'][:60]}'")

    summary = summarize(results, subjects_in_manifest=len(all_jobs),
                        elapsed_sec=time.time() - started, model=args.model,
                        cache_dir=str(cache_dir))
    summary_path = dataset / "_stages" / f"{STAGE}.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Same atomic-write helper the other stages' summaries use, so a killed run cannot leave
    # a half-written summary that funnel.py would then read as truth.
    from ultravid_pipeline.state import atomic_write_json

    atomic_write_json(summary_path, summary)
    print(f"wrote summary -> {summary_path}", flush=True)
    return 1 if fallbacks else 0


if __name__ == "__main__":
    raise SystemExit(main())
