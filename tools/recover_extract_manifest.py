"""Rebuild stage B's ``extracted.jsonl`` from specs + markers after a killed run.

Only needed for datasets extracted before ``extract.partial.jsonl`` existed. A run that
dies before its final write leaves the clips and ref jpgs on disk and a ``passed`` marker
per sample, but no manifest -- and because a resumed run skips marked samples, the manifest
would come out empty. Re-extracting instead would re-download every source video.

A marker records only shape (``video``/``width``/``height``/``frame_count``), so the
per-subject boxes and ref paths are taken from the spec, exactly as ``extract_sample``
would have assembled them. Every referenced artifact is verified to exist on disk; a
sample missing one is skipped and reported rather than written as a row the trainer would
later fail on.

Usage: python tools/recover_extract_manifest.py --dataset <root> --specs <specs.jsonl>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def ref_relpath(sample_id: str, subject_id: int) -> str:
    """Mirrors ``extract.ref_relpath``; imported lazily so this tool needs no BOS deps."""
    return f"ref_frames/{sample_id}_subj{subject_id:02d}.jpg"


def rebuild_row(spec: dict[str, Any], marker: dict[str, Any],
                dataset: Path) -> tuple[dict[str, Any] | None, str | None]:
    """One manifest row, or ``(None, reason)`` when an artifact is missing."""
    sample_id = spec["sample_id"]
    video = marker.get("video") or f"clips/{sample_id}.mp4"
    if not (dataset / video).is_file():
        return None, f"clip missing: {video}"

    width, height = marker.get("width"), marker.get("height")
    frame_count = marker.get("frame_count")
    if not all((width, height, frame_count)):
        return None, "marker lacks width/height/frame_count"

    subjects = []
    for subject in spec["subjects"]:
        subject_id = int(subject["subject_id"])
        relative = ref_relpath(sample_id, subject_id)
        path = dataset / relative
        if not path.is_file():
            return None, f"ref frame missing: {relative}"
        from PIL import Image

        with Image.open(path) as image:
            ref_width, ref_height = image.size
        subjects.append({
            **subject,
            "ref": {
                **subject["ref"],
                "frame": relative,
                "ref_frame_width": int(ref_width),
                "ref_frame_height": int(ref_height),
            },
        })
    if len(subjects) != int(marker.get("subjects") or len(subjects)):
        return None, (f"subject count mismatch: spec has {len(subjects)}, "
                      f"marker recorded {marker.get('subjects')}")

    source = dict(spec["source"])
    return {
        "sample_id": sample_id,
        "video_id": spec["video_id"],
        "phantom_video_id": spec["phantom_video_id"],
        "video": video,
        "caption": spec["caption"],
        "prompt": spec["caption"],
        "width": int(width),
        "height": int(height),
        "frame_count": int(frame_count),
        "fps": int(spec.get("fps") or 16),
        # fps_source / source_total_frames were only ever in the lost manifest; the
        # recovered row omits them rather than inventing values. Stage C does not read them.
        "source": source,
        "subjects": subjects,
        "dropped_subjects": spec.get("dropped_subjects") or [],
        "recovered_from_markers": True,
    }, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--specs", required=True, type=Path)
    parser.add_argument("--out", default="extracted.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dataset = args.dataset.resolve()
    specs = {row["sample_id"]: row for row in read_jsonl(args.specs)}
    marker_dir = dataset / "_stages" / "extract"

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    no_marker = 0
    for sample_id, spec in specs.items():
        path = marker_dir / f"{sample_id}.json"
        if not path.is_file():
            no_marker += 1
            continue
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("status") != "passed":
            continue
        row, reason = rebuild_row(spec, marker, dataset)
        if row is None:
            skipped.append({"sample_id": sample_id, "reason": reason or "unknown"})
            continue
        rows.append(row)

    print(f"specs {len(specs)}  no marker {no_marker}  recovered {len(rows)}  "
          f"skipped {len(skipped)}")
    for item in skipped[:10]:
        print(f"  SKIP {item['sample_id']}: {item['reason']}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0
    out = dataset / args.out
    if out.exists():
        print(f"refusing to overwrite existing {out}")
        return 1
    tmp = out.with_suffix(".jsonl.recovering")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                   encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
