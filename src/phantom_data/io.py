"""Reading manifests, decoding clips, and writing artefacts without leaving a half-written one.

Three functions shared by every stage. They lived in a 674-line rendering module until the
frontends that needed the rest of it were removed; the pipeline only ever used these.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every non-blank line of a jsonl manifest, parsed."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def decode_frames(path: Path) -> list[np.ndarray]:
    """Every frame of a clip as RGB arrays.

    Alpha is dropped rather than carried: the clips are h264 and have none, but a stray 4-channel
    frame would otherwise propagate into the box arithmetic as a silent shape mismatch.
    """
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        return [np.asarray(frame)[..., :3] for frame in reader]
    finally:
        reader.close()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write via a temp file + rename, so a full disk cannot leave a truncated artefact.

    A 0-byte ``gate_report.json`` is worse than a missing one: a reader's ``json.loads`` raises on
    it, and a 0-byte npz reads as corrupt rather than as absent, so the failure surfaces far from
    its cause. Observed for real when the filesystem filled mid-run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(payload)
            sink.flush()
            # fsync before the rename: the rename is atomic with respect to readers, but without
            # the flush the bytes can still be in the page cache when the machine dies.
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
