"""Build a bbox+ref training dataset out of Phantom-Data annotations + Koala videos on BOS.

Stages:

- ``plan``    parquet rows -> sample specs (window choice, seed frames, ref pointers)
- ``extract`` specs -> 81-frame mp4 clips + reference jpgs (via a storage backend)
- ``segment`` clips -> SAM2 masklets + white-matted reference cutouts + bbox json
- ``index``   ``segmented.jsonl`` -> ``indexes/<name>/`` train/eval CSVs + funnel

``segment`` and ``index`` are imported lazily below: they pull in torch / SAM2 and
UltraVidPipeline respectively, which must not be a hard requirement for using ``plan``.
"""
from __future__ import annotations

from .window import (
    FPS,
    WINDOW_FRAMES,
    WINDOW_SEC,
    WindowPlan,
    choose_window,
    sample_id_for,
    seed_frame_index,
)

__all__ = [
    "FPS",
    "WINDOW_FRAMES",
    "WINDOW_SEC",
    "WindowPlan",
    "choose_window",
    "sample_id_for",
    "seed_frame_index",
    "index",
    "segment",
]


def __getattr__(name: str):
    if name in ("index", "segment"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
