"""Pure window-selection math for stage A. No IO, no network.

A training sample is a fixed 81-frame @ 16fps window carved out of one Phantom clip
``<uuid>_<start>_<end>``. Every subject of the clip carries a "seed" annotation at a
normalized position inside the clip; the window must contain as many of those seed
times as possible while staying inside the clip boundaries.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

WINDOW_FRAMES = 81
FPS = 16
WINDOW_SEC = WINDOW_FRAMES / FPS  # 5.0625 s
EPS = 1e-9


@dataclass(frozen=True)
class WindowPlan:
    """Chosen window plus which subject slots it covers.

    ``covered``/``dropped`` hold indices into the ``seed_times`` list passed in.
    """

    window_start: float
    covered: tuple[int, ...] = field(default=())
    dropped: tuple[int, ...] = field(default=())

    @property
    def window_end(self) -> float:
        return self.window_start + WINDOW_SEC


def choose_window(
    start: float,
    end: float,
    seed_times: list[float],
    window_sec: float = WINDOW_SEC,
) -> WindowPlan | None:
    """Pick the window start covering the most seed times, or None if impossible.

    Constraints: ``start <= w0`` and ``w0 + window_sec <= end``. Among the windows with
    maximal coverage the one that centers the covered seeds is returned, which makes the
    result deterministic and keeps seeds away from the window edges. Returns None when
    the clip is shorter than the window or no seed can be covered at all.
    """
    if end - start < window_sec - EPS:
        return None
    low = start
    high = max(end - window_sec, start)

    def clamp(value: float) -> float:
        return min(max(value, low), high)

    candidates = {low, high}
    for time in seed_times:
        candidates.add(clamp(time))
        candidates.add(clamp(time - window_sec))

    best_key: tuple[int, tuple[int, ...]] | None = None
    best_covered: tuple[int, ...] = ()
    for w0 in sorted(candidates):
        covered = tuple(
            index
            for index, time in enumerate(seed_times)
            if w0 - EPS <= time <= w0 + window_sec + EPS
        )
        if not covered:
            continue
        key = (-len(covered), covered)
        if best_key is None or key < best_key:
            best_key, best_covered = key, covered
    if best_key is None:
        return None

    times = [seed_times[index] for index in best_covered]
    # Feasible starts that keep every covered seed inside the window.
    feasible_low = max(low, max(times) - window_sec)
    feasible_high = min(high, min(times))
    if feasible_high < feasible_low:
        feasible_high = feasible_low
    centered = (min(times) + max(times)) / 2.0 - window_sec / 2.0
    window_start = min(max(centered, feasible_low), feasible_high)
    dropped = tuple(index for index in range(len(seed_times)) if index not in set(best_covered))
    return WindowPlan(window_start=window_start, covered=best_covered, dropped=dropped)


def seed_frame_index(
    seed_time: float,
    window_start: float,
    fps: int = FPS,
    num_frames: int = WINDOW_FRAMES,
) -> int:
    """Frame number (0..num_frames-1) of a seed time inside the window."""
    index = int(round((seed_time - window_start) * fps))
    return min(max(index, 0), num_frames - 1)


def sample_id_for(uuid: str, window_start: float) -> str:
    """Filename-safe deterministic id: ``<koala_uuid>_w<window start in ms>``.

    Phantom video ids carry float seconds (dots), which we cannot put in filenames used
    by downstream stages, so the window start is encoded as zero-padded milliseconds.
    Unique per (source video, window start) which is exactly the sample identity.
    """
    milliseconds = int(round(window_start * 1000.0))
    return f"{uuid}_w{milliseconds:09d}"


def source_order_key(video_id: str, seed: int) -> str:
    """Deterministic, seed-dependent shuffle key for a source video / row id."""
    return hashlib.sha256(f"{seed}:{video_id}".encode()).hexdigest()
