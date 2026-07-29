"""Online BOS access: presigned URLs + decord frame grabs, no downloads to disk.

This module is only imported by the viewer, never by the pure-logic tests, so the
decord/bce imports live at top level here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient
from decord import VideoReader

ENDPOINT = "bj.bcebos.com"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_AKSK = REPO_ROOT / "BOS_AKSK"


def load_aksk(path: str | Path | None = None) -> tuple[str, str]:
    """Read (ak, sk) from ``PHANTOM_BOS_AKSK`` env, else the repo-root ``BOS_AKSK``."""
    if path is None:
        path = os.getenv("PHANTOM_BOS_AKSK", str(DEFAULT_AKSK))
    lines = [line.strip() for line in open(path).read().splitlines() if line.strip()][:2]
    ak, sk = lines
    return ak, sk


def make_client(ak: str, sk: str) -> BosClient:
    config = BceClientConfiguration(credentials=BceCredentials(ak, sk), endpoint=ENDPOINT)
    return BosClient(config)


class FrameGrabber:
    """Caches presigned URLs per (bucket, key) and VideoReaders per URL for reuse."""

    def __init__(self, client: BosClient, expiration_in_seconds: int = 3600) -> None:
        self.client = client
        self.expiration_in_seconds = expiration_in_seconds
        self._url_cache: dict[tuple[str, str], str] = {}
        self._reader_cache: dict[str, VideoReader] = {}

    def presigned_url(self, bucket: str, key: str) -> str:
        cache_key = (bucket, key)
        if cache_key not in self._url_cache:
            url = self.client.generate_pre_signed_url(
                bucket, key, expiration_in_seconds=self.expiration_in_seconds
            )
            if isinstance(url, bytes):
                url = url.decode()
            self._url_cache[cache_key] = url
        return self._url_cache[cache_key]

    def reader(self, bucket: str, key: str) -> VideoReader:
        url = self.presigned_url(bucket, key)
        if url not in self._reader_cache:
            self._reader_cache[url] = VideoReader(url)
        return self._reader_cache[url]

    def video_dims(self, bucket: str, key: str) -> tuple[int, int]:
        """(W, H) of the actual decoded frame — the source of truth for bbox scaling."""
        reader = self.reader(bucket, key)
        H, W = reader[0].asnumpy().shape[:2]
        return W, H

    def grab(self, bucket: str, key: str, abs_time: float) -> np.ndarray:
        """Grab the frame nearest ``abs_time`` (seconds) as an HWC uint8 RGB array."""
        reader = self.reader(bucket, key)
        fps = reader.get_avg_fps()
        frame_no = min(int(round(abs_time * fps)), len(reader) - 1)
        return reader[frame_no].asnumpy()
