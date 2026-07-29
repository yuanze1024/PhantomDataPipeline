"""Storage backends for stage-B artifacts.

The pilot writes to a local directory. At scale the clips must live on BOS (local FS is
tight), so every write goes through this interface and callers never build paths
themselves: they ask for a *relative* dataset path and hand over bytes.

A backend is responsible for (a) persisting bytes at a dataset-relative path and (b)
telling callers whether that path already exists. Marker files stay local in every
backend because resumability is driven by ``ultravid_pipeline.state.MarkerStore``.
"""
from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import Protocol

#: Bucket/prefix for the BOS backend. Both are env-overridable with ``${VAR-default}``
#: semantics (``environ.get(name, default)``, never ``or default``) so an explicitly empty
#: ``PHANTOM_BOS_PREFIX=`` writes at the bucket root instead of being silently overridden.
DEFAULT_BOS_BUCKET = "vast-yz"
DEFAULT_BOS_PREFIX = "koala-ref-n-box"

#: Width of the shard directory inserted into per-sample keys. See :func:`bos_key`.
SHARD_WIDTH = 2

#: Where dataset-level files (manifests, summaries) land. Also recognised as an *input*
#: directory, so ``manifests/extracted.jsonl`` is left flat rather than sharded.
FLAT_DIRECTORY = "manifests"


def check_relative_path(relative_path: str) -> Path:
    """Reject traversal, absolute and directory-shaped paths; returns the parsed path.

    Shared by both backends: the sample ids come from an upstream annotation table, so a
    malformed id must not be able to escape the dataset root (local) or the key prefix
    (BOS). Kept as a module-level function so the BOS key mapping can be tested without a
    client.

    The trailing-slash case is checked on the raw string on purpose. ``Path`` normalises
    ``"clips/"`` to ``"clips"``, which is indistinguishable from a bare filename by the time
    it is parsed -- so it would sail through as a dataset-level file and be written as an
    *object* named ``clips`` (or ``manifests/clips`` on BOS). Nothing legitimate asks for a
    directory, so reject rather than guess.
    """
    if not relative_path or relative_path.endswith(("/", os.sep)):
        raise ValueError(f"unsafe relative path: {relative_path!r} (not an object)")
    path = Path(relative_path)
    if ".." in path.parts or path.is_absolute() or not path.name:
        raise ValueError(f"unsafe relative path: {relative_path!r}")
    return path


class StorageBackend(Protocol):
    """Sink for dataset-relative artifact paths (``clips/xx.mp4``, ``ref_frames/..jpg``)."""

    @property
    def root_uri(self) -> str:
        """Human readable location, for logs and manifests."""

    def exists(self, relative_path: str) -> bool: ...

    def write_bytes(self, relative_path: str, payload: bytes) -> str:
        """Persist ``payload``; returns the absolute/URI location actually written."""


class LocalStorage:
    """Writes under a local dataset root, atomically (tmp file + rename)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def root_uri(self) -> str:
        return str(self.root)

    def _absolute(self, relative_path: str) -> Path:
        return self.root / check_relative_path(relative_path)

    def exists(self, relative_path: str) -> bool:
        return self._absolute(relative_path).is_file()

    def write_bytes(self, relative_path: str, payload: bytes) -> str:
        path = self._absolute(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as sink:
                sink.write(payload)
                sink.flush()
                os.fsync(sink.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return str(path)


def bos_key(relative_path: str, prefix: str = DEFAULT_BOS_PREFIX) -> str:
    """Map a dataset-relative path to a BOS key, sharding per-sample artifacts.

    Two-hex-digit sharding, taken from the leading chars of the *filename* (sample ids look
    like ``784cdb6812944b028c70ee5ac14ef6ad_w000050258``, i.e. a 32-hex uuid followed by the
    window offset, so the first two chars are uniform hex)::

        clips/784cdb...ad_w000050258.mp4  ->  <prefix>/clips/78/784cdb...ad_w000050258.mp4
        ref_frames/784cdb..._subj01.jpg   ->  <prefix>/ref_frames/78/784cdb..._subj01.jpg
        extracted.jsonl                   ->  <prefix>/manifests/extracted.jsonl

    Why shard at all, given BOS has a flat keyspace and no directories to ``ls``: the pain
    is ``list_objects``, which pages at 1000 keys. At the 126k-sample target an unsharded
    ``clips/`` prefix is 126 pages per full listing, and every resume/audit scan pays that.
    256 shards puts ~500 objects under each ``clips/<xx>/``, i.e. one page per shard.

    The mapping is *total* by construction: a path with a directory component is sharded, a
    bare filename is a dataset-level file and goes flat under ``manifests/``. There is no
    silent third case -- anything already addressed at ``manifests/...`` is passed through
    unchanged rather than sharded, so callers can name manifests either way.
    """
    # check_relative_path already guarantees a non-empty filename, so ``path.name[:2]`` below
    # cannot silently produce an empty shard.
    path = check_relative_path(relative_path)
    parts = path.parts

    if len(parts) == 1:
        # Dataset-level file (manifest/summary): flat, never sharded. These are a handful of
        # objects and are always fetched by exact key, so sharding would only obscure them.
        key_parts = [FLAT_DIRECTORY, path.name]
    elif parts[0] == FLAT_DIRECTORY:
        key_parts = list(parts)
    else:
        shard = path.name[:SHARD_WIDTH]
        key_parts = [*parts[:-1], shard, path.name]

    if prefix:
        key_parts.insert(0, prefix)
    return "/".join(key_parts)


class BosStorage:
    """Writes stage-B artifacts to a BOS bucket under a sharded key prefix.

    Credentials and the client come from :mod:`phantom_data.bos` (``PHANTOM_BOS_AKSK`` env
    or the repo-root ``BOS_AKSK``); nothing is duplicated here. Note the read path in that
    module only ever generated presigned URLs -- write access to ``vast-yz`` was verified
    separately (put + head with matching size/md5 + delete, from an in-cluster pod).

    Two operational facts worth keeping in mind:

    * BOS egress needs the http(s) proxy env vars UNSET. Inside the training/build pods the
      proxy is exported by default and every request then fails; the launcher must clear
      ``http_proxy``/``https_proxy``/``all_proxy`` (and the uppercase spellings).
    * Unlike :class:`LocalStorage` there is no tmp-file-and-rename dance: a BOS ``PUT`` is
      atomic per object, so a torn write cannot be observed. A *failed* write can leave
      nothing at all, which is exactly what ``exists`` should report.
    """

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str | None = None,
        client=None,
        max_attempts: int = 4,
        base_delay: float = 0.5,
    ) -> None:
        self.bucket = os.environ.get("PHANTOM_BOS_BUCKET", DEFAULT_BOS_BUCKET) if bucket is None else bucket
        self.prefix = os.environ.get("PHANTOM_BOS_PREFIX", DEFAULT_BOS_PREFIX) if prefix is None else prefix
        self.max_attempts = max(1, max_attempts)
        self.base_delay = base_delay
        self._client = client

    @property
    def client(self):
        """Lazily built so constructing the backend needs no credentials (and no network).

        The import is deferred for the same reason: ``phantom_data.bos`` pulls decord at
        module level, which the pure-logic tests must not require.
        """
        if self._client is None:
            from ..bos import load_aksk, make_client

            # Validate before publishing to self._client: assigning first would make the
            # check fire only once, so a second access would hand back the very client that
            # just failed validation and every later call would run against a bad bucket.
            client = make_client(*load_aksk())
            self._check_bucket(client)
            self._client = client
        return self._client

    def _check_bucket(self, client) -> None:
        """Fail loudly if the bucket is not reachable.

        Takes the client as an argument rather than reading ``self._client``: it runs before
        the client is published, so a failed check cannot leave a usable backend behind.

        Needed because a HEAD against a *nonexistent bucket* also answers a bodyless 404,
        which ``exists`` cannot tell apart from a missing key (measured: both surface as
        ``BceHttpClientError(status_code=404, code=None)`` -- ``code`` is empty since a HEAD
        carries no ``NoSuchBucket``/``NoSuchKey`` body). Without this check a typo in
        ``PHANTOM_BOS_BUCKET`` would report every sample as absent and a resume would
        re-extract the whole dataset before the first PUT told anyone.
        """
        if not client.does_bucket_exist(self.bucket):
            raise ValueError(
                f"BOS bucket {self.bucket!r} is not reachable with these credentials "
                f"(check PHANTOM_BOS_BUCKET / PHANTOM_BOS_AKSK)"
            )

    @property
    def root_uri(self) -> str:
        return f"bos://{self.bucket}/{self.prefix}" if self.prefix else f"bos://{self.bucket}"

    def key_for(self, relative_path: str) -> str:
        return bos_key(relative_path, self.prefix)

    def _uri(self, key: str) -> str:
        return f"bos://{self.bucket}/{key}"

    def exists(self, relative_path: str) -> bool:
        """True/False from a HEAD; a missing key is False, anything else raises.

        The distinction matters more than it looks: resume decides whether to redo work off
        this answer, so a connection blip reported as "absent" would silently re-extract,
        and (worse) a blip reported as "present" would skip a sample forever. Absence is
        only ever inferred from an HTTP 404 -- observed shape for a missing key is
        ``BceHttpClientError(status_code=404, code=None)``; ``code`` is None because a HEAD
        has no response body to carry ``NoSuchKey``, so matching on it would never fire.
        """
        key = self.key_for(relative_path)
        try:
            self.client.get_object_meta_data(self.bucket, key)
        except Exception as error:  # noqa: BLE001 - re-raised unless it is a clean 404
            if _status_code(error) == 404:
                return False
            raise
        return True

    def write_bytes(self, relative_path: str, payload: bytes) -> str:
        """PUT ``payload``; retries transient failures, returns the ``bos://`` URI.

        Retry covers 5xx and connection-level errors (``BceHttpClientError`` wrapping a
        socket failure has no ``status_code``), with exponential backoff plus jitter so a
        worker pool does not resynchronise onto the same retry instant. 4xx is not retried:
        AccessDenied or an invalid key will not fix itself, and burning three more attempts
        per sample across 126k samples is a real cost. Exhausting the attempts re-raises, so
        the caller records a failed marker instead of a passing one over missing bytes.

        Note this layer *stacks* on the sdk's own retries: the default client config carries
        ``BackOffRetryPolicy(max_error_retry=3, base_interval_in_millis=300)``, so a genuinely
        dead endpoint costs up to ``max_attempts * 4`` HTTP attempts before the exception
        escapes. That is tolerable per sample but it is why ``max_attempts`` is 4, not 10 --
        and why the extract stage's per-sample timeout stays the real backstop.
        """
        key = self.key_for(relative_path)
        delay = self.base_delay
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.client.put_object_from_string(self.bucket, key, payload)
                return self._uri(key)
            except Exception as error:  # noqa: BLE001 - classified, then retried or raised
                status = _status_code(error)
                retryable = status is None or status >= 500
                if not retryable or attempt == self.max_attempts:
                    raise
                time.sleep(delay * (1.0 + random.random()))
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover


def _status_code(error: BaseException) -> int | None:
    """HTTP status off a bce exception, or None when the failure never got a response.

    ``BceHttpClientError`` carries ``status_code`` as a plain attribute for server errors
    and wraps the underlying exception in ``last_error`` for transport failures (where there
    is no status at all). It is a string in some sdk paths, hence the int() coercion.
    """
    for candidate in (error, getattr(error, "last_error", None)):
        status = getattr(candidate, "status_code", None)
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                return None
    return None


def make_storage(kind: str, root: str | Path) -> StorageBackend:
    """Backend factory.

    ``root`` is the local dataset directory and is ignored by ``bos``, which is addressed
    entirely by ``PHANTOM_BOS_BUCKET``/``PHANTOM_BOS_PREFIX``. Callers still need the local
    root either way: markers and the merged manifest are written there by stage B directly.
    """
    if kind == "local":
        return LocalStorage(root)
    if kind == "bos":
        return BosStorage()
    raise ValueError(f"unknown storage backend: {kind!r} (available: local, bos)")
