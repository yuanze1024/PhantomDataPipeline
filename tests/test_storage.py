"""Pure tests for :mod:`phantom_data.build.storage` key layout. No network, no credentials.

The BOS key mapping is worth testing on its own because it is the one part of the backend
that cannot be checked by looking at a bucket after the fact: a wrong shard silently
scatters a sample's artifacts across prefixes, and resume (which asks ``exists`` for the
same relative path) would then answer "absent" forever and re-extract every sample.

:func:`bos_key` is deliberately a module-level pure function so all of this runs without a
``BosClient`` -- constructing one needs AK/SK and would drag decord in through
``phantom_data.bos``.
"""
from __future__ import annotations

import pytest

from phantom_data.build.storage import (
    DEFAULT_BOS_BUCKET,
    DEFAULT_BOS_PREFIX,
    BosStorage,
    LocalStorage,
    bos_key,
    check_relative_path,
    make_storage,
)

SAMPLE = "784cdb6812944b028c70ee5ac14ef6ad_w000050258"


# ----- sharding -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative,expected",
    [
        (f"clips/{SAMPLE}.mp4", f"koala-ref-n-box/clips/78/{SAMPLE}.mp4"),
        (f"bbox/{SAMPLE}.json", f"koala-ref-n-box/bbox/78/{SAMPLE}.json"),
        (f"ref_frames/{SAMPLE}_subj01.jpg", f"koala-ref-n-box/ref_frames/78/{SAMPLE}_subj01.jpg"),
        (f"object_reference/{SAMPLE}_subj01.jpg",
         f"koala-ref-n-box/object_reference/78/{SAMPLE}_subj01.jpg"),
    ],
)
def test_per_sample_paths_get_a_two_hex_shard(relative: str, expected: str) -> None:
    assert bos_key(relative) == expected


def test_shard_comes_from_the_filename_not_the_directory() -> None:
    """The shard must be a property of the sample, so every artifact of one sample agrees.

    Deriving it from anything else (a counter, the directory) would put ``clips/`` and
    ``ref_frames/`` of the same sample under different shards, which breaks the audit story:
    a per-shard listing could no longer be checked for completeness on its own.
    """
    keys = [
        bos_key(f"clips/{SAMPLE}.mp4"),
        bos_key(f"ref_frames/{SAMPLE}_subj01.jpg"),
        bos_key(f"ref_frames/{SAMPLE}_subj02.jpg"),
    ]
    assert {key.split("/")[-2] for key in keys} == {"784cdb6812944b028c70ee5ac14ef6ad_w000050258"[:2]}


def test_shard_width_is_two_hex_digits() -> None:
    # 256 shards is the number the layout is sized around: ~500 objects per prefix at the
    # 126k-sample target, i.e. one 1000-key list_objects page.
    shard = bos_key(f"clips/{SAMPLE}.mp4").split("/")[-2]
    assert len(shard) == 2
    assert all(char in "0123456789abcdef" for char in shard)


def test_nested_directories_keep_their_parents_and_shard_last() -> None:
    assert bos_key(f"extra/masks/{SAMPLE}.png") == f"koala-ref-n-box/extra/masks/78/{SAMPLE}.png"


# ----- flat manifests -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["extracted.jsonl", "specs.jsonl", "extract.summary.json"])
def test_bare_filenames_go_flat_under_manifests(name: str) -> None:
    assert bos_key(name) == f"koala-ref-n-box/manifests/{name}"


def test_paths_already_under_manifests_are_not_sharded_or_doubled() -> None:
    assert bos_key("manifests/extracted.jsonl") == "koala-ref-n-box/manifests/extracted.jsonl"


def test_mapping_is_total_shard_or_flat_with_no_third_case() -> None:
    """Every accepted path either lands under ``manifests/`` or gains a shard directory.

    Asserted explicitly because the silent third case is the dangerous one: a path that fell
    through both branches would be written at an unsharded key that ``exists`` still
    computes the same way, so nothing would look broken until a listing came up short.
    """
    paths = [
        f"clips/{SAMPLE}.mp4",
        f"bbox/{SAMPLE}.json",
        f"ref_frames/{SAMPLE}_subj01.jpg",
        f"extra/masks/{SAMPLE}.png",
        "extracted.jsonl",
        "manifests/extracted.jsonl",
        "weird",
    ]
    for path in paths:
        key = bos_key(path)
        parts = key.split("/")
        assert parts[0] == "koala-ref-n-box"
        flat = parts[1] == "manifests"
        sharded = len(parts[-2]) == 2 and parts[-2] == parts[-1][:2]
        assert flat != sharded, f"{path!r} -> {key!r} is neither purely flat nor sharded"


# ----- prefix handling ----------------------------------------------------------------


def test_empty_prefix_writes_at_the_bucket_root() -> None:
    # ``${VAR-default}`` semantics: an explicitly empty PHANTOM_BOS_PREFIX is honoured.
    assert bos_key(f"clips/{SAMPLE}.mp4", prefix="") == f"clips/78/{SAMPLE}.mp4"


def test_custom_prefix_is_used_verbatim() -> None:
    assert bos_key("extracted.jsonl", prefix="scratch/v2") == "scratch/v2/manifests/extracted.jsonl"


# ----- unsafe paths -------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "../escape.mp4",
        f"clips/../../{SAMPLE}.mp4",
        f"/abs/clips/{SAMPLE}.mp4",
        "..",
    ],
)
def test_unsafe_paths_are_rejected_by_both_backends(relative: str) -> None:
    with pytest.raises(ValueError, match="unsafe relative path"):
        bos_key(relative)
    with pytest.raises(ValueError, match="unsafe relative path"):
        check_relative_path(relative)
    with pytest.raises(ValueError, match="unsafe relative path"):
        LocalStorage("/tmp/does-not-need-to-exist").exists(relative)


@pytest.mark.parametrize("relative", ["", "clips/"])
def test_empty_targets_are_rejected(relative: str) -> None:
    # ``clips/`` parses to a directory with no name; sharding it would slice an empty string.
    with pytest.raises(ValueError):
        bos_key(relative)


# ----- backend wiring -----------------------------------------------------------------


def test_bos_defaults_and_root_uri() -> None:
    storage = BosStorage()
    assert (storage.bucket, storage.prefix) == (DEFAULT_BOS_BUCKET, DEFAULT_BOS_PREFIX)
    assert storage.root_uri == "bos://vast-yz/koala-ref-n-box"
    assert storage.key_for(f"clips/{SAMPLE}.mp4") == f"koala-ref-n-box/clips/78/{SAMPLE}.mp4"


def test_bos_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHANTOM_BOS_BUCKET", "other-bucket")
    monkeypatch.setenv("PHANTOM_BOS_PREFIX", "trial/run7")
    storage = BosStorage()
    assert storage.root_uri == "bos://other-bucket/trial/run7"
    assert storage.key_for("extracted.jsonl") == "trial/run7/manifests/extracted.jsonl"


def test_bos_empty_prefix_env_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHANTOM_BOS_PREFIX", "")
    storage = BosStorage()
    assert storage.prefix == ""
    assert storage.root_uri == "bos://vast-yz"
    assert storage.key_for(f"clips/{SAMPLE}.mp4") == f"clips/78/{SAMPLE}.mp4"


def test_make_storage_knows_bos_and_still_rejects_junk(tmp_path) -> None:
    assert isinstance(make_storage("local", tmp_path), LocalStorage)
    assert isinstance(make_storage("bos", tmp_path), BosStorage)
    with pytest.raises(ValueError, match="unknown storage backend"):
        make_storage("s3", tmp_path)


def test_unreachable_bucket_keeps_raising_on_every_access() -> None:
    """A failed bucket check must not leave a usable backend behind.

    Regression: the validation used to run *after* ``self._client`` was assigned, so it
    fired only on the first access. Every later call then got the very client that had just
    failed validation -- meaning a typo in ``PHANTOM_BOS_BUCKET`` raised once, was retried,
    and from then on silently reported every sample as absent (see ``_check_bucket``).
    """
    class Unreachable:
        def does_bucket_exist(self, bucket: str) -> bool:
            return False

    storage = BosStorage(bucket="typo-bucket")
    # Repeated calls stand in for repeated lazy builds, without needing credentials or
    # decord. Every one must reject; the old code rejected only the first.
    for _ in range(3):
        with pytest.raises(ValueError, match="not reachable"):
            storage._check_bucket(Unreachable())
    assert storage._client is None, "a rejected client must never be published"
