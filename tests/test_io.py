"""Tests for the shared IO helpers.

Only :func:`atomic_write_bytes` has behaviour worth pinning down, and it is the one function here
whose failure mode is silent: a partial write leaves an artefact that reads as corrupt far from the
run that produced it. The rest is thin enough that the tests are really guarding the contract
(blank lines skipped, alpha dropped) rather than the arithmetic.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from phantom_data.io import atomic_write_bytes, read_jsonl


def test_blank_lines_are_skipped_rather_than_parsed(tmp_path):
    # A trailing newline is normal in a manifest written line by line; json.loads("") raises.
    path = tmp_path / "m.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert [row["a"] for row in read_jsonl(path)] == [1, 2]


def test_an_empty_manifest_reads_as_no_rows(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert read_jsonl(path) == []


def test_a_malformed_line_raises_rather_than_being_skipped(tmp_path):
    # Loud is right here: a manifest row that will not parse means the producer wrote garbage, and
    # silently dropping it would shrink the dataset without telling anyone.
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n{not json\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_writing_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "deep" / "nested" / "out.json"
    atomic_write_bytes(target, b"{}")
    assert target.read_bytes() == b"{}"


def test_writing_replaces_an_existing_file(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_bytes(target, b"old")
    atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"


def test_a_failed_write_leaves_no_temp_file_behind(tmp_path):
    # The reason for the try/except: an abandoned .tmp beside the artefact would be picked up by
    # the next glob over that directory.
    target = tmp_path / "out.json"

    class Boom:
        """Not bytes-like, so sink.write raises the way a full disk would."""

    with pytest.raises(TypeError):
        atomic_write_bytes(target, Boom())
    assert not target.exists()
    assert list(tmp_path.glob(".*tmp")) == []


def test_the_written_file_is_never_partially_visible(tmp_path):
    # os.replace is atomic, so a reader either sees the old bytes or all the new ones. Approximated
    # here by checking the temp file is gone and the target is complete.
    target = tmp_path / "out.json"
    payload = json.dumps({"k": list(range(1000))}).encode()
    atomic_write_bytes(target, payload)
    assert target.read_bytes() == payload
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]
