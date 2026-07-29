"""Pure tests for :mod:`phantom_data.enrich`. No HTTP: the poster is always stubbed.

The load-bearing ones:

* :func:`test_extract_json_survives_quoted_text_inside_a_value` -- the greedy-regex fix. A
  real reply containing ``'I love my life'`` was mis-reported as a parse failure by the
  non-greedy version, and the wasted retry cost money.
* :func:`test_payload_always_disables_thinking` -- the single measured reason this model was
  chosen. If the flag ever drops out, the reply grows ~100x and the JSON stops arriving.
* :func:`test_cached_enrich_retries_a_cached_fallback` -- a cached failure must not be
  mistaken for a cached answer, otherwise one gateway outage poisons the dataset for good.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phantom_data import enrich

CAPTION = ("A bearded man in a green jacket walks along a pier while gulls circle overhead. "
           "The camera pans right across the water.")


def reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def poster_for(*contents: str):
    """A stub poster that returns each content in turn and records the payloads it saw."""
    calls: list[dict] = []
    queue = list(contents)

    def poster(payload, key, **kwargs):
        calls.append(payload)
        return reply(queue.pop(0) if queue else queue_last(contents))

    def queue_last(items):
        return items[-1] if items else ""

    poster.calls = calls
    return poster


GOOD = '{"dis": "bearded man in green jacket"}'


# ----- prompt / payload ---------------------------------------------------------------


def test_build_prompt_includes_the_phantom_phrase_and_caption() -> None:
    text = enrich.build_prompt(CAPTION, "man")
    assert '"man"' in text
    assert "bearded man in a green jacket" in text


def test_build_prompt_truncates_a_long_caption() -> None:
    prompt = enrich.build_prompt("aaa " * 5000, "man", caption_chars=100)
    # The caption contributes exactly the cap and no more: 100 chars of "aaa " is 25 tokens.
    assert prompt.count("aaa") == 25
    assert len(prompt) < 1500  # nowhere near the 20000-char input


def test_build_prompt_collapses_whitespace() -> None:
    assert "a  b" not in enrich.build_prompt("a  \n b", "man")


def test_build_prompt_handles_a_missing_caption() -> None:
    assert "(no caption available)" in enrich.build_prompt("", "man")


def test_payload_always_disables_thinking() -> None:
    """The one measured reason for this model choice; 0/10 leaks with it, 10/10 without."""
    assert enrich.build_payload(CAPTION, "man")["enable_thinking"] is False


def test_payload_is_deterministic_and_bounded() -> None:
    payload = enrich.build_payload(CAPTION, "man")
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == enrich.MAX_TOKENS
    assert payload["model"] == enrich.MODEL
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


# ----- extract_json -------------------------------------------------------------------


def test_extract_json_reads_a_bare_object() -> None:
    assert enrich.extract_json(GOOD)["dis"] == "bearded man in green jacket"


def test_extract_json_survives_quoted_text_inside_a_value() -> None:
    """The measured bug: a non-greedy ``\\{.*?\\}`` truncates here and loses a good reply."""
    raw = ('Here you go: {"det": "woman", "dis": "woman in pink t-shirt with '
           "'I love my life'\", \"prompt\": \"woman cleaning the floor\"}")
    parsed = enrich.extract_json(raw)
    assert parsed is not None
    assert parsed["dis"] == "woman in pink t-shirt with 'I love my life'"


def test_extract_json_reads_through_a_code_fence() -> None:
    assert enrich.extract_json(f"```json\n{GOOD}\n```")["dis"].startswith("bearded man")


def test_extract_json_reads_through_leading_prose() -> None:
    assert enrich.extract_json(f"Sure!\n\n{GOOD}")["dis"].startswith("bearded man")


def test_extract_json_handles_a_nested_object() -> None:
    raw = '{"dis": "bearded man", "meta": {"a": 1}}'
    assert enrich.extract_json(raw)["meta"] == {"a": 1}


@pytest.mark.parametrize("raw", ["", "   ", "no json at all", "{broken", "[1, 2, 3]",
                                 '"just a string"', None])
def test_extract_json_returns_none_when_there_is_no_object(raw) -> None:
    assert enrich.extract_json(raw) is None


# ----- validate -----------------------------------------------------------------------


def test_validate_accepts_a_good_reply() -> None:
    assert enrich.validate(enrich.extract_json(GOOD)) == {
        "dis": "bearded man in green jacket", "text_source": "llm"}


def test_validate_strips_punctuation_and_whitespace() -> None:
    assert enrich.validate({"dis": " bearded  man. "})["dis"] == "bearded man"


def test_validate_accepts_a_reply_longer_than_the_eight_words_asked_for() -> None:
    """Measured: 63 of 138 real replies exceed 8 words and read fine. The cap is guidance
    for brevity, not a rejection criterion -- enforcing it would drop 46% of good data."""
    long_phrase = "small light gray French Bulldog with wrinkled face and large expressive eyes"
    assert len(long_phrase.split()) > 8
    assert enrich.validate({"dis": long_phrase}) is not None


def test_validate_accepts_attributes_before_the_noun() -> None:
    """The removed det check claimed to reject attribute-only phrases but never did: both
    fields came from one reply, so a misread subject got both wrong and passed anyway."""
    assert enrich.validate({"dis": "blue puffer jacket"}) is not None


@pytest.mark.parametrize("dis", ["man in front of the camera", "man against the background",
                                 "man in the video"])
def test_validate_rejects_shot_description(dis) -> None:
    """This phrase becomes the detector query, where "camera" makes it hunt for a camera."""
    assert enrich.validate({"dis": dis}) is None


@pytest.mark.parametrize("parsed", [None, {}, {"dis": ""}, {"dis": "   "},
                                    {"other": "man"}, "not a dict"])
def test_validate_rejects_unusable_replies(parsed) -> None:
    assert enrich.validate(parsed) is None


# ----- fallback -----------------------------------------------------------------------


def test_fallback_uses_the_phantom_phrase() -> None:
    assert enrich.fallback("black cow") == {"dis": "black cow",
                                            "text_source": "phantom_fallback"}


def test_fallback_uses_the_reference_phrase_when_the_target_has_none() -> None:
    assert enrich.fallback("", "white horse")["dis"] == "white horse"


def test_fallback_never_returns_empty_text() -> None:
    """An empty detector query returns every object in the frame, so it must not happen."""
    assert enrich.fallback("", "")["dis"] == "object"


# ----- enrich_subject -----------------------------------------------------------------


def test_enrich_subject_returns_the_first_good_reply() -> None:
    poster = poster_for(GOOD)
    result = enrich.enrich_subject(CAPTION, "man", key="k", poster=poster, sleeper=lambda s: None)
    assert result["text_source"] == "llm" and len(poster.calls) == 1


def test_enrich_subject_retries_an_unusable_reply() -> None:
    poster = poster_for("garbage", GOOD)
    result = enrich.enrich_subject(CAPTION, "man", key="k", poster=poster, sleeper=lambda s: None)
    assert result["dis"] == "bearded man in green jacket" and len(poster.calls) == 2


def test_enrich_subject_retries_a_network_error() -> None:
    calls: list[int] = []

    def poster(payload, key, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("connection reset")
        return reply(GOOD)

    result = enrich.enrich_subject(CAPTION, "man", key="k", poster=poster,
                                   sleeper=lambda s: None)
    assert result["text_source"] == "llm" and len(calls) == 3


def test_enrich_subject_backs_off_exponentially() -> None:
    waits: list[float] = []

    def poster(payload, key, **kwargs):
        raise OSError("down")

    enrich.enrich_subject(CAPTION, "man", key="k", poster=poster, retries=4,
                          sleeper=waits.append)
    assert waits == [1.0, 2.0, 4.0]  # no sleep after the final attempt


def test_enrich_subject_falls_back_rather_than_raising() -> None:
    """One dead subject must not stall a 140-subject pass."""
    def poster(payload, key, **kwargs):
        raise OSError("down")

    result = enrich.enrich_subject(CAPTION, "black cow", key="k", poster=poster, retries=2,
                                   sleeper=lambda s: None)
    assert result["text_source"] == "phantom_fallback" and result["dis"] == "black cow"


def test_enrich_subject_error_field_names_no_detail() -> None:
    """The exception message can echo the request, which carries the credential."""
    def poster(payload, key, **kwargs):
        raise OSError(f"failed with Bearer {key}")

    result = enrich.enrich_subject(CAPTION, "man", key="sk-secret", poster=poster, retries=1,
                                   sleeper=lambda s: None)
    assert result["error"] == "OSError"
    assert "secret" not in json.dumps(result)


# ----- cached_enrich ------------------------------------------------------------------


def test_cached_enrich_writes_then_reads_the_cache(tmp_path: Path) -> None:
    poster = poster_for(GOOD)
    first = enrich.cached_enrich(tmp_path, "sampleA", 0, CAPTION, "man", key="k",
                                poster=poster, sleeper=lambda s: None)
    second = enrich.cached_enrich(tmp_path, "sampleA", 0, CAPTION, "man", key="k",
                                 poster=poster, sleeper=lambda s: None)
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert second["dis"] == first["dis"]
    assert len(poster.calls) == 1  # the second call cost nothing


def test_cached_enrich_keys_on_subject_not_just_sample(tmp_path: Path) -> None:
    poster = poster_for(GOOD, GOOD)
    enrich.cached_enrich(tmp_path, "sampleA", 0, CAPTION, "man", key="k", poster=poster,
                         sleeper=lambda s: None)
    enrich.cached_enrich(tmp_path, "sampleA", 1, CAPTION, "man", key="k", poster=poster,
                         sleeper=lambda s: None)
    assert len(poster.calls) == 2
    assert {p.name for p in tmp_path.glob("*.json")} == {"sampleA_subj00.json",
                                                         "sampleA_subj01.json"}


def test_cached_enrich_does_not_cache_a_fallback(tmp_path: Path) -> None:
    def poster(payload, key, **kwargs):
        raise OSError("down")

    enrich.cached_enrich(tmp_path, "sampleA", 0, CAPTION, "man", key="k", poster=poster,
                         retries=1, sleeper=lambda s: None)
    assert list(tmp_path.glob("*.json")) == []


def test_cached_enrich_retries_a_cached_fallback(tmp_path: Path) -> None:
    """A gateway outage must not become a permanent bad label for the sample."""
    path = tmp_path / enrich.cache_name("sampleA", 0)
    path.write_text(json.dumps(enrich.fallback("man")), encoding="utf-8")
    poster = poster_for(GOOD)
    result = enrich.cached_enrich(tmp_path, "sampleA", 0, CAPTION, "man", key="k",
                                 poster=poster, sleeper=lambda s: None)
    assert result["text_source"] == "llm" and len(poster.calls) == 1


def test_cached_enrich_ignores_a_corrupt_cache_file(tmp_path: Path) -> None:
    (tmp_path / enrich.cache_name("sampleA", 0)).write_text("{not json", encoding="utf-8")
    poster = poster_for(GOOD)
    result = enrich.cached_enrich(tmp_path, "sampleA", 0, CAPTION, "man", key="k",
                                 poster=poster, sleeper=lambda s: None)
    assert result["text_source"] == "llm"


# ----- credential ---------------------------------------------------------------------


def test_read_key_prefers_the_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(enrich.KEY_ENV, "sk-from-env")
    assert enrich.read_key(str(tmp_path / "missing")) == "sk-from-env"


def test_read_key_falls_back_to_the_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(enrich.KEY_ENV, raising=False)
    path = tmp_path / "key"
    path.write_text("sk-from-file\n", encoding="utf-8")
    assert enrich.read_key(str(path)) == "sk-from-file"


def test_read_key_raises_when_there_is_no_credential(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(enrich.KEY_ENV, raising=False)
    with pytest.raises(RuntimeError):
        enrich.read_key(str(tmp_path / "missing"))
