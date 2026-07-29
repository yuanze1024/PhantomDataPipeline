"""Turn a Phantom caption into one short phrase that identifies the subject.

Phantom labels each subject with a noun phrase that is usually too thin to identify it:
65% of subjects carry ``<=2`` words (``woman``, ``fish``), which cannot distinguish one
individual of a class from another. The clip caption does have the detail -- median 207
words -- but CLIP truncates at 77 tokens, so the caption cannot be used as the scoring text
directly. This module asks an LLM to compress it to one phrase.

The output field is ``dis``: adjectives plus the class noun, ending on the noun
(``man wearing blue puffer jacket and glasses``). It is used for **both** jobs downstream --
as the Grounding DINO query and as the CLIP text -- so every score in the pipeline is against
the same words.

Model choice is empirical, not a preference. Measured over repeated identical requests:

===================================================  ==============  ==========
request                                              thinking leaks  completion
===================================================  ==============  ==========
``deepseek-v4-flash`` + ``thinking:{disabled}``      6/12            median 372
``deepseek-v4-flash`` + ``reasoning_effort:none``    5/12            median 34
``qwen3.5-flash`` with no flag                       10/10           median 2604
``qwen3.5-flash`` + ``enable_thinking:false``        **0/10**        **25, flat**
===================================================  ==============  ==========

``thinking:{"type":"disabled"}`` is DeepSeek's own documented field, so the leak is the
gateway not honouring it rather than a wrong parameter -- which is why the fix was to change
model, not to keep tuning flags. Hence :data:`MODEL` and the unconditional
``enable_thinking: False`` in :func:`build_payload`.

Two operational constraints, both learned the hard way:

* **The gateway must be reached directly.** Through the lab proxy 5/20 requests died with
  ``RemoteDisconnected``; direct it was 6/6. :func:`_direct_opener` therefore builds an
  opener with proxies explicitly disabled rather than trusting the ambient environment.
* **JSON extraction needs a greedy regex.** A reply containing ``'I love my life'`` inside a
  value truncates under ``\\{.*?\\}`` and reads as a parse failure. :func:`extract_json`
  tries ``json.loads`` first and falls back to a *greedy* match.

Failure never blocks a run: :func:`enrich_subject` falls back to Phantom's own phrase and
records ``text_source="phantom_fallback"``, so a dead gateway degrades the texts instead of
losing the sample.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

#: Chosen by measurement (see the module docstring), not by preference.
MODEL = "qwen3.5-flash"
API_BASE = "https://api.cometapi.com/v1"
#: Read from a file rather than an env var: the key must not end up in a process listing.
KEY_FILE = "/mnt/pfs/users/yuanze/projects/2026/BboxCondition/APIKEY_COMET"
KEY_ENV = "COMET_API_KEY"

#: The caption is the expensive part of the prompt and its tail is scene description, so it
#: is cut rather than sent whole. 2000 characters covers the subject introduction in every
#: caption inspected.
CAPTION_CHARS = 2000
MAX_TOKENS = 300
TIMEOUT = 120.0
RETRIES = 4

SOURCE_LLM = "llm"
SOURCE_FALLBACK = "phantom_fallback"

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
#: Words that mean the model described the shot instead of the subject. Cheap guard: the
#: phrase goes straight into a detector prompt, where "camera" finds a camera.
_SCENE_WORDS = ("camera", "background", "foreground", "the video", "the clip", "the scene",
                "close-up shot", "footage")

SYSTEM_PROMPT = (
    "You extract a single subject's visual identity from a video caption. "
    "Reply with one JSON object and nothing else. No markdown, no explanation."
)

USER_TEMPLATE = """Caption of a video:
{caption}

The subject of interest is annotated as: "{phrase}"

Return JSON with exactly one key:
{{"dis": "..."}}

dis: that subject's class noun preceded by its most distinctive visible appearance attributes.
Aim for at most 8 words. It MUST contain the class noun, ideally ending on it
(e.g. "man wearing blue puffer jacket and glasses", not "blue puffer jacket").

Describe only the subject's own appearance. Never mention the camera, the background, the shot, or other objects."""


# --------------------------------------------------------------------------------------
# pure helpers (unit tested)
# --------------------------------------------------------------------------------------


def build_prompt(caption: str, phrase: str, caption_chars: int = CAPTION_CHARS) -> str:
    """The user message. Truncation happens here so the tests can pin the boundary."""
    text = " ".join(str(caption or "").split())[:caption_chars]
    return USER_TEMPLATE.format(caption=text or "(no caption available)",
                                phrase=str(phrase or "").strip())


def build_payload(caption: str, phrase: str, model: str = MODEL,
                  max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    """Chat-completions body. ``enable_thinking`` is the field that keeps the reply short."""
    return {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_prompt(caption, phrase)}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
        # Measured: without this the same prompt emits ~2600 completion tokens of thinking
        # and the JSON arrives (or does not) behind it.
        "enable_thinking": False,
    }


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse the model's reply, tolerating prose or a code fence around the object.

    The fallback regex is **greedy** on purpose: a non-greedy ``\\{.*?\\}`` stops at the
    first ``}``, which truncates any reply whose values contain braces or quoted text and
    turns a correct answer into a reported failure.
    """
    body = str(text or "").strip()
    if not body:
        return None
    for candidate in (body, *(m.group(0) for m in [_JSON_OBJECT.search(body)] if m)):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().strip('"').rstrip(".")


def validate(parsed: dict[str, Any] | None) -> dict[str, str] | None:
    """Return the cleaned phrase, or None if the reply is unusable.

    Two checks only, and the ones left out matter as much as the ones kept:

    * **Non-empty.** An empty detector query returns every object in the frame.
    * **No scene words.** ``dis`` goes straight into the detector query, so a phrase saying
      ``camera`` makes it hunt for a camera. Cheap guard against the model describing the shot
      instead of the subject.

    Deliberately *not* checked: the ``<=8 words`` the prompt asks for. Measured over 138 real
    replies, 63 exceed it (up to 12 words) and read fine -- the cap is guidance for brevity,
    and enforcing it would reject 46% of good data.

    An earlier version also asked for a bare class name and required ``dis`` to contain it,
    meaning to block attribute-only answers like ``blue puffer jacket``. It was removed as
    ineffective: both fields came from the same reply, so a model that misread the subject got
    both wrong together and passed (``{"det": "jacket", "dis": "blue puffer jacket"}`` was
    accepted), and across those 138 replies it never once rejected anything.
    """
    if not isinstance(parsed, dict):
        return None
    dis = _clean(parsed.get("dis"))
    if not dis:
        return None
    if any(word in dis.lower() for word in _SCENE_WORDS):
        return None
    return {"dis": dis, "text_source": SOURCE_LLM}


def fallback(phrase: str, ref_phrase: str = "") -> dict[str, str]:
    """What to use when the gateway will not answer: Phantom's own phrase, unembellished."""
    text = _clean(phrase) or _clean(ref_phrase) or "object"
    return {"dis": text, "text_source": SOURCE_FALLBACK}


def cache_name(sample_id: str, subject_id: int) -> str:
    return f"{sample_id}_subj{int(subject_id):02d}.json"


# --------------------------------------------------------------------------------------
# network layer (stubbed in tests)
# --------------------------------------------------------------------------------------


def read_key(key_file: str = KEY_FILE, env: str = KEY_ENV) -> str:
    """Credential from the environment, else from the key file. Never logged."""
    key = (os.environ.get(env) or "").strip()
    if key:
        return key
    path = Path(key_file)
    if not path.is_file():
        raise RuntimeError(f"no comet credential: set ${env} or create {key_file}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"{key_file} is empty")
    return key


def _direct_opener():
    """An opener with proxying disabled.

    Measured: through the lab proxy 5/20 of these requests died with
    ``RemoteDisconnected``; direct, 6/6 succeeded. Ambient ``https_proxy`` in the shell
    would otherwise be picked up silently.
    """
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post_chat(payload: dict[str, Any], key: str, api_base: str = API_BASE,
              timeout: float = TIMEOUT) -> dict[str, Any]:
    endpoint = api_base.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST")
    with _direct_opener().open(request, timeout=timeout) as response:
        return json.loads(response.read())


def message_text(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"].get("content") or "")
    except (KeyError, IndexError, TypeError):
        return ""


def call_llm(caption: str, phrase: str, key: str, *, model: str = MODEL,
             api_base: str = API_BASE, timeout: float = TIMEOUT,
             poster: Callable[..., dict[str, Any]] = post_chat) -> dict[str, str] | None:
    """One request, parsed and validated. None if the reply was unusable."""
    response = poster(build_payload(caption, phrase, model=model), key,
                      api_base=api_base, timeout=timeout)
    return validate(extract_json(message_text(response)))


def enrich_subject(caption: str, phrase: str, ref_phrase: str = "", *, key: str | None = None,
                   model: str = MODEL, api_base: str = API_BASE, retries: int = RETRIES,
                   poster: Callable[..., dict[str, Any]] = post_chat,
                   sleeper: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Enrich one subject, with exponential backoff, falling back rather than raising.

    A subject with no usable texts would stall the whole 140-subject pass, so every failure
    path ends in :func:`fallback` with ``text_source`` recording what happened.
    """
    credential = key if key is not None else read_key()
    last_error = ""
    for attempt in range(max(1, retries)):
        try:
            result = call_llm(caption, phrase, credential, model=model, api_base=api_base,
                              poster=poster)
            if result is not None:
                return result
            last_error = "unusable reply"
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError,
                KeyError) as error:
            # Bare type name only. The message can echo the request, which carries the key.
            last_error = type(error).__name__
        if attempt < max(1, retries) - 1:
            sleeper(2.0 ** attempt)
    result = fallback(phrase, ref_phrase)
    result["error"] = last_error
    return result


def cached_enrich(cache_dir: Path, sample_id: str, subject_id: int, caption: str, phrase: str,
                  ref_phrase: str = "", **kwargs: Any) -> dict[str, Any]:
    """Enrich once and keep the answer on disk, so a re-run costs nothing.

    A cached fallback is *not* honoured: it means the gateway was down, and re-running is
    exactly when it is worth asking again.
    """
    path = Path(cache_dir) / cache_name(sample_id, subject_id)
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            cached = None
        if isinstance(cached, dict) and cached.get("text_source") == SOURCE_LLM:
            cached["cache_hit"] = True
            return cached
    result = enrich_subject(caption, phrase, ref_phrase, **kwargs)
    result["cache_hit"] = False
    if result.get("text_source") == SOURCE_LLM:
        from .inspect import atomic_write_bytes

        atomic_write_bytes(path, (json.dumps(result, ensure_ascii=False, indent=2)
                                  + "\n").encode("utf-8"))
    return result
