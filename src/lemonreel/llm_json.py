"""Robust JSON extraction from LLM replies.

Small local models are good at JSON but not perfect: they wrap it in code fences, add a
sentence before it, leave trailing commas, or emit a reasoning block first. These helpers
recover the object without asking the model again when possible.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_LINE_COMMENT_RE = re.compile(r"^\s*//.*$", re.MULTILINE)


class JSONExtractionError(ValueError):
    pass


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks, including an unterminated one at the start."""
    text = _THINK_RE.sub("", text or "")
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _balanced_span(text: str, start: int) -> str | None:
    """Return the balanced {...} or [...] substring starting at ``start``."""
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair(candidate: str) -> str:
    fixed = candidate.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    fixed = _LINE_COMMENT_RE.sub("", fixed)
    fixed = _TRAILING_COMMA_RE.sub(r"\1", fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    return fixed


def _candidates(text: str) -> list[str]:
    found: list[str] = []
    for match in _FENCE_RE.finditer(text):
        found.append(match.group(1).strip())
    for i, ch in enumerate(text):
        if ch in "{[":
            span = _balanced_span(text, i)
            if span:
                found.append(span)
                if ch == "{":
                    break
    stripped = text.strip()
    if stripped:
        found.append(stripped)
    return found


def extract_json(text: str) -> Any:
    """Parse the first JSON object (or array) found in an LLM reply."""
    cleaned = strip_reasoning(text)
    errors: list[str] = []
    for candidate in _candidates(cleaned):
        for attempt in (candidate, _repair(candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
    snippet = cleaned[:200].replace("\n", " ")
    detail = errors[-1] if errors else "no JSON found"
    raise JSONExtractionError(f"Could not parse JSON from model reply ({detail}): {snippet!r}")


def extract_json_object(text: str) -> dict[str, Any]:
    value = extract_json(text)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict):
        raise JSONExtractionError(f"Expected a JSON object, got {type(value).__name__}")
    return value
