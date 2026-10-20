"""Pronunciation lexicon applied to text before it is spoken.

Captions always show the original spelling. Only the copy sent to the text-to-speech
model is rewritten, so "ROCm" can be read as "Rock M" while the screen still says ROCm.

Respellings come from three places, later ones winning:
1. the built-in list below (common AI and hardware terms),
2. the user's learned lexicon (``~/.lemonreel/lexicon.json``), which LemonReel grows
   automatically whenever the listen-back loop fixes a word,
3. a lexicon file passed on the command line.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

BUILTIN_LEXICON: dict[str, str] = {
    "ROCm": "Rock M",
    "iGPU": "integrated G P U",
    "dGPU": "discrete G P U",
    "LLMs": "L L Ms",
    "NPUs": "N P Us",
    "GPUs": "G P Us",
    "CPUs": "C P Us",
    "GGUF": "G G U F",
    "ONNX": "onyx",
    "VRAM": "V RAM",
    "XDNA": "X D N A",
    "RDNA": "R D N A",
    "Ryzen": "Rye zen",
    "Radeon": "Ray dee on",
    "llama.cpp": "llama C P P",
    "whisper.cpp": "whisper C P P",
    "JSON": "jay son",
    "YAML": "yamel",
    "e.g.": "for example",
    "i.e.": "that is",
    "vs.": "versus",
    "tok/s": "tokens per second",
}

# Units read after a number. Single-letter units must touch the number ("4B", not "4 B")
# so ordinary sentences are left alone.
_SPACED_UNITS = {
    "GB": "gigabytes",
    "MB": "megabytes",
    "TB": "terabytes",
    "GHz": "gigahertz",
    "MHz": "megahertz",
    "ms": "milliseconds",
}
_TIGHT_UNITS = {"B": "billion", "M": "million"}
_UNITS = {**_SPACED_UNITS, **_TIGHT_UNITS}
_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)(?:\s?("
    + "|".join(sorted(_SPACED_UNITS, key=len, reverse=True))
    + r")|("
    + "|".join(_TIGHT_UNITS)
    + r"))\b"
)
_URL_RE = re.compile(r"https?://\S+")

_lock = threading.Lock()


def default_learned_path() -> Path:
    return Path.home() / ".lemonreel" / "lexicon.json"


def load_lexicon_file(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}


def remember(path: Path, respellings: dict[str, str]) -> None:
    """Merge newly learned respellings into a lexicon file (thread-safe, best effort)."""
    if not respellings:
        return
    with _lock:
        current = load_lexicon_file(path)
        current.update(respellings)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            pass


def _pattern(term: str) -> re.Pattern[str]:
    # Terms with punctuation (e.g. "llama.cpp") cannot rely on \b at both ends.
    left = r"(?<![\w.])" if term[0].isalnum() else ""
    right = r"(?![\w])" if term[-1].isalnum() else ""
    flags = 0 if any(c.isupper() for c in term) else re.IGNORECASE
    return re.compile(left + re.escape(term) + right, flags)


class Lexicon:
    """An ordered set of respellings that can be applied to text."""

    def __init__(self, entries: dict[str, str] | None = None, language: str = "en"):
        self.language = language
        self.entries: dict[str, str] = {}
        self._compiled: list[tuple[re.Pattern[str], str]] = []
        if entries:
            self.update(entries)

    def update(self, entries: dict[str, str]) -> None:
        self.entries.update({k: v for k, v in entries.items() if k and v})
        # Longest terms first so "LLMs" wins over "LLM".
        self._compiled = [
            (_pattern(term), spoken)
            for term, spoken in sorted(self.entries.items(), key=lambda kv: -len(kv[0]))
        ]

    def apply(self, text: str) -> str:
        """Return the text the speech model should read aloud."""
        spoken = _URL_RE.sub("", text)
        if self.language == "en":
            spoken = _UNIT_RE.sub(
                lambda m: f"{m.group(1)} {_UNITS[m.group(2) or m.group(3)]}", spoken
            )
        for pattern, replacement in self._compiled:
            spoken = pattern.sub(replacement, spoken)
        return re.sub(r"\s{2,}", " ", spoken).strip()

    @classmethod
    def build(
        cls,
        language: str = "en",
        learned_path: Path | None = None,
        extra_path: Path | None = None,
    ) -> Lexicon:
        lex = cls(language=language)
        if language == "en":
            lex.update(BUILTIN_LEXICON)
        if learned_path is not None:
            lex.update(load_lexicon_file(learned_path))
        if extra_path is not None:
            lex.update(load_lexicon_file(extra_path))
        return lex
