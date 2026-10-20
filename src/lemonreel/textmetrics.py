"""Text normalization, word error rate and alignment helpers.

These power the "listen back" loop: narration is synthesized, transcribed again by a
speech-recognition model, and the two texts are compared word by word.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]

# Languages written without spaces are compared character by character.
CHAR_LEVEL_LANGUAGES = {"ja", "zh"}


def number_to_words(n: int) -> str:
    """Spell out a non-negative integer in English ("128" -> "one hundred twenty eight")."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + ("" if rest == 0 else " " + _ONES[rest])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            return (
                number_to_words(head)
                + " "
                + name
                + ("" if rest == 0 else " " + number_to_words(rest))
            )
    return str(n)  # pragma: no cover - unreachable for ints


def _spell_number(token: str) -> str:
    """Spell a numeric token such as "3.5", "1,024" or "2026"."""
    token = token.replace(",", "")
    if "." in token:
        whole, _, frac = token.partition(".")
        whole_words = number_to_words(int(whole)) if whole else "zero"
        frac_words = " ".join(_ONES[int(d)] for d in frac if d.isdigit())
        return f"{whole_words} point {frac_words}".strip()
    if len(token) == 4 and 1100 <= int(token) <= 2099 and not token.endswith("00"):
        # Years are usually read in pairs: 2026 -> "twenty twenty six".
        head, tail = int(token[:2]), int(token[2:])
        if tail == 0:
            return number_to_words(int(token))
        tail_words = number_to_words(tail) if tail >= 10 else "oh " + _ONES[tail]
        return number_to_words(head) + " " + tail_words
    return number_to_words(int(token))


_SYMBOLS = {
    "&": " and ",
    "%": " percent ",
    "+": " plus ",
    "@": " at ",
    "=": " equals ",
    "/": " ",
    "-": " ",
    "_": " ",
    "–": " ",
    "—": " ",
}
# Spellings a recognizer may legitimately produce for the same spoken word.
_EQUIVALENT = {
    "gb": "gigabytes",
    "gigabyte": "gigabytes",
    "mb": "megabytes",
    "tb": "terabytes",
    "ghz": "gigahertz",
    "mhz": "megahertz",
    "ms": "milliseconds",
    "ok": "okay",
    "vs": "versus",
    "percentage": "percent",
}
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_word(word: str, language: str = "en") -> list[str]:
    """Normalize one whitespace-separated word into zero or more comparison tokens."""
    word = word.lower().replace("’", "'")
    if language in CHAR_LEVEL_LANGUAGES:
        return [c for c in word if c.isalnum()]
    for symbol, spoken in _SYMBOLS.items():
        word = word.replace(symbol, spoken)
    if language == "en":
        word = _NUMBER_RE.sub(lambda m: " " + _spell_number(m.group(0)) + " ", word)
        word = _strip_accents(word)
    tokens = [t.replace("'", "") for t in _WORD_RE.findall(word)]
    return [_EQUIVALENT.get(t, t) for t in tokens if t]


def _collapse_spelled_letters(tokens: list[str], owners: list[int]) -> tuple[list[str], list[int]]:
    """Join runs of single letters, so "n p u" and "npu" compare equal."""
    out: list[str] = []
    out_owners: list[int] = []
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
            j += 1
        if j - i >= 2:
            out.append("".join(tokens[i:j]))
            out_owners.append(owners[i])
            i = j
        else:
            out.append(tokens[i])
            out_owners.append(owners[i])
            i += 1
    return out, out_owners


def tokenize_with_owners(text: str, language: str = "en") -> tuple[list[str], list[int]]:
    """Return normalized tokens and, for each token, the index of the source word it came from."""
    tokens: list[str] = []
    owners: list[int] = []
    for idx, word in enumerate(text.split()):
        for tok in normalize_word(word, language):
            tokens.append(tok)
            owners.append(idx)
    if language not in CHAR_LEVEL_LANGUAGES:
        tokens, owners = _collapse_spelled_letters(tokens, owners)
    return tokens, owners


def normalize_tokens(text: str, language: str = "en") -> list[str]:
    return tokenize_with_owners(text, language)[0]


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str, language: str = "en") -> float:
    """Word error rate of ``hypothesis`` against ``reference`` (0.0 is a perfect match)."""
    ref = normalize_tokens(reference, language)
    hyp = normalize_tokens(hypothesis, language)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


@dataclass(frozen=True)
class Mismatch:
    """A stretch of the reference that the recognizer heard differently."""

    expected: str  # original words from the reference text
    heard: str  # normalized tokens the recognizer produced instead ("" if dropped)


def find_mismatches(reference: str, hypothesis: str, language: str = "en") -> list[Mismatch]:
    """List the reference words that were misheard or dropped, using original spellings."""
    import difflib

    words = reference.split()
    ref, owners = tokenize_with_owners(reference, language)
    hyp = normalize_tokens(hypothesis, language)
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    result: list[Mismatch] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("replace", "delete"):
            continue
        word_ids = sorted(set(owners[i1:i2]))
        expected = " ".join(words[k].strip(".,;:!?\"'()[]") for k in word_ids)
        if expected:
            result.append(Mismatch(expected=expected, heard=" ".join(hyp[j1:j2])))
    return result
