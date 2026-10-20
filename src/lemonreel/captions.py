"""Caption cues timed from the speech recognizer's word timestamps.

The words on screen are always the script's own words. Their timing comes from what the
listener model actually heard: script and transcript are aligned word by word, matched
words take the recognizer's timestamps, and anything in between is interpolated. Without a
listener, timing falls back to an even spread by character count.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from lemonreel.textmetrics import CHAR_LEVEL_LANGUAGES, tokenize_with_owners


@dataclass
class Cue:
    start: float
    end: float
    text: str


TimedWord = tuple[str, float, float]


def timed_words(asr: dict) -> list[TimedWord]:
    """Extract (word, start, end) triples from a ``verbose_json`` transcription."""
    words: list[TimedWord] = []
    top_words = asr.get("words")
    segments = asr.get("segments") or []
    if isinstance(top_words, list) and top_words and not any(s.get("words") for s in segments):
        for w in top_words:
            if isinstance(w, dict) and str(w.get("word", "")).strip():
                words.append(
                    (str(w["word"]).strip(), float(w.get("start", 0)), float(w.get("end", 0)))
                )
        return words
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_words = seg.get("words")
        if isinstance(seg_words, list) and seg_words:
            for w in seg_words:
                text = str(w.get("word", "")).strip()
                if text:
                    words.append(
                        (
                            text,
                            float(w.get("start", seg.get("start", 0))),
                            float(w.get("end", seg.get("end", 0))),
                        )
                    )
            continue
        text = str(seg.get("text", "")).split()
        if not text:
            continue
        start, end = float(seg.get("start", 0)), float(seg.get("end", 0))
        total = sum(len(t) for t in text) or 1
        cursor = start
        for t in text:
            span = (end - start) * len(t) / total
            words.append((t, cursor, cursor + span))
            cursor += span
    return words


_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def chunk_text(text: str, max_chars: int, language: str = "en") -> list[list[int]]:
    """Group the word indices of ``text`` into caption-sized chunks.

    Returns a list of chunks, each a list of indices into ``text.split()``.
    """
    words = text.split()
    if not words:
        return []
    if language in CHAR_LEVEL_LANGUAGES and len(words) <= 2:
        return [list(range(len(words)))]
    chunks: list[list[int]] = []
    current: list[int] = []
    length = 0
    for i, word in enumerate(words):
        extra = len(word) + (1 if current else 0)
        if current and length + extra > max_chars:
            # Prefer to break after a comma in the second half of the chunk.
            cut = next(
                (
                    k
                    for k in range(len(current) - 1, len(current) // 2 - 1, -1)
                    if words[current[k]].endswith((",", ";", ":"))
                ),
                None,
            )
            if cut is not None and cut < len(current) - 1:
                chunks.append(current[: cut + 1])
                current = current[cut + 1 :]
                length = sum(len(words[k]) for k in current) + max(0, len(current) - 1)
            else:
                chunks.append(current)
                current, length = [], 0
            extra = len(word) + (1 if current else 0)
        current.append(i)
        length += extra
        if re.search(r"[.!?]$", word) and length > max_chars * 0.45:
            chunks.append(current)
            current, length = [], 0
    if current:
        chunks.append(current)
    return chunks


def _word_times(
    text: str, words: list[TimedWord], duration: float, language: str
) -> list[tuple[float, float]]:
    """Start/end time for every word of ``text``."""
    n_words = len(text.split())
    disp_tokens, disp_owner = tokenize_with_owners(text, language)
    asr_text = " ".join(w for w, _, _ in words)
    asr_tokens, asr_owner = tokenize_with_owners(asr_text, language)
    known: dict[int, list[float]] = {}
    matcher = difflib.SequenceMatcher(a=disp_tokens, b=asr_tokens, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            word_idx = disp_owner[block.a + k]
            _, start, end = words[asr_owner[block.b + k]]
            slot = known.setdefault(word_idx, [start, end])
            slot[0] = min(slot[0], start)
            slot[1] = max(slot[1], end)

    times: list[tuple[float, float] | None] = [None] * n_words
    for idx, (start, end) in known.items():
        times[idx] = (start, max(start, end))
    anchors = [i for i, t in enumerate(times) if t is not None]
    if not anchors:
        return _proportional(text, duration)
    # Interpolate gaps between anchors (and before the first / after the last).
    lengths = [len(w) + 1 for w in text.split()]

    def fill(lo: int, hi: int, t0: float, t1: float) -> None:
        span = sum(lengths[lo:hi]) or 1
        cursor = t0
        for j in range(lo, hi):
            step = (t1 - t0) * lengths[j] / span
            times[j] = (cursor, cursor + step)
            cursor += step

    first, last = anchors[0], anchors[-1]
    # Unmatched words before the first / after the last anchor are placed at the speaking
    # rate measured on the matched stretch, not stretched to the clip edges.
    span = times[last][1] - times[first][0]
    rate = span / max(1, sum(lengths[first : last + 1])) if span > 0 else 0.07
    lead_start = max(0.0, times[first][0] - rate * sum(lengths[:first]))
    tail_end = min(max(duration, times[last][1]), times[last][1] + rate * sum(lengths[last + 1 :]))
    fill(0, first, lead_start, times[first][0])
    for a, b in zip(anchors, anchors[1:]):
        if b - a > 1:
            fill(a + 1, b, times[a][1], times[b][0])
    fill(last + 1, n_words, times[last][1], tail_end)
    return [t if t is not None else (0.0, 0.0) for t in times]


def _proportional(text: str, duration: float) -> list[tuple[float, float]]:
    lengths = [len(w) + 1 for w in text.split()]
    total = sum(lengths) or 1
    out, cursor = [], 0.0
    for length in lengths:
        step = duration * length / total
        out.append((cursor, cursor + step))
        cursor += step
    return out


def build_cues(
    text: str,
    duration: float,
    words: list[TimedWord] | None = None,
    language: str = "en",
    max_chars: int = 84,
    offset: float = 0.0,
) -> list[Cue]:
    """Caption cues for ``text`` spoken over ``duration`` seconds, shifted by ``offset``."""
    split = text.split()
    if not split or duration <= 0:
        return []
    times = _word_times(text, words, duration, language) if words else _proportional(text, duration)
    cues: list[Cue] = []
    joiner = "" if language in CHAR_LEVEL_LANGUAGES else " "
    for chunk in chunk_text(text, max_chars, language):
        start = times[chunk[0]][0]
        end = times[chunk[-1]][1]
        cues.append(Cue(start=start, end=end, text=joiner.join(split[k] for k in chunk)))
    # Tidy up: monotonic, no overlaps, no flicker, inside [0, duration].
    for i, cue in enumerate(cues):
        cue.start = max(0.0, min(cue.start, duration))
        if i > 0 and cue.start < cues[i - 1].end:
            cues[i - 1].end = cue.start
        cue.end = min(duration, max(cue.end, cue.start + 0.8))
    for i in range(len(cues) - 1):
        if cues[i + 1].start - cues[i].end < 0.6:
            cues[i].end = cues[i + 1].start
    if cues:
        cues[-1].end = min(duration, cues[-1].end + 0.3)
    return [Cue(c.start + offset, c.end + offset, c.text) for c in cues if c.end > c.start]


def _timestamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{i}\n{_timestamp(c.start, ',')} --> {_timestamp(c.end, ',')}\n{c.text}\n"
        for i, c in enumerate(cues, 1)
    ]
    return "\n".join(blocks)


def to_vtt(cues: list[Cue]) -> str:
    body = "\n".join(
        f"{_timestamp(c.start, '.')} --> {_timestamp(c.end, '.')}\n{c.text}\n" for c in cues
    )
    return "WEBVTT\n\n" + body
