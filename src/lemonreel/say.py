"""``lemonreel say``: turn a written script into a voiceover with the Omni model's voice.

Useful for narrating a screen recording or any video you edit yourself. The script is split
into parts on blank lines and headings (see :func:`parse_script`); each part is spoken with the
same lexicon and listen-back checks as a reel, then the parts are joined with a pause. Outputs,
in one folder:

* ``voiceover.wav``: the whole narration
* ``part-NN.wav``: one file per part, to line up with clips in a video editor
* ``voiceover.srt``: captions timed from the listener's word timestamps
* ``timing.txt``: where each part starts and how long it runs
* ``takes/``: the individual takes and the 16 kHz copies sent to the listener
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lemonreel.captions import Cue, build_cues, to_srt
from lemonreel.config import ReelConfig
from lemonreel.lemonade import LemonadeClient
from lemonreel.lexicon import Lexicon, default_learned_path
from lemonreel.media import concat_wavs
from lemonreel.omni import Toolkit
from lemonreel.textmetrics import CHAR_LEVEL_LANGUAGES, normalize_tokens
from lemonreel.voice import ClipResult, Narrator

Log = Callable[[str], None]

MAX_CHUNK_CHARS = 450  # Kokoro sounds best on a few sentences at a time
CJK_CHUNK_CHARS = 150  # the same few sentences in Japanese or Chinese
SENTENCE_GAP = 0.2


@dataclass
class Part:
    index: int
    text: str
    path: Path
    start: float
    duration: float
    wer: float | None
    fixes: dict[str, str]


@dataclass
class Voiceover:
    folder: Path
    wav: Path
    srt: Path
    timing: Path
    parts: list[Part]
    duration: float


class ScriptNotFound(ValueError):
    """The argument looks like a file name, but no such file exists."""


# Line-level patterns: these lines are never spoken.
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+.*)?$")  # section labels; also end a part
_DIRECTION_LINE_RE = re.compile(r"^[ \t]*\[[^\]\n]*\][ \t]*$")  # [show the terminal]
_RULE_RE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$")  # --- or ***
_LINK_DEF_RE = re.compile(r"^[ \t]*\[[^\]]+\]:[ \t]*\S.*$")  # [ref]: https://...
_BULLET_RE = re.compile(r"^[ \t]*[*+-][ \t]+")

# Inline patterns.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_REF_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_INLINE_DIRECTION_RE = re.compile(r"\[[^\]\n]*\]")
_CODE_RE = re.compile(r"`+([^`]*)`+")
# Paired emphasis markers at word edges only, so snake_case and 2*3 survive.
_EMPHASIS_RE = re.compile(r"(?<![\w*])(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1(?![\w*])")

_SCRIPT_SUFFIXES = (".md", ".markdown", ".txt", ".text")


def _clean_inline(text: str) -> str:
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _REF_LINK_RE.sub(r"\1", text)
    text = _INLINE_DIRECTION_RE.sub("", text)
    text = _CODE_RE.sub(r"\1", text)
    for _ in range(2):  # nested emphasis such as **_this_**
        text = _EMPHASIS_RE.sub(r"\2", text)
    return " ".join(text.split())


def parse_script(text: str) -> list[str]:
    """Split a script into spoken parts.

    Blank lines and ``# headings`` end a part. Lines that are only a ``[stage direction]``, a
    horizontal rule or a link definition are skipped without splitting the part around them.
    Inside a part, ``[notes]`` and images are dropped, links keep their text, and Markdown
    emphasis, code ticks and bullets are removed. ``snake_case`` and ``2*3`` are left alone.
    """
    blocks: list[list[str]] = [[]]
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip() or _HEADING_RE.match(line):
            blocks.append([])
        elif not (
            _DIRECTION_LINE_RE.match(line) or _RULE_RE.match(line) or _LINK_DEF_RE.match(line)
        ):
            blocks[-1].append(_BULLET_RE.sub("", line))
    parts = [_clean_inline(" ".join(block)) for block in blocks]
    return [part for part in parts if part]


def _existing_file(source: str) -> Path | None:
    try:
        path = Path(source).expanduser()
        return path if path.is_file() else None
    except (OSError, RuntimeError, ValueError):
        # Long typed-in text (name too long), "~" with no home directory, NUL bytes, ...
        return None


def _looks_like_path(source: str) -> bool:
    text = source.strip()
    if not text or len(text) > 255 or any(ch.isspace() for ch in text):
        return False
    return "/" in text or "\\" in text or text.lower().endswith(_SCRIPT_SUFFIXES)


def _decode(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig", errors="replace")


def resolve_script(source: str) -> tuple[str, Path | None]:
    """Return the script text and the file it came from (``None`` for typed-in text).

    An argument that names an existing file is read (UTF-8 with or without a byte-order mark,
    or UTF-16). Anything else is treated as the script itself, unless it looks like a file
    name, such as ``notes.md`` or ``scripts/demo``, in which case :class:`ScriptNotFound` is
    raised instead of reading the file name aloud.
    """
    path = _existing_file(source)
    if path is not None:
        return _decode(path.read_bytes()), path
    if _looks_like_path(source):
        raise ScriptNotFound(f"Script file not found: {source}")
    return source, None


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*")
_CLAUSE_SEPARATORS = (", ", "，", "、", "; ")


def split_long(text: str, limit: int = MAX_CHUNK_CHARS, joiner: str = " ") -> list[str]:
    """Break a long part at sentence ends so each chunk stays under ``limit`` characters.

    Sentences longer than ``limit`` are cut at a clause separator, then at a space, and only
    as a last resort mid-word. Use ``joiner=""`` for scripts written without spaces.
    """
    if len(text) <= limit:
        return [text]
    sentences = [s for s in _SENTENCE_END_RE.split(text) if s]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > limit:
            cut = max(
                (
                    sentence.rfind(sep, 0, limit) + len(sep)
                    for sep in _CLAUSE_SEPARATORS
                    if sentence.rfind(sep, 0, limit) > limit // 3
                ),
                default=-1,
            )
            if cut <= 0:
                cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if current and len(current) + len(joiner) + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current}{joiner}{sentence}" if current else sentence
    if current:
        chunks.append(current)
    return chunks


def _stamp(seconds: float) -> str:
    m, s = divmod(round(seconds, 1), 60)
    return f"{int(m)}:{s:04.1f}"


def _clear_previous(folder: Path) -> None:
    """Remove outputs of an earlier run so stale part files can't be mistaken for new ones."""
    for old in folder.glob("part-*.wav"):
        old.unlink()
    for name in ("voiceover.wav", "voiceover.srt", "timing.txt"):
        (folder / name).unlink(missing_ok=True)
    shutil.rmtree(folder / "takes", ignore_errors=True)


def make_voiceover(
    client: LemonadeClient,
    toolkit: Toolkit,
    cfg: ReelConfig,
    script: str,
    folder: Path,
    pause: float = 0.7,
    log: Log = print,
) -> Voiceover:
    learned = default_learned_path() if cfg.learn else None
    lexicon = Lexicon.build(
        cfg.base_language,
        learned_path=learned,
        extra_path=Path(cfg.lexicon_path) if cfg.lexicon_path else None,
    )
    char_level = cfg.base_language in CHAR_LEVEL_LANGUAGES
    limit, joiner = (CJK_CHUNK_CHARS, "") if char_level else (MAX_CHUNK_CHARS, " ")

    def speakable(text: str) -> bool:
        # A URL-only line, say, is empty once the lexicon has removed the URL.
        return bool(normalize_tokens(lexicon.apply(text), cfg.base_language))

    planned: list[tuple[str, list[str]]] = []
    for text in parse_script(script):
        chunks = [c for c in split_long(text, limit, joiner) if speakable(c)]
        if chunks:
            planned.append((text, chunks))
    if not planned:
        raise ValueError("Nothing in the script can be spoken.")

    folder.mkdir(parents=True, exist_ok=True)
    _clear_previous(folder)
    narrator = Narrator(client, toolkit, cfg, lexicon, log=log, learned_path=learned)
    voice = cfg.voices[0]
    work = folder / "takes"

    parts_text = [text for text, _ in planned]
    part_files: list[Path] = []
    part_clips: list[list[ClipResult]] = []
    part_layouts = []
    for i, (_, chunks) in enumerate(planned, 1):
        clips = [
            narrator.speak(chunk, "A", voice, work / f"part-{i:02d}-{k:02d}.wav", f"Part {i}")
            for k, chunk in enumerate(chunks, 1)
        ]
        part_path = folder / f"part-{i:02d}.wav"
        layout = concat_wavs([Path(c.path) for c in clips], part_path, gap=SENTENCE_GAP)
        part_files.append(part_path)
        part_clips.append(clips)
        part_layouts.append(layout)
        log(f"Part {i}: {layout.total:.1f}s")

    full = concat_wavs(part_files, folder / "voiceover.wav", gap=pause)
    parts: list[Part] = []
    cues: list[Cue] = []
    for i, (text, clips, layout, start) in enumerate(
        zip(parts_text, part_clips, part_layouts, full.starts), 1
    ):
        for clip, offset in zip(clips, layout.starts):
            cues += build_cues(
                clip.text, clip.duration, clip.words or None, cfg.base_language, 84, start + offset
            )
        checked = [c for c in clips if c.final.wer is not None]
        wer = (
            sum(c.final.wer * len(c.text.split()) for c in checked)
            / max(1, sum(len(c.text.split()) for c in checked))
            if checked
            else None
        )
        fixes: dict[str, str] = {}
        for clip in clips:
            for take in clip.takes:
                if take.helped:
                    fixes.update(take.respellings)
        parts.append(Part(i, text, folder / f"part-{i:02d}.wav", start, layout.total, wer, fixes))

    srt = folder / "voiceover.srt"
    srt.write_text(to_srt(cues), encoding="utf-8")
    timing = folder / "timing.txt"
    lines = [f"Voiceover: {_stamp(full.total)} total, voice {voice}", ""]
    for p in parts:
        preview = p.text if len(p.text) <= 70 else p.text[:67] + "..."
        lines.append(
            f"part-{p.index:02d}.wav  starts {_stamp(p.start)}  lasts {p.duration:4.1f}s  {preview}"
        )
    timing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Voiceover(
        folder=folder,
        wav=folder / "voiceover.wav",
        srt=srt,
        timing=timing,
        parts=parts,
        duration=full.total,
    )
