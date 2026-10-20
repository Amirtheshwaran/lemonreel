"""``lemonreel say``: turn a written script into a voiceover with the Omni model's voice.

Useful for narrating a screen recording or any video you edit yourself. The script is split
into parts on blank lines; each part is spoken with the same lexicon and listen-back checks
as a reel, then the parts are joined with a pause. Outputs, in one folder:

* ``voiceover.wav``: the whole narration
* ``part-NN.wav``: one file per part, to line up with clips in a video editor
* ``voiceover.srt``: captions timed from the listener's word timestamps
* ``timing.txt``: where each part starts and how long it runs
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lemonreel.captions import Cue, build_cues, to_srt
from lemonreel.config import ReelConfig
from lemonreel.lemonade import LemonadeClient
from lemonreel.lexicon import Lexicon, default_learned_path
from lemonreel.media import concat_wavs
from lemonreel.omni import Toolkit
from lemonreel.voice import ClipResult, Narrator

Log = Callable[[str], None]

MAX_CHUNK_CHARS = 450  # Kokoro sounds best on a few sentences at a time
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


_HEADING_LINE_RE = re.compile(r"^\s{0,3}#.*$", re.MULTILINE)
_DIRECTION_RE = re.compile(r"\[[^\]]*\](?!\()")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_EMPHASIS_RE = re.compile(r"[*_`]+")


def clean_script(text: str) -> str:
    """Drop what shouldn't be read aloud.

    ``# Heading`` lines are section labels and ``[bracketed notes]`` are stage directions
    ("[show the terminal]"), so one file can serve as both teleprompter and voiceover script.
    Markdown links keep their text; emphasis markers are removed.
    """
    text = _HEADING_LINE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _DIRECTION_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    return text


def read_script(source: str) -> str:
    """Script text from a file, or the argument itself when it isn't a file."""
    path = Path(source).expanduser()
    raw = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else source
    return clean_script(raw)


def split_parts(text: str) -> list[str]:
    """Parts are separated by blank lines; whitespace inside a part is collapsed."""
    parts = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return [p for p in parts if p]


def split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Break a long part at sentence ends so each chunk stays under ``limit`` characters."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > limit:
            cut = sentence.rfind(", ", 0, limit)
            cut = cut + 1 if cut > limit // 3 else sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _stamp(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def make_voiceover(
    client: LemonadeClient,
    toolkit: Toolkit,
    cfg: ReelConfig,
    script: str,
    folder: Path,
    pause: float = 0.7,
    log: Log = print,
) -> Voiceover:
    parts_text = split_parts(script)
    if not parts_text:
        raise ValueError("The script is empty.")
    folder.mkdir(parents=True, exist_ok=True)
    learned = default_learned_path() if cfg.learn else None
    lexicon = Lexicon.build(
        cfg.base_language,
        learned_path=learned,
        extra_path=Path(cfg.lexicon_path) if cfg.lexicon_path else None,
    )
    narrator = Narrator(client, toolkit, cfg, lexicon, log=log, learned_path=learned)
    voice = cfg.voices[0]
    work = folder / "takes"

    part_files: list[Path] = []
    part_clips: list[list[ClipResult]] = []
    part_layouts = []
    for i, text in enumerate(parts_text, 1):
        chunks = split_long(text)
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
