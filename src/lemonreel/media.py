"""ffmpeg discovery plus the small amount of audio handling LemonReel needs.

LemonReel ships with a known-good ffmpeg through ``imageio-ffmpeg``, so users on Windows
don't have to install anything. ``LEMONREEL_FFMPEG`` or a system ffmpeg can override it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SPEECH_RATE = 24_000  # Kokoro's native rate; every narration clip is normalized to it.
ASR_RATE = 16_000  # what Whisper expects


class FFmpegError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    override = os.environ.get("LEMONREEL_FFMPEG")
    if override:
        return override
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - fall back to whatever is on PATH
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise FFmpegError(
        "ffmpeg was not found. Reinstall LemonReel (it bundles ffmpeg via imageio-ffmpeg) "
        "or set LEMONREEL_FFMPEG to an ffmpeg executable."
    )


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise FFmpegError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def media_duration(path: Path) -> float:
    """Duration of any media file in seconds, read from ffmpeg's banner."""
    if path.suffix.lower() == ".wav":
        try:
            return wav_duration(path)
        except (wave.Error, EOFError):
            pass
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = _DURATION_RE.search(proc.stderr)
    if not match:
        raise FFmpegError(f"Could not read the duration of {path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def normalize_audio(src: Path, dst: Path, rate: int = SPEECH_RATE) -> Path:
    """Convert any audio file to 16-bit mono PCM WAV at ``rate``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(dst)])
    return dst


def write_speech_clip(audio: bytes, dst: Path) -> Path:
    """Store a TTS response as a normalized WAV clip (format-agnostic: wav, mp3, ...)."""
    raw = dst.with_suffix(".raw")
    raw.write_bytes(audio)
    try:
        normalize_audio(raw, dst, SPEECH_RATE)
    finally:
        raw.unlink(missing_ok=True)
    return dst


@dataclass
class ClipLayout:
    """Where each concatenated clip starts inside the combined audio, in seconds."""

    starts: list[float]
    durations: list[float]
    total: float


def concat_wavs(clips: list[Path], dst: Path, gap: float = 0.25) -> ClipLayout:
    """Join normalized WAV clips with a short silence between them."""
    starts: list[float] = []
    durations: list[float] = []
    position = 0.0
    params = None
    frames: list[bytes] = []
    for i, clip in enumerate(clips):
        with wave.open(str(clip), "rb") as w:
            if params is None:
                params = w.getparams()
            elif (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                params.framerate,
                params.nchannels,
                params.sampwidth,
            ):
                raise ValueError(f"{clip} has a different audio format; normalize it first")
            data = w.readframes(w.getnframes())
            duration = w.getnframes() / float(w.getframerate())
        if i > 0 and gap > 0:
            silence_frames = int(gap * params.framerate)
            frames.append(b"\x00" * silence_frames * params.sampwidth * params.nchannels)
            position += silence_frames / params.framerate
        starts.append(position)
        durations.append(duration)
        frames.append(data)
        position += duration
    if params is None:
        raise ValueError("No clips to join")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        for chunk in frames:
            out.writeframes(chunk)
    return ClipLayout(starts=starts, durations=durations, total=position)


def to_asr_wav(src: Path, dst: Path) -> bytes:
    """16 kHz mono copy of a clip, the format Whisper-class models expect."""
    normalize_audio(src, dst, ASR_RATE)
    return dst.read_bytes()
