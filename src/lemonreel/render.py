"""Assemble the final MP4 with ffmpeg.

Each scene becomes one segment: its picture with a slow Ken Burns move, its narration,
a headline badge, and caption overlays timed from the listen-back transcript. Segments
share one encoding profile, so they are joined without re-encoding, and chapters plus an
optional soft-subtitle track are added at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from lemonreel import graphics
from lemonreel.captions import Cue, to_srt
from lemonreel.media import run_ffmpeg

MOTIONS = ("zoom_in", "pan_right", "zoom_out", "pan_left", "rise")
AUDIO_RATE = 48_000


@dataclass
class SegmentSpec:
    image: Path
    audio: Path | None
    audio_duration: float
    cues: list[Cue] = field(default_factory=list)  # relative to the start of the audio
    badge: tuple[int, str] | None = None
    motion: str = "zoom_in"
    lead: float = 0.35
    tail: float = 0.65
    min_duration: float = 0.0
    chapter: str = ""

    @property
    def duration(self) -> float:
        return round(max(self.min_duration, self.lead + self.audio_duration + self.tail), 3)


@dataclass
class Timeline:
    video: Path
    duration: float
    starts: list[float]
    cues: list[Cue]
    chapters: list[tuple[float, str]]


def _zoompan(motion: str, frames: int, size: tuple[int, int], fps: int) -> str:
    w, h = size
    p = f"(on/{max(1, frames - 1)})"
    ease = f"({p}*{p}*(3-2*{p}))"  # smoothstep: gentle start and stop
    center_x, center_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    if motion == "zoom_out":
        z, x, y = f"1.12-0.12*{ease}", center_x, center_y
    elif motion == "pan_right":
        z, x, y = "1.12", f"(iw-iw/zoom)*{ease}", center_y
    elif motion == "pan_left":
        z, x, y = "1.12", f"(iw-iw/zoom)*(1-{ease})", center_y
    elif motion == "rise":
        z, x, y = f"1.07+0.05*{ease}", center_x, f"(ih-ih/zoom)*(1-{ease})"
    else:
        z, x, y = f"1+0.12*{ease}", center_x, center_y
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps}"


def _prepare_canvas(image: Path, size: tuple[int, int], dst: Path) -> Path:
    # Render the still at 2x so the sub-pixel motion of zoompan stays smooth.
    with Image.open(image) as img:
        canvas = graphics.fit_canvas(img, (size[0] * 2, size[1] * 2))
    canvas.save(dst, format="JPEG", quality=93)
    return dst


def _save_cropped(img: Image.Image, dst: Path) -> tuple[int, int]:
    """Save only the visible part of a full-frame overlay; return where it goes."""
    box = img.getbbox() or (0, 0, 1, 1)
    img.crop(box).save(dst)
    return box[0], box[1]


def render_segment(
    spec: SegmentSpec,
    out: Path,
    work: Path,
    size: tuple[int, int],
    fps: int = 30,
    burn_captions: bool = True,
    language: str = "en",
) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    duration = spec.duration
    frames = max(2, int(round(duration * fps)))
    canvas = _prepare_canvas(spec.image, size, work / f"{out.stem}-canvas.jpg")

    args: list[str] = ["-i", str(canvas)]
    if spec.audio is not None:
        args += ["-i", str(spec.audio)]
    else:
        args += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", f"anullsrc=r={AUDIO_RATE}:cl=mono"]
    # (input index, x, y, enable expression or "badge")
    overlays: list[tuple[int, int, int, str]] = []
    next_input = 2
    if spec.badge is not None:
        badge_png = work / f"{out.stem}-badge.png"
        x, y = _save_cropped(
            graphics.badge_overlay(spec.badge[0], spec.badge[1], size, language), badge_png
        )
        args += [
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-t",
            f"{duration:.3f}",
            "-i",
            str(badge_png),
        ]
        overlays.append((next_input, x, y, "badge"))
        next_input += 1
    if burn_captions:
        for i, cue in enumerate(spec.cues):
            cue_png = work / f"{out.stem}-cue{i:02d}.png"
            x, y = _save_cropped(graphics.caption_overlay(cue.text, size, language), cue_png)
            # Static captions only need a few frames per second; overlay repeats the latest.
            args += ["-loop", "1", "-framerate", "5", "-t", f"{duration:.3f}", "-i", str(cue_png)]
            start, end = spec.lead + cue.start, spec.lead + cue.end
            overlays.append((next_input, x, y, f"between(t,{start:.3f},{end:.3f})"))
            next_input += 1

    graph = [f"[0:v]{_zoompan(spec.motion, frames, size, fps)},setsar=1[v0]"]
    current = "v0"
    for n, (idx, x, y, enable) in enumerate(overlays, 1):
        if enable == "badge":
            fade_out = min(4.2, max(1.2, duration - 1.0))
            graph.append(
                f"[{idx}:v]format=rgba,fade=t=in:st=0.3:d=0.4:alpha=1,"
                f"fade=t=out:st={fade_out:.3f}:d=0.5:alpha=1[o{n}]"
            )
            graph.append(f"[{current}][o{n}]overlay={x}:{y}[v{n}]")
        else:
            graph.append(f"[{current}][{idx}:v]overlay={x}:{y}:enable='{enable}'[v{n}]")
        current = f"v{n}"
    fade = min(0.4, duration / 4)
    graph.append(
        f"[{current}]fade=t=in:st=0:d={fade:.3f},fade=t=out:st={duration - fade:.3f}:d={fade:.3f},"
        f"format=yuv420p[vout]"
    )
    delay_ms = int(spec.lead * 1000) if spec.audio is not None else 0
    graph.append(
        f"[1:a]aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=mono,"
        f"adelay={delay_ms}:all=1,apad,atrim=0:{duration:.3f},"
        f"afade=t=out:st={max(0.0, duration - 0.3):.3f}:d=0.3[aout]"
    )
    args += [
        "-filter_complex",
        ";".join(graph),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-t",
        f"{duration:.3f}",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        "1",
        str(out),
    ]
    run_ffmpeg(args)
    return out


def _ffmetadata(title: str, chapters: list[tuple[float, str]], total: float) -> str:
    def esc(text: str) -> str:
        for ch in ("\\", "=", ";", "#", "\n"):
            text = text.replace(ch, "\\" + ch)
        return text

    lines = [";FFMETADATA1", f"title={esc(title)}"]
    for i, (start, name) in enumerate(chapters):
        end = chapters[i + 1][0] if i + 1 < len(chapters) else total
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={int(start * 1000)}",
            f"END={int(end * 1000)}",
            f"title={esc(name)}",
        ]
    return "\n".join(lines) + "\n"


def assemble(
    segments: list[SegmentSpec],
    out: Path,
    work: Path,
    size: tuple[int, int],
    title: str,
    fps: int = 30,
    captions: str = "burn",
    language: str = "en",
    progress=None,
) -> Timeline:
    """Render every segment and join them into ``out``."""
    work.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    starts: list[float] = []
    cues: list[Cue] = []
    chapters: list[tuple[float, str]] = []
    position = 0.0
    burn = captions in ("burn", "both")
    for i, spec in enumerate(segments):
        if progress:
            progress(i, len(segments))
        seg_path = work / f"segment-{i:02d}.mp4"
        render_segment(spec, seg_path, work, size, fps, burn_captions=burn, language=language)
        files.append(seg_path)
        starts.append(position)
        if spec.chapter:
            chapters.append((position, spec.chapter))
        for cue in spec.cues:
            cues.append(
                Cue(position + spec.lead + cue.start, position + spec.lead + cue.end, cue.text)
            )
        position += spec.duration

    concat_list = work / "segments.txt"
    concat_list.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in files), encoding="utf-8"
    )
    meta = work / "chapters.ffmeta"
    meta.write_text(_ffmetadata(title, chapters, position), encoding="utf-8")
    args = ["-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", str(meta)]
    maps = ["-map", "0:v", "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
    codecs = ["-c", "copy"]
    if captions in ("soft", "both") and cues:
        srt = work / "soft.srt"
        srt.write_text(to_srt(cues), encoding="utf-8")
        args += ["-i", str(srt)]
        maps += ["-map", "2:s"]
        codecs += ["-c:s", "mov_text"]
    run_ffmpeg([*args, *maps, *codecs, "-movflags", "+faststart", str(out)])
    if progress:
        progress(len(segments), len(segments))
    return Timeline(
        video=out, duration=round(position, 3), starts=starts, cues=cues, chapters=chapters
    )


def youtube_chapters(chapters: list[tuple[float, str]]) -> str:
    """Chapter list in the format YouTube recognizes in video descriptions."""
    out = []
    for start, name in chapters:
        m, s = divmod(int(start), 60)
        h, m = divmod(m, 60)
        stamp = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        out.append(f"{stamp} {name}")
    return "\n".join(out) + "\n"
