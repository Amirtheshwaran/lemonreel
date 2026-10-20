"""The end-to-end pipeline: source -> storyboard -> pictures + voices -> video -> report.

Pictures (image model) and narration (speech + recognition models) are produced in two
parallel lanes. Lemonade keeps one model of each type loaded at the same time, so on a
Ryzen AI PC the image model can work on the GPU while speech runs on the CPU or NPU.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from lemonreel import __version__, graphics
from lemonreel.captions import Cue, build_cues, to_srt, to_vtt
from lemonreel.config import ReelConfig
from lemonreel.ingest import SourceDoc, load_source
from lemonreel.lemonade import LemonadeClient
from lemonreel.lexicon import Lexicon, default_learned_path
from lemonreel.media import to_asr_wav
from lemonreel.omni import Toolkit, resolve_toolkit
from lemonreel.render import MOTIONS, SegmentSpec, Timeline, assemble, youtube_chapters
from lemonreel.report import write_report
from lemonreel.storyboard import Scene, Storyboard, write_storyboard
from lemonreel.visuals import ImageAttempt, SceneArt, draw_scene
from lemonreel.voice import ClipResult, Narrator, SceneVoice

STAGES = ("connect", "read", "write", "draw", "speak", "edit", "report")


class Cancelled(RuntimeError):
    pass


@dataclass
class Event:
    kind: str  # stage | log | toolkit | storyboard | art | voice | progress | done | error
    stage: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    t: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "stage": self.stage,
            "message": self.message,
            "data": self.data,
            "t": round(self.t, 3),
        }


@dataclass
class SceneOutput:
    art: SceneArt | None = None
    voice: SceneVoice | None = None
    cues: list[Cue] = field(default_factory=list)


@dataclass
class ReelResult:
    workdir: Path
    storyboard: Storyboard
    seconds: float
    video: Path | None = None
    timeline: Timeline | None = None
    report_html: Path | None = None
    report_json: Path | None = None


def slugify(text: str, limit: int = 48) -> str:
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    slug = re.sub(r"[\s_-]+", "-", text).strip("-")
    return slug[:limit].strip("-") or "reel"


def _hash(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Reel:
    """Runs one video job. Create one per job; it is not reusable."""

    def __init__(
        self,
        cfg: ReelConfig,
        client: LemonadeClient | None = None,
        *,
        base_url: str | None = None,
        out_root: Path | str = "reels",
        on_event: Callable[[Event], None] | None = None,
        cancel: threading.Event | None = None,
    ):
        self.cfg = cfg
        self.client = client or LemonadeClient.discover(base_url)
        self.out_root = Path(out_root)
        self.listeners: list[Callable[[Event], None]] = [on_event] if on_event else []
        self.cancel = cancel or threading.Event()
        self.events: list[Event] = []
        self.stage_times: dict[str, dict[str, float]] = {}
        self.devices: dict[str, dict[str, Any]] = {}
        self.server: dict[str, Any] = {}
        self.toolkit: Toolkit | None = None
        self.workdir: Path | None = None
        self._lock = threading.Lock()
        self._started = time.monotonic()

    # -- events ----------------------------------------------------------------------------

    def emit(self, kind: str, stage: str = "", message: str = "", **data: Any) -> None:
        event = Event(kind, stage, message, data, time.monotonic() - self._started)
        with self._lock:
            self.events.append(event)
            if self.workdir is not None:
                with open(self.workdir / "events.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        for listener in self.listeners:
            # A broken UI callback must never take the job down with it.
            with contextlib.suppress(Exception):
                listener(event)

    def log(self, message: str, stage: str = "") -> None:
        self.emit("log", stage, message)

    def _check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled("Cancelled")

    def _stage(self, name: str, message: str) -> Callable[[], None]:
        self._check()
        start = time.monotonic() - self._started
        self.stage_times[name] = {"start": start, "end": start}
        self.emit("stage", name, message, state="start")

        def finish() -> None:
            end = time.monotonic() - self._started
            self.stage_times[name]["end"] = end
            self.emit("stage", name, f"{name} done", state="end", seconds=round(end - start, 2))

        return finish

    def _rel(self, path: Path | str) -> str:
        assert self.workdir is not None
        return Path(path).resolve().relative_to(self.workdir.resolve()).as_posix()

    def _snapshot_devices(self) -> None:
        try:
            health = self.client.health()
        except Exception:  # noqa: BLE001
            return
        for m in health.get("all_models_loaded") or []:
            options = m.get("recipe_options") or {}
            backend = next((v for k, v in options.items() if k.endswith("_backend")), None)
            self.devices[str(m.get("model_name"))] = {
                "device": m.get("device"),
                "recipe": m.get("recipe"),
                "backend": backend,
                "type": m.get("type"),
            }

    # -- main ------------------------------------------------------------------------------

    def run(
        self,
        source: str | None = None,
        *,
        storyboard: Storyboard | Path | None = None,
        workdir: Path | None = None,
        storyboard_only: bool = False,
    ) -> ReelResult:
        """Make a video from ``source`` (or from an existing ``storyboard``).

        With ``storyboard_only`` the run stops once ``storyboard.json`` is written, so it can
        be edited by hand and finished later with ``workdir`` pointing at the same folder.
        """
        if source is None and storyboard is None:
            raise ValueError("Give a source or a storyboard")
        if workdir is not None:
            self.workdir = Path(workdir)
        else:
            label = source or "storyboard"
            as_path = Path(label)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.workdir = (
                self.out_root / f"{slugify(as_path.stem if as_path.is_file() else label)}-{stamp}"
            )
        self.workdir.mkdir(parents=True, exist_ok=True)
        try:
            return self._run(source, storyboard, storyboard_only)
        except Cancelled:
            self.emit("error", message="Cancelled")
            raise
        except Exception as exc:
            self.emit("error", message=str(exc), trace=traceback.format_exc(limit=6))
            raise

    def _run(
        self, source: str | None, storyboard: Storyboard | Path | None, storyboard_only: bool
    ) -> ReelResult:
        assert self.workdir is not None
        cfg, client = self.cfg, self.client
        (self.workdir / "config.json").write_text(
            json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
        )

        # 1. connect ------------------------------------------------------------------------
        done = self._stage("connect", f"Connecting to Lemonade at {client.base_url}")
        health = client.health()
        models = client.models(show_all=True)
        toolkit = resolve_toolkit(
            models,
            cfg.omni,
            writer=cfg.writer,
            illustrator=cfg.illustrator,
            voice=cfg.voice_model,
            listener=cfg.listener,
            want_upscaler=cfg.upscale,
        )
        self.toolkit = toolkit
        info = client.system_info()
        self.server = {
            "url": client.base_url,
            "version": health.get("version"),
            "processor": info.get("Processor")
            or (info.get("devices", {}).get("cpu") or {}).get("name"),
            "memory": info.get("Physical Memory"),
            "os": info.get("OS Version"),
            "devices": info.get("devices") or {},
        }
        self.emit(
            "toolkit", "connect", "Toolkit ready", toolkit=toolkit.summary(), server=self.server
        )
        self.log(
            "Using "
            + (f"Omni model {toolkit.omni}" if toolkit.omni else "individually selected models")
            + f": writer {toolkit.writer.id}, illustrator {toolkit.illustrator.id if toolkit.illustrator else 'none (slides)'}, "
            f"voice {toolkit.voice.id}, listener {toolkit.listener.id if toolkit.listener else 'none'}",
            "connect",
        )
        done()

        # 2. read ---------------------------------------------------------------------------
        done = self._stage("read", "Reading the source")
        if storyboard is not None:
            board = (
                storyboard
                if isinstance(storyboard, Storyboard)
                else Storyboard.load(Path(storyboard))
            )
            doc = SourceDoc(
                title=board.source_title or board.title,
                text="",
                kind="storyboard",
                origin="storyboard",
            )
        else:
            board = None
            doc = load_source(
                source or "", transcriber=self._transcriber() if toolkit.can_listen else None
            )
            (self.workdir / "source.txt").write_text(doc.text, encoding="utf-8")
            self.log(f'Read {doc.kind} source "{doc.title}" ({len(doc.text):,} characters)', "read")
        done()

        # 3. write --------------------------------------------------------------------------
        done = self._stage(
            "write", f"Writing a {cfg.scenes}-scene storyboard with {toolkit.writer.id}"
        )
        board_path = self.workdir / "storyboard.json"
        if board is None and board_path.exists():
            board = Storyboard.load(board_path)
            self.log("Reusing the existing storyboard.json", "write")
        if board is None:
            board = write_storyboard(
                client,
                toolkit.writer.id,
                doc,
                cfg,
                log=lambda m: self.log(m, "write"),
                context_length=toolkit.writer.context_length,
            )
            stats = client.stats()
            if stats:
                self.emit("progress", "write", "Writer speed", llm_stats=stats)
        board.save(board_path)
        self._snapshot_devices()
        self.emit(
            "storyboard",
            "write",
            f'Storyboard: "{board.title}"',
            title=board.title,
            scenes=[
                {
                    "caption": s.caption,
                    "narration": s.spoken_text,
                    "visual": s.visual,
                    "lines": [{"speaker": ln.speaker, "text": ln.text} for ln in s.lines],
                }
                for s in board.scenes
            ],
            hosts=board.hosts,
        )
        done()
        if storyboard_only:
            wall = time.monotonic() - self._started
            self.emit(
                "done",
                message="Storyboard written",
                storyboard=self._rel(board_path),
                workdir=str(self.workdir),
                seconds=round(wall, 1),
            )
            return ReelResult(workdir=self.workdir, storyboard=board, seconds=wall)

        # 4 + 5. draw and speak (in parallel) ------------------------------------------------
        outputs = [SceneOutput() for _ in board.scenes]
        title_clip: list[ClipResult] = []
        lexicon = Lexicon.build(
            cfg.base_language,
            learned_path=default_learned_path() if cfg.learn else None,
            extra_path=Path(cfg.lexicon_path) if cfg.lexicon_path else None,
        )
        narrator = Narrator(
            client,
            toolkit,
            cfg,
            lexicon,
            log=lambda m: self.log(m, "speak"),
            learned_path=default_learned_path() if cfg.learn else None,
        )

        def draw_lane() -> None:
            finish = self._stage(
                "draw",
                "Illustrating scenes"
                + (" (with vision critique)" if cfg.critique and toolkit.can_see else ""),
            )
            for i, scene in enumerate(board.scenes):
                self._check()
                outputs[i].art = self._draw(scene, i, toolkit)
            finish()

        def speak_lane() -> None:
            finish = self._stage(
                "speak",
                "Narrating"
                + (" (with listen-back)" if cfg.listen_back and toolkit.can_listen else ""),
            )
            if cfg.title_card:
                title_clip.append(narrator.narrate_title(board.title, self.workdir / "title"))
            for i, scene in enumerate(board.scenes):
                self._check()
                outputs[i].voice, outputs[i].cues = self._speak(narrator, scene, i)
            finish()

        if cfg.parallel:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="lane") as pool:
                futures = [pool.submit(draw_lane), pool.submit(speak_lane)]
                errors = [f.exception() for f in futures]
            for err in errors:
                if err is not None:
                    raise err
        else:
            draw_lane()
            speak_lane()
        self._snapshot_devices()

        # 6. edit ---------------------------------------------------------------------------
        done = self._stage("edit", "Editing the video")
        timeline = self._edit(board, outputs, title_clip[0] if title_clip else None, toolkit)
        done()

        # 7. report -------------------------------------------------------------------------
        done = self._stage("report", "Writing the run report")
        wall = time.monotonic() - self._started
        report_json, report_html = write_report(
            self.workdir,
            version=__version__,
            cfg=cfg,
            toolkit=toolkit,
            server=self.server,
            devices=self.devices,
            board=board,
            outputs=outputs,
            timeline=timeline,
            trace=client.tracer.to_list(),
            stage_times=self.stage_times,
            wall_seconds=wall,
            video_name=timeline.video.name,
        )
        done()
        result = ReelResult(
            workdir=self.workdir,
            storyboard=board,
            seconds=wall,
            video=timeline.video,
            timeline=timeline,
            report_html=report_html,
            report_json=report_json,
        )
        self.emit(
            "done",
            message=f"Finished in {wall:.0f}s",
            video=self._rel(timeline.video),
            report=self._rel(report_html),
            duration=timeline.duration,
            seconds=round(wall, 1),
            workdir=str(self.workdir),
        )
        return result

    # -- lanes -----------------------------------------------------------------------------

    def _transcriber(self) -> Callable[[Path], str]:
        def transcribe(path: Path) -> str:
            assert self.toolkit is not None and self.toolkit.listener is not None and self.workdir
            self.log(f"Transcribing voice memo {path.name} with {self.toolkit.listener.id}", "read")
            wav = to_asr_wav(path, self.workdir / "memo.16k.wav")
            result = self.client.transcribe(
                self.toolkit.listener.id, wav, response_format="json", note="voice memo"
            )
            return str(result.get("text", ""))

        return transcribe

    def _scene_dir(self, i: int) -> Path:
        assert self.workdir is not None
        return self.workdir / "scenes" / f"{i + 1:02d}"

    def _draw(self, scene: Scene, i: int, toolkit: Toolkit) -> SceneArt:
        cfg = self.cfg
        scene_dir = self._scene_dir(i)
        meta = scene_dir / "art.json"
        key = _hash(
            scene.visual,
            cfg.style,
            toolkit.illustrator.id if toolkit.illustrator else None,
            cfg.aspect,
            cfg.seed,
            cfg.image_steps,
            cfg.critique,
            cfg.upscale,
        )
        if meta.exists() and (scene_dir / "image.png").exists():
            cached = json.loads(meta.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                art = SceneArt(
                    path=cached["path"],
                    attempts=[ImageAttempt(**a) for a in cached.get("attempts", [])],
                    fallback=cached.get("fallback", False),
                    error=cached.get("error", ""),
                )
                self.emit(
                    "art",
                    "draw",
                    f"Scene {i + 1}: reused picture",
                    index=i,
                    image=self._rel(scene_dir / "image.png"),
                    score=cached.get("best_score"),
                    attempts=len(cached.get("attempts", [])),
                    cached=True,
                )
                return art
        art = draw_scene(
            self.client,
            toolkit,
            cfg,
            scene,
            i,
            scene_dir,
            cfg.style,
            log=lambda m: self.log(m, "draw"),
        )
        meta.write_text(json.dumps({"key": key, **art.to_dict()}, indent=2), encoding="utf-8")
        self.emit(
            "art",
            "draw",
            f"Scene {i + 1} illustrated",
            index=i,
            image=self._rel(scene_dir / "image.png"),
            score=art.best_score,
            attempts=len(art.attempts),
            fallback=art.fallback,
        )
        return art

    def _speak(self, narrator: Narrator, scene: Scene, i: int) -> tuple[SceneVoice, list[Cue]]:
        cfg, toolkit = self.cfg, narrator.toolkit
        scene_dir = self._scene_dir(i)
        meta = scene_dir / "voice-cache.json"
        key = _hash(
            scene.narration,
            [(ln.speaker, ln.text) for ln in scene.lines],
            cfg.voices,
            cfg.speed,
            toolkit.voice.id,
            toolkit.listener.id if toolkit.listener else None,
            cfg.listen_back,
            cfg.language,
        )
        if meta.exists():
            cached = json.loads(meta.read_text(encoding="utf-8"))
            if cached.get("key") == key and Path(cached["voice"]["path"]).exists():
                voice = SceneVoice(
                    path=cached["voice"]["path"],
                    duration=cached["voice"]["duration"],
                    clips=[],
                    starts=cached["voice"]["starts"],
                    summary=cached["voice"],
                )
                cues = [Cue(**c) for c in cached["cues"]]
                self.emit(
                    "voice",
                    "speak",
                    f"Scene {i + 1}: reused narration",
                    index=i,
                    audio=self._rel(voice.path),
                    cached=True,
                    wer=cached["voice"].get("wer"),
                )
                return voice, cues
        voice = narrator.narrate_scene(scene, i, scene_dir)
        max_chars = 56 if cfg.aspect == "9:16" else 84
        cues: list[Cue] = []
        for clip, start in zip(voice.clips, voice.starts):
            cues += build_cues(
                clip.text,
                clip.duration,
                clip.words or None,
                cfg.base_language,
                max_chars,
                offset=start,
            )
        meta.write_text(
            json.dumps(
                {"key": key, "voice": voice.to_dict(), "cues": [c.__dict__ for c in cues]}, indent=2
            ),
            encoding="utf-8",
        )
        self.emit(
            "voice",
            "speak",
            f"Scene {i + 1} narrated",
            index=i,
            audio=self._rel(voice.path),
            duration=round(voice.duration, 2),
            wer=None if voice.wer is None else round(voice.wer, 3),
            first_wer=None if voice.first_wer is None else round(voice.first_wer, 3),
            fixes=voice.fixes,
            heard=" ".join(c.final.heard or "" for c in voice.clips).strip(),
        )
        return voice, cues

    def _edit(
        self,
        board: Storyboard,
        outputs: list[SceneOutput],
        title: ClipResult | None,
        toolkit: Toolkit,
    ) -> Timeline:
        assert self.workdir is not None
        cfg = self.cfg
        size = cfg.frame_size
        cards = self.workdir / "cards"
        cards.mkdir(exist_ok=True)
        segments: list[SegmentSpec] = []
        first_image = Path(outputs[0].art.path) if outputs and outputs[0].art else None
        if cfg.title_card:
            from PIL import Image

            bg = Image.open(first_image) if first_image and first_image.exists() else None
            subtitle = (
                f"A conversation between {board.hosts[0]} and {board.hosts[1]}"
                if board.mode == "duo"
                else "An explainer made locally with Lemonade"
            )
            graphics.title_card(board.title, subtitle, size, bg, cfg.language).save(
                cards / "title.png"
            )
            graphics.title_card(board.title, subtitle, (1280, 720), bg, cfg.language).convert(
                "RGB"
            ).save(self.workdir / "thumbnail.jpg", quality=90)
            segments.append(
                SegmentSpec(
                    image=cards / "title.png",
                    audio=Path(title.path) if title else None,
                    audio_duration=title.duration if title else 0.0,
                    motion="zoom_in",
                    lead=0.5,
                    tail=0.9,
                    min_duration=3.0,
                    chapter="Intro",
                )
            )
        for i, (scene, out) in enumerate(zip(board.scenes, outputs)):
            assert out.voice is not None and out.art is not None
            segments.append(
                SegmentSpec(
                    image=Path(out.art.path),
                    audio=Path(out.voice.path),
                    audio_duration=out.voice.duration,
                    cues=out.cues,
                    badge=(i + 1, scene.caption),
                    motion=MOTIONS[i % len(MOTIONS)],
                    chapter=scene.caption,
                )
            )
        if cfg.outro:
            credits = [f"Script and critique: {toolkit.writer.id}"]
            if toolkit.illustrator:
                credits.append(f"Illustrations: {toolkit.illustrator.id}")
            credits.append(
                f"Voice: {toolkit.voice.id} ({', '.join(sorted(set(cfg.voices[: 2 if board.mode == 'duo' else 1])))})"
            )
            if toolkit.listener:
                credits.append(f"Listen-back and captions: {toolkit.listener.id}")
            if toolkit.omni:
                credits.insert(0, f"Omni model: {toolkit.omni}")
            graphics.outro_card(credits, size).save(cards / "outro.png")
            segments.append(
                SegmentSpec(
                    image=cards / "outro.png",
                    audio=None,
                    audio_duration=0.0,
                    motion="zoom_out",
                    lead=0,
                    tail=0,
                    min_duration=4.0,
                )
            )

        video = self.workdir / f"{slugify(board.title)}.mp4"
        timeline = assemble(
            segments,
            video,
            self.workdir / "render",
            size,
            board.title,
            fps=cfg.fps,
            captions=cfg.captions,
            language=cfg.base_language,
            progress=lambda n, total: self.emit(
                "progress", "edit", f"Rendered {n}/{total} segments", done=n, total=total
            ),
        )
        (self.workdir / "captions.srt").write_text(to_srt(timeline.cues), encoding="utf-8")
        (self.workdir / "captions.vtt").write_text(to_vtt(timeline.cues), encoding="utf-8")
        (self.workdir / "chapters.txt").write_text(
            youtube_chapters(timeline.chapters), encoding="utf-8"
        )
        return timeline
