"""LemonReel Studio backend.

A FastAPI app that serves one static page, starts reel jobs in background threads and
streams their progress to the browser with Server-Sent Events. Finished reels are served
straight from the output folder. It binds to localhost by default: nothing leaves the PC.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from lemonreel import __version__
from lemonreel.config import DEFAULT_VOICES, KOKORO_VOICES, LANGUAGE_NAMES, LOOKS, ReelConfig
from lemonreel.lemonade import LemonadeClient, LemonadeError

STATIC = Path(__file__).parent / "static"
UPLOAD_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".html",
    ".htm",
    ".pdf",
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".oga",
    ".opus",
    ".webm",
    ".flac",
    ".aac",
}


@dataclass
class Job:
    id: str
    title: str
    created: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"  # running | done | error | cancelled
    workdir: Path | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


def _config_from_options(options: dict[str, Any]) -> ReelConfig:
    allowed = {
        "scenes": int,
        "minutes": float,
        "mode": str,
        "language": str,
        "look": str,
        "audience": str,
        "voice": str,
        "voice2": str,
        "speed": float,
        "aspect": str,
        "hd": bool,
        "captions": str,
        "critique": bool,
        "listen_back": bool,
        "title_card": bool,
        "outro": bool,
        "omni": str,
        "seed": int,
    }
    kwargs: dict[str, Any] = {}
    for key, cast in allowed.items():
        value = options.get(key)
        if value in (None, ""):
            continue
        kwargs[key] = cast(value)
    return ReelConfig(**kwargs)


def create_app(out: Path, base_url: str | None = None) -> FastAPI:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    uploads = out / ".uploads"
    app = FastAPI(title="LemonReel Studio", version=__version__)
    jobs: dict[str, Job] = {}
    lock = threading.Lock()

    def client() -> LemonadeClient:
        return LemonadeClient.discover(base_url)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        from lemonreel.doctor import diagnose

        presets = {
            "looks": list(LOOKS),
            "languages": LANGUAGE_NAMES,
            "voices": KOKORO_VOICES,
            "default_voices": DEFAULT_VOICES,
            "version": __version__,
        }
        try:
            diag = diagnose(client())
        except LemonadeError as exc:
            return {"connected": False, "error": str(exc), **presets}
        return {
            "connected": True,
            "url": diag.url,
            "lemonade_version": diag.version,
            "hardware": diag.hardware,
            "omni": diag.omni,
            "toolkit": diag.toolkit.summary() if diag.toolkit else None,
            "toolkit_error": str(diag.toolkit_error) if diag.toolkit_error else None,
            "ffmpeg": bool(diag.ffmpeg),
            **presets,
        }

    @app.post("/api/jobs")
    async def create_job(
        kind: str = Form(...),
        text: str = Form(""),
        options: str = Form("{}"),
        file: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        from lemonreel.pipeline import Reel

        try:
            cfg = _config_from_options(json.loads(options or "{}"))
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"Bad options: {exc}") from exc
        job_id = uuid.uuid4().hex[:10]
        job_uploads = uploads / job_id
        if kind in ("text", "markdown"):
            if not text.strip():
                raise HTTPException(400, "Paste some text first.")
            job_uploads.mkdir(parents=True, exist_ok=True)
            first_line = text.strip().splitlines()[0].lstrip("# ").strip()[:60] or "pasted text"
            name = re.sub(r"[^\w\- ]", "", first_line).strip().replace(" ", "-") or "pasted"
            source_path = job_uploads / f"{name}.md"
            source_path.write_text(text, encoding="utf-8")
            source, title = str(source_path), first_line
        elif kind in ("url", "topic"):
            if not text.strip():
                raise HTTPException(400, "Enter a URL or topic first.")
            source, title = text.strip(), text.strip()[:80]
        elif kind in ("file", "memo"):
            if file is None or not file.filename:
                raise HTTPException(400, "Choose a file or record a memo first.")
            suffix = Path(file.filename).suffix.lower() or ".webm"
            if suffix not in UPLOAD_SUFFIXES:
                raise HTTPException(400, f"Unsupported file type {suffix}")
            job_uploads.mkdir(parents=True, exist_ok=True)
            stem = re.sub(r"[^\w\-]", "-", Path(file.filename).stem)[:60] or "upload"
            source_path = job_uploads / f"{stem}{suffix}"
            source_path.write_bytes(await file.read())
            source, title = str(source_path), "Voice memo" if kind == "memo" else file.filename
        else:
            raise HTTPException(400, f"Unknown source kind {kind}")

        job = Job(id=job_id, title=title)
        with lock:
            jobs[job_id] = job

        def record(event) -> None:
            data = event.to_dict()
            if (
                event.kind in ("stage", "storyboard")
                and job.workdir is None
                and reel.workdir is not None
            ):
                job.workdir = reel.workdir
            if reel.workdir is not None:
                data["base"] = f"/reels/{reel.workdir.resolve().relative_to(out).as_posix()}/"
            job.events.append(data)

        try:
            reel = Reel(cfg, client(), out_root=out, on_event=record, cancel=job.cancel)
        except LemonadeError as exc:
            with lock:
                jobs.pop(job_id, None)
            raise HTTPException(503, str(exc)) from exc

        def work() -> None:
            try:
                reel.run(source)
                job.status = "done"
            except Exception:  # noqa: BLE001 - the error event already carries details
                job.status = "cancelled" if job.cancel.is_set() else "error"

        threading.Thread(target=work, name=f"reel-{job_id}", daemon=True).start()
        return {"id": job_id}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, since: int = 0) -> StreamingResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job")

        async def stream():
            index = max(0, since)
            idle = 0
            while True:
                while index < len(job.events):
                    payload = json.dumps(job.events[index], ensure_ascii=False)
                    yield f"id: {index}\ndata: {payload}\n\n"
                    index += 1
                    idle = 0
                if job.status != "running" and index >= len(job.events):
                    yield f"event: end\ndata: {json.dumps({'status': job.status})}\n\n"
                    return
                idle += 1
                if idle % 60 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, str]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        job.cancel.set()
        return {"status": "cancelling"}

    @app.get("/api/reels")
    def reels() -> list[dict[str, Any]]:
        found = []
        for report in sorted(
            out.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:24]:
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            folder = report.parent.name
            found.append(
                {
                    "title": data.get("title"),
                    "base": f"/reels/{folder}/",
                    "video": data.get("video", {}).get("file"),
                    "duration": data.get("video", {}).get("duration_seconds"),
                    "made_in": data.get("wall_seconds"),
                    "omni": data.get("omni_model"),
                    "generated_at": data.get("generated_at"),
                }
            )
        return found

    app.mount("/reels", StaticFiles(directory=out), name="reels")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    out: Path = Path("reels"),
    base_url: str | None = None,
    open_browser: bool = True,
) -> int:
    import uvicorn

    app = create_app(out, base_url)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    print(f"LemonReel Studio is running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
