"""``lemonreel doctor``: check the setup and, optionally, time one call per modality."""

from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from lemonreel.lemonade import LemonadeClient, LemonadeError
from lemonreel.media import ffmpeg_exe, media_duration, to_asr_wav, write_speech_clip
from lemonreel.omni import (
    DEFAULT_OMNI,
    OMNI_RECIPE,
    Toolkit,
    ToolkitError,
    components_of,
    resolve_toolkit,
)
from lemonreel.textmetrics import word_error_rate


@dataclass
class Diagnosis:
    url: str = ""
    version: str | None = None
    hardware: list[str] = field(default_factory=list)
    omni: list[dict[str, Any]] = field(default_factory=list)
    toolkit: Toolkit | None = None
    toolkit_error: ToolkitError | None = None
    ffmpeg: str | None = None
    ffmpeg_error: str | None = None
    smoke: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.toolkit is not None and self.ffmpeg is not None


def hardware_lines(info: dict[str, Any]) -> list[str]:
    lines = []
    if info.get("Processor"):
        lines.append(str(info["Processor"]))
    if info.get("Physical Memory"):
        lines.append(f"{info['Physical Memory']} RAM")
    devices = info.get("devices") or {}
    for gpu in devices.get("amd_gpu") or []:
        if isinstance(gpu, dict) and gpu.get("available", True):
            fam = f" ({gpu['family']})" if gpu.get("family") else ""
            lines.append(f"GPU: {gpu.get('name')}{fam}")
    for gpu in devices.get("nvidia_gpu") or []:
        if isinstance(gpu, dict):
            lines.append(f"GPU: {gpu.get('name')}")
    npu = devices.get("amd_npu")
    if isinstance(npu, dict) and npu.get("available"):
        lines.append(f"NPU: {npu.get('family') or npu.get('name')}")
    return lines


def diagnose(client: LemonadeClient, omni: str | None = None) -> Diagnosis:
    diag = Diagnosis(url=client.base_url)
    health = client.health()
    diag.version = health.get("version")
    diag.hardware = hardware_lines(client.system_info())
    models = client.models(show_all=True)
    index = {str(m.get("id")): m for m in models}
    for m in models:
        if m.get("recipe") != OMNI_RECIPE:
            continue
        parts = components_of(m, index)
        diag.omni.append(
            {
                "id": m.get("id"),
                "downloaded": bool(m.get("downloaded")) and all(p.downloaded for p in parts),
                "components": [p.id for p in parts],
                "size_gb": m.get("size"),
            }
        )
    try:
        diag.toolkit = resolve_toolkit(models, omni)
    except ToolkitError as exc:
        diag.toolkit_error = exc
    try:
        diag.ffmpeg = ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        diag.ffmpeg_error = str(exc)
    return diag


def smoke_test(client: LemonadeClient, toolkit: Toolkit) -> list[dict[str, Any]]:
    """One small request per role, timed. Loading a model the first time is included."""
    results: list[dict[str, Any]] = []

    def record(role: str, model: str, fn) -> Any:
        start = time.monotonic()
        try:
            value, detail = fn()
            results.append(
                {
                    "role": role,
                    "model": model,
                    "ok": True,
                    "seconds": time.monotonic() - start,
                    "detail": detail,
                }
            )
            return value
        except (LemonadeError, OSError, ValueError, wave.Error) as exc:
            results.append(
                {
                    "role": role,
                    "model": model,
                    "ok": False,
                    "seconds": time.monotonic() - start,
                    "detail": str(exc),
                }
            )
            return None

    record(
        "writer",
        toolkit.writer.id,
        lambda: (
            None,
            client.chat(
                toolkit.writer.id,
                [{"role": "user", "content": "Reply with the single word: ready"}],
                max_tokens=16,
            ).text[:40],
        ),
    )
    if toolkit.illustrator:

        def draw():
            png = client.generate_image(
                toolkit.illustrator.id,
                "a ripe lemon on a wooden table, soft light",
                size="256x256",
                seed=7,
            )
            from PIL import Image

            img = Image.open(io.BytesIO(png))
            return png, f"{img.width}x{img.height} image"

        record("illustrator", toolkit.illustrator.id, draw)

    phrase = "Hello from LemonReel. Every word you hear was made on this computer."
    with TemporaryDirectory() as tmp:
        clip = Path(tmp) / "hello.wav"

        def speak():
            write_speech_clip(client.speech(toolkit.voice.id, phrase, voice="af_heart"), clip)
            return clip, f"{media_duration(clip):.1f}s of audio"

        spoken = record("voice", toolkit.voice.id, speak)
        if toolkit.listener and spoken:

            def listen():
                wav = to_asr_wav(clip, Path(tmp) / "hello16.wav")
                heard = str(
                    client.transcribe(
                        toolkit.listener.id, wav, language="en", response_format="json"
                    ).get("text", "")
                )
                return (
                    heard,
                    f'heard "{heard.strip()[:60]}" ({word_error_rate(phrase, heard):.0%} word error)',
                )

            record("listener", toolkit.listener.id, listen)
    return results


__all__ = ["DEFAULT_OMNI", "Diagnosis", "diagnose", "hardware_lines", "smoke_test"]
