"""A stand-in Lemonade Server for tests and offline UI work.

It follows the request/response shapes documented in Lemonade's API spec
(``docs/api/openai.md`` and ``docs/api/lemonade.md``) closely enough to drive LemonReel
end to end without models. Behaviour worth knowing:

* Speech is synthesized as tone bursts, one per word, so durations scale with text. The
  first 0.25 s carries an ID tone that survives resampling, which is how the fake
  recognizer knows which utterance it is hearing.
* The recognizer mishears "Strix" as "tricks" unless the speech input respelled it, so the
  listen-back repair loop has something real to fix.
* The second image critique returns a low score, so the redraw loop runs too.

Run it standalone with ``python tests/fake_lemonade.py --port 13399``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import struct
import threading
import wave
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw

SPEECH_RATE = 24_000
ID_SECONDS = 0.25
WORD_SECONDS = 0.30
MISHEARD = {"strix": "tricks"}
RESPELLED = {"stricks": "Strix"}


def _model(
    mid: str, recipe: str, labels: list[str], downloaded: bool = True, **extra: Any
) -> dict[str, Any]:
    return {
        "id": mid,
        "created": 1744173590,
        "object": "model",
        "owned_by": "lemonade",
        "checkpoint": f"example/{mid}",
        "recipe": recipe,
        "downloaded": downloaded,
        "suggested": True,
        "labels": labels,
        **extra,
    }


def catalog(profile: str = "full") -> list[dict[str, Any]]:
    llm = _model(
        "Qwen3.5-4B-MTP-GGUF",
        "llamacpp",
        ["chat", "vision", "tool-calling", "mtp"],
        size=3.66,
        context_length=8192,
    )
    image = _model(
        "SD-Turbo",
        "sd-cpp",
        ["image"],
        size=5.21,
        image_defaults={
            "steps": 4,
            "cfg_scale": 1.0,
            "width": 512,
            "height": 512,
            "soft_pixel_cap": 525625,
        },
    )
    asr = _model(
        "Whisper-Tiny", "whispercpp", ["transcription", "realtime-transcription"], size=0.075
    )
    tts = _model("kokoro-v1", "kokoro", ["tts"], size=0.354)
    small = _model("Qwen3-0.6B-GGUF", "llamacpp", ["chat", "reasoning"], size=0.38)
    if profile == "minimal":
        # A chat model and a voice only: no image model, no recognizer, no Omni model.
        return [small, tts]
    components = [llm, image, asr, tts]
    lite = _model(
        "LMX-Omni-5.5B-Lite",
        "collection.omni",
        ["chat"],
        size=9.3,
        components=[m["id"] for m in components],
        models=[
            {k: v for k, v in m.items() if k not in ("downloaded", "created")} for m in components
        ],
    )
    halo = _model(
        "LMX-Omni-52B-Halo",
        "collection.omni",
        ["chat"],
        downloaded=False,
        size=44.77,
        components=[
            "Qwen3.6-35B-A3B-MTP-GGUF",
            "Flux-2-Klein-9B-GGUF",
            "Whisper-Large-v3-Turbo",
            "kokoro-v1",
        ],
    )
    return [lite, halo, *components, small]


class FakeState:
    def __init__(self, profile: str = "full", segments_only: bool = False):
        self.profile = profile
        self.segments_only = segments_only
        self.lock = threading.Lock()
        self.utterances: list[str] = []
        self.loaded: dict[str, dict[str, Any]] = {}
        self.critiques = 0
        self.requests: list[dict[str, Any]] = []
        self.asr_rates: list[int] = []

    def touch(self, model: str, kind: str, device: str) -> None:
        self.loaded[model] = {"type": kind, "device": device}


# -- audio ------------------------------------------------------------------------------------


def _tone(freq: float, seconds: float, amp: float = 0.45) -> list[int]:
    n = int(seconds * SPEECH_RATE)
    return [int(32767 * amp * math.sin(2 * math.pi * freq * i / SPEECH_RATE)) for i in range(n)]


def synthesize(text: str, uid: int) -> bytes:
    samples = _tone(300 + 25 * (uid % 60), ID_SECONDS)
    for k, _word in enumerate(text.split()):
        samples += _tone(180 + 15 * (k % 7), WORD_SECONDS - 0.08, 0.25) + [0] * int(
            0.08 * SPEECH_RATE
        )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SPEECH_RATE)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def identify(wav_bytes: bytes) -> tuple[int, float, int]:
    """Return (utterance id, duration, sample rate) for a clip made by :func:`synthesize`."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
        width = w.getsampwidth()
        channels = w.getnchannels()
    count = len(frames) // width
    samples = struct.unpack(f"<{count}h", frames)[::channels] if width == 2 else ()
    window = samples[int(0.03 * rate) : int(0.22 * rate)]
    crossings = sum(1 for a, b in zip(window, window[1:]) if (a < 0) != (b < 0))
    freq = crossings / 2 / (len(window) / rate) if window else 0.0
    uid = round((freq - 300) / 25)
    return uid, len(samples) / rate, rate


def hear(spoken: str) -> str:
    words = []
    for word in spoken.split():
        bare = re.sub(r"[^\w']", "", word).lower()
        if bare in MISHEARD:
            words.append(MISHEARD[bare])
        elif bare in RESPELLED:
            words.append(RESPELLED[bare])
        else:
            words.append(word)
    return " ".join(words)


# -- chat ------------------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [p for p in parts if len(p.split()) >= 4]


def _storyboard(prompt: str) -> str:
    scenes = int(re.search(r"exactly (\d+) scenes", prompt).group(1))
    duo = "two-host" in prompt
    match = re.search(r"<<<\n(.*?)\n>>>", prompt, re.DOTALL)
    source = match.group(1) if match else prompt
    sentences = _sentences(source) or ["Local AI keeps your data on your own computer."]
    out = []
    for i in range(scenes):
        a = sentences[(2 * i) % len(sentences)]
        b = sentences[(2 * i + 1) % len(sentences)]
        caption = " ".join(re.sub(r"[^\w\s]", "", a).split()[:4])
        visual = f"A friendly robot in a sunlit workshop illustrating: {caption.lower()}"
        if duo:
            out.append(
                {
                    "caption": caption,
                    "visual": visual,
                    "lines": [
                        {"speaker": "A", "text": a},
                        {"speaker": "B", "text": "Wait, really? " + b},
                    ],
                }
            )
        else:
            out.append({"caption": caption, "narration": f"{a} {b}", "visual": visual})
    board: dict[str, Any] = {"title": "How Local AI Works", "scenes": out}
    if duo:
        board["hosts"] = ["Maya", "Leo"]
    # Wrap like a chatty model would, to exercise the JSON extractor.
    return "Here is your storyboard:\n```json\n" + json.dumps(board, indent=1) + "\n```"


def chat_reply(state: FakeState, body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    system = " ".join(str(m.get("content")) for m in messages if m.get("role") == "system")
    last = messages[-1] if messages else {}
    content = last.get("content")
    if isinstance(content, list):
        with state.lock:
            state.critiques += 1
            n = state.critiques
        if n == 2:
            return json.dumps(
                {
                    "score": 2,
                    "problems": "the robot is missing",
                    "better_prompt": "A cheerful robot assembling a lemon-shaped computer chip",
                }
            )
        return json.dumps({"score": 4, "problems": "", "better_prompt": ""})
    text = str(content)
    if "head writer" in system or "storyboard" in text.lower():
        if "could not be used" in text:
            prompt = next(str(m["content"]) for m in messages if m.get("role") == "user")
            return _storyboard(prompt)
        return _storyboard(text)
    if "phonetic respelling" in text:
        fixes = {w: "Stricks" for w in ("Strix",) if w in text}
        return json.dumps({"respellings": fixes})
    if "key facts" in text:
        return "\n".join(f"- {s}" for s in _sentences(text)[:6])
    return "OK"


# -- app -------------------------------------------------------------------------------------


def create_app(profile: str = "full", segments_only: bool = False) -> FastAPI:
    app = FastAPI(title="Fake Lemonade")
    state = FakeState(profile, segments_only)
    app.state.fake = state

    def routes(path: str) -> list[str]:
        return [f"/api/v1{path}", f"/v1{path}"]

    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "9.4.1",
            "model_loaded": next(iter(state.loaded), None),
            "all_models_loaded": [
                {
                    "model_name": k,
                    "type": v["type"],
                    "device": v["device"],
                    "recipe": "fake",
                    "recipe_options": {"llamacpp_backend": "vulkan"} if v["type"] == "llm" else {},
                }
                for k, v in state.loaded.items()
            ],
            "max_models": {
                "llm": 1,
                "image": 1,
                "tts": 1,
                "transcription": 1,
                "embedding": 1,
                "reranking": 1,
            },
        }

    async def models(show_all: bool = False) -> dict[str, Any]:
        data = catalog(state.profile)
        if not show_all:
            data = [m for m in data if m["downloaded"]]
        return {"object": "list", "data": data}

    async def system_info() -> dict[str, Any]:
        return {
            "OS Version": "Linux-6.8 (fake)",
            "Processor": "AMD Ryzen AI Max+ 395 w/ Radeon 8060S (fake)",
            "Physical Memory": "128.0 GB",
            "devices": {
                "cpu": {
                    "name": "AMD Ryzen AI Max+ 395",
                    "cores": 16,
                    "threads": 32,
                    "available": True,
                },
                "amd_gpu": [
                    {
                        "name": "AMD Radeon 8060S Graphics",
                        "vram_gb": 96.0,
                        "available": True,
                        "family": "gfx1151",
                    }
                ],
                "amd_npu": {"name": "AMD Ryzen AI Max+ 395", "available": True, "family": "XDNA2"},
            },
        }

    async def stats() -> dict[str, Any]:
        return {
            "time_to_first_token": 0.21,
            "tokens_per_second": 41.5,
            "input_tokens": 512,
            "output_tokens": 640,
        }

    async def chat(request: Request) -> Any:
        body = await request.json()
        state.requests.append({"endpoint": "chat", "body": body})
        model = body.get("model")
        if not model:
            return JSONResponse(
                {"error": {"message": "model is required", "type": "invalid_request_error"}}, 400
            )
        state.touch(model, "llm", "gpu")
        text = chat_reply(state, body)
        return {
            "id": "0",
            "object": "chat.completion",
            "created": 1742927481,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": len(text.split()),
                "total_tokens": 100 + len(text.split()),
            },
        }

    async def images(request: Request) -> Any:
        body = await request.json()
        state.requests.append({"endpoint": "images", "body": body})
        state.touch(body["model"], "image", "gpu")
        w, h = (int(x) for x in str(body.get("size", "512x512")).split("x"))
        seed = int(body.get("seed") or 0)
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        hue = (seed * 37) % 255
        for y in range(h):
            draw.line([(0, y), (w, y)], fill=(hue, (y * 255) // h, 255 - hue))
        draw.ellipse((w * 0.3, h * 0.25, w * 0.7, h * 0.75), fill=(255, 214, 10))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {
            "created": 1742927481,
            "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}],
        }

    async def upscale(request: Request) -> Any:
        body = await request.json()
        img = Image.open(io.BytesIO(base64.b64decode(body["image"])))
        img = img.resize((img.width * 2, img.height * 2))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {
            "created": 1742927481,
            "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}],
        }

    async def speech(request: Request) -> Any:
        body = await request.json()
        state.requests.append({"endpoint": "speech", "body": body})
        state.touch(body["model"], "tts", "cpu")
        with state.lock:
            uid = len(state.utterances)
            state.utterances.append(str(body.get("input", "")))
        return Response(synthesize(body.get("input", ""), uid), media_type="audio/wav")

    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(...),
        response_format: str = Form("json"),
        language: str | None = Form(None),
    ) -> Any:
        state.touch(model, "transcription", "npu")
        data = await file.read()
        uid, duration, rate = identify(data)
        state.asr_rates.append(rate)
        spoken = state.utterances[uid] if 0 <= uid < len(state.utterances) else ""
        text = hear(spoken)
        if response_format != "verbose_json":
            return {"text": text}
        words = text.split()
        segments = []
        for start in range(0, len(words), 12):
            chunk = words[start : start + 12]
            timed = [
                {
                    "word": " " + w,
                    "start": round(ID_SECONDS + (start + k) * WORD_SECONDS, 3),
                    "end": round(ID_SECONDS + (start + k) * WORD_SECONDS + 0.22, 3),
                    "probability": 0.93,
                }
                for k, w in enumerate(chunk)
            ]
            seg = {
                "id": len(segments),
                "start": timed[0]["start"],
                "end": timed[-1]["end"],
                "text": " " + " ".join(chunk),
            }
            if not state.segments_only:
                seg["words"] = timed
            segments.append(seg)
        return {
            "task": "transcribe",
            "language": language or "english",
            "duration": round(duration, 3),
            "text": text,
            "segments": segments,
        }

    async def load(request: Request) -> Any:
        body = await request.json()
        return {"status": "success", "message": f"Loaded model: {body.get('model_name')}"}

    for path, fn, methods in [
        ("/health", health, ["GET"]),
        ("/models", models, ["GET"]),
        ("/system-info", system_info, ["GET"]),
        ("/stats", stats, ["GET"]),
        ("/chat/completions", chat, ["POST"]),
        ("/images/generations", images, ["POST"]),
        ("/images/upscale", upscale, ["POST"]),
        ("/audio/speech", speech, ["POST"]),
        ("/audio/transcriptions", transcriptions, ["POST"]),
        ("/load", load, ["POST"]),
    ]:
        for route in routes(path):
            app.add_api_route(route, fn, methods=methods)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=13399)
    parser.add_argument("--profile", choices=["full", "minimal"], default="full")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.profile), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
