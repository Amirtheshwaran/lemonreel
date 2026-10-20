"""A small, thread-safe client for the Lemonade Server HTTP API.

Only the endpoints LemonReel needs are wrapped. Every request is recorded by a
:class:`RequestTracer`, which is what draws the "who was busy when" timeline in the
run report (LLM, image, speech and transcription engines often run side by side on
different devices).
"""

from __future__ import annotations

import base64
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from lemonreel.llm_json import strip_reasoning

DEFAULT_URL = "http://localhost:13305"
LEGACY_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

START_HINT = (
    "Start Lemonade Server first: launch the Lemonade app (Windows/macOS) or run `lemond` "
    "(Linux), then check it with `lemonade status`. Install it from https://lemonade-server.ai. "
    "If it listens somewhere else, pass --base-url or set LEMONADE_BASE_URL."
)


class LemonadeError(RuntimeError):
    """The server answered, but with an error."""

    def __init__(self, message: str, status: int | None = None, endpoint: str = ""):
        super().__init__(message)
        self.status = status
        self.endpoint = endpoint


class LemonadeUnavailable(LemonadeError):
    """Nothing is listening at the configured address."""


@dataclass
class TraceEvent:
    lane: str
    endpoint: str
    model: str | None
    start: float
    end: float
    ok: bool
    note: str = ""

    @property
    def seconds(self) -> float:
        return self.end - self.start


class RequestTracer:
    """Collects timing for every request, relative to when the tracer was created."""

    def __init__(self) -> None:
        self.origin = time.monotonic()
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()

    def now(self) -> float:
        return time.monotonic() - self.origin

    def add(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    def busy_seconds(self, lane: str) -> float:
        return sum(e.seconds for e in self.events if e.lane == lane)

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.events]


@dataclass
class ChatResult:
    text: str
    reasoning: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


_LANES = {
    "/chat/completions": "llm",
    "/images/generations": "image",
    "/images/upscale": "image",
    "/audio/speech": "tts",
    "/audio/transcriptions": "asr",
}


def normalize_base_url(url: str) -> str:
    """Accept "localhost:13305", "http://host:13305/api/v1", "http://host/v1", ..."""
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    for suffix in ("/api/v1", "/api/v0", "/v1", "/v0"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:500] or response.reason_phrase
    if isinstance(payload, dict):
        err = payload.get("error", payload.get("detail", payload.get("message")))
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
    return str(payload)[:500]


class LemonadeClient:
    """Talks to one Lemonade Server instance."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 900.0,
        tracer: RequestTracer | None = None,
    ):
        self.base_url = normalize_base_url(
            base_url or os.environ.get("LEMONADE_BASE_URL") or DEFAULT_URL
        )
        self.api_key = api_key if api_key is not None else os.environ.get("LEMONADE_API_KEY")
        self.timeout = timeout
        self.tracer = tracer or RequestTracer()
        self._local = threading.local()

    # -- plumbing ---------------------------------------------------------------------------

    @property
    def api_root(self) -> str:
        return self.base_url + API_PREFIX

    def _http(self) -> httpx.Client:
        client = getattr(self._local, "client", None)
        if client is None:
            headers = {"User-Agent": "lemonreel"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            client = httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                headers=headers,
                # Lemonade is local; never route it through a corporate proxy.
                trust_env=False,
            )
            self._local.client = client
        return client

    def _request(
        self,
        method: str,
        path: str,
        *,
        model: str | None = None,
        timeout: float | None = None,
        note: str = "",
        **kwargs: Any,
    ) -> httpx.Response:
        url = self.api_root + path
        lane = _LANES.get(path, "other")
        start = self.tracer.now()
        ok = False
        try:
            if timeout is not None:
                kwargs["timeout"] = timeout
            try:
                response = self._http().request(method, url, **kwargs)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                raise LemonadeUnavailable(
                    f"Can't reach Lemonade Server at {self.base_url}. {START_HINT}",
                    endpoint=path,
                ) from exc
            except httpx.TimeoutException as exc:
                raise LemonadeError(
                    f"Lemonade took longer than {timeout or self.timeout:.0f}s to answer {path}. "
                    "Large models on CPU can be slow; try a smaller Omni model or raise --timeout.",
                    endpoint=path,
                ) from exc
            if response.status_code >= 400:
                raise LemonadeError(
                    f"{path} failed ({response.status_code}): {_error_message(response)}",
                    status=response.status_code,
                    endpoint=path,
                )
            ok = True
            return response
        finally:
            if lane != "other":
                self.tracer.add(TraceEvent(lane, path, model, start, self.tracer.now(), ok, note))

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise LemonadeError(f"{path} returned non-JSON data", endpoint=path) from exc

    # -- discovery ---------------------------------------------------------------------------

    @classmethod
    def discover(
        cls,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 900.0,
        tracer: RequestTracer | None = None,
        candidates: Iterable[str] | None = None,
    ) -> LemonadeClient:
        """Return a client for the first server that answers.

        An explicit ``base_url`` (or ``LEMONADE_BASE_URL``) is used as-is. Otherwise the
        current default port (13305) and the legacy one (8000) are probed.
        """
        explicit = base_url or os.environ.get("LEMONADE_BASE_URL")
        urls = [explicit] if explicit else list(candidates or (DEFAULT_URL, LEGACY_URL))
        last_error: LemonadeError | None = None
        for url in urls:
            client = cls(url, api_key=api_key, timeout=timeout, tracer=tracer)
            try:
                client._json("GET", "/health", timeout=3.0)
                return client
            except LemonadeError as exc:
                last_error = exc
        if isinstance(last_error, LemonadeUnavailable) and not explicit:
            raise LemonadeUnavailable(
                f"Can't reach Lemonade Server on {' or '.join(urls)}. {START_HINT}"
            )
        assert last_error is not None
        raise last_error

    # -- server information -------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health")

    def models(self, show_all: bool = False) -> list[dict[str, Any]]:
        params = {"show_all": "true"} if show_all else None
        payload = self._json("GET", "/models", params=params)
        data = payload.get("data", []) if isinstance(payload, dict) else payload
        return [m for m in data if isinstance(m, dict)]

    def model(self, model_id: str) -> dict[str, Any]:
        return self._json("GET", f"/models/{model_id}")

    def system_info(self) -> dict[str, Any]:
        try:
            return self._json("GET", "/system-info", timeout=30.0)
        except LemonadeError:
            return {}

    def stats(self) -> dict[str, Any]:
        try:
            return self._json("GET", "/stats", timeout=10.0)
        except LemonadeError:
            return {}

    def load(self, model: str, **options: Any) -> dict[str, Any]:
        return self._json("POST", "/load", json={"model_name": model, **options}, model=model)

    # -- inference -------------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        thinking: bool = False,
        note: str = "",
        **extra: Any,
    ) -> ChatResult:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        # Reasoning models spend their budget thinking unless told not to; LemonReel wants
        # direct answers. Templates that don't know this flag simply ignore it.
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(extra)
        started = time.monotonic()
        try:
            payload = self._json("POST", "/chat/completions", json=body, model=model, note=note)
        except LemonadeError as exc:
            # Older backends reject response_format; fall back to prompt-only JSON.
            if json_mode and exc.status == 400 and "response_format" in str(exc):
                body.pop("response_format", None)
                payload = self._json("POST", "/chat/completions", json=body, model=model, note=note)
            else:
                raise
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        reasoning = message.get("reasoning_content") or ""
        text = strip_reasoning(content)
        if not text and reasoning:
            text = strip_reasoning(reasoning)
        return ChatResult(
            text=text,
            reasoning=reasoning,
            finish_reason=choice.get("finish_reason"),
            usage=payload.get("usage") or {},
            seconds=time.monotonic() - started,
            raw=payload,
        )

    def generate_image(
        self,
        model: str,
        prompt: str,
        *,
        size: str | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
        note: str = "",
    ) -> bytes:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
        if size:
            body["size"] = size
        if steps is not None:
            body["steps"] = steps
        if cfg_scale is not None:
            body["cfg_scale"] = cfg_scale
        if seed is not None:
            body["seed"] = seed
        payload = self._json("POST", "/images/generations", json=body, model=model, note=note)
        return _first_image(payload, "/images/generations")

    def upscale_image(self, model: str, png: bytes, note: str = "") -> bytes:
        body = {"model": model, "image": base64.b64encode(png).decode("ascii")}
        payload = self._json("POST", "/images/upscale", json=body, model=model, note=note)
        return _first_image(payload, "/images/upscale")

    def speech(
        self,
        model: str,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        response_format: str = "wav",
        note: str = "",
    ) -> bytes:
        body: dict[str, Any] = {"model": model, "input": text, "response_format": response_format}
        if voice:
            body["voice"] = voice
        if speed is not None:
            body["speed"] = speed
        response = self._request("POST", "/audio/speech", json=body, model=model, note=note)
        if response.headers.get("content-type", "").startswith("application/json"):
            raise LemonadeError(
                f"/audio/speech returned JSON instead of audio: {_error_message(response)}",
                endpoint="/audio/speech",
            )
        if not response.content:
            raise LemonadeError("/audio/speech returned no audio", endpoint="/audio/speech")
        return response.content

    def transcribe(
        self,
        model: str,
        wav: bytes,
        *,
        language: str | None = None,
        response_format: str = "verbose_json",
        filename: str = "audio.wav",
        note: str = "",
    ) -> dict[str, Any]:
        data: dict[str, str] = {"model": model, "response_format": response_format}
        if language:
            data["language"] = language
        files = {"file": (filename, wav, "audio/wav")}
        response = self._request(
            "POST", "/audio/transcriptions", data=data, files=files, model=model, note=note
        )
        try:
            payload = response.json()
        except ValueError:
            return {"text": response.text.strip()}
        if isinstance(payload, str):
            return {"text": payload}
        return payload


def _first_image(payload: Any, endpoint: str) -> bytes:
    try:
        b64 = payload["data"][0]["b64_json"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LemonadeError(f"{endpoint} returned no image data", endpoint=endpoint) from exc
    return base64.b64decode(b64)
