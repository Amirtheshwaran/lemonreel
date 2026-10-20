from __future__ import annotations

import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

from fake_lemonade import create_app  # noqa: E402

SAMPLE_DOC = """# Strix Halo and local AI

Strix Halo is an AMD processor that puts a big GPU and an NPU next to the CPU.
It shares up to 128 GB of memory between all of them, so large models fit without a separate graphics card.
Lemonade is an open source server that runs language, image and speech models on that hardware.
Apps talk to Lemonade through the same API they would use for a cloud service.
Because everything runs on the laptop, prompts and files never leave the machine.
Developers can switch between the CPU, the GPU and the NPU without changing their code.
That makes private AI assistants practical for schools, clinics and small businesses.
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def running_fake(
    profile: str = "full", segments_only: bool = False
) -> Iterator[tuple[str, object]]:
    app = create_app(profile, segments_only)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(url + "/api/v1/health", timeout=0.5)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    try:
        yield url, app.state.fake
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def fake_server():
    with running_fake("full") as (url, state):
        yield url, state


@pytest.fixture
def sample_doc(tmp_path: Path) -> Path:
    path = tmp_path / "strix.md"
    path.write_text(SAMPLE_DOC, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the learned lexicon out of the real home directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("LEMONADE_BASE_URL", raising=False)
    return home
