import json

import pytest
from conftest import SAMPLE_DOC
from fastapi.testclient import TestClient

from lemonreel.cli import main
from lemonreel.lemonade import LemonadeClient, LemonadeUnavailable, normalize_base_url
from lemonreel.studio.app import create_app


def test_normalize_base_url():
    assert normalize_base_url("localhost:13305") == "http://localhost:13305"
    assert normalize_base_url("http://box:8000/api/v1/") == "http://box:8000"
    assert normalize_base_url("https://host/v1") == "https://host"


def test_discover_reports_missing_server():
    with pytest.raises(LemonadeUnavailable, match="Start Lemonade Server"):
        LemonadeClient.discover(candidates=["http://127.0.0.1:9"])


def test_cli_doctor_and_voices(fake_server, capsys):
    url, _ = fake_server
    assert main(["doctor", "--base-url", url, "--smoke"]) == 0
    out = capsys.readouterr().out
    assert "LMX-Omni-5.5B-Lite" in out and "Ready." in out
    assert main(["voices"]) == 0
    assert "af_heart" in capsys.readouterr().out


def test_cli_make(fake_server, sample_doc, tmp_path, capsys):
    url, _ = fake_server
    code = main(
        [
            "make",
            str(sample_doc),
            "--base-url",
            url,
            "--scenes",
            "2",
            "--minutes",
            "0.4",
            "--out",
            str(tmp_path),
            "--no-outro",
            "--look",
            "watercolor",
        ]
    )
    assert code == 0
    assert list(tmp_path.glob("*/*.mp4"))
    config = json.loads(next(tmp_path.glob("*/config.json")).read_text())
    assert "watercolor" in config["resolved_style"]


def test_cli_storyboard_only_and_remake(fake_server, sample_doc, tmp_path, capsys):
    url, _ = fake_server
    assert (
        main(
            [
                "make",
                str(sample_doc),
                "--base-url",
                url,
                "--scenes",
                "2",
                "--out",
                str(tmp_path),
                "--storyboard-only",
            ]
        )
        == 0
    )
    folder = next(tmp_path.glob("*/storyboard.json")).parent
    assert not list(folder.glob("*.mp4"))
    assert (
        main(["remake", str(folder), "--base-url", url, "--scenes", "2", "--out", str(tmp_path)])
        == 0
    )
    assert list(folder.glob("*.mp4"))


def test_studio_job_flow(fake_server, tmp_path):
    url, _ = fake_server
    client = TestClient(create_app(tmp_path, base_url=url))
    status = client.get("/api/status").json()
    assert status["connected"] and status["toolkit"]["omni"] == "LMX-Omni-5.5B-Lite"
    assert client.get("/").status_code == 200

    bad = client.post("/api/jobs", data={"kind": "text", "text": "  ", "options": "{}"})
    assert bad.status_code == 400

    options = json.dumps({"scenes": 2, "minutes": 0.4, "mode": "narrator", "aspect": "1:1"})
    job = client.post(
        "/api/jobs", data={"kind": "text", "text": SAMPLE_DOC, "options": options}
    ).json()
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as stream:
        body = "".join(chunk for chunk in stream.iter_text())
    events = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: {") and '"kind"' in line
    ]
    done = next(e for e in events if e["kind"] == "done")
    video = client.get(done["base"] + done["data"]["video"])
    assert video.status_code == 200 and len(video.content) > 1000
    reels = client.get("/api/reels").json()
    assert reels and reels[0]["title"] == "How Local AI Works"


def test_studio_reports_missing_lemonade(tmp_path):
    client = TestClient(create_app(tmp_path, base_url="http://127.0.0.1:9"))
    status = client.get("/api/status").json()
    assert status["connected"] is False
    assert "languages" in status
