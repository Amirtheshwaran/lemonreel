"""End-to-end runs against the fake Lemonade server. These produce real MP4 files."""

import json
import subprocess

from conftest import running_fake

from lemonreel.config import ReelConfig
from lemonreel.lemonade import LemonadeClient
from lemonreel.lexicon import default_learned_path
from lemonreel.media import ffmpeg_exe, media_duration
from lemonreel.pipeline import Reel


def _streams(video):
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True
    )
    return proc.stderr


def test_narrator_reel(fake_server, sample_doc, tmp_path):
    url, state = fake_server
    events = []
    reel = Reel(
        ReelConfig(scenes=3, minutes=0.6),
        LemonadeClient(url),
        out_root=tmp_path / "out",
        on_event=events.append,
    )
    result = reel.run(str(sample_doc))

    assert result.video.exists()
    duration = media_duration(result.video)
    assert abs(duration - result.timeline.duration) < 0.5
    info = _streams(result.video)
    assert "Video: h264" in info and "Audio: aac" in info
    assert "Chapter" in info

    # The listener always gets 16 kHz audio.
    assert set(state.asr_rates) == {16000}

    report = json.loads(result.report_json.read_text())
    assert report["omni_model"] == "LMX-Omni-5.5B-Lite"
    assert report["quality"]["redraws"] == 1  # the fake critic rejects one picture
    assert report["quality"]["pronunciation_fixes"] == {"Strix": "Stricks"}
    scene1 = report["scenes"][0]
    assert scene1["wer_first"] > 0 and scene1["wer_final"] == 0
    assert report["roles"]["listener"]["device"] == "npu"
    lanes = report["lanes"]
    assert lanes["image"]["requests"] >= 4 and lanes["asr"]["requests"] >= 3

    # Learned pronunciations are remembered for the next video.
    assert json.loads(default_learned_path().read_text()) == {"Strix": "Stricks"}

    srt = (result.workdir / "captions.srt").read_text()
    assert "Strix Halo" in srt and "tricks" not in srt
    assert (result.workdir / "chapters.txt").read_text().startswith("00:00 Intro")
    assert (result.workdir / "thumbnail.jpg").exists()
    kinds = {e.kind for e in events}
    assert {"stage", "toolkit", "storyboard", "art", "voice", "done"} <= kinds


def test_duo_vertical_with_soft_subtitles(fake_server, sample_doc, tmp_path):
    url, _ = fake_server
    cfg = ReelConfig(scenes=2, minutes=0.5, mode="duo", aspect="9:16", captions="both", outro=False)
    result = Reel(cfg, LemonadeClient(url), out_root=tmp_path).run(str(sample_doc))
    info = _streams(result.video)
    assert "720x1280" in info
    assert "Subtitle: mov_text" in info
    board = json.loads((result.workdir / "storyboard.json").read_text())
    assert board["hosts"] == ["Maya", "Leo"]
    assert all(len(s["lines"]) == 2 for s in board["scenes"])


def test_minimal_models_fall_back_gracefully(sample_doc, tmp_path):
    with running_fake("minimal") as (url, _):
        result = Reel(
            ReelConfig(scenes=2, minutes=0.4), LemonadeClient(url), out_root=tmp_path
        ).run(str(sample_doc))
    report = json.loads(result.report_json.read_text())
    assert report["omni_model"] is None
    assert all(s["fallback_slide"] for s in report["scenes"])
    assert all(s["wer_final"] is None for s in report["scenes"])
    assert result.video.exists()


def test_segment_timestamps_only(sample_doc, tmp_path):
    with running_fake("full", segments_only=True) as (url, _):
        cfg = ReelConfig(scenes=2, minutes=0.4, critique=False)
        result = Reel(cfg, LemonadeClient(url), out_root=tmp_path).run(str(sample_doc))
    srt = (result.workdir / "captions.srt").read_text()
    assert srt.count("-->") >= 2


def test_storyboard_only_then_remake_reuses_work(fake_server, sample_doc, tmp_path):
    url, state = fake_server
    cfg = ReelConfig(scenes=2, minutes=0.4)
    first = Reel(cfg, LemonadeClient(url), out_root=tmp_path).run(
        str(sample_doc), storyboard_only=True
    )
    assert first.video is None
    board_path = first.workdir / "storyboard.json"
    board = json.loads(board_path.read_text())
    board["scenes"][1]["narration"] = "An edited line about Lemonade."
    board_path.write_text(json.dumps(board))

    done = Reel(cfg, LemonadeClient(url), out_root=tmp_path).run(
        storyboard=board_path, workdir=first.workdir
    )
    assert done.video.exists()
    images_before = sum(1 for r in state.requests if r["endpoint"] == "images")

    events = []
    Reel(cfg, LemonadeClient(url), out_root=tmp_path, on_event=events.append).run(
        storyboard=board_path, workdir=first.workdir
    )
    images_after = sum(1 for r in state.requests if r["endpoint"] == "images")
    assert images_after == images_before  # every picture came from the cache
    assert all(e.data.get("cached") for e in events if e.kind in ("art", "voice"))
