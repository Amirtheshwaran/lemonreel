from lemonreel.cli import main
from lemonreel.config import ReelConfig
from lemonreel.lemonade import LemonadeClient
from lemonreel.media import media_duration
from lemonreel.omni import resolve_toolkit
from lemonreel.say import clean_script, make_voiceover, read_script, split_long, split_parts

SCRIPT = """# Intro (0:00)
[show the finished video]
This explainer was made on my own laptop. Strix Halo users will *love* it.

# Doctor (0:15)
[terminal: lemonreel doctor --smoke]
LemonReel uses one Lemonade Omni model for everything. See [the repo](https://example.com).
"""


def test_stage_directions_and_headings_are_not_spoken():
    parts = split_parts(clean_script(SCRIPT))
    assert parts == [
        "This explainer was made on my own laptop. Strix Halo users will love it.",
        "LemonReel uses one Lemonade Omni model for everything. See the repo.",
    ]


def test_read_script_accepts_text_or_file(tmp_path):
    assert read_script("Hello [pause] there") == "Hello  there"
    path = tmp_path / "s.md"
    path.write_text(SCRIPT)
    assert "show the finished video" not in read_script(str(path))


def test_split_long_keeps_sentences_whole():
    text = " ".join(f"Sentence {i} talks about local AI for a little while." for i in range(30))
    chunks = split_long(text, 200)
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks) == text
    assert all(c.endswith(".") for c in chunks)
    assert split_long("short", 200) == ["short"]


def test_voiceover_against_fake(fake_server, tmp_path):
    url, _ = fake_server
    client = LemonadeClient(url)
    toolkit = resolve_toolkit(client.models(show_all=True))
    result = make_voiceover(
        client,
        toolkit,
        ReelConfig(),
        clean_script(SCRIPT),
        tmp_path / "vo",
        pause=0.5,
        log=lambda m: None,
    )
    assert len(result.parts) == 2
    assert result.parts[0].fixes == {"Strix": "Stricks"}
    assert result.parts[0].wer == 0
    total = media_duration(result.wav)
    assert abs(total - (result.parts[0].duration + 0.5 + result.parts[1].duration)) < 0.05
    assert abs(result.parts[1].start - (result.parts[0].duration + 0.5)) < 0.01
    srt = result.srt.read_text()
    assert "Strix Halo" in srt and "tricks" not in srt
    timing = result.timing.read_text()
    assert "part-01.wav" in timing and "part-02.wav" in timing
    assert (tmp_path / "vo" / "part-02.wav").exists()


def test_say_cli(fake_server, tmp_path):
    url, _ = fake_server
    script = tmp_path / "demo.md"
    script.write_text(SCRIPT)
    assert main(["say", str(script), "--base-url", url, "--pause", "0.4"]) == 0
    assert (tmp_path / "demo-voiceover" / "voiceover.wav").exists()
    assert main(["say", "   ", "--base-url", url, "--out", str(tmp_path / "empty")]) == 1
