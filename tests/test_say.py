import pytest

from lemonreel.cli import main
from lemonreel.config import ReelConfig
from lemonreel.lemonade import LemonadeClient
from lemonreel.media import media_duration
from lemonreel.omni import resolve_toolkit
from lemonreel.say import (
    ScriptNotFound,
    _stamp,
    make_voiceover,
    parse_script,
    resolve_script,
    split_long,
)

SCRIPT = """# Intro (0:00)
[show the finished video]
This explainer was made on my own laptop. Strix Halo users will *love* it.

# Doctor (0:15)
[terminal: lemonreel doctor --smoke]
LemonReel uses one Lemonade Omni model for everything. See [the repo](https://example.com).
"""


def test_headings_and_stage_directions_are_not_spoken():
    assert parse_script(SCRIPT) == [
        "This explainer was made on my own laptop. Strix Halo users will love it.",
        "LemonReel uses one Lemonade Omni model for everything. See the repo.",
    ]


def test_direction_line_does_not_split_a_paragraph():
    text = "First I open the terminal.\n[cut to terminal]\nThen I type the command."
    assert parse_script(text) == ["First I open the terminal. Then I type the command."]


def test_heading_starts_a_new_part_without_blank_line():
    assert parse_script("Part one.\n## Next\nPart two.") == ["Part one.", "Part two."]


def test_only_real_markdown_headings_are_dropped():
    assert parse_script("#1 rule: keep it short.\n\n#LocalAI is trending.") == [
        "#1 rule: keep it short.",
        "#LocalAI is trending.",
    ]


def test_markdown_cleanup_keeps_identifiers_and_math():
    text = (
        "Rename snake_case to my_var in config_file.py.\n\n"
        "Two times three: 2*3=6, and 5 * 4 = 20.\n\n"
        "Visit https://example.com/my_page for **really** _good_ `lemonreel say` tips.\n\n"
        "- first bullet\n- second bullet\n\n"
        "![diagram](d.png) Read [the guide][guide] now.\n\n"
        "[guide]: https://example.com/guide\n\n"
        "---"
    )
    assert parse_script(text) == [
        "Rename snake_case to my_var in config_file.py.",
        "Two times three: 2*3=6, and 5 * 4 = 20.",
        "Visit https://example.com/my_page for really good lemonreel say tips.",
        "first bullet second bullet",
        "Read the guide now.",
    ]


def test_resolve_script_reads_files_and_text(tmp_path):
    path = tmp_path / "s.md"
    path.write_text(SCRIPT)
    text, found = resolve_script(str(path))
    assert found == path and "show the finished video" in text
    assert resolve_script("Hello there") == ("Hello there", None)


def test_resolve_script_never_crashes_on_typed_text():
    long_text = "This explainer was made on my own laptop without a single cloud call. " * 5
    assert resolve_script(long_text) == (long_text, None)
    assert resolve_script("~2x faster than the cloud.")[1] is None


def test_missing_file_is_an_error_not_narration(tmp_path):
    with pytest.raises(ScriptNotFound):
        resolve_script("scrpt.md")
    with pytest.raises(ScriptNotFound):
        resolve_script(str(tmp_path))  # a directory


def test_byte_order_marks_and_utf16(tmp_path):
    bom = tmp_path / "bom.md"
    bom.write_bytes("﻿# Title\nSpoken line.".encode())
    assert parse_script(resolve_script(str(bom))[0]) == ["Spoken line."]
    utf16 = tmp_path / "u16.txt"
    utf16.write_bytes("# Title\nSpoken line.".encode("utf-16"))
    assert parse_script(resolve_script(str(utf16))[0]) == ["Spoken line."]


def test_split_long_keeps_sentences_whole():
    text = " ".join(f"Sentence {i} talks about local AI for a little while." for i in range(30))
    chunks = split_long(text, 200)
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks) == text
    assert all(c.endswith(".") for c in chunks)
    assert split_long("short", 200) == ["short"]
    assert split_long("Version 3.5 is out. " * 30, 100)[0].startswith("Version 3.5 is out.")


def test_split_long_cjk_cuts_at_sentence_ends():
    text = "今天我们在笔记本电脑上运行人工智能。" * 20
    chunks = split_long(text, 60, joiner="")
    assert "".join(chunks) == text
    assert all(len(c) <= 60 and c.endswith("。") for c in chunks)


def test_stamp_rounds_before_splitting_minutes():
    assert _stamp(59.96) == "1:00.0"
    assert _stamp(75.04) == "1:15.0"


def test_voiceover_against_fake(fake_server, tmp_path):
    url, _ = fake_server
    client = LemonadeClient(url)
    toolkit = resolve_toolkit(client.models(show_all=True))
    folder = tmp_path / "vo"
    folder.mkdir()
    (folder / "part-07.wav").write_bytes(b"stale")
    script = SCRIPT + "\nhttps://example.com/only-a-link\n"
    result = make_voiceover(
        client, toolkit, ReelConfig(), script, folder, pause=0.5, log=lambda m: None
    )
    assert len(result.parts) == 2  # the link-only paragraph has nothing to say
    assert not (folder / "part-07.wav").exists()
    assert result.parts[0].fixes == {"Strix": "Stricks"}
    assert result.parts[0].wer == 0
    total = media_duration(result.wav)
    assert abs(total - (result.parts[0].duration + 0.5 + result.parts[1].duration)) < 0.05
    assert abs(result.parts[1].start - (result.parts[0].duration + 0.5)) < 0.01
    srt = result.srt.read_text()
    assert "Strix Halo" in srt and "tricks" not in srt
    timing = result.timing.read_text()
    assert "part-01.wav" in timing and "part-02.wav" in timing


def test_voiceover_works_without_a_chat_model(fake_server, tmp_path):
    url, _ = fake_server
    client = LemonadeClient(url)
    models = [m for m in client.models(show_all=True) if "chat" not in (m.get("labels") or [])]
    toolkit = resolve_toolkit(models, require_writer=False)
    assert toolkit.writer is None
    result = make_voiceover(
        client, toolkit, ReelConfig(), SCRIPT, tmp_path / "vo", log=lambda m: None
    )
    assert result.parts[0].fixes == {}  # nobody to ask for respellings
    assert result.wav.exists()


def test_say_cli(fake_server, tmp_path, capsys):
    url, _ = fake_server
    script = tmp_path / "demo.md"
    script.write_text(SCRIPT)
    assert main(["say", str(script), "--base-url", url, "--pause", "0.4"]) == 0
    assert (tmp_path / "demo-voiceover" / "voiceover.wav").exists()
    long_text = "This explainer was made on my own laptop without a single cloud call. " * 5
    assert main(["say", long_text, "--base-url", url, "--out", str(tmp_path / "typed")]) == 0
    assert main(["say", "missing.md", "--base-url", url]) == 1
    assert "not found" in capsys.readouterr().out
    assert main(["say", "   ", "--base-url", url, "--out", str(tmp_path / "empty")]) == 1
