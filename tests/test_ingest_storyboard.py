import pytest

from lemonreel.config import ReelConfig
from lemonreel.ingest import (
    SourceDoc,
    SourceError,
    html_to_text,
    load_source,
    markdown_to_text,
    split_into_chunks,
)
from lemonreel.lemonade import LemonadeClient
from lemonreel.llm_json import JSONExtractionError
from lemonreel.omni import ModelInfo
from lemonreel.storyboard import Storyboard, normalize, write_storyboard
from lemonreel.visuals import plan_size

MD = """---
title: front matter
---
# LemonReel Guide

Some **bold** and _italic_ text with a [link](https://example.com) and ![img](x.png).

```python
print("code")
```

| Col | Val |
|-----|-----|
| a   | 1   |

- first item
- second item
"""


def test_markdown_to_text():
    text = markdown_to_text(MD)
    assert "front matter" not in text
    assert "LemonReel Guide." in text
    assert "Some bold and italic text with a link and ." in text
    assert "print" not in text and "[code example omitted]" in text
    assert "first item" in text and "- first" not in text


def test_html_to_text():
    title, text = html_to_text(
        "<html><head><title>Hello</title><style>x{}</style></head>"
        "<body><nav>menu</nav><h1>Big</h1><p>One &amp; two</p><script>evil()</script></body></html>"
    )
    assert title == "Hello"
    assert "Big" in text and "One & two" in text
    assert "menu" not in text and "evil" not in text


def test_load_source_kinds(tmp_path):
    md = tmp_path / "guide.md"
    md.write_text(MD)
    doc = load_source(str(md))
    assert doc.kind == "markdown" and doc.title == "LemonReel Guide"
    topic = load_source("how NPUs speed up local AI")
    assert topic.is_topic
    with pytest.raises(SourceError):
        load_source(str(tmp_path / "missing" / "file.md"))
    wav = tmp_path / "memo.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(SourceError, match="speech-recognition"):
        load_source(str(wav))


def test_split_into_chunks():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    chunks = split_into_chunks(text, 600)
    assert all(len(c) <= 600 for c in chunks)
    assert "".join(chunks).count("Paragraph") == 10


DOC = SourceDoc(title="Doc", text="x", kind="text", origin="test")


def test_normalize_narrator_cleans_and_fills():
    cfg = ReelConfig(scenes=2)
    board = normalize(
        {
            "title": "**Big** Title",
            "scenes": [
                {
                    "caption": "A very long caption that goes on and on",
                    "narration": "Read *this* at https://x.y now.",
                },
                {"narration": "Second scene text."},
                {"narration": "Third scene is dropped."},
            ],
        },
        cfg,
        DOC,
    )
    assert board.title == "Big Title"
    assert len(board.scenes) == 2
    assert board.scenes[0].caption == "A very long caption that goes"
    assert board.scenes[0].narration == "Read this at now."
    assert board.scenes[1].caption and board.scenes[1].visual
    assert board.style == cfg.style


def test_normalize_duo():
    cfg = ReelConfig(scenes=1, mode="duo")
    board = normalize(
        {
            "hosts": ["Maya", "Leo"],
            "scenes": [
                {
                    "caption": "Hi",
                    "visual": "v",
                    "lines": [{"speaker": "a", "text": "Hello."}, {"speaker": "B", "text": "Hey!"}],
                }
            ],
        },
        cfg,
        DOC,
    )
    assert [ln.speaker for ln in board.scenes[0].lines] == ["A", "B"]
    assert board.scenes[0].spoken_text == "Hello. Hey!"
    assert board.hosts == ["Maya", "Leo"]


def test_normalize_rejects_empty():
    with pytest.raises(JSONExtractionError):
        normalize({"scenes": []}, ReelConfig(), DOC)


def test_storyboard_round_trip(tmp_path):
    cfg = ReelConfig(scenes=1, mode="duo")
    board = normalize(
        {"scenes": [{"caption": "c", "visual": "v", "lines": [{"speaker": "A", "text": "t"}]}]},
        cfg,
        DOC,
    )
    board.save(tmp_path / "s.json")
    again = Storyboard.load(tmp_path / "s.json")
    assert again == board


def test_write_storyboard_against_fake(fake_server):
    url, _ = fake_server
    doc = SourceDoc(
        title="Strix", text="Strix Halo is a chip with a big GPU. " * 5, kind="text", origin="t"
    )
    board = write_storyboard(
        LemonadeClient(url), "Qwen3.5-4B-MTP-GGUF", doc, ReelConfig(scenes=4), log=lambda m: None
    )
    assert len(board.scenes) == 4
    assert board.title == "How Local AI Works"


def test_plan_size_is_aligned_and_capped():
    sd = ModelInfo(
        id="SD-Turbo", image_defaults={"width": 512, "height": 512, "soft_pixel_cap": 525625}
    )
    for aspect in ("16:9", "9:16", "1:1"):
        w, h = (int(x) for x in plan_size(sd, aspect).split("x"))
        assert w % 64 == 0 and h % 64 == 0
        assert w * h <= 525625
    w, h = (int(x) for x in plan_size(sd, "16:9").split("x"))
    assert w > h
    flux = ModelInfo(
        id="Flux", image_defaults={"width": 256, "height": 256, "soft_pixel_cap": 2097152}
    )
    assert plan_size(flux, "1:1") == "320x320"


def test_budgets_respect_context_window():
    from lemonreel.storyboard import reply_budget, source_budget

    cfg = ReelConfig(scenes=6)
    assert reply_budget(cfg) >= 1600
    assert source_budget(cfg, None) == cfg.max_source_chars
    small = source_budget(cfg, 4096)
    assert 1500 <= small < cfg.max_source_chars
    assert reply_budget(ReelConfig(scenes=6, language="ja")) > reply_budget(cfg)
