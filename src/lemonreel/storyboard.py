"""Write the storyboard: the scene-by-scene script for the video.

The writer LLM turns the source into JSON (title, visual style and scenes). The reply is
validated and normalized; if it can't be parsed, the parse error is sent back so the model
can fix its own output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lemonreel.config import ReelConfig
from lemonreel.ingest import SourceDoc, split_into_chunks
from lemonreel.lemonade import LemonadeClient
from lemonreel.llm_json import JSONExtractionError, extract_json_object

Log = Callable[[str], None]


@dataclass
class Line:
    speaker: str  # "A" or "B"
    text: str


@dataclass
class Scene:
    caption: str
    visual: str
    narration: str = ""
    lines: list[Line] = field(default_factory=list)

    @property
    def spoken_text(self) -> str:
        if self.lines:
            return " ".join(line.text for line in self.lines)
        return self.narration


@dataclass
class Storyboard:
    title: str
    style: str
    scenes: list[Scene]
    mode: str = "narrator"
    language: str = "en"
    hosts: list[str] = field(default_factory=lambda: ["Alex", "Sam"])
    source_title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Storyboard:
        scenes = []
        for s in data.get("scenes") or []:
            lines = [
                Line(str(x.get("speaker", "A")), str(x.get("text", "")))
                for x in s.get("lines") or []
            ]
            scenes.append(
                Scene(
                    caption=str(s.get("caption", "")),
                    visual=str(s.get("visual", "")),
                    narration=str(s.get("narration", "")),
                    lines=lines,
                )
            )
        return cls(
            title=str(data.get("title", "")),
            style=str(data.get("style", "")),
            scenes=scenes,
            mode=str(data.get("mode", "narrator")),
            language=str(data.get("language", "en")),
            hosts=list(data.get("hosts") or ["Alex", "Sam"]),
            source_title=str(data.get("source_title", "")),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Storyboard:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


SYSTEM_PROMPT = """You are LemonReel's head writer. You turn source material into a short, \
accurate and engaging narrated explainer video. Every scene is one illustration plus spoken words.

Rules:
- Stay faithful to the source. Never invent facts, numbers, names or quotes.
- Write for the ear: short sentences, plain words, natural rhythm. No markdown, lists, URLs or emojis.
- Say symbols as words ("percent", not "%").
- The first scene hooks the viewer; the last scene wraps up with a clear takeaway.
- "visual" is a prompt for an image generator. Describe ONE concrete picture: subject, setting, \
composition, mood. Never ask for text, letters, numbers, logos, charts, diagrams or screenshots.
- "caption" is an on-screen headline of 2 to 6 words.
- Reply with a single JSON object and nothing else."""

NARRATOR_SCHEMA = """{
  "title": "video title, at most 8 words",
  "scenes": [
    {"caption": "2-6 word headline", "narration": "what the narrator says", "visual": "image prompt"}
  ]
}"""

DUO_SCHEMA = """{
  "title": "video title, at most 8 words",
  "hosts": ["first host's first name", "second host's first name"],
  "scenes": [
    {"caption": "2-6 word headline", "visual": "image prompt",
     "lines": [{"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."}]}
  ]
}"""


def _user_prompt(doc: SourceDoc, cfg: ReelConfig, source_text: str) -> str:
    if cfg.mode == "duo":
        voice_rules = (
            "Write it as a lively two-host conversation (speakers A and B) in the style of a "
            "podcast: A explains, B asks the questions a newcomer would ask and reacts. "
            f"Each scene has 2 to 4 short lines, about {cfg.words_per_scene} words per scene in total."
        )
        schema = DUO_SCHEMA
    else:
        voice_rules = (
            f"One narrator. Each scene's narration is about {cfg.words_per_scene} words "
            "(two or three sentences)."
        )
        schema = NARRATOR_SCHEMA
    if doc.is_topic:
        material = (
            f"TOPIC: {source_text}\n\nThere is no source document. Explain the topic from "
            "well-established knowledge only, and stay general where you are unsure."
        )
    else:
        material = f'SOURCE ({doc.kind}, titled "{doc.title}"):\n<<<\n{source_text}\n>>>'
    return f"""Create a storyboard with exactly {cfg.scenes} scenes for an explainer video.
Audience: {cfg.audience}.
Language of the title, captions and spoken words: {cfg.language_name}. Keep "visual" prompts in English.
{voice_rules}

{material}

Return JSON shaped like this:
{schema}"""


_MD_CHARS_RE = re.compile(r"[*_#`>\[\]]")


def _clean_spoken(text: str) -> str:
    text = _MD_CHARS_RE.sub("", str(text))
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _clip_words(text: str, limit: int) -> str:
    words = str(text).split()
    return " ".join(words[:limit]).strip(" .,:;-")


def normalize(data: dict[str, Any], cfg: ReelConfig, doc: SourceDoc) -> Storyboard:
    """Validate the writer's JSON and fill in anything missing."""
    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise JSONExtractionError('The JSON has no "scenes" list.')
    scenes: list[Scene] = []
    for raw in raw_scenes[: cfg.scenes]:
        if not isinstance(raw, dict):
            continue
        lines: list[Line] = []
        if cfg.mode == "duo":
            for item in raw.get("lines") or []:
                if isinstance(item, dict) and str(item.get("text", "")).strip():
                    speaker = str(item.get("speaker", "A")).strip().upper()[:1]
                    lines.append(Line("B" if speaker == "B" else "A", _clean_spoken(item["text"])))
            if not lines and raw.get("narration"):
                lines = [Line("A", _clean_spoken(raw["narration"]))]
            narration = " ".join(line.text for line in lines)
        else:
            narration = _clean_spoken(raw.get("narration") or raw.get("text") or "")
            if not narration and raw.get("lines"):
                narration = " ".join(
                    _clean_spoken(x.get("text", "")) for x in raw["lines"] if isinstance(x, dict)
                )
        if not narration:
            continue
        caption = _clip_words(_clean_spoken(raw.get("caption") or ""), 6) or _clip_words(
            narration, 5
        )
        visual = str(raw.get("visual") or raw.get("image") or "").strip()
        if not visual:
            visual = f"An evocative illustration representing: {caption}"
        scenes.append(Scene(caption=caption, visual=visual, narration=narration, lines=lines))
    if not scenes:
        raise JSONExtractionError("None of the scenes had spoken text.")
    hosts = data.get("hosts")
    if not (isinstance(hosts, list) and len(hosts) >= 2 and all(isinstance(h, str) for h in hosts)):
        hosts = ["Alex", "Sam"]
    title = _clip_words(_clean_spoken(data.get("title") or ""), 10) or doc.title or "Explainer"
    return Storyboard(
        title=title,
        style=cfg.style,
        scenes=scenes,
        mode=cfg.mode,
        language=cfg.language,
        hosts=[h.strip()[:20] for h in hosts[:2]],
        source_title=doc.title,
    )


def condense(
    client: LemonadeClient, model: str, doc: SourceDoc, max_chars: int, log: Log = print
) -> str:
    """Shrink long sources to their key points so they fit the writer's context window."""
    text = doc.text
    if len(text) <= max_chars:
        return text
    chunks = split_into_chunks(text, max(2000, max_chars // 2))
    log(f"Source is {len(text):,} characters; condensing {len(chunks)} parts first")
    notes: list[str] = []
    budget = max(400, max_chars // len(chunks))
    for i, chunk in enumerate(chunks, 1):
        result = client.chat(
            model,
            [
                {
                    "role": "system",
                    "content": "You extract the key facts from documents. Be faithful and concise.",
                },
                {
                    "role": "user",
                    "content": (
                        f'Part {i} of {len(chunks)} of "{doc.title}". List its key facts, ideas '
                        f"and numbers as short bullet points (under {budget // 6} words total).\n\n"
                        f"<<<\n{chunk}\n>>>"
                    ),
                },
            ],
            temperature=0.2,
            note=f"condense {i}/{len(chunks)}",
        )
        notes.append(result.text.strip())
    return "\n".join(notes)[:max_chars]


def reply_budget(cfg: ReelConfig) -> int:
    """Tokens to allow for the storyboard reply: spoken words plus prompts and JSON overhead."""
    tokens = cfg.scenes * (cfg.words_per_scene + 45) * 2.2
    if cfg.base_language in ("ja", "zh", "hi"):
        tokens *= 1.8
    return int(max(1600, min(tokens, 6000)))


def source_budget(cfg: ReelConfig, context_length: int | None) -> int:
    """Characters of source text that fit next to the prompt and the reply."""
    budget = cfg.max_source_chars
    if context_length:
        room = context_length - reply_budget(cfg) - 900  # system prompt, schema, instructions
        budget = max(1500, min(budget, int(room * 3.2)))  # ~3.2 characters per token
    return budget


def write_storyboard(
    client: LemonadeClient,
    model: str,
    doc: SourceDoc,
    cfg: ReelConfig,
    log: Log = print,
    attempts: int = 3,
    context_length: int | None = None,
) -> Storyboard:
    source_text = condense(client, model, doc, source_budget(cfg, context_length), log)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(doc, cfg, source_text)},
    ]
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        result = client.chat(
            model,
            messages,
            temperature=0.6,
            json_mode=True,
            max_tokens=reply_budget(cfg),
            note=f"storyboard attempt {attempt}",
        )
        try:
            board = normalize(extract_json_object(result.text), cfg, doc)
        except JSONExtractionError as exc:
            last_error = exc
            log(f"Storyboard attempt {attempt} was not usable ({exc}); asking the writer to fix it")
            messages += [
                {"role": "assistant", "content": result.text[:6000]},
                {
                    "role": "user",
                    "content": (
                        f"That reply could not be used: {exc}. Reply again with ONLY the corrected "
                        f"JSON object, exactly {cfg.scenes} scenes."
                    ),
                },
            ]
            continue
        if len(board.scenes) < cfg.scenes:
            log(
                f"Writer returned {len(board.scenes)} of {cfg.scenes} scenes; continuing with those"
            )
        return board
    raise RuntimeError(f"The writer model could not produce a valid storyboard: {last_error}")
