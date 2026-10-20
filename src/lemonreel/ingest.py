"""Turn whatever the user hands us into plain source text.

Supported sources:

* Markdown, plain text, reStructuredText and HTML files
* PDF files (``pip install lemonreel[pdf]``)
* web pages, and GitHub repository URLs (their README is used)
* voice memos in any common audio format, transcribed by the Omni model's listener
* a bare topic ("how NPUs speed up local AI"), written from the writer's own knowledge
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import httpx

TEXT_SUFFIXES = {".md", ".markdown", ".mdx", ".txt", ".text", ".rst"}
HTML_SUFFIXES = {".html", ".htm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".webm", ".flac", ".aac"}
PDF_SUFFIXES = {".pdf"}

Transcriber = Callable[[Path], str]


class SourceError(RuntimeError):
    pass


@dataclass
class SourceDoc:
    title: str
    text: str
    kind: str  # markdown | text | html | pdf | url | audio | topic
    origin: str
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def is_topic(self) -> bool:
        return self.kind == "topic"


# -- markdown ---------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^(```|~~~).*?^\1\s*$", re.MULTILINE | re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_REF_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_|`)(?=\S)(.+?)(?<=\S)\1")
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def markdown_to_text(md: str) -> str:
    """Strip Markdown syntax but keep the words and the paragraph structure."""
    text = _FRONT_MATTER_RE.sub("", md)
    text = _FENCE_RE.sub("\n[code example omitted]\n", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _REF_LINK_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = _HEADING_RE.sub(lambda m: "\n" + m.group(1).strip() + ".\n", text)
    for _ in range(2):
        text = _EMPHASIS_RE.sub(r"\2", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"[|:\-\s]+", stripped or "x"):
            continue  # table separators and horizontal rules
        stripped = re.sub(r"^[>*+\-]\s+|^\d+[.)]\s+", "", stripped)
        stripped = stripped.strip("|").replace("|", " - ")
        lines.append(stripped)
    text = "\n".join(lines)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def markdown_title(md: str) -> str | None:
    match = re.search(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", md, re.MULTILINE)
    if match:
        return markdown_to_text(match.group(1)).rstrip(".")
    return None


# -- html -------------------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"}
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "li",
    "br",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
    "blockquote",
    "table",
    "ul",
    "ol",
    "main",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> tuple[str, str]:
    """Return ``(title, text)`` for an HTML document."""
    parser = _TextExtractor()
    parser.feed(markup)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return parser.title.strip(), text.strip()


# -- loaders ----------------------------------------------------------------------------------

_GITHUB_REPO_RE = re.compile(r"^https?://github\.com/([^/\s]+)/([^/\s#?]+)/?$")


def fetch_url(url: str, timeout: float = 30.0) -> SourceDoc:
    repo = _GITHUB_REPO_RE.match(url.rstrip("/"))
    if repo:
        owner, name = repo.group(1), repo.group(2).removesuffix(".git")
        raw = f"https://raw.githubusercontent.com/{owner}/{name}/HEAD/README.md"
        try:
            response = httpx.get(raw, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceError(f"Could not download the README of {owner}/{name}: {exc}") from exc
        md = response.text
        return SourceDoc(
            title=markdown_title(md) or name,
            text=markdown_to_text(md),
            kind="url",
            origin=url,
            meta={"format": "markdown", "repo": f"{owner}/{name}"},
        )
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; lemonreel)"},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SourceError(f"Could not download {url}: {exc}") from exc
    content_type = response.headers.get("content-type", "")
    body = response.text
    if "html" in content_type or body.lstrip().lower().startswith(("<!doctype", "<html")):
        title, text = html_to_text(body)
        return SourceDoc(
            title=title or url, text=text, kind="url", origin=url, meta={"format": "html"}
        )
    if url.lower().endswith((".md", ".markdown")) or "markdown" in content_type:
        return SourceDoc(
            title=markdown_title(body) or url,
            text=markdown_to_text(body),
            kind="url",
            origin=url,
            meta={"format": "markdown"},
        )
    return SourceDoc(title=url, text=body.strip(), kind="url", origin=url, meta={"format": "text"})


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SourceError(
            "Reading PDFs needs the optional pypdf package: pip install 'lemonreel[pdf]'"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()


def load_source(source: str, transcriber: Transcriber | None = None) -> SourceDoc:
    """Load ``source`` (a path, URL or topic) into a :class:`SourceDoc`."""
    source = source.strip()
    if not source:
        raise SourceError("Nothing to make a video about: the source is empty.")
    if re.match(r"^https?://", source, re.IGNORECASE):
        doc = fetch_url(source)
    else:
        path = Path(source).expanduser()
        if path.is_file():
            doc = load_file(path, transcriber)
        elif len(source) < 400 and ("/" in source or "\\" in source) and " " not in source:
            raise SourceError(f"File not found: {source}")
        else:
            topic = " ".join(source.split())
            return SourceDoc(title=topic[:80], text=topic, kind="topic", origin="topic")
    if not doc.text.strip():
        raise SourceError(f"No readable text was found in {source}.")
    return doc


def load_file(path: Path, transcriber: Transcriber | None = None) -> SourceDoc:
    suffix = path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        if transcriber is None:
            raise SourceError(
                "Voice memos need a speech-recognition model; this Omni model has none. "
                "Pull one with `lemonade pull Whisper-Base`."
            )
        text = transcriber(path).strip()
        if not text:
            raise SourceError(f"No speech was recognized in {path.name}.")
        return SourceDoc(title="Voice memo", text=text, kind="audio", origin=str(path))
    if suffix in PDF_SUFFIXES:
        text = _read_pdf(path)
        return SourceDoc(title=path.stem.replace("_", " "), text=text, kind="pdf", origin=str(path))
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in HTML_SUFFIXES:
        title, text = html_to_text(raw)
        return SourceDoc(title=title or path.stem, text=text, kind="html", origin=str(path))
    if suffix in {".md", ".markdown", ".mdx"}:
        return SourceDoc(
            title=markdown_title(raw) or path.stem.replace("-", " ").replace("_", " "),
            text=markdown_to_text(raw),
            kind="markdown",
            origin=str(path),
        )
    return SourceDoc(
        title=path.stem.replace("-", " ").replace("_", " "),
        text=raw.strip(),
        kind="text",
        origin=str(path),
    )


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries into chunks of at most ``max_chars`` characters."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            head, para = para[:cut].strip(), para[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks
