"""Everything LemonReel draws itself: captions, scene badges, title and closing cards.

Text is rendered with Pillow into transparent PNG overlays, so the bundled ffmpeg needs no
font or subtitle libraries and the result looks the same on every OS.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

LEMON = (255, 214, 10)
TEAL = (14, 76, 92)
INK = (18, 22, 28)
WHITE = (255, 255, 255)

_FONT_CANDIDATES = {
    "bold": [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ],
    "regular": [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ],
}
# Scripts the Latin fonts above can't draw.
_SCRIPT_FONTS = {
    "ja": [
        "C:/Windows/Fonts/YuGothB.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ],
    "zh": [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ],
    "hi": [
        "C:/Windows/Fonts/NirmalaB.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "/System/Library/Fonts/Kohinoor.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
    ],
}


@lru_cache(maxsize=64)
def font(size: int, weight: str = "bold", language: str = "en") -> ImageFont.ImageFont:
    override = os.environ.get("LEMONREEL_FONT")
    paths = ([override] if override else []) + _SCRIPT_FONTS.get(language.split("-")[0], [])
    paths += _FONT_CANDIDATES[weight]
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", "C:/Windows")
        paths = [p.replace("C:/Windows", windir) for p in paths]
    for path in paths:
        if path and Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def _text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> float:
    return draw.textlength(text, font=fnt)


def wrap(text: str, fnt: ImageFont.ImageFont, max_width: float, max_lines: int = 3) -> list[str]:
    """Greedy word wrap; falls back to per-character wrapping for scripts without spaces."""
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    units = text.split()
    joiner = " "
    if len(units) <= 1 and _text_width(probe, text, fnt) > max_width:
        units, joiner = list(text), ""
    lines: list[str] = []
    current = ""
    for unit in units:
        trial = f"{current}{joiner}{unit}" if current else unit
        if _text_width(probe, trial, fnt) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = unit
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,") + "\u2026"
    return lines


def _shadowed_text(draw, xy, text, fnt, fill=WHITE, shadow=(0, 0, 0, 160), offset=2):
    x, y = xy
    draw.text((x + offset, y + offset), text, font=fnt, fill=shadow)
    draw.text((x, y), text, font=fnt, fill=fill)


def caption_overlay(text: str, size: tuple[int, int], language: str = "en") -> Image.Image:
    """Full-frame transparent overlay with a subtitle box near the bottom."""
    w, h = size
    vertical = h > w
    fnt = font(max(18, int(min(w, h) * (0.052 if vertical else 0.046))), "bold", language)
    max_width = w * (0.84 if vertical else 0.78)
    lines = wrap(text, fnt, max_width, max_lines=3 if vertical else 2)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ascent, descent = fnt.getmetrics()
    line_h = int((ascent + descent) * 1.18)
    pad_x, pad_y = int(line_h * 0.6), int(line_h * 0.32)
    box_w = int(max(_text_width(draw, ln, fnt) for ln in lines)) + 2 * pad_x
    box_h = line_h * len(lines) + 2 * pad_y
    bottom = int(h * (0.80 if vertical else 0.93))
    left = (w - box_w) // 2
    top = bottom - box_h
    draw.rounded_rectangle(
        (left, top, left + box_w, bottom), radius=int(line_h * 0.35), fill=(10, 12, 16, 178)
    )
    for i, ln in enumerate(lines):
        lw = _text_width(draw, ln, fnt)
        draw.text(((w - lw) / 2, top + pad_y + i * line_h), ln, font=fnt, fill=WHITE)
    return img


def badge_overlay(
    number: int, caption: str, size: tuple[int, int], language: str = "en"
) -> Image.Image:
    """Scene headline chip shown in the top-left corner for the first seconds of a scene."""
    w, h = size
    unit = min(w, h)
    num_font = font(int(unit * 0.034), "bold")
    cap_font = font(int(unit * 0.042), "bold", language)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = int(unit * 0.05)
    lines = wrap(caption, cap_font, w * 0.6, max_lines=2)
    ascent, descent = cap_font.getmetrics()
    line_h = int((ascent + descent) * 1.15)
    num = f"{number:02d}"
    num_w = _text_width(draw, num, num_font)
    pad = int(unit * 0.022)
    chip_w = int(num_w + 2 * pad)
    text_w = int(max(_text_width(draw, ln, cap_font) for ln in lines))
    box_h = line_h * len(lines) + 2 * pad
    x0, y0 = margin, margin
    draw.rounded_rectangle(
        (x0, y0, x0 + chip_w + text_w + 3 * pad, y0 + box_h),
        radius=int(pad * 0.9),
        fill=(10, 12, 16, 170),
    )
    draw.rounded_rectangle(
        (x0, y0, x0 + chip_w, y0 + box_h), radius=int(pad * 0.9), fill=LEMON + (255,)
    )
    n_ascent, n_descent = num_font.getmetrics()
    draw.text((x0 + pad, y0 + (box_h - (n_ascent + n_descent)) / 2), num, font=num_font, fill=INK)
    for i, ln in enumerate(lines):
        draw.text((x0 + chip_w + 1.5 * pad, y0 + pad + i * line_h), ln, font=cap_font, fill=WHITE)
    return img


def _cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = size
    scale = max(w / img.width, h / img.height)
    resized = img.resize(
        (max(w, round(img.width * scale)), max(h, round(img.height * scale))), Image.LANCZOS
    )
    left = (resized.width - w) // 2
    top = (resized.height - h) // 2
    return resized.crop((left, top, left + w, top + h))


def _contain(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = size
    scale = min(w / img.width, h / img.height)
    return img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS
    )


def fit_canvas(img: Image.Image, size: tuple[int, int], tolerance: float = 0.12) -> Image.Image:
    """Fit an image into ``size``.

    Close aspect ratios are center-cropped. Very different ones (a square image in a
    vertical video, say) are shown whole over a blurred, darkened copy of themselves.
    """
    img = img.convert("RGB")
    target = size[0] / size[1]
    actual = img.width / img.height
    if abs(actual - target) / target <= tolerance:
        return _cover(img, size)
    background = _cover(img, size).filter(ImageFilter.GaussianBlur(radius=max(size) // 30))
    background = Image.blend(background, Image.new("RGB", size, (0, 0, 0)), 0.35)
    fg = _contain(img, (int(size[0] * 0.94), int(size[1] * 0.94)))
    background.paste(fg, ((size[0] - fg.width) // 2, (size[1] - fg.height) // 2))
    return background


def gradient(size: tuple[int, int], top=TEAL, bottom=(6, 24, 32)) -> Image.Image:
    w, h = size
    base = Image.new("RGB", (1, 256))
    for y in range(256):
        t = y / 255
        base.putpixel((0, y), tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)))
    return base.resize((w, h), Image.BICUBIC)


def _lemon_mark(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> None:
    draw.ellipse((cx - r * 1.15, cy - r, cx + r * 1.15, cy + r), fill=LEMON)
    draw.ellipse((cx + r * 0.95, cy - r * 0.18, cx + r * 1.35, cy + r * 0.18), fill=LEMON)
    draw.ellipse((cx - r * 0.55, cy - r * 0.55, cx - r * 0.15, cy - r * 0.2), fill=(255, 238, 140))


def slide(caption: str, size: tuple[int, int], language: str = "en") -> Image.Image:
    """Typographic stand-in used when no image model is available or generation fails."""
    img = gradient(size)
    draw = ImageDraw.Draw(img)
    w, h = size
    fnt = font(int(min(w, h) * 0.085), "bold", language)
    lines = wrap(caption, fnt, w * 0.8, max_lines=3)
    ascent, descent = fnt.getmetrics()
    line_h = int((ascent + descent) * 1.15)
    top = (h - line_h * len(lines)) / 2
    for i, ln in enumerate(lines):
        lw = _text_width(draw, ln, fnt)
        _shadowed_text(draw, ((w - lw) / 2, top + i * line_h), ln, fnt)
    _lemon_mark(draw, w / 2, top - line_h * 0.9, min(w, h) * 0.035)
    return img


def title_card(
    title: str,
    subtitle: str,
    size: tuple[int, int],
    background: Image.Image | None,
    language: str = "en",
) -> Image.Image:
    w, h = size
    if background is not None:
        img = _cover(background.convert("RGB"), size).filter(
            ImageFilter.GaussianBlur(radius=max(size) // 60)
        )
        img = Image.blend(img, Image.new("RGB", size, (0, 0, 0)), 0.55)
    else:
        img = gradient(size)
    draw = ImageDraw.Draw(img)
    unit = min(w, h)
    title_font = font(int(unit * (0.095 if w >= h else 0.085)), "bold", language)
    sub_font = font(int(unit * 0.038), "regular", language)
    lines = wrap(title, title_font, w * 0.82, max_lines=3)
    ascent, descent = title_font.getmetrics()
    line_h = int((ascent + descent) * 1.1)
    block_h = line_h * len(lines)
    top = (h - block_h) / 2 - unit * 0.04
    bar_w = unit * 0.12
    draw.rounded_rectangle(
        ((w - bar_w) / 2, top - unit * 0.06, (w + bar_w) / 2, top - unit * 0.045),
        radius=4,
        fill=LEMON,
    )
    for i, ln in enumerate(lines):
        lw = _text_width(draw, ln, title_font)
        _shadowed_text(draw, ((w - lw) / 2, top + i * line_h), ln, title_font, offset=3)
    sub_lines = wrap(subtitle, sub_font, w * 0.8, max_lines=2)
    sy = top + block_h + unit * 0.035
    s_ascent, s_descent = sub_font.getmetrics()
    for ln in sub_lines:
        lw = _text_width(draw, ln, sub_font)
        draw.text(((w - lw) / 2, sy), ln, font=sub_font, fill=(235, 235, 235))
        sy += (s_ascent + s_descent) * 1.2
    return img


def outro_card(lines: list[str], size: tuple[int, int]) -> Image.Image:
    """Closing card crediting the local models that made the video."""
    w, h = size
    img = gradient(size, top=(10, 30, 38), bottom=(4, 12, 16))
    draw = ImageDraw.Draw(img)
    unit = min(w, h)
    head_font = font(int(unit * 0.062), "bold")
    body_font = font(int(unit * 0.034), "regular")
    head = "Made 100% locally with Lemonade"
    head_lines = wrap(head, head_font, w * 0.85, max_lines=2)
    ascent, descent = head_font.getmetrics()
    line_h = int((ascent + descent) * 1.1)
    b_ascent, b_descent = body_font.getmetrics()
    body_h = int((b_ascent + b_descent) * 1.35)
    total = line_h * len(head_lines) + unit * 0.05 + body_h * len(lines)
    y = (h - total) / 2 + unit * 0.03
    _lemon_mark(draw, w / 2, y - unit * 0.07, unit * 0.04)
    for ln in head_lines:
        lw = _text_width(draw, ln, head_font)
        _shadowed_text(draw, ((w - lw) / 2, y), ln, head_font)
        y += line_h
    y += unit * 0.05
    for ln in lines:
        for wrapped in wrap(ln, body_font, w * 0.86, max_lines=2):
            lw = _text_width(draw, wrapped, body_font)
            draw.text(((w - lw) / 2, y), wrapped, font=body_font, fill=(214, 226, 230))
            y += body_h
    return img
