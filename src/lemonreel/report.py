"""Run report: what was made, on which hardware, how long each engine was busy, and how the
self-checks went. Written as ``report.json`` (for tools) and ``report.html`` (for people)."""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from lemonreel.config import ReelConfig
    from lemonreel.omni import Toolkit
    from lemonreel.pipeline import SceneOutput
    from lemonreel.render import Timeline
    from lemonreel.storyboard import Storyboard

LANES = {
    "llm": ("Writer (LLM)", "#e0b100"),
    "image": ("Illustrator", "#1f9e8f"),
    "tts": ("Voice (TTS)", "#e2674a"),
    "asr": ("Listener (ASR)", "#5b6ad8"),
}
ROLE_LANE = {
    "writer": "llm",
    "illustrator": "image",
    "voice": "tts",
    "listener": "asr",
    "upscaler": "image",
}


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def build_report(
    *,
    version: str,
    cfg: ReelConfig,
    toolkit: Toolkit,
    server: dict[str, Any],
    devices: dict[str, dict[str, Any]],
    board: Storyboard,
    outputs: list[SceneOutput],
    timeline: Timeline,
    trace: list[dict[str, Any]],
    stage_times: dict[str, dict[str, float]],
    wall_seconds: float,
    video_name: str,
    workdir: Path,
) -> dict[str, Any]:
    def rel(path: str | None) -> str | None:
        if not path:
            return None
        try:
            return Path(path).resolve().relative_to(workdir.resolve()).as_posix()
        except ValueError:
            return path

    scenes = []
    for i, (scene, out) in enumerate(zip(board.scenes, outputs)):
        art, voice = out.art, out.voice
        scenes.append(
            {
                "index": i + 1,
                "caption": scene.caption,
                "narration": scene.spoken_text,
                "visual": scene.visual,
                "image": rel(art.path) if art else None,
                "score": art.best_score if art else None,
                "image_attempts": [
                    {
                        "number": a.number,
                        "score": a.score,
                        "seconds": a.seconds,
                        "problems": a.problems,
                        "image": rel(a.path),
                        "prompt": a.prompt,
                    }
                    for a in (art.attempts if art else [])
                ],
                "fallback_slide": bool(art and art.fallback),
                "audio": rel(voice.path) if voice else None,
                "duration": round(voice.duration, 2) if voice else None,
                "wer_first": voice.first_wer if voice else None,
                "wer_final": voice.wer if voice else None,
                "fixes": voice.fixes if voice else {},
                "heard": " ".join(c.final.heard or "" for c in voice.clips).strip()
                if voice
                else "",
                "takes": sum(len(c.takes) for c in voice.clips) if voice else 0,
            }
        )

    lanes: dict[str, dict[str, Any]] = {}
    for lane in LANES:
        events = [e for e in trace if e["lane"] == lane]
        lanes[lane] = {
            "requests": len(events),
            "failures": sum(1 for e in events if not e["ok"]),
            "busy_seconds": round(sum(e["end"] - e["start"] for e in events), 2),
            "models": sorted({e["model"] for e in events if e["model"]}),
        }

    redraws = sum(max(0, len(s["image_attempts"]) - 1) for s in scenes)
    fixes: dict[str, str] = {}
    for s in scenes:
        fixes.update(s["fixes"])
    roles = {}
    for role, model in toolkit.roles().items():
        if model is None:
            continue
        dev = devices.get(model.id, {})
        roles[role] = {
            "model": model.id,
            "recipe": model.recipe,
            "device": dev.get("device"),
            "backend": dev.get("backend"),
            "busy_seconds": lanes.get(ROLE_LANE[role], {}).get("busy_seconds"),
        }

    return {
        "lemonreel_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": board.title,
        "video": {
            "file": video_name,
            "duration_seconds": timeline.duration,
            "resolution": "x".join(map(str, cfg.frame_size)),
            "fps": cfg.fps,
            "scenes": len(board.scenes),
            "mode": board.mode,
            "language": board.language,
            "chapters": [{"start": round(s, 2), "title": t} for s, t in timeline.chapters],
        },
        "wall_seconds": round(wall_seconds, 1),
        "seconds_of_video_per_minute": round(timeline.duration / max(wall_seconds, 1e-6) * 60, 1),
        "server": server,
        "omni_model": toolkit.omni,
        "roles": roles,
        "capabilities": toolkit.summary()["capabilities"],
        "lanes": lanes,
        "stages": {
            k: {"start": round(v["start"], 2), "end": round(v["end"], 2)}
            for k, v in stage_times.items()
        },
        "quality": {
            "average_image_score": _mean([s["score"] for s in scenes]),
            "redraws": redraws,
            "average_wer_first_take": _mean([s["wer_first"] for s in scenes]),
            "average_wer_final": _mean([s["wer_final"] for s in scenes]),
            "pronunciation_fixes": fixes,
        },
        "scenes": scenes,
        "config": cfg.to_dict(),
        "trace": [{**e, "start": round(e["start"], 3), "end": round(e["end"], 3)} for e in trace],
    }


# -- HTML ------------------------------------------------------------------------------------

_CSS = """
:root{--bg:#fbfaf6;--card:#fff;--ink:#16191d;--muted:#5d6670;--line:#e7e3d8;--lemon:#ffd60a;--teal:#0e4c5c}
@media (prefers-color-scheme:dark){:root{--bg:#101315;--card:#181c1f;--ink:#eef1f3;--muted:#9aa5ad;--line:#2a3034}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:30px;line-height:1.2;margin:4px 0 6px}h2{font-size:19px;margin:36px 0 12px}
.kicker{color:var(--muted);font-size:13px;letter-spacing:.06em;text-transform:uppercase}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 18px}.chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 12px;font-size:13px}
.chip b{font-variant-numeric:tabular-nums}
video{display:block;width:100%;max-height:72vh;margin:0 auto;border-radius:14px;background:#000;box-shadow:0 10px 30px rgba(0,0,0,.15)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);font-size:14px;vertical-align:top}th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:0}code{font-size:13px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.scene{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.scene img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#222}
.scene .body{padding:12px 14px}.scene h3{margin:0 0 6px;font-size:15px}.scene p{margin:0 0 8px;color:var(--muted);font-size:13.5px}
.badges{display:flex;flex-wrap:wrap;gap:6px}.badge{font-size:12px;border-radius:6px;padding:2px 8px;background:rgba(127,127,127,.12)}
.good{background:rgba(31,158,143,.16)}.warn{background:rgba(226,103,74,.18)}
.muted{color:var(--muted)}footer{margin-top:40px;color:var(--muted);font-size:13px}
svg text{fill:var(--ink);font:12px system-ui,sans-serif}svg .axis{fill:var(--muted)}
"""


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "-"
    if s >= 60:
        return f"{int(s // 60)}m {int(round(s % 60)):02d}s"
    return f"{s:.1f}s"


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def _timeline_svg(report: dict[str, Any]) -> str:
    trace = report["trace"]
    if not trace:
        return '<p class="muted">No inference requests were recorded.</p>'
    end = max(e["end"] for e in trace) or 1.0
    left, width, lane_h, top = 150, 800, 30, 26
    lanes = [k for k in LANES if any(e["lane"] == k for e in trace)]
    height = top + lane_h * len(lanes) + 30
    x = lambda t: left + width * t / end  # noqa: E731
    step = 10 ** math.floor(math.log10(end / 5)) if end > 0 else 1
    for mult in (1, 2, 5, 10):
        if end / (step * mult) <= 8:
            step *= mult
            break
    parts = [
        f'<svg viewBox="0 0 {left + width + 20} {height}" width="100%" role="img" aria-label="Request timeline">'
    ]
    t = 0.0
    while t <= end + 1e-9:
        parts.append(
            f'<line x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{height - 24}" stroke="currentColor" stroke-opacity=".12"/>'
        )
        parts.append(
            f'<text class="axis" x="{x(t):.1f}" y="{height - 8}" text-anchor="middle">{_fmt_seconds(t)}</text>'
        )
        t += step
    for row, lane in enumerate(lanes):
        label, color = LANES[lane]
        y = top + row * lane_h
        model = ", ".join(report["lanes"][lane]["models"])
        parts.append(f'<text x="8" y="{y + 19}">{html.escape(label)}</text>')
        parts.append(f"<title>{html.escape(model)}</title>")
        for e in (ev for ev in trace if ev["lane"] == lane):
            w = max(1.5, x(e["end"]) - x(e["start"]))
            tip = f"{e['note'] or e['endpoint']} - {e['end'] - e['start']:.1f}s ({e['model']})"
            opacity = "1" if e["ok"] else ".35"
            parts.append(
                f'<rect x="{x(e["start"]):.1f}" y="{y + 5}" width="{w:.1f}" height="{lane_h - 10}" rx="3" '
                f'fill="{color}" fill-opacity="{opacity}"><title>{html.escape(tip)}</title></rect>'
            )
    parts.append("</svg>")
    return "".join(parts)


def render_html(report: dict[str, Any]) -> str:
    esc = html.escape
    video = report["video"]
    quality = report["quality"]
    server = report["server"]
    score = quality["average_image_score"]
    score_text = "-" if score is None else f"{score:.1f}/5"
    chips = [
        f"<span class='chip'>Video <b>{_fmt_seconds(video['duration_seconds'])}</b></span>",
        f"<span class='chip'>Made in <b>{_fmt_seconds(report['wall_seconds'])}</b></span>",
        f"<span class='chip'><b>{video['scenes']}</b> scenes &middot; {esc(video['resolution'])} &middot; {video['fps']} fps</span>",
        f"<span class='chip'>Image score <b>{score_text}</b></span>",
        f"<span class='chip'>Listen-back error <b>{_pct(quality['average_wer_first_take'])} &rarr; {_pct(quality['average_wer_final'])}</b></span>",
    ]
    rows = []
    for role, info in report["roles"].items():
        where = " / ".join(x for x in (info.get("device"), info.get("backend")) if x) or "-"
        rows.append(
            f"<tr><td>{esc(role.title())}</td><td><code>{esc(info['model'])}</code></td>"
            f"<td>{esc(info.get('recipe') or '-')}</td><td>{esc(where)}</td><td>{_fmt_seconds(info.get('busy_seconds'))}</td></tr>"
        )
    hw = [server.get("processor"), server.get("memory"), server.get("os")]
    gpus = (server.get("devices") or {}).get("amd_gpu") or []
    if isinstance(gpus, list):
        hw += [g.get("name") for g in gpus if isinstance(g, dict)]
    npu = (server.get("devices") or {}).get("amd_npu")
    if isinstance(npu, dict) and npu.get("available"):
        hw.append(f"NPU: {npu.get('name')} ({npu.get('family', '')})")
    hw_line = " &middot; ".join(esc(str(h)) for h in hw if h)

    cards = []
    for s in report["scenes"]:
        badges = []
        if s["score"] is not None:
            cls = "good" if s["score"] >= 4 else "warn" if s["score"] <= 2 else ""
            tries = len(s["image_attempts"])
            badges.append(
                f"<span class='badge {cls}'>Critic {s['score']}/5{f' after {tries} tries' if tries > 1 else ''}</span>"
            )
        if s["fallback_slide"]:
            badges.append("<span class='badge warn'>Text slide</span>")
        if s["wer_final"] is not None:
            cls = "good" if s["wer_final"] <= 0.1 else "warn" if s["wer_final"] > 0.25 else ""
            change = (
                f"{_pct(s['wer_first'])} &rarr; "
                if s["wer_first"] is not None and s["wer_first"] != s["wer_final"]
                else ""
            )
            badges.append(
                f"<span class='badge {cls}'>Heard back: {change}{_pct(s['wer_final'])} error</span>"
            )
        for word, spoken in s["fixes"].items():
            badges.append(
                f"<span class='badge good'>{esc(word)} &rarr; &ldquo;{esc(spoken)}&rdquo;</span>"
            )
        img = (
            f"<img src='{esc(s['image'])}' alt='{esc(s['visual'])}' loading='lazy'>"
            if s["image"]
            else ""
        )
        cards.append(
            f"<article class='scene'>{img}<div class='body'><h3>{s['index']:02d} &middot; {esc(s['caption'])}</h3>"
            f"<p>{esc(s['narration'])}</p><div class='badges'>{''.join(badges)}</div></div></article>"
        )
    fixes = quality["pronunciation_fixes"]
    fixes_html = (
        "<table><tr><th>Written</th><th>Spoken as</th></tr>"
        + "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in fixes.items())
        + "</table>"
        if fixes
        else "<p class='muted'>No pronunciation problems needed fixing in this video.</p>"
    )
    omni = (
        f"Omni model <code>{esc(report['omni_model'])}</code>"
        if report["omni_model"]
        else "Individually selected models"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(report["title"])} - LemonReel report</title><style>{_CSS}</style></head>
<body><main>
<div class="kicker">LemonReel run report</div>
<h1>{esc(report["title"])}</h1>
<div class="muted">{omni} on Lemonade {esc(str(server.get("version") or ""))} &middot; {esc(report["generated_at"])}</div>
<div class="chips">{"".join(chips)}</div>
<video src="{esc(video["file"])}" controls preload="metadata" poster="thumbnail.jpg">
<track kind="captions" src="captions.vtt" srclang="{esc(video["language"])}" label="Captions"></video>

<h2>Hardware and models</h2>
<p class="muted">{hw_line or "Hardware details were not reported by the server."}</p>
<table><tr><th>Role</th><th>Model</th><th>Engine</th><th>Device / backend</th><th>Busy</th></tr>{"".join(rows)}</table>

<h2>Who was busy when</h2>
<p class="muted">Every request LemonReel sent to Lemonade. Bars on different rows that overlap ran at the same time on different engines.</p>
<div class="panel">{_timeline_svg(report)}</div>

<h2>Scenes</h2>
<div class="grid">{"".join(cards)}</div>

<h2>Pronunciation fixes learned</h2>
{fixes_html}

<footer>Generated by LemonReel {esc(report["lemonreel_version"])}. Raw data: <a href="report.json">report.json</a>
&middot; captions: <a href="captions.srt">SRT</a> / <a href="captions.vtt">VTT</a> &middot; <a href="chapters.txt">chapters</a></footer>
</main></body></html>
"""


def write_report(workdir: Path, **kwargs: Any) -> tuple[Path, Path]:
    report = build_report(workdir=workdir, **kwargs)
    json_path = workdir / "report.json"
    html_path = workdir / "report.html"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path
