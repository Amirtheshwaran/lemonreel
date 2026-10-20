"""Scene artwork: generate, have the writer look at it, and redraw if it misses the brief.

When the Omni model's LLM has vision (``vision`` label), each image is shown back to it
together with the scene description. It scores the match from 1 to 5 and suggests a better
prompt; low scores trigger a redraw with a new seed. The best attempt wins.
"""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from lemonreel import graphics
from lemonreel.config import ASPECTS, ReelConfig
from lemonreel.lemonade import LemonadeClient, LemonadeError
from lemonreel.llm_json import JSONExtractionError, extract_json_object
from lemonreel.omni import ModelInfo, Toolkit
from lemonreel.storyboard import Scene

Log = Callable[[str], None]

CRITIC_PROMPT = """You are reviewing an AI-generated illustration for one scene of an explainer video.

Intended picture: {visual}
Scene headline: {caption}

Score how well the image shows the intended picture, from 1 to 5:
5 = clearly shows it and looks polished; 4 = good with minor issues; 3 = acceptable;
2 = mostly wrong subject, garbled text or broken anatomy; 1 = unrelated or unusable.

Reply with JSON only: {{"score": <1-5>, "problems": "<short>", "better_prompt": "<an improved image prompt, or empty>"}}"""


@dataclass
class ImageAttempt:
    number: int
    prompt: str
    seed: int | None
    path: str
    seconds: float
    score: int | None = None
    problems: str = ""


@dataclass
class SceneArt:
    path: str
    attempts: list[ImageAttempt] = field(default_factory=list)
    fallback: bool = False
    error: str = ""

    @property
    def best_score(self) -> int | None:
        scores = [a.score for a in self.attempts if a.score is not None]
        return max(scores) if scores else None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["best_score"] = self.best_score
        return data


def plan_size(model: ModelInfo, aspect: str) -> str:
    """Choose a generation size near the model's native resolution, in the video's aspect.

    Dimensions are multiples of 64 and stay under the model's ``soft_pixel_cap``.
    """
    defaults = model.image_defaults or {}
    native_w = int(defaults.get("width") or 512)
    native_h = int(defaults.get("height") or 512)
    area = max(native_w * native_h, 256 * 256) * 1.3
    cap = defaults.get("soft_pixel_cap")
    if cap:
        area = min(area, float(cap))
    area = min(area, 1024 * 1024)
    aw, ah = ASPECTS[aspect]
    width = math.sqrt(area * aw / ah)
    height = width * ah / aw
    w = max(256, int(round(width / 64)) * 64)
    h = max(256, int(round(height / 64)) * 64)
    while cap and w * h > cap and min(w, h) > 256:
        if w >= h:
            w -= 64
        else:
            h -= 64
    return f"{w}x{h}"


def build_prompt(scene: Scene, style: str) -> str:
    visual = scene.visual.strip().rstrip(".")
    return f"{visual}. {style}. Highly detailed, no text, no lettering, no watermark."


def _data_url(png: bytes, max_side: int = 448) -> str:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def critique(
    client: LemonadeClient, model: str, png: bytes, scene: Scene, note: str = ""
) -> tuple[int, str, str]:
    """Ask the vision-capable writer to grade an image. Returns (score, problems, better_prompt)."""
    result = client.chat(
        model,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": CRITIC_PROMPT.format(visual=scene.visual, caption=scene.caption),
                    },
                    {"type": "image_url", "image_url": {"url": _data_url(png)}},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=300,
        json_mode=True,
        note=note,
    )
    try:
        data = extract_json_object(result.text)
        score = int(round(float(data.get("score", 3))))
    except (JSONExtractionError, TypeError, ValueError):
        return 3, "critic reply was not readable", ""
    score = min(5, max(1, score))
    return (
        score,
        str(data.get("problems") or "").strip(),
        str(data.get("better_prompt") or "").strip(),
    )


def draw_scene(
    client: LemonadeClient,
    toolkit: Toolkit,
    cfg: ReelConfig,
    scene: Scene,
    index: int,
    scene_dir: Path,
    style: str,
    log: Log = print,
) -> SceneArt:
    """Produce the final picture for one scene inside ``scene_dir``."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    final = scene_dir / "image.png"
    if toolkit.illustrator is None:
        graphics.slide(scene.caption, cfg.frame_size, cfg.language).save(final)
        return SceneArt(path=str(final), fallback=True, error="no image model in this toolkit")

    model = toolkit.illustrator
    size = plan_size(model, cfg.aspect)
    steps = cfg.image_steps or model.image_defaults.get("steps")
    cfg_scale = model.image_defaults.get("cfg_scale")
    base_seed = (cfg.seed if cfg.seed is not None else 1234) + index * 101
    prompt = build_prompt(scene, style)
    art = SceneArt(path=str(final))
    best: tuple[int, bytes] | None = None
    max_attempts = cfg.max_image_attempts if (cfg.critique and toolkit.can_see) else 1

    for number in range(1, max_attempts + 1):
        seed = base_seed + (number - 1) * 7919
        started = client.tracer.now()
        try:
            png = client.generate_image(
                model.id,
                prompt,
                size=size,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                note=f"scene {index + 1} attempt {number}",
            )
        except LemonadeError as exc:
            log(f"Scene {index + 1}: image generation failed ({exc})")
            art.error = str(exc)
            break
        attempt_path = scene_dir / f"attempt-{number}.png"
        attempt_path.write_bytes(png)
        attempt = ImageAttempt(
            number=number,
            prompt=prompt,
            seed=seed,
            path=str(attempt_path),
            seconds=round(client.tracer.now() - started, 2),
        )
        art.attempts.append(attempt)
        if max_attempts == 1:
            best = (0, png)
            break
        try:
            score, problems, better = critique(
                client, toolkit.writer.id, png, scene, note=f"critique scene {index + 1}"
            )
        except LemonadeError as exc:
            log(f"Scene {index + 1}: critique unavailable ({exc}); keeping the image")
            best = best or (0, png)
            break
        attempt.score, attempt.problems = score, problems
        if best is None or score > best[0]:
            best = (score, png)
        detail = f" ({problems})" if problems and score < cfg.min_score else ""
        log(f"Scene {index + 1}: attempt {number} scored {score}/5{detail}")
        if score >= cfg.min_score:
            break
        if better:
            prompt = build_prompt(Scene(caption=scene.caption, visual=better), style)

    if best is None:
        graphics.slide(scene.caption, cfg.frame_size, cfg.language).save(final)
        art.fallback = True
        return art

    png = best[1]
    if cfg.upscale and toolkit.upscaler is not None:
        try:
            png = client.upscale_image(toolkit.upscaler.id, png, note=f"upscale scene {index + 1}")
        except LemonadeError as exc:
            log(f"Scene {index + 1}: upscaling skipped ({exc})")
    final.write_bytes(png)
    return art
