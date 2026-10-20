"""User-facing options and presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Visual looks. Any other string passed to --look is used verbatim as the style.
LOOKS: dict[str, str] = {
    "illustration": (
        "clean modern editorial illustration, soft gradients, warm lemon-yellow and deep teal "
        "palette, gentle cinematic lighting, uncluttered composition"
    ),
    "photo": "cinematic photograph, natural light, shallow depth of field, rich but realistic color",
    "watercolor": "loose watercolor painting on textured paper, soft washes, delicate ink outlines",
    "3d": "stylized 3D render, soft studio lighting, rounded shapes, pastel colors, high detail",
    "comic": "bold comic book art, clean inks, flat vibrant colors, dynamic composition",
    "blueprint": "technical blueprint illustration, white line art on deep blue paper, precise",
    "pixel": "detailed pixel art, 16-bit era palette, crisp pixels, cozy atmosphere",
}

# Kokoro voices that suit each language: (narrator / host A, host B).
DEFAULT_VOICES: dict[str, tuple[str, str]] = {
    "en": ("af_heart", "am_michael"),
    "en-gb": ("bf_emma", "bm_george"),
    "es": ("ef_dora", "em_alex"),
    "fr": ("ff_siwis", "ff_siwis"),
    "hi": ("hf_alpha", "hm_omega"),
    "it": ("if_sara", "im_nicola"),
    "ja": ("jf_alpha", "jm_kumo"),
    "pt": ("pf_dora", "pm_alex"),
    "zh": ("zf_xiaobei", "zm_yunjian"),
}

LANGUAGE_NAMES = {
    "en": "English",
    "en-gb": "British English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "pt": "Brazilian Portuguese",
    "zh": "Mandarin Chinese",
}

KOKORO_VOICES: dict[str, list[str]] = {
    "American English": [
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
    ],
    "British English": [
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
    ],
    "Spanish": ["ef_dora", "em_alex", "em_santa"],
    "French": ["ff_siwis"],
    "Hindi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "Italian": ["if_sara", "im_nicola"],
    "Japanese": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "Brazilian Portuguese": ["pf_dora", "pm_alex", "pm_santa"],
    "Mandarin Chinese": [
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ],
}

ASPECTS: dict[str, tuple[int, int]] = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1)}
WORDS_PER_MINUTE = 150


@dataclass
class ReelConfig:
    # Story
    scenes: int = 6
    minutes: float = 1.5
    mode: str = "narrator"  # narrator | duo
    language: str = "en"
    look: str = "illustration"
    audience: str = "curious people who are new to the topic"
    # Voices
    voice: str | None = None
    voice2: str | None = None
    speed: float = 1.0
    # Picture
    aspect: str = "16:9"
    hd: bool = False
    fps: int = 30
    captions: str = "burn"  # burn | soft | both | none
    title_card: bool = True
    outro: bool = True
    upscale: bool = False
    image_steps: int | None = None
    seed: int | None = None
    # Self-checks
    critique: bool = True
    min_score: int = 3
    max_image_attempts: int = 3
    listen_back: bool = True
    wer_threshold: float = 0.12
    max_voice_attempts: int = 2
    learn: bool = True
    lexicon_path: str | None = None
    # Execution
    parallel: bool = True
    max_source_chars: int = 12_000
    # Model selection
    omni: str | None = None
    writer: str | None = None
    illustrator: str | None = None
    voice_model: str | None = None
    listener: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("narrator", "duo"):
            raise ValueError("mode must be 'narrator' or 'duo'")
        if self.aspect not in ASPECTS:
            raise ValueError(f"aspect must be one of {', '.join(ASPECTS)}")
        if self.captions not in ("burn", "soft", "both", "none"):
            raise ValueError("captions must be burn, soft, both or none")
        if not 1 <= self.scenes <= 24:
            raise ValueError("scenes must be between 1 and 24")
        if self.minutes <= 0:
            raise ValueError("minutes must be positive")
        self.language = self.language.lower()

    @property
    def base_language(self) -> str:
        return self.language.split("-")[0]

    @property
    def language_name(self) -> str:
        return LANGUAGE_NAMES.get(
            self.language, LANGUAGE_NAMES.get(self.base_language, self.language)
        )

    @property
    def voices(self) -> tuple[str, str]:
        default = DEFAULT_VOICES.get(self.language) or DEFAULT_VOICES.get(
            self.base_language, DEFAULT_VOICES["en"]
        )
        return (self.voice or default[0], self.voice2 or default[1])

    @property
    def style(self) -> str:
        return LOOKS.get(self.look, self.look)

    @property
    def words_per_scene(self) -> int:
        return max(12, round(self.minutes * WORDS_PER_MINUTE / self.scenes))

    @property
    def frame_size(self) -> tuple[int, int]:
        short = 1080 if self.hd else 720
        w, h = ASPECTS[self.aspect]
        if w >= h:
            return (round(short * w / h / 2) * 2, short)
        return (short, round(short * h / w / 2) * 2)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resolved_voices"] = list(self.voices)
        data["resolved_style"] = self.style
        return data
