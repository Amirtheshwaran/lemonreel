"""Map a Lemonade Omni model (or loose downloaded models) onto the roles LemonReel needs.

A Lemonade Omni model is a collection (``recipe: "collection.omni"``) of component
models: a planner LLM, an image model, a speech-recognition model and a speech model.
LemonReel reads the components and their labels from ``GET /v1/models`` and assigns:

=============  =================================  ==========================
Role           Picked by                          Used for
=============  =================================  ==========================
``writer``     deployment ``chat``                storyboard, critique, fixes
``vision``     ``vision`` label on the writer     judging generated images
``illustrator`` deployment ``image``              scene artwork
``voice``      deployment ``tts``                 narration
``listener``   deployment ``transcription``       listen-back check, captions
``upscaler``   ``upscaling`` label (optional)     sharper frames
=============  =================================  ==========================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OMNI_RECIPE = "collection.omni"

DEPLOYMENT_LABELS = (
    "chat",
    "transcription",
    "image",
    "tts",
    "embeddings",
    "embedding",
    "reranking",
    "audio-generation",
    "classification",
    "3d",
)

RECIPE_DEFAULT_MODE = {
    "llamacpp": "chat",
    "llamacpp-hrx": "chat",
    "flm": "chat",
    "ryzenai-llm": "chat",
    "vllm": "chat",
    "cloud": "chat",
    "ds4": "chat",
    "whispercpp": "transcription",
    "moonshine": "transcription",
    "sd-cpp": "image",
    "kokoro": "tts",
    "openmoss": "tts",
}

# When several Omni models are downloaded and none was requested, prefer these.
PREFERRED_OMNI = ("LMX-Omni-52B-Halo", "LMX-Omni-5.5B-Lite")
DEFAULT_OMNI = "LMX-Omni-5.5B-Lite"


@dataclass
class ModelInfo:
    id: str
    recipe: str = ""
    labels: list[str] = field(default_factory=list)
    downloaded: bool = True
    size_gb: float | None = None
    image_defaults: dict[str, Any] = field(default_factory=dict)
    context_length: int | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ModelInfo:
        return cls(
            id=str(data.get("id") or data.get("model_name") or ""),
            recipe=str(data.get("recipe") or ""),
            labels=[str(x) for x in data.get("labels") or []],
            downloaded=bool(data.get("downloaded", True)),
            size_gb=data.get("size"),
            image_defaults=dict(data.get("image_defaults") or {}),
            context_length=data.get("context_length") or data.get("max_context_window"),
        )

    @property
    def mode(self) -> str | None:
        return deployment_mode(self.labels, self.recipe)

    def has(self, label: str) -> bool:
        return label in self.labels

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "recipe": self.recipe,
            "labels": self.labels,
            "size_gb": self.size_gb,
        }


def deployment_mode(labels: list[str], recipe: str) -> str | None:
    for label in labels:
        if label in DEPLOYMENT_LABELS:
            return "embeddings" if label == "embedding" else label
    return RECIPE_DEFAULT_MODE.get(recipe)


class ToolkitError(RuntimeError):
    """The server does not have the models LemonReel needs, with suggested fixes."""

    def __init__(self, message: str, hints: list[str] | None = None):
        super().__init__(message)
        self.hints = hints or []

    def __str__(self) -> str:
        text = super().__str__()
        if self.hints:
            text += "\n" + "\n".join(f"  - {h}" for h in self.hints)
        return text


@dataclass
class Toolkit:
    """The concrete models chosen for each role."""

    writer: ModelInfo | None  # only None when resolved with require_writer=False
    voice: ModelInfo
    illustrator: ModelInfo | None = None
    listener: ModelInfo | None = None
    upscaler: ModelInfo | None = None
    omni: str | None = None

    @property
    def can_see(self) -> bool:
        return self.writer is not None and self.writer.has("vision")

    @property
    def can_listen(self) -> bool:
        return self.listener is not None

    @property
    def can_draw(self) -> bool:
        return self.illustrator is not None

    def roles(self) -> dict[str, ModelInfo | None]:
        return {
            "writer": self.writer,
            "illustrator": self.illustrator,
            "voice": self.voice,
            "listener": self.listener,
            "upscaler": self.upscaler,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "omni": self.omni,
            "roles": {k: (v.summary() if v else None) for k, v in self.roles().items()},
            "capabilities": {
                "vision_critique": self.can_see,
                "listen_back": self.can_listen,
                "illustrations": self.can_draw,
            },
        }


def _index(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for m in models:
        mid = str(m.get("id") or m.get("model_name") or "")
        if not mid:
            continue
        index[mid] = m
        for prefix in ("user.", "builtin.", "extra."):
            if mid.startswith(prefix):
                index.setdefault(mid[len(prefix) :], m)
    return index


def _lookup(index: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    for key in (name, f"user.{name}", f"builtin.{name}", f"extra.{name}"):
        if key in index:
            return index[key]
    return None


def omni_collections(models: list[dict[str, Any]]) -> list[ModelInfo]:
    return [ModelInfo.from_api(m) for m in models if m.get("recipe") == OMNI_RECIPE]


def components_of(collection: dict[str, Any], index: dict[str, dict[str, Any]]) -> list[ModelInfo]:
    """Component models of an Omni collection, preferring the embedded definitions."""
    names = [str(n) for n in collection.get("components") or []]
    embedded = {
        str(m.get("id") or m.get("model_name")): m
        for m in collection.get("models") or []
        if isinstance(m, dict)
    }
    result: list[ModelInfo] = []
    for name in names or list(embedded):
        listed = _lookup(index, name)
        data = dict(embedded.get(name) or {})
        if listed:
            # The top-level listing knows the live download state; keep it authoritative.
            data = {**data, **listed}
        if not data:
            data = {"id": name, "downloaded": False}
        data.setdefault("id", name)
        result.append(ModelInfo.from_api(data))
    return result


def _pick(candidates: list[ModelInfo], mode: str, prefer: tuple[str, ...] = ()) -> ModelInfo | None:
    pool = [m for m in candidates if m.mode == mode and not m.has("upscaling")]
    if not pool:
        return None
    for name in prefer:
        for m in pool:
            if m.id == name or m.id.endswith("." + name):
                return m
    return pool[0]


def _require(index: dict[str, dict[str, Any]], name: str, role: str) -> ModelInfo:
    data = _lookup(index, name)
    if data is None:
        raise ToolkitError(
            f"Model '{name}' (requested for the {role} role) is not known to this Lemonade server.",
            ["Run `lemonade list` to see model names."],
        )
    return ModelInfo.from_api(data)


def resolve_toolkit(
    models: list[dict[str, Any]],
    omni: str | None = None,
    *,
    writer: str | None = None,
    illustrator: str | None = None,
    voice: str | None = None,
    listener: str | None = None,
    want_upscaler: bool = False,
    require_writer: bool = True,
) -> Toolkit:
    """Choose a model for every role.

    ``models`` is the ``data`` list of ``GET /v1/models?show_all=true``.
    """
    index = _index(models)
    chosen_omni: str | None = None
    pool: list[ModelInfo]

    if omni:
        collection = _lookup(index, omni)
        if collection is None:
            raise ToolkitError(
                f"Omni model '{omni}' is not known to this Lemonade server.",
                [
                    f"Pull it with `lemonade pull {omni}`",
                    "List Omni models with `lemonreel doctor`",
                ],
            )
        if collection.get("recipe") != OMNI_RECIPE:
            raise ToolkitError(
                f"'{omni}' is a {collection.get('recipe') or 'regular'} model, not an Omni collection.",
                [
                    "Pass an Omni model such as LMX-Omni-5.5B-Lite, or pick roles with --writer/--voice."
                ],
            )
        chosen_omni = str(collection.get("id"))
        pool = components_of(collection, index)
        missing = [m.id for m in pool if not m.downloaded]
        if missing:
            raise ToolkitError(
                f"Omni model '{chosen_omni}' is missing downloaded components: {', '.join(missing)}.",
                [f"Download everything with `lemonade pull {chosen_omni}`"],
            )
    else:
        downloaded_omni = [
            m for m in models if m.get("recipe") == OMNI_RECIPE and m.get("downloaded", True)
        ]
        ranked = sorted(
            downloaded_omni,
            key=lambda m: (
                PREFERRED_OMNI.index(str(m.get("id")))
                if str(m.get("id")) in PREFERRED_OMNI
                else len(PREFERRED_OMNI),
                str(m.get("id")),
            ),
        )
        collection = None
        for candidate in ranked:
            parts = components_of(candidate, index)
            if parts and all(p.downloaded for p in parts):
                collection = candidate
                pool = parts
                break
        if collection is not None:
            chosen_omni = str(collection.get("id"))
        else:
            # No complete Omni model: assemble one from whatever is downloaded.
            pool = [
                ModelInfo.from_api(m)
                for m in models
                if m.get("downloaded", True) and m.get("recipe") != OMNI_RECIPE
            ]

    roles = {
        "writer": _pick(
            sorted(pool, key=lambda m: (not m.has("vision"), not m.has("tool-calling"))), "chat"
        ),
        "illustrator": _pick(pool, "image"),
        "voice": _pick(pool, "tts", prefer=("kokoro-v1",)),
        "listener": _pick(sorted(pool, key=lambda m: -(m.size_gb or 0)), "transcription"),
    }
    overrides = {"writer": writer, "illustrator": illustrator, "voice": voice, "listener": listener}
    for role, name in overrides.items():
        if name:
            roles[role] = _require(index, name, role)

    hints = [f"Download the default Omni model with `lemonade pull {DEFAULT_OMNI}`"]
    if roles["writer"] is None and require_writer:
        raise ToolkitError("No downloaded chat LLM was found for writing the script.", hints)
    if roles["voice"] is None:
        raise ToolkitError("No downloaded text-to-speech model was found for narration.", hints)

    upscaler = None
    if want_upscaler:
        upscaler = next(
            (
                ModelInfo.from_api(m)
                for m in models
                if m.get("downloaded", True) and "upscaling" in (m.get("labels") or [])
            ),
            None,
        )

    return Toolkit(
        writer=roles["writer"],
        illustrator=roles["illustrator"],
        voice=roles["voice"],
        listener=roles["listener"],
        upscaler=upscaler,
        omni=chosen_omni,
    )
