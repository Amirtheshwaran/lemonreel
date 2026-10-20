import pytest
from fake_lemonade import catalog

from lemonreel.omni import ToolkitError, deployment_mode, resolve_toolkit


def test_resolves_downloaded_omni_model():
    tk = resolve_toolkit(catalog("full"))
    assert tk.omni == "LMX-Omni-5.5B-Lite"
    assert tk.writer.id == "Qwen3.5-4B-MTP-GGUF"
    assert tk.illustrator.id == "SD-Turbo"
    assert tk.voice.id == "kokoro-v1"
    assert tk.listener.id == "Whisper-Tiny"
    assert tk.can_see and tk.can_listen and tk.can_draw
    assert tk.illustrator.image_defaults["steps"] == 4


def test_explicit_omni_with_missing_components_explains_fix():
    with pytest.raises(ToolkitError) as err:
        resolve_toolkit(catalog("full"), "LMX-Omni-52B-Halo")
    assert "lemonade pull LMX-Omni-52B-Halo" in str(err.value)


def test_unknown_and_non_collection_models():
    with pytest.raises(ToolkitError):
        resolve_toolkit(catalog("full"), "Nope-Omni")
    with pytest.raises(ToolkitError) as err:
        resolve_toolkit(catalog("full"), "SD-Turbo")
    assert "not an Omni collection" in str(err.value)


def test_falls_back_to_loose_models():
    tk = resolve_toolkit(catalog("minimal"))
    assert tk.omni is None
    assert tk.writer.id == "Qwen3-0.6B-GGUF"
    assert tk.voice.id == "kokoro-v1"
    assert tk.illustrator is None and tk.listener is None
    assert not tk.can_see


def test_overrides_win():
    tk = resolve_toolkit(catalog("full"), writer="Qwen3-0.6B-GGUF")
    assert tk.writer.id == "Qwen3-0.6B-GGUF"
    assert tk.omni == "LMX-Omni-5.5B-Lite"


def test_nothing_usable():
    with pytest.raises(ToolkitError) as err:
        resolve_toolkit([m for m in catalog("minimal") if m["recipe"] != "kokoro"])
    assert "text-to-speech" in str(err.value)


def test_deployment_mode_from_labels_or_recipe():
    assert deployment_mode(["vision", "chat"], "llamacpp") == "chat"
    assert deployment_mode([], "whispercpp") == "transcription"
    assert deployment_mode(["image", "edit"], "sd-cpp") == "image"
    assert deployment_mode(["embedding"], "llamacpp") == "embeddings"
    assert deployment_mode([], "mystery") is None
