import json

from lemonreel.lexicon import Lexicon, load_lexicon_file, remember


def test_builtin_respellings_only_touch_spoken_text():
    lex = Lexicon.build("en")
    spoken = lex.apply("Run it on ROCm or Vulkan with GGUF files.")
    assert "Rock M" in spoken
    assert "G G U F" in spoken


def test_units_and_urls():
    lex = Lexicon.build("en")
    assert lex.apply("It has 128GB and runs a 4B model, see https://example.com") == (
        "It has 128 gigabytes and runs a 4 billion model, see"
    )
    # Single-letter units must touch the number, and unrelated words stay put.
    assert lex.apply("Plan B costs 4 K") == "Plan B costs 4 K"
    assert lex.apply("Watch it in 4K") == "Watch it in 4K"


def test_case_sensitive_acronyms_and_longest_match():
    lex = Lexicon({"LLM": "L L M", "LLMs": "L L Ms"})
    assert lex.apply("LLMs beat one LLM") == "L L Ms beat one L L M"
    assert lex.apply("an llm") == "an llm"


def test_dotted_terms():
    lex = Lexicon.build("en")
    assert lex.apply("llama.cpp is fast") == "llama C P P is fast"


def test_non_english_skips_builtins():
    lex = Lexicon.build("es")
    assert lex.apply("ROCm con 8GB") == "ROCm con 8GB"


def test_learned_lexicon_round_trip(tmp_path):
    path = tmp_path / "lex" / "lexicon.json"
    remember(path, {"Strix": "Stricks"})
    remember(path, {"Qwen": "Chwen"})
    assert load_lexicon_file(path) == {"Strix": "Stricks", "Qwen": "Chwen"}
    lex = Lexicon.build("en", learned_path=path)
    assert lex.apply("Strix runs Qwen") == "Stricks runs Chwen"


def test_bad_lexicon_file_is_ignored(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    assert load_lexicon_file(path) == {}
    path.write_text(json.dumps(["a", "b"]))
    assert load_lexicon_file(path) == {}
