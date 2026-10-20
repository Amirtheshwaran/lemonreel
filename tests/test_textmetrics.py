from lemonreel.textmetrics import (
    find_mismatches,
    normalize_tokens,
    number_to_words,
    word_error_rate,
)


def test_number_to_words():
    assert number_to_words(0) == "zero"
    assert number_to_words(17) == "seventeen"
    assert number_to_words(128) == "one hundred twenty eight"
    assert number_to_words(4096) == "four thousand ninety six"
    assert number_to_words(2_000_000) == "two million"


def test_digits_and_words_compare_equal():
    assert (
        word_error_rate(
            "It has 128 GB of memory.", "it has one hundred twenty eight gigabytes of memory"
        )
        == 0
    )
    assert (
        word_error_rate(
            "Version 3.5 shipped in 2026.", "version three point five shipped in twenty twenty six"
        )
        == 0
    )


def test_spelled_out_acronyms_collapse():
    assert normalize_tokens("The N P U is fast") == normalize_tokens("The NPU is fast")
    assert normalize_tokens("N.P.U.") == ["npu"]


def test_symbols_are_spoken():
    assert (
        word_error_rate(
            "Speed went up 40% & latency fell", "speed went up forty percent and latency fell"
        )
        == 0
    )


def test_wer_counts_substitutions():
    assert word_error_rate("Strix Halo is fast", "tricks halo is fast") == 0.25
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "noise") == 1.0


def test_mismatches_keep_original_spelling():
    found = find_mismatches("Meet Strix Halo, the chip.", "meet tricks halo the chip")
    assert [m.expected for m in found] == ["Strix"]
    assert found[0].heard == "tricks"


def test_dropped_words_are_reported():
    found = find_mismatches("Lemonade runs Qwen locally", "lemonade runs locally")
    assert found[0].expected == "Qwen"
    assert found[0].heard == ""


def test_character_level_languages():
    assert word_error_rate("東京は大きい", "東京は大きい", "ja") == 0
    assert 0 < word_error_rate("東京は大きい", "東京は小さい", "ja") < 1
