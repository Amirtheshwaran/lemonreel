from lemonreel.captions import Cue, build_cues, chunk_text, timed_words, to_srt, to_vtt

TEXT = (
    "Strix Halo puts a big GPU next to the CPU. It shares up to 128 GB of memory, "
    "so large models fit without a separate graphics card."
)


def _asr_for(text: str, start: float = 0.5, step: float = 0.3, words: bool = True) -> dict:
    tokens = text.replace("Strix", "tricks").split()
    timed = [
        {"word": " " + w, "start": start + i * step, "end": start + i * step + 0.25}
        for i, w in enumerate(tokens)
    ]
    seg = {"start": timed[0]["start"], "end": timed[-1]["end"], "text": " ".join(tokens)}
    if words:
        seg["words"] = timed
    return {"text": " ".join(tokens), "segments": [seg]}


def test_timed_words_from_words_and_segments():
    with_words = timed_words(_asr_for("one two three"))
    assert [w for w, _, _ in with_words] == ["one", "two", "three"]
    assert with_words[1][1] == 0.8
    from_segments = timed_words(_asr_for("one two three", words=False))
    assert len(from_segments) == 3
    assert from_segments[0][1] == 0.5
    assert abs(from_segments[-1][2] - 1.35) < 1e-6


def test_top_level_words_openai_style():
    asr = {
        "text": "hi there",
        "words": [
            {"word": "hi", "start": 0.1, "end": 0.3},
            {"word": "there", "start": 0.4, "end": 0.7},
        ],
    }
    assert timed_words(asr) == [("hi", 0.1, 0.3), ("there", 0.4, 0.7)]


def test_chunks_respect_length():
    words = TEXT.split()
    for chunk in chunk_text(TEXT, 42):
        assert len(" ".join(words[i] for i in chunk)) <= 42 or len(chunk) == 1
    covered = [i for chunk in chunk_text(TEXT, 42) for i in chunk]
    assert covered == list(range(len(words)))


def test_cues_follow_recognizer_timing_and_keep_script_words():
    asr = _asr_for(TEXT)
    duration = 0.5 + len(TEXT.split()) * 0.3 + 0.2
    cues = build_cues(TEXT, duration, timed_words(asr), max_chars=60)
    assert cues[0].text.startswith("Strix Halo")  # script spelling, not "tricks"
    # "Strix" was misheard, so its start is extrapolated from the speaking rate, not 0.0.
    assert abs(cues[0].start - 0.5) < 0.15
    for a, b in zip(cues, cues[1:]):
        assert a.start < a.end <= b.start
    assert cues[-1].end <= duration
    assert " ".join(c.text for c in cues) == TEXT


def test_cues_without_recognizer_are_proportional():
    cues = build_cues(TEXT, 10.0, None, max_chars=60, offset=2.0)
    assert cues[0].start == 2.0
    assert cues[-1].end <= 12.0


def test_subtitle_formats():
    cues = [Cue(0.0, 1.5, "Hello"), Cue(61.25, 3725.004, "World")]
    srt = to_srt(cues)
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello" in srt
    assert "00:01:01,250 --> 01:02:05,004" in srt
    vtt = to_vtt(cues)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt
