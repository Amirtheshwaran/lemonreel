"""Narration with a listen-back loop.

Every clip is spoken by the Omni model's voice, then transcribed by its listener and
compared with the script (word error rate). If the listener misheard names or jargon,
the writer proposes phonetic respellings ("ROCm" -> "Rock M"), the clip is spoken again,
and the better take is kept. Respellings that helped are remembered for future videos.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lemonreel import lexicon as lexicon_mod
from lemonreel.captions import TimedWord, timed_words
from lemonreel.config import ReelConfig
from lemonreel.lemonade import LemonadeClient, LemonadeError
from lemonreel.lexicon import Lexicon
from lemonreel.llm_json import JSONExtractionError, extract_json_object
from lemonreel.media import concat_wavs, media_duration, to_asr_wav, write_speech_clip
from lemonreel.omni import Toolkit
from lemonreel.storyboard import Scene
from lemonreel.textmetrics import Mismatch, find_mismatches, normalize_tokens, word_error_rate

Log = Callable[[str], None]

LINE_GAP = 0.3  # seconds of silence between duo lines

RESPELL_PROMPT = """A text-to-speech voice read a sentence aloud and a speech recognizer listened back.

Written: "{written}"
Heard:   "{heard}"

The recognizer did not understand these words: {problems}.
For each one that is a name, acronym, brand or technical term a speech engine could mispronounce,
give a phonetic respelling built from ordinary {language} syllables, for example "ROCm" -> "Rock M"
or "Qwen" -> "Chwen". Skip ordinary words; those are recognizer mistakes, not pronunciation problems.

Reply with JSON only: {{"respellings": {{"<word exactly as written>": "<respelling>"}}}}"""


@dataclass
class Take:
    number: int
    spoken: str
    heard: str | None
    wer: float | None
    seconds: float
    respellings: dict[str, str] = field(default_factory=dict)
    helped: bool = False


@dataclass
class ClipResult:
    """One synthesized utterance (a narrator scene, or one duo line)."""

    text: str
    speaker: str
    voice: str
    path: str
    duration: float
    takes: list[Take] = field(default_factory=list)
    words: list[TimedWord] = field(default_factory=list)
    kept: int = 0  # index of the take that ended up in the video

    @property
    def final(self) -> Take:
        return self.takes[self.kept]


@dataclass
class SceneVoice:
    path: str
    duration: float
    clips: list[ClipResult]
    starts: list[float]
    summary: dict | None = None  # stored results when the narration came from the cache

    @property
    def wer(self) -> float | None:
        if not self.clips and self.summary:
            return self.summary.get("wer")
        values = [c.final.wer for c in self.clips if c.final.wer is not None]
        if not values:
            return None
        weights = [len(c.text.split()) for c in self.clips if c.final.wer is not None]
        return sum(v * w for v, w in zip(values, weights)) / max(1, sum(weights))

    @property
    def first_wer(self) -> float | None:
        if not self.clips and self.summary:
            return self.summary.get("first_wer")
        values = [c.takes[0].wer for c in self.clips if c.takes and c.takes[0].wer is not None]
        if not values:
            return None
        weights = [
            len(c.text.split()) for c in self.clips if c.takes and c.takes[0].wer is not None
        ]
        return sum(v * w for v, w in zip(values, weights)) / max(1, sum(weights))

    @property
    def fixes(self) -> dict[str, str]:
        if not self.clips and self.summary:
            return dict(self.summary.get("fixes") or {})
        out: dict[str, str] = {}
        for clip in self.clips:
            for take in clip.takes:
                if take.helped:
                    out.update(take.respellings)
        return out

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "duration": round(self.duration, 3),
            "starts": [round(s, 3) for s in self.starts],
            "wer": None if self.wer is None else round(self.wer, 4),
            "first_wer": None if self.first_wer is None else round(self.first_wer, 4),
            "fixes": self.fixes,
            "clips": [
                {**{k: v for k, v in asdict(c).items() if k != "words"}, "word_count": len(c.words)}
                for c in self.clips
            ],
        }


# Common words that start sentences; a capitalized sentence opener outside this list is
# probably a name ("Strix Halo is...", "Lemonade runs...").
_COMMON_OPENERS = set(
    """a about after again all also although an and another any are as at back be because
    before being both but by can could did do does each even every few finally first for
    from get give go good great had has have he her here his how however i if imagine in
    instead into is it its just keep last let lets like look make many maybe meanwhile more
    most much my never new next no not now of on once one only or other our over people
    picture plus put right say says see she since so some still such suddenly take than
    thanks that the their them then there these they think this those though thousands
    three through to today together too two under unlike until up us use using very was
    we well were what whatever when where whether which while who why will with without
    would yes yet you your""".split()
)


def special_terms(text: str) -> set[str]:
    """Words likely to be names, acronyms or jargon: the ones worth respelling."""
    terms: set[str] = set()
    words = text.split()
    for i, raw in enumerate(words):
        word = raw.strip(".,;:!?\"'()[]")
        if not word or not any(c.isalpha() for c in word):
            continue
        uppers = sum(c.isupper() for c in word)
        sentence_start = i == 0 or words[i - 1].endswith((".", "!", "?", ":"))
        if (
            any(c.isdigit() for c in word)
            or uppers >= 2
            or (uppers == 1 and not word[0].isupper())
            or (word[0].isupper() and not sentence_start)
            or (word[0].isupper() and word.lower().replace("'", "") not in _COMMON_OPENERS)
        ):
            terms.add(word)
    return terms


class Narrator:
    """Speaks clips and, when a listener is available, checks them."""

    def __init__(
        self,
        client: LemonadeClient,
        toolkit: Toolkit,
        cfg: ReelConfig,
        lexicon: Lexicon,
        log: Log = print,
        learned_path: Path | None = None,
    ):
        self.client = client
        self.toolkit = toolkit
        self.cfg = cfg
        self.lexicon = lexicon
        self.log = log
        self.learned_path = learned_path

    # -- single clip -------------------------------------------------------------------------

    def _synthesize(self, text: str, voice: str, dst: Path, note: str) -> float:
        audio = self.client.speech(
            self.toolkit.voice.id, text, voice=voice, speed=self.cfg.speed, note=note
        )
        write_speech_clip(audio, dst)
        return media_duration(dst)

    def _listen(self, clip: Path, note: str) -> dict:
        assert self.toolkit.listener is not None
        wav16 = to_asr_wav(clip, clip.with_name(clip.stem + ".16k.wav"))
        return self.client.transcribe(
            self.toolkit.listener.id,
            wav16,
            language=self.cfg.base_language,
            response_format="verbose_json",
            note=note,
        )

    def _needs_fix(self, written: str, heard: str, wer: float) -> bool:
        """Retry when the error rate is high, or when any misheard word is a name or term.

        A single mangled product name barely moves the error rate of a 40-word scene, but it
        is exactly the mistake a viewer notices.
        """
        if wer > self.cfg.wer_threshold:
            return True
        if self.cfg.base_language != "en":
            # The name/jargon heuristics rely on English capitalization and openers.
            return False
        terms = special_terms(written)
        return any(
            any(w.strip(".,;:!?\"'()") in terms for w in m.expected.split())
            for m in self._problems(written, heard)
        )

    def _score(self, written: str, spoken: str, heard: str) -> float:
        """Error rate of what was heard, forgiving words heard exactly as respelled."""
        lang = self.cfg.base_language
        wer = word_error_rate(written, heard, lang)
        if spoken != written:
            wer = min(wer, word_error_rate(spoken, heard, lang))
        return wer

    def _problems(self, written: str, heard: str) -> list[Mismatch]:
        lang = self.cfg.base_language
        problems = []
        for m in find_mismatches(written, heard, lang):
            # "ROCm" heard as "rock m" means the respelling worked; that is not a problem.
            if normalize_tokens(self.lexicon.apply(m.expected), lang) == m.heard.split():
                continue
            problems.append(m)
        return problems

    def _respell(self, written: str, heard: str) -> dict[str, str]:
        problems = self._problems(written, heard)
        if not problems:
            return {}
        listed = ", ".join(f'"{m.expected}" (heard "{m.heard or "nothing"}")' for m in problems[:8])
        try:
            result = self.client.chat(
                self.toolkit.writer.id,
                [
                    {
                        "role": "user",
                        "content": RESPELL_PROMPT.format(
                            written=written,
                            heard=heard,
                            problems=listed,
                            language=self.cfg.language_name,
                        ),
                    }
                ],
                temperature=0.2,
                max_tokens=300,
                json_mode=True,
                note="respell",
            )
            data = extract_json_object(result.text)
        except (LemonadeError, JSONExtractionError) as exc:
            self.log(f"Could not get respellings ({exc})")
            return {}
        raw = data.get("respellings", data)
        if not isinstance(raw, dict):
            return {}
        lower_text = written.lower()
        clean: dict[str, str] = {}
        for word, respelled in raw.items():
            word, respelled = str(word).strip(), str(respelled).strip()
            if (
                word
                and respelled
                and word.lower() in lower_text
                and respelled.lower() != word.lower()
                and len(respelled) <= 48
            ):
                clean[word] = respelled
        return clean

    def speak(self, text: str, speaker: str, voice: str, dst: Path, label: str) -> ClipResult:
        """Speak ``text`` into ``dst``; listen back and retry with respellings if needed."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        spoken = self.lexicon.apply(text)
        started = self.client.tracer.now()
        duration = self._synthesize(spoken, voice, dst, note=f"{label} take 1")
        take = Take(1, spoken, None, None, round(self.client.tracer.now() - started, 2))
        result = ClipResult(
            text=text, speaker=speaker, voice=voice, path=str(dst), duration=duration, takes=[take]
        )
        if not (self.cfg.listen_back and self.toolkit.can_listen):
            return result

        try:
            asr = self._listen(dst, note=f"{label} listen 1")
        except LemonadeError as exc:
            self.log(f"{label}: listen-back skipped ({exc})")
            return result
        take.heard = str(asr.get("text", "")).strip()
        take.wer = self._score(text, spoken, take.heard)
        result.words = timed_words(asr)
        best_wer, best_audio = take.wer, dst.read_bytes()

        while (
            take.wer > 0
            and len(result.takes) < self.cfg.max_voice_attempts
            and self._needs_fix(text, take.heard or "", take.wer)
        ):
            number = len(result.takes) + 1
            fixes = self._respell(text, take.heard or "")
            if not fixes:
                break
            candidate = Lexicon(dict(self.lexicon.entries), self.lexicon.language)
            candidate.update(fixes)
            new_spoken = candidate.apply(text)
            if new_spoken == take.spoken:
                break
            started = self.client.tracer.now()
            retry = dst.with_name(f"{dst.stem}.take{number}.wav")
            try:
                new_duration = self._synthesize(
                    new_spoken, voice, retry, note=f"{label} take {number}"
                )
                asr = self._listen(retry, note=f"{label} listen {number}")
            except LemonadeError as exc:
                self.log(f"{label}: retry failed ({exc})")
                break
            heard = str(asr.get("text", "")).strip()
            new_take = Take(
                number,
                new_spoken,
                heard,
                self._score(text, new_spoken, heard),
                round(self.client.tracer.now() - started, 2),
                respellings=fixes,
            )
            result.takes.append(new_take)
            if new_take.wer < best_wer:
                new_take.helped = True
                best_wer, best_audio = new_take.wer, retry.read_bytes()
                result.kept = len(result.takes) - 1
                result.words = timed_words(asr)
                result.duration = new_duration
                self.lexicon.update(fixes)
                if self.cfg.learn and self.learned_path is not None:
                    lexicon_mod.remember(self.learned_path, fixes)
                self.log(
                    f"{label}: fixed pronunciation of {', '.join(fixes)} "
                    f"(listen-back error {take.wer:.0%} -> {new_take.wer:.0%})"
                )
            take = new_take
            retry.unlink(missing_ok=True)

        dst.write_bytes(best_audio)
        return result

    # -- scenes ----------------------------------------------------------------------------

    def narrate_scene(self, scene: Scene, index: int, scene_dir: Path) -> SceneVoice:
        narrator_voice, second_voice = self.cfg.voices
        scene_dir.mkdir(parents=True, exist_ok=True)
        label = f"Scene {index + 1}"
        if scene.lines:
            clips = []
            for n, line in enumerate(scene.lines, 1):
                voice = second_voice if line.speaker == "B" else narrator_voice
                clips.append(
                    self.speak(
                        line.text,
                        line.speaker,
                        voice,
                        scene_dir / f"line-{n:02d}.wav",
                        f"{label} line {n}",
                    )
                )
            layout = concat_wavs(
                [Path(c.path) for c in clips], scene_dir / "voice.wav", gap=LINE_GAP
            )
            path, duration, starts = scene_dir / "voice.wav", layout.total, layout.starts
        else:
            clip = self.speak(scene.narration, "A", narrator_voice, scene_dir / "voice.wav", label)
            clips, path, duration, starts = [clip], Path(clip.path), clip.duration, [0.0]
        voice = SceneVoice(path=str(path), duration=duration, clips=clips, starts=starts)
        (scene_dir / "voice.json").write_text(
            json.dumps(voice.to_dict(), indent=2), encoding="utf-8"
        )
        return voice

    def narrate_title(self, title: str, out_dir: Path) -> ClipResult:
        # Titles are short and often all jargon; listening back adds little.
        spoken = self.lexicon.apply(title)
        dst = out_dir / "title.wav"
        out_dir.mkdir(parents=True, exist_ok=True)
        started = self.client.tracer.now()
        duration = self._synthesize(spoken, self.cfg.voices[0], dst, note="title")
        take = Take(1, spoken, None, None, round(self.client.tracer.now() - started, 2))
        return ClipResult(
            text=title,
            speaker="A",
            voice=self.cfg.voices[0],
            path=str(dst),
            duration=duration,
            takes=[take],
        )
