# How LemonReel works

This page walks through a run from source to MP4 and explains the design decisions, especially the
two self-checking loops. File references point at `src/lemonreel/`.

- [1. Finding the models](#1-finding-the-models)
- [2. Reading the source](#2-reading-the-source)
- [3. Writing the storyboard](#3-writing-the-storyboard)
- [4. Two lanes in parallel](#4-two-lanes-in-parallel)
- [5. The vision critique loop](#5-the-vision-critique-loop)
- [6. The listen-back loop](#6-the-listen-back-loop)
- [7. Captions from word timestamps](#7-captions-from-word-timestamps)
- [8. Editing the video](#8-editing-the-video)
- [9. Caching and remakes](#9-caching-and-remakes)
- [10. The report](#10-the-report)
- [Testing without models](#testing-without-models)

## 1. Finding the models

*`omni.py`, `lemonade.py`*

LemonReel never hard-codes model names. It asks Lemonade for `GET /v1/models?show_all=true` and looks
for Omni collections: models with `recipe: "collection.omni"`. A collection lists its `components`
and, in current Lemonade versions, embeds each component's full model object under `models`. The
top-level listing is used for the live `downloaded` flag, and the embedded objects fill in anything
the listing lacks.

Each component's *deployment mode* comes from its labels (`chat`, `image`, `tts`, `transcription`, ...)
or, when a model has none, from its recipe's default, exactly as Lemonade itself resolves it
(`llamacpp` -> chat, `whispercpp` -> transcription, `sd-cpp` -> image, `kokoro` -> tts). That gives the
four roles:

| Role | Mode | Extra capability |
|------|------|------------------|
| writer | `chat` | `vision` label enables picture critique |
| illustrator | `image` (not `upscaling`) | `image_defaults` sets the native size, steps and CFG |
| voice | `tts` | Kokoro is preferred for its fixed, multilingual voices |
| listener | `transcription` | enables listen-back and timed captions |

With no `--omni`, a fully downloaded collection is chosen (Halo first, then Lite, then others by
name). If none is complete, LemonReel assembles a toolkit from loose downloaded models. Only a writer
and a voice are required. When something is missing, the error names the exact `lemonade pull`
command that fixes it.

The client (`lemonade.py`) discovers the server on port 13305 and then the legacy port 8000, and
accepts any of `host:port`, `/v1` or `/api/v1` URLs. Every inference request is recorded with its
lane, model, start and end time. That trace is what the report's timeline is drawn from.

## 2. Reading the source

*`ingest.py`*

Markdown is reduced to its words while keeping paragraph structure: front matter, images, link
targets, HTML tags and table rules go, headings become sentences, and code blocks become
"[code example omitted]" (narrating code is never useful). HTML is parsed with the standard library,
skipping `script`, `style`, `nav`, `header`, `footer` and `aside`. A GitHub repository URL is turned
into its raw README. PDFs use the optional `pypdf`.

Audio is the interesting case: a voice memo is converted to 16 kHz mono WAV and transcribed by the
Omni model's own listener, so the same Omni model handles input as well as output.

Sources longer than `max_source_chars` (12,000 by default) are split on paragraph boundaries and each
part is condensed by the writer into bullet points first, which keeps small models within their
context window.

## 3. Writing the storyboard

*`storyboard.py`, `llm_json.py`*

The writer gets a system prompt with firm rules: stay faithful to the source, write for the ear, say
symbols as words, and write image prompts that describe one concrete picture with no text in it,
because diffusion models draw letters badly. The user prompt fixes the scene count, the words per
scene (from `--minutes` at 150 words per minute), the audience and the output language. Duo mode asks
for a host A who explains and a host B who asks the questions a newcomer would.

The request uses `response_format: {"type": "json_object"}` and
`chat_template_kwargs: {"enable_thinking": false}`, so reasoning models answer directly instead of
spending the token budget thinking. If the backend rejects `response_format`, the request is repeated
without it.

Small models still wrap JSON in prose or code fences, emit a `<think>` block first, or leave trailing
commas. `extract_json` handles all of that: it strips reasoning, tries fenced blocks, then the first
balanced `{...}` span (string-aware, so braces inside strings don't confuse it), and finally applies
repairs (smart quotes, trailing commas, `//` comments, Python literals). If the result still isn't a
usable storyboard (no scenes, no spoken text), the parse error goes back to the model as a follow-up
message and it gets two more tries.

`normalize` then cleans every field: markdown characters and URLs are removed from spoken text,
captions are clipped to six words, missing image prompts are filled from the caption, and the visual
style is set from `--look` rather than left to the model, so all scenes share one look.

## 4. Two lanes in parallel

*`pipeline.py`*

Pictures and narration don't depend on each other, so they run in two threads:

- the **draw lane** generates and critiques every picture,
- the **speak lane** narrates the title and every scene, with listen-back.

Lemonade loads at most one model *per type* by default (`max_models` in `/v1/health`: one LLM, one
image model, one TTS model, one transcription model), so an image model, Kokoro and Whisper can all be
resident at once. With both lanes busy, Lemonade can run the image model on the GPU while speech runs
on the CPU, and on Windows Ryzen AI machines Whisper can run on the NPU. The writer is shared: picture
critiques and respelling requests from both lanes queue on the one LLM.

After the lanes finish, LemonReel snapshots `/v1/health` to record which device and backend each
loaded model is on. Those values come from the server, not from guesses.

`--sequential` runs the lanes one after the other, which is useful on machines that are short of
memory.

## 5. The vision critique loop

*`visuals.py`*

**Size.** The generation size is derived from the image model's `image_defaults`: the native area is
increased by 30%, capped by the model's `soft_pixel_cap` and at one megapixel, shaped to the video's
aspect ratio, and rounded to multiples of 64. For SD-Turbo (512x512 native) that is 768x448 for 16:9
and 448x768 for 9:16. For Flux-2-Klein the larger cap gives proportionally larger images.

**Critique.** When the writer has the `vision` label, every picture is downscaled to 448 px, sent back
as an `image_url` content part with the intended scene, and graded from 1 to 5 with a short list of
problems and an optional better prompt. A score below `--min-score` (3) triggers another attempt with
a new seed (`base + attempt * 7919`, so runs are reproducible with `--seed`) and the critic's prompt.
After `--max-redraws` extra attempts the highest-scoring picture is used. Every attempt is kept in
`scenes/NN/` and listed in the report.

If image generation fails outright, the scene falls back to a typographic slide instead of failing
the whole video.

## 6. The listen-back loop

*`voice.py`, `textmetrics.py`, `lexicon.py`*

Text-to-speech engines are good at ordinary words and unreliable at exactly the words explainer videos
are full of: product names, acronyms, model names. LemonReel finds those mistakes by listening.

**Speaking.** Before a line is spoken, a pronunciation lexicon rewrites the *spoken copy only*:
`ROCm` -> "Rock M", `GGUF` -> "G G U F", `128GB` -> "128 gigabytes", URLs are dropped. The lexicon is
built from a small built-in list, the user's learned lexicon (`~/.lemonreel/lexicon.json`) and an
optional `--lexicon` file. Captions keep the original spelling.

**Listening.** The clip is resampled to 16 kHz mono and transcribed with `response_format:
verbose_json`. The transcript is compared with the script using a normalized word error rate:

- lower case, punctuation removed, `&` / `%` / `+` spoken as words,
- numbers spelled out (`128` -> "one hundred twenty eight", `3.5` -> "three point five",
  `2026` -> "twenty twenty six"), so "128 GB" and "one hundred twenty eight gigabytes" match,
- unit spellings unified (`gb` -> "gigabytes"),
- runs of single letters joined (`N P U` and `N.P.U.` both become `npu`),
- Japanese and Chinese compared per character.

The score is the lower of the error rates against the written script and against the respelled spoken
copy, because hearing "rock m" for `ROCm` means the respelling worked.

**Deciding to fix.** A pure error-rate threshold misses the important case: one mangled product name in
a 40-word scene is only 2.5% error, but it's the mistake a viewer notices. So a retry is triggered when
the error rate is above `wer_threshold` (12%) **or** any misheard word looks like a term:

- it contains a digit (`Qwen3.5`),
- it has two or more capitals (`AMD`, `iGPU`, `LLMs`) or a capital after the first letter,
- it is capitalized in the middle of a sentence (a proper noun), or
- it is capitalized at the start of a sentence but isn't a common sentence opener ("Strix Halo is...").

Misheard ordinary words are usually recognizer errors (especially with Whisper-Tiny), and retrying them
would waste time, so they don't trigger anything on their own.

**Fixing.** The writer gets the script, what was heard and the list of mismatched words (aligned with
`difflib` on normalized tokens, then mapped back to the original spellings). It returns phonetic
respellings as JSON, and is told to skip ordinary words. Respellings are kept only if the word really
appears in the line, differs from the respelling and is short. The line is spoken again with the
updated lexicon and transcribed again. The take with the lowest error rate goes into the video.

**Learning.** When a respelling lowers the error rate, it is added to the run's lexicon (so later scenes
benefit immediately) and to `~/.lemonreel/lexicon.json` (so later videos do). `--no-learn` turns this
off. The report lists every fix.

## 7. Captions from word timestamps

*`captions.py`*

Captions show the script's words, but their timing comes from the recognizer:

1. `timed_words` reads word timestamps from `verbose_json`. If the backend only returns segments,
   each segment's time is spread over its words by length.
2. The script and the transcript are tokenized with the same normalization as above. Each token
   remembers which original word it came from.
3. `difflib.SequenceMatcher` aligns the two token lists. Matched script words take the recognizer's
   start and end times.
4. Words between two matched words are interpolated by character length. Words before the first match
   or after the last are placed using the speaking rate measured on the matched stretch, so a misheard
   first word doesn't stretch the first caption back to 0:00.
5. The script is chunked into captions of at most 84 characters (56 for vertical video), preferring
   to break after sentence ends and commas, and each caption spans its first word's start to its last
   word's end.
6. A tidy-up pass removes overlaps, enforces a minimum on-screen time and closes gaps shorter than
   0.6 s to avoid flicker.

Duo lines are captioned separately, so a caption never mixes two speakers.

## 8. Editing the video

*`render.py`, `graphics.py`*

Each scene becomes one MP4 segment:

- The picture is fitted to the frame at **2x** resolution: center-cropped when the aspect ratios are
  close, otherwise shown whole over a blurred, darkened copy of itself. Rendering at 2x keeps
  `zoompan`'s sub-pixel motion smooth.
- `zoompan` applies one of five moves (zoom in, pan right, zoom out, pan left, rise) with smoothstep
  easing, cycling through scenes.
- Text is drawn with Pillow, not ffmpeg's `drawtext`, so the bundled ffmpeg needs no font libraries and
  every OS renders the same result. Overlays are cropped to their bounding box before compositing,
  and captions are fed at 5 fps (the overlay filter repeats the latest frame), which made rendering
  about 2.6x faster than full-frame overlays at 30 fps.
- The headline badge fades in and out with alpha fades; captions are switched on and off with
  `enable='between(t,start,end)'`.
- Narration is delayed by a short lead-in, padded and faded; the picture fades from and to black.

All segments share one encoding profile (H.264 `yuv420p`, AAC 48 kHz mono), so the concat demuxer
joins them with `-c copy`. Chapters are written as an `FFMETADATA` file and mapped in. `--captions soft`
adds a `mov_text` subtitle track. `chapters.txt` uses the `00:00 Title` format YouTube recognizes.

ffmpeg comes from `imageio-ffmpeg`, so no separate install is needed; `LEMONREEL_FFMPEG` overrides it.

## 9. Caching and remakes

Each scene folder stores a hash of everything that affects its picture (prompt, style, model, aspect,
seed, steps, critique and upscale settings) and its narration (text, voices, speed, models,
listen-back, language). `lemonreel remake <folder>` re-reads `storyboard.json` and regenerates only
the scenes whose hash changed, then re-edits the video. The usual loop is `--storyboard-only`, edit the
JSON, then `remake`.

## 10. The report

`report.json` contains the configuration, the server's hardware description, the role -> model ->
device/backend table, per-lane request counts and busy time, the start and end of every stage, every
picture attempt with its score, every narration take with what was heard, the pronunciation fixes,
and the full request trace. `report.html` presents the same data with the video, a timeline SVG of
every request by engine, and a card per scene.

## Testing without models

`tests/fake_lemonade.py` implements the endpoints above with the documented request and response
shapes. Details that make it useful:

- It lists `LMX-Omni-5.5B-Lite` with embedded components and a not-yet-downloaded Halo collection, so
  role resolution and the "missing components" error are tested against realistic data.
- Speech is generated as one tone burst per word, which makes durations realistic. The first 0.25 s is
  an ID tone whose frequency survives resampling. The fake recognizer measures it by counting zero
  crossings and knows which utterance it is hearing, then returns `verbose_json` with word timestamps
  that match where the bursts really are.
- It mishears "Strix" as "tricks" unless the speech input was respelled, and its critic gives the second
  picture a 2/5, so end-to-end tests exercise both repair loops.
- A `minimal` profile (chat model and TTS only) and a `segments_only` mode test the fallbacks.

The end-to-end tests render real MP4 files and check them with ffmpeg: streams, duration, chapters,
subtitle track, the 16 kHz audio sent to the listener, the learned lexicon and cache reuse on remake.
