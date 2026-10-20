# Troubleshooting

Start with `lemonreel doctor --smoke`. It checks the connection, the Omni models, the role assignment
and ffmpeg, and times one request per model.

**"Can't reach Lemonade Server"**
Start Lemonade (the Lemonade app on Windows/macOS, `lemond` or the system service on Linux) and check
it with `lemonade status`. LemonReel tries `http://localhost:13305` and then the legacy
`http://localhost:8000`. For anything else, pass `--base-url http://host:port` or set
`LEMONADE_BASE_URL`. If you set `LEMONADE_API_KEY` for the server, set it for LemonReel too.

**"No downloaded ... model was found" or "missing downloaded components"**
Run the `lemonade pull ...` command printed in the message. `lemonade pull LMX-Omni-5.5B-Lite` gives you
every role at once.

**Pictures take minutes each**
Image generation on CPU is slow. Check `lemonade backends` for a GPU backend for `sd-cpp` (Vulkan or
ROCm on Radeon) and install it. Meanwhile, `--scenes 4`, `--no-critique` (no redraws) or `--steps 2`
make runs shorter.

**A request timed out**
Raise the per-request limit, for example `--timeout 1800`. The first request to each model includes
loading it.

**The machine runs out of memory**
Use `--sequential` so only one lane works at a time, or a smaller Omni model.

**A word is still mispronounced**
Add it to a JSON file, `{"Kubernetes": "koo ber net ease"}`, and pass `--lexicon that-file.json`. Or
add it straight to `~/.lemonreel/lexicon.json`, which is used on every run. A larger listener model
(for example `--listener Whisper-Base`) also catches more mistakes.

**Captions drift from the voice**
Caption timing comes from the listener. Without one, timing is estimated from text length. Pull a
Whisper model, or use a larger one with `--listener`.

**Japanese, Chinese or Hindi captions show boxes**
The system has no font for that script. On Linux install Noto CJK / Noto Devanagari fonts, or point
`LEMONREEL_FONT` at any `.ttf`/`.ttc` file that covers it.

**ffmpeg errors**
LemonReel uses the ffmpeg bundled with `imageio-ffmpeg`. Reinstalling LemonReel fixes a broken copy;
`LEMONREEL_FFMPEG=/path/to/ffmpeg` uses another one (it needs libx264).

**Something else**
Every reel folder has `events.jsonl` with each step that ran. Please attach it, `report.json` if
present, and the `lemonreel doctor` output to an issue.
