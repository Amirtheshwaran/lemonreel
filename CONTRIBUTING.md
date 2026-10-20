# Contributing to LemonReel

Thanks for helping. Bug reports, new looks, pronunciation entries, translations and code are all
welcome.

## Getting set up

```bash
git clone https://github.com/Amirtheshwaran/lemonreel && cd lemonreel
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check src tests
pytest
```

You don't need models or a GPU to work on LemonReel. `tests/fake_lemonade.py` stands in for Lemonade
Server; see [docs/how-it-works.md](docs/how-it-works.md#testing-without-models). To try the Studio
against it:

```bash
python tests/fake_lemonade.py --port 13399
lemonreel studio --base-url http://127.0.0.1:13399
```

For a real run, install [Lemonade](https://lemonade-server.ai), `lemonade pull LMX-Omni-5.5B-Lite`,
and `lemonreel doctor --smoke`.

## Making a change

1. Open an issue first for anything bigger than a small fix, so we can agree on the approach.
2. Keep pull requests focused. Add or update tests for behaviour changes; the end-to-end tests in
   `tests/test_pipeline_e2e.py` are the place for anything that touches the pipeline.
3. Run `ruff check src tests`, `ruff format src tests` and `pytest` before pushing.
4. Describe what you changed and how you checked it. For changes to prompts, rendering or the
   self-check loops, a short before/after clip or report screenshot from a real Lemonade run helps a
   lot.

## Good first contributions

- **Pronunciations.** Add terms that text-to-speech engines get wrong to `BUILTIN_LEXICON` in
  `src/lemonreel/lexicon.py`. Please include where you heard the mistake (voice and model).
- **Looks.** New entries in `LOOKS` in `src/lemonreel/config.py`, with a sample frame in the PR.
- **Languages.** Default voices and font candidates for more languages.
- **Hardware notes.** Run `lemonreel doctor --smoke` and a `make` on your machine and share the
  `report.json` timings in a discussion, so the README's hardware table can cite real numbers.

## Reporting bugs

Please include the output of `lemonreel doctor`, your OS, and, if a video was produced, the
`report.json` from its folder. If a run failed, `events.jsonl` in the reel folder shows the last steps.

## Code of conduct

Be kind and assume good intent. Harassment of any kind isn't tolerated.
