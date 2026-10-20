# Examples

| Try | Command |
|-----|---------|
| A short local-AI explainer from a Markdown file | `lemonreel make examples/why-local-ai.md` |
| The same as a two-host vertical short | `lemonreel make examples/why-local-ai.md --mode duo --aspect 9:16 --scenes 4 --minutes 1` |
| In Spanish, watercolor look | `lemonreel make examples/why-local-ai.md --language es --look watercolor` |
| A GitHub project's README | `lemonreel make https://github.com/lemonade-sdk/lemonade` |
| Extra pronunciations | `lemonreel make notes.md --lexicon examples/lexicon.json` |

`lexicon.json` shows the format for pronunciation files: the written word on the left, how the voice
should say it on the right. Captions keep the written form.
