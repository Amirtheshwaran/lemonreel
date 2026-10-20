"""Command line interface: ``lemonreel make | remake | doctor | studio | voices``."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lemonreel import __version__
from lemonreel.config import KOKORO_VOICES, LOOKS, ReelConfig
from lemonreel.lemonade import LemonadeClient, LemonadeError
from lemonreel.omni import DEFAULT_OMNI, ToolkitError

console = Console(highlight=False)


def _add_reel_options(p: argparse.ArgumentParser) -> None:
    story = p.add_argument_group("story")
    story.add_argument("--scenes", type=int, default=6, help="number of scenes (default 6)")
    story.add_argument(
        "--minutes", type=float, default=1.5, help="target length in minutes (default 1.5)"
    )
    story.add_argument(
        "--mode",
        choices=["narrator", "duo"],
        default="narrator",
        help="one narrator, or a two-host conversation (default narrator)",
    )
    story.add_argument(
        "--language",
        default="en",
        help="language of the video: en, en-gb, es, fr, hi, it, ja, pt, zh (default en)",
    )
    story.add_argument(
        "--audience", default="curious people who are new to the topic", help="who the video is for"
    )
    story.add_argument(
        "--look",
        default="illustration",
        help=f"visual style: {', '.join(LOOKS)}, or any description",
    )

    voice = p.add_argument_group("voice")
    voice.add_argument("--voice", help="narrator (or first host) voice, e.g. af_heart")
    voice.add_argument("--voice2", help="second host voice in duo mode, e.g. am_michael")
    voice.add_argument("--speed", type=float, default=1.0, help="speaking speed (default 1.0)")

    picture = p.add_argument_group("picture")
    picture.add_argument(
        "--aspect",
        choices=["16:9", "9:16", "1:1"],
        default="16:9",
        help="16:9 for YouTube, 9:16 for Shorts/Reels/TikTok, 1:1 for feeds",
    )
    picture.add_argument("--hd", action="store_true", help="render at 1080p instead of 720p")
    picture.add_argument("--fps", type=int, default=30)
    picture.add_argument(
        "--captions",
        choices=["burn", "soft", "both", "none"],
        default="burn",
        help="burn into the picture, add as a subtitle track, both, or none",
    )
    picture.add_argument("--no-title", action="store_true", help="skip the title card")
    picture.add_argument("--no-outro", action="store_true", help="skip the closing credits card")
    picture.add_argument(
        "--upscale", action="store_true", help="upscale pictures with a downloaded upscaling model"
    )
    picture.add_argument("--steps", type=int, help="image generation steps (default: model's own)")
    picture.add_argument("--seed", type=int, help="base seed for reproducible pictures")

    checks = p.add_argument_group("self-checks")
    checks.add_argument(
        "--no-critique", action="store_true", help="don't let the vision model grade pictures"
    )
    checks.add_argument(
        "--min-score",
        type=int,
        default=3,
        help="redraw pictures scored below this (1-5, default 3)",
    )
    checks.add_argument(
        "--max-redraws", type=int, default=2, help="extra attempts per picture (default 2)"
    )
    checks.add_argument(
        "--no-listen-back", action="store_true", help="don't transcribe narration to check it"
    )
    checks.add_argument("--no-learn", action="store_true", help="don't save learned pronunciations")
    checks.add_argument(
        "--lexicon", help='JSON file of extra pronunciations {"word": "respelling"}'
    )

    models = p.add_argument_group("models")
    models.add_argument(
        "--omni",
        help=f"Lemonade Omni model to use (default: a downloaded one, e.g. {DEFAULT_OMNI})",
    )
    models.add_argument("--writer", help="override the LLM")
    models.add_argument("--illustrator", help="override the image model")
    models.add_argument("--voice-model", help="override the text-to-speech model")
    models.add_argument("--listener", help="override the speech-recognition model")

    run = p.add_argument_group("run")
    run.add_argument("--out", default="reels", help="folder for finished reels (default ./reels)")
    run.add_argument("--base-url", help="Lemonade Server URL (default http://localhost:13305)")
    run.add_argument(
        "--timeout", type=float, default=900, help="seconds to wait for one model call"
    )
    run.add_argument("--sequential", action="store_true", help="draw and speak one after the other")
    run.add_argument("--open", action="store_true", help="open the report in a browser when done")


def _config(args: argparse.Namespace) -> ReelConfig:
    return ReelConfig(
        scenes=args.scenes,
        minutes=args.minutes,
        mode=args.mode,
        language=args.language,
        audience=args.audience,
        look=args.look,
        voice=args.voice,
        voice2=args.voice2,
        speed=args.speed,
        aspect=args.aspect,
        hd=args.hd,
        fps=args.fps,
        captions=args.captions,
        title_card=not args.no_title,
        outro=not args.no_outro,
        upscale=args.upscale,
        image_steps=args.steps,
        seed=args.seed,
        critique=not args.no_critique,
        min_score=args.min_score,
        max_image_attempts=1 + max(0, args.max_redraws),
        listen_back=not args.no_listen_back,
        learn=not args.no_learn,
        lexicon_path=args.lexicon,
        parallel=not args.sequential,
        omni=args.omni,
        writer=args.writer,
        illustrator=args.illustrator,
        voice_model=args.voice_model,
        listener=args.listener,
    )


def _printer():
    from lemonreel.pipeline import Event

    def show(event: Event) -> None:
        if event.kind == "stage" and event.data.get("state") == "start":
            console.print(f"[bold yellow]>[/] [bold]{event.message}[/]")
        elif event.kind == "stage" and event.data.get("state") == "end":
            console.print(
                f"  [dim]{event.stage} finished in {event.data.get('seconds', 0):.1f}s[/]"
            )
        elif event.kind == "log":
            console.print(f"  [dim]{event.message}[/]")
        elif event.kind == "art":
            score = event.data.get("score")
            extra = f" - critic {score}/5" if score else ""
            if event.data.get("attempts", 1) > 1:
                extra += f" after {event.data['attempts']} tries"
            console.print(f"  [green]picture[/] scene {event.data['index'] + 1}{extra}")
        elif event.kind == "voice":
            wer = event.data.get("wer")
            extra = f" - heard back with {wer:.0%} word error" if wer is not None else ""
            console.print(
                f"  [cyan]voice[/]   scene {event.data['index'] + 1} ({event.data.get('duration', 0):.1f}s){extra}"
            )
        elif event.kind == "storyboard":
            console.print(f'  [bold]"{event.data["title"]}"[/]')
            for i, scene in enumerate(event.data["scenes"], 1):
                console.print(f"  [dim]{i:02d}[/] {scene['caption']}")

    return show


def cmd_make(args: argparse.Namespace, *, remake: bool = False) -> int:
    from lemonreel.pipeline import Reel

    cfg = _config(args)
    try:
        client = LemonadeClient.discover(args.base_url, timeout=args.timeout)
    except LemonadeError as exc:
        console.print(f"[red]{exc}[/]")
        return 2
    reel = Reel(cfg, client, out_root=Path(args.out), on_event=_printer())
    try:
        if remake:
            workdir = Path(args.folder)
            board = workdir / "storyboard.json"
            if not board.exists():
                console.print(f"[red]{board} not found.[/]")
                return 2
            result = reel.run(storyboard=board, workdir=workdir)
        else:
            result = reel.run(args.source, storyboard_only=args.storyboard_only)
    except ToolkitError as exc:
        console.print(f"[red]{exc}[/]")
        return 2
    except (LemonadeError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Failed:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]Stopped.[/]")
        return 130

    if result.video is None:
        board_path = result.workdir / "storyboard.json"
        console.print(
            Panel.fit(
                f"Storyboard written to [bold]{board_path}[/]\n"
                f"Edit it, then finish the video with:\n  lemonreel remake {result.workdir}",
                title="Storyboard ready",
                border_style="yellow",
            )
        )
        return 0
    timeline = result.timeline
    assert timeline is not None and result.report_html is not None
    console.print(
        Panel.fit(
            f"[bold]{result.storyboard.title}[/]\n"
            f"Video     {result.video}\n"
            f"Report    {result.report_html}\n"
            f"Captions  {result.workdir / 'captions.srt'}\n"
            f"Length    {timeline.duration:.0f}s video, made in {result.seconds:.0f}s",
            title="Done",
            border_style="green",
        )
    )
    if args.open:
        webbrowser.open(result.report_html.resolve().as_uri())
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from lemonreel.doctor import diagnose, smoke_test

    try:
        client = LemonadeClient.discover(args.base_url)
    except LemonadeError as exc:
        console.print(f"[red]x[/] {exc}")
        return 2
    diag = diagnose(client, args.omni)
    console.print(f"[green]ok[/] Lemonade Server {diag.version or '?'} at {diag.url}")
    for line in diag.hardware:
        console.print(f"   {line}")
    if diag.omni:
        console.print("\n[bold]Lemonade Omni models[/]")
        for m in diag.omni:
            mark = "[green]ok[/]" if m["downloaded"] else "[dim]--[/]"
            hint = "" if m["downloaded"] else f"  [dim](lemonade pull {m['id']})[/]"
            console.print(f" {mark} {m['id']}{hint}")
    console.print()
    if diag.toolkit:
        tk = diag.toolkit
        table = Table(title=f"Roles ({tk.omni or 'individually selected models'})", show_edge=False)
        table.add_column("Role")
        table.add_column("Model")
        table.add_column("Extras", style="dim")
        table.add_row(
            "Writer",
            tk.writer.id,
            "can judge pictures" if tk.can_see else "no vision: pictures won't be graded",
        )
        table.add_row(
            "Illustrator",
            tk.illustrator.id if tk.illustrator else "-",
            "" if tk.illustrator else "text slides instead of pictures",
        )
        table.add_row("Voice", tk.voice.id, "")
        table.add_row(
            "Listener",
            tk.listener.id if tk.listener else "-",
            "listen-back and timed captions" if tk.listener else "no listen-back",
        )
        console.print(table)
    else:
        console.print(f"[red]x[/] {diag.toolkit_error}")
    if diag.ffmpeg:
        console.print(f"[green]ok[/] ffmpeg: {diag.ffmpeg}")
    else:
        console.print(f"[red]x[/] {diag.ffmpeg_error}")

    if args.smoke and diag.toolkit:
        console.print("\n[bold]Smoke test[/] (first calls include model loading)")
        table = Table(show_edge=False)
        for col in ("Role", "Model", "Time", "Result"):
            table.add_column(col)
        for r in smoke_test(client, diag.toolkit):
            status = r["detail"] if r["ok"] else f"[red]{r['detail']}[/]"
            table.add_row(r["role"], r["model"], f"{r['seconds']:.1f}s", status)
        console.print(table)
    if diag.ready:
        console.print("\n[bold green]Ready.[/] Try: lemonreel make README.md")
        return 0
    return 1


def cmd_studio(args: argparse.Namespace) -> int:
    from lemonreel.studio.app import serve

    return serve(
        host=args.host,
        port=args.port,
        out=Path(args.out),
        base_url=args.base_url,
        open_browser=not args.no_browser,
    )


def cmd_voices(args: argparse.Namespace) -> int:
    for group, voices in KOKORO_VOICES.items():
        console.print(f"[bold]{group}[/]: {', '.join(voices)}")
    console.print(
        "\n[dim]Voice names starting with 'af'/'bf' are female, 'am'/'bm' male. Use --voice and --voice2.[/]"
    )
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    from lemonreel.omni import resolve_toolkit
    from lemonreel.say import make_voiceover, read_script

    script = read_script(args.script)
    script_path = Path(args.script)
    folder = (
        Path(args.out)
        if args.out
        else (
            script_path.with_name(script_path.stem + "-voiceover")
            if script_path.is_file()
            else Path("voiceover")
        )
    )
    try:
        cfg = ReelConfig(
            language=args.language,
            voice=args.voice,
            speed=args.speed,
            listen_back=not args.no_listen_back,
            learn=not args.no_learn,
            lexicon_path=args.lexicon,
        )
        client = LemonadeClient.discover(args.base_url, timeout=args.timeout)
        toolkit = resolve_toolkit(
            client.models(show_all=True),
            args.omni,
            writer=args.writer,
            voice=args.voice_model,
            listener=args.listener,
        )
        console.print(
            f"[bold yellow]>[/] Speaking with {toolkit.voice.id} ({cfg.voices[0]})"
            + (
                f", listening back with {toolkit.listener.id}"
                if toolkit.listener and cfg.listen_back
                else ""
            )
        )
        result = make_voiceover(
            client,
            toolkit,
            cfg,
            script,
            folder,
            pause=args.pause,
            log=lambda m: console.print(f"  [dim]{m}[/]"),
        )
    except ToolkitError as exc:
        console.print(f"[red]{exc}[/]")
        return 2
    except (LemonadeError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Failed:[/] {exc}")
        return 1
    fixes = {k: v for part in result.parts for k, v in part.fixes.items()}
    console.print(
        Panel.fit(
            f"Voiceover  {result.wav}  ({result.duration:.1f}s, {len(result.parts)} parts)\n"
            f"Parts      {result.folder}/part-NN.wav\n"
            f"Captions   {result.srt}\n"
            f"Timing     {result.timing}"
            + (
                f"\nFixed      {', '.join(f'{k} -> {v}' for k, v in fixes.items())}"
                if fixes
                else ""
            ),
            title="Voiceover ready",
            border_style="green",
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lemonreel",
        description="Turn documents, web pages and voice memos into narrated, illustrated videos "
        "with Lemonade Omni models, entirely on your own PC.",
    )
    parser.add_argument("--version", action="version", version=f"lemonreel {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make", help="make a video from a file, URL, voice memo or topic")
    make.add_argument(
        "source",
        help="Markdown/text/HTML/PDF file, URL, GitHub repo URL, audio file, or a topic in quotes",
    )
    make.add_argument(
        "--storyboard-only",
        action="store_true",
        help="stop after writing storyboard.json so you can edit it",
    )
    _add_reel_options(make)
    make.set_defaults(func=cmd_make)

    remake = sub.add_parser(
        "remake", help="finish or redo a reel after editing its storyboard.json"
    )
    remake.add_argument("folder", help="the reel folder that contains storyboard.json")
    _add_reel_options(remake)
    remake.set_defaults(func=lambda a: cmd_make(a, remake=True))

    doctor = sub.add_parser("doctor", help="check Lemonade, models and ffmpeg")
    doctor.add_argument(
        "--smoke", action="store_true", help="also time one small request per model"
    )
    doctor.add_argument("--omni", help="Omni model to check")
    doctor.add_argument("--base-url")
    doctor.set_defaults(func=cmd_doctor)

    studio = sub.add_parser("studio", help="open the browser studio")
    studio.add_argument("--host", default="127.0.0.1")
    studio.add_argument("--port", type=int, default=8765)
    studio.add_argument("--out", default="reels")
    studio.add_argument("--base-url")
    studio.add_argument("--no-browser", action="store_true")
    studio.set_defaults(func=cmd_studio)

    voices = sub.add_parser("voices", help="list Kokoro voices")
    voices.set_defaults(func=cmd_voices)

    say = sub.add_parser(
        "say",
        help="turn a script into a local voiceover (WAV per part, full WAV, SRT, timing sheet)",
        description="Narrate a script with the Omni model's voice. Blank lines separate parts; "
        "'# headings' and '[stage directions]' are not read aloud.",
    )
    say.add_argument("script", help="a .txt/.md script file, or the text itself in quotes")
    say.add_argument("--out", help="output folder (default: <script>-voiceover next to the script)")
    say.add_argument("--voice", help="Kokoro voice (default af_heart; see `lemonreel voices`)")
    say.add_argument("--speed", type=float, default=1.0, help="speaking speed (default 1.0)")
    say.add_argument("--pause", type=float, default=0.7, help="seconds of silence between parts")
    say.add_argument("--language", default="en", help="en, en-gb, es, fr, hi, it, ja, pt, zh")
    say.add_argument("--no-listen-back", action="store_true", help="don't check pronunciation")
    say.add_argument("--no-learn", action="store_true", help="don't save learned pronunciations")
    say.add_argument("--lexicon", help="JSON file of extra pronunciations")
    say.add_argument("--omni", help="Omni model to use")
    say.add_argument("--writer", help="override the LLM used for respellings")
    say.add_argument("--voice-model", help="override the text-to-speech model")
    say.add_argument("--listener", help="override the speech-recognition model")
    say.add_argument("--base-url", help="Lemonade Server URL")
    say.add_argument("--timeout", type=float, default=900)
    say.set_defaults(func=cmd_say)
    return parser


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
