"""`glean` — pull a transcript or still frames out of any lecture video or local file.

Subcommands over one shared media layer:

    glean transcribe <url|file> [-o OUT.md] [--asr] [--model medium.en]
                                [--device auto|cuda|cpu] [--lang en]
                                [--cookies-from-browser B | --cookies FILE]
    glean frames     <url|file> --at TS --window A-B [--every N] [-o DIR]
                                [--cookies-from-browser B | --cookies FILE]
    glean fetch      <url>      [-o DIR|FILE] [--format best]
                                [--cookies-from-browser B | --cookies FILE]

yt-dlp dispatches URLs by site, so YouTube and Udemy are the same path (Udemy needs
`--cookies-from-browser` to read the logged-in session). A local file is transcribed
by ASR directly and frame-seeked without yt-dlp. `fetch` downloads + KEEPS the full
video so both transcribe and frames can run locally afterwards — one request instead
of several (avoids Udemy's rate limiter).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from glean import asr, fetch, frames, source
from glean.source import InputKind
from glean.transcribe import transcribe


def _cmd_transcribe(args: argparse.Namespace) -> int:
    src = source.classify(args.input)
    try:
        md = transcribe(
            args.input,
            asr=args.asr,
            model=args.model,
            device=args.device,
            lang=args.lang,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies,
        )
    except RuntimeError as e:
        raise SystemExit(_error_text(e, src, args)) from e
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({md.count(chr(10))} lines)", file=sys.stderr)
    else:
        sys.stdout.write(md)
    return 0


def _has_cookies(args: argparse.Namespace) -> bool:
    return bool(args.cookies_from_browser or args.cookies)


def _error_text(exc: RuntimeError, src: source.Source, args: argparse.Namespace) -> str:
    """Render a failed grab (review finding 2) — argparse-shaped view of `source.explain`."""
    return source.explain(exc, args.input, is_url=src.is_url, has_cookies=_has_cookies(args))


def _cmd_frames(args: argparse.Namespace) -> int:
    src = source.classify(args.input)
    # parse_ts / expand_window reject bad input with a ValueError carrying an already
    # actionable message — but nothing caught it, so an ordinary typo ("--at 1:2:3:4",
    # a backwards window, "--every 0") printed a raw traceback instead. Funnel it to
    # SystemExit like every other user-facing error in the tool.
    try:
        seconds = frames.collect_seconds(args.at, args.window, args.every)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if not seconds:
        raise SystemExit("nothing to grab — pass --at TS (repeatable) and/or --window A-B")
    out_dir = args.out or Path("frames")
    kind = "file" if src.kind is InputKind.FILE else "stream"
    print(f"grabbing {len(set(seconds))} frame(s) from {kind} …", file=sys.stderr)
    try:
        written = frames.extract_frames(
            src.value,
            seconds,
            out_dir,
            is_url=src.is_url,
            fmt=args.format,
            label=args.label,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies,
        )
    except RuntimeError as e:
        raise SystemExit(_error_text(e, src, args)) from e
    for p in written:
        print(p)
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    src = source.classify(args.input)
    if not src.is_url:
        raise SystemExit(f"{args.input} is already a local file — nothing to fetch.")
    try:
        path = fetch.download_media(
            args.input,
            args.out,
            fmt=args.format,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies,
        )
    except RuntimeError as e:
        raise SystemExit(_error_text(e, src, args)) from e
    print(path)
    print(
        f"kept — now run locally (no more requests):\n"
        f"  glean transcribe {path} -o transcript.md\n"
        f"  glean frames {path} --at MM:SS",
        file=sys.stderr,
    )
    return 0


def _add_cookie_flags(sub: argparse.ArgumentParser) -> None:
    g = sub.add_mutually_exclusive_group()
    g.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="read cookies from a logged-in browser (chrome/firefox/…) — Udemy auth",
    )
    g.add_argument("--cookies", metavar="FILE", help="Netscape cookies.txt file — Udemy auth")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glean", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="export a cleaned [mm:ss] transcript")
    t.add_argument("input", help="video URL (YouTube/Udemy/…) or local media file")
    t.add_argument("-o", "--out", type=Path, help="output .md (default: stdout)")
    t.add_argument(
        "--asr",
        action="store_true",
        help="force local transcription (faster-whisper) even if captions exist; "
        "auto-used as a fallback when a URL has no caption track",
    )
    t.add_argument("--model", default=asr.DEFAULT_MODEL, help="faster-whisper model")
    t.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="ASR device")
    t.add_argument("--lang", default="en", help="caption / transcription language (default en)")
    _add_cookie_flags(t)
    t.set_defaults(func=_cmd_transcribe)

    f = sub.add_parser("frames", help="extract still frames at timestamps")
    f.add_argument("input", help="video URL (YouTube/Udemy/…) or local media file")
    f.add_argument(
        "--at", action="append", default=[], help="timestamp SS/MM:SS/HH:MM:SS (repeatable)"
    )
    f.add_argument("--window", action="append", default=[], help="A-B range (repeatable)")
    f.add_argument(
        "--every", type=int, default=15, help="seconds between window frames (default 15)"
    )
    f.add_argument("-o", "--out", type=Path, help="output dir (default: ./frames)")
    f.add_argument("--format", default=frames.DEFAULT_FORMAT, help="yt-dlp format selector (URL)")
    f.add_argument("--label", help="filename prefix (default 'frame')")
    _add_cookie_flags(f)
    f.set_defaults(func=_cmd_frames)

    d = sub.add_parser(
        "fetch", help="download + KEEP the full video, then transcribe/frames it locally"
    )
    d.add_argument("input", help="video URL (YouTube/Udemy/…)")
    d.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(),
        help="target dir or file (default: ./<title>.<ext>)",
    )
    d.add_argument("--format", default="best", help="yt-dlp format selector (default best)")
    _add_cookie_flags(d)
    d.set_defaults(func=_cmd_fetch)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
