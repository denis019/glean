"""glean over MCP — `transcribe` / `frames` / `fetch` as agent-callable tools.

A second FRONT-END over the existing library, not a second implementation: the same
manual→auto→ASR ladder, the same `build_ydl_opts` cookies + deno seam, and the same
friendly DRM / 403 / deno errors as the `glean` command. Register it with:

    claude mcp add glean -- uv run --directory /path/to/glean glean-mcp

Three things this front-end must get right that the CLI need not:

* **stdout IS the protocol.** On stdio transport one stray `print` corrupts the
  JSON-RPC stream and the session dies with an unexplained parse error. glean already
  writes every progress line to stderr, but a dependency need not, so `_own_stdout`
  points `sys.stdout` at stderr for the whole session.
* **`SystemExit` must not escape.** `source.classify` rejects a bad input that way and
  it descends from `BaseException`, so FastMCP would NOT catch it — a typo'd path
  would take the server down mid-session instead of returning a tool error. Every tool
  funnels through `_classify` / `_tool_errors`.
* **A transcript is big.** A two-hour lecture is tens of thousands of tokens of
  context, so `out` writes it to disk and returns a receipt instead of the text.
  Paths in and out are resolved absolute — the caller does not share glean's cwd.

Every tool blocks for a long time (a download, an ASR pass), so `_offloaded` runs the
body in a worker thread and leaves the event loop free to answer pings and cancels.
"""

from __future__ import annotations

import functools
import sys
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import anyio.to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from glean import asr as _asr
from glean import fetch as _fetch
from glean import frames as _frames
from glean import source
from glean.transcribe import transcribe as _transcribe

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator


@asynccontextmanager
async def _own_stdout(_server: FastMCP) -> AsyncIterator[None]:
    """Point `sys.stdout` at stderr for the session — stdout belongs to the protocol.

    A lifespan, deliberately: it runs INSIDE the stdio transport, which has already
    taken its own handle on the real stdout, so JSON-RPC still reaches the client
    while a stray `print` from anywhere else lands harmlessly on stderr next to
    glean's own progress lines. Doing the swap any earlier would hand the transport
    stderr and break the session outright.
    """
    saved, sys.stdout = sys.stdout, sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved


server = FastMCP(
    "glean",
    instructions=(
        "Extract study material from any lecture video — a YouTube or Udemy URL, or a "
        "local media file. `transcribe` yields [mm:ss]-stamped markdown (author "
        "subtitles → auto-captions → local faster-whisper ASR); `frames` grabs stills "
        "at those timestamps, which are authoritative for on-screen numbers the ASR "
        "garbles. Against a paid, rate-limited source call `fetch` ONCE and then point "
        "the other two at the file it kept. Udemy needs cookies_from_browser."
    ),
    lifespan=_own_stdout,
)


def _offloaded[R](fn: Callable[..., R]) -> Callable[..., Awaitable[R]]:
    """Run a blocking tool body in a worker thread so the session stays responsive.

    A transcribe is minutes of download plus ASR; running it inline would block the
    event loop, and the client's pings and cancellations with it. `functools.wraps`
    keeps the wrapped signature visible, which is what FastMCP builds the tool schema
    from — the wrapper never needs to restate the arguments.
    """

    @functools.wraps(fn)
    async def run(**kwargs: Any) -> R:  # noqa: ANN401 — pass-through of the tool's own args
        return await anyio.to_thread.run_sync(functools.partial(fn, **kwargs))

    return run


def _classify(target: str) -> source.Source:
    """`source.classify`, with its `SystemExit` rejection turned into a tool error."""
    try:
        return source.classify(target)
    except SystemExit as e:
        raise ToolError(str(e)) from e


@contextmanager
def _tool_errors(target: str, *, is_url: bool, has_cookies: bool) -> Iterator[None]:
    """Translate glean's failure modes into MCP tool errors, message intact.

    `ValueError` is a bad timestamp or window; `RuntimeError` is every network, DRM,
    auth and ffmpeg wall, rendered through the shared `source.explain`.
    """
    try:
        yield
    except ValueError as e:
        raise ToolError(str(e)) from e
    except RuntimeError as e:
        raise ToolError(source.explain(e, target, is_url=is_url, has_cookies=has_cookies)) from e


def _cookies(browser: str | None, file: str | None) -> bool:
    return bool(browser or file)


Video = Annotated[str, Field(description="video URL (YouTube/Udemy/…) or local media file path")]
Browser = Annotated[
    str | None,
    Field(description="read cookies from this logged-in browser (chrome/firefox/…) — Udemy auth"),
]
CookieFile = Annotated[
    str | None, Field(description="Netscape cookies.txt file — the other Udemy auth route")
]


@server.tool()
@_offloaded
def transcribe(
    video: Video,
    *,
    out: Annotated[
        str | None,
        Field(description="write the markdown here and return a receipt instead of the text"),
    ] = None,
    asr: Annotated[
        bool, Field(description="force local faster-whisper even when captions exist")
    ] = False,
    model: Annotated[str, Field(description="faster-whisper model, e.g. small.en")] = (
        _asr.DEFAULT_MODEL
    ),
    device: Annotated[Literal["auto", "cuda", "cpu"], Field(description="ASR device")] = "auto",
    lang: Annotated[str, Field(description="caption / transcription language")] = "en",
    cookies_from_browser: Browser = None,
    cookies_file: CookieFile = None,
) -> str:
    """Export a cleaned `[mm:ss]` transcript as markdown.

    Prefers the author's own subtitles, falls back to auto-captions, and only when a
    video has neither does it download the audio and transcribe locally with
    faster-whisper (minutes, and it garbles numbers and proper nouns — read the frames
    for those). A local file has no caption track, so it always goes to ASR.

    Returns the markdown itself, or — when `out` is given — a one-line receipt naming
    the file. Pass `out` for anything longer than a few minutes; a full lecture
    transcript is tens of thousands of tokens.
    """
    src = _classify(video)
    with _tool_errors(
        video, is_url=src.is_url, has_cookies=_cookies(cookies_from_browser, cookies_file)
    ):
        md = _transcribe(
            video,
            asr=asr,
            model=model,
            device=device,
            lang=lang,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
    if out is None:
        return md
    path = Path(out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return f"wrote {path} — {md.count(chr(10))} lines, {len(md)} chars"


@server.tool()
@_offloaded
def frames(
    video: Video,
    *,
    at: Annotated[
        list[str] | None, Field(description="timestamps to grab: SS, MM:SS or HH:MM:SS")
    ] = None,
    window: Annotated[
        list[str] | None, Field(description="ranges to sweep, e.g. '25:00-26:00'")
    ] = None,
    every: Annotated[int, Field(description="seconds between frames inside a window", gt=0)] = 15,
    out: Annotated[str | None, Field(description="output directory (default ./frames)")] = None,
    label: Annotated[str | None, Field(description="filename prefix (default 'frame')")] = None,
    fmt: Annotated[
        str, Field(description="yt-dlp format selector — URL input only")
    ] = _frames.DEFAULT_FORMAT,
    cookies_from_browser: Browser = None,
    cookies_file: CookieFile = None,
) -> list[str]:
    """Grab one still per timestamp. Returns the absolute paths written, in time order.

    The transcript says WHEN something is on screen (`[25:19]`); this grabs that frame
    so it can actually be read. For exact on-screen values these stills are
    authoritative and the transcript is not. Needs `ffmpeg` on PATH.

    Give `at` (exact stamps), `window` (a range, sampled every `every` seconds), or
    both — at least one. A URL is resolved by yt-dlp once and then seeked; a local file
    is seeked directly, with no request at all.
    """
    src = _classify(video)
    has_cookies = _cookies(cookies_from_browser, cookies_file)
    with _tool_errors(video, is_url=src.is_url, has_cookies=has_cookies):
        seconds = _frames.collect_seconds(at or [], window or [], every)
        if not seconds:
            raise ToolError("nothing to grab — pass `at` timestamps and/or a `window` range")
        written = _frames.extract_frames(
            src.value,
            seconds,
            Path(out or "frames").resolve(),
            is_url=src.is_url,
            fmt=fmt,
            label=label,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
    return [str(p) for p in written]


@server.tool()
@_offloaded
def fetch(
    url: Annotated[str, Field(description="video URL (YouTube/Udemy/…)")],
    *,
    out: Annotated[
        str | None, Field(description="target directory or file (default: ./<title>.<ext>)")
    ] = None,
    fmt: Annotated[str, Field(description="yt-dlp format selector")] = "best",
    cookies_from_browser: Browser = None,
    cookies_file: CookieFile = None,
) -> str:
    """Download the full video and KEEP it. Returns the absolute path written.

    `transcribe` and `frames` each hit the source independently, and on Udemy the
    lecture-isolation pass enumerates the whole course — doing both trips the rate
    limiter. Spend ONE request here, then call the other two with the returned path and
    they run locally for free. Use this for anything paid or rate-limited.
    """
    src = _classify(url)
    if not src.is_url:
        raise ToolError(f"{url} is already a local file — nothing to fetch.")
    with _tool_errors(url, is_url=True, has_cookies=_cookies(cookies_from_browser, cookies_file)):
        path = _fetch.download_media(
            url,
            Path(out).resolve() if out else Path.cwd(),
            fmt=fmt,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
    return str(path.resolve())


def main() -> None:
    """`glean-mcp` — serve the tools over stdio, the transport a local client spawns."""
    server.run()


if __name__ == "__main__":
    main()
