"""Fetch + clean a subtitle track for a video URL — author subs preferred over auto.

Two layers:

* **`clean_vtt`** — the pure, tested core. yt-dlp caption VTT is a "rolling" format:
  each cue repeats the previous settled line plus a still-typing active line carrying
  inline ``<00:00:12.000><c>word</c>`` timing tags. The readable text is the SETTLED
  line of each cue; `clean_vtt` extracts those, deduped, entities resolved.
* **`fetch_captions`** — the network path. It requests BOTH manual (author) and auto
  tracks and **prefers the manual one**: asking only for ``writeautomaticsub``
  silently discards accurate author/instructor subtitles — common on Udemy and on
  many YouTube uploads — burning ASR on mishears while a human transcript sits
  unused. `choose_caption_kind` is the pure selection rule; fetch only falls through
  to ASR when *neither* track exists.
"""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from glean import source

if TYPE_CHECKING:
    from collections.abc import Mapping

_TAG = re.compile(r"<[^>]+>")  # inline <00:..> and <c>..</c>
_TS = re.compile(r"^(\d\d):(\d\d):(\d\d)\.\d+\s+-->")
_HEADER_PREFIXES = ("WEBVTT", "Kind:", "Language:")

CaptionKind = Literal["manual", "auto"]


class NoCaptionsError(RuntimeError):
    """Raised when a URL has neither a manual nor an auto caption track — go to ASR."""


def clean_vtt(vtt_text: str) -> list[tuple[int, str]]:
    """Rolling-caption VTT -> [(start_seconds, line)], deduped, tags/entities resolved.

    A line is kept only when it is SETTLED (no inline `<c>` tag) — the still-typing
    active line is skipped, because its finished form re-appears settled in the next
    cue. Consecutive exact repeats are dropped.
    """
    out: list[tuple[int, str]] = []
    last: str | None = None
    for block in vtt_text.split("\n\n"):
        start: int | None = None
        body: list[str] = []
        for ln in block.splitlines():
            m = _TS.match(ln)
            if m:
                start = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            elif ln and not ln.startswith(_HEADER_PREFIXES):
                body.append(ln)
        if start is None:
            continue
        for raw in body:
            if _TAG.search(raw):
                continue  # active (still-typing) line — its settled form follows
            clean = html.unescape(raw).strip()
            if not clean or clean == last:
                continue
            out.append((start, clean))
            last = clean
    return out


def choose_caption_kind(info: Mapping[str, Any], lang: str) -> CaptionKind | None:
    """Pick the caption track for `lang` from a yt-dlp info dict: manual over auto.

    Manual (``subtitles``) is the author's exact words; auto (``automatic_captions``)
    is ASR-in-the-cloud. Returns None when neither has the language — the signal to
    fall back to local ASR. Pure (review S1).
    """
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    if manual.get(lang):
        return "manual"
    if auto.get(lang):
        return "auto"
    return None


def fetch_captions(
    url: str,
    *,
    lang: str = "en",
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> tuple[str, CaptionKind]:  # pragma: no cover — network + yt-dlp
    """Fetch the best available caption VTT for `url`. Returns (vtt_text, kind).

    Probes the video (surfacing BOTH manual and auto tracks), prefers the manual
    track via `choose_caption_kind`, then downloads exactly that kind — setting only
    the matching yt-dlp flag so the two never collide on the same filename. Raises
    `NoCaptionsError` when the video has no track in `lang` (caller goes to ASR).
    All yt-dlp opts go through `source.build_ydl_opts` (cookies + deno — review S2).
    """
    import yt_dlp  # noqa: PLC0415 — lazy: heavy import, network-only path

    # Isolate the exact lecture (Udemy URLs expand to the whole course); resolved
    # once and reused for both the probe and the caption download.
    sel = source.playlist_selection(
        url, cookies_from_browser=cookies_from_browser, cookies_file=cookies_file
    )
    probe = source.build_ydl_opts(
        {"skip_download": True, **sel},
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    with source.ydl_errors(), source.patched_path(probe.env), yt_dlp.YoutubeDL(probe.opts) as ydl:
        info = ydl.extract_info(url, download=False)
    kind = choose_caption_kind(info or {}, lang)
    if kind is None:
        raise NoCaptionsError(
            f"no caption track for {url} (lang={lang}) — neither author nor auto captions."
        )

    with tempfile.TemporaryDirectory() as td:
        call = source.build_ydl_opts(
            {
                "skip_download": True,
                "writesubtitles": kind == "manual",
                "writeautomaticsub": kind == "auto",
                "subtitleslangs": [lang],
                "subtitlesformat": "vtt",
                "outtmpl": {"default": str(Path(td) / "sub")},
                **sel,
            },
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
        with source.ydl_errors(), source.patched_path(call.env), yt_dlp.YoutubeDL(call.opts) as ydl:
            ydl.download([url])
        vtts = list(Path(td).glob("*.vtt"))
        if not vtts:
            raise NoCaptionsError(
                f"yt-dlp reported a {kind} caption track for {url} but wrote no VTT."
            )
        return vtts[0].read_text(encoding="utf-8"), kind
