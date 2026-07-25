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
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from glean import source

if TYPE_CHECKING:
    from collections.abc import Mapping

_TAG = re.compile(r"<[^>]+>")  # inline <00:..> and <c>..</c>
# WebVTT allows the hour field to be omitted, so both `00:01:30.500` and `01:30.500`
# are legal cue timings. Requiring the hour made every short-form cue unmatchable,
# which left `start` None for every block — a silently EMPTY transcript, not an error.
_TS = re.compile(r"^(?:(\d+):)?(\d{2}):(\d{2})\.\d+\s+-->")
_HEADER_PREFIXES = ("WEBVTT", "Kind:", "Language:")

CaptionKind = Literal["manual", "auto"]


class CaptionTrack(NamedTuple):
    """The chosen caption track: how it was produced, and its ACTUAL info-dict key.

    `lang` is the key exactly as yt-dlp reported it (``"en-US"``), NOT the language the
    caller asked for (``"en"``). The download step must request that exact key or
    yt-dlp writes no VTT at all.
    """

    kind: CaptionKind
    lang: str


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
    # Cue blocks are blank-line separated, but the lines within one are read with
    # `splitlines()`. A VTT served with CRLF (or classic-Mac CR) endings therefore
    # never splits — "\r\n\r\n" contains no "\n\n" — collapsing the WHOLE file into a
    # single block whose `start` ends up as the LAST cue's time, which every line then
    # inherits. Wrong-but-plausible timestamps are worse than none, since locating
    # moments for `glean frames` is the transcript's job. Normalise endings first.
    normalised = vtt_text.replace("\r\n", "\n").replace("\r", "\n")
    for block in normalised.split("\n\n"):
        start: int | None = None
        body: list[str] = []
        for ln in block.splitlines():
            m = _TS.match(ln)
            if m:
                hours = int(m[1]) if m[1] else 0
                start = hours * 3600 + int(m[2]) * 60 + int(m[3])
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


def _best_lang_key(tracks: Mapping[str, Any], lang: str) -> str | None:
    """The key in `tracks` that serves `lang`: the exact tag, else a regional variant.

    yt-dlp keys tracks by the uploader's own BCP-47 tag, so an English track routinely
    arrives as ``en-US`` / ``en-GB`` / ``en-orig`` rather than a bare ``en``. Matching
    only the exact tag reported "no captions" for all of those and fell through to a
    full local ASR run — burning minutes of GPU and the number-garbling that comes with
    it on a video that has perfectly good author subtitles. That is the exact failure
    this module exists to prevent (see the module docstring).

    Sub-tags are ``-``-delimited, so a variant of `lang` is `lang`-then-hyphen; ``eng``
    is a *different* tag and must not match ``en``. An exact hit always wins over a
    variant, and variants are sorted so the choice is deterministic.
    """
    available = [key for key, tracks_for_key in tracks.items() if tracks_for_key]
    if lang in available:
        return lang
    variants = sorted(key for key in available if key.startswith(f"{lang}-"))
    return variants[0] if variants else None


def choose_caption_track(info: Mapping[str, Any], lang: str) -> CaptionTrack | None:
    """Pick the caption track for `lang` from a yt-dlp info dict: manual over auto.

    Manual (``subtitles``) is the author's exact words; auto (``automatic_captions``)
    is ASR-in-the-cloud. Returns None when neither has the language — the signal to
    fall back to local ASR. Pure (review S1).

    Returns the track's real key alongside its kind, because the caller must hand that
    key back to yt-dlp as ``subtitleslangs``: asking for ``en`` when the track is filed
    under ``en-US`` downloads nothing.
    """
    key = _best_lang_key(info.get("subtitles") or {}, lang)
    if key is not None:
        return CaptionTrack("manual", key)
    key = _best_lang_key(info.get("automatic_captions") or {}, lang)
    if key is not None:
        return CaptionTrack("auto", key)
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
    track via `choose_caption_track`, then downloads exactly that kind — setting only
    the matching yt-dlp flag so the two never collide on the same filename, and
    requesting the track's OWN language key so a regional tag still downloads. Raises
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
    track = choose_caption_track(info or {}, lang)
    if track is None:
        raise NoCaptionsError(
            f"no caption track for {url} (lang={lang}) — neither author nor auto captions."
        )

    with tempfile.TemporaryDirectory() as td:
        call = source.build_ydl_opts(
            {
                "skip_download": True,
                "writesubtitles": track.kind == "manual",
                "writeautomaticsub": track.kind == "auto",
                "subtitleslangs": [track.lang],
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
                f"yt-dlp reported a {track.kind} caption track for {url} "
                f"(lang={track.lang}) but wrote no VTT."
            )
        return vtts[0].read_text(encoding="utf-8"), track.kind
