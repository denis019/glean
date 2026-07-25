"""Render cleaned cues as a `[mm:ss]` transcript with a source-agnostic provenance.

`to_markdown` is pure. Its provenance line names WHERE the text came from and HOW it
was produced, and is generalized (review M3) over three provenances × two source
kinds:

* provenance: **manual** captions (author's exact words), **auto** captions (cloud
  ASR), or local **faster-whisper** ASR.
* source kind: a **URL** (rendered ``<url>``) — YouTube, Udemy, anything yt-dlp
  supports — or a **local path** (rendered as an inline code span). No hardcoded
  "YouTube" anywhere.
"""

from __future__ import annotations

from typing import Literal

CaptionKind = Literal["manual", "auto"]


class EmptyTranscriptError(RuntimeError):
    """Raised when there are no cues to render — refuse to write an empty transcript.

    Every provenance (captions, ASR-from-URL, ASR-from-file) renders through
    `to_markdown`, so this is the one place that can catch a zero-cue result for all
    of them. Without it a malformed caption track produced a *confident* file — a
    header asserting "author-provided subtitles", a duration of "~0h0m" and no
    content — and the CLI cheerfully reported "wrote transcript.md (8 lines)". A
    silently wrong answer is worse than a loud failure.
    """


def _source_ref(source: str, *, is_url: bool) -> str:
    """Render the source: ``<url>`` for URLs, an inline code span for local paths."""
    return f"<{source}>" if is_url else f"`{source}`"


def to_markdown(
    cues: list[tuple[int, str]],
    *,
    source: str,
    is_url: bool = True,
    asr_model: str | None = None,
    caption_kind: CaptionKind = "auto",
) -> str:
    """Format cleaned cues as a `[mm:ss]` transcript with a provenance header.

    `asr_model` set = the cues came from local speech-to-text (no usable caption
    track); provenance names the model. Otherwise `caption_kind` distinguishes an
    author-provided ("manual") track from an auto-generated one. `is_url` controls
    how the source is rendered (URL vs local path — review M3).

    Raises `EmptyTranscriptError` if there is nothing to render.
    """
    if not cues:
        cause = (
            "the audio produced no speech"
            if asr_model
            else "the caption track parsed to zero cues — malformed VTT, or a layout "
            "this cleaner does not recognise"
        )
        hint = "" if asr_model else " Re-run with --asr to transcribe the audio locally instead."
        raise EmptyTranscriptError(f"no transcript content for {source} — {cause}.{hint}")
    dur = cues[-1][0]
    span = f"~{dur // 3600}h{(dur % 3600) // 60}m"
    ref = _source_ref(source, is_url=is_url)
    if asr_model:
        provenance = (
            f"**Source:** {ref} · **locally transcribed** with faster-whisper "
            f"`{asr_model}` (no usable caption track). {span}."
        )
        caveat = (
            "**Read with care:** these are Whisper mishears, not the speaker's — numbers, "
            "tickers and price levels especially. For exact levels the frames are "
            "authoritative. `[mm:ss]` is VIDEO position, NOT wall-clock time."
        )
    elif caption_kind == "manual":
        provenance = (
            f"**Source:** {ref} · **author-provided** subtitles (yt-dlp), cleaned "
            "(inline tags stripped, rolling duplicates deduped, HTML entities unescaped). "
            f"{span}."
        )
        caveat = (
            "**Read with care:** these are the author's own subtitles — the most accurate "
            "track, but still transcribe timing/typos as written. `[mm:ss]` is VIDEO "
            "position, NOT wall-clock time."
        )
    else:
        provenance = (
            f"**Source:** {ref} · **auto-generated** captions (yt-dlp), cleaned "
            "(inline tags stripped, rolling duplicates deduped, HTML entities unescaped). "
            f"{span}."
        )
        caveat = (
            "**Read with care:** auto-caption mishears are the ASR's, not the speaker's — "
            'numbers especially (it renders "15-minute" as "50-minute"). `[mm:ss]` is '
            "VIDEO position, NOT wall-clock time."
        )
    lines = ["# Transcript", "", provenance, "", caveat, "", "---", ""]
    lines.extend(f"[{s // 60:02d}:{s % 60:02d}] {line}" for s, line in cues)
    return "\n".join(lines) + "\n"
