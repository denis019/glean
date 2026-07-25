"""Caption cleaning + kind selection — the pure, tested core (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import pytest

from glean import captions

FIXTURE = Path(__file__).parent / "fixtures" / "rolling.vtt"


def test_clean_vtt_extracts_settled_lines_only() -> None:
    cues = captions.clean_vtt(FIXTURE.read_text(encoding="utf-8"))
    lines = [line for _, line in cues]
    # Each spoken line appears ONCE, in order — no rolling duplicates, no half-typed
    # active lines, no inline <c> tags.
    assert lines == [
        "[music]",
        "I'm just going to check gold.",
        "Yeah, bearish bias today.",
        "I'll buy above resistance.",
    ]


def test_timestamps_are_cue_start_seconds() -> None:
    cues = captions.clean_vtt(FIXTURE.read_text(encoding="utf-8"))
    starts = {line: s for s, line in cues}
    assert starts["I'm just going to check gold."] == 14  # 00:00:14.400 -> 14s
    assert starts["I'll buy above resistance."] == 24


def test_no_inline_tags_survive() -> None:
    cues = captions.clean_vtt(FIXTURE.read_text(encoding="utf-8"))
    assert not any("<" in line for _, line in cues)


def test_html_entities_unescaped() -> None:
    vtt = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhe said &gt;&gt; buy &amp; hold, it&#39;s fine\n"
    )
    ((_, line),) = captions.clean_vtt(vtt)
    assert line == "he said >> buy & hold, it's fine"


def test_consecutive_exact_repeats_dropped() -> None:
    vtt = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nsame line\n\n"
        "00:00:03.000 --> 00:00:04.000\nsame line\n\n"
        "00:00:05.000 --> 00:00:06.000\nnew line\n"
    )
    assert [line for _, line in captions.clean_vtt(vtt)] == ["same line", "new line"]


def test_empty_vtt_is_empty() -> None:
    assert captions.clean_vtt("WEBVTT\n\n") == []


# ---- caption kind selection (review S1) -------------------------------------


def test_choose_prefers_manual_over_auto() -> None:
    # Both tracks present for the language — the author's own subtitles win.
    info = {
        "subtitles": {"en": [{"ext": "vtt"}]},
        "automatic_captions": {"en": [{"ext": "vtt"}]},
    }
    assert captions.choose_caption_kind(info, "en") == "manual"


def test_choose_auto_when_only_auto() -> None:
    info = {"subtitles": {}, "automatic_captions": {"en": [{"ext": "vtt"}]}}
    assert captions.choose_caption_kind(info, "en") == "auto"


def test_choose_manual_when_only_manual() -> None:
    info = {"subtitles": {"en": [{"ext": "vtt"}]}, "automatic_captions": {}}
    assert captions.choose_caption_kind(info, "en") == "manual"


def test_choose_none_when_neither() -> None:
    # No track in the requested language -> None -> caller falls to ASR.
    info = {"subtitles": {"de": [{"ext": "vtt"}]}, "automatic_captions": {"fr": [{"ext": "vtt"}]}}
    assert captions.choose_caption_kind(info, "en") is None


def test_choose_none_on_empty_info() -> None:
    assert captions.choose_caption_kind({}, "en") is None


# ---- real DownloadError from the yt-dlp call site converts (review finding 1) ----


class _FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL whose extract_info raises a REAL DownloadError."""

    def __init__(self, opts: object) -> None:
        self._opts = opts

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def extract_info(self, url: str, *, download: bool = False) -> dict:
        from yt_dlp.utils import DownloadError

        raise DownloadError(f"HTTP Error 403: Forbidden ({url}, download={download})")


def test_fetch_captions_converts_download_error_to_runtimeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercises the REAL fetch_captions + source.ydl_errors: the inner extract_info
    # raises a genuine yt_dlp DownloadError, which must surface as a RuntimeError the
    # CLI can funnel to friendly_ydl_error (not a raw traceback).
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYDL)
    with pytest.raises(RuntimeError, match="403") as ei:
        captions.fetch_captions("https://x.com/v")
    from yt_dlp.utils import DownloadError

    assert not isinstance(ei.value, DownloadError)
