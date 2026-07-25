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


def test_crlf_vtt_keeps_per_cue_timestamps() -> None:
    # Blocks split on "\n\n" but lines are read with splitlines(): a CRLF file never
    # split, collapsing into ONE block so every line inherited the LAST cue's time
    # (both lines came back stamped 300). Wrong timestamps are worse than no
    # transcript — locating moments for `glean frames` is the whole point.
    crlf = (
        "WEBVTT\r\n\r\n"
        "00:00:01.000 --> 00:00:02.000\r\nfirst\r\n\r\n"
        "00:05:00.000 --> 00:05:02.000\r\nsecond\r\n"
    )
    assert captions.clean_vtt(crlf) == [(1, "first"), (300, "second")]
    # ...and identically to the same file with LF endings.
    assert captions.clean_vtt(crlf) == captions.clean_vtt(crlf.replace("\r\n", "\n"))


def test_lone_cr_vtt_also_splits() -> None:
    cr = "WEBVTT\r\r00:00:01.000 --> 00:00:02.000\rfirst\r\r00:00:09.000 --> 00:00:10.000\rsecond\r"
    assert captions.clean_vtt(cr) == [(1, "first"), (9, "second")]


def test_short_form_timestamps_are_parsed() -> None:
    # WebVTT allows the hour field to be omitted. Requiring HH made every such cue
    # unmatchable, so the file parsed to ZERO cues with no error at all.
    short = "WEBVTT\n\n01:30.500 --> 01:32.000\nhello\n\n02:00.000 --> 02:02.000\nthere\n"
    assert captions.clean_vtt(short) == [(90, "hello"), (120, "there")]


def test_long_form_timestamps_still_parsed() -> None:
    # The optional-hour regex must not regress the ordinary HH:MM:SS.mmm form.
    long_form = "WEBVTT\n\n01:02:03.000 --> 01:02:05.000\nhi\n"
    assert captions.clean_vtt(long_form) == [(3723, "hi")]


# ---- caption track selection (review S1) ------------------------------------


def test_choose_prefers_manual_over_auto() -> None:
    # Both tracks present for the language — the author's own subtitles win.
    info = {
        "subtitles": {"en": [{"ext": "vtt"}]},
        "automatic_captions": {"en": [{"ext": "vtt"}]},
    }
    assert captions.choose_caption_track(info, "en") == ("manual", "en")


def test_choose_auto_when_only_auto() -> None:
    info = {"subtitles": {}, "automatic_captions": {"en": [{"ext": "vtt"}]}}
    assert captions.choose_caption_track(info, "en") == ("auto", "en")


def test_choose_manual_when_only_manual() -> None:
    info = {"subtitles": {"en": [{"ext": "vtt"}]}, "automatic_captions": {}}
    assert captions.choose_caption_track(info, "en") == ("manual", "en")


def test_choose_none_when_neither() -> None:
    # No track in the requested language -> None -> caller falls to ASR.
    info = {"subtitles": {"de": [{"ext": "vtt"}]}, "automatic_captions": {"fr": [{"ext": "vtt"}]}}
    assert captions.choose_caption_track(info, "en") is None


def test_choose_none_on_empty_info() -> None:
    assert captions.choose_caption_track({}, "en") is None


@pytest.mark.parametrize("tag", ["en-US", "en-GB", "en-orig"])
def test_regional_variant_is_found_and_keeps_its_own_tag(tag: str) -> None:
    # yt-dlp keys tracks by the uploader's BCP-47 tag. Exact-matching "en" reported
    # "no captions" for these and burned a full local ASR run on a video that HAS
    # author subtitles — the exact waste this module exists to prevent.
    info = {"subtitles": {tag: [{"ext": "vtt"}]}, "automatic_captions": {}}
    track = captions.choose_caption_track(info, "en")
    assert track is not None
    assert track.kind == "manual"
    # The real key must survive: asking yt-dlp for "en" when the track is filed under
    # "en-US" downloads nothing at all.
    assert track.lang == tag


def test_exact_tag_beats_regional_variant() -> None:
    info = {
        "subtitles": {"en-GB": [{"ext": "vtt"}], "en": [{"ext": "vtt"}]},
        "automatic_captions": {},
    }
    assert captions.choose_caption_track(info, "en").lang == "en"


def test_variant_choice_is_deterministic() -> None:
    info = {"subtitles": {"en-US": [{"ext": "vtt"}], "en-GB": [{"ext": "vtt"}]}}
    assert captions.choose_caption_track(info, "en").lang == "en-GB"  # sorted


def test_manual_variant_beats_exact_auto() -> None:
    # The manual-over-auto preference must still hold across the variant match.
    info = {
        "subtitles": {"en-US": [{"ext": "vtt"}]},
        "automatic_captions": {"en": [{"ext": "vtt"}]},
    }
    assert captions.choose_caption_track(info, "en") == ("manual", "en-US")


def test_different_language_sharing_a_prefix_does_not_match() -> None:
    # Sub-tags are "-"-delimited: "eng" is a different tag, not a variant of "en".
    info = {"subtitles": {"eng": [{"ext": "vtt"}]}, "automatic_captions": {}}
    assert captions.choose_caption_track(info, "en") is None


def test_empty_track_list_is_not_a_match() -> None:
    # A language key mapping to no formats is not a usable track.
    info = {"subtitles": {"en": []}, "automatic_captions": {"en": [{"ext": "vtt"}]}}
    assert captions.choose_caption_track(info, "en") == ("auto", "en")


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
