"""Transcript markdown — provenance generalized over kind and source (review M3)."""

from __future__ import annotations

import pytest

from glean import markdown


def test_header_and_stamps() -> None:
    md = markdown.to_markdown([(14, "check gold"), (74, "buy")], source="http://x")
    assert "**Source:** <http://x>" in md
    assert "Read with care" in md
    assert "[00:14] check gold" in md
    assert "[01:14] buy" in md  # 74s -> 01:14


def test_auto_caption_provenance_by_default() -> None:
    md = markdown.to_markdown([(0, "hi")], source="http://x")
    assert "auto-generated" in md
    assert "author-provided" not in md
    assert "locally transcribed" not in md


def test_manual_caption_provenance() -> None:
    md = markdown.to_markdown([(0, "hi")], source="http://x", caption_kind="manual")
    assert "author-provided" in md
    assert "auto-generated" not in md
    assert "locally transcribed" not in md


def test_asr_provenance_when_model_given() -> None:
    md = markdown.to_markdown([(0, "hi")], source="http://x", asr_model="medium.en")
    assert "locally transcribed" in md
    assert "`medium.en`" in md
    assert "auto-generated" not in md
    assert "[00:00] hi" in md  # cues still render identically


def test_url_source_rendered_with_angle_brackets() -> None:
    md = markdown.to_markdown([(0, "hi")], source="https://udemy.com/x", is_url=True)
    assert "<https://udemy.com/x>" in md


def test_local_path_source_rendered_as_code_span() -> None:
    # A local path is NOT a URL — no angle brackets, rendered as an inline code span
    # (review M3: no hardcoded "YouTube", provenance works for any source).
    md = markdown.to_markdown([(0, "hi")], source="/home/me/lecture.mp4", is_url=False)
    assert "`/home/me/lecture.mp4`" in md
    assert "</home/me/lecture.mp4>" not in md


def test_empty_cues_refuse_to_render() -> None:
    # Zero cues used to render a CONFIDENT empty file — a header asserting provenance,
    # "~0h0m", no content — and the CLI reported "wrote transcript.md (8 lines)".
    # A silently wrong answer is worse than a loud failure.
    with pytest.raises(markdown.EmptyTranscriptError, match="no transcript content"):
        markdown.to_markdown([], source="http://x")


def test_empty_cues_error_points_at_asr_for_the_caption_path() -> None:
    with pytest.raises(markdown.EmptyTranscriptError, match=r"--asr") as ei:
        markdown.to_markdown([], source="http://x", caption_kind="manual")
    assert "caption track parsed to zero cues" in str(ei.value)


def test_empty_cues_error_is_asr_shaped_when_asr_was_used() -> None:
    # The ASR path has no caption track to blame and no --asr to suggest.
    with pytest.raises(markdown.EmptyTranscriptError) as ei:
        markdown.to_markdown([], source="/lecture.mp4", is_url=False, asr_model="medium.en")
    assert "no speech" in str(ei.value)
    assert "--asr" not in str(ei.value)


def test_empty_transcript_error_is_a_runtimeerror() -> None:
    # The CLI funnels on RuntimeError; a bare Exception would escape as a traceback.
    assert issubclass(markdown.EmptyTranscriptError, RuntimeError)
