"""Transcript markdown — provenance generalized over kind and source (review M3)."""

from __future__ import annotations

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


def test_empty_cues_span_is_zero() -> None:
    md = markdown.to_markdown([], source="http://x")
    assert "~0h0m" in md
