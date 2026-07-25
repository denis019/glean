"""Frame extraction — pure ts/window/naming + the NEW local-file dispatch (review S3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glean import frames

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("19", 19),
        ("25:19", 1519),
        ("1:05:03", 3903),
        ("75:00", 4500),  # transcript stamps a 75-min video as [75:00] — MM may exceed 59
        (" 25:19 ", 1519),
    ],
)
def test_parse_ts(text: str, seconds: int) -> None:
    assert frames.parse_ts(text) == seconds


@pytest.mark.parametrize("bad", ["", "aa", "1:2:3:4", "25:xx", "1.5"])
def test_parse_ts_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError, match="bad timestamp"):
        frames.parse_ts(bad)


def test_expand_window() -> None:
    assert frames.expand_window("25:00-25:30", 10) == [1500, 1510, 1520, 1530]


def test_expand_window_inclusive_end() -> None:
    assert frames.expand_window("0:00-0:20", 20) == [0, 20]


def test_expand_window_bad_inputs() -> None:
    with pytest.raises(ValueError, match="want A-B"):
        frames.expand_window("2500", 10)
    with pytest.raises(ValueError, match="before start"):
        frames.expand_window("26:00-25:00", 10)
    with pytest.raises(ValueError, match="positive"):
        frames.expand_window("0:00-0:20", 0)


@pytest.mark.parametrize(
    ("seconds", "label", "expected"),
    [
        (1519, None, "frame-25m19s.jpg"),
        (1519, "gold-entry", "gold-entry-25m19s.jpg"),
        (0, None, "frame-00m00s.jpg"),
        (4500, None, "frame-75m00s.jpg"),  # matches the transcript's [75:00]
    ],
)
def test_frame_name(seconds: int, label: str | None, expected: str) -> None:
    assert frames.frame_name(seconds, label) == expected


def test_extract_frames_empty_is_noop(tmp_path: Path) -> None:
    # No timestamps -> no ffmpeg, no network, no dir created.
    assert frames.extract_frames("http://x", [], tmp_path / "frames") == []


def test_extract_frames_without_ffmpeg_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(frames.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        frames.extract_frames("http://x", [10], tmp_path)


# ---- local-file dispatch (review S3 — NEW code) ------------------------------


def test_local_file_skips_yt_dlp_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A local path must NOT touch yt-dlp: no _resolve_stream_url, and _grab is handed
    # the file path directly (not a resolved CDN URL).
    monkeypatch.setattr(frames.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def _resolve_boom(*_a: object, **_k: object) -> str:
        pytest.fail("local input must not resolve via yt-dlp")

    grabbed: list[tuple[str, int]] = []

    def _fake_grab(target: str, seconds: int, out_path: Path) -> None:
        grabbed.append((target, seconds))
        out_path.write_bytes(b"jpg")

    monkeypatch.setattr(frames, "_resolve_stream_url", _resolve_boom)
    monkeypatch.setattr(frames, "_grab", _fake_grab)

    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"\x00")
    out = frames.extract_frames(str(video), [10, 20], tmp_path / "frames", is_url=False)

    assert [t for t, _ in grabbed] == [str(video), str(video)]  # the file itself, seeked
    assert [s for _, s in grabbed] == [10, 20]
    assert [p.name for p in out] == ["frame-00m10s.jpg", "frame-00m20s.jpg"]


def test_url_dispatch_resolves_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # URL input resolves the stream URL exactly ONCE, then seeks that for every frame.
    monkeypatch.setattr(frames.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    calls = {"resolve": 0}

    def _fake_resolve(*_a: object, **_k: object) -> str:
        calls["resolve"] += 1
        return "https://cdn/stream.mp4"

    grabbed: list[str] = []
    monkeypatch.setattr(frames, "_resolve_stream_url", _fake_resolve)
    monkeypatch.setattr(
        frames, "_grab", lambda target, _s, out: grabbed.append(target) or out.write_bytes(b"j")
    )

    frames.extract_frames("http://x", [10, 20, 30], tmp_path / "f", is_url=True)
    assert calls["resolve"] == 1
    assert grabbed == ["https://cdn/stream.mp4"] * 3
