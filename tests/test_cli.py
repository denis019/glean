"""CLI dispatch — every network/GPU seam stubbed, so this is pure and fast."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import pytest

from glean import asr, captions, cli, fetch, frames, markdown


def _local_file(tmp_path: Path) -> str:
    f = tmp_path / "lecture.mp4"
    f.write_bytes(b"\x00")
    return str(f)


# ---- fetch: download + keep, then work locally -------------------------------


def test_fetch_dispatches_and_prints_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    kept = tmp_path / "lecture11.mp4"
    seen: dict = {}

    def fake(url: str, out: Path, **kw: object) -> Path:
        seen.update(url=url, out=out, **kw)
        return kept

    monkeypatch.setattr(fetch, "download_media", fake)
    rc = cli.main(["fetch", "https://www.udemy.com/course/x/learn/lecture/1", "-o", str(tmp_path)])
    assert rc == 0
    assert str(kept) in capsys.readouterr().out
    assert seen["url"].endswith("/lecture/1")
    assert seen["cookies_from_browser"] is None


def test_fetch_local_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="already a local file"):
        cli.main(["fetch", _local_file(tmp_path)])


# ---- transcribe: caption manual → auto → ASR fallback (review S1) ------------


def test_transcribe_uses_manual_caption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        captions,
        "fetch_captions",
        lambda url, lang="en", **kw: ("WEBVTT\n\n00:00:14.000 --> 00:00:16.000\nhi\n", "manual"),
    )
    monkeypatch.setattr(asr, "export_url", lambda *a, **k: pytest.fail("must not fall to ASR"))
    rc = cli.main(["transcribe", "https://x.com/v"])
    out = capsys.readouterr()
    assert rc == 0
    assert "author-provided" in out.out  # manual provenance
    assert "[00:14] hi" in out.out
    assert "using manual caption track" in out.err


def test_transcribe_uses_auto_caption(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        captions,
        "fetch_captions",
        lambda url, lang="en", **kw: ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nyo\n", "auto"),
    )
    cli.main(["transcribe", "https://x.com/v"])
    out = capsys.readouterr()
    assert "auto-generated" in out.out
    assert "using auto caption track" in out.err


def test_transcribe_falls_back_to_asr_when_no_captions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _none(url: str, lang: str = "en", **_kw: object) -> tuple[str, str]:
        raise captions.NoCaptionsError(f"no caption track for {url}")

    monkeypatch.setattr(captions, "fetch_captions", _none)
    monkeypatch.setattr(asr, "export_url", lambda *a, **k: "# fell back\n")
    cli.main(["transcribe", "https://x.com/v"])
    out = capsys.readouterr()
    assert "# fell back" in out.out
    assert "falling back to local ASR" in out.err


def test_transcribe_asr_flag_forces_local(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        captions, "fetch_captions", lambda *a, **k: pytest.fail("caption path must be skipped")
    )
    monkeypatch.setattr(asr, "export_url", lambda url, **k: f"# ASR {url} {k['model']}\n")
    cli.main(["transcribe", "https://x.com/v", "--asr", "--model", "small.en"])
    assert "# ASR https://x.com/v small.en" in capsys.readouterr().out


def test_transcribe_local_file_goes_to_asr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        captions, "fetch_captions", lambda *a, **k: pytest.fail("no captions for a local file")
    )
    monkeypatch.setattr(asr, "export_file", lambda path, **k: f"# local {path.name}\n")
    cli.main(["transcribe", _local_file(tmp_path)])
    assert "# local lecture.mp4" in capsys.readouterr().out


def test_transcribe_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        captions,
        "fetch_captions",
        lambda url, lang="en", **kw: ("WEBVTT\n\n00:00:14.000 --> 00:00:16.000\nhi\n", "auto"),
    )
    out = tmp_path / "sub" / "t.md"
    rc = cli.main(["transcribe", "https://x.com/v", "-o", str(out)])
    assert rc == 0
    assert "[00:14] hi" in out.read_text(encoding="utf-8")


def test_transcribe_friendly_error_on_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> tuple[str, str]:
        raise RuntimeError("This video is DRM protected")

    monkeypatch.setattr(captions, "fetch_captions", _boom)
    with pytest.raises(SystemExit, match="DRM"):
        cli.main(["transcribe", "https://x.com/v"])


class _DrmYDL:
    """yt_dlp.YoutubeDL stand-in whose extract_info raises a REAL DownloadError."""

    def __init__(self, opts: object) -> None:
        self._opts = opts

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def extract_info(self, url: str, *, download: bool = False) -> dict:  # noqa: ARG002
        from yt_dlp.utils import DownloadError

        raise DownloadError("This video is DRM protected")


def test_transcribe_friendly_error_on_real_download_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end (review finding 1): a REAL yt_dlp DownloadError from the caption call
    # site must reach friendly_ydl_error, not escape as a traceback. Exercises the true
    # chain — fetch_captions + ydl_errors + the CLI funnel — with only the YoutubeDL
    # network object itself stubbed.
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _DrmYDL)
    with pytest.raises(SystemExit, match="DRM"):
        cli.main(["transcribe", "https://x.com/v"])


def test_transcribe_local_file_error_is_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A local-file failure is NOT a yt-dlp failure (review finding 2): surface it
    # verbatim, never prefixed "yt-dlp failed".
    def _no_extra(*_a: object, **_k: object) -> str:
        raise RuntimeError("ASR needs the `asr` extra — install with `uv sync --extra asr`")

    monkeypatch.setattr(asr, "export_file", _no_extra)
    with pytest.raises(SystemExit) as ei:
        cli.main(["transcribe", _local_file(tmp_path)])
    assert "yt-dlp" not in str(ei.value)
    assert "asr` extra" in str(ei.value)


def test_frames_local_file_error_is_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same for frames: a local ffmpeg failure must not be blamed on yt-dlp.
    def _boom(*_a: object, **_k: object) -> list[Path]:
        raise RuntimeError("ffmpeg failed at 60s (often transient — retry)")

    monkeypatch.setattr(frames, "extract_frames", _boom)
    with pytest.raises(SystemExit) as ei:
        cli.main(["frames", _local_file(tmp_path), "--at", "1:00"])
    assert "yt-dlp" not in str(ei.value)
    assert "ffmpeg failed" in str(ei.value)


def test_transcribe_cookies_reach_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _capture(url: str, lang: str = "en", **kw: object) -> tuple[str, str]:
        seen.update(kw)
        return ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n", "manual")

    monkeypatch.setattr(captions, "fetch_captions", _capture)
    cli.main(["transcribe", "https://udemy.com/v", "--cookies-from-browser", "chrome"])
    assert seen["cookies_from_browser"] == "chrome"
    assert seen["cookies_file"] is None


# ---- frames: URL vs local dispatch ------------------------------------------


def _capture_frames(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake(video, seconds, out_dir, *, is_url=True, fmt=frames.DEFAULT_FORMAT, label=None, **kw):
        seen.update(video=video, seconds=seconds, out_dir=out_dir, is_url=is_url, **kw)
        return [out_dir / frames.frame_name(s, label) for s in sorted(set(seconds))]

    monkeypatch.setattr(frames, "extract_frames", fake)
    return seen


def test_frames_url_and_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_frames(monkeypatch)
    cli.main(
        [
            "frames",
            "https://x.com/v",
            "--window",
            "0:00-0:20",
            "--every",
            "10",
            "-o",
            str(tmp_path / "f"),
        ]
    )
    assert seen["video"] == "https://x.com/v"
    assert seen["is_url"] is True
    assert seen["out_dir"] == tmp_path / "f"
    assert seen["seconds"] == [0, 10, 20]


def test_frames_local_file_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_frames(monkeypatch)
    video = _local_file(tmp_path)
    cli.main(["frames", video, "--at", "25:19", "--at", "29:35"])
    assert seen["video"] == video
    assert seen["is_url"] is False  # local -> no yt-dlp resolve
    assert seen["seconds"] == [1519, 1775]
    assert seen["out_dir"] == Path("frames")


def test_frames_default_out_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_frames(monkeypatch)
    cli.main(["frames", "https://x.com/v", "--at", "1:00"])
    assert seen["out_dir"] == Path("frames")


def test_frames_no_timestamps_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_frames(monkeypatch)
    with pytest.raises(SystemExit, match="nothing to grab"):
        cli.main(["frames", "https://x.com/v"])


def test_frames_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> list[Path]:
        raise RuntimeError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(frames, "extract_frames", _boom)
    with pytest.raises(SystemExit, match="403"):
        cli.main(["frames", "https://udemy.com/v", "--at", "1:00"])


def test_frames_cookies_reach_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_frames(monkeypatch)
    cli.main(["frames", "https://udemy.com/v", "--at", "1:00", "--cookies", "cookies.txt"])
    assert seen["cookies_file"] == "cookies.txt"
    assert seen["cookies_from_browser"] is None


# ---- misc --------------------------------------------------------------------


def test_no_subcommand_exits() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_markdown_module_importable() -> None:
    # Guard the public seam the CLI renders through.
    assert markdown.to_markdown([(0, "x")], source="http://y")
