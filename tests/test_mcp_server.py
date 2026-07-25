"""MCP front-end — the library seams stubbed, so this is pure and fast.

The CLI's behaviour is already covered in `test_cli.py`; what is tested here is what
the MCP front-end adds on top: tool registration, the `SystemExit`→`ToolError` funnel
(a leak would take the server down, not fail a call), absolute paths in the results,
and the stdout the JSON-RPC stream depends on.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from glean import captions, fetch, frames, mcp_server


def _call(tool: Any, **kwargs: Any) -> Any:
    """Invoke an offloaded tool from a sync test."""
    return anyio.run(functools.partial(tool, **kwargs))


def _local_file(tmp_path: Path) -> str:
    f = tmp_path / "lecture.mp4"
    f.write_bytes(b"\x00")
    return str(f)


# ---- registration ------------------------------------------------------------


def test_tools_are_registered() -> None:
    tools = {t.name: t for t in anyio.run(mcp_server.server.list_tools)}
    assert set(tools) == {"transcribe", "frames", "fetch"}
    assert "video" in tools["transcribe"].inputSchema["properties"]
    assert tools["transcribe"].inputSchema["required"] == ["video"]
    # The docstring is what the agent reads to choose the tool.
    assert "faster-whisper" in (tools["transcribe"].description or "")


def test_device_is_an_enum_in_the_schema() -> None:
    (t,) = [t for t in anyio.run(mcp_server.server.list_tools) if t.name == "transcribe"]
    assert t.inputSchema["properties"]["device"]["enum"] == ["auto", "cuda", "cpu"]


# ---- transcribe ---------------------------------------------------------------


def _manual_captions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        captions,
        "fetch_captions",
        lambda url, lang="en", **kw: ("WEBVTT\n\n00:00:14.000 --> 00:00:16.000\nhi\n", "manual"),
    )


def test_transcribe_returns_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    _manual_captions(monkeypatch)
    assert "[00:14] hi" in _call(mcp_server.transcribe, video="https://x.com/v")


def test_transcribe_out_writes_a_file_and_returns_a_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The point of `out`: keep a full lecture transcript out of the agent's context.
    _manual_captions(monkeypatch)
    out = tmp_path / "sub" / "t.md"
    receipt = _call(mcp_server.transcribe, video="https://x.com/v", out=str(out))
    assert "[00:14] hi" in out.read_text(encoding="utf-8")
    assert "[00:14] hi" not in receipt
    assert str(out.resolve()) in receipt


def test_transcribe_options_reach_the_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _capture(input_str: str, **kw: object) -> str:
        seen.update(input=input_str, **kw)
        return "# ok\n"

    monkeypatch.setattr(mcp_server, "_transcribe", _capture)
    _call(
        mcp_server.transcribe,
        video="https://udemy.com/v",
        asr=True,
        model="small.en",
        cookies_from_browser="firefox",
    )
    assert seen["asr"] is True
    assert seen["model"] == "small.en"
    assert seen["cookies_from_browser"] == "firefox"
    assert seen["cookies_file"] is None


# ---- the error funnel ---------------------------------------------------------


def test_missing_file_is_a_tool_error_not_systemexit() -> None:
    # `source.classify` rejects a bad input with SystemExit, which descends from
    # BaseException — FastMCP does NOT catch it, so a leak here would take the whole
    # server down mid-session instead of failing one call.
    with pytest.raises(ToolError, match="no such file"):
        _call(mcp_server.transcribe, video="not-a-file.mp4")


def test_malformed_url_is_a_tool_error() -> None:
    with pytest.raises(ToolError, match="missing host"):
        _call(mcp_server.frames, video="https://", at=["1:00"])


def test_runtime_failure_gets_the_friendly_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> tuple[str, str]:
        raise RuntimeError("This video is DRM protected")

    monkeypatch.setattr(captions, "fetch_captions", _boom)
    with pytest.raises(ToolError, match="DRM-protected"):
        _call(mcp_server.transcribe, video="https://x.com/v")


def test_empty_transcript_is_not_blamed_on_yt_dlp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        captions, "fetch_captions", lambda url, lang="en", **kw: ("WEBVTT\n\n", "auto")
    )
    with pytest.raises(ToolError) as ei:
        _call(mcp_server.transcribe, video="https://x.com/v")
    assert "yt-dlp failed" not in str(ei.value)
    assert "no transcript content" in str(ei.value)


def test_bad_timestamp_is_a_tool_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="bad timestamp"):
        _call(mcp_server.frames, video=_local_file(tmp_path), at=["1:2:3:4"])


# ---- frames -------------------------------------------------------------------


def _capture_frames(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake(video, seconds, out_dir, *, is_url=True, fmt="", label=None, **kw):
        seen.update(video=video, seconds=seconds, out_dir=out_dir, is_url=is_url, **kw)
        return [out_dir / frames.frame_name(s, label) for s in sorted(set(seconds))]

    monkeypatch.setattr(frames, "extract_frames", fake)
    return seen


def test_frames_expands_a_window_and_returns_absolute_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _capture_frames(monkeypatch)
    out = _call(
        mcp_server.frames,
        video="https://x.com/v",
        at=["1:00"],
        window=["0:00-0:20"],
        every=10,
        out=str(tmp_path / "f"),
    )
    assert seen["seconds"] == [60, 0, 10, 20]
    assert seen["is_url"] is True
    # Absolute: the caller does not share glean's cwd, so a relative path is unusable.
    assert out == [
        str(tmp_path / "f" / n)
        for n in ("frame-00m00s.jpg", "frame-00m10s.jpg", "frame-00m20s.jpg", "frame-01m00s.jpg")
    ]


def test_frames_local_file_skips_yt_dlp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _capture_frames(monkeypatch)
    video = _local_file(tmp_path)
    _call(mcp_server.frames, video=video, at=["25:19"])
    assert seen["video"] == video
    assert seen["is_url"] is False


def test_frames_without_timestamps_is_a_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_frames(monkeypatch)
    with pytest.raises(ToolError, match="nothing to grab"):
        _call(mcp_server.frames, video="https://x.com/v")


def test_frames_default_out_dir_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_frames(monkeypatch)
    _call(mcp_server.frames, video="https://x.com/v", at=["1:00"])
    assert seen["out_dir"] == Path("frames").resolve()


# ---- fetch --------------------------------------------------------------------


def test_fetch_returns_the_kept_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    kept = tmp_path / "lecture11.mp4"
    seen: dict = {}

    def fake(url: str, out: Path, **kw: object) -> Path:
        seen.update(url=url, out=out, **kw)
        return kept

    monkeypatch.setattr(fetch, "download_media", fake)
    got = _call(
        mcp_server.fetch,
        url="https://www.udemy.com/course/x/learn/lecture/1",
        out=str(tmp_path),
        cookies_from_browser="chrome",
    )
    assert got == str(kept)
    assert seen["out"] == tmp_path.resolve()
    assert seen["cookies_from_browser"] == "chrome"


def test_fetch_rejects_a_local_file(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="already a local file"):
        _call(mcp_server.fetch, url=_local_file(tmp_path))


# ---- the stdio contract -------------------------------------------------------


def test_lifespan_hands_stdout_to_the_protocol() -> None:
    # stdout IS the JSON-RPC stream: for the length of a session `print` must land on
    # stderr, and the real stdout must come back afterwards.
    async def exercise() -> None:
        assert sys.stdout is not sys.stderr
        async with mcp_server._own_stdout(mcp_server.server):
            assert sys.stdout is sys.stderr
        assert sys.stdout is not sys.stderr

    anyio.run(exercise)


def test_main_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[bool] = []
    monkeypatch.setattr(mcp_server.server, "run", lambda: ran.append(True))
    mcp_server.main()
    assert ran == [True]
