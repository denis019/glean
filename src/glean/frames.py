"""Extract still frames from a video at transcript timestamps (throwaway study aids).

The transcript tells you WHEN something is explained on screen (`[25:19]`); this grabs
that frame so it can be looked at. Two input kinds:

* **URL**: yt-dlp resolves the direct video URL once — through
  `source.build_ydl_opts` so cookies + deno reach this call site too — then ffmpeg
  input-seeks (`-ss` before `-i`) to each timestamp.
* **Local file**: no yt-dlp resolve and no ``--format``; ffmpeg seeks the file on
  disk directly.

The pure timestamp/window/naming helpers are shared by both.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from glean import source

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# Best VIDEO-ONLY stream, H.264 preferred. Frames need no audio, so a video-only DASH
# format (itag 299 = 1080p60 avc1) is both highest quality AND smaller to seek than the
# combined stream. avc1 (H.264) first for fast ffmpeg decode; then any mp4 video (av1);
# then a combined fallback. Resolves WITHOUT a JS runtime. Override with --format.
DEFAULT_FORMAT = "bestvideo[vcodec^=avc1]/bestvideo[ext=mp4]/best"


def parse_ts(text: str) -> int:
    """`SS`, `MM:SS`, or `HH:MM:SS` -> total seconds. MM may exceed 59 (the transcript
    stamps a 75-minute video as `[75:00]`), so parts are NOT range-checked, only summed.
    """
    parts = text.strip().split(":")
    if not parts or len(parts) > 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"bad timestamp {text!r} — want SS, MM:SS, or HH:MM:SS")
    nums = [0] * (3 - len(parts)) + [int(p) for p in parts]
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def expand_window(text: str, every: int) -> list[int]:
    """`A-B` (each side a timestamp) + step `every` seconds -> [A, A+every, …, <=B]."""
    if every <= 0:
        raise ValueError("--every must be positive")
    lo_s, _, hi_s = text.partition("-")
    if not hi_s:
        raise ValueError(f"bad window {text!r} — want A-B, e.g. 25:00-26:00")
    lo, hi = parse_ts(lo_s), parse_ts(hi_s)
    if hi < lo:
        raise ValueError(f"window end {hi_s} is before start {lo_s}")
    return list(range(lo, hi + 1, every))


def collect_seconds(at: Sequence[str], window: Sequence[str], every: int) -> list[int]:
    """Every second named by the `--at` stamps and the `A-B` windows, in argument order.

    The timestamp grammar is a user-facing contract, so both front-ends (the CLI and
    the MCP server) resolve it HERE rather than each re-walking `parse_ts` /
    `expand_window`. Raises `ValueError` with an already-actionable message on a typo;
    an empty result means the caller named no timestamps at all.
    """
    seconds = [parse_ts(a) for a in at]
    for w in window:
        seconds.extend(expand_window(w, every))
    return seconds


def frame_name(seconds: int, label: str | None = None) -> str:
    """Filename for a frame, matching the transcript's MM:SS style: `frame-25m19s.jpg`."""
    stem = f"{seconds // 60:02d}m{seconds % 60:02d}s"
    return f"{f'{label}-' if label else 'frame-'}{stem}.jpg"


def _resolve_stream_url(
    video_url: str,
    fmt: str,
    *,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> str:  # pragma: no cover — network
    import yt_dlp  # noqa: PLC0415 — lazy: heavy import, network-only path

    call = source.build_ydl_opts(
        {
            "format": fmt,
            # Fetch exactly the lecture the URL names (Udemy expands it to the course).
            **source.playlist_selection(
                video_url, cookies_from_browser=cookies_from_browser, cookies_file=cookies_file
            ),
        },
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    with source.ydl_errors(), source.patched_path(call.env), yt_dlp.YoutubeDL(call.opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    url = info.get("url") if info else None
    if not url and info and info.get("requested_formats"):
        url = info["requested_formats"][0].get("url")
    if not url:
        raise RuntimeError(f"could not resolve a direct stream URL for {video_url} (fmt={fmt})")
    return url


def _grab(target: str, seconds: int, out_path: Path) -> None:  # pragma: no cover — ffmpeg
    # `-ss` before `-i` = fast input seek (grabs one frame without reading the whole
    # file). `target` is a local path OR, for the URL path, a time-limited CDN link that
    # occasionally throttles; surface ffmpeg's own stderr so a transient 403/429 (URL
    # only) is diagnosable (a re-run usually succeeds) rather than a bare non-zero exit.
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-ss",
            str(seconds),
            "-i",
            target,
            "-frames:v",
            "1",
            "-q:v",
            "3",
            "-y",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
        raise RuntimeError(f"ffmpeg failed at {seconds}s (often transient — retry):\n{tail}")


def extract_frames(
    video: str,
    seconds: list[int],
    out_dir: Path,
    *,
    is_url: bool = True,
    fmt: str = DEFAULT_FORMAT,
    label: str | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> list[Path]:
    """Grab a frame per timestamp into `out_dir`. Requires `ffmpeg` on PATH.

    URL input resolves the stream URL ONCE via yt-dlp, then ffmpeg-seeks it. Local
    input (review S3) skips the resolve + ``--format`` entirely and ffmpeg-seeks the
    file directly. Returns the written paths, in timestamp order.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — install it to extract frames")
    if not seconds:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    target = (
        _resolve_stream_url(
            video, fmt, cookies_from_browser=cookies_from_browser, cookies_file=cookies_file
        )
        if is_url
        else video
    )
    written: list[Path] = []
    for sec in sorted(set(seconds)):
        path = out_dir / frame_name(sec, label)
        _grab(target, sec, path)
        written.append(path)
    return written
