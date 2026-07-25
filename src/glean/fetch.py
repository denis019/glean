"""Download and KEEP the full media locally — `glean fetch`.

`transcribe` and `frames` each re-hit the source — and the lecture-isolation
flat-extract enumerates the WHOLE course — so doing both against a paid,
rate-limited source (Udemy) trips its rate limiter. `glean fetch` downloads the
full lecture ONCE and keeps it; `glean transcribe <file>` and `glean frames
<file>` then run locally against the file with zero further requests.
"""

from __future__ import annotations

from pathlib import Path

from glean import source


def download_media(
    url: str,
    out: Path,
    *,
    fmt: str = "best",
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> Path:  # pragma: no cover — network
    """Download the full media for `url` and KEEP it. Returns the written path.

    `out` is a target **directory** (→ ``<title>.<ext>``) or a **file path** (its
    extension is replaced by the real container's). Reuses glean's cookie +
    lecture-isolation handling, so a Udemy lecture URL fetches exactly that lecture.
    """
    import yt_dlp  # noqa: PLC0415 — lazy: heavy import, network-only path

    stem = out / "%(title)s" if out.is_dir() else out.with_suffix("")
    call = source.build_ydl_opts(
        {
            "format": fmt,
            "outtmpl": {"default": f"{stem}.%(ext)s"},
            **source.playlist_selection(
                url, cookies_from_browser=cookies_from_browser, cookies_file=cookies_file
            ),
        },
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    with source.ydl_errors(), source.patched_path(call.env), yt_dlp.YoutubeDL(call.opts) as ydl:
        info = ydl.extract_info(url)
        entry = info["entries"][0] if info and info.get("entries") else info
        produced = Path(ydl.prepare_filename(entry)) if entry else None
    if produced and produced.exists():
        return produced
    # An HLS/merge remux can change the extension from what prepare_filename guessed;
    # fall back to the newest file matching the stem.
    parent = out if out.is_dir() else out.parent
    matches = sorted(parent.glob(f"{Path(stem).name}.*"), key=lambda p: p.stat().st_mtime)
    if matches:
        return matches[-1]
    raise RuntimeError(f"fetch produced no file for {url}")
