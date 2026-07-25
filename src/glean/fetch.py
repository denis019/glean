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
) -> Path:
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
    if produced is None:
        raise RuntimeError(f"fetch produced no file for {url}")
    if produced.exists():
        return produced
    # An HLS/merge remux can change the extension from what prepare_filename guessed;
    # fall back to the newest sibling sharing its stem.
    #
    # Match on `produced`, NOT on `stem`: for a target DIRECTORY — which is the default
    # (`-o` omitted) and the whole Udemy workflow — `stem` still holds the unexpanded
    # "%(title)s" template, and globbing that searches for a file *literally* named
    # "%(title)s.*". It can never match, so the fallback was dead precisely where it
    # was needed. prepare_filename has already expanded the outtmpl, so its stem is the
    # real one. Comparing stems also sidesteps glob metacharacters in video titles
    # (a "Lecture [Part 1]" would have been read as a character class).
    if not produced.parent.is_dir():
        raise RuntimeError(f"fetch produced no file for {url}")
    siblings = [p for p in produced.parent.iterdir() if p.is_file() and p.stem == produced.stem]
    if siblings:
        return max(siblings, key=lambda p: p.stat().st_mtime)
    raise RuntimeError(f"fetch produced no file for {url}")
