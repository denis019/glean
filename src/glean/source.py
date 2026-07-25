"""Input resolution + the ONE yt-dlp options builder (cookies + deno) for glean.

Two concerns live here, both kept pure so the rest of the tool is testable without
the network:

* **Local-file vs URL dispatch.** `classify` decides whether ``<input>`` is a local
  media file or a yt-dlp-supported URL. yt-dlp itself dispatches URLs by site, so
  YouTube and Udemy are *not* branched on here — only file-vs-URL is.
* **`build_ydl_opts` — the single builder every yt-dlp call site goes through**
  (review S2/N3). There are THREE sites — caption fetch (`captions.py`), audio
  download (`asr.py`) and frame-stream resolve (`frames.py`) — and each must stamp
  the same two things: the Udemy auth **cookies** (``cookiesfrombrowser`` /
  ``cookiefile``) and the **deno PATH shim** (yt-dlp needs a JS runtime or YouTube
  serves a throttled client that 403s). Routing all three through one pure builder
  is the only way to guarantee none is missed. It returns a `YdlCall` = the opts
  dict **plus** the PATH-augmented env; call sites apply the env with
  `patched_path` around the actual (no-cover) `YoutubeDL` call.
"""

from __future__ import annotations

import os
import re
import shutil
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class InputKind(Enum):
    """How `classify` read the ``<input>`` argument."""

    URL = "url"
    FILE = "file"


class Source(NamedTuple):
    """A resolved input: its kind and the value to hand to yt-dlp / ffmpeg."""

    kind: InputKind
    value: str

    @property
    def is_url(self) -> bool:
        return self.kind is InputKind.URL


class YdlCall(NamedTuple):
    """A ready-to-use yt-dlp invocation: the opts dict and the PATH-patched env."""

    opts: dict[str, Any]
    env: dict[str, str]


def classify(target: str) -> Source:
    """Resolve ``<input>`` to a URL or a local file.

    An ``http(s)://`` input with a host is a URL (yt-dlp handles the site). Anything
    else is treated as a local path and must exist — a bare typo'd URL (no scheme)
    or a missing file raises a friendly `SystemExit` rather than a yt-dlp stack trace.
    """
    parsed = urlparse(target)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return Source(InputKind.URL, target)
    if parsed.scheme in ("http", "https"):
        raise SystemExit(f"malformed URL {target!r} — missing host after the scheme")
    path = Path(target)
    if path.exists():
        return Source(InputKind.FILE, str(path))
    raise SystemExit(
        f"no such file: {target!r}\n"
        "Pass a local media file (.mp4/.mkv/.m4a/.wav) or an http(s):// video URL."
    )


def deno_on_path() -> str | None:
    """Path to a usable `deno`, or None. Checks PATH then the default install dir.

    yt-dlp needs a JS runtime for YouTube *downloads* (audio + caption fetch);
    without one YouTube throttles the client and the request 403s. `deno` is the
    runtime yt-dlp enables by default.
    """
    found = shutil.which("deno")
    if found:
        return found
    fallback = Path.home() / ".deno" / "bin" / "deno"
    return str(fallback) if fallback.is_file() else None


def build_ydl_opts(
    extra: dict[str, Any] | None = None,
    *,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    deno_path: str | None = None,
) -> YdlCall:
    """Build the yt-dlp opts + env stamped with cookies and the deno PATH shim.

    Pure: given the inputs it computes the opts dict and a copy of the environment
    with deno's directory prepended to ``PATH`` — it does NOT mutate ``os.environ``
    or touch the network (apply the env with `patched_path`). This is the single
    seam every yt-dlp call site shares (review S2/N3).

    * ``cookies_from_browser`` → ``cookiesfrombrowser=(BROWSER,)`` (reads the live
      logged-in session — the Udemy auth path). ``cookies_file`` → ``cookiefile``.
      Both are harmless for public YouTube.
    * ``deno_path`` defaults to `deno_on_path()`; when found, its directory is
      prepended to ``PATH`` in the returned env so yt-dlp's runtime probe sees it.
    """
    opts: dict[str, Any] = {"quiet": True, "no_warnings": True}
    if extra:
        opts.update(extra)
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies_file:
        opts["cookiefile"] = str(cookies_file)

    env = dict(os.environ)
    deno = deno_path if deno_path is not None else deno_on_path()
    if deno:
        env["PATH"] = f"{Path(deno).parent}{os.pathsep}{env.get('PATH', '')}"
    return YdlCall(opts, env)


_LECTURE_RE = re.compile(r"/lecture/(\d+)")


def lecture_id(url: str) -> str | None:
    """The Udemy lecture id in a ``/lecture/<id>`` URL, else None.

    Udemy's yt-dlp extractor expands such a URL into the WHOLE course playlist —
    the id in the path is NOT used to isolate the single lecture — so we detect it
    and select the matching entry ourselves (see `playlist_selection`).
    """
    m = _LECTURE_RE.search(url)
    return m.group(1) if m else None


def playlist_selection(
    url: str,
    *,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
) -> dict[str, Any]:
    """yt-dlp opts to fetch exactly the video `url` names, not a sibling playlist.

    Udemy ``/lecture/<id>`` URLs expand to the whole course → resolve the matching
    entry's 1-based index and pass ``playlist_items``. Any other URL → ``noplaylist``
    (so a YouTube ``&list=`` URL grabs the one video, not the list). Applied at every
    yt-dlp call site (transcribe, captions, frames) so isolation is consistent.
    """
    idx = _resolve_lecture_index(
        url, cookies_from_browser=cookies_from_browser, cookies_file=cookies_file
    )
    return {"playlist_items": str(idx)} if idx is not None else {"noplaylist": True}


def _resolve_lecture_index(
    url: str,
    *,
    cookies_from_browser: str | None,
    cookies_file: str | Path | None,
) -> int | None:
    """1-based index of the lecture `url` names within its course, or None when the
    URL is not a Udemy lecture (no isolation needed) or the lecture isn't found."""
    lid = lecture_id(url)
    if lid is None:
        return None
    return _flat_lecture_index(url, lid, cookies_from_browser, cookies_file)


def entry_matches_lecture(entry: Mapping[str, Any], lecture: str) -> bool:
    """Does this flat-playlist `entry` name lecture id `lecture`?

    The id is matched as a WHOLE ``/lecture/<id>`` path segment. A bare substring test
    lets a sibling win on a shared prefix — ``"1234567"`` is inside
    ``".../lecture/12345678"`` — and since the first match wins, that silently returns
    the index of the WRONG lecture, so glean would transcribe a different video than
    the URL names with nothing to indicate it.

    In a FULL extraction an entry's `id` is the internal media id rather than the
    lecture id, so the id field is compared exactly and the URLs by path segment.
    """
    if str(entry.get("id")) == lecture:
        return True
    segment = re.compile(rf"/lecture/{re.escape(lecture)}(?![0-9])")
    return any(
        segment.search(field) for field in (entry.get("url") or "", entry.get("webpage_url") or "")
    )


def _flat_lecture_index(
    url: str, lid: str, cookies_from_browser: str | None, cookies_file: str | Path | None
) -> int | None:  # pragma: no cover — network
    """Flat-extract the course (cheap: no per-lecture download) and find the entry
    carrying `lid`, matched by `entry_matches_lecture`."""
    import yt_dlp  # noqa: PLC0415 — lazy: heavy import, network-only path

    call = build_ydl_opts(
        {"extract_flat": True, "skip_download": True},
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    with ydl_errors(), patched_path(call.env), yt_dlp.YoutubeDL(call.opts) as ydl:
        info = ydl.extract_info(url, download=False)
    for i, entry in enumerate((info or {}).get("entries") or [], start=1):
        # A lazy playlist can yield None for an unavailable entry — skip, don't crash.
        if entry and entry_matches_lecture(entry, lid):
            return i
    return None


@contextmanager
def patched_path(env: dict[str, str]) -> Iterator[None]:  # pragma: no cover — env I/O
    """Temporarily install ``env['PATH']`` so yt-dlp's runtime probe finds deno."""
    old = os.environ.get("PATH")
    os.environ["PATH"] = env.get("PATH", old or "")
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old


def friendly_ydl_error(exc: Exception, target: str, *, has_cookies: bool) -> str:
    """Translate a raw yt-dlp failure into an actionable one-liner.

    Covers the three walls a study grab hits: **DRM** (Widevine — unextractable,
    we stop), **auth/403** (needs cookies — suggest ``--cookies-from-browser``) and
    a **missing JS runtime** (deno). Anything else is passed through verbatim so no
    diagnostic is swallowed.
    """
    msg = str(exc)
    low = msg.lower()
    host = urlparse(target).netloc
    if "drm" in low or "widevine" in low or "encrypted" in low:
        return (
            f"{host or target} is DRM-protected (Widevine) — yt-dlp cannot extract it "
            "and glean does not circumvent DRM. Stopping."
        )
    if "403" in low or "forbidden" in low or "login" in low or "account" in low:
        hint = (
            "the cookies may be stale or lack access to this content"
            if has_cookies
            else "pass --cookies-from-browser chrome (or --cookies FILE) to authenticate"
        )
        return f"access denied fetching {target} (403 / login required) — {hint}."
    if "js runtime" in low or "javascript" in low or "jsinterp" in low or "deno" in low:
        return (
            f"yt-dlp needs a JS runtime for {target}. Install deno: "
            "`curl -fsSL https://deno.land/install.sh | sh`, then ensure ~/.deno/bin is on PATH."
        )
    return f"yt-dlp failed for {target}: {msg}"


@contextmanager
def ydl_errors() -> Iterator[None]:
    """Convert a yt-dlp `DownloadError` into a plain `RuntimeError`.

    `yt_dlp.utils.DownloadError` is NOT a `RuntimeError` (it descends straight from
    `Exception` via `YoutubeDLError`), so a live DRM / 403 / auth failure would sail
    past the CLI's `except RuntimeError` and print a raw traceback — the friendly-error
    UX would never run. Every yt-dlp call site wraps its network op in this so the
    single `RuntimeError` funnel (and `friendly_ydl_error`) catches real failures too.

    yt_dlp is imported lazily HERE (network time), never at module load, so importing
    `source` — and therefore `cli` — stays cheap.
    """
    from yt_dlp.utils import DownloadError  # noqa: PLC0415 — lazy: heavy import

    try:
        yield
    except DownloadError as e:
        raise RuntimeError(str(e)) from e
