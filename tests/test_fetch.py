"""`glean fetch` — download-and-keep, with the yt-dlp network object stubbed.

`download_media` had no tests at all despite being the headline Udemy workflow, and
its remux fallback was dead code in the default `-o DIR` case. These pin both the
happy path and that fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import pytest

from glean import fetch

_URL = "https://example.com/v"  # non-Udemy: playlist_selection does no network for it


class _FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL that "downloads" by touching a file.

    `writes` is the extension actually produced; `guesses` is what prepare_filename
    reports. When they differ we are simulating the HLS/merge remux the fallback
    exists for.
    """

    written: Path | None = None

    def __init__(self, opts: dict, *, writes: str = "mp4", guesses: str = "mp4") -> None:
        self._tmpl = opts["outtmpl"]["default"]
        self._writes = writes
        self._guesses = guesses

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def _resolve(self, ext: str) -> Path:
        return Path(self._tmpl.replace("%(title)s", "Lecture 01").replace("%(ext)s", ext))

    def extract_info(self, url: str, *, download: bool = True) -> dict:  # noqa: ARG002
        real = self._resolve(self._writes)
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_bytes(b"video")
        type(self).written = real
        return {"title": "Lecture 01", "ext": self._writes}

    def prepare_filename(self, _entry: object) -> str:
        return str(self._resolve(self._guesses))


def _install(monkeypatch: pytest.MonkeyPatch, **kw: str) -> None:
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", lambda opts: _FakeYDL(opts, **kw))


def test_fetch_into_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install(monkeypatch)
    got = fetch.download_media(_URL, tmp_path)
    assert got == tmp_path / "Lecture 01.mp4"
    assert got.read_bytes() == b"video"


def test_fetch_to_explicit_file_replaces_the_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # -o lecture.mp4 -> the real container's extension wins over the one given.
    _install(monkeypatch, writes="mkv", guesses="mkv")
    got = fetch.download_media(_URL, tmp_path / "lecture.mp4")
    assert got == tmp_path / "lecture.mkv"


def test_remux_fallback_finds_the_file_for_a_target_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # THE BUG: for a target directory the fallback globbed the UNEXPANDED template —
    # a file literally named "%(title)s.*" — so it never matched and fetch raised
    # "produced no file" on a video sitting right there on disk. This is the default
    # invocation (-o omitted) and the whole Udemy path, where an HLS remux changing
    # the extension is exactly what the fallback is for.
    _install(monkeypatch, writes="mkv", guesses="mp4")
    got = fetch.download_media(_URL, tmp_path)
    assert got == tmp_path / "Lecture 01.mkv"
    assert got.read_bytes() == b"video"


def test_remux_fallback_for_the_default_cwd_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `-o` omitted -> Path(), i.e. the cwd. Same dead-glob path as above.
    monkeypatch.chdir(tmp_path)
    _install(monkeypatch, writes="mkv", guesses="mp4")
    assert fetch.download_media(_URL, Path()) == Path("Lecture 01.mkv")


def test_title_with_glob_metacharacters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # "Lecture [Part 1]" would have been read as a glob character class. Stems are
    # compared directly now, so bracketed titles resolve.
    import yt_dlp

    class _Bracketed(_FakeYDL):
        def _resolve(self, ext: str) -> Path:
            return Path(self._tmpl.replace("%(title)s", "Lecture [Part 1]").replace("%(ext)s", ext))

    monkeypatch.setattr(yt_dlp, "YoutubeDL", lambda opts: _Bracketed(opts, writes="mkv"))
    assert fetch.download_media(_URL, tmp_path) == tmp_path / "Lecture [Part 1].mkv"


def test_playlist_entry_is_unwrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A Udemy lecture URL comes back wrapped in a playlist; the selected entry is what
    # prepare_filename must be handed.
    import yt_dlp

    seen: dict = {}

    class _Playlist(_FakeYDL):
        def extract_info(self, url: str, *, download: bool = True) -> dict:
            entry = super().extract_info(url, download=download)
            return {"entries": [entry], "_type": "playlist"}

        def prepare_filename(self, entry: object) -> str:
            seen["entry"] = entry
            return super().prepare_filename(entry)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _Playlist)
    fetch.download_media(_URL, tmp_path)
    assert seen["entry"]["title"] == "Lecture 01"  # the entry, not the playlist wrapper


def test_no_file_produced_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import yt_dlp

    class _Nothing(_FakeYDL):
        def extract_info(self, url: str, *, download: bool = True) -> dict:  # noqa: ARG002
            return {"title": "Lecture 01", "ext": "mp4"}  # downloads nothing

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _Nothing)
    with pytest.raises(RuntimeError, match="produced no file"):
        fetch.download_media(_URL, tmp_path)


def test_download_error_becomes_runtimeerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # fetch goes through source.ydl_errors like every other call site.
    import yt_dlp

    class _Boom(_FakeYDL):
        def extract_info(self, url: str, *, download: bool = True) -> dict:  # noqa: ARG002
            from yt_dlp.utils import DownloadError

            raise DownloadError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _Boom)
    with pytest.raises(RuntimeError, match="403") as ei:
        fetch.download_media(_URL, tmp_path)
    from yt_dlp.utils import DownloadError

    assert not isinstance(ei.value, DownloadError)


def test_cookies_and_format_reach_ydl_opts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import yt_dlp

    seen: dict = {}

    def _capture(opts: dict) -> _FakeYDL:
        seen.update(opts)
        return _FakeYDL(opts)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _capture)
    fetch.download_media(_URL, tmp_path, fmt="bestvideo", cookies_from_browser="firefox")
    assert seen["format"] == "bestvideo"
    assert seen["cookiesfrombrowser"] == ("firefox",)
    assert seen["noplaylist"] is True
