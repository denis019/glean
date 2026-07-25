"""Input classification + the pure `build_ydl_opts` seam (cookies + deno — S2/N3)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from glean import source
from glean.source import InputKind

if TYPE_CHECKING:
    from pathlib import Path


def test_classify_url() -> None:
    s = source.classify("https://www.youtube.com/watch?v=abc")
    assert s.kind is InputKind.URL
    assert s.is_url is True
    assert s.value == "https://www.youtube.com/watch?v=abc"


def test_classify_udemy_url_is_plain_url() -> None:
    # No Udemy special-casing — yt-dlp dispatches by site, glean only splits file/URL.
    s = source.classify("https://www.udemy.com/course/x/learn/lecture/123")
    assert s.kind is InputKind.URL


def test_classify_local_file(tmp_path: Path) -> None:
    f = tmp_path / "lecture.mp4"
    f.write_bytes(b"\x00")
    s = source.classify(str(f))
    assert s.kind is InputKind.FILE
    assert s.is_url is False
    assert s.value == str(f)


def test_classify_missing_file_exits() -> None:
    with pytest.raises(SystemExit, match="no such file"):
        source.classify("/does/not/exist.mp4")


def test_classify_malformed_url_exits() -> None:
    with pytest.raises(SystemExit, match="missing host"):
        source.classify("https://")


# ---- build_ydl_opts (review S2/N3) — pure ------------------------------------


def test_build_opts_base_has_quiet() -> None:
    call = source.build_ydl_opts(deno_path=None)
    assert call.opts["quiet"] is True
    assert call.opts["no_warnings"] is True


def test_build_opts_merges_extra() -> None:
    call = source.build_ydl_opts({"skip_download": True}, deno_path=None)
    assert call.opts["skip_download"] is True


def test_build_opts_cookies_from_browser_stamped() -> None:
    call = source.build_ydl_opts(cookies_from_browser="chrome", deno_path=None)
    assert call.opts["cookiesfrombrowser"] == ("chrome",)
    assert "cookiefile" not in call.opts


def test_build_opts_cookies_file_stamped() -> None:
    call = source.build_ydl_opts(cookies_file="cookies.txt", deno_path=None)
    assert call.opts["cookiefile"] == "cookies.txt"
    assert "cookiesfrombrowser" not in call.opts


def test_build_opts_no_cookies_by_default() -> None:
    call = source.build_ydl_opts(deno_path=None)
    assert "cookiesfrombrowser" not in call.opts
    assert "cookiefile" not in call.opts


def test_build_opts_deno_path_prepended() -> None:
    # deno's directory is prepended to PATH so yt-dlp's runtime probe finds it.
    call = source.build_ydl_opts(deno_path="/opt/deno/bin/deno")
    assert call.env["PATH"].startswith(f"/opt/deno/bin{os.pathsep}")


def test_build_opts_no_deno_leaves_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # deno absent -> PATH is passed through untouched.
    monkeypatch.setattr(source, "deno_on_path", lambda: None)
    call = source.build_ydl_opts()
    assert call.env["PATH"] == os.environ.get("PATH", "")


def test_build_opts_env_is_a_copy() -> None:
    # build_ydl_opts must not mutate os.environ (pure).
    before = dict(os.environ)
    source.build_ydl_opts(deno_path="/opt/deno/bin/deno")
    assert dict(os.environ) == before


# ---- deno discovery ----------------------------------------------------------


def test_deno_found_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source.shutil, "which", lambda _: "/usr/bin/deno")
    assert source.deno_on_path() == "/usr/bin/deno"


def test_deno_falls_back_to_home_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(source.shutil, "which", lambda _: None)
    monkeypatch.setattr(source.Path, "home", classmethod(lambda _cls: tmp_path))
    deno = tmp_path / ".deno" / "bin" / "deno"
    deno.parent.mkdir(parents=True)
    deno.touch()
    assert source.deno_on_path() == str(deno)


def test_deno_missing_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(source.shutil, "which", lambda _: None)
    monkeypatch.setattr(source.Path, "home", classmethod(lambda _cls: tmp_path))
    assert source.deno_on_path() is None


def test_build_opts_auto_discovers_deno(monkeypatch: pytest.MonkeyPatch) -> None:
    # deno_path omitted -> build_ydl_opts calls deno_on_path() itself.
    monkeypatch.setattr(source, "deno_on_path", lambda: "/x/deno/bin/deno")
    call = source.build_ydl_opts()
    assert call.env["PATH"].startswith(f"/x/deno/bin{os.pathsep}")


# ---- friendly errors ---------------------------------------------------------


def test_friendly_drm_error() -> None:
    msg = source.friendly_ydl_error(
        RuntimeError("This video is DRM protected"), "u", has_cookies=True
    )
    assert "DRM" in msg
    assert "circumvent" in msg


def test_friendly_403_suggests_cookies_when_absent() -> None:
    msg = source.friendly_ydl_error(
        RuntimeError("HTTP Error 403: Forbidden"), "u", has_cookies=False
    )
    assert "--cookies-from-browser" in msg


def test_friendly_403_notes_stale_when_cookies_present() -> None:
    msg = source.friendly_ydl_error(RuntimeError("login required"), "u", has_cookies=True)
    assert "stale" in msg


def test_friendly_deno_error() -> None:
    msg = source.friendly_ydl_error(RuntimeError("no JS runtime found"), "u", has_cookies=False)
    assert "deno" in msg


def test_friendly_jsinterp_error() -> None:
    msg = source.friendly_ydl_error(RuntimeError("jsinterp: failed to run"), "u", has_cookies=False)
    assert "deno" in msg


def test_friendly_json_not_mistaken_for_js() -> None:
    # "JSON" contains "js" — the old bare `"js" in low` wrongly fired the deno branch
    # on a 404 metadata error (review finding 3). It must fall through to passthrough.
    msg = source.friendly_ydl_error(
        RuntimeError("Unable to download JSON metadata: HTTP Error 404"), "u", has_cookies=False
    )
    assert "deno" not in msg
    assert "yt-dlp failed" in msg


def test_friendly_passthrough_for_unknown() -> None:
    msg = source.friendly_ydl_error(RuntimeError("kaboom"), "u", has_cookies=False)
    assert "kaboom" in msg


# ---- ydl_errors: DownloadError -> RuntimeError (review finding 1) ------------


def test_ydl_errors_converts_real_download_error() -> None:
    # yt_dlp.utils.DownloadError is NOT a RuntimeError — the CLI's `except RuntimeError`
    # would miss it. ydl_errors must convert it so the friendly funnel catches it.
    from yt_dlp.utils import DownloadError

    assert not issubclass(DownloadError, RuntimeError)  # the whole reason this exists
    with pytest.raises(RuntimeError, match="DRM") as ei:  # noqa: SIM117
        with source.ydl_errors():
            raise DownloadError("This video is DRM protected")
    assert not isinstance(ei.value, DownloadError)  # genuinely a plain RuntimeError now


def test_ydl_errors_is_transparent_on_success() -> None:
    with source.ydl_errors():
        result = 1 + 1
    assert result == 2


# ---- lecture isolation: Udemy /lecture/<id> expands to the whole course -------


def test_lecture_id_extracted_from_udemy_url() -> None:
    assert source.lecture_id("https://www.udemy.com/course/x/learn/lecture/31552430") == "31552430"


def test_lecture_id_none_for_non_lecture_url() -> None:
    assert source.lecture_id("https://www.youtube.com/watch?v=abc") is None


def test_playlist_selection_noplaylist_for_non_lecture() -> None:
    # A non-Udemy URL needs no isolation (and does NO network) -> plain noplaylist,
    # so a YouTube &list= URL still grabs the one video.
    sel = source.playlist_selection("https://www.youtube.com/watch?v=abc&list=PL1")
    assert sel == {"noplaylist": True}


def test_playlist_selection_uses_index_for_lecture(monkeypatch: pytest.MonkeyPatch) -> None:
    # A Udemy lecture URL -> resolve the matching entry's index and pass playlist_items.
    monkeypatch.setattr(source, "_flat_lecture_index", lambda *a, **k: 7)
    sel = source.playlist_selection("https://www.udemy.com/course/x/learn/lecture/31552430")
    assert sel == {"playlist_items": "7"}


def test_playlist_selection_falls_back_when_lecture_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source, "_flat_lecture_index", lambda *a, **k: None)
    sel = source.playlist_selection("https://www.udemy.com/course/x/learn/lecture/999")
    assert sel == {"noplaylist": True}


# ---- entry matching: the id is a whole path segment, not a substring ---------

_COURSE = "https://www.udemy.com/course/x/learn/lecture"


def test_entry_matches_lecture_by_url() -> None:
    assert source.entry_matches_lecture({"id": "media7", "url": f"{_COURSE}/1234567"}, "1234567")


def test_entry_matches_lecture_by_webpage_url() -> None:
    entry = {"id": "media7", "webpage_url": f"{_COURSE}/1234567"}
    assert source.entry_matches_lecture(entry, "1234567")


def test_entry_matches_lecture_by_exact_id() -> None:
    # A flat extraction can carry the lecture id as the entry id itself.
    assert source.entry_matches_lecture({"id": "1234567"}, "1234567")


def test_longer_sibling_id_is_not_a_match() -> None:
    # THE BUG: `lid in url` made "1234567" match ".../lecture/12345678". First match
    # wins, so glean silently transcribed a DIFFERENT lecture than the URL named.
    assert not source.entry_matches_lecture({"id": "m", "url": f"{_COURSE}/12345678"}, "1234567")


def test_prefix_id_is_not_matched_by_a_longer_request() -> None:
    assert not source.entry_matches_lecture({"id": "m", "url": f"{_COURSE}/1234"}, "12345")


def test_id_substring_elsewhere_in_the_url_is_not_a_match() -> None:
    # The course slug happens to contain the digits — not a lecture match.
    entry = {"id": "m", "url": "https://www.udemy.com/course/1234567/learn/lecture/99"}
    assert not source.entry_matches_lecture(entry, "1234567")


def test_entry_with_no_url_fields_does_not_match() -> None:
    assert not source.entry_matches_lecture({"id": "m"}, "1234567")


def test_colliding_siblings_resolve_to_the_requested_lecture() -> None:
    # End-to-end over the ordering that made the bug bite: the longer-id sibling comes
    # FIRST, so a substring match returned index 1 instead of 2.
    entries = [{"id": "a", "url": f"{_COURSE}/12345678"}, {"id": "b", "url": f"{_COURSE}/1234567"}]
    hits = [i for i, e in enumerate(entries, start=1) if source.entry_matches_lecture(e, "1234567")]
    assert hits == [2]
