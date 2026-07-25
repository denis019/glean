"""Public transcribe orchestrator — the manual→auto→ASR ladder as a library call.

`transcribe_url` runs the ladder for a URL (author captions → auto captions → local
faster-whisper ASR, or `asr=True` to force ASR); `transcribe` classifies a
local-file-vs-URL input first (a local file has no caption track, so it goes straight
to ASR). Both live here rather than inside the CLI so there is ONE ladder that the
`glean` command *and* library callers share — no reinvention.

Progress is written to stderr; the return value is transcript markdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

from glean import asr as _asr
from glean import captions, markdown, source
from glean.source import InputKind


def transcribe_url(
    url: str,
    *,
    asr: bool = False,
    model: str = _asr.DEFAULT_MODEL,
    device: str = "auto",
    lang: str = "en",
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> str:
    """Transcript markdown for a URL: author captions → auto captions → local ASR.

    `asr=True` forces local faster-whisper transcription even when captions exist;
    otherwise captions are tried first (manual preferred over auto — review S1) and
    ASR is the fallback only when the URL has no caption track at all.
    """
    if asr:
        print(f"transcribing URL locally (faster-whisper {model}) …", file=sys.stderr)
        return _asr.export_url(
            url,
            model=model,
            device=device,
            lang=lang,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )

    try:
        vtt, kind = captions.fetch_captions(
            url, lang=lang, cookies_from_browser=cookies_from_browser, cookies_file=cookies_file
        )
    except captions.NoCaptionsError as e:
        print(f"{e}\nfalling back to local ASR …", file=sys.stderr)
        return _asr.export_url(
            url,
            model=model,
            device=device,
            lang=lang,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
    print(f"using {kind} caption track", file=sys.stderr)
    cues = captions.clean_vtt(vtt)
    return markdown.to_markdown(cues, source=url, is_url=True, caption_kind=kind)


def transcribe(
    input_str: str,
    *,
    asr: bool = False,
    model: str = _asr.DEFAULT_MODEL,
    device: str = "auto",
    lang: str = "en",
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> str:
    """Transcript markdown for a URL *or* a local media file.

    Classifies the input; a local file has no yt-dlp caption track so it goes straight
    to ASR, while a URL runs the full `transcribe_url` ladder.
    """
    src = source.classify(input_str)
    if src.kind is InputKind.FILE:
        print(f"transcribing local file (faster-whisper {model}) …", file=sys.stderr)
        return _asr.export_file(Path(src.value), model=model, device=device, lang=lang)
    return transcribe_url(
        src.value,
        asr=asr,
        model=model,
        device=device,
        lang=lang,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
