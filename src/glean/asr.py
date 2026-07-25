"""Local ASR fallback — transcribe a video's audio when it has no caption track.

When `captions.fetch_captions` finds no track (`NoCaptionsError`), or the caller
forces ``--asr``, this transcribes the audio locally with faster-whisper and returns
the same cue shape. Two things this path needs beyond captions:

* **A JS runtime for yt-dlp.** Without one, yt-dlp falls back to a throttled client
  and every *audio* download 403s. `deno` is the default runtime; `source.deno_on_path`
  finds it, and the download goes through `source.build_ydl_opts` (cookies + the deno
  PATH shim — review S2), so a Udemy audio grab authenticates like the others.
* **CUDA runtime libs, in-process.** ctranslate2 (faster-whisper's backend) needs
  CUDA-12 cuBLAS + cuDNN-9. We ship them as the ``asr`` extra's pip wheels and preload
  them with ``ctypes`` (`enable_cuda_libs`) so the GPU works without the caller setting
  ``LD_LIBRARY_PATH``. Absent the wheels it runs on CPU.

The download / transcribe paths are network / GPU / heavy-import and marked
``pragma: no cover``; the pure formatting lives in `markdown.to_markdown`.
"""

from __future__ import annotations

import ctypes
import tempfile
from pathlib import Path

from glean import markdown, source

# English-only default: the `.en` models beat the multilingual ones per-parameter on
# English. `medium.en` is the accuracy/speed sweet spot on a small GPU (~1 GB VRAM);
# drop to `small.en` for CPU-only.
DEFAULT_MODEL = "medium.en"


def enable_cuda_libs() -> bool:
    """Preload the pip-installed CUDA runtime libs so ctranslate2's GPU backend can
    find them in-process (the loader reads ``LD_LIBRARY_PATH`` only at startup, so we
    ``dlopen`` the ``.so``s directly instead of re-execing).

    Returns True if the ``asr`` extra's CUDA wheels are present (GPU is possible),
    False otherwise (caller runs on CPU). Never raises.
    """
    lib_dirs: list[Path] = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            mod = __import__(pkg, fromlist=["__path__"])
        except ImportError:
            return False  # CUDA wheels not installed -> CPU
        lib_dir = Path(next(iter(mod.__path__))) / "lib"
        if lib_dir.is_dir():
            lib_dirs.append(lib_dir)
    sos = [so for d in lib_dirs for so in d.iterdir() if ".so" in so.name]
    # Retry a few passes so inter-lib deps (cublasLt<-cublas, cudnn engines<-cudnn)
    # resolve regardless of iterdir order.
    pending = list(sos)
    for _ in range(3):
        pending = [so for so in pending if not _try_load(so)]
        if not pending:
            break
    return True


def _try_load(so_path: Path) -> bool:  # pragma: no cover — needs the CUDA wheels
    try:
        ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    return True


def download_audio(
    url: str,
    dest_dir: Path,
    *,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> Path:  # pragma: no cover — network
    """Download the video's audio into `dest_dir` via yt-dlp. Returns the path.

    deno is NOT required in general — only **YouTube** needs a JS runtime (its
    signature/nsig challenge); Udemy (cookies + HLS) and other sources download
    without it, and local files never reach here. So we don't hard-require deno:
    `source.build_ydl_opts` prepends it to PATH *if present*, and if a YouTube grab
    throttles/403s for lack of it, `friendly_ydl_error` surfaces the deno hint.
    Cookies + the deno shim are stamped via `build_ydl_opts` (review S2).
    """
    import yt_dlp  # noqa: PLC0415 — lazy: heavy import, network-only path

    call = source.build_ydl_opts(
        {
            # Prefer an audio-only stream (YouTube: itag 140 / m4a), but fall back to
            # `best` — sources like Udemy serve only COMBINED video+audio, so there is
            # no audio-only format; faster-whisper reads the audio out of the mp4/HLS.
            "format": "140/bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": {"default": str(dest_dir / "audio.%(ext)s")},
            # Fetch exactly the lecture the URL names (Udemy expands it to the whole
            # course); noplaylist for everything else.
            **source.playlist_selection(
                url, cookies_from_browser=cookies_from_browser, cookies_file=cookies_file
            ),
        },
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    with source.ydl_errors(), source.patched_path(call.env), yt_dlp.YoutubeDL(call.opts) as ydl:
        ydl.download([url])
    audio = next(iter(dest_dir.glob("audio.*")), None)
    if audio is None:
        raise RuntimeError(f"yt-dlp produced no audio file for {url}")
    return audio


def transcribe_audio(
    audio: Path,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    lang: str = "en",
    beam: int = 5,
) -> list[tuple[int, str]]:  # pragma: no cover — heavy import + GPU/CPU inference
    """Transcribe `audio` to `[(start_second, text)]` cues with faster-whisper.

    `device` is ``auto`` (try CUDA, fall back to CPU int8), ``cuda`` or ``cpu``.
    """
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415 — optional `asr` extra
    except ImportError as e:
        raise RuntimeError(
            "ASR needs the `asr` extra — install with `uv sync --extra asr` "
            "(GPU support is included on Linux)."
        ) from e

    if device in ("auto", "cuda"):
        enable_cuda_libs()

    order = {
        "auto": [("cuda", "int8_float16"), ("cpu", "int8")],
        "cuda": [("cuda", "int8_float16")],
        "cpu": [("cpu", "int8")],
    }[device]
    last_err: Exception | None = None
    for dev, compute in order:
        try:
            whisper = WhisperModel(model, device=dev, compute_type=compute)
            segments, _ = whisper.transcribe(
                str(audio),
                language=lang,
                vad_filter=True,  # skip long "starting soon" silence
                beam_size=beam,
                condition_on_previous_text=False,  # faster, avoids repetition loops
            )
            return [(int(s.start), s.text.strip()) for s in segments if s.text.strip()]
        except Exception as e:  # noqa: BLE001 — GPU libs may be missing; CPU always works
            last_err = e
    raise RuntimeError(f"transcription failed on all backends: {last_err}")


def export_url(
    url: str,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    lang: str = "en",
    beam: int = 5,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> str:  # pragma: no cover — orchestrates network + inference
    """Download + transcribe a URL + format. Returns transcript markdown."""
    with tempfile.TemporaryDirectory() as td:
        audio = download_audio(
            url,
            Path(td),
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
        cues = transcribe_audio(audio, model=model, device=device, lang=lang, beam=beam)
    return markdown.to_markdown(cues, source=url, is_url=True, asr_model=model)


def export_file(
    path: Path,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    lang: str = "en",
    beam: int = 5,
) -> str:  # pragma: no cover — heavy import + inference
    """Transcribe a LOCAL media file (no yt-dlp) + format. Returns transcript markdown.

    faster-whisper reads .mp4/.mkv/.m4a/.wav directly (it feeds ffmpeg), so no audio
    download step — the file IS the audio source.
    """
    cues = transcribe_audio(path, model=model, device=device, lang=lang, beam=beam)
    return markdown.to_markdown(cues, source=str(path), is_url=False, asr_model=model)
