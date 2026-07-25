---
name: glean
description: >
  glean — a general "extract study material from any video" CLI: transcribe (video →
  [mm:ss] markdown, captions-or-local-ASR), frames (stills at timestamps), and fetch
  (download + keep). Use when transcribing or grabbing frames from ANY video — a
  YouTube lecture, a paid Udemy course, or a local media file. Keywords: transcribe,
  transcript, captions, subtitles, frames, thumbnails, ASR, faster-whisper, whisper,
  yt-dlp, Udemy, cookies, deno, lecture, course video.
---

# glean — extract study material from any video

A **source-agnostic** CLI that pulls a **transcript** or **still frames** out of any
video (YouTube, Udemy, local file). Read-only against sources — it produces study
material for a human and nothing else.

## Run it

```bash
uv run glean <args>                          # from the repo root
uv run --extra asr glean <args>              # when the local-ASR path is needed
```
⚠ The `asr` extra (faster-whisper + Linux CUDA wheels) is optional — pass `--extra asr`
(or `uv sync --extra asr` once) for any local-ASR run. A plain `uv run glean` does NOT
pull it in.

## Three subcommands

```bash
glean transcribe <url|file> [-o out.md] [--asr] [--model medium.en]
                            [--device auto|cuda|cpu] [--lang en]
                            [--cookies-from-browser firefox | --cookies FILE]
glean frames     <url|file> --at 2:30 --window 8:00-9:00 [--every N] [-o DIR]
                            [--cookies-from-browser firefox]
glean fetch      <url>      [-o DIR|FILE] [--format best] [--cookies-from-browser firefox]
```

- **transcribe** — the ladder: **author subtitles → auto-captions → local faster-whisper
  ASR**. A local file skips straight to ASR. Output is `[mm:ss]`-stamped markdown.
  The model flag is `--model`.
- **frames** — resolves the stream URL once + ffmpeg input-seeks each timestamp (no full
  download); a local file is ffmpeg-seeked directly.
- **fetch** — download + **KEEP** the whole video, then transcribe/frames the LOCAL file.
  This is the Udemy answer (below).

## Sources — the per-source gotchas

**YouTube** — needs **deno** on PATH (`~/.deno/bin`, auto-checked) for downloads: without a
JS runtime YouTube throttles and audio 403s. deno is a **YouTube-only** need.

**Udemy (paid, behind login)** — the whole reason `fetch` exists:
- **Auth = cookies:** `--cookies-from-browser firefox` reads your logged-in session. No deno
  needed (Udemy has no JS challenge).
- **DRM:** Widevine-encrypted courses are **unextractable** — glean stops with a clear
  message, no circumvention. Non-DRM courses extract fine.
- **A `/lecture/<id>` URL expands to the WHOLE course** — glean isolates the matching entry
  via `source.playlist_selection()` (matches the id in each entry's `webpage_url`, because
  the entry `id` is the internal media id). So `transcribe <lecture-url>` gets THAT lecture.
- **Rate-limiting / 403:** after ~8 rapid requests Udemy 403s **even with valid cookies** —
  that's WAF/rate-limit, NOT auth (auth failure = 401). Cooldown minutes-to-hour. **Use
  `fetch` once, then work on the local file** to avoid re-hitting it.
- **Workflow:**
  ```bash
  glean fetch <udemy-url> --cookies-from-browser firefox -o lecture.mp4   # one request
  glean transcribe lecture.mp4 -o transcript.md                          # local, free
  glean frames     lecture.mp4 --at 2:30                                 # local, free
  ```

**Local file** — no yt-dlp, no cookies, no deno; ASR / ffmpeg-seek directly.

## ASR quality + GPU

- **GPU-automatic:** the CUDA wheels preload in-process (no `LD_LIBRARY_PATH`); CPU fallback
  otherwise. On a 4 GB laptop GPU: 84-min video ≈ 3 min; `medium.en` (default) fits ~1 GB
  VRAM. `small.en` for CPU-only speed.
- ⚠ **ASR garbles numbers, tickers, proper nouns** regardless of model ("Jayce Pham"→"Jay
  Spam", "volume"→"Valium", "5-minute"→"50 minute", "weekly"→"Whitney"). Prose is very
  readable; **for exact on-screen values the FRAMES stay authoritative** — the transcript is
  for the reasoning.

## Where things live

- Code: `src/glean/` — `source.py` (classify + `build_ydl_opts` + cookies/deno +
  `playlist_selection`), `captions.py`, `asr.py`, `frames.py`, `fetch.py`, `markdown.py`,
  `transcribe.py` (the public `transcribe`/`transcribe_url` ladder), `cli.py`.
- Library use: `from glean import transcribe, transcribe_url, frames, asr` — downstream
  callers get the same ladder; don't re-implement media logic against yt-dlp directly.
- Gates: `uv run pytest -q && uv run ruff check . && uv run ty check src/`.
  Network/GPU paths are `# pragma: no cover`; hold `--cov-fail-under=80`.
