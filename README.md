# glean

Pull study material — a cleaned `[mm:ss]` transcript and still frames — out of **any**
lecture video. Read-only against every source: it produces material for a human to
study, and nothing else.

`<url|file>` is a local media file **or** any yt-dlp-supported URL. yt-dlp dispatches by
site, so YouTube and Udemy are the *same* code path — Udemy just needs
`--cookies-from-browser` (or `--cookies FILE`) to read your logged-in session.

## Install

```
uv sync                         # + --extra asr for local transcription
uv run glean --help
```

## Usage

```
glean transcribe <url|file> [-o out.md] [--asr] [--model medium.en]
                            [--device auto|cuda|cpu] [--lang en]
                            [--cookies-from-browser chrome | --cookies FILE]
glean frames     <url|file> --at 25:19 --window 8:00-9:00 [--every 15] [-o dir]
                            [--cookies-from-browser chrome | --cookies FILE]
glean fetch      <url>      [-o DIR|FILE] [--format best]
                            [--cookies-from-browser chrome | --cookies FILE]
```

### transcribe

Prefers the **author's own subtitles**, falls back to **auto-captions**, and only when a
video has neither does it download the audio and transcribe **locally** with
faster-whisper. `--asr` forces local transcription; a **local file** always goes to ASR
(there is no separate caption track to fetch).

```
glean transcribe https://www.youtube.com/watch?v=ID -o transcript.md
glean transcribe lecture.mp4 --asr --model small.en
glean transcribe "$UDEMY_LECTURE_URL" --cookies-from-browser chrome -o lec.md
```

⚠ ASR garbles numbers, tickers and proper nouns regardless of model size. Prose reads
well; for exact on-screen values the **frames** are authoritative, not the transcript.

### frames

Grabs one still per timestamp. A **URL** is resolved by yt-dlp once, then ffmpeg
input-seeks the stream; a **local file** is ffmpeg-seeked directly (no yt-dlp).

```
glean frames https://www.youtube.com/watch?v=ID --at 25:19 --at 29:35
glean frames lecture.mp4 --window 8:00-9:00 --every 15 -o frames/
```

### fetch

Downloads and **keeps** the whole video. `transcribe` and `frames` each hit the source
independently, which trips the rate limiter on paid sites; `fetch` spends one request,
then both run locally against the file for free.

```
glean fetch "$UDEMY_LECTURE_URL" --cookies-from-browser firefox -o lecture.mp4
glean transcribe lecture.mp4 -o transcript.md
glean frames     lecture.mp4 --at 2:30
```

## MCP server

The same three commands are exposed as MCP tools, so an agent can call them directly
instead of shelling out. Register it once:

```
claude mcp add glean -s user -- uv run --project /path/to/glean --extra asr --extra mcp glean-mcp
```

`--project` (not `--directory`) deliberately: the server runs out of glean's
environment but keeps the **caller's** working directory, so a relative `out` lands in
the project you're working in. Paths come back absolute either way. Drop `-s user` to
register it for the current project only.

| tool | notes |
|---|---|
| `transcribe(video, out?, asr?, model?, device?, lang?, cookies_*)` | returns the markdown, or with `out` a receipt — pass `out` for a full lecture, it is tens of thousands of tokens |
| `frames(video, at?, window?, every?, out?, label?, fmt?, cookies_*)` | returns the paths written |
| `fetch(url, out?, fmt?, cookies_*)` | returns the kept path — call this **once** for a paid source, then point the other two at it |

Everything else is shared with the CLI: one ladder, one yt-dlp options builder, one set
of friendly DRM/403/deno errors. On stdio, stdout is the JSON-RPC stream, so the server
hands it to the protocol and points every `print` at stderr for the session.

## Requirements

- **ffmpeg** on `PATH` — for `frames` and (for local ASR) audio decoding.
- **deno** — yt-dlp needs a JS runtime for YouTube *downloads*, else the client is
  throttled and requests 403. `curl -fsSL https://deno.land/install.sh | sh`. This is a
  YouTube-only need; Udemy and local files don't use it.
- **`asr` extra** for local transcription: `uv sync --extra asr`. On Linux this pulls the
  CUDA-12 cuBLAS + cuDNN wheels, preloaded in-process so the GPU works without
  `LD_LIBRARY_PATH`; absent a GPU it runs on CPU.

## Notes

- **DRM.** Widevine-encrypted content (some Udemy courses) cannot be extracted by yt-dlp;
  glean says so plainly and stops. It does **not** circumvent DRM.
- **Auth vs rate-limit.** With valid cookies a 403 from Udemy is its rate limiter, not an
  auth failure (that would be a 401). Wait it out; use `fetch` to avoid provoking it.
- **Provenance.** Every transcript header names where the text came from (author subs /
  auto-captions / local ASR) and warns that `[mm:ss]` is VIDEO position, not wall-clock.

## Layout

| module | responsibility |
|---|---|
| `source.py` | local-file-vs-URL dispatch; the one `build_ydl_opts()` that stamps cookies + the deno PATH shim onto every yt-dlp call; playlist/lecture isolation; friendly DRM/403/deno errors |
| `captions.py` | fetch (author **and** auto, prefer author) + clean rolling-VTT |
| `asr.py` | audio download, ctypes CUDA preload, faster-whisper transcribe (cuda→cpu) |
| `frames.py` | stills at timestamps — URL (yt-dlp resolve + ffmpeg) and local-file (ffmpeg direct) |
| `fetch.py` | download-and-keep the full media |
| `transcribe.py` | the public author→auto→ASR ladder shared by the CLI and library callers |
| `markdown.py` | `[mm:ss]` transcript markdown with source-agnostic provenance |
| `cli.py` | the `transcribe` / `frames` / `fetch` subcommands |
| `mcp_server.py` | the same three as MCP tools — stdio-safe stdout, `SystemExit` funneled to tool errors |

## Dev

```
uv sync --all-groups            # + --extra asr for local transcription / ty
uv run pytest -q                # unit tests (no network/GPU), coverage ≥ 80%
uv run ruff format --check . && uv run ruff check && uv run ty check src/
```
