"""glean — pull study material (transcript + frames) out of any lecture video.

Source-agnostic: yt-dlp dispatches by URL, so a YouTube live VOD and a paid Udemy
lecture are the *same* code path, not special cases; a local media file is the
third input kind. Read-only against every source — it produces `[mm:ss]`
transcripts and still frames for a human to study, and nothing else.

The public transcribe orchestrator (the manual→auto→ASR ladder) is re-exported here
so downstream callers get ONE ladder: `glean.transcribe_url` / `glean.transcribe`.
"""

from glean.transcribe import transcribe, transcribe_url

__all__ = ["transcribe", "transcribe_url"]
