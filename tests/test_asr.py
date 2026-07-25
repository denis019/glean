"""ASR fallback — the pure, testable bits (no network, no GPU, no faster-whisper).

The download / transcribe paths are network+GPU and `pragma: no cover`; these tests
pin CUDA-loader discovery, device selection, and the missing-extra error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from glean import asr


def test_enable_cuda_libs_returns_false_without_wheels(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the CUDA wheels being absent (import fails) — deterministic whether or
    # not the `asr` extra happens to be installed. Must return False (CPU) and not raise.
    monkeypatch.setitem(sys.modules, "nvidia.cublas", None)
    assert asr.enable_cuda_libs() is False


def test_transcribe_audio_raises_clear_error_without_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate faster-whisper being absent; the error must name the extra to install.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(RuntimeError, match="asr` extra"):
        asr.transcribe_audio(Path("nope.m4a"), device="cpu")


def test_default_model_is_medium_en() -> None:
    assert asr.DEFAULT_MODEL == "medium.en"
