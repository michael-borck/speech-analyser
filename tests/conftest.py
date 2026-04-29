"""Shared test fixtures for audio-lens."""

import wave
from pathlib import Path

import pytest


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """A minimal valid WAV file containing 1 second of silence."""
    path = tmp_path / "silent.wav"
    sample_rate = 16000
    num_samples = sample_rate  # 1 second

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)

    return path


@pytest.fixture
def sample_audio_dir() -> Path:
    """Points to video-lens test fixtures if present, else returns None."""
    candidate = Path(__file__).parents[3] / "video-lens" / "tests"
    return candidate if candidate.exists() else None
