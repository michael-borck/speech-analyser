"""Shared test fixtures for audio-lens."""

import math
import struct
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


@pytest.fixture
def silent_wav_bytes(silent_wav: Path) -> bytes:
    """Raw bytes of the silent WAV fixture."""
    return silent_wav.read_bytes()


@pytest.fixture
def two_speaker_wav(tmp_path: Path) -> Path:
    """A WAV with two tones at different frequencies, simulating two speakers.

    4 seconds total: 440 Hz for first 2s, 880 Hz for second 2s.
    Useful for testing speaker assignment logic with mocked diarization.
    """
    path = tmp_path / "two_speaker.wav"
    sample_rate = 16000
    duration = 4  # seconds

    frames = bytearray()
    for i in range(sample_rate * duration):
        t = i / sample_rate
        freq = 440.0 if t < 2.0 else 880.0
        sample = int(32767 * math.sin(2 * math.pi * freq * t))
        frames += struct.pack("<h", sample)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))

    return path
