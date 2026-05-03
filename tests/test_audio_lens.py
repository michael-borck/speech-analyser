"""Integration tests for AudioLens."""

from pathlib import Path

import pytest

from audio_lens import AudioLens
from audio_lens.exceptions import AudioLensError


class TestAudioLensSilent:
    def test_unsupported_format_raises(self, tmp_path: Path):
        lens = AudioLens()
        p = tmp_path / "file.xyz"
        p.write_bytes(b"not audio")
        with pytest.raises(AudioLensError, match="Unsupported"):
            lens.analyse(p)

    def test_missing_file_raises(self, tmp_path: Path):
        lens = AudioLens()
        with pytest.raises(AudioLensError, match="not found"):
            lens.analyse(tmp_path / "missing.wav")

    def test_string_path_accepted(self, tmp_path: Path):
        lens = AudioLens()
        p = tmp_path / "file.xyz"
        p.write_bytes(b"not audio")
        with pytest.raises(AudioLensError, match="Unsupported"):
            lens.analyse(str(p))

    def test_success_shape(self, silent_wav: Path):
        """Full transcription of silent audio — requires faster-whisper installed."""
        lens = AudioLens()
        result = lens.analyse(silent_wav)
        assert "transcript" in result
        assert "language" in result
        assert "duration" in result
        assert "segments" in result
        assert "speech_metrics" in result
        assert "file_path" in result
        assert "file_size" in result
        assert result["file_size"] > 0
        assert "success" not in result
        assert "data" not in result
