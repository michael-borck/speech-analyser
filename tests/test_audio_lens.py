"""Integration tests for AudioLens."""

from pathlib import Path

import pytest

from audio_lens import AudioLens


class TestAudioLensSilent:
    """Tests using the silent WAV fixture — no Whisper model required for format checks."""

    def test_unsupported_format_returns_failure(self, tmp_path: Path):
        lens = AudioLens()
        p = tmp_path / "file.xyz"
        p.write_bytes(b"not audio")
        result = lens.analyse(p)
        assert result["success"] is False
        assert "Unsupported" in result["error"]

    def test_missing_file_returns_failure(self, tmp_path: Path):
        lens = AudioLens()
        result = lens.analyse(tmp_path / "missing.wav")
        assert result["success"] is False
        assert "error" in result

    def test_string_path_accepted(self, tmp_path: Path):
        lens = AudioLens()
        p = tmp_path / "file.xyz"
        p.write_bytes(b"not audio")
        result = lens.analyse(str(p))
        # Should fail on unsupported format, not on path type
        assert result["success"] is False
        assert "Unsupported" in result["error"]

    def test_success_shape(self, silent_wav: Path):
        """Full transcription of silent audio — requires faster-whisper installed."""
        lens = AudioLens()
        result = lens.analyse(silent_wav)
        assert result["success"] is True
        data = result["data"]
        assert "transcript" in data
        assert "language" in data
        assert "duration" in data
        assert "segments" in data
        assert "speech_metrics" in data
        assert "file_path" in data
        assert "file_size" in data
        assert data["file_size"] > 0
