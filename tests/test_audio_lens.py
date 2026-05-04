"""Integration tests for AudioLens."""

import json
import subprocess
import sys
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

    def test_model_not_available_is_subclass_of_audio_lens_error(self):
        from audio_lens.exceptions import ModelNotAvailableError, AudioLensError
        assert issubclass(ModelNotAvailableError, AudioLensError)

    def test_model_not_available_exported_from_package(self):
        from audio_lens import ModelNotAvailableError  # noqa: F401


class TestCLI:
    def test_analyse_unsupported_exits_1(self, tmp_path: Path):
        p = tmp_path / "file.xyz"
        p.write_bytes(b"data")
        proc = subprocess.run(
            [sys.executable, "-m", "audio_lens.cli", "analyse", str(p), "--json"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 1
        err = json.loads(proc.stderr)
        assert "error" in err
        assert "success" not in err

    def test_serve_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "audio_lens.cli", "serve", "--help"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert "--port" in proc.stdout
        assert "--host" in proc.stdout
