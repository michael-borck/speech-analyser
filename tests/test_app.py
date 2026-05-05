"""Tests for the audio-lens FastAPI app."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from audio_lens.app import app

client = TestClient(app)

_FAKE_ANALYSIS = {
    "transcript": "hello world",
    "language": "en",
    "duration": 5.0,
    "segments": [{"start": 0.0, "end": 5.0, "text": "hello world", "speaker": None}],
    "speech_metrics": {
        "word_count": 2,
        "speaking_rate_wpm": 24.0,
        "pace_category": "slow",
        "filler_word_count": 0,
        "filler_word_rate": 0.0,
        "filler_words_found": [],
        "silence_ratio": 0.0,
        "actual_speaking_time": 5.0,
        "quality_score": 62,
        "quality_factors": {"clarity": 25, "depth": 5, "balance": 18, "pace": 14},
        "quality_ratings": {"clarity": "excellent", "depth": "low", "balance": "good", "pace": "fair"},
        "insights": {
            "strengths": ["Very few filler words — speech is clear"],
            "observations": [],
        },
    },
    "diarization_available": False,
    "speakers": None,
    "talk_time": None,
    "file_path": "/tmp/test.wav",
    "file_size": 1234,
}


class TestHealthEndpoint:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_has_required_fields(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert isinstance(data["uptime"], float)


class TestRootEndpoint:
    def test_returns_200(self):
        assert client.get("/").status_code == 200

    def test_has_service_info(self):
        data = client.get("/").json()
        assert data["service"] == "audio-lens"
        assert "health" in data["endpoints"]
        assert "analyse" in data["endpoints"]


class TestAnalyseEndpoint:
    def test_no_file_returns_422(self):
        assert client.post("/analyse").status_code == 422

    def test_unsupported_format_returns_400(self):
        response = client.post(
            "/analyse",
            files={"file": ("test.xyz", b"not audio", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_invalid_model_returns_400(self):
        response = client.post(
            "/analyse",
            files={"file": ("test.wav", b"fake", "audio/wav")},
            data={"model": "giant"},
        )
        assert response.status_code == 400
        assert "Invalid model" in response.json()["detail"]

    def test_valid_file_returns_analysis_shape(self, silent_wav_bytes: bytes):
        with patch("audio_lens.app._get_lens") as mock_get_lens:
            mock_get_lens.return_value.analyse.return_value = _FAKE_ANALYSIS.copy()
            response = client.post(
                "/analyse",
                files={"file": ("test.wav", silent_wav_bytes, "audio/wav")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "hello world"
        assert data["language"] == "en"
        assert "speech_metrics" in data
        assert "success" not in data
        assert "data" not in data

    def test_response_has_no_envelope(self, silent_wav_bytes: bytes):
        with patch("audio_lens.app._get_lens") as mock_get_lens:
            mock_get_lens.return_value.analyse.return_value = _FAKE_ANALYSIS.copy()
            data = client.post(
                "/analyse",
                files={"file": ("test.wav", silent_wav_bytes, "audio/wav")},
            ).json()
        assert "success" not in data
        assert "error" not in data


class TestDiarizeEndpoint:
    def test_diarize_param_accepted(self, silent_wav_bytes: bytes):
        with patch("audio_lens.app._get_lens") as mock_get_lens:
            mock_get_lens.return_value.analyse.return_value = _FAKE_ANALYSIS.copy()
            response = client.post(
                "/analyse",
                files={"file": ("test.wav", silent_wav_bytes, "audio/wav")},
                data={"diarize": "false"},
            )
        assert response.status_code == 200
        mock_get_lens.return_value.analyse.assert_called_once()
        call_kwargs = mock_get_lens.return_value.analyse.call_args
        assert call_kwargs.kwargs.get("diarize") is False or call_kwargs.args[1] is False

    def test_diarize_env_var_default(self, silent_wav_bytes: bytes, monkeypatch):
        monkeypatch.setenv("AUDIO_LENS_DIARIZE", "true")
        with patch("audio_lens.app._get_lens") as mock_get_lens:
            mock_get_lens.return_value.analyse.return_value = _FAKE_ANALYSIS.copy()
            response = client.post(
                "/analyse",
                files={"file": ("test.wav", silent_wav_bytes, "audio/wav")},
            )
        assert response.status_code == 200
        call_kwargs = mock_get_lens.return_value.analyse.call_args
        # When AUDIO_LENS_DIARIZE=true and diarize not sent, diarize should be True
        diarize_value = call_kwargs.kwargs.get("diarize", call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
        assert diarize_value is True

    def test_diarize_model_unavailable_returns_503(self, silent_wav_bytes: bytes):
        from audio_lens.exceptions import ModelNotAvailableError
        with patch("audio_lens.app._get_lens") as mock_get_lens:
            mock_get_lens.return_value.analyse.side_effect = ModelNotAvailableError(
                "pyannote.audio is not installed"
            )
            response = client.post(
                "/analyse",
                files={"file": ("test.wav", silent_wav_bytes, "audio/wav")},
                data={"diarize": "true"},
            )
        assert response.status_code == 503
        assert "pyannote" in response.json()["detail"]
