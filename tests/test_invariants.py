"""Invariant tests — fast, no real ML models, run by default.

These tests guard against the failure modes that motivated the recent
audit and clean-up: packaging bugs (records-analyser style), silent
graceful-degradation in analysers, eager model loading at construction
time, and version-string drift across hardcoded literals. They're cheap;
they should always run.
"""

from importlib.metadata import version
from unittest.mock import patch

import pytest


def test_package_imports_cleanly() -> None:
    """The package must import without optional ML deps actually loading.

    Smoke alarm: catches packaging bugs (records-analyser style) where
    a stray top-level import or syntax error would make the package
    un-importable. The faster-whisper / pyannote imports are deferred,
    so neither needs to actually load for this test to pass.
    """
    import speech_analyser  # noqa: F401
    from speech_analyser.app import app  # noqa: F401


def test_health_version_matches_installed_package() -> None:
    """/health must report the actual installed package version.

    Drift trap: a hardcoded version literal in the route would pass any
    test that hardcoded the same wrong string. Pin the route to
    importlib.metadata and verify it stays pinned.
    """
    from fastapi.testclient import TestClient

    from speech_analyser.app import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] == version("speech-analyser")


def test_speech_analyser_init_does_not_load_model() -> None:
    """SpeechAnalyser construction must be lazy — no model load at __init__.

    Loading the Whisper model is expensive and should only happen on the
    first analyse() call. Catches the regression where someone moves the
    load into __init__.
    """
    from speech_analyser.speech_analyser import SpeechAnalyser

    with patch("faster_whisper.WhisperModel") as mock_model:
        analyser = SpeechAnalyser(model_size="tiny")
        # WhisperModel must not have been instantiated yet.
        mock_model.assert_not_called()
        # The transcriber's lazy-load slot should still be None.
        assert analyser._transcriber._model is None


def test_diarizer_load_raises_loudly_when_pyannote_missing() -> None:
    """Diarizer.diarize() must raise ModelNotAvailableError when pyannote is unimportable.

    Family pattern: optional-dep failures must be loud, not silent.
    """
    from speech_analyser.diarizer import Diarizer
    from speech_analyser.exceptions import ModelNotAvailableError

    diarizer = Diarizer()
    with patch.object(diarizer, "_import_pipeline", side_effect=ImportError("no pyannote")):
        with pytest.raises(ModelNotAvailableError, match="not installed"):
            diarizer.diarize("/tmp/nonexistent.wav")
