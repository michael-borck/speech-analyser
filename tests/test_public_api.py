"""Canonical public-surface checks shared across the analyser family.

These assert the names import and have the right shape. They deliberately do
NOT call ``analyse`` — that would load a whisper model / hit the network.
"""

import speech_analyser


def test_canonical_names_import():
    assert speech_analyser.SpeechAnalyser is not None
    assert speech_analyser.AudioAnalysis is not None
    assert speech_analyser.SpeechAnalysis is not None
    assert speech_analyser.SpeechAnalyserError is not None
    assert speech_analyser.ModelNotAvailableError is not None


def test_speech_analysis_is_alias():
    assert speech_analyser.SpeechAnalysis is speech_analyser.AudioAnalysis


def test_analyse_is_callable():
    assert callable(speech_analyser.analyse)


def test_manifest_name():
    assert speech_analyser.MANIFEST["name"] == "speech-analyser"


def test_version_is_str():
    assert isinstance(speech_analyser.__version__, str)


def test_all_lists_canonical_names():
    for name in (
        "SpeechAnalyser",
        "AudioAnalysis",
        "SpeechAnalysis",
        "analyse",
        "MANIFEST",
        "__version__",
        "SpeechAnalyserError",
        "ModelNotAvailableError",
    ):
        assert name in speech_analyser.__all__
