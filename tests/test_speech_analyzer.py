"""Unit tests for SpeechAnalyzer — no audio files needed."""

from audio_lens.speech_analyzer import SpeechAnalyzer
from audio_lens.transcriber import Segment, TranscriptionResult


def _make_result(text: str, duration: float, segments=None) -> TranscriptionResult:
    if segments is None:
        segments = [Segment(start=0.0, end=duration * 0.8, text=text, avg_logprob=-0.2)]
    return TranscriptionResult(text=text, segments=segments, language="en", duration=duration)


class TestSpeechMetrics:
    def test_word_count(self):
        result = _make_result("hello world foo bar", duration=60.0)
        metrics = SpeechAnalyzer().analyse(result)
        assert metrics["word_count"] == 4

    def test_speaking_rate_wpm(self):
        # 60 words in 60 seconds = 60 wpm
        text = " ".join(["word"] * 60)
        result = _make_result(text, duration=60.0)
        metrics = SpeechAnalyzer().analyse(result)
        assert metrics["speaking_rate_wpm"] == 60.0

    def test_filler_word_detection(self):
        result = _make_result("um so basically I um think", duration=10.0)
        metrics = SpeechAnalyzer().analyse(result)
        assert metrics["filler_word_count"] >= 2

    def test_silence_ratio_between_0_and_1(self):
        result = _make_result("hello", duration=10.0)
        metrics = SpeechAnalyzer().analyse(result)
        assert 0.0 <= metrics["silence_ratio"] <= 1.0

    def test_empty_transcript(self):
        result = _make_result("", duration=10.0, segments=[])
        metrics = SpeechAnalyzer().analyse(result)
        assert metrics["word_count"] == 0
        assert metrics["speaking_rate_wpm"] == 0.0
