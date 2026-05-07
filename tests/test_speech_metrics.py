"""Unit tests for SpeechMetrics — no audio files needed."""

from speech_analyser.speech_metrics import SpeechMetrics, _detect_fillers, _pace_category, _quality_score, _insights
from speech_analyser.transcriber import Segment, TranscriptionResult


def _make_result(text: str, duration: float, segments=None) -> TranscriptionResult:
    if segments is None:
        segments = [Segment(start=0.0, end=duration * 0.8, text=text, avg_logprob=-0.2)]
    return TranscriptionResult(text=text, segments=segments, language="en", duration=duration)


class TestFillerDetection:
    def test_single_word_filler(self):
        count, found = _detect_fillers("um well I um think so")
        assert count >= 3
        assert "um" in found

    def test_multi_word_filler(self):
        count, found = _detect_fillers("I you know think you know")
        assert count >= 2
        assert "you know" in found

    def test_i_mean_detected(self):
        count, found = _detect_fillers("I mean that is i mean fine")
        assert "i mean" in found

    def test_empty_text(self):
        count, found = _detect_fillers("")
        assert count == 0
        assert found == []

    def test_no_fillers(self):
        count, found = _detect_fillers("The quick brown fox jumps")
        assert count == 0

    def test_multi_word_filler_with_punctuation(self):
        count, found = _detect_fillers("you know, I think you know.")
        assert count >= 2
        assert "you know" in found


class TestPaceCategory:
    def test_slow(self):
        assert _pace_category(70.0) == "slow"

    def test_natural(self):
        assert _pace_category(150.0) == "natural"

    def test_fast(self):
        assert _pace_category(220.0) == "fast"

    def test_boundary_90_is_natural(self):
        assert _pace_category(90.0) == "natural"

    def test_boundary_200_is_natural(self):
        assert _pace_category(200.0) == "natural"


class TestQualityScore:
    def test_score_in_range(self):
        score, factors, ratings = _quality_score(
            filler_rate=0.02, avg_words_per_segment=15.0, wpm=150.0, speaker_percentages=[]
        )
        assert 0 <= score <= 100

    def test_excellent_pace_scores_25(self):
        _, factors, _ = _quality_score(
            filler_rate=0.0, avg_words_per_segment=15.0, wpm=150.0, speaker_percentages=[]
        )
        assert factors["pace"] == 25

    def test_single_speaker_balance_neutral(self):
        _, factors, _ = _quality_score(
            filler_rate=0.0, avg_words_per_segment=15.0, wpm=150.0, speaker_percentages=[]
        )
        assert factors["balance"] == 18

    def test_balanced_two_speakers(self):
        _, factors, _ = _quality_score(
            filler_rate=0.0, avg_words_per_segment=15.0, wpm=150.0,
            speaker_percentages=[50.0, 50.0],
        )
        assert factors["balance"] == 25

    def test_dominant_speaker_penalised(self):
        _, factors, _ = _quality_score(
            filler_rate=0.0, avg_words_per_segment=15.0, wpm=150.0,
            speaker_percentages=[90.0, 10.0],
        )
        assert factors["balance"] < 18

    def test_ratings_are_valid_strings(self):
        _, _, ratings = _quality_score(
            filler_rate=0.05, avg_words_per_segment=10.0, wpm=130.0, speaker_percentages=[]
        )
        valid = {"excellent", "good", "fair", "low"}
        for v in ratings.values():
            assert v in valid


class TestSpeechMetrics:
    def test_word_count(self):
        result = _make_result("hello world foo bar", duration=60.0)
        m = SpeechMetrics().analyse(result)
        assert m["word_count"] == 4

    def test_speaking_rate_wpm(self):
        text = " ".join(["word"] * 60)
        result = _make_result(text, duration=60.0)
        m = SpeechMetrics().analyse(result)
        assert m["speaking_rate_wpm"] == 60.0

    def test_filler_word_rate_is_fraction(self):
        # "um" appears 1 time in 10 words → rate = 0.1
        result = _make_result("um " + " ".join(["word"] * 9), duration=10.0)
        m = SpeechMetrics().analyse(result)
        assert 0.0 <= m["filler_word_rate"] <= 1.0

    def test_silence_ratio_between_0_and_1(self):
        result = _make_result("hello", duration=10.0)
        m = SpeechMetrics().analyse(result)
        assert 0.0 <= m["silence_ratio"] <= 1.0

    def test_pace_category_present(self):
        result = _make_result("hello world", duration=10.0)
        m = SpeechMetrics().analyse(result)
        assert m["pace_category"] in {"slow", "natural", "fast"}

    def test_quality_score_present(self):
        result = _make_result("hello world", duration=10.0)
        m = SpeechMetrics().analyse(result)
        assert "quality_score" in m
        assert 0 <= m["quality_score"] <= 100

    def test_quality_factors_keys(self):
        result = _make_result("hello world", duration=10.0)
        m = SpeechMetrics().analyse(result)
        assert set(m["quality_factors"].keys()) == {"clarity", "depth", "balance", "pace"}

    def test_insights_structure(self):
        result = _make_result("hello world", duration=10.0)
        m = SpeechMetrics().analyse(result)
        assert "insights" in m
        assert "strengths" in m["insights"]
        assert "observations" in m["insights"]
        assert isinstance(m["insights"]["strengths"], list)
        assert isinstance(m["insights"]["observations"], list)

    def test_empty_transcript(self):
        result = _make_result("", duration=10.0, segments=[])
        m = SpeechMetrics().analyse(result)
        assert m["word_count"] == 0
        assert m["speaking_rate_wpm"] == 0.0

    def test_multi_word_fillers_detected(self):
        result = _make_result("you know I think you know maybe", duration=10.0)
        m = SpeechMetrics().analyse(result)
        assert "you know" in m["filler_words_found"]
        assert m["filler_word_count"] >= 2

    def test_empty_transcript_pace_category_is_unknown(self):
        result = _make_result("", duration=10.0, segments=[])
        m = SpeechMetrics().analyse(result)
        assert m["pace_category"] == "unknown"


class TestInsights:
    def test_high_filler_rate_observation(self):
        _, _, ratings = _quality_score(filler_rate=0.15, avg_words_per_segment=10.0, wpm=140.0, speaker_percentages=[])
        result = _insights(filler_rate=0.15, avg_words_per_segment=10.0, wpm=140.0, quality_ratings=ratings, speaker_data=[])
        assert any("Filler" in o for o in result["observations"])

    def test_short_turns_observation(self):
        _, _, ratings = _quality_score(filler_rate=0.0, avg_words_per_segment=3.0, wpm=140.0, speaker_percentages=[])
        result = _insights(filler_rate=0.0, avg_words_per_segment=3.0, wpm=140.0, quality_ratings=ratings, speaker_data=[])
        assert any("short turns" in o for o in result["observations"])

    def test_dominant_speaker_observation(self):
        speaker_data = [{"id": "SPEAKER_00", "percentage": 85.0}, {"id": "SPEAKER_01", "percentage": 15.0}]
        _, _, ratings = _quality_score(filler_rate=0.0, avg_words_per_segment=15.0, wpm=150.0, speaker_percentages=[85.0, 15.0])
        result = _insights(filler_rate=0.0, avg_words_per_segment=15.0, wpm=150.0, quality_ratings=ratings, speaker_data=speaker_data)
        assert any("SPEAKER_00" in o for o in result["observations"])

    def test_fallback_strength_when_no_strengths_fire(self):
        _, _, ratings = _quality_score(filler_rate=0.15, avg_words_per_segment=3.0, wpm=50.0, speaker_percentages=[])
        result = _insights(filler_rate=0.15, avg_words_per_segment=3.0, wpm=50.0, quality_ratings=ratings, speaker_data=[])
        assert len(result["strengths"]) >= 1  # fallback fires
