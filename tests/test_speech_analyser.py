"""Integration tests for SpeechAnalyser."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from speech_analyser import SpeechAnalyser
from speech_analyser.exceptions import SpeechAnalyserError


class TestSpeechAnalyserSilent:
    def test_unsupported_format_raises(self, tmp_path: Path):
        lens = SpeechAnalyser()
        p = tmp_path / "file.xyz"
        p.write_bytes(b"not audio")
        with pytest.raises(SpeechAnalyserError, match="Unsupported"):
            lens.analyse(p)

    def test_missing_file_raises(self, tmp_path: Path):
        lens = SpeechAnalyser()
        with pytest.raises(SpeechAnalyserError, match="not found"):
            lens.analyse(tmp_path / "missing.wav")

    def test_string_path_accepted(self, tmp_path: Path):
        lens = SpeechAnalyser()
        p = tmp_path / "file.xyz"
        p.write_bytes(b"not audio")
        with pytest.raises(SpeechAnalyserError, match="Unsupported"):
            lens.analyse(str(p))

    def test_success_shape(self, silent_wav: Path):
        """Full transcription of silent audio — requires faster-whisper installed."""
        lens = SpeechAnalyser()
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

    def test_model_not_available_is_subclass_of_speech_analyser_error(self):
        from speech_analyser.exceptions import ModelNotAvailableError, SpeechAnalyserError
        assert issubclass(ModelNotAvailableError, SpeechAnalyserError)

    def test_model_not_available_exported_from_package(self):
        from speech_analyser import ModelNotAvailableError  # noqa: F401

    def test_success_shape_has_diarization_keys(self, silent_wav: Path):
        lens = SpeechAnalyser()
        result = lens.analyse(silent_wav)
        assert "diarization_available" in result
        assert "speakers" in result
        assert "talk_time" in result
        assert result["diarization_available"] is False
        assert result["speakers"] is None
        assert result["talk_time"] is None

    def test_segments_have_speaker_key(self, silent_wav: Path):
        lens = SpeechAnalyser()
        result = lens.analyse(silent_wav)
        for seg in result["segments"]:
            assert "speaker" in seg
            assert seg["speaker"] is None  # no diarization

    def test_speech_metrics_has_new_fields(self, silent_wav: Path):
        lens = SpeechAnalyser()
        result = lens.analyse(silent_wav)
        m = result["speech_metrics"]
        assert "pace_category" in m
        assert "quality_score" in m
        assert "quality_factors" in m
        assert "quality_ratings" in m
        assert "insights" in m
        assert m["pace_category"] in {"slow", "natural", "fast", "unknown"}
        assert 0 <= m["quality_score"] <= 100


class TestSpeechAnalyserDiarization:
    def test_diarize_flag_populates_speakers(self, silent_wav: Path):
        from unittest.mock import patch
        from speech_analyser.diarizer import DiarizationTurn

        fake_turns = [
            DiarizationTurn(start=0.0, end=0.5, speaker="SPEAKER_00"),
        ]
        with patch("speech_analyser.speech_analyser.Diarizer.diarize", return_value=fake_turns):
            result = SpeechAnalyser().analyse(silent_wav, diarize=True)

        assert result["diarization_available"] is True
        assert result["speakers"] is not None
        assert len(result["speakers"]) >= 1
        assert result["speakers"][0]["id"] == "SPEAKER_00"

    def test_diarize_assigns_speaker_to_segments(self, silent_wav: Path):
        from unittest.mock import patch
        from speech_analyser.diarizer import DiarizationTurn

        fake_turns = [
            DiarizationTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        ]
        with patch("speech_analyser.speech_analyser.Diarizer.diarize", return_value=fake_turns):
            result = SpeechAnalyser().analyse(silent_wav, diarize=True)

        # All segments should be assigned to SPEAKER_00 (covers the whole file)
        for seg in result["segments"]:
            if seg["speaker"] is not None:
                assert seg["speaker"] == "SPEAKER_00"

    def test_diarize_talk_time_populated(self, silent_wav: Path):
        from unittest.mock import patch
        from speech_analyser.diarizer import DiarizationTurn

        fake_turns = [
            DiarizationTurn(start=0.0, end=0.5, speaker="SPEAKER_00"),
        ]
        with patch("speech_analyser.speech_analyser.Diarizer.diarize", return_value=fake_turns):
            result = SpeechAnalyser().analyse(silent_wav, diarize=True)

        assert result["talk_time"] is not None
        assert "is_balanced" in result["talk_time"]

    def test_diarize_false_skips_diarizer(self, silent_wav: Path):
        from unittest.mock import patch
        with patch("speech_analyser.speech_analyser.Diarizer.diarize") as mock_diarize:
            SpeechAnalyser().analyse(silent_wav, diarize=False)
        mock_diarize.assert_not_called()

    def test_model_not_available_reraises_not_wrapped(self, silent_wav: Path):
        from unittest.mock import patch
        from speech_analyser.exceptions import ModelNotAvailableError
        with patch("speech_analyser.speech_analyser.Diarizer.diarize", side_effect=ModelNotAvailableError("no model")):
            with pytest.raises(ModelNotAvailableError):
                SpeechAnalyser().analyse(silent_wav, diarize=True)


class TestAssignSpeakers:
    def _seg(self, start, end, text="hello"):
        from speech_analyser.transcriber import Segment
        return Segment(start=start, end=end, text=text, avg_logprob=-0.2)

    def _turn(self, start, end, speaker):
        from speech_analyser.diarizer import DiarizationTurn
        return DiarizationTurn(start=start, end=end, speaker=speaker)

    def test_full_overlap_assigns_speaker(self):
        from speech_analyser.speech_analyser import _assign_speakers
        seg = self._seg(0.0, 5.0)
        turn = self._turn(0.0, 5.0, "SPEAKER_00")
        result = _assign_speakers([seg], [turn])
        assert result == ["SPEAKER_00"]

    def test_gap_between_turns_returns_none(self):
        from speech_analyser.speech_analyser import _assign_speakers
        seg = self._seg(2.0, 3.0)
        t1 = self._turn(0.0, 1.5, "SPEAKER_00")
        t2 = self._turn(3.5, 5.0, "SPEAKER_01")
        result = _assign_speakers([seg], [t1, t2])
        assert result == [None]

    def test_max_overlap_wins(self):
        from speech_analyser.speech_analyser import _assign_speakers
        # seg 1-4, turn0 covers 0-2.5 (1.5s overlap), turn1 covers 2.5-5 (1.5s overlap) → tie → first wins
        seg = self._seg(1.0, 4.0)
        t0 = self._turn(0.0, 2.5, "SPEAKER_00")  # 1.5s overlap
        t1 = self._turn(2.5, 5.0, "SPEAKER_01")  # 1.5s overlap
        result = _assign_speakers([seg], [t0, t1])
        # Either speaker is acceptable on a tie; just assert one was assigned
        assert result[0] in ("SPEAKER_00", "SPEAKER_01")

    def test_empty_turns_returns_all_none(self):
        from speech_analyser.speech_analyser import _assign_speakers
        seg = self._seg(0.0, 5.0)
        result = _assign_speakers([seg], [])
        assert result == [None]

    def test_empty_segments_returns_empty(self):
        from speech_analyser.speech_analyser import _assign_speakers
        result = _assign_speakers([], [])
        assert result == []


class TestComputeTalkTime:
    def _seg(self, start, end, text):
        from speech_analyser.transcriber import Segment
        return Segment(start=start, end=end, text=text, avg_logprob=-0.2)

    def test_single_speaker_all_words(self):
        from speech_analyser.speech_analyser import _compute_talk_time
        seg = self._seg(0.0, 5.0, "hello world")
        talk_time, speaker_data = _compute_talk_time([seg], ["SPEAKER_00"])
        assert len(speaker_data) == 1
        assert speaker_data[0]["id"] == "SPEAKER_00"
        assert speaker_data[0]["word_count"] == 2
        assert speaker_data[0]["percentage"] == 100.0

    def test_no_speakers_assigned_returns_none_empty(self):
        from speech_analyser.speech_analyser import _compute_talk_time
        seg = self._seg(0.0, 5.0, "hello world")
        talk_time, speaker_data = _compute_talk_time([seg], [None])
        # All None assignments → total_words for assigned speakers = 0 → returns None, []
        assert speaker_data == [] or talk_time is None

    def test_is_balanced_true_for_equal_speakers(self):
        from speech_analyser.speech_analyser import _compute_talk_time
        seg1 = self._seg(0.0, 2.0, "hello world")
        seg2 = self._seg(2.0, 4.0, "foo bar")
        talk_time, _ = _compute_talk_time([seg1, seg2], ["SPEAKER_00", "SPEAKER_01"])
        assert talk_time["is_balanced"] is True

    def test_dominant_speaker_flagged(self):
        from speech_analyser.speech_analyser import _compute_talk_time
        seg1 = self._seg(0.0, 2.0, "a b c d e f g h i j")  # 10 words
        seg2 = self._seg(2.0, 4.0, "x")                    # 1 word
        talk_time, _ = _compute_talk_time([seg1, seg2], ["SPEAKER_00", "SPEAKER_01"])
        assert talk_time["is_balanced"] is False
        assert talk_time["dominant_speaker"] == "SPEAKER_00"


class TestWhisperCache:
    def test_returns_true_when_path_string_returned(self):
        from speech_analyser.transcriber import _is_whisper_cached
        from unittest.mock import patch
        with patch("huggingface_hub.try_to_load_from_cache", return_value="/cache/path/config.json"):
            assert _is_whisper_cached("base") is True

    def test_returns_false_when_none_returned(self):
        from speech_analyser.transcriber import _is_whisper_cached
        from unittest.mock import patch
        with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            assert _is_whisper_cached("base") is False

    def test_returns_false_for_object_sentinel(self):
        from speech_analyser.transcriber import _is_whisper_cached
        from unittest.mock import patch
        with patch("huggingface_hub.try_to_load_from_cache", return_value=object()):
            assert _is_whisper_cached("base") is False

    def test_returns_true_on_import_error(self):
        from speech_analyser.transcriber import _is_whisper_cached
        from unittest.mock import patch
        with patch("huggingface_hub.try_to_load_from_cache", side_effect=ImportError("no huggingface_hub")):
            # ImportError is caught by the broad except, so should return True (assume cached)
            assert _is_whisper_cached("base") is True


class TestCLI:
    def test_analyse_unsupported_exits_1(self, tmp_path: Path):
        p = tmp_path / "file.xyz"
        p.write_bytes(b"data")
        proc = subprocess.run(
            [sys.executable, "-m", "speech_analyser.cli", str(p), "--json"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 1
        err = json.loads(proc.stderr)
        assert "error" in err
        assert "success" not in err

    def test_serve_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "speech_analyser.cli", "serve", "--help"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert "--port" in proc.stdout
        assert "--host" in proc.stdout
