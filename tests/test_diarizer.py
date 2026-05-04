"""Unit tests for Diarizer — pyannote pipeline is always mocked."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audio_lens.diarizer import Diarizer, DiarizationTurn
from audio_lens.exceptions import AudioLensError, ModelNotAvailableError


class TestDiarizerImportGuard:
    def test_raises_model_not_available_when_pyannote_missing(self, tmp_path):
        d = Diarizer()
        with patch.object(d, "_import_pipeline", side_effect=ImportError("no module named pyannote")):
            with pytest.raises(ModelNotAvailableError, match="not installed"):
                d.diarize(tmp_path / "x.wav")

    def test_model_not_available_is_audio_lens_error(self, tmp_path):
        d = Diarizer()
        with patch.object(d, "_import_pipeline", side_effect=ImportError("no module")):
            with pytest.raises(AudioLensError):
                d.diarize(tmp_path / "x.wav")

    def test_raises_model_not_available_when_from_pretrained_fails(self, tmp_path):
        d = Diarizer()
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.side_effect = RuntimeError("401 Unauthorized")
        with patch.object(d, "_import_pipeline", return_value=mock_pipeline_cls):
            with patch.object(d, "_resolve_token", return_value="fake-token"):
                with patch("huggingface_hub.try_to_load_from_cache", return_value="/cache/path"):
                    with pytest.raises(ModelNotAvailableError, match="Could not load"):
                        d.diarize(tmp_path / "x.wav")


class TestDiarizerNoToken:
    def test_raises_when_no_token(self, tmp_path):
        mock_pipeline_cls = MagicMock()
        d = Diarizer()
        with patch.object(d, "_import_pipeline", return_value=mock_pipeline_cls):
            with patch.object(d, "_resolve_token", return_value=None):
                with pytest.raises(ModelNotAvailableError, match="No Hugging Face token"):
                    d.diarize(tmp_path / "x.wav")


class TestDiarizerTurns:
    def _make_diarizer_with_annotation(self, tracks):
        """Create a Diarizer whose pipeline returns a mock annotation."""
        mock_ann = MagicMock()
        mock_ann.itertracks.return_value = tracks

        d = Diarizer()
        d._pipeline = MagicMock(return_value=mock_ann)
        return d

    def test_returns_sorted_turns(self, silent_wav):
        t1 = MagicMock(start=0.0, end=2.0)
        t2 = MagicMock(start=2.5, end=5.0)
        d = self._make_diarizer_with_annotation([
            (t2, None, "SPEAKER_01"),
            (t1, None, "SPEAKER_00"),
        ])
        turns = d.diarize(silent_wav)
        assert len(turns) == 2
        assert turns[0].start == 0.0
        assert turns[0].speaker == "SPEAKER_00"
        assert turns[1].start == 2.5
        assert turns[1].speaker == "SPEAKER_01"

    def test_returns_diarization_turn_objects(self, silent_wav):
        t = MagicMock(start=1.0, end=3.0)
        d = self._make_diarizer_with_annotation([(t, None, "SPEAKER_00")])
        turns = d.diarize(silent_wav)
        assert isinstance(turns[0], DiarizationTurn)

    def test_empty_annotation_returns_empty_list(self, silent_wav):
        d = self._make_diarizer_with_annotation([])
        turns = d.diarize(silent_wav)
        assert turns == []

    def test_num_speakers_passed_to_pipeline(self, silent_wav):
        t = MagicMock(start=0.0, end=1.0)
        mock_ann = MagicMock()
        mock_ann.itertracks.return_value = [(t, None, "SPEAKER_00")]
        mock_pipeline = MagicMock(return_value=mock_ann)
        d = Diarizer()
        d._pipeline = mock_pipeline
        d.diarize(silent_wav, num_speakers=2)
        mock_pipeline.assert_called_once_with(str(silent_wav), num_speakers=2)

    def test_pipeline_error_raises_audio_lens_error(self, silent_wav):
        d = Diarizer()
        d._pipeline = MagicMock(side_effect=RuntimeError("CUDA error"))
        with pytest.raises(AudioLensError, match="Diarization failed"):
            d.diarize(silent_wav)

    def test_pipeline_cached_after_first_load(self, silent_wav):
        t = MagicMock(start=0.0, end=1.0)
        mock_ann = MagicMock()
        mock_ann.itertracks.return_value = [(t, None, "SPEAKER_00")]

        d = Diarizer()
        mock_pipeline_instance = MagicMock(return_value=mock_ann)

        with patch.object(d, "_import_pipeline") as mock_import:
            mock_import.return_value = MagicMock(
                from_pretrained=MagicMock(return_value=mock_pipeline_instance)
            )
            with patch.object(d, "_resolve_token", return_value="fake-token"):
                with patch("huggingface_hub.try_to_load_from_cache", return_value="/cache/path"):
                    d.diarize(silent_wav)
                    d.diarize(silent_wav)

        mock_import.assert_called_once()  # _import_pipeline only called once
