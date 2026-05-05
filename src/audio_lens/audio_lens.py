from pathlib import Path
from typing import Any

from .diarizer import Diarizer, DiarizationTurn
from .exceptions import AudioLensError, ModelNotAvailableError
from .speech_analyzer import SpeechAnalyzer
from .transcriber import Segment, Transcriber


def _assign_speakers(
    segments: list[Segment], turns: list[DiarizationTurn]
) -> list[str | None]:
    """Assign each segment its speaker by maximum time overlap with diarization turns."""
    result: list[str | None] = []
    for seg in segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for turn in turns:
            overlap = max(0.0, min(seg.end, turn.end) - max(seg.start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        result.append(best_speaker)
    return result


def _compute_talk_time(
    segments: list[Segment],
    speaker_assignments: list[str | None],
    turns: list[DiarizationTurn] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Compute per-speaker word count, duration, and percentage.

    Percentage is based on word count when speech is present, otherwise on segment
    duration. If segments yield no speaker data (e.g. silent audio), falls back to
    computing duration totals directly from diarization turns.

    Note: percentages are rounded per-speaker (1 decimal place) and may not sum to exactly 100.0.

    Returns (talk_time dict, speaker_data list). Both are None/[] if no data available.
    """
    counts: dict[str, dict[str, Any]] = {}
    for seg, spk in zip(segments, speaker_assignments):
        if spk is None:
            continue
        if spk not in counts:
            counts[spk] = {"words": 0, "duration": 0.0}
        counts[spk]["words"] += len(seg.text.split())
        counts[spk]["duration"] += seg.end - seg.start

    # Fall back to raw turn durations when segments produced no speaker data.
    if not counts and turns:
        for turn in turns:
            if turn.speaker not in counts:
                counts[turn.speaker] = {"words": 0, "duration": 0.0}
            counts[turn.speaker]["duration"] += turn.end - turn.start

    if not counts:
        return None, []

    total_words = sum(v["words"] for v in counts.values())
    total_duration = sum(v["duration"] for v in counts.values())

    # Use word count for percentage when speech is present; fall back to duration.
    if total_words > 0:
        total_metric = total_words
        metric_key = "words"
    elif total_duration > 0:
        total_metric = total_duration
        metric_key = "duration"
    else:
        return None, []

    speaker_data = sorted(
        [
            {
                "id": spk,
                "word_count": data["words"],
                "duration_seconds": round(data["duration"], 1),
                "percentage": round(data[metric_key] / total_metric * 100, 1),
            }
            for spk, data in counts.items()
        ],
        key=lambda s: s["percentage"],
        reverse=True,
    )

    dominant = speaker_data[0]
    is_balanced = dominant["percentage"] <= 70
    talk_time = {
        "is_balanced": is_balanced,
        "dominant_speaker": None if is_balanced else dominant["id"],
    }
    return talk_time, speaker_data


class AudioLens:
    """Transcribes audio files and returns speech metrics.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
    """

    def __init__(self, model_size: str = "base") -> None:
        self._transcriber = Transcriber(model_size=model_size)
        self._analyzer = SpeechAnalyzer()
        self._diarizer = Diarizer()

    def analyse(self, file_path: Path | str, diarize: bool = False) -> dict[str, Any]:
        """Analyse an audio file.

        Args:
            file_path: path to the audio file.
            diarize: if True, run speaker diarization (requires audio-lens[diarization]
                     and HF_TOKEN env var). Default False.

        Returns:
            Analysis dict with transcript, language, duration, segments, speech_metrics,
            diarization_available, speakers, talk_time, file_path, file_size.

        Raises:
            AudioLensError: file missing, format unsupported, or transcription failed.
            ModelNotAvailableError: diarize=True but pyannote.audio not installed or
                                    no HF token configured.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            raise AudioLensError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self._transcriber.SUPPORTED_EXTENSIONS:
            raise AudioLensError(
                f"Unsupported audio format: {file_path.suffix}. "
                f"Supported: {', '.join(sorted(self._transcriber.SUPPORTED_EXTENSIONS))}"
            )

        try:
            file_size = file_path.stat().st_size
            result = self._transcriber.transcribe(file_path)

            diarization_available = False
            speaker_assignments: list[str | None] = [None] * len(result.segments)
            talk_time: dict[str, Any] | None = None
            speaker_data: list[dict[str, Any]] = []

            if diarize:
                turns = self._diarizer.diarize(file_path)
                speaker_assignments = _assign_speakers(result.segments, turns)
                diarization_available = True
                talk_time, speaker_data = _compute_talk_time(result.segments, speaker_assignments, turns)

            metrics = self._analyzer.analyse(
                result,
                speaker_data=speaker_data if speaker_data else None,
            )

            return {
                "transcript": result.text,
                "language": result.language,
                "duration": result.duration,
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "speaker": speaker_assignments[i],
                    }
                    for i, s in enumerate(result.segments)
                ],
                "speech_metrics": metrics,
                "diarization_available": diarization_available,
                "speakers": speaker_data if speaker_data else None,
                "talk_time": talk_time,
                "file_path": str(file_path),
                "file_size": file_size,
            }
        except (AudioLensError, ModelNotAvailableError):
            raise
        except Exception as e:
            raise AudioLensError(str(e)) from e
