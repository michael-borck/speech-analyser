from pathlib import Path
from typing import Any

from .diarizer import Diarizer, DiarizationTurn
from .embedding import embed_document
from .exceptions import SpeechAnalyserError, ModelNotAvailableError
from .speech_metrics import SpeechMetrics
from .transcriber import Segment, Transcriber, extracted_audio, is_video_container


def _speaker_for_span(
    start: float, end: float, turns: list[DiarizationTurn]
) -> str | None:
    """Speaker whose diarization turn overlaps [start, end] the most."""
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    return best_speaker


def _assign_speakers(
    segments: list[Segment], turns: list[DiarizationTurn]
) -> list[str | None]:
    """Assign each segment its speaker by maximum time overlap with diarization turns."""
    return [_speaker_for_span(seg.start, seg.end, turns) for seg in segments]


def _assign_word_speakers(
    segments: list[Segment], turns: list[DiarizationTurn]
) -> None:
    """Fill in Word.speaker for every word, by maximum overlap (in place).

    Per-word attribution is what lets consumers split a Whisper segment at
    the exact word where the voice changes; segment-level assignment alone
    credits a straddling segment entirely to one speaker.
    """
    for seg in segments:
        for word in seg.words:
            word.speaker = _speaker_for_span(word.start, word.end, turns)


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


class SpeechAnalyser:
    """Transcribes audio files and returns speech metrics.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
    """

    def __init__(self, model_size: str = "base") -> None:
        self._transcriber = Transcriber(model_size=model_size)
        self._analyzer = SpeechMetrics()
        self._diarizer = Diarizer()

    def analyse(
        self,
        file_path: Path | str,
        diarize: bool = False,
        language: str | None = None,
        vad_filter: bool = False,
    ) -> dict[str, Any]:
        """Analyse an audio or video file.

        Args:
            file_path: path to the audio or video file. Video containers
                (mp4/mov/mkv/webm/…) have their audio extracted via ffmpeg.
            diarize: if True, run speaker diarization (requires speech-analyser[diarization]
                     and HF_TOKEN env var). Default False.
            language: force a transcription language code (e.g. "en")
                instead of auto-detection. Default None (auto-detect).
            vad_filter: skip silence with faster-whisper's voice-activity
                filter. Default False.

        Returns:
            Analysis dict with transcript, language, duration, segments, speech_metrics,
            diarization_available, speakers, talk_time, file_path, file_size.

        Raises:
            SpeechAnalyserError: file missing, format unsupported, or transcription failed.
            ModelNotAvailableError: diarize=True but pyannote.audio not installed or
                                    no HF token configured.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            raise SpeechAnalyserError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self._transcriber.ALL_EXTENSIONS:
            raise SpeechAnalyserError(
                f"Unsupported audio format: {file_path.suffix}. "
                f"Supported: {', '.join(sorted(self._transcriber.ALL_EXTENSIONS))}"
            )

        # Video: extract the audio track ONCE and feed the same canonical
        # WAV to both transcription and diarization (the diarizer's decoder
        # cannot be relied on for video containers).
        if is_video_container(file_path):
            try:
                with extracted_audio(file_path) as wav_path:
                    return self._analyse_audio(
                        wav_path, file_path, diarize, language, vad_filter
                    )
            except ValueError as e:
                raise SpeechAnalyserError(str(e)) from e
        return self._analyse_audio(file_path, file_path, diarize, language, vad_filter)

    def _analyse_audio(
        self,
        audio_path: Path,
        original_path: Path,
        diarize: bool,
        language: str | None,
        vad_filter: bool,
    ) -> dict[str, Any]:
        try:
            file_size = original_path.stat().st_size
            result = self._transcriber.transcribe(
                audio_path, language=language, vad_filter=vad_filter
            )

            diarization_available = False
            speaker_assignments: list[str | None] = [None] * len(result.segments)
            talk_time: dict[str, Any] | None = None
            speaker_data: list[dict[str, Any]] = []
            diarization_turns: list[DiarizationTurn] = []

            speaker_embeddings: dict[str, list[float]] = {}
            if diarize:
                diarization_turns, speaker_embeddings = (
                    self._diarizer.diarize_with_embeddings(audio_path)
                )
                speaker_assignments = _assign_speakers(result.segments, diarization_turns)
                _assign_word_speakers(result.segments, diarization_turns)
                diarization_available = True
                talk_time, speaker_data = _compute_talk_time(result.segments, speaker_assignments, diarization_turns)

            metrics = self._analyzer.analyse(
                result,
                speaker_data=speaker_data if speaker_data else None,
            )

            return {
                "transcript": result.text,
                "embedding": embed_document(result.text),
                "language": result.language,
                "duration": result.duration,
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "speaker": speaker_assignments[i],
                        # Per-word timings + confidence (word_timestamps=True). Additive:
                        # existing consumers that read start/end/text/speaker are unaffected.
                        "words": [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end,
                                "probability": w.probability,
                                # Per-word diarization label (None unless
                                # diarize=True): the exact word where the
                                # voice changes inside a segment.
                                "speaker": w.speaker,
                            }
                            for w in s.words
                        ],
                    }
                    for i, s in enumerate(result.segments)
                ],
                "speech_metrics": metrics,
                "diarization_available": diarization_available,
                # The diarizer's raw turns ({start,end,speaker} in recording
                # order). Previously computed to label segments and then
                # discarded, forcing consumers (Debrief) to approximate them
                # from Whisper segment boundaries. Empty unless diarize=True.
                "turns": [
                    {"start": t.start, "end": t.end, "speaker": t.speaker}
                    for t in diarization_turns
                ],
                # One voiceprint embedding per diarized speaker, keyed by
                # cluster label, for cross-recording identity matching by
                # cosine similarity. {} unless diarized (or unavailable).
                "speaker_embeddings": speaker_embeddings,
                "speakers": speaker_data if speaker_data else None,
                "talk_time": talk_time,
                "file_path": str(original_path),
                "file_size": file_size,
            }
        except (SpeechAnalyserError, ModelNotAvailableError):
            raise
        except Exception as e:
            raise SpeechAnalyserError(str(e)) from e
