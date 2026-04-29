from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    language: str
    duration: float


class Transcriber:
    """Wraps Faster-Whisper for audio transcription.

    Model is loaded lazily on first call to transcribe().
    """

    SUPPORTED_EXTENSIONS = {
        ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus",
    }

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file. Raises ValueError for unsupported formats."""
        if audio_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        model = self._load()
        raw_segments, info = model.transcribe(str(audio_path), word_timestamps=False)

        segments = []
        texts = []
        for seg in raw_segments:
            segments.append(Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                avg_logprob=seg.avg_logprob,
            ))
            texts.append(seg.text.strip())

        return TranscriptionResult(
            text=" ".join(texts),
            segments=segments,
            language=info.language,
            duration=info.duration,
        )
