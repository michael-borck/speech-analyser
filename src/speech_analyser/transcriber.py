import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Word:
    """A single transcribed word with its timing and confidence.

    `probability` is whisper's per-word confidence (0–1). Word-level timings let
    consumers compute precise within-window metrics (per-segment WPM, pauses from
    word gaps, filler localisation) that segment-level timing can't support.
    """

    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    language: str
    duration: float


# Maps model size → approximate download size for the user-facing warning.
_MODEL_SIZES = {
    "tiny": "39 MB",
    "tiny.en": "39 MB",
    "base": "74 MB",
    "base.en": "74 MB",
    "small": "244 MB",
    "small.en": "244 MB",
    "medium": "769 MB",
    "medium.en": "769 MB",
    "large": "1.5 GB",
    "large-v1": "1.5 GB",
    "large-v2": "1.5 GB",
    "large-v3": "1.5 GB",
}

SUPPORTED_MODELS: frozenset[str] = frozenset(_MODEL_SIZES.keys())


def _is_whisper_cached(model_size: str) -> bool:
    """Return True if the faster-whisper model is already in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        repo_id = f"Systran/faster-whisper-{model_size}"
        result = try_to_load_from_cache(repo_id, "config.json")
        return isinstance(result, str)
    except Exception:
        return True  # assume cached on any error to avoid false warnings


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
            if not _is_whisper_cached(self._model_size):
                size_hint = _MODEL_SIZES.get(self._model_size, "unknown size")
                print(
                    f"[speech-analyser] Downloading Whisper '{self._model_size}' model "
                    f"({size_hint}) — this only happens once.",
                    file=sys.stderr,
                    flush=True,
                )
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file."""
        if audio_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        model = self._load()
        # word_timestamps=True yields per-word start/end/probability — the granularity
        # consumers need for precise within-window metrics (e.g. video-analyser's
        # per-scene WPM, pause and filler detection). Slightly slower than segment-only.
        raw_segments, info = model.transcribe(str(audio_path), word_timestamps=True)

        segments = []
        texts = []
        for seg in raw_segments:
            words = [
                Word(word=w.word, start=w.start, end=w.end, probability=w.probability)
                for w in (seg.words or [])
            ]
            segments.append(Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                avg_logprob=seg.avg_logprob,
                words=words,
            ))
            texts.append(seg.text.strip())

        return TranscriptionResult(
            text=" ".join(texts),
            segments=segments,
            language=info.language,
            duration=info.duration,
        )
