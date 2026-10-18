import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


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
    speaker: str | None = None  # filled in by SpeechAnalyser when diarizing


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

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus",
    }

    # Video containers whose audio track is extracted with ffmpeg before
    # transcription (see extract_audio_to_wav). Faster-Whisper's PyAV decode
    # handles some of these natively, but the diarizer's decoder is less
    # forgiving, so video always goes through a canonical 16 kHz mono WAV —
    # the same normalisation Debrief's sidecar applies before /analyse.
    VIDEO_EXTENSIONS: ClassVar[set[str]] = {
        ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi",
    }

    # Every suffix analyse()/transcribe() will accept.
    ALL_EXTENSIONS: ClassVar[set[str]] = SUPPORTED_EXTENSIONS | VIDEO_EXTENSIONS

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

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        vad_filter: bool = False,
    ) -> TranscriptionResult:
        """Transcribe an audio or video file.

        Video containers (VIDEO_EXTENSIONS) have their audio extracted to a
        temporary 16 kHz mono WAV via ffmpeg first. `language` forces a
        language code (e.g. "en") instead of auto-detection; `vad_filter`
        enables faster-whisper's voice-activity filter, which skips silence.
        Both default to the previous behaviour.
        """
        suffix = audio_path.suffix.lower()
        if suffix not in self.ALL_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {', '.join(sorted(self.ALL_EXTENSIONS))}"
            )

        if suffix in self.VIDEO_EXTENSIONS:
            with extracted_audio(audio_path) as wav_path:
                return self._transcribe_audio(wav_path, language, vad_filter)
        return self._transcribe_audio(audio_path, language, vad_filter)

    def _transcribe_audio(
        self, audio_path: Path, language: str | None, vad_filter: bool
    ) -> TranscriptionResult:
        model = self._load()
        # word_timestamps=True yields per-word start/end/probability — the granularity
        # consumers need for precise within-window metrics (e.g. video-analyser's
        # per-scene WPM, pause and filler detection). Slightly slower than segment-only.
        raw_segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=language,
            vad_filter=vad_filter,
        )

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


def is_video_container(path: Path) -> bool:
    """True when `path` is a video container needing audio extraction."""
    return path.suffix.lower() in Transcriber.VIDEO_EXTENSIONS


def extract_audio_to_wav(source: Path, dest: Path) -> Path:
    """Extract `source`'s audio track to a 16 kHz mono WAV at `dest` (ffmpeg).

    Raises ValueError with an actionable message when ffmpeg is missing or
    the file has no decodable audio. Only video inputs take this path, so
    plain-audio users never need ffmpeg installed.
    """
    if shutil.which("ffmpeg") is None:
        raise ValueError(
            "ffmpeg is required to read video files but was not found on PATH. "
            "Install it, e.g.: brew install ffmpeg"
        )
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000", str(dest),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"ffmpeg could not extract audio from {source}: {detail}")
    return dest


@contextmanager
def extracted_audio(source: Path) -> Iterator[Path]:
    """Yield a temporary WAV holding `source`'s audio; delete it on exit."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        yield extract_audio_to_wav(source, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
