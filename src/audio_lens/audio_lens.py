from pathlib import Path
from typing import Any

from .exceptions import AudioLensError
from .speech_analyzer import SpeechAnalyzer
from .transcriber import Transcriber


class AudioLens:
    """Transcribes audio files and returns speech metrics.

    Args:
        model_size: Whisper model size. Options: tiny, base, small, medium, large-v3.
    """

    def __init__(self, model_size: str = "base") -> None:
        self._transcriber = Transcriber(model_size=model_size)
        self._analyzer = SpeechAnalyzer()

    def analyse(self, file_path: Path | str) -> dict[str, Any]:
        """Analyse an audio file. Returns the analysis dict directly.

        Raises:
            AudioLensError: if the file is missing, format is unsupported,
                            or transcription fails.
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
            metrics = self._analyzer.analyse(result)

            return {
                "transcript": result.text,
                "language": result.language,
                "duration": result.duration,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in result.segments
                ],
                "speech_metrics": metrics,
                "file_path": str(file_path),
                "file_size": file_size,
            }
        except AudioLensError:
            raise
        except Exception as e:
            raise AudioLensError(repr(e)) from e
