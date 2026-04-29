from pathlib import Path
from typing import Any

from .transcriber import Transcriber
from .speech_analyzer import SpeechAnalyzer


class AudioLens:
    """Transcribes audio files and returns speech metrics.

    Args:
        model_size: Whisper model size. Options: tiny, base, small, medium, large-v3.
                    Larger = more accurate, slower. Default 'base' suits most cases.
    """

    def __init__(self, model_size: str = "base") -> None:
        self._transcriber = Transcriber(model_size=model_size)
        self._analyzer = SpeechAnalyzer()

    def analyse(self, file_path: Path | str) -> dict[str, Any]:
        """Analyse an audio file.

        Returns:
            dict with keys:
              success (bool)
              data (dict): transcript, language, duration, segments,
                           speech_metrics, file_path, file_size
              error (str): present only on failure
        """
        try:
            if isinstance(file_path, str):
                file_path = Path(file_path)

            if file_path.suffix.lower() not in self._transcriber.SUPPORTED_EXTENSIONS:
                return {
                    "success": False,
                    "error": (
                        f"Unsupported audio format: {file_path.suffix}. "
                        f"Supported: {', '.join(sorted(self._transcriber.SUPPORTED_EXTENSIONS))}"
                    ),
                    "data": {},
                }

            file_size = file_path.stat().st_size
            result = self._transcriber.transcribe(file_path)
            metrics = self._analyzer.analyse(result)

            return {
                "success": True,
                "data": {
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
                },
            }
        except Exception as e:
            return {"success": False, "error": repr(e), "data": {}}
