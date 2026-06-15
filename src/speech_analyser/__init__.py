from importlib.metadata import version as _v
from pathlib import Path
from typing import Any

from .exceptions import ModelNotAvailableError, SpeechAnalyserError
from .manifest import MANIFEST
from .schemas import AudioAnalysis
from .speech_analyser import SpeechAnalyser

# Canonical family alias for the result model. The analyser declares
# ``produces="AudioAnalysis"``; ``SpeechAnalysis`` is the uniform family name.
SpeechAnalysis = AudioAnalysis

__version__ = _v("speech-analyser")
del _v


def analyse(file_path: str | Path, *, diarize: bool = False) -> dict[str, Any]:
    """Analyse an audio file (module-level convenience wrapper).

    Constructs a :class:`SpeechAnalyser` and delegates to its ``.analyse``.
    The whisper model is loaded lazily by the analyser when called, so
    importing this package never triggers a model download.
    """
    return SpeechAnalyser().analyse(Path(file_path), diarize=diarize)


__all__ = [
    "SpeechAnalyser",
    "AudioAnalysis",
    "SpeechAnalysis",
    "analyse",
    "MANIFEST",
    "__version__",
    "SpeechAnalyserError",
    "ModelNotAvailableError",
]
