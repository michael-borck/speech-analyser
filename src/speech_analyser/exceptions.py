class SpeechAnalyserError(Exception):
    """Raised when speech-analyser cannot analyse a file."""


class ModelNotAvailableError(SpeechAnalyserError):
    """Raised when a required model is not installed or not yet downloaded.

    The message includes instructions for resolving the issue.
    Callers should treat this as a recoverable condition (e.g. HTTP 503).
    """
