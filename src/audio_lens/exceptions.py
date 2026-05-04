class AudioLensError(Exception):
    """Raised when audio-lens cannot analyse a file."""


class ModelNotAvailableError(AudioLensError):
    """Raised when a required model is not installed or not yet downloaded.

    The message includes instructions for resolving the issue.
    Callers should treat this as a recoverable condition (e.g. HTTP 503).
    """
