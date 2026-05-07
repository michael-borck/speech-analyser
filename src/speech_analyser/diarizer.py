import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import AudioLensError, ModelNotAvailableError

_MODEL_ID = "pyannote/speaker-diarization-3.1"


@dataclass
class DiarizationTurn:
    start: float
    end: float
    speaker: str


class Diarizer:
    """Speaker diarization via pyannote.audio.

    Requires:
      - pip install 'audio-lens[diarization]'  (installs pyannote.audio)
      - HF_TOKEN env var with access granted to pyannote/speaker-diarization-3.1
        Get a token: https://huggingface.co/settings/tokens
        Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
    """

    def __init__(self) -> None:
        self._pipeline: Any = None

    def _import_pipeline(self):
        """Import Pipeline from pyannote.audio. Separate method for testability."""
        from pyannote.audio import Pipeline
        return Pipeline

    def _resolve_token(self) -> str | None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            val = token_file.read_text().strip()
            return val or None
        return None

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        try:
            Pipeline = self._import_pipeline()
        except ImportError as e:
            raise ModelNotAvailableError(
                "pyannote.audio is not installed. "
                "Install with: pip install 'audio-lens[diarization]'"
            ) from e

        token = self._resolve_token()

        # Token only required when the model isn't already cached. Once cached
        # (e.g. bundled inside an installer + HF_HUB_OFFLINE=1), Pipeline can
        # load without contacting the gated repo.
        is_cached = False
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(_MODEL_ID, "config.yaml")
            is_cached = isinstance(cached, str)
        except Exception:
            pass

        if not token and not is_cached:
            raise ModelNotAvailableError(
                "No Hugging Face token found. Diarization requires a token with access to "
                f"{_MODEL_ID}. Set the HF_TOKEN environment variable. "
                "Get a free token at https://huggingface.co/settings/tokens and accept "
                f"the model terms at https://huggingface.co/{_MODEL_ID}"
            )

        if not is_cached:
            print(
                f"[speech-analyser] Downloading diarization model '{_MODEL_ID}' "
                f"(~2 GB, first use only)...",
                file=sys.stderr,
                flush=True,
            )

        try:
            self._pipeline = Pipeline.from_pretrained(_MODEL_ID, token=token)
        except Exception as e:
            raise ModelNotAvailableError(
                f"Could not load diarization model: {e}. "
                "Ensure you have accepted the model terms at "
                f"https://huggingface.co/{_MODEL_ID}"
            ) from e

        return self._pipeline

    def diarize(
        self, audio_path: Path, num_speakers: int | None = None
    ) -> list[DiarizationTurn]:
        """Run speaker diarization on an audio file.

        Returns list of DiarizationTurn sorted by start time.

        Raises:
            ModelNotAvailableError: pyannote.audio not installed or no HF token.
            AudioLensError: diarization pipeline failed.
        """
        pipeline = self._load()

        kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        try:
            output = pipeline(str(audio_path), **kwargs)
        except Exception as e:
            raise AudioLensError(f"Diarization failed: {e}") from e

        # pyannote 4.x returns DiarizeOutput wrapping the Annotation as
        # .speaker_diarization; pyannote 3.x returns the Annotation directly.
        # Type-name check rather than getattr so test mocks (MagicMock auto-
        # creates any attribute) fall through to the legacy path.
        annotation = output.speaker_diarization if type(output).__name__ == "DiarizeOutput" else output

        turns = [
            DiarizationTurn(
                start=round(turn.start, 3),
                end=round(turn.end, 3),
                speaker=speaker,
            )
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: t.start)
        return turns
