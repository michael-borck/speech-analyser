import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from lens_contract import add_contract_routes, add_cors, add_rate_limit

from .speech_analyser import SpeechAnalyser
from .exceptions import SpeechAnalyserError, ModelNotAvailableError
from .schemas import AudioAnalysis
from .transcriber import SUPPORTED_MODELS
from .manifest import MANIFEST

# Cache SpeechAnalyser instances by model size — model loading is expensive
_lens_cache: dict[str, SpeechAnalyser] = {}


def _get_lens(model_size: str) -> SpeechAnalyser:
    if model_size not in _lens_cache:
        _lens_cache[model_size] = SpeechAnalyser(model_size=model_size)
    return _lens_cache[model_size]


# MANIFEST["version"] is the installed package version (resolved by lens-contract),
# so the FastAPI service version always matches the package — no manual sync.
app = FastAPI(
    title="speech-analyser",
    description="Audio transcription and speech analysis API",
    version=MANIFEST["version"],
    docs_url="/docs",
    redoc_url="/redoc",
)

# GET /health and GET /manifest (the family contract, via lens-contract).
add_contract_routes(app, MANIFEST)
# CORS — env-driven: SPEECH_ANALYSER_MODE=desktop (Electron) or SPEECH_ANALYSER_ALLOWED_ORIGINS.
add_cors(app, env_prefix="SPEECH_ANALYSER")
# Opt-in rate limiting — SPEECH_ANALYSER_RATE_LIMIT_ENABLED=true (needs the [ratelimit] extra).
add_rate_limit(app, env_prefix="SPEECH_ANALYSER")


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "speech-analyser",
        "version": MANIFEST["version"],
        "status": "running",
        "endpoints": {"health": "/health", "analyse": "/analyse"},
    }


@app.post("/analyse", response_model=AudioAnalysis)
async def analyse(
    file: UploadFile = File(..., description="Audio file to analyse"),
    model: str | None = Form(default=None, description="Whisper model size (optional)"),
    diarize: bool = Form(
        default=False,
        description="Run speaker diarization (requires speech-analyser[diarization] and HF_TOKEN)",
    ),
) -> AudioAnalysis:
    model_size = model if model is not None else os.getenv("SPEECH_ANALYSER_MODEL", "base")

    if model_size not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model_size}'. Must be one of: {', '.join(sorted(SUPPORTED_MODELS))}",
        )

    # Respect env-var default for diarize if not explicitly passed
    if not diarize:
        diarize = os.getenv("SPEECH_ANALYSER_DIARIZE", "false").lower() == "true"

    # Audio formats are sniffed from content by ffmpeg, but keep the upload's suffix
    # (defaulting to .wav) so the analyser's extension check sees the real format.
    suffix = Path(file.filename or "upload").suffix or ".wav"
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        data = _get_lens(model_size).analyse(tmp_path, diarize=diarize)
        return AudioAnalysis(**data)
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SpeechAnalyserError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
