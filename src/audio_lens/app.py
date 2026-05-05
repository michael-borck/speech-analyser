import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .audio_lens import AudioLens
from .exceptions import AudioLensError, ModelNotAvailableError
from .schemas import AudioAnalysis, HealthResponse

_VERSION = "0.1.0"
_START_TIME = time.time()
_VALID_MODELS = {"tiny", "base", "small", "medium", "large-v3"}

# Cache AudioLens instances by model size — model loading is expensive
_lens_cache: dict[str, AudioLens] = {}


def _get_lens(model_size: str) -> AudioLens:
    if model_size not in _lens_cache:
        _lens_cache[model_size] = AudioLens(model_size=model_size)
    return _lens_cache[model_size]


app = FastAPI(
    title="audio-lens",
    description="Audio transcription and speech analysis API",
    version=_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — desktop mode allows any localhost origin (for Electron)
if os.getenv("AUDIO_LENS_MODE") == "desktop":
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r"^(https?://localhost(:\d+)?"
            r"|https?://127\.0\.0\.1(:\d+)?"
            r"|file://.*"
            r"|null)$"
        ),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    _origins = os.getenv(
        "AUDIO_LENS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _origins],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

# Optional rate limiting — off by default, enable with AUDIO_LENS_RATE_LIMIT_ENABLED=true
if os.getenv("AUDIO_LENS_RATE_LIMIT_ENABLED", "false").lower() == "true":
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    _limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "audio-lens",
        "version": _VERSION,
        "status": "running",
        "endpoints": {"health": "/health", "analyse": "/analyse"},
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=_VERSION,
        uptime=round(time.time() - _START_TIME, 1),
    )


@app.post("/analyse", response_model=AudioAnalysis)
async def analyse(
    file: UploadFile = File(..., description="Audio file to analyse"),
    model: str | None = Form(default=None, description="Whisper model size (optional)"),
    diarize: bool = Form(
        default=False,
        description="Run speaker diarization (requires audio-lens[diarization] and HF_TOKEN)",
    ),
) -> AudioAnalysis:
    model_size = model if model is not None else os.getenv("AUDIO_LENS_MODEL", "base")

    if model_size not in _VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model_size}'. Must be one of: {', '.join(sorted(_VALID_MODELS))}",
        )

    # Respect env-var default for diarize if not explicitly passed
    if not diarize:
        diarize = os.getenv("AUDIO_LENS_DIARIZE", "false").lower() == "true"

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
    except AudioLensError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
