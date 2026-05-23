"""Capability manifest for the lens family (consumed by auto-analyser)."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    try:
        return version("speech-analyser")
    except PackageNotFoundError:
        return "0.0.0"


MANIFEST: dict = {
    "name": "speech-analyser",
    "version": _version(),
    "role": "analyser",
    "accepts": ["audio"],
    "extensions": [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"],
    "auto_routable": True,
    "produces": "AudioAnalysis",
}
