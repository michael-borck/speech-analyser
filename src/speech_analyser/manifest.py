"""Capability manifest for the lens family (consumed by auto-analyser)."""
from __future__ import annotations

from lens_contract import make_manifest

MANIFEST = make_manifest(
    name="speech-analyser",
    accepts=["audio"],
    extensions=[".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"],
    auto_routable=True,
    produces="AudioAnalysis",
)
