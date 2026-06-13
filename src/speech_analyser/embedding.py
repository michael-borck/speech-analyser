"""Transcript embedding via the family's shared helper (lens-embed).

A single pinned model across the family means this vector is comparable to
other members' vectors (e.g. a video narration vs a report) — the basis for
cross-artefact and cohort-distinctiveness signals downstream. Opt-in and
degradable: install the [embeddings] extra to populate it; without it (or on
any failure) this returns None.
"""

from __future__ import annotations


def embed_document(text: str) -> list[float] | None:
    """Pooled, L2-normalised vector, or None if embeddings are off."""
    if not text or not text.strip():
        return None
    try:
        from lens_embed import backend_available, embed_long_text
    except ImportError:
        return None
    if not backend_available("text"):
        return None
    try:
        return embed_long_text(text)
    except Exception:
        return None
