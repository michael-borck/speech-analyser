import re
from typing import Any

from .transcriber import TranscriptionResult

_SINGLE_WORD_FILLERS = {
    "um", "uh", "er", "ah", "hmm", "mm",
    "like", "so", "well", "actually", "basically",
    "literally", "honestly", "obviously", "right",
    "okay", "yeah", "anyway",
}

_MULTI_WORD_FILLERS = [
    "you know", "i mean", "kind of", "sort of", "you see",
    "i guess", "or something", "or whatever",
]


def _detect_fillers(text: str) -> tuple[int, list[str]]:
    """Return (total_count, sorted list of distinct filler expressions found)."""
    if not text:
        return 0, []

    lower = text.lower()
    tokens = lower.split()
    counts: dict[str, int] = {}

    # Build cleaned tokens once — used for both single-word and multi-word matching
    cleaned_tokens = [re.sub(r"^[^\w']+|[^\w']+$", "", tok) for tok in tokens]

    for clean in cleaned_tokens:
        if clean in _SINGLE_WORD_FILLERS:
            counts[clean] = counts.get(clean, 0) + 1

    # Use cleaned tokens so punctuation doesn't block multi-word phrase matching
    joined = " " + " ".join(cleaned_tokens) + " "
    for phrase in _MULTI_WORD_FILLERS:
        needle = " " + phrase + " "
        idx, n = 0, 0
        while (idx := joined.find(needle, idx)) != -1:
            n += 1
            idx += len(needle) - 1
        if n:
            counts[phrase] = counts.get(phrase, 0) + n

    total = sum(counts.values())
    return total, sorted(counts.keys())


def _pace_category(wpm: float) -> str:
    if wpm < 90:
        return "slow"
    if wpm > 200:
        return "fast"
    return "natural"


def _quality_score(
    filler_rate: float,
    avg_words_per_segment: float,
    wpm: float | None,
    speaker_percentages: list[float],
) -> tuple[int, dict[str, int], dict[str, str]]:
    # Clarity — inverse of filler rate (fraction 0-1)
    if filler_rate <= 0.02:
        clarity = 25
    elif filler_rate <= 0.05:
        clarity = 21
    elif filler_rate <= 0.08:
        clarity = 16
    elif filler_rate <= 0.12:
        clarity = 11
    else:
        clarity = 6

    # Depth — average words per Whisper segment
    w = avg_words_per_segment
    if 12 <= w <= 25:
        depth = 25
    elif 8 <= w <= 30:
        depth = 20
    elif 5 <= w <= 40:
        depth = 15
    elif w >= 3:
        depth = 10
    else:
        depth = 5

    # Balance — only meaningful with multiple speakers; neutral (18) otherwise
    if len(speaker_percentages) > 1:
        dominant = max(speaker_percentages)
        ideal = 100.0 / len(speaker_percentages)
        deviation = abs(dominant - ideal)
        if deviation <= 10:
            balance = 25
        elif deviation <= 20:
            balance = 20
        elif deviation <= 30:
            balance = 15
        elif deviation <= 45:
            balance = 10
        else:
            balance = 5
    else:
        balance = 18

    # Pace — natural conversational range: 130-170 wpm
    if wpm is None or wpm <= 0:
        pace = 18
    elif 130 <= wpm <= 170:
        pace = 25
    elif 110 <= wpm <= 190:
        pace = 20
    elif 90 <= wpm <= 210:
        pace = 15
    elif 60 <= wpm <= 240:
        pace = 10
    else:
        pace = 5

    score = clarity + depth + balance + pace

    def _rate(v: int) -> str:
        if v >= 22:
            return "excellent"
        if v >= 17:
            return "good"
        if v >= 12:
            return "fair"
        return "low"

    factors = {"clarity": clarity, "depth": depth, "balance": balance, "pace": pace}
    ratings = {k: _rate(v) for k, v in factors.items()}
    return score, factors, ratings


def _insights(
    filler_rate: float,
    avg_words_per_segment: float,
    wpm: float | None,
    quality_ratings: dict[str, str],
    speaker_data: list[dict],
) -> dict[str, list[str]]:
    strengths: list[str] = []
    observations: list[str] = []

    if quality_ratings["clarity"] == "excellent":
        strengths.append("Very few filler words — speech is clear")
    if quality_ratings["depth"] == "excellent":
        strengths.append("Turns are substantive (12–25 words on average)")
    if len(speaker_data) > 1 and quality_ratings["balance"] == "excellent":
        strengths.append("Speakers contribute roughly evenly")
    if quality_ratings["pace"] == "excellent":
        strengths.append("Speaking pace is in the natural conversational range (130–170 wpm)")

    if filler_rate > 0.08:
        observations.append(f"Filler words make up {filler_rate:.1%} of spoken words")
    if avg_words_per_segment < 5:
        observations.append("Very short turns — could indicate hesitation or rapid exchange")
    if avg_words_per_segment > 30:
        observations.append("Long turns — may be a lecture or monologue format")
    if wpm is not None and wpm < 90:
        observations.append(f"Slow pace ({wpm:.0f} wpm) — may include long pauses")
    if wpm is not None and wpm > 200:
        observations.append(f"Fast pace ({wpm:.0f} wpm) — speakers talking quickly")
    if len(speaker_data) > 1:
        dominant = max(speaker_data, key=lambda s: s["percentage"])
        if dominant["percentage"] > 70:
            observations.append(
                f"{dominant['id']} dominates with {dominant['percentage']:.0f}% of spoken words"
            )

    if not strengths:
        strengths.append("Transcript analysed successfully")

    return {"strengths": strengths, "observations": observations}


class SpeechAnalyzer:
    """Derives speech quality metrics from a TranscriptionResult."""

    def analyse(
        self,
        result: TranscriptionResult,
        speaker_data: list[dict] | None = None,
    ) -> dict[str, Any]:
        words = result.text.split() if result.text else []
        word_count = len(words)
        duration_minutes = result.duration / 60

        wpm = round(word_count / duration_minutes, 1) if duration_minutes > 0 else 0.0

        filler_count, filler_words_found = _detect_fillers(result.text)
        filler_rate = round(filler_count / word_count, 4) if word_count > 0 else 0.0

        speaking_time = sum(s.end - s.start for s in result.segments)
        silence_ratio = max(0.0, min(1.0, round(
            1.0 - (speaking_time / result.duration), 3
        ))) if result.duration > 0 else 0.0

        avg_words_per_segment = (
            sum(len(s.text.split()) for s in result.segments) / len(result.segments)
            if result.segments else 0.0
        )

        pace_cat = _pace_category(wpm) if wpm > 0 else "unknown"

        spk = speaker_data or []
        speaker_percentages = [s["percentage"] for s in spk]

        quality_score, quality_factors, quality_ratings = _quality_score(
            filler_rate=filler_rate,
            avg_words_per_segment=avg_words_per_segment,
            wpm=wpm if wpm > 0 else None,
            speaker_percentages=speaker_percentages,
        )

        insights = _insights(
            filler_rate=filler_rate,
            avg_words_per_segment=avg_words_per_segment,
            wpm=wpm if wpm > 0 else None,
            quality_ratings=quality_ratings,
            speaker_data=spk,
        )

        return {
            "word_count": word_count,
            "speaking_rate_wpm": wpm,
            "pace_category": pace_cat,
            "filler_word_count": filler_count,
            "filler_word_rate": filler_rate,
            "filler_words_found": filler_words_found,
            "silence_ratio": silence_ratio,
            "actual_speaking_time": round(speaking_time, 1),
            "quality_score": quality_score,
            "quality_factors": quality_factors,
            "quality_ratings": quality_ratings,
            "insights": insights,
        }
