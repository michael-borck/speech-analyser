import re
from typing import Any
from .transcriber import TranscriptionResult

_FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically", "literally",
    "actually", "sort of", "kind of", "right", "okay", "so",
}


class SpeechAnalyzer:
    """Derives speech quality metrics from a TranscriptionResult."""

    def analyse(self, result: TranscriptionResult) -> dict[str, Any]:
        words = result.text.split() if result.text else []
        word_count = len(words)
        duration_minutes = result.duration / 60

        speaking_rate = round(word_count / duration_minutes, 1) if duration_minutes > 0 else 0.0

        text_lower = result.text.lower()
        filler_words_found = []
        filler_count = 0
        for fw in _FILLER_WORDS:
            pattern = r'\b' + re.escape(fw) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                filler_words_found.append(fw)
                filler_count += len(matches)
        filler_rate = round(filler_count / duration_minutes, 2) if duration_minutes > 0 else 0.0

        speaking_time = sum(s.end - s.start for s in result.segments)
        silence_ratio = round(
            1.0 - (speaking_time / result.duration), 3
        ) if result.duration > 0 else 0.0

        return {
            "word_count": word_count,
            "speaking_rate_wpm": speaking_rate,
            "filler_word_count": filler_count,
            "filler_word_rate": filler_rate,
            "filler_words_found": sorted(filler_words_found),
            "silence_ratio": silence_ratio,
            "actual_speaking_time": round(speaking_time, 1),
        }
