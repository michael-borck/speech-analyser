from typing import Any

from pydantic import BaseModel


class QualityFactors(BaseModel):
    clarity: int
    depth: int
    balance: int
    pace: int


class QualityRatings(BaseModel):
    clarity: str
    depth: str
    balance: str
    pace: str


class Insights(BaseModel):
    strengths: list[str]
    observations: list[str]


class SpeechMetrics(BaseModel):
    word_count: int
    speaking_rate_wpm: float
    pace_category: str
    filler_word_count: int
    filler_word_rate: float
    filler_words_found: list[str]
    silence_ratio: float
    actual_speaking_time: float
    quality_score: int
    quality_factors: QualityFactors
    quality_ratings: QualityRatings
    insights: Insights


class SpeakerInfo(BaseModel):
    id: str
    word_count: int
    duration_seconds: float
    percentage: float


class TalkTime(BaseModel):
    is_balanced: bool
    dominant_speaker: str | None = None


class AudioAnalysis(BaseModel):
    transcript: str
    language: str
    duration: float
    segments: list[dict[str, Any]]
    speech_metrics: SpeechMetrics
    diarization_available: bool
    speakers: list[SpeakerInfo] | None = None
    talk_time: TalkTime | None = None
    file_path: str
    file_size: int


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
