from pydantic import BaseModel


class SpeechMetrics(BaseModel):
    word_count: int
    speaking_rate_wpm: float
    filler_word_count: int
    filler_word_rate: float
    filler_words_found: list[str]
    silence_ratio: float
    actual_speaking_time: float


class AudioAnalysis(BaseModel):
    transcript: str
    language: str
    duration: float
    segments: list[dict]
    speech_metrics: SpeechMetrics
    file_path: str
    file_size: int


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
