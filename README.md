# speech-analyser

Transcribes audio and video files and returns speech metrics: word count, speaking rate, filler word detection, silence ratio, and a quality score with natural-language insights. Optionally identifies individual speakers.

Part of the [analyser family](#the-analyser-family).

## Install

```bash
pip install speech-analyser
```

Requires Python 3.11+. Uses CPU by default; GPU (CUDA) is used automatically if available.

For speaker diarization, install the extra and set a Hugging Face token:

```bash
pip install "speech-analyser[diarization]"
export HF_TOKEN=hf_...
```

## Usage

### Python

```python
from speech_analyser import SpeechAnalyser

lens = SpeechAnalyser()             # model_size="base" by default
result = lens.analyse("recording.mp3")

m = result["speech_metrics"]
print(f"Duration:  {result['duration']:.1f}s")
print(f"Words:     {m['word_count']} ({m['speaking_rate_wpm']} wpm, {m['pace_category']})")
print(f"Quality:   {m['quality_score']}/100")
print(result["transcript"])
```

### CLI

```bash
# Human-readable summary
speech-analyser analyse recording.mp3

# Larger model for better accuracy
speech-analyser analyse lecture.wav --model small

# Machine-readable JSON
speech-analyser analyse recording.m4a --json

# Speaker diarization
speech-analyser analyse interview.mp3 --diarize

# Start the HTTP server
speech-analyser serve --port 8001
```

### HTTP API

```bash
curl -X POST http://localhost:8001/analyse \
  -F "file=@recording.mp3"
```

## Supported formats

Audio: `.mp3` `.wav` `.m4a` `.ogg` `.flac` `.aac` `.wma` `.opus`

Video: `.mp4` `.mov` `.avi` `.mkv` `.webm` — audio track is extracted automatically.

## Model sizes

| Model | Speed | Accuracy |
|---|---|---|
| `tiny` | fastest | lowest |
| `base` | fast | good (default) |
| `small` | medium | better |
| `medium` | slow | very good |
| `large-v3` | slowest | best |

Models download on first use (~75 MB for `base`, ~1.5 GB for `large-v3`).

## Output

```json
{
  "transcript": "Good morning everyone...",
  "language": "en",
  "duration": 62.4,
  "segments": [{"start": 0.0, "end": 3.2, "text": "Good morning everyone", "speaker": null}],
  "speech_metrics": {
    "word_count": 120,
    "speaking_rate_wpm": 115.4,
    "pace_category": "natural",
    "filler_word_count": 3,
    "filler_word_rate": 0.025,
    "filler_words_found": ["um", "basically"],
    "silence_ratio": 0.18,
    "actual_speaking_time": 51.2,
    "quality_score": 78,
    "quality_factors": {"clarity": 23, "depth": 18, "balance": 18, "pace": 19},
    "quality_ratings": {"clarity": "excellent", "depth": "good", "balance": "good", "pace": "good"},
    "insights": {
      "strengths": ["Very few filler words — speech is clear"],
      "observations": ["Speaking rate is slightly slow — aim for 130–170 wpm"]
    }
  },
  "diarization_available": false,
  "speakers": null,
  "talk_time": null,
  "file_path": "/path/to/recording.mp3",
  "file_size": 2048000
}
```

When diarization is enabled, `speakers` contains per-speaker word count, duration, and percentage; `talk_time.is_balanced` flags whether one speaker dominates.

## The analyser family

Low-level analysis tools. Each accepts files directly and returns structured JSON. Build your own UI or pipeline on top.

| Package | Handles |
|---|---|
| [speech-analyser](https://github.com/michael-borck/speech-analyser) | audio and video files — transcript and speech metrics |
| [video-analyser](https://github.com/michael-borck/video-analyser) | video files — frames, scenes, and visual quality |
| [document-analyser](https://github.com/michael-borck/document-analyser) | PDF, DOCX, PPTX, TXT — text and readability |
| [code-analyser](https://github.com/michael-borck/code-analyser) | source code — style, complexity, and quality metrics |
| [records-analyser](https://github.com/michael-borck/records-analyser) | CSV, Excel, SQLite, Parquet, JSON — data profiling |
| [auto-analyser](https://github.com/michael-borck/auto-analyser) | any file — detects format and routes to the right tool |

## Licence

MIT
