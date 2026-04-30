# audio-lens

Audio transcription and speech analysis for the [prism lens family](https://github.com/michael-borck/prism).

Transcribes audio files using [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) and returns speech metrics: word count, speaking rate, filler word detection, and silence ratio.

## Install

```bash
pip install audio-lens
```

Requires Python 3.11+. Uses CPU by default; GPU (CUDA) is used automatically if available.

## Usage

### Python

```python
from audio_lens import AudioLens

lens = AudioLens()  # model_size="base" by default
result = lens.analyse("recording.mp3")

if result["success"]:
    data = result["data"]
    print(f"Language: {data['language']}")
    print(f"Duration: {data['duration']:.1f}s")
    print(f"Words:    {data['speech_metrics']['word_count']}")
    print(f"WPM:      {data['speech_metrics']['speaking_rate_wpm']}")
    print(data["transcript"])
```

### CLI

```bash
# Human-readable output
audio-lens analyse recording.mp3

# Choose a larger model for better accuracy
audio-lens analyse lecture.wav --model small

# Machine-readable JSON (for piping to prism or other tools)
audio-lens analyse recording.m4a --json
```

## Supported formats

`.mp3` `.wav` `.m4a` `.ogg` `.flac` `.aac` `.wma` `.opus`

## Model sizes

| Model     | Speed   | Accuracy |
|-----------|---------|----------|
| `tiny`    | fastest | lowest   |
| `base`    | fast    | good (default) |
| `small`   | medium  | better   |
| `medium`  | slow    | very good |
| `large-v3`| slowest | best     |

Models are downloaded on first use (~75MB for `base`, ~1.5GB for `large-v3`).

## Output shape

```json
{
  "success": true,
  "data": {
    "transcript": "Hello world...",
    "language": "en",
    "duration": 62.4,
    "segments": [{"start": 0.0, "end": 3.2, "text": "Hello world"}],
    "speech_metrics": {
      "word_count": 120,
      "speaking_rate_wpm": 115.4,
      "filler_word_count": 3,
      "filler_word_rate": 2.88,
      "filler_words_found": ["um", "basically"],
      "silence_ratio": 0.18,
      "actual_speaking_time": 51.2
    },
    "file_path": "/path/to/recording.mp3",
    "file_size": 2048000
  }
}
```

## Part of the prism family

audio-lens is one lens in a family of analysis tools routed by [prism](https://github.com/michael-borck/prism):

- [document-lens](https://github.com/michael-borck/document-lens) — PDFs, DOCX, PPTX, Markdown
- [data-lens](https://github.com/michael-borck/data-lens) — CSV, XLSX, SQLite, JSON, YAML
- [audio-lens](https://github.com/michael-borck/audio-lens) — audio transcription and speech metrics
- [prism](https://github.com/michael-borck/prism) — meta-router: detects format and calls the right lens

## License

MIT
