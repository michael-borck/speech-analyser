# Basic usage

`speech-analyser` transcribes audio and reports speech metrics (pace, filler words, silence, quality) for the analyser family.

## Install

```bash
pip install speech-analyser
```

Speaker diarization is optional: `pip install "speech-analyser[diarization]"` (also needs an `HF_TOKEN`).

## CLI

```bash
speech-analyser path/to/talk.mp3 --json
```

## Python

```python
from speech_analyser import SpeechAnalyser

analyser = SpeechAnalyser(model_size="base")
result = analyser.analyse("path/to/talk.mp3")  # returns a dict
print(result["speech_metrics"]["speaking_rate_wpm"])
```

## HTTP

```bash
speech-analyser serve
curl -F file=@path/to/talk.mp3 http://localhost:8001/analyse
```
