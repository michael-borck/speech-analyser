"""CLI entry point for audio-lens.

Usage:
  audio-lens analyse recording.mp3
  audio-lens analyse recording.wav --model small
  audio-lens analyse recording.m4a --json
"""

import json
import sys
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="audio-lens",
        description="Audio transcription and speech analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyse = sub.add_parser("analyse", help="Analyse an audio file")
    analyse.add_argument("file", type=Path, help="Path to audio file")
    analyse.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    analyse.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output raw JSON",
    )

    args = parser.parse_args()

    if args.command == "analyse":
        from .audio_lens import AudioLens

        lens = AudioLens(model_size=args.model)
        result = lens.analyse(args.file)

        if args.as_json:
            print(json.dumps(result, indent=2))
            if not result["success"]:
                sys.exit(1)
        else:
            if not result["success"]:
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)

            data = result["data"]
            print(f"Language:      {data['language']}")
            print(f"Duration:      {data['duration']:.1f}s")
            print(f"Words:         {data['speech_metrics']['word_count']}")
            print(f"Speaking rate: {data['speech_metrics']['speaking_rate_wpm']} wpm")
            print(f"Filler words:  {data['speech_metrics']['filler_word_count']}")
            print(f"Silence ratio: {data['speech_metrics']['silence_ratio']:.1%}")
            print()
            print("Transcript:")
            print(data["transcript"])
