"""CLI entry point for speech-analyser.

Usage:
  speech-analyser recording.mp3
  speech-analyser recording.wav --model small
  speech-analyser recording.m4a --json
  speech-analyser interview.mp4 --diarize
  speech-analyser serve
  speech-analyser serve --port 8001 --host 0.0.0.0
"""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    import argparse

    from lens_contract import run_contract_subcommands

    from .manifest import MANIFEST

    # `serve` and `manifest` are the family's shared subcommands (lens-contract).
    if run_contract_subcommands(
        MANIFEST,
        app_path="speech_analyser.api:app",
        default_port=8001,
        env_prefix="SPEECH_ANALYSER",
    ):
        return

    parser = argparse.ArgumentParser(
        prog="speech-analyser",
        description="Audio transcription and speech analysis",
    )
    parser.add_argument("file", type=Path, help="Audio or video file to analyse")
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted(["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                        "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3"]),
        help="Whisper model size (default: SPEECH_ANALYSER_MODEL env var or 'base')",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Run speaker diarization (requires speech-analyser[diarization] and HF_TOKEN)",
    )
    _cmd_analyse(parser.parse_args())


def _cmd_analyse(args) -> None:
    from .speech_analyser import SpeechAnalyser
    from .exceptions import SpeechAnalyserError, ModelNotAvailableError

    model = args.model if args.model is not None else os.getenv("SPEECH_ANALYSER_MODEL", "base")
    diarize = args.diarize or os.getenv("SPEECH_ANALYSER_DIARIZE", "false").lower() == "true"
    lens = SpeechAnalyser(model_size=model)

    try:
        result = lens.analyse(args.file, diarize=diarize)
    except ModelNotAvailableError as e:
        if args.as_json:
            print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        else:
            print(f"Diarization unavailable: {e}", file=sys.stderr)
        sys.exit(2)
    except SpeechAnalyserError as e:
        if args.as_json:
            print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        print(json.dumps(result, indent=2))
        return

    print(f"Language:      {result['language']}")
    print(f"Duration:      {result['duration']:.1f}s")
    print(f"Words:         {result['speech_metrics']['word_count']}")
    print(f"Speaking rate: {result['speech_metrics']['speaking_rate_wpm']} wpm "
          f"({result['speech_metrics']['pace_category']})")
    print(f"Filler words:  {result['speech_metrics']['filler_word_count']} "
          f"({result['speech_metrics']['filler_word_rate']:.1%})")
    print(f"Silence ratio: {result['speech_metrics']['silence_ratio']:.1%}")
    print(f"Quality score: {result['speech_metrics']['quality_score']}/100")

    insights = result["speech_metrics"]["insights"]
    if insights["strengths"]:
        print("\nStrengths:")
        for s in insights["strengths"]:
            print(f"  • {s}")
    if insights["observations"]:
        print("\nObservations:")
        for o in insights["observations"]:
            print(f"  • {o}")

    if result.get("speakers"):
        print(f"\nSpeakers ({len(result['speakers'])}):")
        for spk in result["speakers"]:
            print(f"  {spk['id']}: {spk['percentage']:.0f}% ({spk['word_count']} words)")

    print("\nTranscript:")
    print(result["transcript"])


if __name__ == "__main__":
    main()
