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

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        _main_serve(sys.argv[2:])
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


def _main_serve(argv: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="speech-analyser serve", description="Start the HTTP server")
    parser.add_argument("--port", type=int, default=int(os.getenv("SPEECH_ANALYSER_PORT", "8001")))
    parser.add_argument("--host", default=os.getenv("SPEECH_ANALYSER_HOST", "127.0.0.1"))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    _cmd_serve(parser.parse_args(argv))


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


def _cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run(
        "speech_analyser.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
