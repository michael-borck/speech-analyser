"""CLI entry point for audio-lens.

Usage:
  audiolens analyse recording.mp3
  audiolens analyse recording.wav --model small
  audiolens analyse recording.m4a --json
  audiolens serve
  audiolens serve --port 8001 --host 0.0.0.0
"""

import json
import os
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
        default=None,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: AUDIO_LENS_MODEL env var or 'base')",
    )
    analyse.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output raw JSON",
    )

    serve = sub.add_parser("serve", help="Start the FastAPI HTTP server")
    serve.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("AUDIO_LENS_PORT", "8001")),
        help="Port to listen on (default: AUDIO_LENS_PORT or 8001)",
    )
    serve.add_argument(
        "--host",
        default=os.getenv("AUDIO_LENS_HOST", "127.0.0.1"),
        help="Host to bind (default: AUDIO_LENS_HOST or 127.0.0.1)",
    )
    serve.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)",
    )

    args = parser.parse_args()

    if args.command == "analyse":
        _cmd_analyse(args)
    elif args.command == "serve":
        _cmd_serve(args)


def _cmd_analyse(args) -> None:
    from .audio_lens import AudioLens
    from .exceptions import AudioLensError

    model = args.model or os.getenv("AUDIO_LENS_MODEL", "base")
    lens = AudioLens(model_size=model)

    try:
        result = lens.analyse(args.file)
    except AudioLensError as e:
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
    print(f"Speaking rate: {result['speech_metrics']['speaking_rate_wpm']} wpm")
    print(f"Filler words:  {result['speech_metrics']['filler_word_count']}")
    print(f"Silence ratio: {result['speech_metrics']['silence_ratio']:.1%}")
    print()
    print("Transcript:")
    print(result["transcript"])


def _cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run(
        "audio_lens.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
