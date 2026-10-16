"""CLI entry point for speech-analyser.

Usage:
  speech-analyser recording.mp3
  speech-analyser recording.wav --model small
  speech-analyser recording.m4a --json
  speech-analyser interview.mp4 --diarize
  speech-analyser lecture.mp4 -o lecture.txt --srt --timestamps
  speech-analyser podcast.mp3 --language en --vad
  speech-analyser serve
  speech-analyser serve --port 8001 --host 0.0.0.0

Video files (mp4/mov/mkv/webm/…) are supported: their audio track is
extracted with ffmpeg (must be on PATH) before transcription.
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
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Also write the transcript to this text file",
    )
    parser.add_argument(
        "--timestamps", action="store_true",
        help="Prefix each transcript line with [HH:MM:SS.mmm] (report and -o file)",
    )
    parser.add_argument(
        "--srt", action="store_true",
        help="Also write a .srt subtitle file (next to --output, or next to the input)",
    )
    parser.add_argument(
        "--language", default=None,
        help="Force a language code, e.g. en, es, fr (default: auto-detect)",
    )
    parser.add_argument(
        "--vad", action="store_true", dest="vad_filter",
        help="Skip silence with faster-whisper's voice-activity filter",
    )
    _cmd_analyse(parser.parse_args())


def _cmd_analyse(args) -> None:
    from .speech_analyser import SpeechAnalyser
    from .exceptions import SpeechAnalyserError, ModelNotAvailableError

    model = args.model if args.model is not None else os.getenv("SPEECH_ANALYSER_MODEL", "base")
    diarize = args.diarize or os.getenv("SPEECH_ANALYSER_DIARIZE", "false").lower() == "true"
    lens = SpeechAnalyser(model_size=model)

    try:
        result = lens.analyse(
            args.file,
            diarize=diarize,
            language=args.language,
            vad_filter=args.vad_filter,
        )
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

    _write_outputs(args, result)

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
    if args.timestamps:
        print(render_transcript(result["segments"], timestamps=True))
    else:
        print(result["transcript"])



# --- Transcript file rendering (ported from the standalone transcribe.py) ---

def format_timestamp(seconds: float) -> str:
    """Seconds -> "HH:MM:SS.mmm"."""
    clamped = max(0.0, float(seconds))
    h = int(clamped // 3600)
    m = int((clamped % 3600) // 60)
    sec = clamped % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def format_srt_timestamp(seconds: float) -> str:
    """Seconds -> SRT "HH:MM:SS,mmm"."""
    return format_timestamp(seconds).replace(".", ",")


def render_transcript(segments: list[dict], timestamps: bool = False) -> str:
    """Render analyse() segments as plain (or timestamp-prefixed) text."""
    lines: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if timestamps:
            prefix = f"[{format_timestamp(seg.get('start', 0.0))}] "
            if speaker:
                prefix += f"{speaker}: "
            lines.append(f"{prefix}{text}")
        else:
            lines.append(f"{speaker}: {text}" if speaker else text)
    return "\n".join(lines) + ("\n" if lines else "")


def render_srt(segments: list[dict]) -> str:
    """Render analyse() segments as SubRip subtitles."""
    cues: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        body = f"{speaker}: {text}" if speaker else text
        cues.append(
            f"{len(cues) + 1}\n"
            f"{format_srt_timestamp(seg.get('start', 0.0))} --> "
            f"{format_srt_timestamp(seg.get('end', 0.0))}\n"
            f"{body}\n"
        )
    return "\n".join(cues) + ("\n" if cues else "")


def _write_outputs(args, result: dict) -> None:
    """Write -o transcript and/or --srt files. Confirmations go to stderr
    so --json stdout stays machine-readable."""
    written: list[Path] = []
    if args.output is not None:
        args.output.write_text(
            render_transcript(result["segments"], timestamps=args.timestamps),
            encoding="utf-8",
        )
        written.append(args.output)
    if args.srt:
        base = args.output if args.output is not None else args.file
        srt_path = base.with_suffix(".srt")
        srt_path.write_text(render_srt(result["segments"]), encoding="utf-8")
        written.append(srt_path)
    for path in written:
        print(f"Written: {path}", file=sys.stderr)

if __name__ == "__main__":
    main()
