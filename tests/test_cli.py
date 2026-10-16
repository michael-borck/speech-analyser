"""Tests for the CLI transcript-file outputs and video-container support.

The -o/--timestamps/--srt/--language/--vad options and video handling were
ported from a standalone transcribe.py script. Model-dependent paths are
faked; the rendering and routing logic run for real.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from speech_analyser import SpeechAnalyser
from speech_analyser.cli import (
    _write_outputs,
    format_srt_timestamp,
    format_timestamp,
    render_srt,
    render_transcript,
)
from speech_analyser.exceptions import SpeechAnalyserError
from speech_analyser.transcriber import (
    Transcriber,
    extract_audio_to_wav,
    is_video_container,
)

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "Hello there.", "speaker": "SPEAKER_00", "words": []},
    {"start": 3.0, "end": 4.25, "text": "General Kenobi.", "speaker": "SPEAKER_01", "words": []},
    {"start": 5.0, "end": 5.5, "text": "  ", "speaker": None, "words": []},
]


class TestFormatting:
    def test_timestamp_format(self):
        assert format_timestamp(0) == "00:00:00.000"
        assert format_timestamp(3661.5) == "01:01:01.500"
        assert format_timestamp(-3) == "00:00:00.000"

    def test_srt_timestamp_uses_comma(self):
        assert format_srt_timestamp(62.25) == "00:01:02,250"

    def test_render_transcript_plain(self):
        out = render_transcript(SEGMENTS)
        assert out == "SPEAKER_00: Hello there.\nSPEAKER_01: General Kenobi.\n"

    def test_render_transcript_without_speakers(self):
        segs = [{**s, "speaker": None} for s in SEGMENTS]
        assert render_transcript(segs) == "Hello there.\nGeneral Kenobi.\n"

    def test_render_transcript_timestamped(self):
        out = render_transcript(SEGMENTS, timestamps=True)
        assert "[00:00:00.000] SPEAKER_00: Hello there." in out
        assert "[00:00:03.000] SPEAKER_01: General Kenobi." in out

    def test_render_srt_skips_blank_and_numbers_from_one(self):
        out = render_srt(SEGMENTS)
        assert out.startswith("1\n00:00:00,000 --> 00:00:02,500\nSPEAKER_00: Hello there.\n")
        assert "\n2\n00:00:03,000 --> 00:00:04,250\n" in out


class TestWriteOutputs:
    def _args(self, tmp_path: Path, **kw):
        base = {"output": None, "srt": False, "timestamps": False, "file": tmp_path / "in.mp4"}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_output_file_written(self, tmp_path: Path):
        out = tmp_path / "notes.txt"
        _write_outputs(self._args(tmp_path, output=out), {"segments": SEGMENTS})
        assert "Hello there." in out.read_text(encoding="utf-8")

    def test_srt_lands_next_to_output(self, tmp_path: Path):
        out = tmp_path / "notes.txt"
        _write_outputs(self._args(tmp_path, output=out, srt=True), {"segments": SEGMENTS})
        assert (tmp_path / "notes.srt").exists()

    def test_srt_without_output_lands_next_to_input(self, tmp_path: Path):
        _write_outputs(self._args(tmp_path, srt=True), {"segments": SEGMENTS})
        assert (tmp_path / "in.srt").exists()

    def test_no_flags_writes_nothing(self, tmp_path: Path):
        _write_outputs(self._args(tmp_path), {"segments": SEGMENTS})
        assert list(tmp_path.iterdir()) == []


class TestVideoSupport:
    def test_video_extensions_classified(self):
        assert is_video_container(Path("a.mp4"))
        assert is_video_container(Path("a.MOV"))
        assert not is_video_container(Path("a.wav"))
        assert ".mp4" in Transcriber.ALL_EXTENSIONS

    def test_missing_ffmpeg_gives_actionable_error(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "speech_analyser.transcriber.shutil.which", lambda _: None
        )
        with pytest.raises(ValueError, match="ffmpeg is required"):
            extract_audio_to_wav(tmp_path / "x.mp4", tmp_path / "x.wav")

    def test_analyse_accepts_mp4_suffix(self, tmp_path: Path, monkeypatch):
        # An .mp4 must get PAST the format gate; with ffmpeg "missing" it
        # then fails on extraction, proving the video path was taken.
        monkeypatch.setattr(
            "speech_analyser.transcriber.shutil.which", lambda _: None
        )
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"not really a video")
        with pytest.raises(SpeechAnalyserError, match="ffmpeg is required"):
            SpeechAnalyser().analyse(p)

    def test_unknown_suffix_still_unsupported(self, tmp_path: Path):
        p = tmp_path / "file.xyz"
        p.write_bytes(b"not audio")
        with pytest.raises(SpeechAnalyserError, match="Unsupported"):
            SpeechAnalyser().analyse(p)


class TestCliWiring:
    def test_cmd_analyse_passes_options_and_writes_files(self, tmp_path: Path, monkeypatch, capsys):
        captured = {}

        class FakeAnalyser:
            def __init__(self, model_size="base"):
                captured["model_size"] = model_size

            def analyse(self, file_path, diarize=False, language=None, vad_filter=False):
                captured.update(
                    file_path=file_path, diarize=diarize,
                    language=language, vad_filter=vad_filter,
                )
                return {
                    "transcript": "Hello there. General Kenobi.",
                    "language": "en",
                    "duration": 5.5,
                    "segments": SEGMENTS,
                    "speech_metrics": {
                        "word_count": 4, "speaking_rate_wpm": 100,
                        "pace_category": "normal", "filler_word_count": 0,
                        "filler_word_rate": 0.0, "silence_ratio": 0.1,
                        "quality_score": 90,
                        "insights": {"strengths": [], "observations": []},
                    },
                    "speakers": None,
                }

        monkeypatch.setattr(
            "speech_analyser.speech_analyser.SpeechAnalyser", FakeAnalyser
        )
        from speech_analyser.cli import _cmd_analyse

        out = tmp_path / "out.txt"
        _cmd_analyse(SimpleNamespace(
            file=tmp_path / "in.mp4", model="small", diarize=True,
            as_json=True, output=out, srt=True, timestamps=True,
            language="en", vad_filter=True,
        ))

        assert captured == {
            "model_size": "small", "file_path": tmp_path / "in.mp4",
            "diarize": True, "language": "en", "vad_filter": True,
        }
        # stdout stays pure JSON even though files were written.
        payload = json.loads(capsys.readouterr().out)
        assert payload["transcript"].startswith("Hello")
        assert "[00:00:00.000]" in out.read_text(encoding="utf-8")
        assert (tmp_path / "out.srt").exists()
