import json
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

import transcribe_backends as tb


class FakeFloat(float):
    """Stands in for numpy scalar types (float subclasses) from onnx-asr."""


def _seg(start, end, text, tokens, timestamps):
    return SimpleNamespace(
        start=FakeFloat(start), end=FakeFloat(end), text=text,
        tokens=tokens, timestamps=[FakeFloat(t) for t in timestamps],
    )


# --- token -> word reconstruction ------------------------------------------

def test_words_from_tokens_groups_by_leading_space():
    words = tb._words_from_tokens(
        [" T", "odo", " el", " mundo", "."],
        [0.0, 0.16, 0.32, 0.40, 0.60],
        seg_start=10.0, seg_end=12.0,
    )
    assert [w["word"] for w in words] == [" Todo", " el", " mundo."]
    # Absolute (segment-offset) start times.
    assert words[0]["start"] == pytest.approx(10.0)
    assert words[1]["start"] == pytest.approx(10.32)
    # End inferred from the next word's start; last word ends at segment end.
    assert words[0]["end"] == pytest.approx(10.32)
    assert words[2]["end"] == pytest.approx(11.0, abs=0.3)


def test_words_from_tokens_first_token_without_space_gets_one():
    words = tb._words_from_tokens([
        "Hola", " que", " tal"], [0.0, 0.5, 1.0], seg_start=0.0, seg_end=2.0)
    assert words[0]["word"] == " Hola"


def test_words_from_tokens_end_capped_after_long_silence():
    # Second word starts 5s later; first word's end must stay near its
    # own tokens (cap = last token + 0.6), not stretch across the gap.
    words = tb._words_from_tokens(
        [" uno", " dos"], [0.0, 5.0], seg_start=0.0, seg_end=6.0)
    assert words[0]["end"] <= 0.7


def test_words_from_tokens_all_floats_are_native():
    words = tb._words_from_tokens(
        [" a", "b", " c"], [0.0, 0.1, 0.2], seg_start=FakeFloat(1), seg_end=FakeFloat(2))
    for w in words:
        assert type(w["start"]) is float
        assert type(w["end"]) is float


# --- parakeet transcript assembly ------------------------------------------

@pytest.fixture
def fake_parakeet(monkeypatch):
    segs = [
        _seg(0.35, 2.43, "Todo el mundo habla.",
             [" Todo", " el", " mundo", " hab", "la", "."],
             [0.0, 0.32, 0.40, 0.56, 0.72, 0.80]),
        _seg(2.59, 5.57, "Pero, ¿qué es esto?",
             [" Pero", ",", " ¿", "qu", "é", " es", " esto", "?"],
             [0.0, 0.24, 0.32, 0.48, 0.64, 0.80, 1.04, 1.36]),
    ]
    model = SimpleNamespace(recognize=lambda path: iter(segs))
    monkeypatch.setattr(tb, "_get_parakeet_model", lambda: model)
    monkeypatch.setattr(tb, "_extract_wav", lambda path: "/tmp/fake.wav")
    monkeypatch.setattr(tb.os, "remove", lambda path: None)
    return segs


def test_parakeet_transcript_matches_contract(fake_parakeet, monkeypatch):
    monkeypatch.setattr(tb, "_detect_language", lambda text: "es")
    t = tb._transcribe_with_parakeet("video.mp4")

    assert t["text"] == "Todo el mundo habla. Pero, ¿qué es esto?"
    assert t["language"] == "es"
    assert len(t["segments"]) == 2

    seg = t["segments"][1]
    assert type(seg["start"]) is float and type(seg["end"]) is float
    # Continuations (",", "qu", "é") merged; leading spaces preserved.
    assert [w["word"] for w in seg["words"]] == [" Pero,", " ¿qué", " es", " esto?"]
    # Segment offset applied to word times.
    assert seg["words"][0]["start"] == pytest.approx(2.59)
    # Whole transcript is JSON-serializable (metadata json.dump path).
    json.dumps(t)


def test_parakeet_words_survive_merge_continuation_words(fake_parakeet, monkeypatch):
    from subtitles import merge_continuation_words
    monkeypatch.setattr(tb, "_detect_language", lambda text: "es")
    t = tb._transcribe_with_parakeet("video.mp4")
    for seg in t["segments"]:
        assert merge_continuation_words(seg["words"]) == seg["words"]


# --- fallback policy --------------------------------------------------------

def _valid_transcript(n_words=200, duration=120.0, language="es"):
    words = [
        {"word": f" w{i}", "start": i * duration / n_words,
         "end": (i + 1) * duration / n_words}
        for i in range(n_words)
    ]
    return {"text": "x " * n_words, "language": language,
            "segments": [{"start": 0.0, "end": duration, "text": "x", "words": words}]}


def test_fallback_reason_none_for_good_result():
    assert tb._parakeet_fallback_reason(_valid_transcript()) is None


def test_fallback_when_no_words():
    t = {"text": "", "language": "es",
         "segments": [{"start": 0, "end": 5, "text": "x", "words": []}]}
    assert "no words" in tb._parakeet_fallback_reason(t)


def test_fallback_when_language_unsupported():
    assert "outside" in tb._parakeet_fallback_reason(
        _valid_transcript(language="ja"))


def test_fallback_when_word_rate_absurdly_low():
    assert tb._parakeet_fallback_reason(
        _valid_transcript(n_words=5, duration=600.0)) is not None


def test_transcribe_media_falls_back_on_parakeet_exception(monkeypatch):
    def boom(path):
        raise RuntimeError("onnx exploded")

    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "parakeet")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(tb, "_transcribe_with_parakeet", boom)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


def test_transcribe_media_default_is_whisper(monkeypatch):
    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(
        tb, "_transcribe_with_parakeet",
        lambda path: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


def test_transcribe_media_raises_on_silent_video(monkeypatch):
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: False)
    with pytest.raises(tb.NoAudioError):
        tb.transcribe_media("silent.mp4")


# --- groq cloud backend ------------------------------------------------------

def _groq_payload():
    return {
        "text": "Hello world. Second part.",
        "language": "en",
        "duration": 6.0,
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "Hello world."},
            {"start": 3.0, "end": 6.0, "text": "Second part."},
        ],
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world.", "start": 0.6, "end": 1.0},
            {"word": "Second", "start": 3.0, "end": 3.4},
            {"word": "part.", "start": 3.5, "end": 4.0},
        ],
    }


def test_groq_payload_maps_to_contract():
    t = tb._groq_payload_to_transcript(_groq_payload())

    assert t["text"] == "Hello world. Second part."
    assert t["language"] == "en"
    assert len(t["segments"]) == 2
    # OpenAI-style complete words get the faster-whisper leading-space
    # convention, assigned to the segment containing their start.
    assert [w["word"] for w in t["segments"][0]["words"]] == [" Hello", " world."]
    assert [w["word"] for w in t["segments"][1]["words"]] == [" Second", " part."]
    for seg in t["segments"]:
        for w in seg["words"]:
            assert type(w["start"]) is float and type(w["end"]) is float
    # Whole transcript is JSON-serializable (metadata json.dump path).
    json.dumps(t)


def test_groq_offset_shifts_words_and_segments():
    t = tb._groq_payload_to_transcript(_groq_payload(), offset=120.0)
    assert t["segments"][1]["start"] == pytest.approx(123.0)
    assert t["segments"][1]["words"][0]["start"] == pytest.approx(123.0)


def test_groq_words_survive_merge_continuation_words():
    from subtitles import merge_continuation_words
    t = tb._groq_payload_to_transcript(_groq_payload())
    for seg in t["segments"]:
        assert merge_continuation_words(seg["words"]) == seg["words"]


def test_groq_without_word_timestamps_synthesizes_them():
    payload = _groq_payload()
    payload.pop("words")
    t = tb._groq_payload_to_transcript(payload)
    assert len(t["segments"]) == 2
    for seg in t["segments"]:
        assert seg["words"]
        assert seg["words"][0]["word"].startswith(" ")


def test_groq_without_segments_synthesizes_one_from_words():
    payload = _groq_payload()
    payload.pop("segments")
    t = tb._groq_payload_to_transcript(payload)
    assert len(t["segments"]) == 1
    assert len(t["segments"][0]["words"]) == 4
    assert t["text"] == "Hello world. Second part."


def test_groq_part_plan():
    # 100 MB of FLAC over an hour, 24 MB cap -> ~778 s parts (90% of the cap;
    # no floor: dense audio must not be forced into oversized parts).
    part = tb._groq_part_plan(100e6, 3600.0, 24e6)
    assert 700 <= part <= 800, part
    assert part * (100e6 / 3600.0) < 24e6, "planned part must fit the cap"
    assert tb._groq_part_plan(10e6, 600.0, 24e6) is None  # single upload
    assert tb._groq_part_plan(10e6, 0.0, 24e6) is None  # unknown duration


def test_groq_part_plan_no_floor_for_dense_audio():
    # Dense game/music audio (~12 MB/min): a 300s floor would send 58 MB parts
    # and Groq answers 413 — that killed cloud ASR on 17-oct-2026.
    part = tb._groq_part_plan(291e6, 1500.0, 24e6)
    assert part is not None and part < 300
    assert part * (291e6 / 1500.0) < 24e6


def test_groq_part_plan_splits_when_duration_unreadable():
    # Returning None here handed the whole over-cap file to the API for a 413:
    # a duration ffprobe cannot read is a reason to slice conservatively, not
    # to stop slicing.
    assert tb._groq_part_plan(100e6, 0.0, 24e6) == tb._UNKNOWN_DURATION_PART_SECS


# --- chunk offsets (the 2-nov-2026 "captions don't match" bug) ---------------

def _ffmpeg_available():
    import shutil
    return shutil.which("ffmpeg") and shutil.which("ffprobe")


pytestmark_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="needs ffmpeg/ffprobe on PATH")


@pytestmark_ffmpeg
def test_segment_muxer_parts_report_a_duration(tmp_path):
    """The regression the old offsets died of: muxer FLAC parts probe N/A.

    ``format=duration`` and ``stream=duration`` both answer N/A for these
    files, so _probe_duration has to fall back to the packet table — otherwise
    every chunk measures 0 s and every offset collapses onto the start.
    """
    src = _tone_flac(tmp_path, seconds=20)
    outdir = tmp_path / "seg"
    outdir.mkdir()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-f", "segment", "-segment_time", "5", "-c:a", "flac",
                    str(outdir / "part_%05d.flac")], check=True, timeout=120)
    parts = sorted(outdir.glob("part_*.flac"))
    assert len(parts) >= 3
    # Header really is unreadable (that is what makes this a regression test).
    header = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of", "csv=p=0", str(parts[1])],
                            capture_output=True, text=True, timeout=60).stdout.strip()
    assert header in ("N/A", ""), f"expected an unreadable header, got {header!r}"
    assert tb._probe_duration(parts[1]) == pytest.approx(5.0, abs=0.3)
    # And the muxer stamps them against the source, so part 2 starts at 5 s.
    assert tb._probe_pts_range(parts[1])[0] == pytest.approx(5.0, abs=0.3)


@pytestmark_ffmpeg
def test_groq_split_offsets_tile_the_timeline(tmp_path):
    """Parts must land at absolute, strictly increasing, gap-free offsets.

    End-to-end form of the bug: 23 parts of a 4-hour video all got offset 0.0,
    so the whole video was transcribed as its first 9 minutes.
    """
    src = _tone_flac(tmp_path, seconds=60)
    import os
    full = tb._extract_groq_audio(str(src), str(tmp_path))
    parts = tb._split_audio_for_groq(str(src), str(tmp_path),
                                     max_bytes=os.path.getsize(full) / 4)
    assert len(parts) >= 3, [p for p, _ in parts]
    offsets = [off for _, off in parts]
    assert offsets[0] == pytest.approx(0.0, abs=0.2)
    assert all(b > a for a, b in zip(offsets, offsets[1:])), offsets
    # Every gap is a real part, so consecutive offsets advance by more than a
    # rounding error and the last part still sits inside the source.
    assert all(b - a > 1.0 for a, b in zip(offsets, offsets[1:])), offsets
    assert offsets[-1] < 60.0
    # Coverage: parts concatenated by offset span the whole audio. Measured from
    # the packet table — a muxer part's header carries the WHOLE file's sample
    # count, not its own slice.
    covered = offsets[-1] + tb._probe_pts_range(parts[-1][0])[1]
    assert covered == pytest.approx(60.0, abs=1.0)


@pytestmark_ffmpeg
def test_groq_split_resplit_keeps_both_parts(tmp_path):
    """Two oversized parts must not share sub-file names (the collision bug).

    A shared ``sub_%05d`` prefix made each ffmpeg run overwrite the previous
    part's sub files, so both parts collected the same last-written files and
    their offsets drifted.
    """
    src = _tone_flac(tmp_path, seconds=60)
    import os
    full = tb._extract_groq_audio(str(src), str(tmp_path))
    parts = tb._split_audio_for_groq(str(src), str(tmp_path),
                                     max_bytes=os.path.getsize(full) / 8)
    names = [os.path.basename(p) for p, _ in parts]
    assert len(names) == len(set(names)), names
    offsets = [off for _, off in parts]
    assert all(b > a for a, b in zip(offsets, offsets[1:])), offsets


def _tone_flac(tmp_path, seconds):
    """Small deterministic test audio, so chunk sizes are predictable."""
    path = tmp_path / "source.flac"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds}",
                    "-ac", "1", "-c:a", "flac", str(path)],
                   check=True, timeout=120)
    return path


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        tb._transcribe_with_groq("video.mp4")


def test_groq_split_single_part_for_small_audio(tmp_path, monkeypatch):
    fake_flac = tmp_path / "extracted.flac"
    fake_flac.write_bytes(b"x" * 1000)
    monkeypatch.setattr(tb, "_extract_groq_audio",
                        lambda media, tmpdir: str(fake_flac))
    parts = tb._split_audio_for_groq("video.mp4", str(tmp_path), 24e6)
    assert parts == [(str(fake_flac), 0.0)]


def test_transcribe_media_falls_back_on_groq_exception(monkeypatch):
    def boom(path):
        raise RuntimeError("groq exploded")

    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(tb, "_transcribe_with_groq", boom)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


def test_transcribe_media_groq_happy_path(monkeypatch):
    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(tb, "_transcribe_with_groq", lambda path: sentinel)
    monkeypatch.setattr(
        tb, "_transcribe_with_whisper",
        lambda path: (_ for _ in ()).throw(AssertionError("should not run")))
    assert tb.transcribe_media("video.mp4") is sentinel


# --- whisper singleton ------------------------------------------------------

@pytest.fixture
def fake_faster_whisper(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self, model_size, device=None, compute_type=None):
            self.model_size = model_size
            self.device = device
            created.append(self)

        def transcribe(self, path, **params):
            segs = (s for s in [SimpleNamespace(
                start=0.0, end=1.0, text=" hola", words=None)])
            return segs, SimpleNamespace(language="es")

    fake_module = types.SimpleNamespace(WhisperModel=FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(tb, "_whisper_model", None)
    monkeypatch.setattr(tb, "_whisper_key", None)
    return created


def test_whisper_model_is_singleton(fake_faster_whisper, monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    a, _ = tb._get_whisper_model()
    b, _ = tb._get_whisper_model()
    assert a is b
    assert len(fake_faster_whisper) == 1


def test_whisper_model_rebuilds_when_env_changes(fake_faster_whisper, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small")
    a, _ = tb._get_whisper_model()
    monkeypatch.setenv("WHISPER_MODEL", "large-v3-turbo")
    b, _ = tb._get_whisper_model()
    assert a is not b
    assert b.model_size == "large-v3-turbo"


def test_run_whisper_transcription_materializes_segments(fake_faster_whisper, monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    segments, info = tb.run_whisper_transcription("video.mp4")
    assert isinstance(segments, list)
    assert info.language == "es"
