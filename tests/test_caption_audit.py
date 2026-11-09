"""Unit tests for the caption-sense audit.

Pure logic: no ffmpeg, no transcript files, no API. The fixtures are word
lists shaped exactly like faster-whisper / Groq verbose_json output so the
checks are exercised on real forms (leading-space tokens, quotes glued to the
terminator, hyphenated fragments).
"""
import pytest

import caption_audit as ca


def W(text, start, end):
    return {"word": text, "start": start, "end": end}


def blocks(*lines):
    """blocks(("Go.", 0.0, 0.4), ("now.", 0.5, 0.9)) -> renderer-shaped blocks."""
    out = []
    for line in lines:
        group, start, end = line
        toks = str(group).split()
        n = max(len(toks), 1)
        span = (end - start) / n
        out.append([W(tok, start + i * span, start + (i + 1) * span)
                    for i, tok in enumerate(toks)])
    return out


def transcript(words, per_segment=None, spaces=True):
    """Wrap absolute-time words into the {'segments': [{'words': [...]}]} shape.

    Every token but the first gets the leading space faster-whisper and Groq
    both emit as their word-boundary marker; without it,
    ``merge_continuation_words`` glues the whole fixture into one word and the
    tests stop describing real transcripts. Pass ``spaces=False`` to keep the
    tokens exactly as written — that is how a fragment fixture says so.
    """
    labelled = []
    for i, word in enumerate(words):
        text = str(word.get("word", ""))
        if spaces and i and not text.startswith(" "):
            word = {**word, "word": " " + text}
        labelled.append(word)
    per_segment = per_segment or len(labelled) or 1
    segs = [{"words": labelled[i:i + per_segment]}
            for i in range(0, len(labelled), per_segment)]
    return {"segments": segs, "language": "en"}


def codes(result):
    return {i["code"] for i in result["issues"]}


class TestSentencePredicates:
    def test_terminators_with_trailing_quotes_count(self):
        assert ca.ends_sentence('end.')
        assert ca.ends_sentence('end!"')
        assert ca.ends_sentence('wait...')
        assert not ca.ends_sentence('end')
        assert not ca.ends_sentence('the')
        assert not ca.ends_sentence('')

    def test_openings(self):
        assert ca.starts_sentence('Go')
        assert ca.starts_sentence('"So')
        assert ca.starts_sentence('2024')
        assert not ca.starts_sentence('because')
        assert not ca.starts_sentence('')

    def test_vowel_ratio_flags_consonant_mush(self):
        assert ca.vowel_ratio('tk') == 0.0
        assert ca.vowel_ratio('the') > 0.3
        assert ca.vowel_ratio('...') == 0.0


class TestReview:
    def test_a_finished_sentence_is_clean(self):
        r = ca.review(blocks(("So that is the whole idea.", 0.0, 2.4)),
                      text_before="and here is why", text_after="",
                      clip_seconds=3.0)
        assert r["ok"], r["issues"]
        assert r["text"].endswith("idea.")
        assert r["stats"]["words"] == 6

    def test_silence_is_not_a_caption(self):
        r = ca.review([], clip_seconds=30.0)
        assert not r["ok"]
        assert codes(r) == {"no_speech"}

    def test_opening_mid_sentence_needs_both_sides_of_evidence(self):
        # lowercase first word + the preceding speech did not end a sentence.
        r = ca.review(blocks(("because that is why it broke.", 0.0, 2.0)),
                      text_before="we looked at the engine", clip_seconds=4.0)
        assert "opens_mid_sentence" in codes(r)
        # Same opening after a finished sentence: nothing to report.
        r2 = ca.review(blocks(("Because that is why it broke.", 0.0, 2.0)),
                       text_before="we looked at the engine.", clip_seconds=4.0)
        assert "opens_mid_sentence" not in codes(r2)
        # No context passed: the check stays quiet instead of guessing.
        r3 = ca.review(blocks(("because that is why it broke.", 0.0, 2.0)),
                       clip_seconds=4.0)
        assert "opens_mid_sentence" not in codes(r3)

    def test_ending_mid_sentence_names_the_word_that_continues(self):
        r = ca.review(blocks(("and the reason for that is.", 0.0, 2.0)),
                      text_after="it works", clip_seconds=2.4)
        assert "ends_mid_sentence" not in codes(r)
        r = ca.review(blocks(("and the reason for that", 0.0, 2.0)),
                      text_after="is simple", clip_seconds=2.0)
        assert "ends_mid_sentence" in codes(r)
        detail = [i for i in r["issues"] if i["code"] == "ends_mid_sentence"][0]
        assert detail["fixable"] and "that" in detail["detail"]

    def test_hallucination_loop(self):
        r = ca.review(blocks(("Thank you for watching.", 0.0, 1.5),
                             ("Thank you for watching.", 1.6, 3.1),
                             ("Thank you for watching.", 3.2, 4.7),
                             ("Thank you for watching.", 4.8, 6.3)),
                      clip_seconds=7.0)
        assert "repeated_text" in codes(r)

    def test_adjacent_duplicates_without_a_loop(self):
        r = ca.review(blocks(("Nice one.", 0.0, 1.0), ("Nice one.", 1.1, 2.1),
                             ("Nice one.", 2.2, 3.2)), clip_seconds=4.0)
        assert "repeated_text" in codes(r)

    def test_text_a_viewer_cannot_read_in_time(self):
        # 26 chars on screen for 0.5s is ~52 cps.
        r = ca.review(blocks(("the quarterly reconciliation statements are", 0.0, 0.5)),
                      clip_seconds=2.0)
        assert "too_fast_to_read" in codes(r)

    def test_music_bed_reads_as_almost_no_speech(self):
        r = ca.review(blocks(("Da.", 0.0, 0.3), ("da.", 40.0, 40.4)),
                      clip_seconds=45.0)
        assert "sparse_speech" in codes(r)
        assert r["stats"]["words_per_sec"] < ca.MIN_WORDS_PER_SEC

    def test_consonant_mush(self):
        r = ca.review(blocks(("tkk ngg shx vrr", 0.0, 1.2)), clip_seconds=2.0)
        assert "gibberish" in codes(r)

    def test_asr_boilerplate_is_junk_even_though_it_grammar_is_fine(self):
        r = ca.review(blocks(("If you liked this please like and subscribe.", 0.0, 2.5)),
                      clip_seconds=3.0)
        assert "junk_text" in codes(r)

    def test_gibberish_check_ignores_punctuation_tokens(self):
        r = ca.review(blocks(("Well — it worked, didn't it?", 0.0, 2.0)),
                      clip_seconds=2.5)
        assert "gibberish" not in codes(r)

    # --- precision: the false positives a first pass produced on real clips ---

    def test_a_shouted_filler_word_is_not_an_asr_loop(self):
        # Gaming commentary says "you" twenty times a minute. Only a phrase
        # repeated BACK TO BACK is the transcript stalling.
        r = ca.review(blocks(("you", 0.0, 0.3), ("go", 1.0, 1.3), ("you", 2.0, 2.3),
                             ("see", 3.0, 3.3), ("you", 4.0, 4.3)),
                      clip_seconds=6.0)
        assert "repeated_text" not in codes(r)

    def test_repetition_must_be_contiguous(self):
        lines = [("here they come", 0.0, 0.9), ("go now", 1.0, 1.9),
                 ("here they come", 2.0, 2.9), ("move up", 3.0, 3.9),
                 ("here they come", 4.0, 4.9)]
        r = ca.review(blocks(*lines), clip_seconds=6.0)
        assert "repeated_text" not in codes(r)

    def test_a_real_loop_still_fires(self):
        r = ca.review(blocks(*[("never mind", i * 0.8, i * 0.8 + 0.7)
                               for i in range(5)]), clip_seconds=4.5)
        assert "repeated_text" in codes(r)

    def test_an_interjection_is_not_judged_on_reading_speed(self):
        # "Huh?" in 0.05s is a timestamp artefact, not 400 chars/second.
        r = ca.review(blocks(("Huh?", 0.0, 0.05)), clip_seconds=3.0)
        assert "too_fast_to_read" not in codes(r)

    def test_a_long_phrase_held_too_short_is_unreadable(self):
        r = ca.review(blocks(("the reconciliation of the accounts failed", 0.0, 1.0)),
                      clip_seconds=2.0)
        assert "too_fast_to_read" in codes(r)

    def test_karaoke_blocks_are_not_judged_on_reading_speed(self):
        # Same block, revealed one word at a time: the 1.0s is how fast the
        # speaker talked, not how long the viewer has to parse a whole line.
        # This was the false positive on every energetic clip in a live job.
        b = blocks(("the reconciliation of the accounts failed", 0.0, 1.0))
        assert "too_fast_to_read" not in codes(ca.review(b, clip_seconds=2.0,
                                                         karaoke=True))

    def test_karaoke_still_reports_a_firehose(self):
        # The aggregate load is a real problem in every style: 200 chars of
        # speech in 5s is unreadable however it is revealed.
        b = blocks(("the reconciliation of the accounts failed "
                    "and nobody signed the paperwork", 0.0, 1.0),
                   ("so the whole quarter had to be rebuilt from "
                    "the raw ledger exports by hand", 1.0, 2.0))
        r = ca.review(b, clip_seconds=2.0, karaoke=True)
        assert "too_fast_to_read" in codes(r)

    def test_a_transcript_denser_than_the_audio_cannot_hold_is_flagged(self):
        # The ASR offset bug in one picture: four minutes of speech compressed
        # into ten seconds. Every sentence looks fine; the arithmetic doesn't.
        dense = [[W(f"word{i}", i * 0.05, i * 0.05 + 0.04)
                  for _ in range(1)] for i in range(120)]
        r = ca.review(dense, clip_seconds=10.0)
        assert "impossible_density" in codes(r)

    def test_normal_speech_does_not_trip_the_density_check(self):
        r = ca.review(blocks(("That is the entire point of the argument.", 0.0, 2.6)),
                      clip_seconds=3.0)
        assert r["stats"]["words_per_sec"] < ca.MAX_WORDS_PER_SEC
        assert codes(r) == set()


class TestFlatten:
    def test_merges_fragments_and_sorts(self):
        # "-Kanal." has no leading space: faster-whisper delivered it as a
        # fragment of the word before it, not as a new word.
        t = transcript([W(" You", 1.0, 1.2), W("-Kanal.", 1.2, 1.4),
                        W(" Tube", 0.2, 0.4)], per_segment=2, spaces=False)
        words = ca.flatten_words(t)
        assert [w["word"] for w in words] == ["Tube", "You-Kanal."]
        assert words[0]["start"] == 0.2

    def test_garbage_input_is_empty_not_fatal(self):
        assert ca.flatten_words(None) == []
        assert ca.flatten_words({"segments": None}) == []
        assert ca.flatten_words({"segments": [{"words": [{}]}]}) == []


class TestSentences:
    def test_groups_on_terminators(self):
        words = [W("Go.", 0.0, 0.3), W("then", 0.4, 0.7), W("stop.", 0.8, 1.1),
                 W("and", 1.5, 1.6)]
        sents = ca.sentence_boundaries(words)
        assert [s["text"] for s in sents] == ["Go.", "then stop.", "and"]
        assert sents[0]["start"] == 0.0 and sents[0]["end"] == 0.3
        assert sents[1]["start"] == 0.4 and sents[1]["end"] == 1.1
        # the trailing fragment is a sentence too — the audio just ends there
        assert sents[2]["text"] == "and"

    def test_a_pause_belongs_to_the_sentence_that_ended(self):
        sents = ca.sentence_boundaries([W("Go.", 0.0, 0.3), W("Stop.", 3.0, 3.3)])
        assert ca.containing_sentence(sents, 1.2) is sents[0]
        assert ca.containing_sentence(sents, 3.1) is sents[1]


class TestRepairRange:
    def test_end_runs_to_finish_the_sentence(self):
        t = transcript([W("The", 0.0, 0.2), W("punchline", 0.3, 0.6),
                        W("lands", 0.7, 0.9), W("here.", 1.0, 2.6),
                        W("Next", 5.0, 5.2)])
        start, end, notes = ca.repair_range(t, 0.0, 2.0, video_duration=60.0,
                                            min_duration=5.0, max_duration=30.0)
        assert end == pytest.approx(2.6)
        assert start == 0.0
        assert any("sentence" in n for n in notes)

    def test_refuses_when_the_band_would_break(self):
        t = transcript([W("The", 0.0, 0.2), W("tail", 0.3, 0.5), W("goes.", 0.6, 12.0)])
        start, end, notes = ca.repair_range(t, 0.0, 10.0, video_duration=60.0,
                                            min_duration=5.0, max_duration=11.0,
                                            max_shift=6.0)
        assert (start, end) == (0.0, 10.0)
        assert notes == []

    def test_start_steps_back_to_the_sentence_it_cut(self):
        t = transcript([W("Setup", 0.0, 0.2), W("comes", 0.3, 0.5),
                        W("first.", 0.6, 0.9), W("then", 1.0, 1.2),
                        W("the.", 1.3, 1.6), W("Payoff", 2.0, 2.4)])
        # A cut at 1.45 lands in the middle of "then the."; the sentence
        # begins at 1.0, so the clip starts there instead of on "the.".
        start, end, notes = ca.repair_range(
            t, 1.45, 6.0, video_duration=60.0, min_duration=2.0, max_duration=30.0)
        assert start == 1.0
        assert any("begin the sentence" in n for n in notes)

    def test_a_cut_inside_the_final_word_still_extends(self):
        # "here." carries a period, so a naive terminator check would call this
        # clip finished; it actually stops 1.6s into the word.
        t = transcript([W("The", 0.0, 0.2), W("joke", 0.3, 0.5),
                        W("lands", 0.6, 0.8), W("here.", 0.9, 2.6)])
        start, end, notes = ca.repair_range(t, 0.0, 1.6, video_duration=60.0,
                                            min_duration=2.0, max_duration=30.0)
        assert end == pytest.approx(2.6)

    def test_a_clean_range_moves_nowhere(self):
        t = transcript([W("One.", 0.0, 0.3), W("Two.", 1.0, 1.3),
                        W("Three.", 2.0, 2.3)])
        assert ca.repair_range(t, 0.0, 2.3, 60.0, 1.0, 30.0) == (0.0, 2.3, [])

    def test_env_off_switch(self, monkeypatch):
        monkeypatch.setenv("CAPTION_AUDIT", "0")
        t = transcript([W("Go", 0.0, 0.2), W("now.", 0.3, 1.4)])
        assert ca.repair_range(t, 0.0, 1.0, 60.0, 1.0, 30.0) == (0.0, 1.0, [])

    def test_no_transcript_is_a_no_op(self):
        assert ca.repair_range(None, 3.0, 9.0, 60.0) == (3.0, 9.0, [])
        assert ca.repair_range({"segments": []}, 3.0, 9.0, 60.0) == (3.0, 9.0, [])

    def test_a_broken_transcript_never_raises(self):
        # 'start' as a string is what a hand-edited cache file looks like.
        bad = {"segments": [{"words": [{"word": "x", "start": "nope", "end": None}]}]}
        assert ca.repair_range(bad, 0.0, 1.0, 10.0)[2] == []


class TestAuditIntegration:
    def test_uses_the_renderers_own_blocks(self):
        t = transcript([W(" This", 0.0, 0.3), W(" clip", 0.3, 0.6),
                        W(" stops", 0.6, 0.9), W(" mid", 0.9, 1.2),
                        W(" of", 1.2, 1.4), W(" a", 1.4, 1.5),
                        W(" sentence.", 1.5, 2.0)], per_segment=7)
        r = ca.audit(t, 0.0, 1.0, max_chars=40, max_duration=2.0)
        assert r["clip"] == [0.0, 1.0]
        assert r["text"] == "This clip stops mid"
        assert not r["ok"]
        assert "ends_mid_sentence" in codes(r)

    def test_audit_of_silence_in_the_range(self):
        t = transcript([W("Elsewhere.", 100.0, 101.0)])
        r = ca.audit(t, 0.0, 10.0)
        assert codes(r) == {"no_speech"}

    def test_summary_is_one_loggable_line(self):
        ok = ca.review(blocks(("All good.", 0.0, 1.0)), clip_seconds=1.2)
        assert "clean" in ca.summary(ok)
        bad = ca.review([], clip_seconds=5.0)
        s = ca.summary(bad)
        assert "no_speech" in s
        assert ca.summary(None) == "caption audit unavailable"


class TestPipelineWiring:
    """The audit only matters if the pipeline runs it.

    Source inspection, the way test_key_pool.py guards the ASR rules: these
    few lines of main.py are the difference between a library and a feature,
    and a refactor that drops them would silence the audit without failing one
    of the tests above.
    """
    def _main(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "main.py")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_selection_repairs_the_cut_before_rendering(self):
        src = self._main()
        loop = src[src.index("# Snap each proposed clip"):
                   src.index("# Aggregate cost across both passes")]
        assert "caption_audit.repair_range(" in loop
        # Ordered: re-anchoring after the clip dict is written back would
        # render the un-repaired range and report the repaired one.
        assert (loop.index("caption_audit.repair_range(")
                < loop.index('s["start"], s["end"] = rs, re_'))
        assert "caption_audit.audit(" in loop

    def test_selected_clips_ship_the_verdict_in_metadata(self):
        assert 's["caption_review"]' in self._main()

    def test_the_burn_path_reports_what_it_shipped(self):
        src = self._main()
        fn = src[src.index("def auto_caption_clip("):
                 src.index("def auto_hook_clip(")]
        assert "caption_audit.summary(" in fn

    def test_the_audit_is_told_how_captions_reveal(self):
        # A parameter nobody passes is a feature nobody has: without karaoke=
        # at both call sites every energetic clip ships a false warning.
        src = self._main()
        assert src.count("karaoke=(style.get(\"style\") == \"karaoke\")") \
            + src.count("karaoke=(_style.get(\"style\") == \"karaoke\")") == 2

    def test_the_audit_can_never_lose_a_clip(self):
        src = self._main()
        fn = src[src.index("def auto_caption_clip("):
                 src.index("def auto_hook_clip(")]
        # The audit call sits inside the try that already degrades to "ship
        # the clip uncaptioned"; the selection loop likewise swallows it.
        assert "try:" in fn[:fn.index("caption_audit.audit(")]
        assert src.count("Caption audit skipped") >= 2


class TestVerdictPayload:
    """One stored shape for every path that captions a clip.

    The library pill, the editor pill and the MCP tools all read
    ``clip["caption_review"]``. If selection writes three keys and a recut
    writes five, the same clip means two different things depending on which
    button was last pressed — and the UI silently shows the union.
    """
    def test_a_verdict_becomes_the_stored_dict(self):
        verdict = ca.review(blocks(("Hello there.", 0.0, 1.4)),
                            clip_seconds=2.0)
        payload = ca.verdict_payload(verdict)
        assert set(payload) == {"ok", "issues", "detail", "stats", "clip"}
        assert payload["ok"] is True
        assert payload["issues"] == [] and payload["detail"] == []
        assert payload["stats"]["words"] == 2
        # "clip" is the range audit() judged; review() alone has no range.
        assert payload["clip"] is None

    def test_a_failing_verdict_keeps_its_explanation(self):
        verdict = ca.review(blocks(("cut off mid", 0.0, 1.0)),
                            text_after="of the sentence that goes on.",
                            clip_seconds=1.0)
        payload = ca.verdict_payload(verdict)
        assert payload["ok"] is False
        assert "ends_mid_sentence" in payload["issues"]
        # Detail is what the UI quotes, so it has to survive the trip.
        assert any(payload["detail"])

    def test_a_judged_range_is_recorded_with_the_verdict(self):
        # The editor shows when a verdict was measured; a verdict that outlives
        # the range it described is the bug this key exists to catch.
        verdict = ca.audit(transcript([W("Hello.", 10.0, 10.6)]), 10.0, 12.0)
        assert ca.verdict_payload(verdict)["clip"] == [10.0, 12.0]

    def test_no_verdict_stores_nothing(self):
        assert ca.verdict_payload(None) is None
        assert ca.verdict_payload({}) is None

    def test_nobody_hand_builds_the_dict_anymore(self):
        import os
        root = os.path.join(os.path.dirname(__file__), "..")
        for name in ("main.py", "app.py"):
            with open(os.path.join(root, name), encoding="utf-8") as fh:
                src = fh.read()
            assert 'for i in verdict["issues"]' not in src, name
            assert src.count("caption_audit.verdict_payload(") >= (
                1 if name == "main.py" else 2), name
