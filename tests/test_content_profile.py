"""The genre layer: classification mapping, the brief, and how it splices in."""
import pytest

import content_profile as cp
import gemini_worker


# --- normalisation -----------------------------------------------------------

def test_normalize_genre_accepts_model_shaped_answers():
    # Models answer with labels and punctuation, not keys.
    assert cp.normalize_genre("Gaming") == "gaming"
    assert cp.normalize_genre("gaming / let's-play") == "gaming"
    assert cp.normalize_genre("Podcast-Interview") == "podcast_interview"
    assert cp.normalize_genre("  tutorial howto ") == "tutorial_howto"


def test_normalize_genre_falls_back_to_general():
    for raw in (None, "", "a cat video", "unknown"):
        assert cp.normalize_genre(raw) == "general"


def test_normalize_genre_maps_what_models_actually_say():
    assert cp.normalize_genre("interview") == "podcast_interview"
    assert cp.normalize_genre("Video Game") == "gaming"
    assert cp.normalize_genre("Cooking Show") == "cooking_craft"
    assert cp.normalize_genre("fitness") == "sports_fitness"


def test_a_real_key_beats_an_alias():
    # "general" contains no alias, but a loose answer that mentions both must
    # still land on the stated genre.
    assert cp.normalize_genre("music performance") == "music_performance"


def test_every_genre_key_round_trips():
    for key in cp.GENRES:
        assert cp.normalize_genre(key) == key
        genre = cp.get_genre(key)
        assert genre.what_matters and genre.avoid and genre.caption_rules
        assert genre.hook_angle and genre.notes_for_editor


def test_normalize_style_aliases():
    assert cp.normalize_style("funny") == "funny"
    assert cp.normalize_style("Comedy") == "funny"
    assert cp.normalize_style("clutch") == "peak"
    assert cp.normalize_style("peaks") == "peak"
    assert cp.normalize_style("whatever") == "general"


# --- the brief ---------------------------------------------------------------

def test_brief_is_empty_for_the_neutral_pair():
    # An unclassified video with no style request must score exactly the way it
    # did before this module existed.
    assert cp.build_brief("general", "general") == ""


def test_brief_carries_all_three_axes():
    brief = cp.build_brief("gaming", "funny", evidence="gg bro")
    assert "GENRE: gaming" in brief
    assert "clutch or impossible outcome" in brief
    assert "WHAT THE USER ASKED FOR: funny moments" in brief
    assert "kill feed" in brief              # genre caption rules
    assert "gg bro" in brief                 # evidence line
    assert "HOOK ANGLE" in brief
    assert "punch-ins and short zooms help" in brief


def test_brief_warns_against_motion_for_a_talking_head():
    brief = cp.build_brief("podcast_interview", "general")
    assert "zoom and punch-in are usually a defect here" in brief


def test_style_changes_the_ranking_instructions():
    funny = cp.build_brief("gaming", "funny")
    peak = cp.build_brief("gaming", "peak")
    assert "actually funny" in funny
    assert "genuinely in doubt" in peak
    assert funny != peak


def test_style_from_env(monkeypatch):
    monkeypatch.delenv("CONTENT_STYLE", raising=False)
    assert cp.style_from_env() == "general"
    monkeypatch.setenv("CONTENT_STYLE", "funny")
    assert cp.style_from_env() == "funny"


def test_genre_from_env_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("CONTENT_GENRE", raising=False)
    assert cp.genre_from_env() is None
    monkeypatch.setenv("CONTENT_GENRE", "Gaming")
    assert cp.genre_from_env() == "gaming"


# --- splicing into the prompts -----------------------------------------------

def test_apply_brief_removes_the_marker_when_there_is_no_brief():
    prompt = cp.apply_brief("head\n<<EDITORIAL_BRIEF>>\ntail", "")
    assert cp.BRIEF_MARKER not in prompt
    assert "head" in prompt and "tail" in prompt


def test_both_templates_carry_the_marker_once():
    for template in (gemini_worker.SCORE_PROMPT_TEMPLATE,
                     gemini_worker.DETAIL_PROMPT_TEMPLATE):
        assert template.count(cp.BRIEF_MARKER) == 1


def test_brief_survives_format_because_it_is_spliced_afterwards():
    # The brief is prose full of characters .format() would interpret; splicing
    # it after the format call is what keeps every existing call site working.
    brief = cp.build_brief("gaming", "funny")
    prompt = gemini_worker.SCORE_PROMPT_TEMPLATE.format(
        video_duration=100, language="en", windows_json="[]")
    final = cp.apply_brief(prompt, brief)
    assert brief in final
    assert cp.BRIEF_MARKER not in final
    assert "WINDOWS_JSON" in final


def test_detail_prompt_defers_to_a_detected_genre():
    assert "If an EDITORIAL BRIEF above" in gemini_worker.DETAIL_PROMPT_TEMPLATE


# --- the classification prompt and sample ------------------------------------

def test_genre_prompt_lists_every_genre():
    prompt = cp.GENRE_PROMPT_TEMPLATE.format(
        genre_list=cp.genre_labels(), sample="hello")
    for key in cp.GENRES:
        assert key in prompt


def test_genre_response_validates_a_free_text_answer():
    # A loose answer still validates; normalize_genre does the mapping, so a
    # capital or a label never raises and fails the job over a label.
    parsed = cp.GenreResponse.model_validate({"genre": "Gaming", "evidence": "gg"}).model_dump()
    assert cp.normalize_genre(parsed["genre"]) == "gaming"


def test_prompt_sample_takes_beginning_middle_and_end():
    text = " ".join(f"word{i}" for i in range(4000))
    sample = cp.prompt_sample({"text": text}, chars_per_slice=100, slices=3)
    assert sample.startswith("word0")
    assert sample.endswith("word3999")
    assert "[..]" in sample
    assert len(sample) < len(text)


def test_prompt_sample_falls_back_to_segments():
    result = {"segments": [{"text": " one "}, {"text": " two "}]}
    assert cp.prompt_sample(result) == "one two"


def test_prompt_sample_survives_an_empty_transcript():
    assert cp.prompt_sample(None) == ""
    assert cp.prompt_sample({"text": "", "segments": []}) == ""


# --- request parameters ------------------------------------------------------

def test_parse_style_param():
    assert cp.parse_style_param(None) is None
    assert cp.parse_style_param("") is None
    assert cp.parse_style_param("auto") is None      # explicit no-op
    assert cp.parse_style_param(" Funny ") == "funny"


def test_parse_style_param_rejects_a_typo():
    # Silently downgrading "funny" to "general" is how a user loses a request.
    with pytest.raises(ValueError, match="content_style"):
        cp.parse_style_param("hilarious")


def test_parse_genre_param_rejects_a_typo():
    assert cp.parse_genre_param("gaming") == "gaming"
    assert cp.parse_genre_param("auto") is None
    with pytest.raises(ValueError, match="content_genre"):
        cp.parse_genre_param("cartoons")
