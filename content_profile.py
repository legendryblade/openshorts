"""Genre-aware editorial brief: decide what KIND of video this is before scoring it.

The picker used to run one generic rubric over every source, so "find me the
funny moments" and "find me the peak moments" returned the same clips, and a
gaming VOD was judged with podcast criteria. That is the complaint in
HANDOVER 3.1: one lens cannot tell a clutch play from a punchline.

Two axes, combined into the block injected into the scoring and detail prompts:

  * genre    — what the video IS, detected from the transcript before scoring.
               Tells the model what matters in THIS material.
  * style    — what the user ASKED for (funny | peak | general), from
               CONTENT_STYLE. Tells the model which of the available moments to
               prefer when more than one qualifies.

Why a categorical choice and not a score: layout_picker.py measured that
Gemini's numeric answers drift run to run while a choice among closed options
is stable (94/92/96% over a 48-clip corpus). Genre classification uses the same
shape for the same reason.

Everything here is pure — no network, no SDK — so the rubric text and the
fallback behaviour stay unit-testable. The model call lives in main.py, which
already has the pacer, the retry policy and the dual Gemini/LLM-backend path.
"""
from __future__ import annotations

import os

from pydantic import BaseModel, Field

VALID_STYLES = ("general", "funny", "peak")

#: Sentinel the prompts carry instead of a format field, so callers that do not
#: know about the brief still format the template unchanged.
BRIEF_MARKER = "<<EDITORIAL_BRIEF>>"


class _Genre:
    """One genre's editorial lens. Plain object: cheap to construct, hashable
    by key, and never serialised into the transcript contract."""

    __slots__ = ("key", "label", "what_matters", "avoid", "hook_angle",
                 "caption_rules", "motion_ok", "notes_for_editor")

    def __init__(self, key, label, what_matters, avoid, hook_angle,
                 caption_rules, motion_ok, notes_for_editor):
        self.key = key
        self.label = label
        self.what_matters = what_matters
        self.avoid = avoid
        self.hook_angle = hook_angle
        self.caption_rules = caption_rules
        self.motion_ok = motion_ok
        self.notes_for_editor = notes_for_editor


def _g(*args, **kwargs):
    return _Genre(*args, **kwargs)


GENRES = {
    "gaming": _g(
        key="gaming",
        label="gaming / let's-play / stream",
        what_matters=[
            "a clutch or impossible outcome the player did not expect",
            "a failure that is funny specifically because the player is invested",
            "a reaction spike: scream, laugh, disbelief, rage at an absurd event",
            "a near-miss or countdown where the result is genuinely in doubt",
            "an unexpected event from another player, an NPC, or the game itself",
        ],
        avoid=[
            "menu navigation, inventory management, looting and building boilerplate",
            "sponsor reads, sub goals, and 'drop a like' housekeeping",
            "coordination chatter that only makes sense to the people in the call",
            "a kill or win narrated flatly, with no reaction attached to it",
        ],
        hook_angle="Name the impossible thing and the stakes, in the player's own words where they are colourful.",
        caption_rules=[
            "in-game text (kill feed, chat, HUD, subtitles the game burned in) is NOT speech — never merge it into a caption line",
            "several players talk over each other: caption the line the reaction lands on, not the loudest voice",
            "game and item names are proper nouns; keep the exact spelling the transcript already has",
        ],
        motion_ok=True,
        notes_for_editor="punch-ins belong on the reaction word or the visual beat, never on narration",
    ),
    "podcast_interview": _g(
        key="podcast_interview",
        label="podcast / interview / talk show",
        what_matters=[
            "a claim a viewer would argue with, stated in one breath",
            "a story with tension that resolves inside the clip",
            "a confession, an admission, or something the guest clearly almost did not say",
            "disagreement between two people, with both sides present",
            "a specific number, name or outcome that makes an abstract point concrete",
        ],
        avoid=[
            "host housekeeping: intros, 'welcome back', ad breaks, subscription pleas",
            "agreement and acknowledgement: 'yeah exactly', 'for sure', 'that's right'",
            "a question whose answer is the whole point but is cut off before the answer",
            " throat-clearing at the start of an answer that never gets to the point",
        ],
        hook_angle="Lead with the claim or the consequence, not with who is speaking.",
        caption_rules=[
            "speakers overlap: one caption line is one speaker's sentence, never both at once",
            "cut 'um' and 'like' only when the sentence still reads as the speaker meant it",
            "a name or a number the guest is famous for must be transcribed exactly; if the ASR mangled it, prefer the phrase around it",
        ],
        motion_ok=False,
        notes_for_editor="a talking head takes zoom poorly; hold on faces and let the words carry it",
    ),
    "tutorial_howto": _g(
        key="tutorial_howto",
        label="tutorial / how-to / explainer",
        what_matters=[
            "the moment the thing finally works, with the result visible",
            "the mistake almost everybody makes, named and shown",
            "a specific payoff: the number, the speed-up, the cost, the output",
            "the step people skip, explained in one sentence",
        ],
        avoid=[
            "installing, opening menus, creating files, and typing boilerplate",
            "'in this video we will' and any other preview of the clip's own content",
            "a step that needs the previous three steps to mean anything",
        ],
        hook_angle="Name the concrete outcome or the mistake — never the lesson plan.",
        caption_rules=[
            "commands, file names, versions, flags and identifiers are literal: never fold them into surrounding prose or normalise their casing",
            "code terms ASR commonly mangles (camelCase, snake_case, dots, slashes) must stay readable as typed",
            "if the speech says 'click here' or 'this button', the caption must carry the actual label being referred to or the clip is unusable",
        ],
        motion_ok=False,
        notes_for_editor="screen legibility beats motion; do not zoom a screencast the viewer is trying to read",
    ),
    "news_commentary": _g(
        key="news_commentary",
        label="news / analysis / commentary",
        what_matters=[
            "the fact that changes the reader's picture, stated plainly",
            "the consequence nobody has said out loud yet",
            "a take with a defensible edge, including the disagreement",
        ],
        avoid=[
            "recapping an event the audience already knows",
            "attribution with no claim in it ('sources say', 'people are reporting')",
            "speculation that never lands on what it means for the viewer",
        ],
        hook_angle="The surprise or the stake, in one clause. No hedging.",
        caption_rules=[
            "names, organisations, figures and dates must survive ASR intact; a mangled proper noun is worse than no caption",
            "quoted numbers keep their units — 'million', '%', 'per year' are part of the fact",
        ],
        motion_ok=False,
        notes_for_editor="graphics and lower-thirds carry the meaning; keep them unobstructed",
    ),
    "vlog_story": _g(
        key="vlog_story",
        label="vlog / personal story / travel",
        what_matters=[
            "the turn: what was promised, what actually happened",
            "an emotional peak that is visible, not just described",
            "a consequence shown on screen — the broken thing, the missed flight, the surprise",
        ],
        avoid=[
            "transitions and connective tissue: 'so anyway', 'before we get into it'",
            "scene-setting with no payoff inside the clip",
            "an inside joke that needs the previous videos to land",
        ],
        hook_angle="Open on the moment everything changed, then let the setup arrive after.",
        caption_rules=[
            "a story needs its referents: a clip that opens on 'he', 'that', or 'then' is missing the line that names them",
            "background noise and music often corrupt ASR here — drop caption lines that are only noise",
        ],
        motion_ok=True,
        notes_for_editor="push-ins work on the emotional beat; don't punch in on walking and talking",
    ),
    "sports_fitness": _g(
        key="sports_fitness",
        label="sport / fitness / athletic attempt",
        what_matters=[
            "the attempt and its result, both inside the clip",
            "a record, a distance, a time, a weight — the number that makes it impressive",
            "a fall, a miss or a save: the moment the outcome was in doubt",
        ],
        avoid=[
            "warm-up, rest between sets, and equipment setup",
            "coaching talk with no visible attempt attached to it",
            "the replay of the moment already shown, cut a second time",
        ],
        hook_angle="The result first, then the effort that produced it.",
        caption_rules=[
            "crowd noise and echo destroy ASR: short shouted phrases beat a garbled long line",
            "scores, times and distances keep their units exactly as spoken",
        ],
        motion_ok=True,
        notes_for_editor="the action carries the clip; keep captions out of the play area",
    ),
    "music_performance": _g(
        key="music_performance",
        label="music / performance / DJ set",
        what_matters=[
            "the drop, the hardest passage, or the moment the room reacts",
            "a visible skill moment: a solo, a switch-up, a crowd singing along",
        ],
        avoid=[
            "tuning, soundcheck, stage banter with no music attached",
            "a clip that starts mid-phrase and cuts the resolution off",
        ],
        hook_angle="Name the track, the artist, or the feat — the sound is the pitch.",
        caption_rules=[
            "do not caption lyrics as speech: either caption nothing or caption the one line that carries the meaning",
            "ASR invents words over instrumental passages — an instrumental section with a confident transcript is a transcription error, not content",
        ],
        motion_ok=True,
        notes_for_editor="cut on the bar, not on the word; a mid-phrase edit is the only real defect here",
    ),
    "cooking_craft": _g(
        key="cooking_craft",
        label="cooking / craft / build",
        what_matters=[
            "the transformation: before and after inside one clip",
            "the failure and the reaction to it",
            "the taste or use test — the verdict, not the recipe",
        ],
        avoid=[
            "ingredient lists read aloud, prep, and measurements with no result attached",
            "a finished reveal with no process visible before it",
        ],
        hook_angle="The result or the disaster, never the ingredient list.",
        caption_rules=[
            "quantities, temperatures and timings are the useful content — keep every number and unit",
            "unfamiliar ingredient and technique names are commonly mangled by ASR; keep only lines that read correctly",
        ],
        motion_ok=True,
        notes_for_editor="a slow push onto the reveal works; keep hands and the work surface unobstructed",
    ),
    "general": _g(
        key="general",
        label="unclassifiable footage",
        what_matters=[
            "a self-contained moment that lands without any prior context",
            "emotion, surprise, or a concrete fact in the first two seconds",
        ],
        avoid=["housekeeping, outros, and anything that only makes sense to the creator"],
        hook_angle="Say what happens, specifically.",
        caption_rules=["the caption must read as a sentence a person would say out loud"],
        motion_ok=False,
        notes_for_editor="restraint: default to no effects unless the content asks for one",
    ),
}

class GenreResponse(BaseModel):
    """The classification answer.

    `genre` is a free string rather than a closed enum on purpose: the answer
    is normalised with normalize_genre, so "Gaming", "gaming / let's-play" and
    "video_game" all land on a real key. A Literal would turn a formatting
    difference into a validation error, and no part of clip selection may fail
    the job over an editorial label.
    """

    genre: str = Field(description="One of: " + ", ".join(sorted(GENRES)))
    evidence: str = Field(description="The few transcript words that decided it")


GENRE_PROMPT_TEMPLATE = """
Read the transcript excerpt below and decide what KIND of video this is.
Choose exactly one genre from:
{genre_list}

Pick "general" only if the excerpt genuinely fits nothing else. Judge the
CONTENT of the speech, not the fact that it is a video.

EXCERPT:
{sample}

Return only:
{{"genre": "<key>", "evidence": "<a few words from the excerpt>"}}
"""


STYLES = {
    "general": {
        "label": "best moments",
        "emphasis": [
            "rank by how strongly the moment stands on its own for a cold viewer",
            "variety across the source: different moments, not the same beat four times",
        ],
    },
    "funny": {
        "label": "funny moments",
        "emphasis": [
            "rank by whether the moment is actually funny: the surprise, the timing, "
            "the delivery, the fail — not by how loud it is",
            "a laugh needs a setup and a punchline inside the clip; a punchline without "
            "its setup is just noise, so extend the start to carry it",
            "prefer genuine reactions and deadpan delivery over scripted or announced jokes "
            "('this is going to be hilarious' is not the joke)",
            "an anti-climax, a mistake, or something going visibly wrong is the strongest "
            "comedy signal; a flat recap of a funny event is the weakest",
            "if a moment is exciting but not funny, it is the WRONG clip for this request",
        ],
    },
    "peak": {
        "label": "peak / highest-stakes moments",
        "emphasis": [
            "rank by stakes: the outcome must be genuinely in doubt inside the clip",
            "clutch plays, last-second results, countdowns, betrayals, reversals, and "
            "the moment a bet is called",
            "peak needs the build AND the payoff; a clip that opens after the result is "
            "already on screen has no tension left and scores low",
            "intensity of delivery is evidence, not the definition — a quiet moment with "
            "enormous stakes beats a shouted moment with none",
        ],
    },
}


def normalize_genre(raw) -> str:
    """Map any model answer onto a GENRES key; unknown -> "general".

    Models answer with labels ("Gaming", "gaming / let's-play"), sometimes with
    punctuation, so match on the leading word rather than the exact string.
    """
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in GENRES:
        return text
    if not text:
        # "" is a substring of every key, so the scan below would answer
        # "gaming" — the first entry — for an empty or blank answer.
        return "general"
    for key in GENRES:
        if key in text:
            return key
    for alias, key in _GENRE_ALIASES.items():
        if alias in text:
            return key
    first = text.split("/")[0].strip()
    for key in GENRES:
        if key.split("_")[0] == first:
            return key
    return "general"


# What the models actually say when they mean a genre above. Only used after the
# key scan, so a real key always wins over an alias.
_GENRE_ALIASES = {
    "interview": "podcast_interview",
    "talk": "podcast_interview",
    "conversation": "podcast_interview",
    "streamer": "gaming",
    "video_game": "gaming",
    "gameplay": "gaming",
    "lets_play": "gaming",
    "how_to": "tutorial_howto",
    "tutorial": "tutorial_howto",
    "educational": "tutorial_howto",
    "explainer": "tutorial_howto",
    "vlog": "vlog_story",
    "travel": "vlog_story",
    "documentary": "vlog_story",
    "news": "news_commentary",
    "politics": "news_commentary",
    "commentary": "news_commentary",
    "sport": "sports_fitness",
    "fitness": "sports_fitness",
    "gym": "sports_fitness",
    "workout": "sports_fitness",
    "music": "music_performance",
    "concert": "music_performance",
    "performance": "music_performance",
    "dj": "music_performance",
    "song": "music_performance",
    "cooking": "cooking_craft",
    "recipe": "cooking_craft",
    "woodworking": "cooking_craft",
    "diy": "cooking_craft",
    "craft": "cooking_craft",
}


def normalize_style(raw) -> str:
    """Map CONTENT_STYLE onto a STYLES key; anything unknown -> "general"."""
    text = str(raw or "").strip().lower()
    if text in STYLES:
        return text
    if text in ("comedy", "funny_moments", "humor", "humour", "lol"):
        return "funny"
    if text in ("peaks", "peak_moments", "hype", "clutch", "highlights"):
        return "peak"
    return "general"


def style_from_env() -> str:
    return normalize_style(os.environ.get("CONTENT_STYLE"))


def parse_style_param(raw):
    """Validate a request's content_style; None when absent.

    Raises ValueError rather than downgrading: a user who asked for funny
    moments and quietly got generic ones has no way to tell what happened. The
    web layer turns that into a 400 listing the choices.
    """
    text = str(raw).strip() if raw is not None else ""
    if not text or text.lower() in ("none", "null", "auto"):
        return None
    if text.lower() not in VALID_STYLES:
        raise ValueError("content_style must be one of " + ", ".join(VALID_STYLES))
    return text.lower()


def parse_genre_param(raw):
    """Validate a request's content_genre; None when absent (auto-detect)."""
    text = str(raw).strip() if raw is not None else ""
    if not text or text.lower() in ("none", "null", "auto"):
        return None
    if text.lower() not in GENRES:
        raise ValueError("content_genre must be one of " + ", ".join(sorted(GENRES)))
    return text.lower()


def genre_from_env():
    """An operator-pinned genre from CONTENT_GENRE, or None to let the model decide.

    The override exists for the material a transcript cannot classify — a
    silent screen recording, a streamer whose VOD is all game noise — and for
    re-running a job with a different lens without paying for detection.
    """
    raw = (os.environ.get("CONTENT_GENRE") or "").strip()
    if not raw:
        return None
    return normalize_genre(raw)


def get_genre(key: str) -> _Genre:
    return GENRES[normalize_genre(key)]


def _bullets(items, prefix="- "):
    return "\n".join(f"{prefix}{line}" for line in items)


def build_brief(genre_key, style_key, evidence=None) -> str:
    """The block spliced into the scoring and detail prompts.

    Returns "" when the genre and style are both the neutral default, so an
    unclassified video scores exactly the way it did before this existed.
    """
    genre = get_genre(genre_key)
    style = STYLES[normalize_style(style_key)]
    if genre.key == "general" and style["label"] == "best moments":
        return ""

    lines = [
        "EDITORIAL BRIEF — derived from this video, applies to every decision below.",
        "",
        f"GENRE: {genre.label}" + (f" (evidence: {evidence})" if evidence else ""),
        "",
        "WHAT MAKES A CLIP WORK IN THIS VIDEO:",
        _bullets(genre.what_matters),
        "",
        "DO NOT PICK:",
        _bullets(genre.avoid),
        "",
        "WHAT THE USER ASKED FOR: {label}.".format(label=style["label"]),
        _bullets(style["emphasis"]),
        "",
        "CAPTION RULES FOR THIS GENRE — a caption that breaks these does not read "
        "as speech and the clip is not publishable:",
        _bullets(genre.caption_rules),
        "",
        f"HOOK ANGLE: {genre.hook_angle}",
        "",
        "HOW TO SCORE A WINDOW AGAINST THE ABOVE: the generic viral criteria "
        "(2-second hook, stands alone, payoff) still apply, but they are gates, not "
        "the ranking. Rank by what matters in this genre and by what the user asked "
        "for; a window that passes the gates but misses both scores below one that "
        "hits them. When two windows are equally viral, the one that fits the brief "
        "wins.",
        "",
        "MOTION IN THIS GENRE: {}. {}".format(
            "punch-ins and short zooms help" if genre.motion_ok
            else "zoom and punch-in are usually a defect here",
            genre.notes_for_editor),
    ]
    return "\n".join(lines)


def caption_rules_for(genre_key) -> list:
    return get_genre(genre_key).caption_rules


def genre_labels() -> str:
    """"gaming, podcast_interview, ..." — the closed list handed to the model."""
    return ", ".join(sorted(GENRES))


def apply_brief(prompt: str, brief: str) -> str:
    """Splice the editorial brief into a formatted prompt.

    Deliberately a replace of a sentinel rather than a new ``.format()`` field:
    every existing call site and test formats these templates without knowing
    the brief exists, and an empty brief leaves the prompt byte-identical to
    what it rendered before.
    """
    if not brief:
        return prompt.replace(BRIEF_MARKER, "")
    return prompt.replace(BRIEF_MARKER, "\n" + brief.strip() + "\n")


def prompt_sample(transcript_result, chars_per_slice=900, slices=3):
    """A cheap, representative slice of the transcript for classification.

    Beginning, middle and end, joined with markers: genre shows up in the
    opening framing, the middle's routine, and the close's payoff. Sampling
    keeps the classification call inside a small token budget, which matters
    on a free-tier per-minute limit.
    """
    text = str((transcript_result or {}).get("text") or "").strip()
    if not text:
        segments = (transcript_result or {}).get("segments") or []
        text = " ".join(str(s.get("text") or "").strip() for s in segments)
        text = text.strip()
    if len(text) <= chars_per_slice * slices:
        return text
    if slices <= 1:
        return text[:chars_per_slice] + " [..]"
    step = (len(text) - chars_per_slice) / float(slices - 1)
    picks = [text[int(i * step):int(i * step) + chars_per_slice]
             for i in range(slices)]
    # The last slice ends where the transcript does: the close of a video is
    # where a payoff or an outro lives, and a step-sized window that stops
    # short of it is the one part of the sample that must not be missing.
    picks[-1] = text[-chars_per_slice:]
    return " [..] ".join(picks)
