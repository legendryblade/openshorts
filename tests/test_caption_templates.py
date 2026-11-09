"""Caption templates: named looks for the karaoke burn.

A template can never break the burn — it only supplies style fields to the
existing generate_ass path, and an unknown name degrades to the default look
(a bad template must never cost the clip its captions).
"""
import asyncio
import json
import re

import httpx
import pytest

import subtitles
from subtitles import CAPTION_TEMPLATES, AUTO_CAPTION_STYLE, get_caption_style

REQUIRED_FIELDS = ("font_name", "font_size", "font_color", "highlight_color",
                   "border_color", "border_width", "effect", "base_opacity",
                   "uppercase", "max_chars", "max_duration")


class TestRegistry:
    def test_at_least_eight_templates(self):
        assert len(CAPTION_TEMPLATES) >= 8

    def test_every_template_is_complete_and_safe(self):
        for name, style in CAPTION_TEMPLATES.items():
            for field in REQUIRED_FIELDS:
                assert field in style, f"{name} missing {field}"
            assert style["effect"] in {"none", "glow", "pop", "box"}, name
            for color in ("font_color", "highlight_color", "border_color"):
                assert re.fullmatch(r"#[0-9A-Fa-f]{6}", style[color]), \
                    f"{name}.{color} not a hex colour"
            assert 20 <= style["font_size"] <= 60, name
            assert 0.05 <= style["base_opacity"] <= 1.0, name
            assert style["max_chars"] > 0 and style["max_duration"] > 0, name

    def test_default_template_is_the_original_look(self):
        assert CAPTION_TEMPLATES["default"] == AUTO_CAPTION_STYLE


class TestResolution:
    def test_explicit_name_wins(self, monkeypatch):
        monkeypatch.setenv("CAPTION_TEMPLATE", "default")
        style = get_caption_style("hormozi")
        assert style["highlight_color"] == "#FFE500"
        assert style["base_opacity"] == 0.45  # the dimmed-base signature

    def test_env_var_selects_when_no_name(self, monkeypatch):
        monkeypatch.setenv("CAPTION_TEMPLATE", "neon_glow")
        assert get_caption_style()["highlight_color"] == "#00FFF7"

    def test_unknown_name_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("CAPTION_TEMPLATE", raising=False)
        style = get_caption_style("does-not-exist")
        assert style == AUTO_CAPTION_STYLE

    def test_name_is_case_and_space_insensitive(self):
        assert get_caption_style("  TikTok_Bold ")["font_size"] == 50

    def test_returns_a_copy_not_the_registry(self):
        style = get_caption_style("hormozi")
        style["font_size"] = 999
        assert CAPTION_TEMPLATES["hormozi"]["font_size"] != 999

    def test_malformed_fields_repaired_to_safe_values(self, monkeypatch):
        monkeypatch.setenv("CAPTION_TEMPLATE", "default")
        subtitles.CAPTION_TEMPLATES["broken"] = {
            **AUTO_CAPTION_STYLE, "effect": "explode", "font_color": "nope"}
        try:
            style = get_caption_style("broken")
            assert style["effect"] == "pop"
            assert style["font_color"] == "#FFFFFF"
        finally:
            del subtitles.CAPTION_TEMPLATES["broken"]


class TestBurnIntegration:
    def test_ass_carries_the_template_look(self, tmp_path):
        style = get_caption_style("hormozi")
        transcript = {"segments": [{"start": 0.0, "end": 2.0, "text": "hello world",
                                    "words": [{"word": " hello", "start": 0.0, "end": 0.8},
                                              {"word": " world", "start": 0.9, "end": 1.8}]}]}
        out = tmp_path / "subs.ass"
        assert subtitles.generate_ass(
            transcript, 0, 2.0, str(out),
            alignment=style["alignment"], fontsize=style["font_size"],
            font_name=style["font_name"], font_color=style["font_color"],
            highlight_color=style["highlight_color"],
            border_color=style["border_color"], border_width=style["border_width"],
            effect=style["effect"], base_opacity=style["base_opacity"],
            uppercase=style["uppercase"]) is True
        text = out.read_text(encoding="utf-8-sig")
        style_line = [l for l in text.splitlines() if l.startswith("Style: Default")][0]
        # Anton at half size (44 * 0.85), dimmed primary colour — the template
        # fields reach the ASS with their conversions applied.
        assert "Anton" in style_line
        assert "HELLO" in text and "WORLD" in text  # uppercase flag applied
        assert "\\c" in text  # the active-word colour override exists

    def test_active_word_uses_the_template_highlight(self, tmp_path):
        style = get_caption_style("mrbeast")
        transcript = {"segments": [{"start": 0.0, "end": 1.0, "text": "go",
                                    "words": [{"word": " go", "start": 0.0, "end": 0.9}]}]}
        out = tmp_path / "subs.ass"
        subtitles.generate_ass(transcript, 0, 1.0, str(out),
                               font_color=style["font_color"],
                               highlight_color=style["highlight_color"],
                               effect=style["effect"], uppercase=True)
        # #00FF66 → the inline \c override uses the 6-digit BBGGRR form
        assert "&H66FF00&" in out.read_text(encoding="utf-8-sig")


class TestEndpoint:
    def _get(self, path):
        async def go():
            transport = httpx.ASGITransport(app=pytest.importorskip("app").app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.get(path)
        return asyncio.run(go())

    def test_endpoint_lists_registry_and_active(self, monkeypatch):
        monkeypatch.setenv("CAPTION_TEMPLATE", "hormozi")
        data = self._get("/api/caption-templates").json()
        names = [t["name"] for t in data["templates"]]
        assert "hormozi" in names and "default" in names
        assert data["active"] == "hormozi"
        for t in data["templates"]:
            assert {"font_color", "highlight_color", "effect"} <= set(t)
