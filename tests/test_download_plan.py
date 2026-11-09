"""Download attempt ordering: cheapest bandwidth first.

Direct (server IP) -> flat-rate static ISP proxies (uncapped 1080p, free
bytes) -> per-GB paid proxy (720p cost cap) -> conservative fallback. The
``capped`` flag marks the attempts whose bytes are billed per GB (drives the
720p format cap and the PROXY_BYTES monthly counter).
"""
import os

import pytest

main = pytest.importorskip("main")

plan = main.plan_download_attempts

STATICS = ["http://s1", "http://s2", "http://s3"]
PAID = "http://paid"


class TestOrdering:
    def test_full_chain(self):
        got = plan(True, STATICS, PAID, True)
        assert got == [
            ('HD-direct', False, None),
            ('HD-static1', False, "http://s1"),
            ('HD-static2', False, "http://s2"),
            ('HD-static3', False, "http://s3"),
            # Free rescue for the fake "Video unavailable" the HD client gets
            # from datacenter-ISP IPs (4-sep-2026): conservative clients
            # through a static BEFORE any per-GB attempt.
            ('fallback-static', False, "http://s1"),
            ('HD', True, PAID),
            ('fallback', True, PAID),
        ]

    def test_statics_only_fallback_uses_a_static(self):
        got = plan(False, STATICS, None, True)
        assert got[-1] == ('fallback', False, "http://s1")
        assert ('HD', False, None) in got  # direct HD still tried, uncapped

    def test_selfhost_no_proxies_matches_legacy_chain(self):
        assert plan(False, [], None, True) == [
            ('HD', False, None),
            ('fallback', False, None),
        ]

    def test_no_hd_args_still_has_fallback(self):
        assert plan(False, [], PAID, False) == [('fallback', True, PAID)]

    def test_paid_attempts_are_the_only_capped_ones(self):
        got = plan(True, STATICS, PAID, True)
        for _label, capped, proxy in got:
            assert capped == (proxy == PAID)


def test_fallback_static_only_exists_when_paid_is_configured():
    # Without a paid proxy the final fallback already runs through a static;
    # a fallback-static before it would be the same attempt twice.
    got = plan(False, STATICS, None, True)
    assert ('fallback-static', False, "http://s1") not in got
    assert got[-1] == ('fallback', False, "http://s1")


def test_direct_file_urls_skip_the_proxy_chain():
    """A catbox/tmpfiles/CDN mp4 has no YouTube ban to dodge: own IP first,
    one static as fallback, never the per-GB proxy (prod 27-aug: 9 KB/s
    through the ISP proxy vs 17 MB/s direct)."""
    got = plan(False, ["s1", "s2"], "paid", True, youtube=False)
    assert got == [("direct", False, None), ("static-fallback", False, "s1")]
    assert plan(False, [], "paid", True, youtube=False) == [("direct", False, None)]
    assert main.is_youtube_url("https://www.youtube.com/watch?v=x")
    assert main.is_youtube_url("https://youtu.be/x")
    assert not main.is_youtube_url("https://litter.catbox.moe/u90j4q.mp4")
    assert not main.is_youtube_url("https://tmpfiles.org/dl/1/2/v.mp4")


class TestCookieResolution:
    """YOUTUBE_COOKIES is the one durable fix for the "Sign in to confirm
    you're not a bot" wall. It used to accept only pasted Netscape content and
    wrote it to /app/cookies.txt — a path that exists only inside the Docker
    image, so every self-hosted box silently downloaded with NO cookies (the
    Windows write failed and the path became None)."""

    def test_path_form_is_used_as_is(self, tmp_path, monkeypatch):
        f = tmp_path / "cookies.txt"
        f.write_text("# Netscape HTTP Cookie File\n")
        monkeypatch.setenv("YOUTUBE_COOKIES", str(f))
        path, msg = main.resolve_youtube_cookies()
        assert path == str(f)
        assert "Using cookies file" in msg

    def test_pasted_content_lands_in_a_writable_dir(self, tmp_path, monkeypatch):
        blob = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tsid\tx\n"
        monkeypatch.setenv("YOUTUBE_COOKIES", blob)
        # tempfile caches its dir on first use; pin the cache so the test is
        # order-independent.
        import tempfile
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        path, msg = main.resolve_youtube_cookies()
        assert path and os.path.isfile(path)
        assert open(path, encoding="utf-8").read() == blob
        assert "Wrote YOUTUBE_COOKIES content" in msg
        assert "sid" not in msg, "status must never echo live session cookies"

    def test_unset_means_no_cookies_and_a_clear_warning(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_COOKIES", raising=False)
        path, msg = main.resolve_youtube_cookies()
        assert path is None
        assert "not found" in msg

    def test_no_container_path_leaks_into_the_result(self, monkeypatch):
        # The old code wrote to /app/cookies.txt; on Windows that write fails
        # and cookies silently vanish. The temp-dir location must win.
        monkeypatch.setenv("YOUTUBE_COOKIES", "# Netscape HTTP Cookie File\n")
        path, _ = main.resolve_youtube_cookies()
        assert path is None or "/app/" not in path
