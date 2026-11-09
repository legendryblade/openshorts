"""The self-host Library: GET /api/jobs rebuilds history from disk.

The live ``jobs`` dict dies with the process; the Library view reads this
endpoint instead, so it must resolve every clip to the canonical (newest
derived) file exactly like the download bundle does, and refuse to list
anything in cloud mode where projects are R2-owned.
"""
import asyncio
import json
import os

import httpx
import pytest

app_module = pytest.importorskip("app")


@pytest.fixture
def out(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    return root


def _seed(out, job_id, base="vid_1", clips=1):
    d = out / job_id
    d.mkdir()
    shorts = [{"start": i * 10.0, "end": i * 10.0 + 5.0,
               "video_title_for_youtube_short": f"Clip {i}",
               "viral_hook_text": "hook 😳"} for i in range(clips)]
    (d / f"{base}_metadata.json").write_text(json.dumps({"shorts": shorts}))
    for i in range(clips):
        (d / f"{base}_clip_{i + 1}.mp4").write_bytes(b"\x00")  # existence is all the API checks
    return d


def _get(path="/api/jobs"):
    async def go():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.get(path)
    return asyncio.run(go())


class TestJobListing:

    def test_no_output_dir_means_empty_library(self, out):
        assert _get().json() == {"jobs": []}

    def test_dirs_without_metadata_are_not_jobs(self, out):
        (out / "stray_dir").mkdir()
        assert _get().json() == {"jobs": []}

    def test_seeded_job_comes_back_with_its_clips(self, out):
        _seed(out, "job-abc", base="Great_Video", clips=2)
        jobs = _get().json()["jobs"]
        assert len(jobs) == 1
        job = jobs[0]
        assert job["job_id"] == "job-abc"
        assert job["label"] == "Great_Video"
        assert len(job["clips"]) == 2
        first = job["clips"][0]
        assert first["video_url"] == "/videos/job-abc/Great_Video_clip_1.mp4"
        assert first["title"] == "Clip 0"
        assert first["viral_hook_text"] == "hook 😳"

    def test_derived_captioned_file_wins_over_the_clean_clip(self, out):
        # The pipeline keeps <base>_clip_N.mp4 and writes
        # subtitled_<ts>_<base>_clip_N.mp4 on top; the Library must serve the
        # captioned one, or history silently loses captions.
        d = _seed(out, "job-1", base="v", clips=1)
        (d / "subtitled_1700000000_v_clip_1.mp4").write_bytes(b"\x00")
        clip = _get().json()["jobs"][0]["clips"][0]
        assert clip["video_url"] == "/videos/job-1/subtitled_1700000000_v_clip_1.mp4"

    def test_newest_job_first(self, out):
        _seed(out, "job-old")
        os.utime(out / "job-old" / "vid_1_metadata.json", (1000, 1000))
        _seed(out, "job-new")
        os.utime(out / "job-new" / "vid_1_metadata.json", (9_000_000_000, 9_000_000_000))
        jobs = _get().json()["jobs"]
        assert [j["job_id"] for j in jobs] == ["job-new", "job-old"]

    def test_cloud_mode_refuses_to_list(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
        _seed(out, "job-abc")
        resp = _get()
        assert resp.status_code == 403
