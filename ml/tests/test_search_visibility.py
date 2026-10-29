"""Search must never return frames of non-ready or deleted videos."""

import uuid

import numpy as np

from app import config
from app.db import SessionLocal
from app.models import Frame, Status, Video

from .conftest import FakeClip, _bump, search, upload, upload_ready


def _insert_unready_video_with_vectors(client, status: str) -> str:
    """Simulate a video mid-indexing (or failed) whose vectors are already in the index."""
    vid = str(uuid.uuid4())
    with SessionLocal() as s:
        s.add(Video(id=vid, filename=f"{status}.mp4", stored_path="videos/none.mp4", duration_sec=5,
                    size_bytes=1, sample_interval_sec=1, status=status, stage="embedding"))
        frames = [Frame(video_id=vid, timestamp_sec=t, frame_path="frames/none.jpg") for t in range(5)]
        s.add_all(frames)
        s.commit()
        ids = [f.id for f in frames]
    # Perfect matches for "second 3" so they'd rank first if they leaked through.
    client.app.state.services.frame_index.add(ids, np.stack([_bump(3)] * len(ids)))
    return vid


def test_processing_queued_and_failed_videos_are_not_searchable(client, video_file):
    ready = upload_ready(client, video_file(8), interval=1)
    hidden = [_insert_unready_video_with_vectors(client, st) for st in (Status.PROCESSING, Status.QUEUED, Status.FAILED)]

    results = search(client, "second 3", top_k=50, dedupe_window_sec=0)
    assert results
    assert {r["video_id"] for r in results} == {ready}
    assert not set(hidden) & {r["video_id"] for r in results}

    # Filtering on a non-ready video is refused rather than leaking results.
    r = client.post("/search", json={"query": "second 3", "video_id": hidden[0]})
    assert r.status_code == 409


def test_deleted_video_is_gone_everywhere(client, video_file):
    keep = upload_ready(client, video_file(6, name="keep.mp4"), interval=1)
    gone = upload_ready(client, video_file(6, name="gone.mp4"), interval=1)
    services = client.app.state.services
    assert services.frame_index.ntotal == 12

    with SessionLocal() as s:
        gone_ids = {f.id for f in s.query(Frame).filter_by(video_id=gone)}
    assert client.delete(f"/videos/{gone}").status_code == 204

    assert services.frame_index.ntotal == 6
    assert not gone_ids & services.frame_index.ids()
    assert {r["video_id"] for r in search(client, "second 2", top_k=50, dedupe_window_sec=0)} == {keep}
    assert client.get(f"/videos/{gone}").status_code == 404
    assert not (config.FRAMES_DIR / gone).exists()
    assert not (config.VIDEOS_DIR / f"{gone}.mp4").exists()
    for fid in gone_ids:
        assert client.get(f"/frames/{fid}/image").status_code == 404
    assert client.delete(f"/videos/{gone}").status_code == 404


def test_no_ready_videos_returns_empty(client):
    assert search(client, "second 1") == []


def test_video_filter(client, video_file):
    a = upload_ready(client, video_file(6, name="a.mp4"), interval=1)
    upload_ready(client, video_file(6, name="b.mp4"), interval=1)
    results = search(client, "second 2", top_k=50, video_id=a, dedupe_window_sec=0)
    assert len(results) == 6
    assert {r["video_id"] for r in results} == {a}
    assert client.post("/search", json={"query": "x", "video_id": "nope"}).status_code == 404


def test_search_does_not_embed_frames(client, video_file):
    upload_ready(client, video_file(6), interval=1)
    before = FakeClip.calls["images"]
    search(client, "second 2")
    search(client, "second 4", top_k=5)
    assert FakeClip.calls["images"] == before  # only the text query is embedded


def test_reindex_with_new_interval_replaces_everything(client, video_file):
    vid = upload_ready(client, video_file(10), interval=1)
    services = client.app.state.services
    with SessionLocal() as s:
        old_ids = {f.id for f in s.query(Frame).filter_by(video_id=vid)}
    assert len(old_ids) == 10

    r = client.post(f"/videos/{vid}/reindex", json={"sample_interval_sec": 5})
    assert r.status_code == 202 and r.json()["status"] == "queued"
    assert client.app.state.worker.wait_idle(60)

    video = client.get(f"/videos/{vid}").json()
    assert (video["status"], video["frame_count"], video["sample_interval_sec"]) == ("ready", 2, 5)
    with SessionLocal() as s:
        new_ids = {f.id for f in s.query(Frame).filter_by(video_id=vid)}
    assert not old_ids & new_ids
    assert min(new_ids) > max(old_ids)  # ids are never reused
    assert services.frame_index.ids() == new_ids
    assert [r["timestamp_sec"] for r in search(client, "second 5", top_k=1)] == [5]


def test_reindex_rejects_bad_interval_and_busy_video(client, video_file):
    vid = upload_ready(client, video_file(4), interval=1)
    assert client.post(f"/videos/{vid}/reindex", json={"sample_interval_sec": 3}).status_code == 400
    with SessionLocal() as s:
        s.get(Video, vid).status = Status.PROCESSING
        s.commit()
    assert client.post(f"/videos/{vid}/reindex").status_code == 409
    with SessionLocal() as s:
        s.get(Video, vid).status = Status.READY
        s.commit()


class _BrokenClip(FakeClip):
    def encode_images(self, paths):
        raise RuntimeError("model exploded")


def test_failure_leaves_nothing_searchable_and_retry_recovers(client, video_file):
    services = client.app.state.services
    services._clip.value = _BrokenClip()
    r = upload(client, video_file(5), interval=1)
    vid = r.json()["id"]
    assert client.app.state.worker.wait_idle(60)

    video = client.get(f"/videos/{vid}").json()
    assert video["status"] == "failed"
    assert "model exploded" in video["error_message"]
    assert video["frame_count"] == 0
    assert services.frame_index.ntotal == 0
    assert not (config.FRAMES_DIR / vid).exists()
    assert search(client, "second 1") == []

    services._clip.value = FakeClip()  # "fix" the problem, then retry
    assert client.post(f"/videos/{vid}/reindex").status_code == 202
    assert client.app.state.worker.wait_idle(60)
    assert client.get(f"/videos/{vid}").json()["status"] == "ready"
    assert search(client, "second 1", top_k=1)[0]["timestamp_sec"] == 1


def test_resolve_handles_more_ids_than_sqlite_variable_limit(client, video_file):
    """Over-fetching on a big index must never build an IN (...) list SQLite rejects."""
    from app.search import _resolve

    vid = upload_ready(client, video_file(4), interval=1)
    with SessionLocal() as s:
        real = [f.id for f in s.query(Frame).filter_by(video_id=vid)]
        fake = list(range(10_000_000, 10_040_000))  # 40k ids that don't exist
        hits = _resolve(s, [(i, 0.5) for i in fake + real], None)
    assert sorted(h.frame_id for h in hits) == sorted(real)
