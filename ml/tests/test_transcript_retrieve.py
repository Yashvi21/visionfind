"""Transcripts (chunking, indexing, no-audio) and /retrieve for Q&A."""

from fractions import Fraction

import numpy as np
import pytest

from app import worker as worker_mod
from app.db import SessionLocal
from app.models import Status, TranscriptSegment, Video
from app.transcribe import Segment, chunk_segments, has_audio

from .conftest import FakeTranscriber, upload, upload_ready


def _segs(spans):
    return [Segment(a, b, f"s{i}") for i, (a, b) in enumerate(spans)]


# -- chunking ---------------------------------------------------------------
def test_chunks_are_20_to_30_seconds():
    segs = _segs([(i * 4, i * 4 + 4) for i in range(30)])  # 120 s of 4 s segments
    chunks = chunk_segments(segs)
    assert len(chunks) == 6
    assert all(20 <= c.end - c.start <= 30 for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == 120
    assert " ".join(c.text for c in chunks) == " ".join(s.text for s in segs)  # nothing lost or reordered


def test_chunk_never_exceeds_max_unless_single_long_segment():
    chunks = chunk_segments(_segs([(0, 15), (15, 35), (35, 80), (80, 85)]))
    assert [(c.start, c.end) for c in chunks] == [(0, 15), (15, 35), (35, 80), (80, 85)]


def test_short_tail_and_blank_text():
    chunks = chunk_segments([Segment(0, 5, "hello"), Segment(5, 6, "   "), Segment(6, 9, "world")])
    assert [(c.start, c.end, c.text) for c in chunks] == [(0, 9, "hello world")]
    assert chunk_segments([]) == []


# -- audio detection ----------------------------------------------------------
def _mp4_with_audio(path, seconds=2):
    import av

    with av.open(str(path), "w") as out:
        v = out.add_stream("libx264", rate=10)
        v.width, v.height, v.pix_fmt = 64, 64, "yuv420p"
        a = out.add_stream("aac", rate=16000)
        for i in range(seconds * 10):
            for p in v.encode(av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")):
                out.mux(p)
        samples = np.zeros((1, 1024), np.float32)
        for i in range(seconds * 16000 // 1024):
            f = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            f.sample_rate, f.pts, f.time_base = 16000, i * 1024, Fraction(1, 16000)
            for p in a.encode(f):
                out.mux(p)
        for s in (v, a):
            for p in s.encode():
                out.mux(p)
    return path


def test_has_audio(tmp_path, video_file):
    assert has_audio(_mp4_with_audio(tmp_path / "a.mp4")) is True
    assert has_audio(video_file(2)) is False  # OpenCV writes video-only files


# -- worker -------------------------------------------------------------------
@pytest.fixture
def with_audio(monkeypatch):
    """Pretend synthetic (silent, video-only) uploads have audio, and chunk small so each scripted
    segment becomes its own chunk."""
    monkeypatch.setattr(worker_mod, "has_audio", lambda p: True)
    monkeypatch.setattr(worker_mod, "chunk_segments", lambda segs: chunk_segments(segs, min_sec=3, max_sec=5))


def _segments(vid):
    with SessionLocal() as s:
        return s.query(TranscriptSegment).filter_by(video_id=vid).order_by(TranscriptSegment.start_sec).all()


def test_transcript_indexed_with_db_ids(client, video_file, with_audio):
    vid = upload_ready(client, video_file(12), interval=1)
    segs = _segments(vid)
    assert [(s.start_sec, s.end_sec, s.text) for s in segs] == FakeTranscriber.script
    assert client.app.state.services.transcript_index.ids() == {s.id for s in segs}


def test_no_audio_skips_transcription(client, video_file):
    before = FakeTranscriber.calls
    vid = upload_ready(client, video_file(4), interval=1)
    assert FakeTranscriber.calls == before
    assert _segments(vid) == []
    assert client.app.state.services.transcript_index.ntotal == 0


def test_transcription_failure_rolls_back_everything(client, video_file, with_audio, monkeypatch):
    def boom(self, path):
        raise RuntimeError("whisper crashed")

    monkeypatch.setattr(FakeTranscriber, "transcribe", boom)
    vid = upload(client, video_file(6), interval=1).json()["id"]
    assert client.app.state.worker.wait_idle(60)
    video = client.get(f"/videos/{vid}").json()
    assert video["status"] == "failed" and "whisper crashed" in video["error_message"]
    services = client.app.state.services
    assert services.frame_index.ntotal == 0 and services.transcript_index.ntotal == 0
    assert _segments(vid) == []


def test_delete_removes_transcript_vectors(client, video_file, with_audio):
    vid = upload_ready(client, video_file(12), interval=1)
    assert client.app.state.services.transcript_index.ntotal == 3
    assert client.delete(f"/videos/{vid}").status_code == 204
    assert client.app.state.services.transcript_index.ntotal == 0


# -- /retrieve -------------------------------------------------------------------
def test_retrieve_frames_matches_and_overlaps(client, video_file, with_audio):
    vid = upload_ready(client, video_file(12), interval=1)
    upload_ready(client, video_file(12, name="other.mp4"), interval=1)  # must never leak in

    r = client.post("/retrieve", json={"video_id": vid, "question": "second 2 budget", "k_frames": 1, "k_chunks": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_transcript"] is True and body["video_id"] == vid
    assert [f["timestamp_sec"] for f in body["frames"]] == [2]
    assert body["frames"][0]["timestamp_label"] == "00:02"
    # "budget" matches the 4-8 s chunk by text; the frame at 2 s overlaps the 0-4 s chunk.
    assert [(c["start_sec"], c["reason"]) for c in body["chunks"]] == [(0.0, "overlap"), (4.0, "match")]
    assert body["chunks"][1]["text"] == "budget numbers for next year"


def test_retrieve_defaults_sorted_by_time_and_scoped_to_video(client, video_file, with_audio):
    vid = upload_ready(client, video_file(12), interval=1)
    other = upload_ready(client, video_file(12, name="other.mp4"), interval=1)
    body = client.post("/retrieve", json={"video_id": vid, "question": "second 6 hiring"}).json()
    ts = [f["timestamp_sec"] for f in body["frames"]]
    assert 1 <= len(ts) <= 6 and ts == sorted(ts)
    starts = [c["start_sec"] for c in body["chunks"]]
    assert starts == sorted(starts)
    with SessionLocal() as s:
        other_ids = {x.id for x in s.query(TranscriptSegment).filter_by(video_id=other)}
    assert not other_ids & {c["segment_id"] for c in body["chunks"]}


def test_retrieve_without_transcript(client, video_file):
    vid = upload_ready(client, video_file(6), interval=1)
    body = client.post("/retrieve", json={"video_id": vid, "question": "second 3"}).json()
    assert body["has_transcript"] is False and body["chunks"] == [] and body["frames"]


def test_retrieve_errors(client, video_file):
    assert client.post("/retrieve", json={"video_id": "nope", "question": "x"}).status_code == 404
    vid = upload_ready(client, video_file(3), interval=1)
    assert client.post("/retrieve", json={"video_id": vid, "question": "   "}).status_code == 400
    for status in (Status.PROCESSING, Status.QUEUED, Status.FAILED):
        with SessionLocal() as s:
            s.get(Video, vid).status = status
            s.commit()
        r = client.post("/retrieve", json={"video_id": vid, "question": "second 1"})
        assert r.status_code == 409, status  # never partial results


def test_transcript_index_survives_restart(video_file, with_audio):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        vid = upload_ready(c, video_file(12), interval=1)
        before = c.post("/retrieve", json={"video_id": vid, "question": "hiring engineers", "k_frames": 1, "k_chunks": 1}).json()
    with TestClient(app) as c:  # indexes reloaded from disk
        assert c.app.state.services.transcript_index.ids() == {s.id for s in _segments(vid)}
        after = c.post("/retrieve", json={"video_id": vid, "question": "hiring engineers", "k_frames": 1, "k_chunks": 1}).json()
    assert after == before
    assert any(ch["reason"] == "match" and "hiring" in ch["text"] for ch in after["chunks"])
