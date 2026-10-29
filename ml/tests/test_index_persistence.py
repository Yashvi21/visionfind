"""Indexes are saved to disk and reload correctly; startup reconciles them with the database."""

import numpy as np
from fastapi.testclient import TestClient

from app import config
from app.db import SessionLocal
from app.main import app
from app.models import Status, Video
from app.vector_index import VectorIndex

from .conftest import _bump, search, upload_ready


def test_vector_index_save_load_roundtrip(tmp_path):
    path = tmp_path / "frames.faiss"
    idx = VectorIndex(512, path)
    idx.add([5, 99, 3], np.stack([_bump(1), _bump(5), _bump(9)]))
    idx.save()
    assert not path.with_suffix(".faiss.tmp").exists()

    loaded = VectorIndex.load_or_create(512, path)
    assert loaded.ids() == {5, 99, 3}
    assert loaded.search(_bump(5), 1)[0][0] == 99


def test_missing_index_starts_empty(tmp_path):
    assert VectorIndex.load_or_create(512, tmp_path / "nope.faiss").ntotal == 0


def test_search_results_identical_after_restart(video_file):
    with TestClient(app) as c:
        upload_ready(c, video_file(8, name="a.mp4"), interval=1)
        upload_ready(c, video_file(6, name="b.mp4"), interval=2)
        before = search(c, "second 3", top_k=20, dedupe_window_sec=0)
        ntotal = c.app.state.services.frame_index.ntotal
    assert (config.INDEX_DIR / "frames.faiss").exists()

    with TestClient(app) as c:  # fresh Services: indexes loaded from disk
        assert c.app.state.services.frame_index.ntotal == ntotal == 11
        after = search(c, "second 3", top_k=20, dedupe_window_sec=0)
    assert after == before


def test_startup_drops_stale_vectors(video_file):
    with TestClient(app) as c:
        upload_ready(c, video_file(4), interval=1)
        services = c.app.state.services
        services.frame_index.add([987654], _bump(2)[None, :])  # no such Frame row
        services.save_indexes()

    with TestClient(app) as c:
        ids = c.app.state.services.frame_index.ids()
        assert 987654 not in ids and len(ids) == 4


def test_ready_video_with_missing_vectors_is_reindexed(video_file):
    with TestClient(app) as c:
        vid = upload_ready(c, video_file(5), interval=1)
    # Simulate losing the index file (e.g. deleted by hand).
    (config.INDEX_DIR / "frames.faiss").unlink()

    with TestClient(app) as c:
        assert c.app.state.worker.wait_idle(60)
        with SessionLocal() as s:
            assert s.get(Video, vid).status == Status.READY
        assert c.app.state.services.frame_index.ntotal == 5
        assert search(c, "second 2", top_k=1)[0]["timestamp_sec"] == 2
