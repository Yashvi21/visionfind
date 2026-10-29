"""FAISS vector id == SQLite primary key, never insertion order."""

import numpy as np
import pytest

from app.db import SessionLocal
from app.models import Frame
from app.vector_index import VectorIndex

from .conftest import _bump, search, upload_ready


def _vec(center):
    return _bump(center)[None, :]


def test_ids_are_the_given_ids_not_positions(tmp_path):
    idx = VectorIndex(512, tmp_path / "i.faiss")
    idx.add([1000, 7, 42], np.stack([_bump(1), _bump(5), _bump(9)]))
    assert idx.search(_vec(5), 1) == [(7, pytest.approx(1.0, abs=1e-5))]
    assert idx.search(_vec(9), 1)[0][0] == 42
    assert idx.search(_vec(1), 1)[0][0] == 1000


def test_remove_does_not_shift_other_ids(tmp_path):
    idx = VectorIndex(512, tmp_path / "i.faiss")
    idx.add([10, 20, 30], np.stack([_bump(1), _bump(5), _bump(9)]))
    assert idx.remove([20]) == 1
    assert idx.ids() == {10, 30}
    assert idx.search(_vec(9), 1)[0][0] == 30  # would be wrong if ids followed positions


def test_vectors_are_normalized_and_add_is_idempotent(tmp_path):
    idx = VectorIndex(512, tmp_path / "i.faiss")
    idx.add([1], _vec(3) * 37.0)  # unnormalized input
    idx.add([1], _vec(3))  # re-adding the same id replaces, never duplicates
    assert idx.ntotal == 1
    (_, score), = idx.search(_vec(3) * 0.01, 1)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_restrict_ids(tmp_path):
    idx = VectorIndex(512, tmp_path / "i.faiss")
    idx.add([1, 2, 3], np.stack([_bump(1), _bump(5), _bump(9)]))
    hits = idx.search(_vec(1), 3, restrict_ids=[2, 3])
    assert {i for i, _ in hits} == {2, 3}


def test_wrong_dim_rejected(tmp_path):
    with pytest.raises(ValueError):
        VectorIndex(512, tmp_path / "i.faiss").add([1], np.ones((1, 384)))


def test_every_result_maps_to_its_own_frame_row(client, video_file):
    """End to end: each hit's frame_id resolves to the row (video + timestamp) the result claims,
    and the image behind it is from that second."""
    a = upload_ready(client, video_file(10, name="a.mp4"), interval=1)
    b = upload_ready(client, video_file(6, name="b.mp4"), interval=2)

    for second in (0, 3, 4, 8):
        results = search(client, f"second {second}", top_k=20, dedupe_window_sec=0)
        assert results, second
        with SessionLocal() as s:
            for r in results:
                row = s.get(Frame, r["frame_id"])
                assert row.video_id == r["video_id"]
                assert row.timestamp_sec == r["timestamp_sec"]
                assert r["video_filename"] == ("a.mp4" if row.video_id == a else "b.mp4")
                assert r["thumbnail_url"] == f"/ml/frames/{row.id}/image"
        top = results[0]
        assert top["timestamp_sec"] == second
        assert top["similarity"] == pytest.approx(1.0, abs=0.01)
    assert {r["video_id"] for r in search(client, "second 2", top_k=50, dedupe_window_sec=0)} == {a, b}


def test_timestamp_label(client, video_file):
    upload_ready(client, video_file(8), interval=1)
    top = search(client, "second 7", top_k=1)[0]
    assert top["timestamp_label"] == "00:07"
