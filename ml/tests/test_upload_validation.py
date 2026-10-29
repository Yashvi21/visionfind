from app import config

from .conftest import upload


def _stored_videos():
    return list(config.VIDEOS_DIR.iterdir())


def test_valid_mp4_is_accepted_and_queued(client, video_file):
    r = upload(client, video_file(6), interval=2)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] in ("queued", "processing")
    assert body["filename"] == "clip.mp4"
    assert abs(body["duration_sec"] - 6.0) < 0.2
    assert body["sample_interval_sec"] == 2
    assert len(_stored_videos()) == 1


def test_non_mp4_extension_rejected(client, video_file):
    r = upload(client, video_file(2), filename="clip.mov")
    assert r.status_code == 400
    assert ".mp4" in r.json()["detail"]
    assert _stored_videos() == []


def test_corrupt_mp4_rejected_and_deleted(client, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"this is not a video" * 1000)
    r = upload(client, bad)
    assert r.status_code == 400
    assert "video" in r.json()["detail"].lower()
    assert _stored_videos() == []


def test_empty_file_rejected(client, tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    r = upload(client, empty)
    assert r.status_code == 400
    assert _stored_videos() == []


def test_oversized_file_rejected_and_deleted(client, video_file, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1024)
    r = upload(client, video_file(3))
    assert r.status_code == 413
    assert "limit" in r.json()["detail"]
    assert _stored_videos() == []


def test_invalid_interval_rejected(client, video_file):
    r = upload(client, video_file(2), interval=3)
    assert r.status_code == 400
    assert "sample_interval_sec" in r.json()["detail"]
    assert _stored_videos() == []


def test_get_unknown_video_404(client):
    assert client.get("/videos/does-not-exist").status_code == 404


def test_oversized_content_length_rejected_before_body_is_read(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    # Claim a huge body; the middleware must answer 413 from the header alone.
    r = client.post("/videos", content=b"x" * 10, headers={"content-type": "multipart/form-data; boundary=x",
                                                           "content-length": str(10 * 1024 * 1024 * 1024)})
    assert r.status_code == 413
    assert _stored_videos() == []


def test_health_exposes_limits(client):
    body = client.get("/health").json()
    assert body["max_upload_mb"] == config.MAX_UPLOAD_MB
    assert body["allowed_intervals"] == [1, 2, 5]
