import cv2
import pytest

from app.video_io import extract_frames, sample_timestamps

from .conftest import GRAY_STEP, upload


@pytest.mark.parametrize(
    ("duration", "interval", "expected"),
    [(10, 1, 10), (10, 2, 5), (10, 5, 2), (11, 5, 3), (11, 2, 6), (0.5, 2, 1)],
)
def test_sample_timestamps_count(duration, interval, expected):
    ts = sample_timestamps(duration, interval)
    assert len(ts) == expected
    assert ts[0] == 0
    assert all(t < duration for t in ts)


@pytest.mark.parametrize(("interval", "expected"), [(1, 10), (2, 5), (5, 2)])
def test_extract_frames_per_interval(tmp_path, video_file, interval, expected):
    frames = extract_frames(video_file(10), interval, tmp_path / "out", max_width=512)
    assert [f.timestamp_sec for f in frames] == [i * interval for i in range(expected)]
    for f in frames:
        img = cv2.imread(str(f.path))
        assert img.shape[1] == 512  # 640px source resized to max 512 wide
        # The frame's brightness encodes which second it came from: wrong seek -> wrong level.
        assert abs(img.mean() - int(f.timestamp_sec) * GRAY_STEP) < 6, f"wrong frame at {f.timestamp_sec}s"


def test_small_video_not_upscaled(tmp_path, video_file):
    frames = extract_frames(video_file(3, size=(320, 240)), 1, tmp_path / "out", max_width=512)
    assert cv2.imread(str(frames[0].path)).shape[1] == 320


@pytest.mark.parametrize(("interval", "expected"), [(1, 8), (2, 4), (5, 2)])
def test_worker_extracts_and_records_frames(client, video_file, interval, expected):
    r = upload(client, video_file(8), interval=interval)
    vid = r.json()["id"]
    assert client.app.state.worker.wait_idle(60)

    video = client.get(f"/videos/{vid}").json()
    assert video["frame_count"] == expected
    assert video["status"] == "ready"
    assert video["stage"] == "ready"
    assert video["error_message"] is None

    from app.db import SessionLocal
    from app.models import Frame

    with SessionLocal() as s:
        rows = s.query(Frame).filter_by(video_id=vid).order_by(Frame.timestamp_sec).all()
    assert [f.timestamp_sec for f in rows] == [i * interval for i in range(expected)]
    assert all(f.abs_path.is_file() for f in rows)

    # Frame endpoints serve the right image
    img = client.get(f"/frames/{rows[1].id}/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    b64 = client.get(f"/frames/{rows[1].id}/base64").json()
    assert b64["video_id"] == vid and b64["timestamp_sec"] == interval and b64["mime_type"] == "image/jpeg"


def test_video_streaming_supports_range(client, video_file):
    path = video_file(4)
    vid = upload(client, path).json()["id"]
    full = client.get(f"/videos/{vid}/file")
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    assert full.content == path.read_bytes()

    part = client.get(f"/videos/{vid}/file", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 100-199/{path.stat().st_size}"
    assert part.content == path.read_bytes()[100:200]


def test_missing_frame_404(client):
    assert client.get("/frames/999999/image").status_code == 404
