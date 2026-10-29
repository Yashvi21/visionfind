"""Frames must come from the time they are labelled with, including on variable-frame-rate videos
(where OpenCV's frame-number seeking is off by seconds) and with sparse keyframes."""

from fractions import Fraction

import av
import cv2
import numpy as np
import pytest

from app.video_io import _to_bgr, extract_frames, probe

STEP = 6
BASE = 40  # keep away from the dark end, where limited-range YUV distorts flat gray levels


def _level(second: float) -> int:
    return BASE + int(second) * STEP


def _encode(path, times, gop=None):
    """H.264 file whose frame at presentation time t is a flat gray of _level(t)."""
    with av.open(str(path), "w") as out:
        s = out.add_stream("libx264", rate=30, options={"crf": "12", **({"g": str(gop), "keyint_min": str(gop)} if gop else {})})
        s.width, s.height, s.pix_fmt = 320, 240, "yuv420p"
        s.codec_context.time_base = Fraction(1, 1000)
        for t in times:
            f = av.VideoFrame.from_ndarray(np.full((240, 320, 3), _level(t), np.uint8), format="rgb24")
            f.pts, f.time_base = int(round(t * 1000)), Fraction(1, 1000)
            for p in s.encode(f):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    return path


def _vfr_times():
    """0-10 s at 30 fps, 10-20 s at 5 fps, 20-30 s at 30 fps (average ~21.7 fps)."""
    t, out = 0.0, []
    while t < 30:
        out.append(round(t, 4))
        t += 1 / 5 if 10 <= t < 20 else 1 / 30
    return out


def _second_of(path) -> float:
    return (cv2.imread(str(path)).mean() - BASE) / STEP


@pytest.mark.parametrize("interval", [1, 2, 5])
def test_variable_frame_rate_timestamps_are_exact(tmp_path, interval):
    video = _encode(tmp_path / "vfr.mp4", _vfr_times())
    assert probe(video).duration_sec == pytest.approx(30, abs=0.1)
    frames = extract_frames(video, interval, tmp_path / "out", 512)
    assert len(frames) == 30 // interval
    for f in frames:
        # Stored time is within one frame of the requested sample (5 fps stretch -> 0.2 s)...
        assert f.timestamp_sec == pytest.approx(round(f.timestamp_sec / interval) * interval, abs=0.21)
        # ...and the image really is from that second (OpenCV was ~6 s off at 15 s on this file).
        assert _second_of(f.path) == pytest.approx(int(f.timestamp_sec), abs=0.5), f.timestamp_sec


def test_sparse_keyframes_forward_decoding(tmp_path):
    """GOP of 10 s: samples between keyframes are reached by decoding forward, still exact."""
    video = _encode(tmp_path / "gop.mp4", [i / 30 for i in range(30 * 25)], gop=300)
    frames = extract_frames(video, 1, tmp_path / "out", 512)
    assert [f.timestamp_sec for f in frames] == pytest.approx(list(range(25)), abs=1 / 30 + 1e-6)
    for f in frames:
        assert _second_of(f.path) == pytest.approx(int(round(f.timestamp_sec)), abs=0.5)


class _StubFrame:
    def __init__(self, rotation):
        self.rotation = rotation

    def to_ndarray(self, format):
        img = np.zeros((20, 40, 3), np.uint8)
        img[0, 0] = 255  # marker in the top-left corner of the stored frame
        return img


@pytest.mark.parametrize(
    ("rotation", "shape", "marker"),
    [(0, (20, 40), (0, 0)), (-90, (40, 20), (0, 19)), (90, (40, 20), (39, 0)), (180, (20, 40), (19, 39))],
)
def test_display_rotation_applied_like_ffmpeg(rotation, shape, marker):
    """Display-matrix rotation is counter-clockwise; ffmpeg displays it rotated by -rotation."""
    img = _to_bgr(_StubFrame(rotation))
    assert img.shape[:2] == shape
    assert tuple(np.argwhere(img[:, :, 0] == 255)[0]) == marker
