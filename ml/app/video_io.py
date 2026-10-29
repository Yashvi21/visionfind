"""Probe a video file and extract frames at fixed time intervals.

OpenCV validates uploads and resizes/writes JPEGs. Decoding and seeking use PyAV, because OpenCV maps
times to frame numbers with the *average* frame rate: on variable-frame-rate videos (phones, screen
recordings) its frames can be several seconds away from the requested time. PyAV gives each frame's
real presentation timestamp, and that exact time is what we store.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import av
import cv2

log = logging.getLogger(__name__)

JPEG_QUALITY = 85
# If the next sample is at most this far ahead of the decoder, keep decoding forward instead of
# seeking back to a keyframe (cheaper for short intervals).
FORWARD_DECODE_MAX_SEC = 2.0


class InvalidVideoError(ValueError):
    """The file can't be used as a video. The message is shown to the user."""


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class ExtractedFrame:
    timestamp_sec: float  # the decoded frame's real presentation time
    path: Path


def _container_duration(path: Path) -> float | None:
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                return None
            s = c.streams.video[0]
            if s.duration and s.time_base:
                return float(s.duration * s.time_base) + (float(s.start_time * s.time_base) if s.start_time else 0.0)
            if c.duration:
                return c.duration / av.time_base
    except Exception:
        log.warning("PyAV could not read the duration of %s", path.name, exc_info=True)
    return None


def probe(path: Path) -> VideoInfo:
    """Validate with OpenCV (it must open the file and decode a frame), then read the duration."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidVideoError("The file could not be opened as a video. It may be corrupt or use an unsupported codec.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, _ = cap.read()
        if not ok:
            raise InvalidVideoError("The video contains no readable frames.")
    finally:
        cap.release()

    duration = _container_duration(path)
    if not duration and fps > 0 and frame_count > 0:
        duration = frame_count / fps  # fallback; only exact for constant frame rate
    if not duration or duration <= 0 or width <= 0 or height <= 0:
        raise InvalidVideoError("Could not read the video's duration. The file may be corrupt or incomplete.")
    return VideoInfo(duration, fps, width, height)


def sample_timestamps(duration_sec: float, interval_sec: float) -> list[float]:
    """0, N, 2N, ... strictly before the end of the video: ceil(duration / N) timestamps."""
    n = math.ceil(duration_sec / interval_sec - 1e-9)
    return [round(i * interval_sec, 3) for i in range(max(n, 0))]


def _resize(img, max_width: int):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    return cv2.resize(img, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)


_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _to_bgr(frame: "av.VideoFrame"):
    img = frame.to_ndarray(format="bgr24")
    # Apply the display matrix like ffmpeg's autorotate (phones store portrait video rotated).
    theta = round(-(getattr(frame, "rotation", 0) or 0)) % 360
    code = _ROTATIONS.get(theta)
    return cv2.rotate(img, code) if code is not None else img


def extract_frames(path: Path, interval_sec: float, out_dir: Path, max_width: int) -> list[ExtractedFrame]:
    """One JPEG per sample time: the first frame whose presentation time is >= that time.

    Seeks (to the preceding keyframe) for each sample rather than decoding the whole video, but keeps
    decoding forward when the next sample is close. A sample past the last frame is skipped, and two
    samples that land on the same frame (low frame-rate stretches) produce it once.
    """
    info = probe(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExtractedFrame] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        decoder = None
        last_time: float | None = None  # time of the last decoded frame
        for ts in sample_timestamps(info.duration_sec, interval_sec):
            if decoder is None or last_time is None or ts - last_time > FORWARD_DECODE_MAX_SEC or ts < last_time:
                container.seek(int(ts / stream.time_base), stream=stream, backward=True, any_frame=False)
                decoder = container.decode(stream)
            frame = None
            for f in decoder:
                if f.time is None:
                    continue
                last_time = f.time
                if f.time >= ts - 1e-3:
                    frame = f
                    break
            if frame is None:
                log.warning("No frame at or after %.2fs in %s; skipping", ts, path.name)
                decoder = None
                continue
            actual = round(float(frame.time), 3)
            if results and actual <= results[-1].timestamp_sec:
                continue  # same frame as the previous sample
            out_path = out_dir / f"{int(round(actual * 1000)):09d}.jpg"
            img = _resize(_to_bgr(frame), max_width)
            if not cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                raise OSError(f"Failed to write frame image {out_path}")
            results.append(ExtractedFrame(actual, out_path))
    if not results:
        raise InvalidVideoError("No frames could be extracted from this video.")
    return results
