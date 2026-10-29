import os
import shutil
import tempfile
from pathlib import Path

# Isolate all test data before any app module reads its config.
_TEST_DATA = Path(tempfile.mkdtemp(prefix="visionfind-test-"))
os.environ["VS_DATA_DIR"] = str(_TEST_DATA)
os.environ["VS_PRELOAD_MODELS"] = "0"

import re  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Video  # noqa: E402
from app.vector_index import l2_normalize  # noqa: E402

GRAY_STEP = 20  # each second of a synthetic video has brightness (second * GRAY_STEP)


def _bump(center: float, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    idx = np.arange(16)
    v[:16] = np.exp(-((idx - center) ** 2) / 2.0)
    return v


class FakeClip:
    """Deterministic stand-in for CLIP. A frame embeds as a bump at its second (from brightness);
    the text "second N" embeds as a bump at N. So "second N" ranks the frame at N first, then
    its neighbours by distance. Lets tests check exact ids/timestamps without the real model."""

    dim = 512
    calls = {"images": 0, "text": 0}

    def encode_images(self, paths):
        FakeClip.calls["images"] += len(paths)
        levels = [cv2.imread(str(p)).mean() / GRAY_STEP for p in paths]
        return l2_normalize(np.stack([_bump(lv) for lv in levels]))

    def encode_text(self, texts):
        FakeClip.calls["text"] += len(texts)
        out = []
        for t in texts:
            m = re.search(r"second (\d+)", t)
            out.append(_bump(int(m.group(1))) if m else np.ones(512, np.float32))
        return l2_normalize(np.stack(out))


class FakeText:
    """Hashed bag-of-words: texts sharing words get high similarity. Stand-in for MiniLM."""

    dim = 384

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, sum(map(ord, w)) % self.dim] += 1.0
        out[:, -1] += 1e-3  # avoid zero vectors
        return l2_normalize(out)


class FakeTranscriber:
    """Returns a scripted transcript. Only called when the worker sees an audio track."""

    script = [
        (0.0, 4.0, "welcome to the roadmap review"),
        (4.0, 8.0, "budget numbers for next year"),
        (8.0, 12.0, "hiring plan for engineers"),
    ]
    calls = 0

    def transcribe(self, path):
        from app.transcribe import Segment

        FakeTranscriber.calls += 1
        return [Segment(a, b, t) for a, b, t in self.script]


app.state.clip_factory = FakeClip
app.state.text_factory = FakeText
app.state.transcriber_factory = FakeTranscriber


def make_video(path: Path, seconds: float, fps: int = 10, size: tuple[int, int] = (640, 360)) -> Path:
    """Silent mp4 where every frame in second s is filled with gray level s*GRAY_STEP,
    so a frame's brightness tells us which timestamp it came from."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for i in range(int(round(seconds * fps))):
        level = min(int(i // fps) * GRAY_STEP, 255)
        writer.write(np.full((h, w, 3), level, np.uint8))
    writer.release()
    return path


@pytest.fixture
def video_file(tmp_path):
    def _make(seconds: float = 10, name: str = "clip.mp4", **kw) -> Path:
        return make_video(tmp_path / name, seconds, **kw)

    return _make


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
        c.app.state.worker.wait_idle(60)


@pytest.fixture(autouse=True)
def _clean_between_tests():
    yield
    _reset_data()


def _reset_data():
    init_db()
    with SessionLocal() as s:
        for v in s.query(Video).all():
            s.delete(v)
        s.commit()
    for d in (config.VIDEOS_DIR, config.FRAMES_DIR, config.INDEX_DIR):
        shutil.rmtree(d, ignore_errors=True)
    config.ensure_dirs()


def upload(client, path: Path, interval: int = 2, filename: str | None = None):
    with path.open("rb") as fh:
        return client.post(
            "/videos",
            files={"file": (filename or path.name, fh, "video/mp4")},
            data={"sample_interval_sec": str(interval)},
        )


def upload_ready(client, path: Path, interval: int = 1) -> str:
    r = upload(client, path, interval)
    assert r.status_code == 201, r.text
    vid = r.json()["id"]
    assert client.app.state.worker.wait_idle(60)
    video = client.get(f"/videos/{vid}").json()
    assert video["status"] == "ready", video
    return vid


def search(client, query: str, **kw):
    r = client.post("/search", json={"query": query, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DATA, ignore_errors=True)
