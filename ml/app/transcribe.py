"""Speech-to-text (faster-whisper), chunking into ~20-30 s segments, and MiniLM text embeddings."""

import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app import config
from app.vector_index import l2_normalize

log = logging.getLogger(__name__)

CHUNK_MIN_SEC = 20.0
CHUNK_MAX_SEC = 30.0


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


class Transcriber(Protocol):
    def transcribe(self, path: Path) -> list[Segment]: ...


class TextEmbedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def has_audio(path: Path) -> bool:
    import av

    try:
        with av.open(str(path)) as container:
            return len(container.streams.audio) > 0
    except Exception:  # unreadable container: let the transcriber decide
        log.warning("Could not inspect audio streams of %s", path.name, exc_info=True)
        return True


def chunk_segments(segments: Sequence[Segment], min_sec: float = CHUNK_MIN_SEC, max_sec: float = CHUNK_MAX_SEC) -> list[Segment]:
    """Merge consecutive speech segments into chunks of ~min_sec..max_sec.

    A chunk is closed once it reaches min_sec, or before adding a segment would push it past
    max_sec. A single segment longer than max_sec becomes its own chunk (never split mid-sentence).
    """
    chunks: list[Segment] = []
    cur: list[Segment] = []

    def flush() -> None:
        if cur:
            text = " ".join(s.text.strip() for s in cur if s.text.strip())
            if text:
                chunks.append(Segment(cur[0].start, cur[-1].end, text))
            cur.clear()

    for seg in segments:
        if cur and seg.end - cur[0].start > max_sec:
            flush()
        cur.append(seg)
        if cur[-1].end - cur[0].start >= min_sec:
            flush()
    flush()
    return chunks


class WhisperTranscriber:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        # cpu_threads explicitly: OMP_NUM_THREADS=1 (see app/__init__.py) would otherwise
        # leave CTranslate2 single-threaded.
        threads = os.cpu_count() or 4
        log.info("Loading faster-whisper %s (cpu, int8, %d threads)", config.WHISPER_MODEL, threads)
        self._model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=threads)
        self._lock = threading.Lock()

    def transcribe(self, path: Path) -> list[Segment]:
        with self._lock:
            segments, info = self._model.transcribe(str(path), vad_filter=True)
            out = [Segment(float(s.start), float(s.end), s.text) for s in segments]
        log.info("Transcribed %s: language=%s, %d segments", path.name, info.language, len(out))
        return out


class MiniLMEmbedder:
    dim = config.TEXT_DIM

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        from app.embeddings import pick_device

        device = pick_device()
        log.info("Loading %s on %s", config.TEXT_MODEL, device)
        self._model = SentenceTransformer(config.TEXT_MODEL, device=device)
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        with self._lock:
            vecs = self._model.encode(list(texts), batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        return l2_normalize(vecs)
