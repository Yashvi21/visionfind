"""Long-lived shared objects: the two FAISS indexes and the (lazily loaded) embedding models."""

import logging
import shutil
import threading
from collections.abc import Callable

from sqlalchemy import delete, select

from app import config
from app.db import session_scope
from app.embeddings import ClipEmbedder, ImageTextEmbedder
from app.transcribe import MiniLMEmbedder, TextEmbedder, Transcriber, WhisperTranscriber
from app.models import Frame, Stage, Status, TranscriptSegment, Video
from app.vector_index import VectorIndex

log = logging.getLogger(__name__)


def frames_dir_for(video_id: str):
    return config.FRAMES_DIR / video_id


class _Lazy:
    """Build an expensive model on first use, once, thread-safely."""

    def __init__(self, factory: Callable) -> None:
        self.factory = factory
        self.value = None
        self._lock = threading.Lock()

    def get(self):
        if self.value is None:
            with self._lock:
                if self.value is None:
                    self.value = self.factory()
        return self.value


class Services:
    def __init__(
        self,
        clip_factory: Callable[[], ImageTextEmbedder] = ClipEmbedder,
        text_factory: Callable[[], TextEmbedder] = MiniLMEmbedder,
        transcriber_factory: Callable[[], Transcriber] = WhisperTranscriber,
    ) -> None:
        self.frame_index = VectorIndex.load_or_create(config.CLIP_DIM, config.INDEX_DIR / "frames.faiss")
        self.transcript_index = VectorIndex.load_or_create(config.TEXT_DIM, config.INDEX_DIR / "transcript.faiss")
        self._clip = _Lazy(clip_factory)
        self._text = _Lazy(text_factory)
        self._transcriber = _Lazy(transcriber_factory)

    @property
    def clip(self) -> ImageTextEmbedder:
        """Frames AND text queries for visual search always go through this one model."""
        return self._clip.get()

    @property
    def text(self) -> TextEmbedder:
        """MiniLM for transcript chunks and Ask questions."""
        return self._text.get()

    @property
    def transcriber(self) -> Transcriber:
        return self._transcriber.get()

    def save_indexes(self) -> None:
        self.frame_index.save()
        self.transcript_index.save()

    def remove_video_vectors(self, session, video_id: str) -> None:
        frame_ids = session.scalars(select(Frame.id).where(Frame.video_id == video_id)).all()
        seg_ids = session.scalars(select(TranscriptSegment.id).where(TranscriptSegment.video_id == video_id)).all()
        self.frame_index.remove(frame_ids)
        self.transcript_index.remove(seg_ids)

    def clear_video_data(self, session, video_id: str) -> None:
        """Remove everything indexing produced for a video (vectors, frame/transcript rows, frame files).
        The video row and file are kept. Callers save the indexes when appropriate."""
        self.remove_video_vectors(session, video_id)
        session.execute(delete(Frame).where(Frame.video_id == video_id))
        session.execute(delete(TranscriptSegment).where(TranscriptSegment.video_id == video_id))
        shutil.rmtree(frames_dir_for(video_id), ignore_errors=True)

    def reconcile(self) -> None:
        """Make the on-disk indexes agree with the database after a restart or crash.

        - Vectors whose row is missing or whose video isn't ready are dropped (they'd be unsearchable anyway).
        - A "ready" video whose vectors are missing is re-queued for indexing.
        """
        with session_scope() as s:
            ready_frames = dict(
                s.execute(
                    select(Frame.id, Frame.video_id).join(Video).where(Video.status == Status.READY)
                ).all()
            )
            ready_segs = set(
                s.scalars(
                    select(TranscriptSegment.id).join(Video).where(Video.status == Status.READY)
                ).all()
            )
            frame_ids, seg_ids = self.frame_index.ids(), self.transcript_index.ids()
            stale_frames = frame_ids - ready_frames.keys()
            stale_segs = seg_ids - ready_segs
            missing = {vid for fid, vid in ready_frames.items() if fid not in frame_ids}
            missing_segs = ready_segs - seg_ids
            if missing_segs:
                missing |= set(
                    s.scalars(select(TranscriptSegment.video_id).where(TranscriptSegment.id.in_(missing_segs))).all()
                )
            for vid in missing:
                log.warning("Video %s is marked ready but its vectors are missing; re-queueing", vid)
                video = s.get(Video, vid)
                video.status, video.stage = Status.QUEUED, Stage.UPLOADED

        if stale_frames or stale_segs:
            log.warning("Removing %d stale frame and %d stale transcript vectors", len(stale_frames), len(stale_segs))
            self.frame_index.remove(stale_frames)
            self.transcript_index.remove(stale_segs)
            self.save_indexes()
