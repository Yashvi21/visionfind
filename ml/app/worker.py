"""Single background thread that indexes videos one at a time. Never called from a request handler."""

import logging
import queue
import shutil
import threading
import time

from sqlalchemy import select

from app import config
from app.db import session_scope
from app.models import Frame, Stage, Status, TranscriptSegment, Video
from app.services import Services, frames_dir_for
from app.transcribe import chunk_segments, has_audio
from app.video_io import InvalidVideoError, extract_frames

log = logging.getLogger(__name__)

_STOP = object()


class _Deleted(Exception):
    """The video was deleted while it was being processed."""


class Worker:
    def __init__(self, services: Services) -> None:
        self.services = services
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="indexer", daemon=True)
        self._thread.start()
        self._requeue_unfinished()

    def stop(self, timeout: float = 30) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join(timeout)
        self._thread = None

    def enqueue(self, video_id: str) -> None:
        self._queue.put(video_id)

    def wait_idle(self, timeout: float = 120) -> bool:
        """Block until the queue is drained (used by tests and scripts)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.05)
        return False

    def _requeue_unfinished(self) -> None:
        """After a restart, videos left queued/processing are restarted from scratch."""
        with session_scope() as s:
            ids = s.scalars(
                select(Video.id)
                .where(Video.status.in_([Status.QUEUED, Status.PROCESSING]))
                .order_by(Video.created_at)
            ).all()
            for vid in ids:
                s.get(Video, vid).status = Status.QUEUED
        for vid in ids:
            log.info("Re-queueing unfinished video %s", vid)
            self.enqueue(vid)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self.process(item)
            except Exception:  # process() handles its own errors; this is a last resort
                log.exception("Unexpected worker error for %s", item)
            finally:
                self._queue.task_done()

    # -- pipeline --------------------------------------------------------
    def process(self, video_id: str) -> None:
        started = time.monotonic()
        with session_scope() as s:
            video = s.get(Video, video_id)
            if video is None or video.status not in (Status.QUEUED, Status.PROCESSING):
                log.info("Video %s is gone or no longer queued; skipping", video_id)
                return
            self.services.clear_video_data(s, video_id)
            video.status = Status.PROCESSING
            video.error_message = None
            video.frame_count = 0
            src, interval = video.abs_path, video.sample_interval_sec

        added_frame_ids: list[int] = []
        added_seg_ids: list[int] = []
        try:
            # 1. Frames
            self._set_stage(video_id, Stage.EXTRACTING_FRAMES)
            extracted = extract_frames(src, interval, frames_dir_for(video_id), config.FRAME_MAX_WIDTH)
            with session_scope() as s:
                self._require(s, video_id)
                rows = [
                    Frame(
                        video_id=video_id,
                        timestamp_sec=f.timestamp_sec,
                        frame_path=str(f.path.relative_to(config.DATA_DIR)),
                    )
                    for f in extracted
                ]
                s.add_all(rows)
                s.flush()  # assigns Frame.id, which becomes the FAISS id
                frames = [(r.id, r.abs_path) for r in rows]
                s.get(Video, video_id).frame_count = len(rows)
            t_frames = time.monotonic()

            # 2. CLIP embeddings, batch by batch, so a delete mid-way stops quickly
            self._set_stage(video_id, Stage.EMBEDDING)
            clip = self.services.clip
            batch = config.CLIP_BATCH_SIZE
            for start in range(0, len(frames), batch):
                chunk = frames[start : start + batch]
                vecs = clip.encode_images([p for _, p in chunk])
                ids = [fid for fid, _ in chunk]
                self.services.frame_index.add(ids, vecs)
                added_frame_ids.extend(ids)
                self._check_exists(video_id)
            t_embed = time.monotonic()

            # 3. Transcript: Whisper -> ~20-30 s chunks -> MiniLM -> transcript index
            self._set_stage(video_id, Stage.TRANSCRIBING)
            chunks = []
            if has_audio(src):
                chunks = chunk_segments(self.services.transcriber.transcribe(src))
            else:
                log.info("Video %s has no audio track; skipping transcription", video_id)
            if chunks:
                with session_scope() as s:
                    self._require(s, video_id)
                    seg_rows = [
                        TranscriptSegment(video_id=video_id, start_sec=round(c.start, 2), end_sec=round(c.end, 2), text=c.text)
                        for c in chunks
                    ]
                    s.add_all(seg_rows)
                    s.flush()  # assigns TranscriptSegment.id, the transcript FAISS id
                    seg_ids = [r.id for r in seg_rows]
                vecs = self.services.text.encode([c.text for c in chunks])
                self.services.transcript_index.add(seg_ids, vecs)
                added_seg_ids.extend(seg_ids)
                self._check_exists(video_id)
            t_transcribe = time.monotonic()

            # 4. Persist indexes, and only then mark ready.
            self._set_stage(video_id, Stage.BUILDING_INDEX)
            self.services.save_indexes()
            with session_scope() as s:
                video = self._require(s, video_id)
                video.status = Status.READY
                video.stage = Stage.READY
            log.info(
                "Video %s ready: %d frames, %d transcript chunks (extract %.1fs, embed %.1fs, transcribe %.1fs, total %.1fs)",
                video_id, len(frames), len(chunks), t_frames - started, t_embed - t_frames,
                t_transcribe - t_embed, time.monotonic() - started,
            )
        except _Deleted:
            log.info("Video %s was deleted during processing; discarding its vectors", video_id)
            self.services.frame_index.remove(added_frame_ids)
            self.services.transcript_index.remove(added_seg_ids)
            shutil.rmtree(frames_dir_for(video_id), ignore_errors=True)
        except Exception as exc:
            log.exception("Indexing failed for %s", video_id)
            self.services.frame_index.remove(added_frame_ids)
            self.services.transcript_index.remove(added_seg_ids)
            self._fail(video_id, exc)

    @staticmethod
    def _require(session, video_id: str) -> Video:
        video = session.get(Video, video_id)
        if video is None:
            raise _Deleted()
        return video

    def _check_exists(self, video_id: str) -> None:
        with session_scope() as s:
            self._require(s, video_id)

    def _set_stage(self, video_id: str, stage: str) -> None:
        with session_scope() as s:
            self._require(s, video_id).stage = stage

    def _fail(self, video_id: str, exc: Exception) -> None:
        if isinstance(exc, InvalidVideoError):
            message = str(exc)
        else:
            message = f"Indexing failed: {type(exc).__name__}: {exc}"
        with session_scope() as s:
            video = s.get(Video, video_id)
            if video is None:
                return
            self.services.clear_video_data(s, video_id)
            video.status = Status.FAILED
            video.error_message = message
            video.frame_count = 0
        self.services.save_indexes()
