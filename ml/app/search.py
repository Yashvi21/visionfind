"""Visual search over the frames index. Read-only: never extracts or embeds frames."""

from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.models import Frame, Status, TranscriptSegment, Video
from app.services import Services


def format_timestamp(seconds: float) -> str:
    """mm:ss, or h:mm:ss for videos of an hour or more."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class Hit:
    frame_id: int
    video_id: str
    video_filename: str
    timestamp_sec: float
    similarity: float
    interval_sec: int


def temporal_dedupe(hits: Iterable[Hit], window_sec: float | None) -> list[Hit]:
    """Hits must be sorted best-first. Within a video, drop a hit if a better hit already kept
    is within `window_sec` (inclusive). None -> 2x that video's sample interval; 0 disables."""
    kept: list[Hit] = []
    kept_times: dict[str, list[float]] = {}
    for h in hits:
        window = 2 * h.interval_sec if window_sec is None else window_sec
        times = kept_times.setdefault(h.video_id, [])
        if window > 0 and any(abs(h.timestamp_sec - t) <= window for t in times):
            continue
        times.append(h.timestamp_sec)
        kept.append(h)
    return kept


_SQL_CHUNK = 900


def _resolve(db: Session, scored: list[tuple[int, float]], video_id: str | None) -> list[Hit]:
    """Map FAISS ids back to rows via primary key, keeping only frames of ready videos."""
    if not scored:
        return []
    ids = [fid for fid, _ in scored]
    rows = {}
    for start in range(0, len(ids), _SQL_CHUNK):  # stay well under SQLite's bound-variable limit
        q = (
            select(Frame.id, Frame.video_id, Frame.timestamp_sec, Video.filename, Video.sample_interval_sec)
            .join(Video, Frame.video_id == Video.id)
            .where(Frame.id.in_(ids[start : start + _SQL_CHUNK]), Video.status == Status.READY)
        )
        if video_id:
            q = q.where(Frame.video_id == video_id)
        rows.update({r.id: r for r in db.execute(q)})
    hits = []
    for fid, score in scored:  # keep FAISS ranking order
        r = rows.get(fid)
        if r is not None:
            hits.append(Hit(fid, r.video_id, r.filename, r.timestamp_sec, score, r.sample_interval_sec))
    return hits


def search_frames(
    services: Services,
    db: Session,
    query_vec,
    top_k: int,
    video_id: str | None = None,
    dedupe_window_sec: float | None = None,
) -> list[Hit]:
    restrict = None
    if video_id:
        restrict = db.scalars(
            select(Frame.id).join(Video).where(Frame.video_id == video_id, Video.status == Status.READY)
        ).all()
        if not restrict:
            return []

    total = services.frame_index.ntotal if restrict is None else len(restrict)
    k = top_k * 5  # over-fetch so filtering and dedupe still leave enough results
    while True:
        scored = services.frame_index.search(query_vec, k, restrict_ids=restrict)
        hits = temporal_dedupe(_resolve(db, scored, video_id), dedupe_window_sec)
        if len(hits) >= top_k or k >= total:
            break
        k *= 4  # many hits were filtered out (non-ready videos or duplicates); widen the net
    hits.sort(key=lambda h: h.similarity, reverse=True)
    return hits[:top_k]


def thumbnail_url(frame_id: int) -> str:
    return f"{config.PUBLIC_PREFIX}/frames/{frame_id}/image"


@dataclass
class ChunkHit:
    segment: TranscriptSegment
    reason: str  # "match" | "overlap"
    similarity: float | None


def retrieve_for_question(
    services: Services, db: Session, video_id: str, question: str, k_frames: int, k_chunks: int
) -> tuple[list[Hit], list[ChunkHit], bool]:
    """Context for answering a question about one (ready) video. Read-only: uses stored vectors.

    Frames: top CLIP matches (temporally deduped), returned in time order.
    Chunks: top MiniLM matches + any chunk spoken while a retrieved frame was on screen, in time order.
    """
    frames = search_frames(services, db, services.clip.encode_text([question]), k_frames, video_id)
    frames.sort(key=lambda h: h.timestamp_sec)

    segments = db.scalars(select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)).all()
    chunks: dict[int, ChunkHit] = {}
    if segments and k_chunks > 0:
        by_id = {s.id: s for s in segments}
        q = services.text.encode([question])
        for sid, score in services.transcript_index.search(q, k_chunks, restrict_ids=list(by_id)):
            chunks[sid] = ChunkHit(by_id[sid], "match", score)
    for f in frames:
        for s in segments:
            if s.start_sec <= f.timestamp_sec <= s.end_sec and s.id not in chunks:
                chunks[s.id] = ChunkHit(s, "overlap", None)
    ordered = sorted(chunks.values(), key=lambda c: c.segment.start_sec)
    return frames, ordered, bool(segments)
