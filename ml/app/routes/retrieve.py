import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Status, Video
from app.schemas import RetrievedChunk, RetrievedFrame, RetrieveRequest, RetrieveResponse
from app.search import format_timestamp, retrieve_for_question, thumbnail_url

log = logging.getLogger(__name__)
router = APIRouter(tags=["ask"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, request: Request, db: Session = Depends(get_db)):
    question = req.question.strip()
    if not question:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Question must not be empty.")
    video = db.get(Video, req.video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found.")
    if video.status != Status.READY:
        raise HTTPException(status.HTTP_409_CONFLICT, "This video is not ready yet.")

    started = time.perf_counter()
    frames, chunks, has_transcript = retrieve_for_question(
        request.app.state.services, db, video.id, question, req.k_frames, req.k_chunks
    )
    log.info("retrieve video=%s -> %d frames, %d chunks in %.0f ms",
             video.id, len(frames), len(chunks), (time.perf_counter() - started) * 1000)
    return RetrieveResponse(
        video_id=video.id,
        video_filename=video.filename,
        duration_sec=video.duration_sec,
        has_transcript=has_transcript,
        frames=[
            RetrievedFrame(
                frame_id=f.frame_id,
                timestamp_sec=f.timestamp_sec,
                timestamp_label=format_timestamp(f.timestamp_sec),
                thumbnail_url=thumbnail_url(f.frame_id),
                similarity=round(f.similarity, 4),
            )
            for f in frames
        ],
        chunks=[
            RetrievedChunk(
                segment_id=c.segment.id,
                start_sec=c.segment.start_sec,
                end_sec=c.segment.end_sec,
                start_label=format_timestamp(c.segment.start_sec),
                end_label=format_timestamp(c.segment.end_sec),
                text=c.segment.text,
                reason=c.reason,
                similarity=None if c.similarity is None else round(c.similarity, 4),
            )
            for c in chunks
        ],
    )
