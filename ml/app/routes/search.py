import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Status, Video
from app.schemas import SearchRequest, SearchResult
from app.search import format_timestamp, search_frames, thumbnail_url

log = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


@router.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest, request: Request, db: Session = Depends(get_db)):
    query = req.query.strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Query must not be empty.")
    if req.video_id:
        video = db.get(Video, req.video_id)
        if video is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found.")
        if video.status != Status.READY:
            raise HTTPException(status.HTTP_409_CONFLICT, "This video is not ready for search yet.")

    services = request.app.state.services
    started = time.perf_counter()
    query_vec = services.clip.encode_text([query])
    hits = search_frames(services, db, query_vec, req.top_k, req.video_id, req.dedupe_window_sec)
    log.info("search %r top_k=%d video=%s -> %d hits in %.0f ms",
             query, req.top_k, req.video_id, len(hits), (time.perf_counter() - started) * 1000)
    return [
        SearchResult(
            frame_id=h.frame_id,
            video_id=h.video_id,
            video_filename=h.video_filename,
            timestamp_sec=h.timestamp_sec,
            timestamp_label=format_timestamp(h.timestamp_sec),
            thumbnail_url=thumbnail_url(h.frame_id),
            similarity=round(h.similarity, 4),
        )
        for h in hits
    ]
