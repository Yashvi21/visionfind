import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.db import get_db
from app.models import Stage, Status, Video
from app.schemas import ReindexRequest, VideoOut
from app.video_io import InvalidVideoError, probe

log = logging.getLogger(__name__)
router = APIRouter(prefix="/videos", tags=["videos"])

CHUNK = 1024 * 1024


def _get_video(db: Session, video_id: str) -> Video:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video not found.")
    return video


def _save_upload(upload: UploadFile, dest: Path) -> int:
    """Copy the upload to disk in chunks, enforcing the size limit while writing."""
    limit = config.MAX_UPLOAD_BYTES
    written = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(CHUNK):
            written += len(chunk)
            if written > limit:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    f"File is larger than the {config.MAX_UPLOAD_MB} MB limit.",
                )
            out.write(chunk)
    return written


@router.post("", response_model=VideoOut, status_code=status.HTTP_201_CREATED)
def upload_video(
    request: Request,
    file: UploadFile = File(...),
    sample_interval_sec: int = Form(config.DEFAULT_INTERVAL),
    db: Session = Depends(get_db),
):
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .mp4 files are supported.")
    if sample_interval_sec not in config.ALLOWED_INTERVALS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"sample_interval_sec must be one of {list(config.ALLOWED_INTERVALS)}.",
        )

    video_id = str(uuid.uuid4())
    dest = config.VIDEOS_DIR / f"{video_id}.mp4"
    try:
        size = _save_upload(file, dest)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
        info = probe(dest)
    except InvalidVideoError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except BaseException:
        dest.unlink(missing_ok=True)
        raise

    video = Video(
        id=video_id,
        filename=filename,
        stored_path=str(dest.relative_to(config.DATA_DIR)),
        duration_sec=round(info.duration_sec, 3),
        size_bytes=size,
        sample_interval_sec=sample_interval_sec,
        status=Status.QUEUED,
        stage=Stage.UPLOADED,
    )
    db.add(video)
    db.commit()
    request.app.state.worker.enqueue(video_id)
    log.info("Uploaded %s as %s (%.1fs, %d bytes, every %ds)", filename, video_id, info.duration_sec, size, sample_interval_sec)
    return video


@router.get("", response_model=list[VideoOut])
def list_videos(db: Session = Depends(get_db)):
    return db.scalars(select(Video).order_by(Video.created_at.desc())).all()


@router.get("/{video_id}", response_model=VideoOut)
def get_video(video_id: str, db: Session = Depends(get_db)):
    return _get_video(db, video_id)


@router.get("/{video_id}/file")
def stream_video(video_id: str, db: Session = Depends(get_db)):
    """Starlette's FileResponse answers Range requests with 206, so the browser can seek."""
    video = _get_video(db, video_id)
    path = video.abs_path
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The video file is missing from storage.")
    return FileResponse(path, media_type="video/mp4", filename=video.filename, content_disposition_type="inline")


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: str, request: Request, db: Session = Depends(get_db)):
    """Removes vectors from both indexes, frame files, DB rows and the video file.
    Safe while the video is being indexed: the worker notices and discards its work."""
    services = request.app.state.services
    video = _get_video(db, video_id)
    path = video.abs_path
    services.clear_video_data(db, video_id)  # vectors first, so nothing can point at deleted rows
    db.delete(video)
    db.commit()
    path.unlink(missing_ok=True)
    services.save_indexes()
    log.info("Deleted video %s (%s)", video_id, video.filename)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{video_id}/reindex", response_model=VideoOut, status_code=status.HTTP_202_ACCEPTED)
def reindex_video(
    video_id: str,
    request: Request,
    body: ReindexRequest | None = Body(None),
    db: Session = Depends(get_db),
):
    """Clear all derived data and queue the video again, optionally with a new interval.
    Also used to retry a failed video."""
    services = request.app.state.services
    video = _get_video(db, video_id)
    if video.status in (Status.QUEUED, Status.PROCESSING):
        raise HTTPException(status.HTTP_409_CONFLICT, "This video is already being indexed.")
    interval = body.sample_interval_sec if body and body.sample_interval_sec else video.sample_interval_sec
    if interval not in config.ALLOWED_INTERVALS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"sample_interval_sec must be one of {list(config.ALLOWED_INTERVALS)}.",
        )
    if not video.abs_path.is_file():
        raise HTTPException(status.HTTP_409_CONFLICT, "The original video file is missing, so it can't be re-indexed.")

    services.clear_video_data(db, video_id)
    video.sample_interval_sec = interval
    video.status = Status.QUEUED
    video.stage = Stage.UPLOADED
    video.error_message = None
    video.frame_count = 0
    db.commit()
    services.save_indexes()
    request.app.state.worker.enqueue(video_id)
    log.info("Re-indexing video %s every %ds", video_id, interval)
    return video
