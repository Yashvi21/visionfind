import base64

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Frame
from app.schemas import FrameBase64Out

router = APIRouter(prefix="/frames", tags=["frames"])


def _get_frame(db: Session, frame_id: int) -> Frame:
    frame = db.get(Frame, frame_id)
    if frame is None or not frame.abs_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Frame not found.")
    return frame


@router.get("/{frame_id}/image")
def frame_image(frame_id: int, db: Session = Depends(get_db)):
    frame = _get_frame(db, frame_id)
    # Frame ids are never reused (AUTOINCREMENT), so the image for an id never changes.
    return FileResponse(frame.abs_path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/{frame_id}/base64", response_model=FrameBase64Out)
def frame_base64(frame_id: int, db: Session = Depends(get_db)):
    frame = _get_frame(db, frame_id)
    return FrameBase64Out(
        frame_id=frame.id,
        video_id=frame.video_id,
        timestamp_sec=frame.timestamp_sec,
        mime_type="image/jpeg",
        data=base64.b64encode(frame.abs_path.read_bytes()).decode("ascii"),
    )
