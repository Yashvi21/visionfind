"""SQLite schema. Frame.id and TranscriptSegment.id are also the FAISS vector ids."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app import config
from app.db import Base


class Status:
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Stage:
    UPLOADED = "uploaded"
    EXTRACTING_FRAMES = "extracting_frames"
    EMBEDDING = "embedding"
    TRANSCRIBING = "transcribing"
    BUILDING_INDEX = "building_index"
    READY = "ready"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(512))
    # Paths are stored relative to config.DATA_DIR so the data folder can move.
    stored_path: Mapped[str] = mapped_column(String(1024))
    duration_sec: Mapped[float] = mapped_column(Float)
    size_bytes: Mapped[int] = mapped_column(Integer)
    sample_interval_sec: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default=Status.QUEUED, index=True)
    stage: Mapped[str] = mapped_column(String(32), default=Stage.UPLOADED)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    frames: Mapped[list["Frame"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", passive_deletes=True
    )
    segments: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def abs_path(self) -> Path:
        return config.DATA_DIR / self.stored_path


class Frame(Base):
    __tablename__ = "frames"

    # AUTOINCREMENT: SQLite never reuses ids of deleted rows, so a stale FAISS id can
    # never silently point at a different row.
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    timestamp_sec: Mapped[float] = mapped_column(Float)
    frame_path: Mapped[str] = mapped_column(String(1024))

    video: Mapped[Video] = relationship(back_populates="frames")

    @property
    def abs_path(self) -> Path:
        return config.DATA_DIR / self.frame_path


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    # AUTOINCREMENT: SQLite never reuses ids of deleted rows, so a stale FAISS id can
    # never silently point at a different row.
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)

    video: Mapped[Video] = relationship(back_populates="segments")
