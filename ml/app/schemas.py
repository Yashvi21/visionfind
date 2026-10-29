from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app import config


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    duration_sec: float
    size_bytes: int
    sample_interval_sec: int
    status: str
    stage: str
    error_message: str | None
    frame_count: int
    created_at: datetime


class FrameBase64Out(BaseModel):
    frame_id: int
    video_id: str
    timestamp_sec: float
    mime_type: str
    data: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=config.MAX_TOP_K)
    video_id: str | None = None
    dedupe_window_sec: float | None = Field(None, ge=0)


class SearchResult(BaseModel):
    frame_id: int
    video_id: str
    video_filename: str
    timestamp_sec: float
    timestamp_label: str
    thumbnail_url: str
    similarity: float


class ReindexRequest(BaseModel):
    sample_interval_sec: int | None = None


class RetrieveRequest(BaseModel):
    video_id: str
    question: str = Field(min_length=1, max_length=2000)
    k_frames: int = Field(6, ge=1, le=12)
    k_chunks: int = Field(4, ge=0, le=10)


class RetrievedFrame(BaseModel):
    frame_id: int
    timestamp_sec: float
    timestamp_label: str
    thumbnail_url: str
    similarity: float


class RetrievedChunk(BaseModel):
    segment_id: int
    start_sec: float
    end_sec: float
    start_label: str
    end_label: str
    text: str
    # "match": retrieved by similarity to the question; "overlap": spoken while a retrieved frame was on screen
    reason: str
    similarity: float | None


class RetrieveResponse(BaseModel):
    video_id: str
    video_filename: str
    duration_sec: float
    has_transcript: bool
    frames: list[RetrievedFrame]
    chunks: list[RetrievedChunk]
