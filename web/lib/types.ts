export type VideoStatus = "queued" | "processing" | "ready" | "failed";
export type VideoStage =
  | "uploaded"
  | "extracting_frames"
  | "embedding"
  | "transcribing"
  | "building_index"
  | "ready";

export interface Video {
  id: string;
  filename: string;
  duration_sec: number;
  size_bytes: number;
  sample_interval_sec: number;
  status: VideoStatus;
  stage: VideoStage;
  error_message: string | null;
  frame_count: number;
  created_at: string;
}

export interface SearchResult {
  frame_id: number;
  video_id: string;
  video_filename: string;
  timestamp_sec: number;
  timestamp_label: string;
  thumbnail_url: string;
  similarity: number;
}

export const INTERVALS = [1, 2, 5] as const;
export type Interval = (typeof INTERVALS)[number];
export const DEFAULT_INTERVAL: Interval = 2;

/** Fallback only; the real limit comes from the ML service's /health (VS_MAX_UPLOAD_MB). */
export const DEFAULT_MAX_UPLOAD_MB = 500;
