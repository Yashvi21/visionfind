import type { SearchResult, Video } from "./types";

export const ML_DOWN_MESSAGE =
  "Can't reach the ML service. Make sure it is running on port 8000 (uvicorn app.main:app --port 8000).";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

/** Turn an error response into a readable message. The Next.js proxy answers 500/502/504 with
 *  a non-JSON body when the ML service isn't running. */
export async function errorFromResponse(res: Response): Promise<ApiError> {
  let detail: string | undefined;
  try {
    const body = await res.json();
    detail = typeof body?.detail === "string" ? body.detail : undefined;
    if (!detail && Array.isArray(body?.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  if (!detail && res.status >= 500) detail = ML_DOWN_MESSAGE;
  return new ApiError(detail ?? `Request failed (${res.status})`, res.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/ml${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(ML_DOWN_MESSAGE, 0);
  }
  if (!res.ok) throw await errorFromResponse(res);
  return (res.status === 204 ? undefined : await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<{ status: string; max_upload_mb: number; allowed_intervals: number[] }>("/health"),
  listVideos: () => request<Video[]>("/videos"),
  getVideo: (id: string) => request<Video>(`/videos/${id}`),
  deleteVideo: (id: string) => request<void>(`/videos/${id}`, { method: "DELETE" }),
  reindexVideo: (id: string, sampleIntervalSec?: number) =>
    request<Video>(`/videos/${id}/reindex`, json(sampleIntervalSec ? { sample_interval_sec: sampleIntervalSec } : {})),
  search: (body: { query: string; top_k: number; video_id?: string; dedupe_window_sec?: number }) =>
    request<SearchResult[]>("/search", json(body)),
};

export function isActive(v: Video): boolean {
  return v.status === "queued" || v.status === "processing";
}
