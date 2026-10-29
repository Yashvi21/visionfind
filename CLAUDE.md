# VisionFind

Web app: users upload their own .mp4 videos, then (A) find moments by describing a scene in plain language (local CLIP search) and (B) ask questions about a video, answered by Gemini Flash from retrieved frames + transcript with clickable `[mm:ss]` citations.

## Architecture

```
browser ──► web/ (Next.js 15, :3000) ──rewrite /ml/:path*──► ml/ (FastAPI, :8000)
                 │                                              │
                 └─ app/api/ask (server only) ──► Gemini API    ├─ SQLite  data/app.db
                        └─ calls ML /retrieve, /frames/{id}/base64
                                                                ├─ FAISS   data/index/{frames,transcript}.faiss
                                                                └─ files   data/videos/, data/frames/{video_id}/
```

- `web/` — Next.js 15 App Router, TypeScript, Tailwind. All UI + `app/api/ask/route.ts` (the only Gemini caller; key never reaches the browser).
- `ml/` — Python 3.11 FastAPI. Storage, frame sampling, CLIP/MiniLM embeddings, faster-whisper transcription, FAISS search.
- `data/` — runtime storage, git-ignored.

### Indexing pipeline (background worker: single thread + `queue.Queue`)
`Video.stage`: uploaded → extracting_frames → embedding → transcribing → building_index → ready
1. Seek to each timestamp (every N s, N ∈ {1,2,5}, default 2) with OpenCV, save JPEG ≤512px wide to `data/frames/{video_id}/`.
2. CLIP `ViT-B-32` / `laion2b_s34b_b79k`, batches of 32, L2-normalize, add to frames index with id = `Frame.id`.
3. faster-whisper `base` (int8 on CPU); skip gracefully if no audio. Chunk into ~20–30 s segments, embed with `all-MiniLM-L6-v2`, normalize, add to transcript index with id = `TranscriptSegment.id`.
4. Save both indexes to disk, then status=ready.
On error: remove this video's vectors from both indexes, status=failed, readable `error_message`.

### Data model (SQLAlchemy / SQLite)
- **Video**: id (uuid), filename, stored_path, duration_sec, size_bytes, sample_interval_sec, status (queued|processing|ready|failed), stage, error_message, frame_count, created_at
- **Frame**: id (int PK == FAISS id), video_id (FK cascade), timestamp_sec, frame_path
- **TranscriptSegment**: id (int PK == FAISS id), video_id (FK cascade), start_sec, end_sec, text

### FAISS
`IndexIDMap2(IndexFlatIP)` — 512 dims (CLIP), 384 dims (MiniLM). Each wrapped in a thread-safe class with add/remove/search/save/load. Loaded on startup or created empty.

### ML API
- `GET /health`
- `POST /videos` (multipart file + sample_interval_sec; .mp4 only, ≤500 MB configurable, must open in OpenCV with readable duration, else 400 + file deleted)
- `GET /videos`, `GET /videos/{id}`, `GET /videos/{id}/file` (HTTP Range), `DELETE /videos/{id}`, `POST /videos/{id}/reindex`
- `GET /frames/{id}/image`, `GET /frames/{id}/base64`
- `POST /search {query, top_k=10 (≤50), video_id?, dedupe_window_sec?}` — search k = top_k*5, keep ready videos only, temporal dedupe per video (default window 2× interval, 0 disables), sorted by similarity desc
- `POST /retrieve {video_id, question, k_frames=6, k_chunks=4}` — top deduped frames + top transcript chunks + chunks overlapping those frames, sorted by time

### Web
- `/` Library: upload (XHR progress, size check, interval choice + trade-off text, estimated frames), video cards, re-index/delete/retry, polls every 2 s while anything is queued/processing.
- `/search`: query + example chips, Top-K 5/10/20, video filter, state in URL params, results link to `/videos/[id]?t=<sec>`. Show "similarity", never "confidence".
- `/videos/[id]`: player (`/ml/videos/{id}/file`), seek to `?t`, in-video search strip, Ask chat (streaming, citation chips, frame thumbnails, last 4 turns).
- `api/ask`: ≤6 frames at 512px, last 4 turns, `generateContentStream`, systemInstruction = answer only from material, cite `[mm:ss]`, say when not visible/mentioned, concise. 429/RESOURCE_EXHAUSTED → exponential backoff, 3 retries, then a clear "rate limit reached, try again in a minute" message. Clear errors for missing key / ML down.

## Hard rules
- Frames and text queries use the SAME CLIP model. Always L2-normalize before inner-product search.
- FAISS vector id == SQLite primary key. Never rely on insertion order.
- A video is `ready` only after frames, embeddings, transcript and saved indexes are all complete. Partial indexes are never searchable.
- Indexing never runs inside a request handler.
- Search and Ask never re-extract or re-embed anything.
- A result must never point to the wrong video or timestamp.
- Be honest about limits: static visual content matches well; actions over time don't; no face identification.
- Ask the user before adding any dependency not in the spec.

## Allowed dependencies
- Python: fastapi, uvicorn, sqlalchemy, opencv-python, open_clip_torch, torch, faiss-cpu, faster-whisper, sentence-transformers, pytest (+ approved additions listed below)
- Web: next, react, tailwind, @google/genai
- Approved additions: python-multipart, httpx (TestClient), numpy, pillow (direct imports), av/PyAV (frame-accurate decoding; also a faster-whisper dep). Approved: MPS for CLIP on Apple Silicon (CPU fallback); Whisper stays CPU int8.

## Config
- `web/.env.local`: `GEMINI_API_KEY`, `GEMINI_MODEL`, `ML_URL=http://localhost:8000`
- ML env (optional): `VS_DATA_DIR` (default `../data`), `VS_MAX_UPLOAD_MB` (default 500)

## Run commands (macOS, Apple Silicon)
```bash
# ML service
cd ml
uv venv --python 3.11 .venv          # uv lives in ~/.local/bin
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Tests
cd ml && source .venv/bin/activate && pytest -q   # fast suite (fake models)
cd ml && .venv/bin/pytest -m slow                 # real CLIP sanity check
cd web && npm test                                # Gemini retry logic (node:test, no deps)

# Web
cd web
npm install
cp .env.example .env.local   # then fill in GEMINI_API_KEY and GEMINI_MODEL
npm run dev                  # http://localhost:3000

# Evaluation (Phase 6)
cd ml && python scripts/evaluate.py
```

## Workflow
Build in phases (0–6), stop after each with a summary and exact test steps, and wait for the user's OK before continuing.
Current phase: **All phases (0-6) complete.** Further work: ask the user.

## Environment notes
- macOS arm64. Python 3.11 installed via uv (no Homebrew). Node 24.
- Installed versions: torch 2.14, OpenCV 5.0, faiss 1.15, Next 15.5, Tailwind v4, @google/genai 2.x.
- Harmless at import: objc "AVFFrameReceiver implemented in both" warning (PyAV and OpenCV each bundle libavdevice; we never use camera capture). Also a Starlette TestClient httpx deprecation warning.
- Gemini: `GEMINI_MODEL=gemini-flash-latest` (alias, resolved to gemini-3.8-flash on 2026-10-15). Key verified. Gemini also returns 503 UNAVAILABLE ("high demand"), so Ask retries must cover 503 as well as 429/RESOURCE_EXHAUSTED.
- web/next.config.ts: `experimental.proxyTimeout` (10 min) AND `experimental.middlewareClientMaxBodySize: "520mb"`. Without the latter Next.js truncates proxied bodies at 10 MB and large uploads hang.

## Implementation notes
- Paths in DB (`stored_path`, `frame_path`) are relative to `config.DATA_DIR`; use the `abs_path` properties.
- Frame/TranscriptSegment tables use SQLite AUTOINCREMENT so ids (== FAISS ids) are never reused.
- Upload errors: 400 (wrong type, empty, unreadable, bad interval), 413 (over size limit). A rejected file is always deleted.
- Frames: decoded with **PyAV**, not OpenCV seeking (OpenCV uses average fps -> wrong by seconds on VFR video). For each sample t in 0, N, 2N... < duration: seek to the keyframe before t (or keep decoding forward if t is <= 2 s ahead), take the first frame with pts >= t, store its REAL pts as timestamp_sec. Display-matrix rotation applied like ffmpeg. OpenCV still validates uploads and resizes/writes JPEGs. Files: `data/frames/{video_id}/{ms:09d}.jpg`. Duration from the container (PyAV), fallback frame_count/fps.
- Worker re-queues queued/processing videos on startup and clears derived data before (re)processing (`clear_derived_data`).
- **OMP_NUM_THREADS=1 is forced in `ml/app/__init__.py`.** faiss-cpu and torch each bundle libomp; on macOS, multi-threaded torch CPU ops + faiss in one process segfault (reproduced while instantiating CLIP). Never import torch/faiss before `app`. CLIP runs on MPS so speed is unaffected; check Whisper's speed under this in Phase 5 (set faster-whisper `cpu_threads` explicitly).
- `Services` (app/services.py) owns both indexes + lazy CLIP; `app.state.services`. `clear_video_data()` = vectors -> rows -> frame files; used by worker, delete, reindex, failure.
- Startup `reconcile()`: drops index vectors not belonging to ready videos; re-queues ready videos whose vectors are missing.
- Search over-fetches top_k*5, maps ids->rows via SQL join filtered to status=ready, dedupes, and widens k (x4) if too few remain. Video filter uses a FAISS IDSelector over that video's ids.
- Dedupe: per video, drop if within window (inclusive) of a better kept hit; default window = 2 x that video's interval; 0 disables.
- `thumbnail_url` = `/ml/frames/{id}/image` (`VS_PUBLIC_PREFIX`).
- Upload limit: single source of truth is `VS_MAX_UPLOAD_MB`; `/health` exposes it and the UploadPanel reads it. A middleware returns 413 from Content-Length before the body is spooled.
- `_resolve` chunks `IN (...)` lists (900 ids) to stay under SQLite's variable limit.
- Test videos: `ml/scripts/make_test_videos.py` (macOS: wallpapers, Apple emoji, `say`) -> `data/test_videos/` (10 videos incl. no-audio, VFR, portrait, 5-min lecture, MPEG-4 Part 2; `invalid/`; TEST_GUIDE.md; eval_set.json). Baseline on them: Recall@5 ~0.96, MRR@10 ~0.90.
- Evaluation: `ml/scripts/evaluate.py` (synthetic benchmark by default, `--eval-set` for labelled videos; runs in a temp data dir).
- Web: `lib/api.ts` (all /ml calls; maps proxy 5xx/network errors to a "can't reach the ML service" message), `lib/format.ts`, `lib/types.ts`. Modals use native `<dialog>` (never window.confirm/alert).
- Browsers can't play every .mp4: Chrome can't decode MPEG-4 Part 2 (OpenCV's `mp4v`) or usually HEVC. OpenCV still indexes them, so the video page must show a clear "your browser can't play this codec" message. Use H.264 test videos for UI work (PyAV has libx264 + aac).
- Search page state lives in URL params `q`, `k` (5/10/20, default 10 omitted), `video`. Result links go to `/videos/{id}?t=<sec>`.
- `VideoPlayer` (forwardRef `seek(sec)`): applies ?t after `loadedmetadata`, clamps, shows "Jumped to mm:ss" for 4s, shows a codec message on MediaError instead of a broken player.
- `<body suppressHydrationWarning>` is there because extensions (Grammarly) inject body attributes.
- Transcripts: `app/transcribe.py`. `has_audio()` (PyAV) gates Whisper; `chunk_segments()` merges Whisper segments into 20-30 s chunks (pure, tested). Whisper uses `cpu_threads=os.cpu_count()` explicitly (needed because OMP_NUM_THREADS=1). `vad_filter=True`.
- `Services` has lazy `clip`, `text` (MiniLM), `transcriber` (Whisper); tests inject FakeClip/FakeText/FakeTranscriber via `app.state.*_factory`.
- `/retrieve`: frames (CLIP, deduped, time-sorted) + chunks (MiniLM "match" + "overlap" with retrieved frames), time-sorted. 409 unless ready.
- Ask: `web/app/api/ask/route.ts` -> `lib/gemini.ts` (prompt + `startStreamWithRetry`: 429/503, 3 retries, 1/2/4 s + jitter; first chunk read inside retry loop so nothing is streamed before success; 45 s per-attempt first-chunk timeout because Gemini sometimes hangs under load (observed >5 min); no new attempt after 90 s; client disconnect is never retried). Response is NDJSON events (`lib/ask.ts`): sources, delta*, done|error. History = last 4 Q/A turns, text only. Errors: 500 missing key/model, 503 ML down or Gemini overloaded, 429 rate limit, 502 other Gemini errors.
- Retrieval uses only the current question (not history), so vague follow-ups ("and after that?") retrieve poorly; known limitation.
- Chat UI: `components/AskPanel.tsx`, `components/AnswerText.tsx` (citations -> chips incl. ranges/lists; times past the video end stay plain text; light markdown).
- Layout gotcha: grids that contain horizontal scrollers need `grid-cols-1`/`minmax(0,1fr)` + `min-w-0`, otherwise mobile overflows.
- Browser testing: the Claude-in-Chrome window is usually `document.hidden`, so Chrome never loads <video> media there. For player/seek tests use headless Chrome via CDP (`--headless=new --remote-debugging-port=9333`) with a small Node driver script (Node 24 has fetch + WebSocket built in); no npm deps needed.
- Stop the ML server with `pkill -f "app.main:app"` (matches both `uvicorn app.main:app` and `--app-dir` launches).
- Tests use `FakeClip` (conftest): frame embedding = bump at its second (from brightness); text "second N" = bump at N. Real-model test: `pytest -m slow`.
- Tests: conftest points `VS_DATA_DIR` at a temp dir; synthetic videos encode the second in frame brightness (`GRAY_STEP`) so wrong seeks are detectable.
