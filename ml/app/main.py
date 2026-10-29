import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config
from app.db import init_db
from app.embeddings import ClipEmbedder
from app.routes import frames, health, retrieve, search, videos
from app.services import Services
from app.transcribe import MiniLMEmbedder, WhisperTranscriber
from app.worker import Worker

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Third-party noise: keep model-download chatter at WARNING unless debugging.
for noisy in ("httpx", "huggingface_hub", "urllib3", "faster_whisper"):
    logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLevelName(config.LOG_LEVEL)))
log = logging.getLogger("visionfind")


def _preload(services: Services) -> None:
    for name in ("clip", "text", "transcriber"):
        try:
            getattr(services, name)
        except Exception:
            log.exception("Preloading %s failed; it will be retried on first use", name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    init_db()
    # Tests swap in fake models by setting app.state.*_factory before startup.
    services = Services(
        clip_factory=getattr(app.state, "clip_factory", ClipEmbedder),
        text_factory=getattr(app.state, "text_factory", MiniLMEmbedder),
        transcriber_factory=getattr(app.state, "transcriber_factory", WhisperTranscriber),
    )
    services.reconcile()
    app.state.services = services
    app.state.worker = Worker(services)
    app.state.worker.start()
    if config.PRELOAD_MODELS:
        threading.Thread(target=_preload, args=(services,), name="preload", daemon=True).start()
    log.info("Data dir: %s | frames index: %d vectors", config.DATA_DIR, services.frame_index.ntotal)
    yield
    app.state.worker.stop()


app = FastAPI(title="VisionFind ML", version="0.5.0", lifespan=lifespan)


@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    """Refuse oversized uploads from the Content-Length header, before the body is received and spooled.
    (The route also counts bytes while saving, for clients that don't send Content-Length.)"""
    if request.method == "POST" and request.url.path == "/videos":
        length = request.headers.get("content-length")
        # multipart framing adds a little overhead on top of the file itself
        if length and length.isdigit() and int(length) > config.MAX_UPLOAD_BYTES + 1024 * 1024:
            return JSONResponse({"detail": f"File is larger than the {config.MAX_UPLOAD_MB} MB limit."}, status_code=413)
    return await call_next(request)


app.include_router(health.router)
app.include_router(videos.router)
app.include_router(frames.router)
app.include_router(search.router)
app.include_router(retrieve.router)
