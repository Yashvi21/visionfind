"""Central configuration. Every value can be overridden with a VS_* environment variable."""

import os
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ML_ROOT.parent

DATA_DIR = Path(os.environ.get("VS_DATA_DIR", REPO_ROOT / "data")).resolve()
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "app.db"

MAX_UPLOAD_MB = int(os.environ.get("VS_MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_INTERVALS = (1, 2, 5)
DEFAULT_INTERVAL = 2
FRAME_MAX_WIDTH = 512

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_DIM = 512
CLIP_BATCH_SIZE = 32
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_DIM = 384
WHISPER_MODEL = os.environ.get("VS_WHISPER_MODEL", "base")


def ensure_dirs() -> None:
    for d in (DATA_DIR, VIDEOS_DIR, FRAMES_DIR, INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Prefix the browser uses to reach this service (the Next.js rewrite), used in thumbnail URLs.
PUBLIC_PREFIX = os.environ.get("VS_PUBLIC_PREFIX", "/ml")
# Load CLIP in the background at startup so the first search isn't slow. Tests turn this off.
PRELOAD_MODELS = os.environ.get("VS_PRELOAD_MODELS", "1") == "1"
MAX_TOP_K = 50
LOG_LEVEL = os.environ.get("VS_LOG_LEVEL", "INFO").upper()
