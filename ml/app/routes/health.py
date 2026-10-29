from fastapi import APIRouter

from app import config

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Liveness + the limits the web client should enforce (single source of truth)."""
    return {
        "status": "ok",
        "service": "visionfind-ml",
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "allowed_intervals": list(config.ALLOWED_INTERVALS),
    }
