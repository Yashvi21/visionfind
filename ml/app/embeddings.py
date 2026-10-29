"""CLIP embeddings. Frames and text queries MUST use this same model; outputs are L2-normalized."""

import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from app import config
from app.vector_index import l2_normalize

log = logging.getLogger(__name__)


class ImageTextEmbedder(Protocol):
    dim: int

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray: ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...


def pick_device() -> str:
    import torch

    forced = os.environ.get("VS_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEmbedder:
    dim = config.CLIP_DIM

    def __init__(self, device: str | None = None) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.device = device or pick_device()
        log.info("Loading CLIP %s/%s on %s", config.CLIP_MODEL, config.CLIP_PRETRAINED, self.device)
        try:
            self._load(open_clip)
        except Exception:
            if self.device == "cpu":
                raise
            log.exception("Could not load CLIP on %s; falling back to CPU", self.device)
            self.device = "cpu"
            self._load(open_clip)
        # One forward pass at a time: the worker and search requests share the model.
        self._lock = threading.Lock()

    def _load(self, open_clip) -> None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED, device=self.device
        )
        model.eval()
        self.model, self.preprocess = model, preprocess
        self.tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        out = []
        for start in range(0, len(paths), config.CLIP_BATCH_SIZE):
            batch_paths = paths[start : start + config.CLIP_BATCH_SIZE]
            images = []
            for p in batch_paths:
                with Image.open(p) as im:
                    images.append(self.preprocess(im.convert("RGB")))
            batch = torch.stack(images).to(self.device)
            with self._lock, torch.inference_mode():
                feats = self.model.encode_image(batch)
            out.append(feats.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return l2_normalize(np.concatenate(out))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        tokens = self.tokenizer(list(texts)).to(self.device)
        with self._lock, torch.inference_mode():
            feats = self.model.encode_text(tokens)
        return l2_normalize(feats.float().cpu().numpy())
