"""Sanity check with the real CLIP model (slow; downloads weights on first run). Run: pytest -m slow"""

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def clip():
    from app.embeddings import ClipEmbedder

    return ClipEmbedder()


def _solid(tmp_path, name, bgr):
    p = tmp_path / f"{name}.jpg"
    cv2.imwrite(str(p), np.full((224, 224, 3), bgr, np.uint8))
    return p


def test_text_and_images_share_a_space(clip, tmp_path):
    red, blue = _solid(tmp_path, "red", (0, 0, 255)), _solid(tmp_path, "blue", (255, 0, 0))
    imgs = clip.encode_images([red, blue])
    txt = clip.encode_text(["a plain red image", "a plain blue image"])
    assert imgs.shape == (2, 512) and txt.shape == (2, 512)
    assert np.allclose(np.linalg.norm(imgs, axis=1), 1, atol=1e-4)
    assert np.allclose(np.linalg.norm(txt, axis=1), 1, atol=1e-4)
    sims = txt @ imgs.T
    assert sims[0, 0] > sims[0, 1]  # "red" closer to the red image
    assert sims[1, 1] > sims[1, 0]  # "blue" closer to the blue image
