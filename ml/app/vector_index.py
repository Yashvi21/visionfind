"""Thread-safe FAISS index whose vector ids are SQLite primary keys (never insertion order)."""

import logging
import os
import threading
from pathlib import Path

import faiss
import numpy as np

log = logging.getLogger(__name__)


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.ascontiguousarray(vecs / np.maximum(norms, 1e-12), dtype=np.float32)


class VectorIndex:
    """IndexIDMap2(IndexFlatIP). Inputs are L2-normalized here, so inner product == cosine similarity."""

    def __init__(self, dim: int, path: Path) -> None:
        self.dim = dim
        self.path = Path(path)
        self._lock = threading.RLock()
        self._index = self._new()

    def _new(self) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    # -- persistence -----------------------------------------------------
    @classmethod
    def load_or_create(cls, dim: int, path: Path) -> "VectorIndex":
        idx = cls(dim, path)
        if idx.path.exists():
            loaded = faiss.read_index(str(idx.path))
            if not isinstance(loaded, faiss.IndexIDMap2) or loaded.d != dim:
                raise ValueError(f"Index {path} is not an IndexIDMap2 with dim {dim}")
            idx._index = loaded
            log.info("Loaded %s with %d vectors", path.name, idx.ntotal)
        else:
            log.info("Created empty index %s", path.name)
        return idx

    def save(self) -> None:
        """Write to a temp file then rename, so a crash never leaves a half-written index."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            faiss.write_index(self._index, str(tmp))
            os.replace(tmp, self.path)

    # -- mutation --------------------------------------------------------
    def add(self, ids, vectors) -> None:
        ids = np.asarray(list(ids), dtype=np.int64)
        vecs = l2_normalize(vectors)
        if len(ids) != len(vecs):
            raise ValueError("ids and vectors must have the same length")
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vecs.shape[1]}")
        if len(ids) == 0:
            return
        with self._lock:
            # Re-adding an id would create a duplicate entry; remove first so add is idempotent.
            self._index.remove_ids(faiss.IDSelectorBatch(ids))
            self._index.add_with_ids(vecs, ids)

    def remove(self, ids) -> int:
        ids = np.asarray(list(ids), dtype=np.int64)
        if len(ids) == 0:
            return 0
        with self._lock:
            return int(self._index.remove_ids(faiss.IDSelectorBatch(ids)))

    # -- queries ---------------------------------------------------------
    @property
    def ntotal(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    def ids(self) -> set[int]:
        with self._lock:
            return set(faiss.vector_to_array(self._index.id_map).tolist())

    def search(self, query, k: int, restrict_ids=None) -> list[tuple[int, float]]:
        """Return [(id, similarity)] best first. `restrict_ids` limits the search to those ids."""
        q = l2_normalize(query)
        with self._lock:
            k = min(k, self._index.ntotal)
            if k <= 0:
                return []
            params = None
            if restrict_ids is not None:
                allowed = np.asarray(list(restrict_ids), dtype=np.int64)
                if len(allowed) == 0:
                    return []
                k = min(k, len(allowed))
                params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
            scores, ids = self._index.search(q, k, params=params)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
