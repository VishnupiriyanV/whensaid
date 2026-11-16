import os

import numpy as np

from .paths import models_dir

MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self, model=MODEL, allow_download=False):
        self.model_name = model
        self.allow_download = allow_download
        self._m = None

    def _load(self):
        from fastembed import TextEmbedding
        self._m = TextEmbedding(
            self.model_name,
            cache_dir=str(models_dir() / "fastembed"),
            local_files_only=not self.allow_download,
            threads=min(4, os.cpu_count() or 1),  # see rerank.THREADS
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._m is None:
            self._load()
        vecs = np.array(list(self._m.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)
