import os
import sys

from .paths import models_dir

MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"


class Reranker:
    """Cross-encoder that reads the query and a passage together and scores the pair.
    Slower than the embedding lookup, so it only sees the top few fused hits."""

    def __init__(self, model=MODEL, allow_download=False):
        self.model_name = model
        self.allow_download = allow_download
        self._m = None

    def _load(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self._m = TextCrossEncoder(
            self.model_name,
            cache_dir=str(models_dir() / "fastembed"),
            local_files_only=not self.allow_download,
        )

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._m is None:
            self._load()
        return [float(s) for s in self._m.rerank(query, texts)]


def load_reranker():
    """The reranker if its model is on disk, otherwise None and search runs without it.
    HEARSAY_RERANK=0 turns it off."""
    if os.environ.get("HEARSAY_RERANK") == "0":
        return None
    r = Reranker()
    try:
        r._load()
    except Exception:
        print("hearsay: reranker model not found, run `hearsay models pull`; searching without it", file=sys.stderr)
        return None
    return r


# Scores seen so far: off-topic queries top out around -10.5 (one outlier at -7.5), real
# queries are above -8. Set low on purpose, a dropped real answer costs more than a weak one.
DEFAULT_MIN_RELEVANCE = -9.0


def min_relevance() -> float | None:
    """HEARSAY_MIN_RELEVANCE overrides the default; 'off' disables the cutoff."""
    v = os.environ.get("HEARSAY_MIN_RELEVANCE")
    if v is None:
        return DEFAULT_MIN_RELEVANCE
    return None if v.lower() == "off" else float(v)
