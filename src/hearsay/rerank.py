import os
import sys

from .paths import models_dir

# The int8 file of the same model: same ranking on our query sets, under half the time on cpu.
REPO = "Xenova/ms-marco-MiniLM-L-12-v2"
MODEL_FILE = "onnx/model_quantized.onnx"
MODEL = "hearsay/ms-marco-MiniLM-L-12-v2-q"
# More threads made it slower here: the embedder has its own pool and they fight for cores.
THREADS = min(4, os.cpu_count() or 1)


class Reranker:
    """Cross-encoder that reads the query and a passage together and scores the pair.
    Slower than the embedding lookup, so it only sees the top few fused hits."""

    def __init__(self, model=MODEL, allow_download=False):
        self.model_name = model
        self.allow_download = allow_download
        self._m = None

    def _load(self):
        from fastembed.common.model_description import ModelSource
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        if self.model_name == MODEL:
            try:
                TextCrossEncoder.add_custom_model(model=MODEL, sources=ModelSource(hf=REPO), model_file=MODEL_FILE)
            except ValueError:
                pass  # already registered in this process
        self._m = TextCrossEncoder(
            self.model_name,
            cache_dir=str(models_dir() / "fastembed"),
            local_files_only=not self.allow_download,
            threads=THREADS,
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


# Measured on 65 h with 91 real and 40 off-topic queries (top hit of each):
#   below -9: no real query, 33 of the 40 off-topic ones -> dropped
#   -9 to -5: 7 real queries and 6 off-topic ones        -> kept, but marked weak
#   above -5: 84 real queries, 1 off-topic one
# Dropping is kept conservative on purpose: losing a real answer costs more than showing a
# weak one that is labelled as such.
DEFAULT_MIN_RELEVANCE = -9.0
WEAK_BELOW = -5.0


def strength(relevance: float | None) -> str | None:
    if relevance is None:
        return None
    return "weak" if relevance < WEAK_BELOW else "strong"


def min_relevance() -> float | None:
    """HEARSAY_MIN_RELEVANCE overrides the default; 'off' disables the cutoff."""
    v = os.environ.get("HEARSAY_MIN_RELEVANCE")
    if v is None:
        return DEFAULT_MIN_RELEVANCE
    return None if v.lower() == "off" else float(v)
