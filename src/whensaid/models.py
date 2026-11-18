"""Which models are on disk, and getting them. Nothing here loads a model onto the GPU."""
import contextlib
import os
import socket
import sys
import threading

from .paths import models_dir

# rough sizes, only for the message shown before a download
SIZES = {"whisper": "1.5 GB", "embedder": "70 MB", "reranker": "35 MB"}
WHISPER_SIZES = {"distil-large-v3": "1.5 GB", "small.en": "0.5 GB", "tiny.en": "75 MB", "base.en": "150 MB"}


def size_label(kind: str) -> str:
    return WHISPER_SIZES.get(whisper_name(), "large") if kind == "whisper" else SIZES[kind]
SEARCH = ("embedder", "reranker")  # what `search` and the server need
INDEX = ("whisper", "embedder")    # what `index` needs


class ModelsMissing(Exception):
    pass


def offline() -> bool:
    return os.environ.get("WHENSAID_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"


def whisper_name() -> str:
    from .transcribe import MODEL

    return os.environ.get("WHENSAID_WHISPER_MODEL") or os.environ.get("HEARSAY_WHISPER_MODEL") or MODEL


def present(kind: str) -> bool:
    """On disk and complete. A download that was cut off does not count."""
    if kind not in SIZES:
        raise ValueError(f"unknown model kind {kind!r}")
    try:
        from loguru import logger  # fastembed logs "ERROR ... could not find the model" for a plain "not there yet"

        logger.disable("fastembed")
    except ImportError:
        logger = None
    try:
        if kind == "whisper":
            from faster_whisper.utils import download_model

            path = download_model(whisper_name(), local_files_only=True, cache_dir=str(models_dir()))
            return os.path.isfile(os.path.join(path, "model.bin"))
        if kind == "embedder":
            from .embed import Embedder

            Embedder()._load()
        else:
            from .rerank import Reranker

            Reranker()._load()
        return True
    except Exception:  # fastembed raises ValueError for "not on disk", hf raises various things
        return False
    finally:
        if logger is not None:
            logger.enable("fastembed")


@contextlib.contextmanager
def ipv4_only():
    """Some networks have a broken IPv6 route: a browser or curl shrugs it off, python sits on it
    until the timeout. Downloads from a CDN work fine over IPv4, so use only that."""
    real = socket.getaddrinfo

    def v4(host, port, family=0, type=0, proto=0, flags=0):
        return real(host, port, socket.AF_INET if family in (0, socket.AF_UNSPEC) else family, type, proto, flags)

    socket.getaddrinfo = v4
    try:
        yield
    finally:
        socket.getaddrinfo = real


def download(kinds, log=sys.stderr) -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")  # the default is 10 s per read
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    with ipv4_only():
        for kind in kinds:
            print(f"downloading the {kind} model ({size_label(kind)}) ...", file=log, flush=True)
            if kind == "whisper":
                from faster_whisper.utils import download_model

                download_model(whisper_name(), cache_dir=str(models_dir()))
            elif kind == "embedder":
                from .embed import Embedder

                Embedder(allow_download=True)._load()
            else:
                from .rerank import Reranker

                Reranker(allow_download=True)._load()
            if not present(kind):
                raise RuntimeError(f"the {kind} model is still not usable after downloading")


def ensure(kinds, log=sys.stderr) -> None:
    """Make sure these models are on disk, downloading what is missing."""
    missing = [k for k in kinds if not present(k)]
    if not missing:
        return
    if offline():
        raise ModelsMissing(
            f"missing models: {', '.join(missing)}. Offline mode is on (WHENSAID_OFFLINE / HF_HUB_OFFLINE); "
            "run `whensaid models pull` once with a connection."
        )
    try:
        download(missing, log)
    except Exception as e:
        raise ModelsMissing(
            f"could not download the {', '.join(missing)} model: {e}. "
            "Check the connection and try `whensaid models pull`."
        ) from e


def wanted_for_search() -> tuple:
    return tuple(k for k in SEARCH if not (k == "reranker" and os.environ.get("WHENSAID_RERANK", os.environ.get("HEARSAY_RERANK")) == "0"))


class Prep:
    """Gets the search models in the background so the server can start at once and answer
    "still downloading" instead of making the first tool call wait minutes."""

    def __init__(self, kinds=None):
        self.kinds = tuple(kinds or wanted_for_search())
        self.state = {k: "waiting" for k in self.kinds}
        self.error: str | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> "Prep":
        self.thread = threading.Thread(target=self._run, daemon=True, name="models")
        self.thread.start()
        return self

    def _run(self):
        for k in self.kinds:
            try:
                if present(k):
                    self.state[k] = "ready"
                    continue
                if offline():
                    raise ModelsMissing(f"the {k} model is missing and offline mode is on")
                self.state[k] = "downloading"
                download([k], log=open(os.devnull, "w"))  # stdout belongs to the protocol, and stderr is noise here
                self.state[k] = "ready"
            except Exception as e:
                self.state[k] = "failed"
                self.error = f"{k}: {e}"
                return

    def ready(self) -> bool:
        return all(v == "ready" for v in self.state.values())

    def describe(self) -> dict:
        return {"ready": self.ready(), "models": dict(self.state), "error": self.error}

    def message(self) -> str:
        if self.error:
            return (f"The search models could not be downloaded ({self.error}). Tell the user to run "
                    "`whensaid models pull` in a terminal.")
        left = [f"{k} ({size_label(k)})" for k, v in self.state.items() if v != "ready"]
        return ("The search models are still downloading the first time: " + ", ".join(left) +
                ". Try again in a minute.")
