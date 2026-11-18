"""Which models are on disk, and getting them. Nothing here loads a model onto the GPU."""
import contextlib
import os
import socket
import sys
import threading
import time

from .paths import home, models_dir

# read by huggingface_hub when it is first imported; download() also sets the live values
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

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


_ACTIVE = {"v4": False}  # for tests: is the IPv4-only filter on right now


def _flag():
    return home() / "ipv4"


def _ipv4_known_needed() -> bool:
    f = _flag()
    return f.exists() and time.time() - f.stat().st_mtime < 86400


def ipv6_broken(host: str = "huggingface.co", timeout: float = 4.0) -> bool:
    """True when the host has an IPv6 address that does not answer while IPv4 does. That is the
    case where python (and the Rust downloader) wait for minutes and a browser does not notice."""
    def connects(family) -> bool | None:
        try:
            addrs = socket.getaddrinfo(host, 443, family, socket.SOCK_STREAM)
        except OSError:
            return None  # no address of that kind at all
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            try:
                sock.connect(addrs[0][4])
                return True
            except OSError:
                return False

    if connects(socket.AF_INET6) is not False:  # works, or there is no IPv6 to be broken
        return False
    return connects(socket.AF_INET) is True


def _set_timeouts():
    # huggingface_hub reads these when it is imported, which has usually happened already
    # (present() imports it), so set the live values as well as the environment
    try:
        from huggingface_hub import constants

        constants.HF_HUB_DOWNLOAD_TIMEOUT = 30
        constants.HF_HUB_ETAG_TIMEOUT = 30
    except Exception:
        pass


def _no_xet():
    """hf_xet moves the bytes in Rust, past python's sockets, so the IPv4 filter would not reach it."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import constants

        constants.HF_HUB_DISABLE_XET = True
    except Exception:
        pass


def _fetch(kinds, log):
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


def download(kinds, log=sys.stderr) -> None:
    """Normally first. If that fails (a stalled IPv6 route is the usual cause: curl or a browser
    shrug it off, python waits for the timeout), once more over IPv4 with the Rust downloader
    off, and remember that for a day so the next command does not sit through the stall again."""
    _set_timeouts()
    if not _ipv4_known_needed() and ipv6_broken():
        print("the IPv6 route to huggingface.co does not answer; using IPv4 only", file=log, flush=True)
        _flag().touch()
    attempts = [True] if _ipv4_known_needed() else [False, True]
    last = None
    for v4 in attempts:
        try:
            if v4:
                _no_xet()
                with ipv4_only():
                    _ACTIVE["v4"] = True
                    _fetch(kinds, log)
                if attempts[0] is False:  # the normal way failed and this worked
                    _flag().touch()
            else:
                _fetch(kinds, log)
            return
        except Exception as e:
            last = e
            if not v4:
                print(f"download failed ({str(e)[:100]}); trying again over IPv4 only ...", file=log, flush=True)
        finally:
            _ACTIVE["v4"] = False
    raise last


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
