"""ctranslate2 wants libcublas/libcudnn from CUDA 12. The pip wheels
(nvidia-cublas-cu12, nvidia-cudnn-cu12) put them in site-packages where the
dynamic loader never looks, so load them ourselves before ctranslate2 starts."""
import ctypes
import glob
import importlib.util
import os
import sys


def _spec(pkg: str):
    try:
        return importlib.util.find_spec(pkg)
    except (ModuleNotFoundError, ValueError):  # "nvidia" itself is missing when the cuda extra isn't installed
        return None


def preload() -> None:
    if not sys.platform.startswith("linux"):
        return
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = _spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        libdir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        # load in name order so e.g. libcudnn.so.9 pulls in its sub-libraries
        for so in sorted(glob.glob(os.path.join(libdir, "lib*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def libs_load() -> bool:
    """Can the CUDA 12 libraries ctranslate2 needs be found, from the pip wheels or the system?"""
    if not sys.platform.startswith("linux"):
        return False
    try:
        for name in ("libcublas.so.12", "libcudnn.so.9"):
            ctypes.CDLL(name)
        return True
    except OSError:
        return False
