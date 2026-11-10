"""ctranslate2 wants libcublas/libcudnn from CUDA 12. The pip wheels
(nvidia-cublas-cu12, nvidia-cudnn-cu12) put them in site-packages where the
dynamic loader never looks, so load them ourselves before ctranslate2 starts."""
import ctypes
import glob
import importlib.util
import os
import sys


def preload() -> None:
    if not sys.platform.startswith("linux"):
        return
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        libdir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        # load in name order so e.g. libcudnn.so.9 pulls in its sub-libraries
        for so in sorted(glob.glob(os.path.join(libdir, "lib*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
