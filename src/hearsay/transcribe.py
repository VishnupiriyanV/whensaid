import sys

from .chunk import Segment
from .paths import models_dir

# distil-large-v3 is english only, which is what we target first
GPU_MODEL = "distil-large-v3"
CPU_MODEL = "small.en"


def pick_device() -> tuple[str, str, str]:
    """Returns (device, compute_type, model_name)."""
    from . import cudalibs
    cudalibs.preload()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16", GPU_MODEL
    except Exception:
        pass
    return "cpu", "int8", CPU_MODEL


class NoAudio(Exception):
    pass


class Transcriber:
    def __init__(self, device=None, compute_type=None, model=None, allow_download=False):
        d, c, m = pick_device()
        self.device = device or d
        self.compute_type = compute_type or c
        self.model_name = model or m
        self.allow_download = allow_download
        self._pipe = None

    def _load(self):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(models_dir()),
            local_files_only=not self.allow_download,
        )
        self._pipe = BatchedInferencePipeline(model)

    def transcribe(self, path: str) -> tuple[list[Segment], float]:
        """Returns (segments, audio_duration_seconds). Raises NoAudio."""
        from faster_whisper.audio import decode_audio

        try:
            audio = decode_audio(path, sampling_rate=16000)
        except IndexError:
            # pyav gives us an IndexError when there is no audio stream at all
            raise NoAudio(path)
        if len(audio) == 0:
            raise NoAudio(path)

        if self._pipe is None:
            self._load()

        # The batched pipeline returns one big segment per vad block unless we ask
        # for timestamps, and even then a block can glue a stray noise at 0:02 to
        # speech at 2:26. Word times are the only reliable ones, so use those.
        segs, info = self._pipe.transcribe(
            audio,
            batch_size=8 if self.device == "cuda" else 4,
            vad_filter=True,
            language="en",
            without_timestamps=False,
            word_timestamps=True,
        )
        out = []
        for s in segs:
            out += split_on_gaps(s.words or [], fallback=s)
        return out, len(audio) / 16000


def split_on_gaps(words, fallback=None, max_gap: float = 2.5) -> list[Segment]:
    """Turn a segment's words into Segments, cutting wherever the speaker went
    quiet for more than max_gap seconds."""
    if not words:
        if fallback is not None and fallback.text.strip():
            return [Segment(fallback.start, fallback.end, fallback.text.strip())]
        return []

    out, cur = [], [words[0]]
    for w in words[1:]:
        if w.start - cur[-1].end > max_gap:
            out.append(cur)
            cur = []
        cur.append(w)
    out.append(cur)

    return [
        Segment(float(g[0].start), float(g[-1].end), "".join(w.word for w in g).strip())
        for g in out
    ]
