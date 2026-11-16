import os
import sys

from .chunk import Segment
from .paths import models_dir

# distil-large-v3 is english only, which is what we target first. On a 12 core cpu it ran at
# ~6x realtime with ~4% difference from the gpu transcript, small.en was ~16x but ~17% off.
MODEL = "distil-large-v3"
FAST_MODEL = "small.en"  # for slow machines: WHENSAID_WHISPER_MODEL=small.en


def pick_device() -> tuple[str, str, str]:
    """Returns (device, compute_type, model_name)."""
    from . import cudalibs
    cudalibs.preload()
    model = os.environ.get("WHENSAID_WHISPER_MODEL", MODEL)
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16", model
    except Exception:
        pass
    return "cpu", "int8", model


class NoAudio(Exception):
    pass


class Unintelligible(Exception):
    """The model was guessing: most likely the speech is not English, or it is not speech."""


# Mean probability the model gives its own words. Measured on 3 minute samples: English of every
# kind we have (clean, 1940s radio, film, sports commentary, Indian-accented) 0.90 to 0.99; French,
# German, Spanish, Japanese, Russian, Hindi and Chinese forced through as English 0.46 to 0.71.
# The model's own language detection is no use here, it answers "en" for all of them.
MIN_CONFIDENCE = 0.75
MIN_WORDS = 40  # below this there is too little to judge, keep whatever came out


def min_confidence() -> float:
    """WHENSAID_MIN_CONFIDENCE overrides the default, 0 turns the check off."""
    return float(os.environ.get("WHENSAID_MIN_CONFIDENCE", MIN_CONFIDENCE))


class Transcriber:
    def __init__(self, device=None, compute_type=None, model=None, allow_download=False):
        d, c, m = pick_device()
        self.device = device or d
        self.compute_type = compute_type or c
        self.model_name = model or m
        self.allow_download = allow_download
        self._pipe = None
        self.last_confidence = None  # of the most recent transcribe() call

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
        out, probs = [], []
        for s in segs:
            out += split_on_gaps(s.words or [], fallback=s)
            probs += [w.probability for w in (s.words or [])]
        self.last_confidence = sum(probs) / len(probs) if probs else None
        if len(probs) >= MIN_WORDS and self.last_confidence < min_confidence():
            raise Unintelligible(
                f"transcript unreliable (mean word confidence {self.last_confidence:.2f}): "
                "probably not English, or not speech"
            )
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
