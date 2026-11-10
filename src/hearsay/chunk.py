from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    text: str


def make_chunks(segments: list[Segment], target: float = 45.0, overlap: float = 6.0) -> list[Segment]:
    """Group whisper segments into windows of roughly `target` seconds.

    Each new window starts with the last few segments of the previous one
    (up to `overlap` seconds) so a sentence on the boundary shows up in both.
    """
    chunks: list[Segment] = []
    cur: list[Segment] = []

    for seg in segments:
        if cur and seg.end - cur[0].start > target:
            chunks.append(_join(cur))
            cur = _tail(cur, overlap)
        cur.append(seg)

    if cur:
        # don't emit a window that is only overlap carried over from the last one
        if not chunks or cur[-1].end > chunks[-1].end:
            chunks.append(_join(cur))
    return chunks


def _tail(segs: list[Segment], overlap: float) -> list[Segment]:
    out: list[Segment] = []
    total = 0.0
    for s in reversed(segs[1:]):  # never carry the whole window over
        total += s.end - s.start
        if total > overlap:
            break
        out.insert(0, s)
    return out


def _join(segs: list[Segment]) -> Segment:
    return Segment(segs[0].start, segs[-1].end, " ".join(s.text.strip() for s in segs))
