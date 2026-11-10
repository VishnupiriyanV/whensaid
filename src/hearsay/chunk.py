from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    text: str


def make_chunks(
    segments: list[Segment], target: float = 45.0, overlap: float = 6.0, max_gap: float = 8.0
) -> list[Segment]:
    """Group whisper segments into windows of roughly `target` seconds.

    A new window starts with the last few segments of the previous one (anything
    that started within `overlap` seconds of its end) so a sentence on the
    boundary shows up in both. A silence longer than `max_gap` (music, a title
    card) ends the window outright, with no overlap across it.
    """
    chunks: list[Segment] = []
    cur: list[Segment] = []

    for seg in segments:
        if cur and seg.start - cur[-1].end > max_gap:
            chunks.append(_join(cur))
            cur = []
        elif cur and seg.end - cur[0].start > target:
            chunks.append(_join(cur))
            cur = _tail(cur, overlap)
            if cur and seg.end - cur[0].start > target:
                cur = []  # the carried tail would blow the window right away
        cur.append(seg)

    if cur and (not chunks or cur[-1].end > chunks[-1].end):
        chunks.append(_join(cur))
    return chunks


def _tail(segs: list[Segment], overlap: float) -> list[Segment]:
    cutoff = segs[-1].end - overlap
    return [s for s in segs[1:] if s.start >= cutoff]  # never carry the whole window


def _join(segs: list[Segment]) -> Segment:
    return Segment(segs[0].start, segs[-1].end, " ".join(s.text.strip() for s in segs))
