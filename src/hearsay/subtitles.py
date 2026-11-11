"""Read .srt and .vtt subtitle files into segments, so a recording that already has
captions doesn't need to be transcribed."""
import html
import re
from pathlib import Path

from .chunk import Segment

EXTS = (".srt", ".vtt")
# 1:02:03,500 or 02:03.500 (vtt may leave out the hours, srt uses a comma)
_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE = re.compile(rf"{_TIME}\s*-->\s*{_TIME}")
_TAGS = re.compile(r"<[^>]*>|\{\\[^}]*\}")  # <i>, <c.colorE5E5E5>, <00:00:01.000>, {\an8}


def _seconds(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _drop_overlap(prev: list[str], cur: list[str]) -> list[str]:
    """Auto-captions roll: each cue repeats the line before it and adds a new one. Drop the
    leading lines of cur that are the trailing lines of prev, the longest such overlap."""
    for k in range(min(len(prev), len(cur)), 0, -1):
        if prev[-k:] == cur[:k]:
            return cur[k:]
    return cur


_SOUND = re.compile(r"^[\[(][^\])]{1,40}[\])]$")  # [Music], (applause)


def _cues(text: str):
    """Yield (start, end, text lines) per cue. A cue runs from its timing line to the next
    truly empty line or timing line. Whitespace-only lines do NOT end it: YouTube puts them
    inside cues, and some srt files use them between cues (the index line before the next
    timing is dropped, so both work)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").split("\n")
    n = len(lines)
    for i, line in enumerate(lines):
        m = _CUE.search(line)
        if not m:
            continue
        body = []
        j = i + 1
        while j < n and lines[j] != "" and not _CUE.search(lines[j]):
            body.append(lines[j])
            j += 1
        if j < n and _CUE.search(lines[j]):  # ran into the next cue: its index line is ours to drop
            while body and (not body[-1].strip() or body[-1].strip().isdigit()):
                if body.pop().strip().isdigit():
                    break
        yield _seconds(*m.group(1, 2, 3, 4)), _seconds(*m.group(5, 6, 7, 8)), body


def parse(text: str) -> list[Segment]:
    """Works for srt and vtt. Strips markup, sound labels like [Music], the vtt header and
    NOTE/STYLE blocks, removes the repeated lines of rolling auto-captions, and merges a cue
    that only shows the same text again."""
    out: list[Segment] = []
    prev_lines: list[str] = []
    for start, end, body in _cues(text):
        cur = [re.sub(r"\s+", " ", html.unescape(_TAGS.sub("", ln))).strip() for ln in body]
        cur = [ln for ln in cur if ln and not _SOUND.match(ln)]
        if end < start or not cur:
            continue
        new = _drop_overlap(prev_lines, cur)
        prev_lines = cur
        if not new:  # nothing but lines we already have: just keep the earlier cue on screen
            if out:
                out[-1].end = max(out[-1].end, end)
            continue
        text_ = " ".join(new)
        if out and out[-1].text == text_:
            out[-1].end = max(out[-1].end, end)
            continue
        out.append(Segment(start, end, text_))
    out.sort(key=lambda s: s.start)
    return out


def find_sidecar(media_path: str) -> Path | None:
    """A subtitle file next to the recording: talk.srt, talk.vtt, or talk.en.srt style."""
    p = Path(media_path)
    for ext in EXTS:
        for stem in (p.stem, p.stem + ".en", p.stem + ".en-US", p.stem + ".en-GB"):
            cand = p.with_name(stem + ext)
            if cand.is_file():
                return cand
    return None


def read(path: Path) -> list[Segment]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return parse(raw.decode(enc))
        except UnicodeError:
            continue
    return []
