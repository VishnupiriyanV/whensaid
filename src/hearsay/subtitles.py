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


def parse(text: str) -> list[Segment]:
    """Works for srt and vtt. Skips the vtt header, NOTE/STYLE blocks and cue settings,
    strips markup, and drops a cue whose text just repeats the previous one (auto-captions
    show every line twice while it scrolls)."""
    out: list[Segment] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")):
        lines = block.strip("\n").split("\n")
        i = next((k for k, ln in enumerate(lines) if _CUE.search(ln)), None)
        if i is None:  # header, NOTE, STYLE, or junk
            continue
        m = _CUE.search(lines[i])
        start = _seconds(*m.group(1, 2, 3, 4))
        end = _seconds(*m.group(5, 6, 7, 8))
        body = " ".join(
            html.unescape(_TAGS.sub("", ln)).strip() for ln in lines[i + 1 :]
        )
        body = re.sub(r"\s+", " ", body).strip()
        if not body or end < start:
            continue
        if out and out[-1].text == body:
            out[-1].end = max(out[-1].end, end)  # same line shown again, extend it
            continue
        out.append(Segment(start, end, body))
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
