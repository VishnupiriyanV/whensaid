import re
from pathlib import Path

from .util import timecode


def render_txt(segments) -> str:
    # one segment per line with its start time, so `grep -n` output is already a citation
    return "".join(f"[{timecode(s['start'])}] {s['text']}\n" for s in segments)


def render_srt(segments) -> str:
    out = []
    for i, s in enumerate(segments, 1):
        out.append(f"{i}\n{timecode(s['start'], srt=True)} --> {timecode(s['end'], srt=True)}\n{s['text']}\n")
    return "\n".join(out)


RENDERERS = {"txt": render_txt, "srt": render_srt}


def target_path(media_path: str, out_dir: Path | None, fmt: str, kind: str = "file") -> Path:
    if kind == "url":  # a key like youtube:ID, there is no file to sit next to
        return out_dir / "urls" / (re.sub(r"[^\w.-]+", "_", media_path) + "." + fmt)
    if out_dir is None:  # next to the media file, same name
        return Path(media_path).with_suffix("." + fmt)
    # mirror the full path so two files with the same name never collide
    mirrored = media_path.replace(":", "").lstrip("/\\")  # drive letters have no place in a relative path
    return out_dir / (mirrored + "." + fmt)


def export_all(conn, out_dir: Path | None, fmt: str = "txt", overwrite: bool = False) -> tuple[list[Path], list[Path]]:
    """Returns (written, skipped). Next to the recordings (out_dir None) an existing file may
    be someone's real subtitles, so it is left alone unless overwrite is set."""
    render = RENDERERS[fmt]
    written, skipped = [], []
    for src in conn.execute("select id, path, kind from sources where status = 'done' order by path").fetchall():
        if src["kind"] == "url" and out_dir is None:
            continue  # web videos have no recording to write beside
        segs = conn.execute(
            "select start, end, text from segments where source_id = ? order by start", (src["id"],)
        ).fetchall()
        dest = target_path(src["path"], out_dir, fmt, src["kind"])
        text = render(segs)
        if out_dir is None and dest.exists() and not overwrite and dest.read_text(encoding="utf-8", errors="replace") != text:
            skipped.append(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        written.append(dest)
    return written, skipped
