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


def target_path(media_path: str, out_dir: Path | None, fmt: str) -> Path:
    if out_dir is None:  # next to the media file, same name
        return Path(media_path).with_suffix("." + fmt)
    # mirror the full path so two files with the same name never collide
    return out_dir / (media_path.lstrip("/") + "." + fmt)


def export_all(conn, out_dir: Path | None, fmt: str = "txt", overwrite: bool = False) -> tuple[list[Path], list[Path]]:
    """Returns (written, skipped). Next to the recordings (out_dir None) an existing file may
    be someone's real subtitles, so it is left alone unless overwrite is set."""
    render = RENDERERS[fmt]
    written, skipped = [], []
    for src in conn.execute("select id, path from sources where status = 'done' order by path").fetchall():
        segs = conn.execute(
            "select start, end, text from segments where source_id = ? order by start", (src["id"],)
        ).fetchall()
        dest = target_path(src["path"], out_dir, fmt)
        text = render(segs)
        if out_dir is None and dest.exists() and not overwrite and dest.read_text(encoding="utf-8", errors="replace") != text:
            skipped.append(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        written.append(dest)
    return written, skipped
