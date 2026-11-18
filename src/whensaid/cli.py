import argparse
import os
import sys
from pathlib import Path

from . import db, paths


def cmd_add(args):
    folder = os.path.abspath(os.path.expanduser(args.folder))
    if not os.path.isdir(folder):
        sys.exit(f"not a folder: {folder}")
    conn = db.connect(paths.db_path())
    conn.execute("insert or ignore into roots (path) values (?)", (folder,))
    conn.commit()
    print(f"added {folder}")


def cmd_index(args):
    from .embed import Embedder
    from .fetch import YtDlp
    from .indexer import AlreadyRunning, ModelChanged, index_lock, run, watch
    from .transcribe import Transcriber

    from . import models

    try:
        models.ensure(models.INDEX)
    except models.ModelsMissing as e:
        sys.exit(str(e))
    conn = db.connect(paths.db_path())
    try:
        with index_lock():
            if args.retry_skipped:
                from .urls import retry_skipped

                print(f"{retry_skipped(conn)} skipped items set to try again")
            if args.watch:
                try:
                    watch(conn, Transcriber(), Embedder(), interval=args.interval, use_subtitles=args.subtitles,
                          fetcher=YtDlp())
                except KeyboardInterrupt:
                    print("stopped")
                return
            counts = run(conn, Transcriber(), Embedder(), use_subtitles=args.subtitles, fetcher=YtDlp())
    except AlreadyRunning:
        sys.exit("another whensaid index is already running")
    except ModelChanged as e:
        sys.exit(str(e))
    print(counts)


def cmd_add_url(args):
    from .fetch import FetchError, YtDlp
    from .urls import add_url

    conn = db.connect(paths.db_path())
    try:
        r = add_url(conn, YtDlp(), args.url, limit=args.limit, no_playlist=args.no_playlist)
    except FetchError as e:
        sys.exit(f"could not read {args.url}: {e}")
    print(f"run `whensaid index` to fetch {r['new']} new videos" if r["new"] else "nothing new to fetch")


def cmd_remove(args):
    from .indexer import AlreadyRunning, index_lock, remove_root
    from .urls import remove as remove_url

    conn = db.connect(paths.db_path())
    try:
        with index_lock():
            n = remove_url(conn, args.folder) if args.folder.startswith(("http://", "https://")) else remove_root(conn, args.folder)
    except AlreadyRunning:
        sys.exit("another whensaid index is already running")
    print(f"removed {n} files from the index (the recordings themselves are untouched)")


def cmd_rechunk(args):
    from .embed import Embedder
    from .indexer import AlreadyRunning, index_lock, rechunk

    conn = db.connect(paths.db_path())
    try:
        with index_lock():
            rechunk(conn, Embedder())
    except AlreadyRunning:
        sys.exit("another whensaid index is already running")


def cmd_export(args):
    from .export import export_all

    conn = db.connect(paths.db_path())
    out = None if args.beside else (Path(args.out).expanduser() if args.out else paths.home() / "transcripts")
    files, skipped = export_all(conn, out, args.format, overwrite=args.overwrite)
    where = "next to each recording" if out is None else str(out)
    print(f"wrote {len(files)} {args.format} files, {where}")
    if skipped:
        print(f"left {len(skipped)} existing files alone (pass --overwrite to replace them), e.g. {skipped[0]}")


def cmd_search(args):
    import json

    from .embed import Embedder
    from .rerank import load_reranker, min_relevance, strength
    from .search import Searcher
    from .util import deep_link, timecode

    from . import models

    try:
        conn = db.connect_readonly(paths.db_path())
    except FileNotFoundError:
        sys.exit("nothing indexed yet: whensaid add <folder>, then whensaid index")
    try:
        models.ensure(models.wanted_for_search())
    except models.ModelsMissing as e:
        sys.exit(str(e))
    try:
        hits = Searcher(conn, Embedder(), reranker=load_reranker(), min_relevance=min_relevance()).search(args.query, limit=args.limit)
    except RuntimeError as e:
        sys.exit(str(e))
    if args.json:
        print(json.dumps([
            {"file": h.title or os.path.basename(h.path), "path": h.path, "url": h.url, "link": deep_link(h.url, h.start),
             "start": round(h.start, 1),
             "at": timecode(h.start), "similarity": h.similarity and round(h.similarity, 3),
             "relevance": h.relevance and round(h.relevance, 1), "match": strength(h.relevance), "text": h.text}
            for h in hits
        ], indent=1))
        return
    if not hits:
        print("nothing relevant found")
    elif strength(hits[0].relevance) == "weak":
        print("only weak matches, the recordings may not cover this:")
    for h in hits:
        mark = {"weak": "weak  ", "strong": "      "}.get(strength(h.relevance), "      ")
        where = deep_link(h.url, h.start) or timecode(h.start)
        print(f"{mark}{h.title or os.path.basename(h.path)}  {where}  {h.text[:200]}")


def cmd_serve(args):
    from .server import main as serve

    serve()


def cmd_doctor(args):
    from .doctor import run

    sys.exit(run())


def cmd_status(args):
    conn = db.connect(paths.db_path())
    rows = conn.execute(
        "select status, sum(kind = 'file') files, sum(kind = 'url') web, coalesce(sum(duration), 0) d "
        "from sources where status != 'removed' group by status"
    ).fetchall()
    for r in rows:
        print(f"{r['status']:10} {r['files']:5} files {r['web']:5} web videos  {r['d'] / 3600:.2f} h")
    n = conn.execute("select count(*) from chunks").fetchone()[0]
    print(f"{n} chunks")
    feeds = conn.execute("select url, title from feeds").fetchall()
    if feeds:
        print(f"following {len(feeds)} playlists or channels:")
        for f in feeds:
            print(f"  {f['title'] or ''}  {f['url']}")
    for status, what in (("failed", "failed, will be tried again"), ("skipped", "skipped for good (whensaid index --retry-skipped to try anyway)")):
        why = conn.execute(
            "select coalesce(error, '?') e, count(*) n, min(coalesce(title, path)) eg from sources where status = ? group by e order by n desc limit 5",
            (status,),
        ).fetchall()
        if why:
            print(f"{what}:")
            for w in why:
                print(f"  {w['n']:4}  {w['e'][:90]}  (e.g. {os.path.basename(w['eg'])[:50]})")


def cmd_models_pull(args):
    from . import models

    todo = [k for k in ("whisper", "embedder", "reranker") if not models.present(k)]
    if not todo:
        print("all models are already on disk")
        return
    try:
        models.download(todo)
    except Exception as e:
        sys.exit(f"download failed: {e}")
    print("ok")


def main(argv=None):
    p = argparse.ArgumentParser(prog="whensaid")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="register a folder of recordings")
    a.add_argument("folder")
    a.set_defaults(fn=cmd_add)

    au = sub.add_parser("add-url", help="add a video, playlist or channel from the web (needs yt-dlp)")
    au.add_argument("url")
    au.add_argument("--limit", type=int, default=200, help="most videos to take from a playlist or channel")
    au.add_argument("--no-playlist", action="store_true", help="for a video url that also names a playlist: just the video")
    au.set_defaults(fn=cmd_add_url)

    rm = sub.add_parser("remove", help="stop watching a folder, playlist or video and drop it from the index")
    rm.add_argument("folder", metavar="folder-or-url")
    rm.set_defaults(fn=cmd_remove)
    ix = sub.add_parser("index", help="transcribe and index anything new or changed")
    ix.add_argument("--watch", action="store_true", help="keep running and pick up new files as they appear")
    ix.add_argument("--interval", type=float, default=30, help="seconds between scans with --watch")
    ix.add_argument("--subtitles", action="store_true",
                    help="use a .srt/.vtt next to a recording instead of transcribing it (English assumed)")
    ix.add_argument("--retry-skipped", action="store_true", help="try again what was skipped for good (live then, private then, ...)")
    ix.set_defaults(fn=cmd_index)
    sub.add_parser("rechunk", help="rebuild chunks and embeddings from stored transcripts").set_defaults(fn=cmd_rechunk)
    e = sub.add_parser("export", help="write transcripts as text files an agent can grep")
    e.add_argument("--format", choices=["txt", "srt"], default="txt")
    e.add_argument("--out", help="folder to write into (default: ~/.whensaid/transcripts)")
    e.add_argument("--beside", action="store_true", help="write next to each recording instead")
    e.add_argument("--overwrite", action="store_true", help="with --beside: replace files that are already there")
    e.set_defaults(fn=cmd_export)

    q = sub.add_parser("search", help="search from the terminal")
    q.add_argument("query")
    q.add_argument("-n", "--limit", type=int, default=8)
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_search)

    sub.add_parser("serve", help="run the MCP server on stdio").set_defaults(fn=cmd_serve)
    sub.add_parser("status", help="show what is indexed").set_defaults(fn=cmd_status)
    sub.add_parser("doctor", help="check models, GPU, yt-dlp and the index, and say how to fix what is wrong").set_defaults(fn=cmd_doctor)

    m = sub.add_parser("models", help="model management")
    msub = m.add_subparsers(dest="mcmd", required=True)
    msub.add_parser("pull", help="download the models").set_defaults(fn=cmd_models_pull)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
