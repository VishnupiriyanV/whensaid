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
    from .indexer import AlreadyRunning, index_lock, run, watch
    from .transcribe import Transcriber

    conn = db.connect(paths.db_path())
    try:
        with index_lock():
            if args.watch:
                try:
                    watch(conn, Transcriber(), Embedder(), interval=args.interval)
                except KeyboardInterrupt:
                    print("stopped")
                return
            counts = run(conn, Transcriber(), Embedder())
    except AlreadyRunning:
        sys.exit("another hearsay index is already running")
    print(counts)


def cmd_remove(args):
    from .indexer import AlreadyRunning, index_lock, remove_root

    conn = db.connect(paths.db_path())
    try:
        with index_lock():
            n = remove_root(conn, args.folder)
    except AlreadyRunning:
        sys.exit("another hearsay index is already running")
    print(f"removed {n} files from the index (the recordings themselves are untouched)")


def cmd_rechunk(args):
    from .embed import Embedder
    from .indexer import AlreadyRunning, index_lock, rechunk

    conn = db.connect(paths.db_path())
    try:
        with index_lock():
            rechunk(conn, Embedder())
    except AlreadyRunning:
        sys.exit("another hearsay index is already running")


def cmd_export(args):
    from .export import export_all

    conn = db.connect(paths.db_path())
    out = None if args.beside else (Path(args.out).expanduser() if args.out else paths.home() / "transcripts")
    files = export_all(conn, out, args.format)
    where = "next to each recording" if out is None else str(out)
    print(f"wrote {len(files)} {args.format} files, {where}")


def cmd_search(args):
    import json

    from .embed import Embedder
    from .rerank import load_reranker, min_relevance
    from .search import Searcher
    from .util import timecode

    try:
        conn = db.connect_readonly(paths.db_path())
    except FileNotFoundError:
        sys.exit("nothing indexed yet: hearsay add <folder>, then hearsay index")
    hits = Searcher(conn, Embedder(), reranker=load_reranker(), min_relevance=min_relevance()).search(args.query, limit=args.limit)
    if args.json:
        print(json.dumps([
            {"file": os.path.basename(h.path), "path": h.path, "start": round(h.start, 1),
             "at": timecode(h.start), "similarity": h.similarity and round(h.similarity, 3), "text": h.text}
            for h in hits
        ], indent=1))
        return
    if not hits:
        print("nothing relevant found")
    for h in hits:
        sim = "  -  " if h.similarity is None else f"{h.similarity:.2f}"
        print(f"{os.path.basename(h.path)}  {timecode(h.start)}  sim {sim}  {h.text[:200]}")


def cmd_serve(args):
    from .server import main as serve

    serve()


def cmd_status(args):
    conn = db.connect(paths.db_path())
    rows = conn.execute("select status, count(*) n, coalesce(sum(duration), 0) d from sources group by status")
    for r in rows:
        print(f"{r['status']:10} {r['n']:5} files  {r['d'] / 3600:.2f} h")
    n = conn.execute("select count(*) from chunks").fetchone()[0]
    print(f"{n} chunks")


def cmd_models_pull(args):
    from .embed import Embedder
    from .rerank import Reranker
    from .transcribe import Transcriber

    t = Transcriber(allow_download=True)
    print(f"whisper: {t.model_name} ({t.device})")
    t._load()
    e = Embedder(allow_download=True)
    print(f"embedder: {e.model_name}")
    e.encode(["warm up"])
    r = Reranker(allow_download=True)
    print(f"reranker: {r.model_name}")
    r.score("warm up", ["warm up"])
    print("ok")


def main(argv=None):
    p = argparse.ArgumentParser(prog="hearsay")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="register a folder of recordings")
    a.add_argument("folder")
    a.set_defaults(fn=cmd_add)

    rm = sub.add_parser("remove", help="stop watching a folder and drop its files from the index")
    rm.add_argument("folder")
    rm.set_defaults(fn=cmd_remove)
    ix = sub.add_parser("index", help="transcribe and index anything new or changed")
    ix.add_argument("--watch", action="store_true", help="keep running and pick up new files as they appear")
    ix.add_argument("--interval", type=float, default=30, help="seconds between scans with --watch")
    ix.set_defaults(fn=cmd_index)
    sub.add_parser("rechunk", help="rebuild chunks and embeddings from stored transcripts").set_defaults(fn=cmd_rechunk)
    e = sub.add_parser("export", help="write transcripts as text files an agent can grep")
    e.add_argument("--format", choices=["txt", "srt"], default="txt")
    e.add_argument("--out", help="folder to write into (default: ~/.hearsay/transcripts)")
    e.add_argument("--beside", action="store_true", help="write next to each recording instead")
    e.set_defaults(fn=cmd_export)

    q = sub.add_parser("search", help="search from the terminal")
    q.add_argument("query")
    q.add_argument("-n", "--limit", type=int, default=8)
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_search)

    sub.add_parser("serve", help="run the MCP server on stdio").set_defaults(fn=cmd_serve)
    sub.add_parser("status", help="show what is indexed").set_defaults(fn=cmd_status)

    m = sub.add_parser("models", help="model management")
    msub = m.add_subparsers(dest="mcmd", required=True)
    msub.add_parser("pull", help="download the models").set_defaults(fn=cmd_models_pull)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
