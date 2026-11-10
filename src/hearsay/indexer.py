import fcntl
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .chunk import Segment, make_chunks
from .db import forget_chunks, forget_source
from .paths import home
from .transcribe import NoAudio

MEDIA_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac",
}


class AlreadyRunning(Exception):
    pass


@contextmanager
def index_lock():
    # only one indexer at a time, mcp servers just read.
    # windows: swap fcntl for msvcrt.locking when we get there
    f = open(home() / "index.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise AlreadyRunning()
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def indexer_running() -> bool:
    """True if some process holds the index lock. Takes a shared lock for an
    instant, which only fails while the indexer has the exclusive one."""
    path = home() / "index.lock"
    if not path.exists():
        return False
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def find_media(root: str):
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in sorted(files):
            if Path(name).suffix.lower() in MEDIA_EXTS and not name.startswith("."):
                yield os.path.join(dirpath, name)


def _needs_work(row, st, retry_failed=True) -> bool:
    if row is None or row["status"] == "pending":
        return True
    if row["status"] == "failed" and retry_failed:
        return True
    return row["size"] != st.st_size or row["mtime_ns"] != st.st_mtime_ns


def scan(conn, root: str, settle: float = 0, retry_failed: bool = True) -> list[str]:
    """Register new or changed files under root, return the paths to process.
    settle: skip files modified in the last this-many seconds, they may still be copying."""
    todo = []
    now = time.time()
    for path in find_media(root):
        st = os.stat(path)
        if settle and now - st.st_mtime < settle:
            continue
        row = conn.execute("select * from sources where path = ?", (path,)).fetchone()
        if not _needs_work(row, st, retry_failed):
            continue
        conn.execute(
            """insert into sources (path, size, mtime_ns, status) values (?, ?, ?, 'pending')
               on conflict(path) do update set size = excluded.size,
               mtime_ns = excluded.mtime_ns, status = 'pending', error = null""",
            (path, st.st_size, st.st_mtime_ns),
        )
        todo.append(path)
    conn.commit()
    return todo


def index_file(conn, path: str, transcriber, embedder) -> str:
    row = conn.execute("select id from sources where path = ?", (path,)).fetchone()
    source_id = row["id"]
    try:
        segments, duration = transcriber.transcribe(path)
    except NoAudio:
        conn.execute("update sources set status = 'no_audio' where id = ?", (source_id,))
        conn.commit()
        return "no_audio"
    except Exception as e:
        conn.execute(
            "update sources set status = 'failed', error = ? where id = ?", (str(e), source_id)
        )
        conn.commit()
        return "failed"

    chunks = make_chunks(segments)
    vecs = embedder.encode([c.text for c in chunks]) if chunks else []

    # one transaction so a crash never leaves a half indexed file
    with conn:
        forget_source(conn, source_id)
        conn.executemany(
            "insert into segments (source_id, start, end, text) values (?, ?, ?, ?)",
            [(source_id, s.start, s.end, s.text) for s in segments],
        )
        _insert_chunks(conn, source_id, chunks, vecs)
        conn.execute(
            "update sources set status = 'done', duration = ?, error = null, indexed_at = datetime('now') where id = ?",
            (duration, source_id),
        )
    return "done"


def _insert_chunks(conn, source_id, chunks, vecs):
    for c, v in zip(chunks, vecs):
        cur = conn.execute(
            "insert into chunks (source_id, start, end, text, embedding) values (?, ?, ?, ?, ?)",
            (source_id, c.start, c.end, c.text, v.tobytes()),
        )
        conn.execute("insert into chunks_fts (rowid, text) values (?, ?)", (cur.lastrowid, c.text))


def rechunk(conn, embedder, log=sys.stderr) -> int:
    """Rebuild chunks and embeddings from the stored segments, no transcribing.
    Use after changing the chunking or the embedding model."""
    ids = [r["id"] for r in conn.execute("select id from sources where status = 'done'")]
    for source_id in ids:
        rows = conn.execute(
            "select start, end, text from segments where source_id = ? order by start", (source_id,)
        ).fetchall()
        chunks = make_chunks([Segment(r["start"], r["end"], r["text"]) for r in rows])
        vecs = embedder.encode([c.text for c in chunks]) if chunks else []
        with conn:
            forget_chunks(conn, source_id)
            _insert_chunks(conn, source_id, chunks, vecs)
    print(f"rechunked {len(ids)} files", file=log)
    return len(ids)


def run(conn, transcriber, embedder, log=sys.stderr, settle: float = 0, retry_failed: bool = True) -> dict:
    roots = [r["path"] for r in conn.execute("select path from roots order by path")]
    todo = []
    for root in roots:
        todo += scan(conn, root, settle, retry_failed)

    counts = {"done": 0, "no_audio": 0, "failed": 0}
    for i, path in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {path}", file=log, flush=True)
        result = index_file(conn, path, transcriber, embedder)
        counts[result] += 1
        print(f"    {result} in {time.time() - t0:.1f}s", file=log, flush=True)
    return counts


def watch(conn, transcriber, embedder, log=sys.stderr, interval: float = 30, settle: float = 15,
          rounds: int | None = None, sleep=time.sleep) -> dict:
    """Keep indexing as files show up. A failed file is retried only once it changes, so a
    broken one doesn't loop forever. rounds and sleep exist so tests can run it."""
    total = {"done": 0, "no_audio": 0, "failed": 0}
    n = 0
    print(f"watching every {interval:.0f}s, ctrl-c to stop", file=log, flush=True)
    while rounds is None or n < rounds:
        # the first pass still retries earlier failures, after that only changed files
        counts = run(conn, transcriber, embedder, log, settle=settle, retry_failed=(n == 0))
        for k, v in counts.items():
            total[k] += v
        n += 1
        if rounds is None or n < rounds:
            sleep(interval)
    return total
