import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import subtitles, urls
from .chunk import Segment, make_chunks
from .db import forget_chunks, forget_source, get_meta, set_meta
from .paths import home
from .fetch import Skip
from .transcribe import NoAudio, Unintelligible

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

MEDIA_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac",
}


class AlreadyRunning(Exception):
    pass


class ModelChanged(Exception):
    pass


def claim_embed_model(conn, embedder) -> None:
    """Vectors from two embedding models can't be compared, so remember which one built
    the index and refuse to add to it with another."""
    name = getattr(embedder, "model_name", None)
    if not name:
        return
    have = get_meta(conn, "embed_model")
    if have is None:
        set_meta(conn, "embed_model", name)
        conn.commit()
    elif have != name:
        raise ModelChanged(
            f"this index was embedded with {have}, but the embedder is now {name}. "
            "Run `whensaid rechunk` to re-embed everything with the new model."
        )


def _try_lock(f, shared: bool = False) -> bool:
    """Take the lock without waiting. False if another process holds it."""
    if sys.platform == "win32":
        # no shared locks here; locking one byte is enough to tell who is running
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    try:
        fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(f):
    if sys.platform == "win32":
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def index_lock():
    # only one indexer at a time, mcp servers just read.
    f = open(home() / "index.lock", "a+")
    if not _try_lock(f):
        f.close()
        raise AlreadyRunning()
    try:
        yield
    finally:
        _unlock(f)
        f.close()


def indexer_running() -> bool:
    """True if some process holds the index lock. Takes a shared lock for an
    instant, which only fails while the indexer has the exclusive one."""
    path = home() / "index.lock"
    if not path.exists():
        return False
    with open(path, "a+") as f:
        if not _try_lock(f, shared=True):
            return True
        _unlock(f)
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
        try:
            st = os.stat(path)
        except OSError:
            continue  # dangling symlink, or the file vanished since the walk
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


def _root_of(roots, path):
    for r in roots:
        if path.startswith(r.rstrip(os.sep) + os.sep):
            return r
    return None


def prune(conn, log=sys.stderr) -> int:
    """Drop index rows for files that no longer exist, returns how many.
    Careful with unplugged drives: a root that is missing, or that exists but holds no media at
    all (an empty mount point), is left alone, since wiping a library over that would be bad."""
    roots = [r["path"] for r in conn.execute("select path from roots")]
    gone: dict[str, list[int]] = {}
    for row in conn.execute("select id, path from sources").fetchall():
        root = _root_of(roots, row["path"])
        if root is not None and not os.path.exists(row["path"]):
            gone.setdefault(root, []).append(row["id"])
    removed = 0
    for root, ids in gone.items():
        if not os.path.isdir(root):
            print(f"skipping {len(ids)} missing files: {root} is not there (unplugged?)", file=log)
            continue
        if next(find_media(root), None) is None:
            print(f"skipping {len(ids)} missing files: {root} has no media at all (unmounted?)", file=log)
            continue
        with conn:
            for sid in ids:
                forget_source(conn, sid)
                conn.execute("delete from sources where id = ?", (sid,))
        removed += len(ids)
    if removed:
        print(f"removed {removed} files that no longer exist", file=log)
    return removed


def remove_root(conn, folder: str) -> int:
    """Stop watching a folder and forget everything indexed from it, returns files removed."""
    folder = os.path.abspath(os.path.expanduser(folder)).rstrip(os.sep)
    prefix = folder + os.sep
    ids = [r["id"] for r in conn.execute("select id, path from sources") if r["path"].startswith(prefix)]
    with conn:
        for sid in ids:
            forget_source(conn, sid)
            conn.execute("delete from sources where id = ?", (sid,))
        conn.execute("delete from roots where path = ?", (folder,))
    return len(ids)


def _from_subtitles(path: str):
    """(segments, duration) from a subtitle file next to the recording, or None if there is
    none or it holds no text. The duration is the media's when it can be read quickly."""
    side = subtitles.find_sidecar(path)
    segs = subtitles.read(side) if side else []
    if not segs:
        return None
    duration = segs[-1].end
    try:
        import av

        with av.open(path) as c:
            if c.duration:
                duration = max(duration, c.duration / 1_000_000)
    except Exception:
        pass
    return segs, duration


def index_file(conn, path: str, transcriber, embedder, use_subtitles: bool = False, fetcher=None) -> str:
    row = conn.execute("select id, kind, url from sources where path = ?", (path,)).fetchone()
    source_id = row["id"]
    how = "whisper"
    try:
        if row["kind"] == "url":
            if fetcher is None:
                raise RuntimeError("no fetcher available for a web video")
            segments, duration, how = fetcher.acquire(row["url"], transcriber)
        else:
            found = _from_subtitles(path) if use_subtitles else None
            if found:
                how = "subtitle-file"
            segments, duration = found if found else transcriber.transcribe(path)
    except NoAudio:
        conn.execute("update sources set status = 'no_audio' where id = ?", (source_id,))
        conn.commit()
        return "no_audio"
    except (Skip, Unintelligible) as e:
        # will not work on a retry either, so it is not asked for again (see urls.retry_skipped)
        conn.execute("update sources set status = 'skipped', error = ? where id = ?", (str(e), source_id))
        conn.commit()
        return "skipped"
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
            "update sources set status = 'done', duration = ?, error = null, captions = ?, confidence = ?, "
            "indexed_at = datetime('now') where id = ?",
            (duration, how, getattr(transcriber, "last_confidence", None) if how == "whisper" else None, source_id),
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
    name = getattr(embedder, "model_name", None)
    if name:
        set_meta(conn, "embed_model", name)
        conn.commit()
    print(f"rechunked {len(ids)} files", file=log)
    return len(ids)


def _progress(log):
    """A callback for the transcriber: one line per 10% of a long file, so a three hour film
    is not silent for ten minutes. Short files finish before anyone wonders."""
    from .util import timecode

    state = {"step": -1}

    def cb(done, total):
        if total < 300:
            return
        step = int(min(done / total, 1.0) * 10)
        if step > state["step"]:
            state["step"] = step
            print(f"    {step * 10:3d}%  {timecode(done)} of {timecode(total)}", file=log, flush=True)

    cb.reset = lambda: state.update(step=-1)
    return cb


WEB_PAUSE = 3.0  # seconds between two web videos; each one costs a few yt-dlp requests


def run(conn, transcriber, embedder, log=sys.stderr, settle: float = 0, retry_failed: bool = True,
        use_subtitles: bool = False, fetcher=None) -> dict:
    claim_embed_model(conn, embedder)
    prune(conn, log)
    roots = [r["path"] for r in conn.execute("select path from roots order by path")]
    todo = []
    for root in roots:
        todo += scan(conn, root, settle, retry_failed)
    if fetcher is not None:
        urls.refresh_feeds(conn, fetcher, log)
        todo += urls.pending(conn, retry_failed)

    counts = {"done": 0, "no_audio": 0, "failed": 0, "skipped": 0}
    last_was_web = False
    web = set(urls.pending(conn)) if fetcher is not None else set()
    progress = _progress(log)
    if hasattr(transcriber, "on_progress"):
        transcriber.on_progress = progress
    for i, path in enumerate(todo, 1):
        progress.reset()
        if path in web and last_was_web:
            time.sleep(WEB_PAUSE)
        last_was_web = path in web
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {path}", file=log, flush=True)
        result = index_file(conn, path, transcriber, embedder, use_subtitles, fetcher)
        counts[result] += 1
        print(f"    {result} in {time.time() - t0:.1f}s", file=log, flush=True)
    return counts


def watch(conn, transcriber, embedder, log=sys.stderr, interval: float = 30, settle: float = 15,
          rounds: int | None = None, sleep=time.sleep, use_subtitles: bool = False, fetcher=None) -> dict:
    """Keep indexing as files show up. A failed file is retried only once it changes, so a
    broken one doesn't loop forever. rounds and sleep exist so tests can run it."""
    total = {"done": 0, "no_audio": 0, "failed": 0, "skipped": 0}
    n = 0
    print(f"watching every {interval:.0f}s, ctrl-c to stop", file=log, flush=True)
    while rounds is None or n < rounds:
        # the first pass still retries earlier failures, after that only changed files
        counts = run(conn, transcriber, embedder, log, settle=settle, retry_failed=(n == 0),
                     use_subtitles=use_subtitles, fetcher=fetcher)
        for k, v in counts.items():
            total[k] += v
        n += 1
        if rounds is None or n < rounds:
            sleep(interval)
    return total
