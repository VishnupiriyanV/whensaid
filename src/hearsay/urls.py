"""Videos that live on the web: registering them, and keeping playlists and channels fresh.
The transcripts themselves are fetched by indexer.index_file through a fetcher (see fetch.py)."""
import sys
import time

from .db import forget_source

REFRESH_AFTER = 3600  # seconds between re-listing a playlist or channel, so --watch doesn't hammer the site


def register(conn, entry: dict, feed_url: str | None = None) -> bool:
    """Add a video as pending. False if it is already known (the same video reached by another route)."""
    cur = conn.execute(
        "insert or ignore into sources (path, kind, url, title, feed_url, status) values (?, 'url', ?, ?, ?, 'pending')",
        (entry["key"], entry["url"], entry.get("title"), feed_url),
    )
    return cur.rowcount > 0


def add_url(conn, fetcher, url: str, limit: int = 200, no_playlist: bool = False, log=sys.stderr) -> dict:
    kind, title, entries = fetcher.entries(url, limit, no_playlist)
    feed = url if kind == "playlist" else None
    new = 0
    with conn:
        if feed:
            conn.execute(
                "insert into feeds (url, kind, title, limit_n, checked_at) values (?, ?, ?, ?, ?) "
                "on conflict(url) do update set title = excluded.title, limit_n = excluded.limit_n, checked_at = excluded.checked_at",
                (url, kind, title, limit, time.time()),
            )
        for e in entries:
            new += register(conn, e, feed)
    print(f"{kind} {title or url}: {len(entries)} videos, {new} new", file=log)
    return {"kind": kind, "title": title, "found": len(entries), "new": new}


def refresh_feeds(conn, fetcher, log=sys.stderr, force: bool = False) -> int:
    """List playlists and channels again, add videos that are new. Returns how many."""
    new = 0
    for f in conn.execute("select * from feeds").fetchall():
        if not force and time.time() - (f["checked_at"] or 0) < REFRESH_AFTER:
            continue
        try:
            _, _, entries = fetcher.entries(f["url"], f["limit_n"] or 200)
        except Exception as e:  # offline, blocked, removed: try again next time
            print(f"could not refresh {f['url']}: {e}", file=log)
            continue
        with conn:
            for e in entries:
                new += register(conn, e, f["url"])
            conn.execute("update feeds set checked_at = ? where url = ?", (time.time(), f["url"]))
    if new:
        print(f"{new} new videos from playlists", file=log)
    return new


def pending(conn, retry_failed: bool = True) -> list[str]:
    """Keys of web videos still to do."""
    states = "('pending', 'failed')" if retry_failed else "('pending')"
    return [r["path"] for r in conn.execute(
        f"select path from sources where kind = 'url' and status in {states} order by id")]


def remove(conn, url: str) -> int:
    """Forget a playlist/channel and its videos, or a single video. Returns videos removed."""
    ids = [r["id"] for r in conn.execute("select id from sources where kind = 'url' and (url = ? or feed_url = ?)", (url, url))]
    with conn:
        for sid in ids:
            forget_source(conn, sid)
            conn.execute("delete from sources where id = ?", (sid,))
        conn.execute("delete from feeds where url = ?", (url,))
    return len(ids)
