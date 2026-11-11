import sqlite3
from pathlib import Path

SCHEMA = """
create table if not exists roots (
    path text primary key
);

create table if not exists sources (
    id integer primary key,
    path text unique not null,
    size integer,
    mtime_ns integer,
    duration real,
    status text not null default 'pending',  -- pending, done, no_audio, failed
    error text,
    indexed_at text
);

create table if not exists segments (
    id integer primary key,
    source_id integer not null references sources(id) on delete cascade,
    start real not null,
    end real not null,
    text text not null
);
create index if not exists segments_source on segments(source_id, start);

create table if not exists chunks (
    id integer primary key,
    source_id integer not null references sources(id) on delete cascade,
    start real not null,
    end real not null,
    text text not null,
    embedding blob
);
create index if not exists chunks_source on chunks(source_id, start);

create table if not exists meta (
    key text primary key,
    value text
);

create virtual table if not exists chunks_fts using fts5(text, tokenize='porter unicode61');
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode = wal")
    conn.execute("pragma foreign_keys = on")
    conn.executescript(SCHEMA)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    try:
        row = conn.execute("select value from meta where key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:  # index from before the meta table, opened read-only
        return None
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "insert into meta (key, value) values (?, ?) on conflict(key) do update set value = excluded.value",
        (key, value),
    )


def forget_chunks(conn: sqlite3.Connection, source_id: int) -> None:
    # fts rows are not tied to chunks by a foreign key, so clear them by hand
    conn.execute(
        "delete from chunks_fts where rowid in (select id from chunks where source_id = ?)",
        (source_id,),
    )
    conn.execute("delete from chunks where source_id = ?", (source_id,))


def forget_source(conn: sqlite3.Connection, source_id: int) -> None:
    forget_chunks(conn, source_id)
    conn.execute("delete from segments where source_id = ?", (source_id,))


def connect_readonly(path: Path) -> sqlite3.Connection:
    """For the mcp server: never writes, never creates the file. The connection is
    shared between the sdk's worker threads, so the caller has to serialise use."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
