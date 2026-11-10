import os
import sys
import threading
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import db, paths
from .indexer import indexer_running
from .search import Searcher
from .util import timecode  # noqa: F401 (also used by tests via server.timecode)

INSTRUCTIONS = """\
Search what was said in the user's local video and audio recordings.
Start with `search`: it returns the matching passage, the file, and where in the file it was
said (seconds and h:mm:ss). Use `get_transcript` to read more around a hit. All times are in
seconds from the start of the file. `similarity` is cosine similarity to the query: roughly
below 0.55 the passage is probably unrelated. If `search` says nothing is indexed, tell the
user to run `hearsay add <folder>` and `hearsay index`.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

mcp = MCPServer("hearsay", instructions=INSTRUCTIONS)

_lock = threading.Lock()  # sync tools run on worker threads and share one connection
_state: dict = {}


def _conn():
    if "conn" not in _state:
        try:
            _state["conn"] = db.connect_readonly(paths.db_path())
        except FileNotFoundError:
            raise ToolError("Nothing is indexed yet. Run `hearsay add <folder>` then `hearsay index`.")
    return _state["conn"]


def _searcher() -> Searcher:
    if "searcher" not in _state:
        from .embed import Embedder

        _state["searcher"] = Searcher(_conn(), Embedder())
    return _state["searcher"]


def _resolve_source(conn, file: str):
    """Accepts a full path or a file name. Only files that are in the index can be read."""
    rows = conn.execute("select * from sources where path = ?", (file,)).fetchall()
    if not rows:
        rows = conn.execute("select * from sources where path like ?", (f"%{os.sep}{file}",)).fetchall()
    if not rows:
        raise ToolError(f"{file!r} is not in the index. Use list_sources to see what is.")
    if len(rows) > 1:
        raise ToolError(f"{file!r} matches several files, pass the full path: " + ", ".join(r["path"] for r in rows[:5]))
    return rows[0]


@mcp.tool(annotations=READ_ONLY)
def search(
    query: Annotated[str, Field(description="What to look for, in plain words. A question or topic works.")],
    limit: Annotated[int, Field(ge=1, le=25, description="Max passages to return.")] = 8,
) -> dict:
    """Find passages in the indexed recordings that match a query, best first.
    Mixes keyword and meaning-based matching, so paraphrases work."""
    with _lock:
        if not query.strip():
            raise ToolError("query is empty")
        try:
            hits = _searcher().search(query, limit=limit)
        except RuntimeError as e:
            raise ToolError(str(e))
        total = _conn().execute("select count(*) from chunks").fetchone()[0]
    if total == 0:
        raise ToolError("Nothing is indexed yet. Run `hearsay add <folder>` then `hearsay index`.")
    return {
        "query": query,
        "results": [
            {
                "file": os.path.basename(h.path),
                "path": h.path,
                "start": round(h.start, 1),
                "end": round(h.end, 1),
                "at": timecode(h.start),
                "text": h.text,
                "similarity": None if h.similarity is None else round(h.similarity, 3),
            }
            for h in hits
        ],
    }


@mcp.tool(annotations=READ_ONLY)
def get_transcript(
    file: Annotated[str, Field(description="Full path or file name from a search result.")],
    start: Annotated[float, Field(ge=0, description="Seconds from the start of the file.")] = 0,
    end: Annotated[float | None, Field(description="Seconds; omit to read to the end (capped).")] = None,
    max_segments: Annotated[int, Field(ge=1, le=500)] = 150,
) -> dict:
    """Read the transcript of one indexed file, optionally just a time range."""
    with _lock:
        conn = _conn()
        src = _resolve_source(conn, file)
        rows = conn.execute(
            "select start, end, text from segments where source_id = ? and end > ? and start < ? order by start limit ?",
            (src["id"], start, end if end is not None else 1e12, max_segments + 1),
        ).fetchall()
    truncated = len(rows) > max_segments
    return {
        "path": src["path"],
        "duration": src["duration"],
        "truncated": truncated,
        "segments": [
            {"start": round(r["start"], 1), "at": timecode(r["start"]), "text": r["text"]}
            for r in rows[:max_segments]
        ],
    }


@mcp.tool(annotations=READ_ONLY)
def list_sources(limit: Annotated[int, Field(ge=1, le=500)] = 100) -> dict:
    """List the recordings in the index with their length and status."""
    with _lock:
        conn = _conn()
        rows = conn.execute(
            "select path, duration, status, indexed_at from sources order by path limit ?", (limit,)
        ).fetchall()
        total = conn.execute("select count(*) from sources").fetchone()[0]
    return {
        "total": total,
        "sources": [
            {"path": r["path"], "duration": r["duration"], "status": r["status"], "indexed_at": r["indexed_at"]}
            for r in rows
        ],
    }


@mcp.tool(annotations=READ_ONLY)
def index_status() -> dict:
    """Say whether indexing is running and how many files are done, waiting or failed."""
    with _lock:
        conn = _conn()
        counts = {r["status"]: r["n"] for r in conn.execute("select status, count(*) n from sources group by status")}
        failed = conn.execute("select path, error from sources where status = 'failed' limit 5").fetchall()
        chunks = conn.execute("select count(*) from chunks").fetchone()[0]
    return {
        "indexer_running": indexer_running(),
        "files": counts,
        "chunks": chunks,
        "recent_failures": [{"path": r["path"], "error": r["error"]} for r in failed],
    }


def main():
    # stdout belongs to the protocol, anything we print has to go to stderr
    mcp.run("stdio")


if __name__ == "__main__":
    main()
