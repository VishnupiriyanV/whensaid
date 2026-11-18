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
from .rerank import strength
from .search import Searcher
from .util import deep_link, timecode  # noqa: F401 (timecode is also used by tests via server.timecode)

INSTRUCTIONS = """\
Search what was said in the user's local video and audio recordings.
Start with `search`: it returns the matching passage, the file, and where in the file it was
said (seconds and h:mm:ss). Use `get_transcript` to read more around a hit, passing the hit's
`path`. All times are in seconds from the start of the file. For videos from the web a hit also
has `title`, `url` and `link`, a url that opens the video at that moment: give it to the user. Passages the reranker judges unrelated are left out, so an
empty result (it comes with a note) means the recordings most likely don't cover the topic: say
so rather than guessing. Each hit has `match`: "strong" or "weak". A weak hit is often off-topic,
so check its text before using it; when even the best hit is weak the result says so in `note`.
`relevance` is the reranker score behind that (higher is better), `similarity` is cosine similarity. If `search` says nothing is indexed, tell the user to
run `whensaid add <folder>` and `whensaid index`.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

mcp = MCPServer("whensaid", instructions=INSTRUCTIONS)

_lock = threading.Lock()  # sync tools run on worker threads and share one connection
_state: dict = {}


def _conn():
    if "conn" not in _state:
        try:
            _state["conn"] = db.connect_readonly(paths.db_path())
        except FileNotFoundError:
            raise ToolError("Nothing is indexed yet. Run `whensaid add <folder>` then `whensaid index`.")
    return _state["conn"]


def _searcher() -> Searcher:
    prep = _state.get("prep")
    if prep is not None and not prep.ready():
        # never build the searcher around a model that is still downloading: it would run without
        # the reranker (no weak labels, no cutoff) until the server restarts, and nobody would see it
        raise ToolError(prep.message())
    if "searcher" not in _state:
        from .embed import Embedder
        from .rerank import load_reranker, min_relevance

        _state["searcher"] = Searcher(_conn(), Embedder(), reranker=load_reranker(), min_relevance=min_relevance())
    return _state["searcher"]


def _has_web_cols(conn) -> bool:
    return "url" in {r[1] for r in conn.execute("pragma table_info(sources)")}  # not cached: a writer may upgrade the index later


def _col(row, name):
    return row[name] if name in row.keys() else None  # old indexes have no web columns


def _resolve_source(conn, file: str):
    """Accepts a full path, a web video's key or url, or a file name. Only what is in the index can be read."""
    rows = conn.execute("select * from sources where path = ?", (file,)).fetchall()
    if not rows and _has_web_cols(conn):
        rows = conn.execute("select * from sources where url = ?", (file,)).fetchall()
    if not rows:
        # compared in python: a LIKE here would treat _ and % in file names as wildcards
        tail = os.sep + file.lstrip(os.sep)
        rows = [r for r in conn.execute("select * from sources") if r["path"].endswith(tail)]
    if not rows and _has_web_cols(conn):
        rows = conn.execute("select * from sources where title = ?", (file,)).fetchall()  # search shows web videos by title
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
        raise ToolError("Nothing is indexed yet. Run `whensaid add <folder>` then `whensaid index`.")
    out = {
        "query": query,
        "results": [
            {
                "file": h.title or os.path.basename(h.path),
                "path": h.path,
                "start": round(h.start, 1),
                "end": round(h.end, 1),
                "at": timecode(h.start),
                "title": h.title,
                "url": h.url,
                "link": deep_link(h.url, h.start),
                "text": h.text,
                "similarity": None if h.similarity is None else round(h.similarity, 3),
                "relevance": None if h.relevance is None else round(h.relevance, 1),
                "match": strength(h.relevance),
            }
            for h in hits
        ],
    }
    if not hits:
        out["note"] = "No passage was relevant enough. The recordings probably don't cover this."
    elif strength(hits[0].relevance) == "weak":
        out["note"] = (
            "Only weak matches. The recordings may not cover this: read the passages before "
            "relying on them, and say so if they don't answer the question."
        )
    return out


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
        "title": _col(src, "title"),
        "url": _col(src, "url"),
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
            "select * from sources where status != 'removed' order by path limit ?", (limit,)
        ).fetchall()
        total = conn.execute("select count(*) from sources where status != 'removed'").fetchone()[0]
    return {
        "total": total,
        "sources": [
            {"path": r["path"], "title": _col(r, "title"), "url": _col(r, "url"), "captions": _col(r, "captions"),
             "duration": r["duration"], "status": r["status"], "indexed_at": r["indexed_at"]}
            for r in rows
        ],
    }


@mcp.tool(annotations=READ_ONLY)
def index_status() -> dict:
    """Say whether indexing is running and how many files are done, waiting or failed."""
    with _lock:
        conn = _conn()
        counts = {r["status"]: r["n"] for r in conn.execute(
            "select status, count(*) n from sources where status != 'removed' group by status")}
        failed = conn.execute("select path, error from sources where status = 'failed' limit 5").fetchall()
        chunks = conn.execute("select count(*) from chunks").fetchone()[0]
    prep = _state.get("prep")
    return {
        "models": prep.describe() if prep is not None else None,
        "indexer_running": indexer_running(),
        "files": counts,
        "chunks": chunks,
        "recent_failures": [{"path": r["path"], "error": r["error"]} for r in failed],
    }


def main():
    # stdout belongs to the protocol, anything we print has to go to stderr
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)  # one log line per request otherwise
    from .models import Prep

    _state["prep"] = Prep().start()  # first use: download in the background, answer "still downloading" meanwhile
    mcp.run("stdio")


if __name__ == "__main__":
    main()
