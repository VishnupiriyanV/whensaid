import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from whensaid import db, paths, server
from whensaid.search import Searcher


class TinyEmbedder:
    def encode(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            v = np.array([t.count("cat") + t.count("kitten"), t.count("money") + t.count("cash"), 0.01], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WHENSAID_HOME", str(tmp_path))
    server._state.clear()
    yield tmp_path
    server._state.clear()


def fill(home):
    conn = db.connect(paths.db_path())
    conn.execute("insert into sources (id, path, duration, status) values (1, '/media/talk.mp4', 60, 'done')")
    conn.execute("insert into sources (id, path, status, error) values (2, '/media/bad.mp4', 'failed', 'decode error')")
    emb = TinyEmbedder()
    segs = [(0, 10, "my cat sleeps all day"), (10, 20, "a kitten chases a ball"), (20, 30, "the bank keeps your money safe")]
    for s, e, t in segs:
        conn.execute("insert into segments (source_id, start, end, text) values (1, ?, ?, ?)", (s, e, t))
    for i, (s, e, t) in enumerate(segs, 1):
        conn.execute(
            "insert into chunks (id, source_id, start, end, text, embedding) values (?, 1, ?, ?, ?, ?)",
            (i, s, e, t, emb.encode([t])[0].tobytes()),
        )
        conn.execute("insert into chunks_fts (rowid, text) values (?, ?)", (i, t))
    conn.commit()
    conn.close()


def use_fake_embedder():
    server._state["searcher"] = Searcher(server._conn(), TinyEmbedder(), query_prefix="")


def test_nothing_indexed_gives_a_helpful_error(home):
    with pytest.raises(ToolError, match="whensaid add"):
        server.search("anything")


def test_search_returns_timecodes(home):
    fill(home)
    use_fake_embedder()
    out = server.search("money", limit=2)
    top = out["results"][0]
    assert top["file"] == "talk.mp4" and top["path"] == "/media/talk.mp4"
    assert top["start"] == 20.0 and top["at"] == "0:00:20"
    assert top["similarity"] > 0.9


def test_empty_query_is_an_error(home):
    fill(home)
    use_fake_embedder()
    with pytest.raises(ToolError, match="empty"):
        server.search("  ")


def test_get_transcript_range_and_name_lookup(home):
    fill(home)
    out = server.get_transcript("talk.mp4", start=12, end=25)  # by name, not full path
    assert [s["text"] for s in out["segments"]] == ["a kitten chases a ball", "the bank keeps your money safe"]
    assert out["truncated"] is False
    assert server.get_transcript("/media/talk.mp4", max_segments=1)["truncated"] is True


def test_only_indexed_files_can_be_read(home):
    fill(home)
    with pytest.raises(ToolError, match="not in the index"):
        server.get_transcript("/etc/passwd")


def test_status_and_sources(home):
    fill(home)
    st = server.index_status()
    assert st["files"] == {"done": 1, "failed": 1} and st["chunks"] == 3
    assert st["recent_failures"] == [{"path": "/media/bad.mp4", "error": "decode error"}]
    assert server.list_sources()["total"] == 2


def test_server_is_read_only(home):
    fill(home)
    with pytest.raises(Exception):
        server._conn().execute("delete from chunks")


def test_timecode():
    assert server.timecode(0) == "0:00:00"
    assert server.timecode(3725.9) == "1:02:05"


def test_underscore_in_a_file_name_is_not_a_wildcard(home):
    fill(home)
    conn = db.connect(paths.db_path())
    conn.execute("insert into sources (path, status) values ('/media/a_b.mp4', 'done')")
    conn.execute("insert into sources (path, status) values ('/media/aXb.mp4', 'done')")
    conn.commit()
    assert server._resolve_source(conn, "a_b.mp4")["path"] == "/media/a_b.mp4"
    with pytest.raises(ToolError):
        server._resolve_source(conn, "%")
    with pytest.raises(ToolError):
        server._resolve_source(conn, "b.mp4")  # must match a whole name, not the end of one


def test_weak_matches_are_labelled(home, monkeypatch):
    from whensaid import rerank

    assert rerank.strength(None) is None
    assert rerank.strength(-5.1) == "weak" and rerank.strength(-4.9) == "strong"

    class Lukewarm:
        def score(self, query, texts):
            return [-6.0 - i for i in range(len(texts))]

    fill(home)
    server._state["searcher"] = Searcher(server._conn(), TinyEmbedder(), query_prefix="", reranker=Lukewarm(), min_relevance=-9.0)
    out = server.search("cat kitten money", limit=5)
    assert out["results"] and all(r["match"] == "weak" for r in out["results"])
    assert "weak" in out["note"]
    assert all(r["relevance"] >= -9 for r in out["results"])  # below the floor is still dropped
