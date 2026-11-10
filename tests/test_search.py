import numpy as np

from hearsay import db
from hearsay.search import Searcher, fts_query, rrf


def test_fts_query_quotes_and_drops_stopwords():
    assert fts_query("What did they say about C++ and gradient descent?") == '"say" OR "c" OR "gradient" OR "descent"'


def test_fts_query_survives_nasty_input():
    assert fts_query('he said "no') == '"said" OR "no"'  # no stray quote reaches fts5
    assert fts_query("***") is None
    assert fts_query("the of") == '"the" OR "of"'  # only stopwords: keep them rather than search nothing


def test_rrf_prefers_agreement():
    out = rrf({"a": [1, 2, 3], "b": [3, 2, 9]})
    assert {cid for cid, _, _ in out[:2]} == {2, 3}  # the two ids that appear in both lists
    assert dict((cid, r) for cid, _, r in out)[2] == {"a": 2, "b": 2}


class TinyEmbedder:
    """Two made-up dimensions: 'cat-ness' and 'money-ness'."""
    def encode(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            v = np.array([t.count("cat") + t.count("kitten"), t.count("money") + t.count("cash")], dtype=np.float32)
            out.append(v / max(np.linalg.norm(v), 1e-9))
        return np.array(out)


def make_db(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    conn.execute("insert into sources (id, path, status) values (1, '/a.mp4', 'done')")
    emb = TinyEmbedder()
    texts = ["my cat sleeps all day", "a kitten chases a ball", "the bank keeps your money safe", "weather is nice"]
    for i, t in enumerate(texts, 1):
        conn.execute(
            "insert into chunks (id, source_id, start, end, text, embedding) values (?, 1, ?, ?, ?, ?)",
            (i, i * 10.0, i * 10.0 + 8, t, emb.encode([t])[0].tobytes()),
        )
        conn.execute("insert into chunks_fts (rowid, text) values (?, ?)", (i, t))
    conn.commit()
    return conn


def test_modes(tmp_path):
    s = Searcher(make_db(tmp_path), TinyEmbedder(), query_prefix="")
    kw = s.search("cat", mode="keyword")
    assert [h.chunk_id for h in kw] == [1]  # keyword only finds the literal word
    sem = s.search("kitten", mode="semantic", limit=2)
    assert {h.chunk_id for h in sem} == {1, 2}  # semantic also finds the cat chunk
    hy = s.search("kitten", mode="hybrid")
    assert hy[0].chunk_id == 2 and set(hy[0].matched_by) == {"keyword", "semantic"}
    assert hy[0].path == "/a.mp4" and hy[0].start == 20.0


def test_empty_db(tmp_path):
    s = Searcher(db.connect(tmp_path / "e.db"), TinyEmbedder(), query_prefix="")
    assert s.search("anything") == []


def test_overlapping_hits_from_one_passage_are_merged(tmp_path):
    conn = db.connect(tmp_path / "o.db")
    conn.execute("insert into sources (id, path, status) values (1, '/a.mp4', 'done')")
    emb = TinyEmbedder()
    # two chunks overlapping by 6s, plus one far away; all about cats
    for i, (s, e) in enumerate([(0, 45), (39, 84), (200, 245)], 1):
        t = f"cat talk number {i}"
        conn.execute(
            "insert into chunks (id, source_id, start, end, text, embedding) values (?, 1, ?, ?, ?, ?)",
            (i, s, e, t, emb.encode([t])[0].tobytes()),
        )
        conn.execute("insert into chunks_fts (rowid, text) values (?, ?)", (i, t))
    conn.commit()
    hits = Searcher(conn, emb, query_prefix="").search("cat", limit=5)
    assert len(hits) == 2
    assert not (hits[0].start < hits[1].end and hits[1].start < hits[0].end)
