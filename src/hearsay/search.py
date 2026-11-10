import re
from dataclasses import dataclass, field

import numpy as np

# bge models want this in front of the query (not the passages) for retrieval
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

STOPWORDS = set(
    "a an the of to in on at is are was were be been it its this that and or for with as by "
    "from i you he she we they what who how when where why did do does about".split()
)


@dataclass
class Hit:
    chunk_id: int
    path: str
    start: float
    end: float
    text: str
    score: float
    matched_by: dict = field(default_factory=dict)  # mode -> rank


def fts_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 query: quoted words joined with OR.
    Quoting stops characters like " or - from being read as fts syntax."""
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]
    if not words:
        words = re.findall(r"\w+", text.lower())
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


def rrf(rankings: dict[str, list[int]], k: int = 60) -> list[tuple[int, float, dict]]:
    """Reciprocal rank fusion. rankings maps a name to chunk ids, best first."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict] = {}
    for name, ids in rankings.items():
        for rank, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(cid, {})[name] = rank
    return sorted(((cid, s, ranks[cid]) for cid, s in scores.items()), key=lambda x: -x[1])


class Searcher:
    def __init__(self, conn, embedder, query_prefix: str = BGE_QUERY_PREFIX):
        self.conn = conn
        self.embedder = embedder
        self.query_prefix = query_prefix
        self._key = None
        self._ids = np.zeros(0, dtype=np.int64)
        self._mat = np.zeros((0, 0), dtype=np.float32)

    def _load_vectors(self):
        # reload only when the db changed (another process may be indexing)
        key = (
            self.conn.execute("pragma data_version").fetchone()[0],
            *self.conn.execute("select count(*), coalesce(max(id), 0) from chunks").fetchone(),
        )
        if key == self._key:
            return
        rows = self.conn.execute("select id, embedding from chunks where embedding is not null").fetchall()
        self._ids = np.array([r["id"] for r in rows], dtype=np.int64)
        self._mat = (
            np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        self._key = key

    def keyword_ids(self, query: str, n: int) -> list[int]:
        q = fts_query(query)
        if q is None:
            return []
        rows = self.conn.execute(
            "select rowid from chunks_fts where chunks_fts match ? order by bm25(chunks_fts) limit ?",
            (q, n),
        )
        return [r[0] for r in rows]

    def semantic_ids(self, query: str, n: int) -> list[int]:
        self._load_vectors()
        if len(self._ids) == 0:
            return []
        qv = self.embedder.encode([self.query_prefix + query])[0]
        sims = self._mat @ qv  # vectors are normalised, so this is cosine similarity
        top = np.argsort(-sims)[:n]
        return [int(self._ids[i]) for i in top]

    def search(self, query: str, limit: int = 10, mode: str = "hybrid", pool: int = 50, k: int = 60) -> list[Hit]:
        rankings = {}
        if mode in ("hybrid", "keyword"):
            rankings["keyword"] = self.keyword_ids(query, pool)
        if mode in ("hybrid", "semantic"):
            rankings["semantic"] = self.semantic_ids(query, pool)
        fused = rrf(rankings, k)[:limit]
        if not fused:
            return []

        marks = ",".join("?" * len(fused))
        rows = {
            r["id"]: r
            for r in self.conn.execute(
                f"select c.id, c.start, c.end, c.text, s.path from chunks c "
                f"join sources s on s.id = c.source_id where c.id in ({marks})",
                [cid for cid, _, _ in fused],
            )
        }
        return [
            Hit(cid, rows[cid]["path"], rows[cid]["start"], rows[cid]["end"], rows[cid]["text"], score, ranks)
            for cid, score, ranks in fused
            if cid in rows
        ]
