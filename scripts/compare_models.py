"""Try other embedding models and rerankers on the golden set, without touching the index.

    WHENSAID_HOME=<home> python scripts/compare_models.py --embedders bge-base,gte-base --rerankers jina-turbo

Chunks are re-embedded in memory for each candidate. Hybrid is keyword + semantic fused with
rrf as in the real searcher, and a reranker (if given) reorders the top 20 of that.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from whensaid import db, paths
from whensaid.search import Searcher, rrf

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden.json"
BGE = "Represent this sentence for searching relevant passages: "

# name -> (model id, query prefix, passage prefix)
EMBEDDERS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", BGE, ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", BGE, ""),
    "bge-large": ("BAAI/bge-large-en-v1.5", BGE, ""),
    "arctic-m": ("snowflake/snowflake-arctic-embed-m", BGE, ""),
    "gte-base": ("thenlper/gte-base", "", ""),
    "nomic-q": ("nomic-ai/nomic-embed-text-v1.5-Q", "search_query: ", "search_document: "),
    "mxbai": ("mixedbread-ai/mxbai-embed-large-v1", BGE, ""),
}
RERANKERS = {
    "minilm-12": "Xenova/ms-marco-MiniLM-L-12-v2",
    "jina-turbo": "jinaai/jina-reranker-v1-turbo-en",
    "bge-rr-base": "BAAI/bge-reranker-base",
}


def is_hit(path, start, end, q):
    return any(os.path.basename(path) == a["file"] and start < a["end"] and end > a["start"] for a in [q, *q.get("alt", [])])


def score(ranked, golden):
    """ranked: one list of (path, start, end) per query, best first."""
    rows = []
    for q, hits in zip(golden, ranked):
        r = next((i for i, h in enumerate(hits, 1) if is_hit(*h[:3], q)), None)
        rows.append(r)
    n = len(rows)
    at = lambda k: sum(1 for r in rows if r and r <= k) / n
    return at(1), at(5), at(10), sum(1 / r for r in rows if r) / n


def dedupe(hits, limit=10):
    out = []
    for h in hits:
        if any(o[0] == h[0] and o[1] < h[2] and h[1] < o[2] for o in out):
            continue
        out.append(h)
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", default="bge-small,bge-base,gte-base,arctic-m,nomic-q")
    ap.add_argument("--rerankers", default="minilm-12,jina-turbo,bge-rr-base")
    ap.add_argument("--depth", type=int, default=20, help="how many fused hits the reranker sees")
    args = ap.parse_args()

    golden = [q for q in json.load(open(GOLDEN))["queries"]]
    conn = db.connect(paths.db_path())
    chunks = conn.execute(
        "select c.id, c.start, c.end, c.text, s.path from chunks c join sources s on s.id = c.source_id order by c.id"
    ).fetchall()
    ids = [c["id"] for c in chunks]
    by_id = {c["id"]: c for c in chunks}
    print(f"{len(chunks)} chunks, {len(golden)} queries")

    kw = Searcher(conn, None)  # only used for its keyword side
    keyword = [kw.keyword_ids(q["query"], 50) for q in golden]

    def as_hits(chunk_ids):
        return [(by_id[i]["path"], by_id[i]["start"], by_id[i]["end"], i) for i in chunk_ids]

    rerank_models = {}
    print(f"\n{'embedder':10} {'stage':22} {'R@1':>5} {'R@5':>5} {'R@10':>5} {'MRR':>5}  {'embed s':>7}")
    for name in args.embedders.split(","):
        model_id, qp, pp = EMBEDDERS[name]
        t0 = time.time()
        emb = TextEmbedding(model_id)
        mat = np.array(list(emb.embed([pp + c["text"] for c in chunks], batch_size=32)), dtype=np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        took = time.time() - t0
        qv = np.array(list(emb.embed([qp + q["query"] for q in golden])), dtype=np.float32)
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)

        sem = [[ids[j] for j in np.argsort(-(mat @ v))[:50]] for v in qv]
        fused = [[cid for cid, _, _ in rrf({"keyword": k, "semantic": s})] for k, s in zip(keyword, sem)]

        def show(stage, lists, t=None):
            ranked = [dedupe(as_hits(l)) for l in lists]
            r1, r5, r10, mrr = score(ranked, golden)
            print(f"{name:10} {stage:22} {r1:5.2f} {r5:5.2f} {r10:5.2f} {mrr:5.2f}  {t if t else '':>7}")

        show("semantic", sem, f"{took:.0f}")
        show("hybrid", fused)
        for rname in filter(None, args.rerankers.split(",")):
            if rname not in rerank_models:
                rerank_models[rname] = TextCrossEncoder(RERANKERS[rname])
            rr = rerank_models[rname]
            out = []
            for q, f in zip(golden, fused):
                top = f[: args.depth]
                scores = list(rr.rerank(q["query"], [by_id[i]["text"] for i in top]))
                out.append([i for _, i in sorted(zip(scores, top), key=lambda x: -x[0])])
            show(f"hybrid+{rname}", out)


if __name__ == "__main__":
    main()
