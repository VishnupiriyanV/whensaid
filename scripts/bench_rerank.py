"""Compare reranker variants for speed and accuracy on the golden sets.

    WHENSAID_HOME=<home with everything indexed> python scripts/bench_rerank.py

The fused keyword+semantic candidates are computed once per query, then each variant only
reorders them, so differences between rows come from the reranker alone. Latency is measured
with a query embedding run just before each rerank, the way a real search does it.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder

from whensaid import db, paths
from whensaid.embed import Embedder
from whensaid.paths import models_dir
from whensaid.search import Searcher, rrf

TESTS = Path(__file__).resolve().parent.parent / "tests"
SETS = ["golden.json", "golden_noisy.json", "golden_scale.json"]

# name -> (hf repo, onnx file)
MODELS = {
    "L12": ("Xenova/ms-marco-MiniLM-L-12-v2", "onnx/model.onnx"),
    "L12-q": ("Xenova/ms-marco-MiniLM-L-12-v2", "onnx/model_quantized.onnx"),
    "L6": ("Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx"),
    "L6-q": ("Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model_quantized.onnx"),
}


def load(name, threads):
    repo, file = MODELS[name]
    custom = f"whensaid-bench/{name}"
    try:
        TextCrossEncoder.add_custom_model(model=custom, sources=ModelSource(hf=repo), model_file=file)
    except ValueError:
        pass  # registered on an earlier call
    return TextCrossEncoder(custom, cache_dir=str(models_dir() / "fastembed"), threads=threads)


def is_hit(row, q):
    return any(
        os.path.basename(row["path"]) == a["file"] and row["start"] < a["end"] and row["end"] > a["start"]
        for a in [q, *q.get("alt", [])]
    )


def first_hit(rows, q, limit=10):
    kept = []
    for r in rows:
        if any(k["path"] == r["path"] and k["start"] < r["end"] and r["start"] < k["end"] for k in kept):
            continue  # same passage as a better hit, as the searcher does
        kept.append(r)
        if is_hit(r, q):
            return len(kept)
        if len(kept) >= limit:
            break
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="L12:0,L12:4,L12-q:4,L6:4,L6-q:4", help="model:threads, 0 = library default")
    ap.add_argument("--depths", default="20,10")
    args = ap.parse_args()

    conn = db.connect_readonly(paths.db_path())
    embedder = Embedder()
    searcher = Searcher(conn, embedder)
    queries = []
    for name in SETS:
        for q in json.load(open(TESTS / name))["queries"]:
            queries.append((name, q))

    # candidates once per query: fused order, with the rows needed to score a hit
    cands = []
    for _, q in queries:
        qv = searcher.query_vector(q["query"])
        fused = rrf({"keyword": searcher.keyword_ids(q["query"], 50), "semantic": searcher.semantic_ids(q["query"], 50, qv)})
        ids = [cid for cid, _, _ in fused[:40]]
        marks = ",".join("?" * len(ids))
        rows = {r["id"]: r for r in conn.execute(
            f"select c.id, c.start, c.end, c.text, s.path from chunks c join sources s on s.id = c.source_id where c.id in ({marks})", ids)}
        cands.append([rows[i] for i in ids])

    def report(label, ranked, ms=None):
        cells = []
        for name in SETS:
            rs = [first_hit(r, q) for (n, q), r in zip(queries, ranked) if n == name]
            cells.append(f"{sum(1 for r in rs if r == 1) / len(rs):.2f}/{sum(1 for r in rs if r and r <= 5) / len(rs):.2f}")
        t = "" if ms is None else f"{statistics.median(ms):5.0f} {max(ms):5.0f}"
        print(f"{label:16} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}   {t}")

    print(f"{len(queries)} queries, R@1/R@5 per set, then median and max rerank ms")
    print(f"{'variant':16} {'clean':>10} {'noisy':>10} {'scale':>10}   {'med':>5} {'max':>5}")
    report("no reranker", cands)
    for spec in args.variants.split(","):
        name, threads = spec.split(":")
        rr = load(name, int(threads) or None)
        list(rr.rerank("warm up", ["warm up"]))
        for depth in [int(d) for d in args.depths.split(",")]:
            ranked, ms = [], []
            for (_, q), rows in zip(queries, cands):
                embedder.encode([q["query"]])  # a search embeds the query right before reranking
                head = rows[:depth]
                t = time.time()
                scores = list(rr.rerank(q["query"], [r["text"] for r in head]))
                ms.append((time.time() - t) * 1000)
                order = [r for _, r in sorted(zip(scores, head), key=lambda x: -x[0])]
                ranked.append(order + rows[depth:])
            report(f"{name} t={threads} d={depth}", ranked, ms)


if __name__ == "__main__":
    main()
