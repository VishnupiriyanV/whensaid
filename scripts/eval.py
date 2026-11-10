"""Score retrieval against tests/golden.json.

    HEARSAY_HOME=<home with the test files indexed> python scripts/eval.py [--modes keyword,semantic,hybrid]

A hit counts if a returned chunk is from the right file and overlaps the expected window.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from hearsay import db, paths
from hearsay.embed import Embedder
from hearsay.search import BGE_QUERY_PREFIX, Searcher

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden.json"


def is_hit(h, q):
    return os.path.basename(h.path) == q["file"] and h.start < q["end"] and h.end > q["start"]


def first_hit_rank(hits, q):
    for i, h in enumerate(hits, 1):
        if is_hit(h, q):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="keyword,semantic,hybrid")
    ap.add_argument("--no-prefix", action="store_true", help="don't add the bge query instruction")
    ap.add_argument("--verbose", action="store_true", help="show the misses")
    args = ap.parse_args()

    golden = json.load(open(GOLDEN))
    conn = db.connect(paths.db_path())
    searcher = Searcher(conn, Embedder(), query_prefix="" if args.no_prefix else BGE_QUERY_PREFIX)

    print(f"{'mode':10} {'kind':6} {'n':>3} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}")
    for mode in args.modes.split(","):
        ranks = [(q, first_hit_rank(searcher.search(q["query"], limit=10, mode=mode), q)) for q in golden["queries"]]
        for kind in ("all", "exact", "para"):
            rs = [r for q, r in ranks if kind == "all" or q["kind"] == kind]
            n = len(rs)
            at = lambda k: sum(1 for r in rs if r and r <= k) / n
            mrr = sum(1 / r for r in rs if r) / n
            print(f"{mode:10} {kind:6} {n:3} {at(1):6.2f} {at(5):6.2f} {at(10):6.2f} {mrr:6.2f}")
        if args.verbose:
            for q, r in ranks:
                if r is None or r > 3:
                    print(f"   miss ({mode}) rank={r}: {q['query']}")
    print()


if __name__ == "__main__":
    main()
