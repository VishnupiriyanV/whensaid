"""End-to-end check used by CI: index one short clip and find a phrase from it.

    WHENSAID_HOME=<empty dir> python scripts/smoke.py tests/data

Runs the real commands as a user would, in a subprocess each.
"""
import json
import subprocess
import sys

folder = sys.argv[1]


def run(*args):
    print("$ whensaid", *args, flush=True)
    r = subprocess.run(["whensaid", *args], capture_output=True, text=True, timeout=1500)
    print(r.stdout[-1500:], r.stderr[-1500:], sep="", flush=True)
    if r.returncode != 0:
        sys.exit(f"whensaid {args[0]} exited with {r.returncode}")
    return r.stdout


run("doctor")
run("add", folder)
run("index")  # downloads the models on first use
out = run("search", "ask not what your country can do for you", "--json")
hits = json.loads(out)
assert hits, "no results"
top = hits[0]
assert top["file"] == "jfk.wav", top
assert "country" in top["text"].lower(), top
assert top["match"] == "strong", top  # the reranker ran, and was sure
print("smoke test passed:", top["file"], top["at"], top["relevance"])
