# hearsay-mcp

Search what was *said* in your local video and audio files, from your AI agent.

Point it at folders of recordings (talks, courses, meetings, interviews, podcasts).
Your agent (Claude Code, Cursor, ...) calls `search("when did they talk about X")` and gets back
the file, the exact timestamp, and the passage. Everything runs on your machine, nothing is uploaded.

**Status: pre-alpha.** Works on Linux with an NVIDIA GPU (CPU fallback exists but is untested).
See [docs/PLAN.md](docs/PLAN.md) for the research, decisions and measured results.

## Use it

```bash
uv venv && uv pip install -e ".[cuda]"      # drop [cuda] if you have no NVIDIA GPU
hearsay models pull                         # one-off download, ~1.6 GB (whisper, embedder, reranker)
hearsay add ~/Videos/talks                  # register a folder
hearsay index                               # transcribe + index anything new or changed
hearsay index --watch                       # or keep running and pick up new files (scans every 30 s)
hearsay remove ~/Videos/old                 # stop watching a folder and drop it from the index
hearsay status
```

Then give your agent the MCP server (stdio):

```bash
claude mcp add hearsay -- /path/to/.venv/bin/hearsay serve
```

Tools the agent gets (all read-only): `search`, `get_transcript`, `list_sources`, `index_status`.
On CPU-only machines it uses the same model in int8 (about 6x realtime on 12 cores); `HEARSAY_WHISPER_MODEL=small.en` trades accuracy for speed.
Indexing is only ever done by the `hearsay index` command, never by the agent.
`hearsay index` also drops files that were deleted or moved (unless their folder is missing or holds no media at all, which looks like an unplugged drive). With `--watch`, files are indexed once they have stopped changing for 15 s, and a file that failed is retried only after it changes.

Set `HEARSAY_HOME` to keep the index somewhere other than `~/.hearsay`.

## How it works

ffmpeg-free decode (PyAV) -> faster-whisper with VAD and word timestamps -> ~45 s overlapping
chunks -> bge-small embeddings -> one SQLite file (FTS5 keyword index + vectors).
Search fuses keyword and vector ranks (reciprocal rank fusion), then a small cross-encoder reranks the top 20 (about 0.8 s on CPU; `HEARSAY_RERANK=0` skips it). Passages it scores as unrelated are dropped, so off-topic questions return nothing (`HEARSAY_MIN_RELEVANCE` tunes this, `off` disables).

## Develop

```bash
uv pip install -e ".[dev]" && pytest
HEARSAY_HOME=<index with tests/data indexed> python scripts/eval.py --verbose
```

Test media is public domain and not checked in; see [tests/data/README.md](tests/data/README.md).

License: MIT
