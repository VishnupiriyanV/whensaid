# whensaid

Search what was *said* in your local video and audio files, from your AI agent.

Point it at folders of recordings (talks, courses, meetings, interviews, podcasts).
Your agent (Claude Code, Cursor, ...) calls `search("when did they talk about X")` and gets back
the file, the exact timestamp, and the passage. Everything runs on your machine, nothing is uploaded.

**Status: pre-alpha.** Linux only so far. Runs on an NVIDIA GPU or on CPU. English only.
See [docs/ROADMAP.md](docs/ROADMAP.md) for what is next and [docs/PLAN.md](docs/PLAN.md) for the research and measured results.

## Use it

```bash
pip install -e ".[url]"                     # from a clone, until it is on PyPI. Add cuda on an NVIDIA GPU: ".[cuda,url]"
whensaid doctor                             # checks the GPU, models, yt-dlp and the index, says how to fix problems
whensaid add ~/Videos/talks                 # register a folder of recordings
whensaid add-url https://www.youtube.com/playlist?list=...   # or a video, playlist or channel
whensaid index                              # transcribe and index what is new (models download on first use)
whensaid index --watch                      # or keep running and pick up new files every 30 s
whensaid index --subtitles                  # use a .srt/.vtt next to a recording instead of transcribing it
whensaid remove ~/Videos/old                # stop watching a folder, playlist or video and drop it from the index
whensaid status
```

The models (about 1.6 GB for the default set) are downloaded the first time they are needed. `whensaid models pull`
fetches them ahead of time, and `WHENSAID_OFFLINE=1` forbids any download.

Then give your agent the MCP server (stdio):

```bash
claude mcp add whensaid -- whensaid serve
```

Tools the agent gets (all read-only): `search`, `get_transcript`, `list_sources`, `index_status`.
On CPU-only machines it uses the same model in int8 (about 6x realtime on 12 cores); `WHENSAID_WHISPER_MODEL=small.en` trades accuracy for speed.
Indexing is only ever done by the `whensaid index` command, never by the agent.
`whensaid index` also drops files that were deleted or moved (unless their folder is missing or holds no media at all, which looks like an unplugged drive). With `--watch`, files are indexed once they have stopped changing for 15 s, and a file that failed is retried only after it changes.

Web videos use their captions when there are any (seconds, no GPU) and are transcribed from the audio otherwise. Search results for them carry the title and a `link` that opens the video at that moment. Playlists and channels are checked for new videos at most once an hour. Use this for videos you are allowed to download; the site's terms still apply.

Things that will never work are skipped once and not asked for again: live streams, private or removed videos, videos that are not in English, and recordings whose transcript the model itself was unsure of (usually another language, or music). `whensaid status` lists them with the reason; `whensaid index --retry-skipped` tries them again, and `WHENSAID_MIN_CONFIDENCE=0` turns the transcript check off.

Set `WHENSAID_HOME` to keep the index somewhere other than `~/.whensaid`.

## How it works

ffmpeg-free decode (PyAV) -> faster-whisper with VAD and word timestamps -> ~45 s overlapping
chunks -> bge-small embeddings -> one SQLite file (FTS5 keyword index + vectors).
Search fuses keyword and vector ranks (reciprocal rank fusion), then a small cross-encoder reranks the top 20 (about half a second per search on CPU; `WHENSAID_RERANK=0` skips it). Passages it scores as unrelated are dropped, so off-topic questions usually return nothing, and borderline hits come back marked `weak` (`WHENSAID_MIN_RELEVANCE` tunes this, `off` disables).

## Develop

```bash
uv pip install -e ".[dev]" && pytest
WHENSAID_HOME=<index with tests/data indexed> python scripts/eval.py --verbose
```

Test media is public domain and not checked in; see [tests/data/README.md](tests/data/README.md).

License: MIT
