# whensaid

Search what was *said* in your local video and audio files, from your AI agent.

Point it at folders of recordings (talks, courses, meetings, interviews, podcasts) or at a YouTube playlist.
Your agent (Claude Code, Cursor, ...) calls `search("when did they talk about X")` and gets back the file, the exact
timestamp and the passage; for web videos, a link that opens the video at that moment. Everything runs on your
machine, and nothing is uploaded.

**Status: pre-alpha.** Tested on Linux (CI also runs on macOS). Runs on an NVIDIA GPU or on CPU. English only.
See [docs/ROADMAP.md](docs/ROADMAP.md) for what is next and [docs/PLAN.md](docs/PLAN.md) for the research and measured results.

## A worked example

The index here is 198 public-domain recordings (65 hours of audiobooks and old radio). The question is paraphrased,
not a quote:

```
$ whensaid search "a mouse tribe claiming rights over everyone else" -n 1
strong  the_consolation_of_philosophy_..._boethius_64kb.mp3  0:01:38  and over whom ye exercise your fancied
lordship suppose now that in the mouse tribe there should rise up one claiming rights and powers for himself above the rest ...
```

Through MCP the agent gets the same thing as JSON (`file`, `path`, `start`, `at`, `text`, `match`, `relevance`, and
`link` for web videos) and can call `get_transcript` to read around the hit. About half a second per search on CPU.

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
Indexing is only ever done by the `whensaid index` command, never by the agent.

## How good is it

Measured on the 65-hour public-domain index above, with questions written as paraphrases (details and caveats in
[docs/PLAN.md](docs/PLAN.md)):

- The right passage is the top result for 83-97% of questions and in the top 5 for 96-100%, depending on the question set.
- Off-topic questions: 33 of 40 return nothing, and 6 of the other 7 come back marked `weak`.
- Against an agent that greps plain transcript files (`whensaid export`), with Claude Sonnet agents answering 36
  questions: grep got 33 right and whensaid 31. That is a tie within noise for one run each, **so whensaid is not
  more accurate than grep at this size**. What it adds is speed (an agent finished a batch of 8 questions in about a
  minute instead of about nine), no export step, timestamp links for web videos, and a library that persists between
  sessions. The write-up is in [docs/experiments](docs/experiments/2026-11-grep-vs-whensaid-65h/README.md), including
  the display bug that made the first two runs look much worse.
- Not measured: word error rate against human transcripts, noisy or multi-speaker recordings beyond old radio, and any
  hardware other than one Linux laptop (12 cores, RTX 3050).

## Limits

- English only. Videos in other languages, and recordings the model is unsure of, are skipped (see below).
- Linux is the tested platform; macOS runs in CI but nobody has used it daily. Windows is untried.
- Speech only. It does not look at the picture.
- Transcription is the slow part: about 14x realtime on the GPU it was tested on, about 6x on 12 CPU cores with the
  default model. `WHENSAID_WHISPER_MODEL=small.en` trades accuracy for speed.
- Web videos are for videos you are allowed to download; each site's terms still apply and are your responsibility.

## Details

`whensaid index` also drops files that were deleted or moved (unless their folder is missing or holds no media at all,
which looks like an unplugged drive). With `--watch`, files are indexed once they have stopped changing for 15 s, and
a file that failed is retried only after it changes.

Web videos use their captions when there are any (seconds, no GPU) and are transcribed from the audio otherwise.
Playlists and channels are checked for new videos at most once an hour.

Things that will never work are skipped once and not asked for again: live streams, private or removed videos, videos
that are not in English, and recordings whose transcript the model itself was unsure of (usually another language, or
music). `whensaid status` lists them with the reason; `whensaid index --retry-skipped` tries them again, and
`WHENSAID_MIN_CONFIDENCE=0` turns the transcript check off.

Set `WHENSAID_HOME` to keep the index somewhere other than `~/.whensaid`.

## How it works

ffmpeg-free decode (PyAV) -> faster-whisper with VAD and word timestamps -> ~45 s overlapping
chunks -> bge-small embeddings -> one SQLite file (FTS5 keyword index + vectors).
Search fuses keyword and vector ranks (reciprocal rank fusion), then a small cross-encoder reranks the top 20
(`WHENSAID_RERANK=0` skips it). Passages it scores as unrelated are dropped, so off-topic questions usually return
nothing, and borderline hits come back marked `weak` (`WHENSAID_MIN_RELEVANCE` tunes this, `off` disables).

## Develop

```bash
uv pip install -e ".[dev]" && pytest
WHENSAID_HOME=<index with tests/data indexed> python scripts/eval.py --verbose
```

Test media is public domain; see [tests/data/README.md](tests/data/README.md).

License: MIT
