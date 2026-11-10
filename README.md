# hearsay-mcp

Search what was *said* in your local video and audio files, from your AI agent.

Point it at folders of recordings (talks, courses, meetings, interviews, podcasts).
Your agent (Claude Code, Cursor, ...) calls `search("what did they say about X")` and gets back
the file, the exact timestamp, and the surrounding transcript. Everything runs on your machine.

**Status: pre-alpha, planning only.** See [docs/PLAN.md](docs/PLAN.md) for the research, stack
decisions, and milestones.

Planned scope (v0): video + audio only, Linux + CUDA, English, stdio MCP server, CLI indexer.

License: MIT
