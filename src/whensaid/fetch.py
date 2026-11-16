"""Getting a transcript for a video that lives on the web, through yt-dlp.

Captions come first (no GPU, seconds); only a video without any English captions has its audio
downloaded and transcribed. yt-dlp is an optional extra and is always run as a subprocess, with
`--` before the url, so a hostile url can't be read as an option.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import subtitles
from .paths import home


class FetchError(Exception):
    pass


def ytdlp_cmd() -> list[str]:
    # the copy in our own environment first: an old system yt-dlp is the usual reason youtube fails
    try:
        import yt_dlp  # noqa: F401

        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        pass
    found = shutil.which("yt-dlp")
    if found:
        return [found]
    raise FetchError("yt-dlp is not installed. Run: pip install 'whensaid[url]'")


def choose_track(info: dict) -> tuple[str, str] | None:
    """Pick one English caption track: manual beats auto. Returns (kind, language key)."""
    def english(tracks, prefer):
        keys = [k for k in (tracks or {}) if k == "en" or k.startswith("en-") or k.startswith("en_")]
        for k in prefer:
            if k in keys:
                return k
        return keys[0] if keys else None

    k = english(info.get("subtitles"), ["en", "en-US", "en-GB"])
    if k:
        return "manual", k
    k = english(info.get("automatic_captions"), ["en", "en-orig"])
    if k:
        return "auto", k
    return None


class YtDlp:
    def __init__(self, timeout: int = 900):
        self.timeout = timeout

    def _run(self, args: list[str], url: str | None = None) -> str:
        cmd = ytdlp_cmd() + ["--no-warnings", "--ignore-config", *args]
        if url:
            cmd += ["--", url]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise FetchError("yt-dlp timed out")
        if r.returncode != 0:
            errs = [ln for ln in r.stderr.splitlines() if ln.startswith("ERROR")]
            last = errs[-1] if errs else (r.stderr.strip().splitlines() or ["yt-dlp failed"])[-1]
            raise FetchError(last[:400])
        return r.stdout

    def entries(self, url: str, limit: int = 200, no_playlist: bool = False) -> tuple[str, str | None, list[dict]]:
        """What a url points at: ("video" or "playlist", its title, [{key, url, title}])."""
        args = ["-J", "--flat-playlist", "--skip-download", "--playlist-end", str(limit)]
        if no_playlist:
            args.append("--no-playlist")
        data = json.loads(self._run(args, url))
        if data.get("_type") != "playlist":
            return "video", data.get("title"), [_entry(data)]
        found: list[dict] = []

        def walk(node):
            for e in node.get("entries") or []:
                if len(found) >= limit:
                    return
                if e.get("_type") == "playlist":  # a channel is a playlist of tabs
                    walk(e)
                elif e.get("id"):
                    found.append(_entry(e))

        walk(data)
        return "playlist", data.get("title"), found

    def acquire(self, url: str, transcriber) -> tuple[list, float, str]:
        """(segments, duration, how) for one video. how is manual, auto or whisper."""
        base = home() / "tmp"
        base.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=base))  # not /tmp, that is often RAM
        try:
            info = json.loads(self._run(["-J", "--skip-download", "--no-playlist"], url))
            duration = float(info.get("duration") or 0)
            orig = [k for k in (info.get("automatic_captions") or {}) if k.endswith("-orig") and k != "en-orig"]
            if orig and not info.get("subtitles", {}).get("en"):
                # youtube offers a machine-translated English track for any language; that is not
                # what was said, and whisper here is English-only, so don't pretend
                raise FetchError(f"video is not in English (original language {orig[0][:-5]}); only English is supported so far")
            pick = choose_track(info)
            if pick:
                kind, lang = pick
                self._run(
                    ["--skip-download", "--no-playlist", "--write-subs" if kind == "manual" else "--write-auto-subs",
                     "--sub-langs", lang, "--sub-format", "vtt", "--sleep-subtitles", "1", "-o", str(work / "%(id)s")],
                    url,
                )
                for vtt in sorted(work.glob("*.vtt")):
                    segs = subtitles.read(vtt)
                    if segs:
                        return segs, max(duration, segs[-1].end), kind
            # no usable captions: take the audio, which pyav reads as it is (no ffmpeg needed)
            self._run(["-f", "bestaudio/best", "--no-playlist", "-o", str(work / "%(id)s.%(ext)s")], url)
            audio = next((p for p in work.iterdir() if p.suffix != ".vtt"), None)
            if audio is None:
                raise FetchError("no captions and no audio could be downloaded")
            segs, dur = transcriber.transcribe(str(audio))
            return segs, dur, "whisper"
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _entry(e: dict) -> dict:
    extractor = (e.get("extractor_key") or e.get("ie_key") or "web").lower()
    vid = e.get("id") or e.get("webpage_url") or e.get("url")
    url = e.get("webpage_url") or e.get("url")
    if extractor == "youtube" and e.get("id"):
        url = f"https://www.youtube.com/watch?v={e['id']}"  # drop &list=, &t=, &si= so one video is one source
    return {"key": f"{extractor}:{vid}", "url": url, "title": e.get("title")}
