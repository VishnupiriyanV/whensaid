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
import time
from pathlib import Path

from . import subtitles
from .paths import home


class FetchError(Exception):
    pass


class Skip(FetchError):
    """This video will not work however often we try: live, private, another language."""


LIVE = {"is_live": "a live stream", "is_upcoming": "not published yet", "post_live": "a stream that just ended, not processed yet"}
# yt-dlp's wording for things that will not fix themselves. Anything else is treated as
# temporary (network, rate limit, bot check) and retried on the next run.
PERMANENT = ("private video", "video unavailable", "video is unavailable", "is not available", "has been removed", "members-only", "members only",
             "confirm your age", "account associated with this video has been terminated", "not available in your country")


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


def is_english(code: str) -> bool:
    return code == "en" or code.startswith(("en-", "en_"))


def spoken_language(info: dict) -> str | None:
    """What the site says is spoken. Its `language` field when there is one. Otherwise the
    "-orig" caption tracks, but only when they agree: a video with dubbed audio lists one per
    dub (22 on a real NASA video), which says nothing about the original."""
    if info.get("language"):
        return info["language"]
    orig = {k[:-5] for k in (info.get("automatic_captions") or {}) if k.endswith("-orig")}
    if any(is_english(k) for k in orig):
        return "en"
    return orig.pop() if len(orig) == 1 else None


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


FIRST_TRY = 45  # seconds before deciding the connection is stuck and trying again over IPv4


class YtDlp:
    def __init__(self, timeout: int = 120, download_timeout: int = 1800):
        self.timeout = timeout                    # listing a playlist, reading metadata, fetching captions
        self.download_timeout = download_timeout  # pulling the audio of a video without captions
        # set after a stall: networks with broken IPv6 hang python but not a browser. Remembered
        # for a day so each command doesn't sit through the same stall again.
        self._flag = home() / "ipv4"
        self.ipv4 = self._flag.exists() and time.time() - self._flag.stat().st_mtime < 86400

    def _run(self, args: list[str], url: str | None = None, timeout: int | None = None) -> str:
        # the socket timeout makes a dead connection fail in seconds instead of holding the whole run
        cmd = ytdlp_cmd() + ["--no-warnings", "--ignore-config", "--socket-timeout", "20", "--retries", "2", *args]
        if url:
            cmd += ["--", url]
        limit = timeout or self.timeout
        try:
            if self.ipv4:
                r = subprocess.run(cmd[:-2] + ["--force-ipv4"] + cmd[-2:] if url else cmd + ["--force-ipv4"],
                                   capture_output=True, text=True, timeout=limit)
            else:
                try:
                    # a healthy connection answers the first request well inside this
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=min(limit, FIRST_TRY))
                except subprocess.TimeoutExpired:
                    if limit <= FIRST_TRY:
                        raise
                    self.ipv4 = True
                    out = self._run(args, url, timeout)
                    self._flag.touch()  # only once IPv4 really did work
                    return out
        except subprocess.TimeoutExpired:
            raise FetchError(f"yt-dlp gave no answer in {limit} s (slow or blocked connection?)")
        if r.returncode != 0:
            errs = [ln for ln in r.stderr.splitlines() if ln.startswith("ERROR")]
            last = errs[-1] if errs else (r.stderr.strip().splitlines() or ["yt-dlp failed"])[-1]
            if any(p in last.lower() for p in PERMANENT):
                raise Skip(last[:400])
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
            if info.get("live_status") in LIVE:
                raise Skip(LIVE[info["live_status"]])  # its captions are a moving window, not a recording
            lang = spoken_language(info)
            if lang and not is_english(lang):
                # youtube offers a machine-translated English track for any language; that is not
                # what was said, and whisper here is English-only, so don't pretend
                raise Skip(f"not in English (original language {lang}); only English is supported so far")
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
            self._run(["-f", "bestaudio/best", "--no-playlist", "-o", str(work / "%(id)s.%(ext)s")], url,
                      timeout=self.download_timeout)
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
    return {"key": f"{extractor}:{vid}", "url": url, "title": e.get("title"), "live": LIVE.get(e.get("live_status"))}
