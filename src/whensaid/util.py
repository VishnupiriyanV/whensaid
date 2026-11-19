import re

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean_text(text: str | None) -> str | None:
    """Text that came from the web (a video title) with control characters removed, so a hostile title cannot
    carry terminal escape sequences into `whensaid status` or `search`."""
    if text is None:
        return None
    return _CONTROL.sub("", text).strip()


def is_web_url(url: str) -> bool:
    return re.match(r"https?://[^\s]+$", url or "", re.I) is not None


def canonical_url(url: str) -> str:
    """The form a youtube video is stored under (no list, t or si noise), others unchanged."""
    from urllib.parse import parse_qs, urlparse

    u = urlparse(url)
    host = u.netloc.lower().removeprefix("www.").removeprefix("m.")
    if host == "youtube.com" and u.path == "/watch" and "v" in parse_qs(u.query):
        return f"https://www.youtube.com/watch?v={parse_qs(u.query)['v'][0]}"
    if host == "youtu.be" and u.path.strip("/"):
        return f"https://www.youtube.com/watch?v={u.path.strip('/')}"
    return url


def deep_link(url: str | None, seconds: float) -> str | None:
    """A link that opens the video at that second, for sites where we know the format."""
    if not url:
        return None
    from urllib.parse import parse_qs, urlparse

    u = urlparse(url)
    host = u.netloc.lower().removeprefix("www.").removeprefix("m.")
    t = int(max(seconds, 0))
    if host == "youtube.com" and u.path == "/watch" and "v" in parse_qs(u.query):
        return f"https://www.youtube.com/watch?v={parse_qs(u.query)['v'][0]}&t={t}s"
    if host == "youtu.be" and u.path.strip("/"):
        return f"https://www.youtube.com/watch?v={u.path.strip('/')}&t={t}s"
    if host == "vimeo.com" and u.path.strip("/").isdigit():
        return f"https://vimeo.com/{u.path.strip('/')}#t={t}s"
    return None


def timecode(seconds: float, srt: bool = False) -> str:
    """h:mm:ss for people, or hh:mm:ss,mmm for srt files."""
    if srt:
        ms = int(round(seconds * 1000))
        return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
