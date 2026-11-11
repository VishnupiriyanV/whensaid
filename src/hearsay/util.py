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
