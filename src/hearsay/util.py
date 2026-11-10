def timecode(seconds: float, srt: bool = False) -> str:
    """h:mm:ss for people, or hh:mm:ss,mmm for srt files."""
    if srt:
        ms = int(round(seconds * 1000))
        return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
