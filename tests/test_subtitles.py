import numpy as np

from whensaid import db, indexer, subtitles

SRT = """1
00:00:01,000 --> 00:00:03,500
Hello <i>there</i>,
how are you?

2
00:01:05,200 --> 00:01:08,000
{\\an8}Gradient &amp; descent
"""

VTT = """WEBVTT
Kind: captions
Language: en

NOTE this is a comment
and so is this

STYLE
::cue { color: red }

00:01.000 --> 00:03.000 align:start position:0%
<c.colorE5E5E5>first</c> line

00:03.000 --> 00:05.000
first line

1:02:03.500 --> 1:02:05.000
much later
"""


def test_srt_markup_entities_and_multiline():
    segs = subtitles.parse(SRT)
    assert [(s.start, s.end, s.text) for s in segs] == [
        (1.0, 3.5, "Hello there, how are you?"),
        (65.2, 68.0, "Gradient & descent"),
    ]


def test_vtt_header_notes_settings_and_repeats():
    segs = subtitles.parse(VTT)
    assert [(s.start, s.end, s.text) for s in segs] == [(1.0, 5.0, "first line"), (3723.5, 3725.0, "much later")]


def test_junk_and_empty_input():
    assert subtitles.parse("") == []
    assert subtitles.parse("not subtitles at all\n\njust text") == []
    assert subtitles.parse("1\n00:00:05,000 --> 00:00:01,000\nbackwards\n") == []


def test_sidecar_lookup_and_encodings(tmp_path):
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"x")
    assert subtitles.find_sidecar(str(media)) is None
    side = tmp_path / "talk.en.srt"
    side.write_bytes("﻿1\n00:00:01,000 --> 00:00:02,000\ncafé\n".encode("utf-8"))
    assert subtitles.find_sidecar(str(media)) == side
    assert subtitles.read(side)[0].text == "café"
    side.write_bytes("1\n00:00:01,000 --> 00:00:02,000\ncafé\n".encode("latin-1"))
    assert subtitles.read(side)[0].text == "café"


class Boom:
    def transcribe(self, path):
        raise AssertionError("should not transcribe a file that has subtitles")


class Fake:
    def transcribe(self, path):
        from whensaid.chunk import Segment
        return [Segment(0, 3, "from whisper")], 3.0


class Emb:
    def encode(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def test_index_uses_subtitles_only_when_asked_and_present(tmp_path):
    media = tmp_path / "m"
    media.mkdir()
    for n in ("with.mp4", "without.mp4"):
        (media / n).write_bytes(b"not real media")
    (media / "with.srt").write_text(SRT)
    conn = db.connect(tmp_path / "s.db")
    conn.execute("insert into roots values (?)", (str(media),))
    log = open("/dev/null", "w")

    class Either:
        def transcribe(self, path):
            if path.endswith("with.mp4"):
                raise AssertionError("should not transcribe a file that has subtitles")
            return Fake().transcribe(path)

    assert indexer.run(conn, Either(), Emb(), log=log, use_subtitles=True) == {"done": 2, "no_audio": 0, "failed": 0, "skipped": 0}
    texts = [r[0] for r in conn.execute(
        "select group_concat(text, ' ') from segments s join sources o on o.id = s.source_id group by o.path order by o.path")]
    assert "Gradient & descent" in texts[0] and texts[1] == "from whisper"
    dur = conn.execute("select duration from sources where path like '%with.mp4'").fetchone()[0]
    assert dur == 68.0  # not real media, so the duration falls back to the last cue

    # default behaviour is untouched: without the flag the subtitle file is ignored
    conn2 = db.connect(tmp_path / "s2.db")
    conn2.execute("insert into roots values (?)", (str(media),))
    indexer.run(conn2, Fake(), Emb(), log=log)
    assert conn2.execute("select count(*) from segments where text = 'from whisper'").fetchone()[0] == 2


def test_rolling_auto_captions_from_youtube():
    # structure copied from a real auto-caption file (own text): whitespace-only lines inside
    # cues, a blank first line, every cue repeating the previous line, 10 ms transition cues
    from pathlib import Path

    segs = subtitles.parse((Path(__file__).parent / "fixtures" / "autocaptions.vtt").read_text())
    assert [(s.start, s.end, s.text) for s in segs] == [
        (0.8, 5.49, "Welcome to the tour of the harbor"),
        (5.49, 18.98, "First we visit the lighthouse"),
        (42.44, 46.48, "the keeper lives alone here"),
        (46.48, 50.4, "and tends the lamp"),
    ]


def test_srt_with_whitespace_only_separator_lines():
    srt = "1\n00:00:01,000 --> 00:00:02,000\nfirst\n \n2\n00:00:03,000 --> 00:00:04,000\nsecond\n  \n3\n00:00:05,000 --> 00:00:06,000\nthird"
    assert [s.text for s in subtitles.parse(srt)] == ["first", "second", "third"]
