from pathlib import Path

from hearsay import db, export
from hearsay.util import timecode


def test_timecode_formats():
    assert timecode(3725.9) == "1:02:05"
    assert timecode(3.5, srt=True) == "00:00:03,500"
    assert timecode(3725.9, srt=True) == "01:02:05,900"


SEGS = [{"start": 1.0, "end": 3.5, "text": "hello there"}, {"start": 65.2, "end": 70.0, "text": "gradient descent"}]


def test_txt_has_one_timestamped_line_per_segment():
    assert export.render_txt(SEGS) == "[0:00:01] hello there\n[0:01:05] gradient descent\n"


def test_srt():
    out = export.render_srt(SEGS)
    assert out.startswith("1\n00:00:01,000 --> 00:00:03,500\nhello there\n\n2\n")


def test_export_all_mirrors_paths_and_skips_unfinished(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    conn.execute("insert into sources (id, path, status) values (1, '/media/a/talk.mp4', 'done')")
    conn.execute("insert into sources (id, path, status) values (2, '/media/b/talk.mp4', 'done')")
    conn.execute("insert into sources (id, path, status) values (3, '/media/c/bad.mp4', 'failed')")
    for sid in (1, 2):
        conn.execute("insert into segments (source_id, start, end, text) values (?, 0, 2, ?)", (sid, f"file {sid}"))
    conn.commit()

    out = tmp_path / "out"
    files = export.export_all(conn, out)
    assert len(files) == 2  # the failed one is not exported
    # same file name in two folders does not collide
    assert (out / "media/a/talk.mp4.txt").read_text() == "[0:00:00] file 1\n"
    assert (out / "media/b/talk.mp4.txt").read_text() == "[0:00:00] file 2\n"


def test_beside_writes_next_to_media(tmp_path):
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"x")
    conn = db.connect(tmp_path / "t.db")
    conn.execute("insert into sources (id, path, status) values (1, ?, 'done')", (str(media),))
    conn.execute("insert into segments (source_id, start, end, text) values (1, 0, 2, 'hi')")
    conn.commit()
    export.export_all(conn, None, "srt")
    assert (tmp_path / "talk.srt").exists()
