import numpy as np

from hearsay import db, indexer
from hearsay.chunk import Segment
from hearsay.transcribe import NoAudio


class FakeTranscriber:
    def transcribe(self, path):
        if "silent" in path:
            raise NoAudio(path)
        if "broken" in path:
            raise RuntimeError("decode blew up")
        return [Segment(0, 4, "hello there"), Segment(4, 9, "gradient descent is neat")], 9.0


class FakeEmbedder:
    def encode(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def make_tree(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    for name in ("talk.mp4", "silent.mkv", "broken.wav", "notes.txt"):
        (media / name).write_bytes(b"x" * 10)
    return media


def test_run_and_reindex(tmp_path):
    media = make_tree(tmp_path)
    conn = db.connect(tmp_path / "t.db")
    conn.execute("insert into roots values (?)", (str(media),))

    counts = indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=open("/dev/null", "w"))
    assert counts == {"done": 1, "no_audio": 1, "failed": 1}
    assert conn.execute("select count(*) from chunks").fetchone()[0] == 1
    hit = conn.execute("select rowid from chunks_fts where chunks_fts match 'gradient'").fetchall()
    assert len(hit) == 1

    # second run: the done and no_audio files are untouched, only the failed one retries
    counts = indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=open("/dev/null", "w"))
    assert counts == {"done": 0, "no_audio": 0, "failed": 1}

    # a changed file is picked up and its old rows replaced, not duplicated
    (media / "talk.mp4").write_bytes(b"y" * 20)
    indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=open("/dev/null", "w"))
    assert conn.execute("select count(*) from chunks").fetchone()[0] == 1
    assert conn.execute("select count(*) from chunks_fts").fetchone()[0] == 1


def test_second_indexer_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARSAY_HOME", str(tmp_path))
    with indexer.index_lock():
        try:
            with indexer.index_lock():
                assert False, "should not get the lock twice"
        except indexer.AlreadyRunning:
            pass


def test_rechunk_rebuilds_from_segments(tmp_path):
    media = make_tree(tmp_path)
    conn = db.connect(tmp_path / "t.db")
    conn.execute("insert into roots values (?)", (str(media),))
    log = open("/dev/null", "w")
    indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=log)
    conn.execute("delete from chunks")  # pretend the chunks are stale or missing
    conn.execute("delete from chunks_fts")
    conn.commit()

    assert indexer.rechunk(conn, FakeEmbedder(), log=log) == 1
    assert conn.execute("select count(*) from chunks").fetchone()[0] == 1
    assert len(conn.execute("select rowid from chunks_fts where chunks_fts match 'gradient'").fetchall()) == 1


def test_watch_picks_up_new_files_and_does_not_retry_failures(tmp_path):
    media = make_tree(tmp_path)
    conn = db.connect(tmp_path / "w.db")
    conn.execute("insert into roots values (?)", (str(media),))
    log = open("/dev/null", "w")
    calls = {"n": 0}

    def sleep(_):
        calls["n"] += 1
        if calls["n"] == 1:
            (media / "later.mp4").write_bytes(b"z" * 10)  # shows up between scans

    seen = []

    class Counting(FakeTranscriber):
        def transcribe(self, path):
            seen.append(path)
            return super().transcribe(path)

    total = indexer.watch(conn, Counting(), FakeEmbedder(), log=log, interval=0, settle=0, rounds=3, sleep=sleep)
    assert total == {"done": 2, "no_audio": 1, "failed": 1}
    assert sum(p.endswith("broken.wav") for p in seen) == 1  # tried once, not every round
    assert sum(p.endswith("later.mp4") for p in seen) == 1


def test_settle_skips_files_still_being_written(tmp_path):
    media = make_tree(tmp_path)
    conn = db.connect(tmp_path / "s.db")
    assert indexer.scan(conn, str(media), settle=3600) == []  # all modified just now
    assert len(indexer.scan(conn, str(media), settle=0)) == 3
