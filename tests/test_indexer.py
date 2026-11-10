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
