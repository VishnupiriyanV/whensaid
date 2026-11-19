import os
import numpy as np

from whensaid import db, indexer
from whensaid.chunk import Segment
from whensaid.transcribe import NoAudio


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

    counts = indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=open(os.devnull, "w"))
    assert counts == {"done": 1, "no_audio": 1, "failed": 1, "skipped": 0}
    assert conn.execute("select count(*) from chunks").fetchone()[0] == 1
    hit = conn.execute("select rowid from chunks_fts where chunks_fts match 'gradient'").fetchall()
    assert len(hit) == 1

    # second run: the done and no_audio files are untouched, only the failed one retries
    counts = indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=open(os.devnull, "w"))
    assert counts == {"done": 0, "no_audio": 0, "failed": 1, "skipped": 0}

    # a changed file is picked up and its old rows replaced, not duplicated
    (media / "talk.mp4").write_bytes(b"y" * 20)
    indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=open(os.devnull, "w"))
    assert conn.execute("select count(*) from chunks").fetchone()[0] == 1
    assert conn.execute("select count(*) from chunks_fts").fetchone()[0] == 1


def test_second_indexer_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("WHENSAID_HOME", str(tmp_path))
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
    log = open(os.devnull, "w")
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
    log = open(os.devnull, "w")
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
    assert total == {"done": 2, "no_audio": 1, "failed": 1, "skipped": 0}
    assert sum(p.endswith("broken.wav") for p in seen) == 1  # tried once, not every round
    assert sum(p.endswith("later.mp4") for p in seen) == 1


def test_settle_skips_files_still_being_written(tmp_path):
    media = make_tree(tmp_path)
    conn = db.connect(tmp_path / "s.db")
    assert indexer.scan(conn, str(media), settle=3600) == []  # all modified just now
    assert len(indexer.scan(conn, str(media), settle=0)) == 3


def indexed(tmp_path, names=("talk.mp4", "other.mp4")):
    media = tmp_path / "media"
    media.mkdir()
    for n in names:
        (media / n).write_bytes(b"x" * 10)
    conn = db.connect(tmp_path / "p.db")
    conn.execute("insert into roots values (?)", (str(media),))
    log = open(os.devnull, "w")
    indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=log)
    return media, conn, log


def counts(conn):
    return [conn.execute(f"select count(*) from {t}").fetchone()[0] for t in ("sources", "segments", "chunks", "chunks_fts")]


def test_prune_removes_only_missing_files(tmp_path):
    media, conn, log = indexed(tmp_path)
    assert counts(conn) == [2, 4, 2, 2]
    (media / "talk.mp4").unlink()
    assert indexer.prune(conn, log) == 1
    assert counts(conn) == [1, 2, 1, 1]  # segments, chunks and fts rows went with it
    assert conn.execute("select path from sources").fetchone()[0].endswith("other.mp4")


def test_run_prunes_too(tmp_path):
    media, conn, log = indexed(tmp_path)
    (media / "other.mp4").unlink()
    indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=log)
    assert counts(conn)[0] == 1


def test_prune_leaves_everything_when_the_folder_is_gone_or_empty(tmp_path):
    media, conn, log = indexed(tmp_path)
    for f in media.iterdir():
        f.unlink()
    assert indexer.prune(conn, log) == 0  # mount point exists but holds nothing: looks unmounted
    media.rmdir()
    assert indexer.prune(conn, log) == 0  # folder itself missing
    assert counts(conn)[0] == 2


def test_prune_ignores_sources_outside_any_root(tmp_path):
    media, conn, log = indexed(tmp_path)
    conn.execute("insert into sources (path, status) values ('/nowhere/else.mp4', 'done')")
    assert indexer.prune(conn, log) == 0


def test_remove_root_only_touches_that_folder(tmp_path):
    media, conn, log = indexed(tmp_path)
    twin = tmp_path / "media2"  # shares a name prefix with "media"
    twin.mkdir()
    (twin / "keep.mp4").write_bytes(b"x" * 10)
    conn.execute("insert into roots values (?)", (str(twin),))
    indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=log)
    assert counts(conn)[0] == 3
    assert indexer.remove_root(conn, str(media)) == 2
    assert counts(conn) == [1, 2, 1, 1]
    assert [r[0] for r in conn.execute("select path from roots")] == [str(twin)]
    assert (media / "talk.mp4").exists()  # recordings are never deleted


def test_dangling_symlink_is_skipped(tmp_path):
    media, conn, log = indexed(tmp_path)
    (media / "gone.mp4").symlink_to(tmp_path / "nowhere.mp4")
    assert indexer.run(conn, FakeTranscriber(), FakeEmbedder(), log=log) == {"done": 0, "no_audio": 0, "failed": 0, "skipped": 0}


class Named(FakeEmbedder):
    def __init__(self, name):
        self.model_name = name


def test_index_refuses_a_different_embedding_model(tmp_path):
    import pytest
    from whensaid.search import Searcher

    media, conn, log = indexed(tmp_path)
    indexer.run(conn, FakeTranscriber(), Named("model-a"), log=log)  # first named run claims the index
    assert db.get_meta(conn, "embed_model") == "model-a"
    with pytest.raises(indexer.ModelChanged):
        indexer.run(conn, FakeTranscriber(), Named("model-b"), log=log)
    with pytest.raises(RuntimeError):
        Searcher(conn, Named("model-b"), query_prefix="").search("hello")
    indexer.rechunk(conn, Named("model-b"), log=log)  # the documented way out
    assert db.get_meta(conn, "embed_model") == "model-b"
    indexer.run(conn, FakeTranscriber(), Named("model-b"), log=log)


def test_unintelligible_audio_is_skipped_with_the_reason(tmp_path):
    from whensaid.transcribe import Unintelligible

    media = tmp_path / "media"
    media.mkdir()
    for n in ("english.mp4", "foreign.mp4"):
        (media / n).write_bytes(b"x")
    conn = db.connect(tmp_path / "q.db")
    conn.execute("insert into roots values (?)", (str(media),))
    log = open(os.devnull, "w")
    calls = []

    class T(FakeTranscriber):
        def transcribe(self, path):
            calls.append(path)
            if "foreign" in path:
                self.last_confidence = 0.46
                raise Unintelligible("transcript unreliable (mean word confidence 0.46): probably not English, or not speech")
            self.last_confidence = 0.97
            return super().transcribe(path)

    assert indexer.run(conn, T(), FakeEmbedder(), log=log) == {"done": 1, "no_audio": 0, "failed": 0, "skipped": 1}
    got = {os.path.basename(r["path"]): (r["status"], r["confidence"], r["error"]) for r in conn.execute("select * from sources")}
    assert got["english.mp4"] == ("done", 0.97, None)
    assert got["foreign.mp4"][0] == "skipped" and "not English" in got["foreign.mp4"][2]
    assert conn.execute("select count(*) from chunks").fetchone()[0] == 1  # nothing from the garbage transcript is searchable
    calls.clear()
    indexer.run(conn, T(), FakeEmbedder(), log=log)
    assert calls == []  # an unchanged skipped file is not transcribed again


def test_confidence_check_thresholds(monkeypatch):
    from types import SimpleNamespace as W

    import numpy as np

    from whensaid import transcribe

    def fake_pipe(p, n):
        words = [W(start=i * 0.5, end=i * 0.5 + 0.4, word=" w", probability=p) for i in range(n)]
        return SimpleNamespaceLike(words)

    class SimpleNamespaceLike:
        def __init__(self, words):
            self.words = words

        def transcribe(self, audio, **kw):
            return [W(words=self.words, start=0, end=1, text="x")], None

    monkeypatch.setattr("faster_whisper.audio.decode_audio", lambda path, sampling_rate: np.zeros(16000, dtype=np.float32))
    t = transcribe.Transcriber(device="cpu", compute_type="int8", model="x")
    t._pipe = fake_pipe(0.5, 100)
    try:
        t.transcribe("a.wav")
        assert False, "low confidence over many words should be refused"
    except transcribe.Unintelligible as e:
        assert "0.50" in str(e)
    t._pipe = fake_pipe(0.5, 10)  # too few words to judge
    assert len(t.transcribe("a.wav")[0]) == 1 and t.last_confidence == 0.5
    t._pipe = fake_pipe(0.95, 100)
    assert t.transcribe("a.wav")[0] and abs(t.last_confidence - 0.95) < 1e-6
    monkeypatch.setenv("WHENSAID_MIN_CONFIDENCE", "0")
    t._pipe = fake_pipe(0.5, 100)
    assert t.transcribe("a.wav")[0]  # check switched off


def test_long_files_report_progress_in_tens_of_percent(tmp_path):
    import io

    media = tmp_path / "m"
    media.mkdir()
    for n in ("big.mp4", "tiny.mp4"):
        (media / n).write_bytes(b"x")
    conn = db.connect(tmp_path / "pg.db")
    conn.execute("insert into roots values (?)", (str(media),))

    class Slow(FakeTranscriber):
        on_progress = None

        def transcribe(self, path):
            total = 7200.0 if path.endswith("big.mp4") else 60.0
            for t in range(0, int(total), 30):  # a segment every 30 s
                self.on_progress(t + 30, total)
            return super().transcribe(path)

    log = io.StringIO()
    indexer.run(conn, Slow(), FakeEmbedder(), log=log)
    lines = [ln.strip() for ln in log.getvalue().splitlines() if "%" in ln]
    assert [ln.split("%")[0] for ln in lines] == [str(p) for p in range(0, 101, 10)]  # 0,10,...,100 once each
    assert lines[5].endswith("1:00:00 of 2:00:00") and len(lines) == 11  # nothing at all for the 60 s file
