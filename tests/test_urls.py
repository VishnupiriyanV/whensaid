import sqlite3

import numpy as np
import pytest

from hearsay import db, export, fetch, indexer, paths, server, urls
from hearsay.chunk import Segment
from hearsay.search import Searcher
from hearsay.util import deep_link

LOG = open("/dev/null", "w")



def vid(i, title=None):
    return {"key": f"youtube:{i}", "url": f"https://www.youtube.com/watch?v={i}", "title": title or f"Video {i}"}


class FakeFetcher:
    def __init__(self, playlists=None, captions=None):
        self.playlists = playlists or {}
        self.captions = captions or {}
        self.listed = []
        self.acquired = []

    def entries(self, url, limit=200, no_playlist=False):
        self.listed.append(url)
        if url in self.playlists:
            return "playlist", "A playlist", self.playlists[url][:limit]
        i = url.split("v=")[1].split("&")[0]
        return "video", f"Video {i}", [vid(i)]

    def acquire(self, url, transcriber):
        self.acquired.append(url)
        how = self.captions.get(url, "manual")
        if how == "boom":
            raise fetch.FetchError("ERROR: Sign in to confirm you're not a bot")
        if how == "whisper":
            segs, dur = transcriber.transcribe("audio.webm")
            return segs, dur, "whisper"
        return [Segment(0, 30, "gradient descent is neat"), Segment(754, 760, "and then the lighthouse")], 800.0, how


class Emb:
    model_name = "fake"

    def encode(self, texts):
        return np.array([[1.0, 0.0, float("lighthouse" in t), 0.1] for t in texts], dtype=np.float32)


class Tr:
    def transcribe(self, path):
        return [Segment(0, 5, "spoken words")], 5.0


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(indexer, "WEB_PAUSE", 0)


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "u.db")


def test_add_video_and_playlist_dedupe_the_same_video(conn):
    f = FakeFetcher(playlists={"https://y/p": [vid("a"), vid("b")]})
    assert urls.add_url(conn, f, "https://www.youtube.com/watch?v=a&t=30s", log=LOG)["new"] == 1
    r = urls.add_url(conn, f, "https://y/p", log=LOG)
    assert (r["found"], r["new"]) == (2, 1)  # "a" is already known, reached another way
    assert conn.execute("select count(*) from sources where kind = 'url'").fetchone()[0] == 2
    assert [x["url"] for x in conn.execute("select url from feeds")] == ["https://y/p"]


def test_index_web_videos_records_how_and_survives_failures(conn):
    f = FakeFetcher(
        playlists={"https://y/p": [vid("a"), vid("b"), vid("c")]},
        captions={"https://www.youtube.com/watch?v=b": "whisper", "https://www.youtube.com/watch?v=c": "boom"},
    )
    urls.add_url(conn, f, "https://y/p", log=LOG)
    counts = indexer.run(conn, Tr(), Emb(), log=LOG, fetcher=f)
    assert counts == {"done": 2, "no_audio": 0, "failed": 1}
    got = {r["path"]: (r["captions"], r["status"]) for r in conn.execute("select * from sources")}
    assert got == {"youtube:a": ("manual", "done"), "youtube:b": ("whisper", "done"), "youtube:c": (None, "failed")}
    assert "not a bot" in conn.execute("select error from sources where path = 'youtube:c'").fetchone()[0]
    # done videos are never fetched again, the failed one is retried
    f.acquired.clear()
    indexer.run(conn, Tr(), Emb(), log=LOG, fetcher=f)
    assert f.acquired == ["https://www.youtube.com/watch?v=c"]


def test_web_videos_are_not_pruned_or_exported_beside(conn, tmp_path):
    f = FakeFetcher()
    urls.add_url(conn, f, "https://www.youtube.com/watch?v=a", log=LOG)
    indexer.run(conn, Tr(), Emb(), log=LOG, fetcher=f)
    conn.execute("insert into roots values (?)", (str(tmp_path),))  # some folder is watched, the key is not under it
    (tmp_path / "x.mp4").write_bytes(b"x")
    assert indexer.prune(conn, LOG) == 0
    written, skipped = export.export_all(conn, None, "txt")
    assert written == [] and skipped == []
    written, _ = export.export_all(conn, tmp_path / "out", "txt")
    assert [p.relative_to(tmp_path) for p in written] == [__import__("pathlib").Path("out/urls/youtube_a.txt")]
    assert "[0:12:34] and then the lighthouse" in written[0].read_text()


def test_feed_refresh_adds_new_videos_but_not_too_often(conn):
    f = FakeFetcher(playlists={"https://y/p": [vid("a")]})
    urls.add_url(conn, f, "https://y/p", log=LOG)
    f.playlists["https://y/p"] = [vid("a"), vid("b")]
    assert urls.refresh_feeds(conn, f, LOG) == 0  # checked a moment ago
    assert urls.refresh_feeds(conn, f, LOG, force=True) == 1
    conn.execute("update feeds set checked_at = 0")
    f.playlists["https://y/p"] = [vid("a"), vid("b"), vid("c")]
    assert urls.refresh_feeds(conn, f, LOG) == 1  # an hour later


def test_remove_playlist_or_single_video(conn):
    f = FakeFetcher(playlists={"https://y/p": [vid("a"), vid("b")]})
    urls.add_url(conn, f, "https://y/p", log=LOG)
    urls.add_url(conn, f, "https://www.youtube.com/watch?v=z", log=LOG)
    indexer.run(conn, Tr(), Emb(), log=LOG, fetcher=f)
    assert urls.remove(conn, "https://www.youtube.com/watch?v=z") == 1
    assert urls.remove(conn, "https://y/p") == 2
    assert [conn.execute(f"select count(*) from {t}").fetchone()[0] for t in ("sources", "segments", "chunks", "feeds")] == [0, 0, 0, 0]


def test_search_returns_title_and_a_link_that_opens_at_the_moment(conn, monkeypatch, tmp_path):
    f = FakeFetcher()
    urls.add_url(conn, f, "https://www.youtube.com/watch?v=abc123", log=LOG)
    indexer.run(conn, Tr(), Emb(), log=LOG, fetcher=f)
    hit = Searcher(conn, Emb(), query_prefix="").search("lighthouse", mode="keyword")[0]
    assert (hit.title, hit.url) == ("Video abc123", "https://www.youtube.com/watch?v=abc123")
    conn.commit()
    monkeypatch.setenv("HEARSAY_HOME", str(tmp_path))
    server._state.clear()
    server._state["conn"] = db.connect_readonly(tmp_path / "u.db")
    server._state["searcher"] = Searcher(server._state["conn"], Emb(), query_prefix="")
    out = server.search("lighthouse")["results"][0]
    assert out["file"] == "Video abc123" and out["path"] == "youtube:abc123"
    assert out["link"] == "https://www.youtube.com/watch?v=abc123&t=754s"
    tr = server.get_transcript("youtube:abc123", start=700)  # by key, as search returns it
    assert tr["title"] == "Video abc123" and tr["segments"][0]["text"] == "and then the lighthouse"
    assert server.get_transcript("https://www.youtube.com/watch?v=abc123")["path"] == "youtube:abc123"  # or by url
    server._state.clear()


def test_an_index_from_before_web_videos_still_searches_read_only(tmp_path):
    old = sqlite3.connect(tmp_path / "old.db")
    old.executescript(db.SCHEMA)  # the old schema: no kind/url/title columns
    old.execute("insert into sources (id, path, status) values (1, '/a.mp4', 'done')")
    old.execute("insert into chunks (id, source_id, start, end, text, embedding) values (1, 1, 0, 5, 'the lighthouse', ?)",
                (Emb().encode(["the lighthouse"])[0].tobytes(),))
    old.execute("insert into chunks_fts (rowid, text) values (1, 'the lighthouse')")
    old.execute("insert into segments (source_id, start, end, text) values (1, 0, 5, 'the lighthouse')")
    old.commit()
    old.close()
    ro = db.connect_readonly(tmp_path / "old.db")
    assert "url" not in {r[1] for r in ro.execute("pragma table_info(sources)")}
    hit = Searcher(ro, Emb(), query_prefix="").search("lighthouse")[0]
    assert (hit.url, hit.title) == (None, None)
    server._state.clear()
    server._state["conn"] = ro
    assert server.get_transcript("a.mp4")["segments"][0]["text"] == "the lighthouse"
    assert server.list_sources()["sources"][0]["url"] is None
    server._state.clear()
    db.connect(tmp_path / "old.db")  # and a writer upgrades it in place
    assert db.connect(tmp_path / "old.db").execute("select kind from sources").fetchone()[0] == "file"


def test_choose_track_prefers_manual_then_plain_auto():
    assert fetch.choose_track({"subtitles": {"fr": [], "en-GB": [], "en": []}, "automatic_captions": {"en": []}}) == ("manual", "en")
    assert fetch.choose_track({"subtitles": {"fr": []}, "automatic_captions": {"en-orig": [], "en": [], "de": []}}) == ("auto", "en")
    assert fetch.choose_track({"automatic_captions": {"en-orig": []}}) == ("auto", "en-orig")
    assert fetch.choose_track({"subtitles": {"de": []}, "automatic_captions": {"de": []}}) is None


def test_entry_keys_drop_playlist_and_time_noise():
    e = fetch._entry({"id": "abc", "extractor_key": "Youtube", "webpage_url": "https://www.youtube.com/watch?v=abc&list=PL1&t=30s", "title": "T"})
    assert e == {"key": "youtube:abc", "url": "https://www.youtube.com/watch?v=abc", "title": "T"}
    assert fetch._entry({"id": "9", "ie_key": "Vimeo", "url": "https://vimeo.com/9"})["key"] == "vimeo:9"


def test_deep_links():
    assert deep_link("https://www.youtube.com/watch?v=abc&list=x", 754.9) == "https://www.youtube.com/watch?v=abc&t=754s"
    assert deep_link("https://youtu.be/abc", 5) == "https://www.youtube.com/watch?v=abc&t=5s"
    assert deep_link("https://vimeo.com/123456", 61) == "https://vimeo.com/123456#t=61s"
    assert deep_link("https://example.com/talk.mp4", 61) is None and deep_link(None, 1) is None


def test_server_notices_an_index_upgraded_while_it_runs(tmp_path):
    old = sqlite3.connect(tmp_path / "m.db")
    old.executescript(db.SCHEMA)
    old.commit()
    old.close()
    ro = db.connect_readonly(tmp_path / "m.db")
    s = Searcher(ro, Emb(), query_prefix="")
    assert s.search("lighthouse") == []  # old schema, nothing indexed, no error
    w = db.connect(tmp_path / "m.db")  # a writer upgrades it, as `hearsay add-url` would
    f = FakeFetcher()
    urls.add_url(w, f, "https://www.youtube.com/watch?v=abc123", log=LOG)
    indexer.run(w, Tr(), Emb(), log=LOG, fetcher=f)
    w.commit()
    hit = s.search("lighthouse", mode="keyword")[0]  # same Searcher, same read-only connection
    assert (hit.title, hit.url) == ("Video abc123", "https://www.youtube.com/watch?v=abc123")


def test_remove_accepts_other_forms_of_the_same_video_url(conn):
    f = FakeFetcher()
    urls.add_url(conn, f, "https://www.youtube.com/watch?v=abc123", log=LOG)
    assert urls.remove(conn, "https://youtu.be/abc123") == 1
    urls.add_url(conn, f, "https://www.youtube.com/watch?v=abc123", log=LOG)
    assert urls.remove(conn, "https://www.youtube.com/watch?v=abc123&list=PL1&t=30s") == 1


def test_pause_only_between_web_videos(conn, monkeypatch):
    pauses = []
    monkeypatch.setattr(indexer, "WEB_PAUSE", 7)
    monkeypatch.setattr(indexer.time, "sleep", pauses.append)
    f = FakeFetcher(playlists={"https://y/p": [vid("a"), vid("b"), vid("c")]})
    urls.add_url(conn, f, "https://y/p", log=LOG)
    indexer.run(conn, Tr(), Emb(), log=LOG, fetcher=f)
    assert pauses == [7, 7]  # three videos, two gaps
