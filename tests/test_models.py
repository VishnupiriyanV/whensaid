import socket
import threading
import time

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from whensaid import models, server


def test_ipv4_only_filters_lookups_and_restores(monkeypatch):
    seen = []

    def fake(host, port, family=0, type=0, proto=0, flags=0):
        seen.append(family)
        return [(family or socket.AF_INET6, 1, 6, "", ("::1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    with models.ipv4_only():
        socket.getaddrinfo("example.com", 443)
        socket.getaddrinfo("example.com", 443, socket.AF_INET6)  # an explicit ask is left alone
    assert seen == [socket.AF_INET, socket.AF_INET6]
    assert socket.getaddrinfo is fake  # put back, also after an error
    with pytest.raises(RuntimeError), models.ipv4_only():
        raise RuntimeError
    assert socket.getaddrinfo is fake


def test_a_cut_off_whisper_download_is_not_present(tmp_path, monkeypatch):
    monkeypatch.setenv("WHENSAID_HOME", str(tmp_path))
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")  # the snapshot folder exists, the weights never arrived
    import faster_whisper.utils as fw

    monkeypatch.setattr(fw, "download_model", lambda *a, **k: str(snap))
    assert models.present("whisper") is False
    (snap / "model.bin").write_bytes(b"x")
    assert models.present("whisper") is True
    with pytest.raises(ValueError):
        models.present("nonsense")


def test_a_model_that_is_not_on_disk_is_reported_as_missing_not_as_an_error(tmp_path, monkeypatch):
    # fastembed raises ValueError("Could not load model ... from any source") for a missing model;
    # that must come out as False so the download starts, not as a crash that blocks it
    monkeypatch.setenv("WHENSAID_HOME", str(tmp_path))
    from whensaid import embed

    def load(self):
        raise ValueError("Could not load model BAAI/bge-small-en-v1.5 from any source.")

    monkeypatch.setattr(embed.Embedder, "_load", load)
    assert models.present("embedder") is False


def test_ensure_downloads_only_what_is_missing_and_respects_offline(monkeypatch):
    have = {"whisper": True, "embedder": False, "reranker": False}
    got = []
    monkeypatch.setattr(models, "present", lambda k: have[k])
    monkeypatch.setattr(models, "download", lambda kinds, log=None: got.append(list(kinds)))
    models.ensure(("whisper", "embedder", "reranker"))
    assert got == [["embedder", "reranker"]]
    got.clear()
    monkeypatch.setenv("WHENSAID_OFFLINE", "1")
    with pytest.raises(models.ModelsMissing) as e:
        models.ensure(("embedder",))
    assert got == [] and "models pull" in str(e.value)

    monkeypatch.delenv("WHENSAID_OFFLINE")

    def boom(kinds, log=None):
        raise OSError("name resolution failed")

    monkeypatch.setattr(models, "download", boom)
    with pytest.raises(models.ModelsMissing) as e:
        models.ensure(("embedder",))
    assert "name resolution" in str(e.value) and "models pull" in str(e.value)


def test_search_models_skip_the_reranker_when_it_is_switched_off(monkeypatch):
    monkeypatch.delenv("WHENSAID_RERANK", raising=False)
    monkeypatch.delenv("HEARSAY_RERANK", raising=False)
    assert models.wanted_for_search() == ("embedder", "reranker")
    monkeypatch.setenv("WHENSAID_RERANK", "0")
    assert models.wanted_for_search() == ("embedder",)


def test_prep_downloads_in_the_background_and_reports(monkeypatch):
    on_disk = {"embedder": False, "reranker": False}
    gate = threading.Event()

    def download(kinds, log=None):
        gate.wait(5)
        for k in kinds:
            on_disk[k] = True

    monkeypatch.setattr(models, "present", lambda k: on_disk[k])
    monkeypatch.setattr(models, "download", download)
    p = models.Prep(("embedder", "reranker")).start()
    time.sleep(0.1)
    assert not p.ready() and p.state["embedder"] == "downloading"
    assert "still downloading" in p.message() and "embedder" in p.message()
    gate.set()
    p.thread.join(5)
    assert p.ready() and p.describe() == {"ready": True, "models": {"embedder": "ready", "reranker": "ready"}, "error": None}


def test_prep_failure_names_the_fix(monkeypatch):
    monkeypatch.setattr(models, "present", lambda k: False)

    def boom(kinds, log=None):
        raise OSError("no route to host")

    monkeypatch.setattr(models, "download", boom)
    p = models.Prep(("embedder",)).start()
    p.thread.join(5)
    assert not p.ready() and p.state["embedder"] == "failed"
    assert "no route to host" in p.message() and "whensaid models pull" in p.message()


def test_server_waits_for_models_and_only_then_builds_a_searcher_with_the_reranker(tmp_path, monkeypatch):
    from whensaid import db, indexer
    from whensaid import embed, rerank

    monkeypatch.setenv("WHENSAID_HOME", str(tmp_path))
    db.connect(tmp_path / "whensaid.db").close()
    server._state.clear()
    built = []

    class FakeRerank:
        def score(self, q, texts):
            return [0.0] * len(texts)

    monkeypatch.setattr(embed, "Embedder", lambda *a, **k: object())
    monkeypatch.setattr(rerank, "load_reranker", lambda: built.append("reranker") or FakeRerank())
    monkeypatch.setattr(server, "Searcher", lambda conn, e, reranker=None, **k: built.append(("searcher", reranker is not None)) or object())

    prep = models.Prep(("embedder", "reranker"))
    prep.state["embedder"] = "ready"
    prep.state["reranker"] = "downloading"
    server._state["prep"] = prep
    with pytest.raises(ToolError) as e:
        server._searcher()
    assert "still downloading" in str(e.value) and built == []  # nothing was built around the missing model
    assert server.index_status()["models"]["models"]["reranker"] == "downloading"

    prep.state["reranker"] = "ready"
    server._searcher()
    assert built == ["reranker", ("searcher", True)]  # built once, with its reranker
    server._searcher()
    assert built == ["reranker", ("searcher", True)]  # and kept

    server._state.clear()
    prep = models.Prep(("embedder", "reranker"))
    prep.state.update(embedder="ready", reranker="failed")
    prep.error = "reranker: no route to host"
    server._state["prep"] = prep
    with pytest.raises(ToolError) as e:
        server._searcher()
    assert "models pull" in str(e.value)  # a failed reranker is an error, not a silent search without it
    server._state.clear()
