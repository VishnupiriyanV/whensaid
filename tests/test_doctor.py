import io

from whensaid import db, doctor, models


def fake_models(monkeypatch, have=True):
    monkeypatch.setattr(models, "present", lambda k: have)


def names(results):
    return {n: (s, d, f) for s, n, d, f in results}


def test_doctor_on_a_fresh_machine_warns_but_does_not_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("WHENSAID_HOME", str(tmp_path))
    fake_models(monkeypatch, have=False)
    r = names(doctor.checks())
    assert r["whisper model"][0] == "warn" and "first use" in r["whisper model"][1]
    assert r["whisper model"][2] == "whensaid models pull"
    assert r["index"][0] == "ok" and "none yet" in r["index"][1]
    out = io.StringIO()
    assert doctor.run(out) == 0  # warnings only
    assert "warnings" in out.getvalue() and "-> whensaid models pull" in out.getvalue()


def test_doctor_flags_an_index_built_with_another_model(tmp_path, monkeypatch):
    monkeypatch.setenv("WHENSAID_HOME", str(tmp_path))
    fake_models(monkeypatch)
    conn = db.connect(tmp_path / "whensaid.db")
    db.set_meta(conn, "embed_model", "some/other-model")
    conn.commit()
    conn.close()
    r = names(doctor.checks())
    assert r["embedding model"][0] == "fail" and r["embedding model"][2] == "whensaid rechunk"
    assert doctor.run(io.StringIO()) == 1


def test_doctor_reports_gpu_without_its_libraries(tmp_path, monkeypatch):
    import sys
    import types

    from whensaid import cudalibs

    monkeypatch.setenv("WHENSAID_HOME", str(tmp_path))
    fake_models(monkeypatch)
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1))
    monkeypatch.setattr(cudalibs, "libs_load", lambda: False)
    r = names(doctor.checks())
    assert r["gpu"][0] == "warn" and "whensaid[cuda]" in r["gpu"][2]
