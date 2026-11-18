from types import SimpleNamespace as NS

from whensaid.transcribe import split_on_gaps


def w(word, start, end):
    return NS(word=word, start=start, end=end)


def test_stray_word_is_split_from_later_speech():
    # the real case: "Randall" at 2s, rest of the sentence at 146s
    words = [w("Randall", 2.0, 2.4), w(" is", 146.8, 146.9), w(" more", 147.0, 147.2)]
    out = split_on_gaps(words)
    assert [(s.start, s.text) for s in out] == [(2.0, "Randall"), (146.8, "is more")]


def test_normal_sentence_stays_whole():
    words = [w(" Ask", 1.0, 1.2), w(" not", 1.3, 1.5), w(" what", 1.6, 1.8)]
    out = split_on_gaps(words)
    assert len(out) == 1
    assert out[0].text == "Ask not what"
    assert (out[0].start, out[0].end) == (1.0, 1.8)


def test_no_words_falls_back_to_segment_text():
    fb = NS(start=3.0, end=5.0, text="  hello  ")
    assert split_on_gaps([], fallback=fb)[0].text == "hello"
    assert split_on_gaps([], fallback=NS(start=0, end=1, text="  ")) == []


def test_no_cuda_extra_installed_does_not_crash(monkeypatch):
    import importlib.util

    from whensaid import cudalibs

    real = importlib.util.find_spec

    def find_spec(name, *a):  # what python does when the parent package "nvidia" does not exist
        if name.startswith("nvidia"):
            raise ModuleNotFoundError("No module named 'nvidia'")
        return real(name, *a)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    cudalibs.preload()  # used to raise


def test_gpu_without_its_libraries_falls_back_to_cpu(monkeypatch, capsys):
    import sys
    import types

    from whensaid import cudalibs, transcribe

    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1))
    monkeypatch.setattr(cudalibs, "libs_load", lambda: False)
    assert transcribe.pick_device()[:2] == ("cpu", "int8")
    assert "whensaid[cuda]" in capsys.readouterr().err
    monkeypatch.setattr(cudalibs, "libs_load", lambda: True)
    assert transcribe.pick_device()[:2] == ("cuda", "float16")
