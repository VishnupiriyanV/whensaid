from hearsay.chunk import Segment, make_chunks


def segs(n, length=5.0):
    return [Segment(i * length, (i + 1) * length, f"line {i}") for i in range(n)]


def test_empty():
    assert make_chunks([]) == []


def test_short_input_is_one_chunk():
    out = make_chunks(segs(3))
    assert len(out) == 1
    assert out[0].start == 0 and out[0].end == 15
    assert out[0].text == "line 0 line 1 line 2"


def test_windows_overlap_and_cover_everything():
    out = make_chunks(segs(40), target=45, overlap=6)  # 200s of speech
    assert len(out) > 1
    assert out[0].start == 0
    assert out[-1].end == 200
    for a, b in zip(out, out[1:]):
        assert b.start < a.end  # overlap
        assert b.start > a.start  # but always moves forward


def test_no_chunk_far_over_target():
    for c in make_chunks(segs(60), target=45, overlap=6):
        assert c.end - c.start <= 45 + 6 + 5
