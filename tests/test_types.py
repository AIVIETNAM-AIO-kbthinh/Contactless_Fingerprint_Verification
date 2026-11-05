import json

from fpsys.types import CaptureError, ErrorCode, Minutia, Template


def _template(embedding):
    mins = [Minutia(10.5, 20.25, 1.2345678901234, "ending", 0.8),
            Minutia(100.0, 3.0, 6.0, "bifurcation", 0.1)]
    return Template(mins, (320, 240), 9.03, embedding=embedding,
                    meta={"cfg_hash": "abcd1234", "timing_ms": {"total": 12.5}})


def test_roundtrip_without_embedding():
    t = _template(None)
    assert Template.from_dict(t.to_dict()) == t
    assert Template.from_dict(json.loads(json.dumps(t.to_dict()))) == t


def test_roundtrip_with_embedding():
    t = _template([0.1, -0.25, 3.0])
    back = Template.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back == t
    assert back.embedding == [0.1, -0.25, 3.0]
    assert isinstance(back.shape, tuple)


def test_capture_error_message():
    e = CaptureError(ErrorCode.BLURRY, "x")
    assert e.code is ErrorCode.BLURRY and "BLURRY" in str(e)
