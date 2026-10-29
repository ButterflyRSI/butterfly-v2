import io
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from butterfly.llm import MockLLM
from butterfly.server import create_app
from butterfly.tts import MODEL_DIR, KokoroTTS, TTSUnavailable, clean_for_speech, to_wav


def test_clean_for_speech_strips_formatting():
    text = "# Title\n- **Bold** point\nSee [the docs](http://x.com) or https://y.com\n```py\nx=1\n```\nUse `pip`."
    out = clean_for_speech(text)
    assert out == "Title Bold point See the docs or a link code block omitted. Use pip."


def test_to_wav_is_valid_16bit_mono():
    np = pytest.importorskip("numpy")
    data = to_wav(np.array([0.0, 0.5, -0.5, 2.0, float("nan")], dtype="float32"), 24000)
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 5)


def test_missing_model_files_reported_not_crashed(tmp_path):
    status = KokoroTTS(tmp_path).status()
    assert status["available"] is False and status["error"]
    with pytest.raises(TTSUnavailable):
        KokoroTTS(tmp_path).synthesize("hello")


class FakeTTS:
    def __init__(self):
        self.calls = []

    def status(self):
        return {"available": True, "engine": "fake", "voices": ["af_nova", "am_michael"], "error": None}

    def synthesize(self, text, voice="af_heart", speed=1.0):
        if voice not in ("af_nova", "am_michael"):
            raise ValueError(f"unknown voice '{voice}'")
        self.calls.append((text, voice, speed))
        return b"RIFF....WAVE"


@pytest.fixture
def fake():
    return FakeTTS()


@pytest.fixture
def client(tmp_path, fake):
    return TestClient(create_app(llm=MockLLM(), data_dir=tmp_path, tts=fake))


def test_tts_uses_persona_default_voice(client, fake):
    r = client.post("/api/tts", json={"persona": "nova", "text": "Hi there"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert fake.calls[-1] == ("Hi there", "af_nova", 1.0)
    client.post("/api/tts", json={"persona": "echo", "text": "Hi"})
    assert fake.calls[-1][1] == "am_michael"


def test_tts_voice_override_and_bad_voice(client, fake):
    client.post("/api/tts", json={"persona": "nova", "text": "Hi", "voice": "am_michael", "speed": 1.2})
    assert fake.calls[-1] == ("Hi", "am_michael", 1.2)
    assert client.post("/api/tts", json={"persona": "nova", "text": "Hi", "voice": "zz_nope"}).status_code == 400
    assert client.post("/api/tts", json={"persona": "nova", "text": "Hi", "speed": 9}).status_code == 422


def test_tts_unavailable_is_503(tmp_path):
    c = TestClient(create_app(llm=MockLLM(), data_dir=tmp_path, tts=KokoroTTS(tmp_path / "none")))
    assert c.get("/api/tts").json()["available"] is False
    assert c.post("/api/tts", json={"persona": "nova", "text": "Hi"}).status_code == 503


def test_personas_include_voice(client):
    nova = client.get("/api/personas").json()[0]
    assert nova["voice"]["name"] == "af_nova"


@pytest.mark.skipif(not (Path(MODEL_DIR) / "kokoro-v1.0.onnx").exists(), reason="voice model not downloaded")
def test_real_kokoro_speaks():
    audio = KokoroTTS().synthesize("Hello from Butterfly.", voice="af_nova")
    with wave.open(io.BytesIO(audio)) as w:
        assert w.getnframes() / w.getframerate() > 0.5
