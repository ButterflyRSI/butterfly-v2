"""Local text-to-speech with Kokoro.

Kokoro is a small open-weight voice model that runs offline on CPU.
It's optional: install it and fetch the model once with

    pip install -e ".[voice]"
    python -m butterfly.tts download

If it isn't installed, the server reports it as unavailable and the UI
falls back to the browser's built-in voice.
"""

from __future__ import annotations

import io
import os
import re
import sys
import threading
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("BUTTERFLY_VOICE_DIR", ROOT / "models" / "kokoro"))
MODEL_FILE = "kokoro-v1.0.onnx"   # full precision; the int8 build can produce silent audio
VOICES_FILE = "voices-v1.0.bin"
DOWNLOAD_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MAX_CHARS = 2000

# Kokoro voice names start with a language/gender code. Map the code to
# the language setting Kokoro needs for pronunciation.
LANG_BY_PREFIX = {
    "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi",
    "i": "it", "j": "ja", "p": "pt-br", "z": "cmn",
}


class TTSUnavailable(RuntimeError):
    pass


def clean_for_speech(text: str) -> str:
    """Strip formatting that sounds bad read aloud."""
    text = re.sub(r"```.*?```", " code block omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)            # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)          # links keep their label
    text = re.sub(r"https?://\S+", " a link ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s*[-*\u2022]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"[*_~]{1,3}", "", text)                        # bold/italic marks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_wav(samples, sample_rate: int) -> bytes:
    """Float samples in [-1, 1] to a 16-bit mono WAV file."""
    import numpy as np

    pcm = (np.clip(np.nan_to_num(samples), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class KokoroTTS:
    def __init__(self, model_dir: Path | str = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._engine = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        return self.model_dir / MODEL_FILE

    @property
    def voices_path(self) -> Path:
        return self.model_dir / VOICES_FILE

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise TTSUnavailable('Kokoro isn\'t installed. Run: pip install -e ".[voice]"') from exc
        if not self.model_path.exists() or not self.voices_path.exists():
            raise TTSUnavailable("Kokoro voice files are missing. Run: python -m butterfly.tts download")
        self._engine = Kokoro(str(self.model_path), str(self.voices_path))
        return self._engine

    def status(self) -> dict:
        try:
            with self._lock:
                engine = self._load()
            return {"available": True, "engine": "kokoro", "voices": sorted(engine.get_voices()), "error": None}
        except TTSUnavailable as exc:
            return {"available": False, "engine": "kokoro", "voices": [], "error": str(exc)}
        except Exception as exc:  # broken model file, onnxruntime problem, etc.
            return {"available": False, "engine": "kokoro", "voices": [], "error": f"Kokoro failed to load: {exc}"}

    def synthesize(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> bytes:
        text = clean_for_speech(text)[:MAX_CHARS]
        if not text:
            raise ValueError("nothing to say")
        speed = max(0.5, min(2.0, float(speed)))
        with self._lock:
            engine = self._load()
            if voice not in engine.get_voices():
                raise ValueError(f"unknown voice '{voice}'")
            lang = LANG_BY_PREFIX.get(voice[:1], "en-us")
            samples, sample_rate = engine.create(text, voice=voice, speed=speed, lang=lang)
        if len(samples) == 0:
            raise RuntimeError("Kokoro produced no audio for that text")
        return to_wav(samples, sample_rate)


def download(model_dir: Path | str = MODEL_DIR) -> None:
    """Fetch the model (~325 MB) and voices (~28 MB) once."""
    import httpx

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in (MODEL_FILE, VOICES_FILE):
        dest = model_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {name} already downloaded")
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"  downloading {name}...")
        with httpx.stream("GET", f"{DOWNLOAD_BASE}/{name}", follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    {done * 100 // total}%", end="", flush=True)
        print()
        tmp.replace(dest)
    print(f"  Voice files are in {model_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download()
    elif len(sys.argv) > 1 and sys.argv[1] == "voices":
        print("\n".join(KokoroTTS().status()["voices"]))
    else:
        print("usage: python -m butterfly.tts download | voices")
