"""Pruebas unitarias de conversión de audio a WAV PCM (Plan 77).

Cubre §B.2 de .claude/workflow/77-plan-aplicacion.md:
4. Un webm/opus, un ogg/opus y un m4a/aac generados en el test (con av) salen como WAV
   de 16 000 Hz, un canal y 16 bits, con la misma duración ±0,1 s.
5. Un archivo que no es audio levanta (no escribe un WAV vacío y sigue).
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

av = pytest.importorskip("av")
import numpy as np  # noqa: E402

from aida.arms.health.audio import to_wav  # noqa: E402


def _crear_audio(
    path: Path,
    fmt: str,
    codec: str,
    rate: int,
    duration: float = 1.0,
) -> None:
    container = av.open(str(path), mode="w", format=fmt)
    stream = container.add_stream(codec, rate=rate)
    stream.layout = "mono"
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    sig = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(sig.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = rate
    frame.pts = 0
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


@pytest.mark.parametrize(
    ("filename", "fmt", "codec", "sample_rate"),
    [
        ("audio.webm", "webm", "opus", 48000),
        ("audio.ogg", "ogg", "opus", 48000),
        ("audio.m4a", "mp4", "aac", 44100),
    ],
)
def test_to_wav_convierte_formatos_a_pcm16_mono_16khz(
    tmp_path: Path,
    filename: str,
    fmt: str,
    codec: str,
    sample_rate: int,
) -> None:
    """Caso 4 (§B.2): webm/opus, ogg/opus y m4a/aac se convierten a WAV PCM 16 bits,
    16 kHz, mono con la misma duración +-0,1 s."""
    src = tmp_path / filename
    dst = tmp_path / f"{filename}.wav"
    _crear_audio(src, fmt, codec, sample_rate, duration=1.0)

    dur = to_wav(src, dst)

    assert abs(dur - 1.0) <= 0.1
    assert dst.exists()

    with wave.open(str(dst), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        nframes = w.getnframes()
        dur_wave = nframes / w.getframerate()
        assert abs(dur_wave - 1.0) <= 0.1


def test_to_wav_archivo_no_audio_levanta_y_no_crea_wav_vacio(tmp_path: Path) -> None:
    """Caso 5 (§B.2): un archivo que no es audio levanta excepción y no escribe
    un WAV vacío para seguir silenciosamente."""
    src = tmp_path / "falso_audio.txt"
    src.write_text("esto no es audio de verdad", encoding="utf-8")
    dst = tmp_path / "salida.wav"

    with pytest.raises(Exception):  # noqa: B017
        to_wav(src, dst)

    assert not dst.exists()
