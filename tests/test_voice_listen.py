"""Pruebas unitarias de escucha con corte por silencio y escritura de WAV.

Cubren `aida.voice.listen.SilenceGatedListener` y `write_wav` (§6.1 de
33-plan-consultorio-voz.md) sin micrófono ni modelo: todos los trozos y
probabilidades de habla se inyectan mediante dobles deterministas.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from aida.voice.listen import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    SilenceGatedListener,
    check_capture_available,
    write_wav,
)


def test_listen_silence_speech_silence_cuts_at_silence_limit() -> None:
    """Caso 1: Guion silencio(10) + habla(20) + silencio(60), silence_seconds=1.0
    devuelve exactamente 10 + 20 + 31 trozos (int(1.0 * 16000 / 512) = 31), no 60.
    """
    guion = [0.0] * 10 + [0.9] * 20 + [0.0] * 60
    it = iter(guion)

    def chunk_source() -> np.ndarray:
        return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def vad(_tail: np.ndarray) -> np.ndarray:
        return np.array([next(it)])

    listener = SilenceGatedListener(chunk_source=chunk_source, vad=vad)
    result = listener.listen(silence_seconds=1.0, max_seconds=10.0)

    expected_chunks = 10 + 20 + 31
    assert result.size == expected_chunks * CHUNK_SAMPLES


def test_listen_initial_silence_is_preserved_and_does_not_trigger_cutoff() -> None:
    """Caso 2: Silencio inicial de 40 trozos y luego habla: los 40 sí están en el
    resultado y no disparan el corte — solo cuenta el silencio posterior al habla.
    Mata M1.
    """
    guion = [0.0] * 40 + [0.9] * 10 + [0.0] * 50
    it = iter(guion)

    def chunk_source() -> np.ndarray:
        return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def vad(_tail: np.ndarray) -> np.ndarray:
        return np.array([next(it)])

    listener = SilenceGatedListener(chunk_source=chunk_source, vad=vad)
    result = listener.listen(silence_seconds=1.0, max_seconds=10.0)

    expected_chunks = 40 + 10 + 31
    assert result.size == expected_chunks * CHUNK_SAMPLES


def test_listen_never_speaks_returns_empty_array_at_max_seconds() -> None:
    """Caso 3: Nunca habla -> array vacío al llegar a max_seconds, y chunk_source
    se llama exactamente int(max_seconds * 16000 / 512) veces.
    Mata M3.
    """
    calls = 0

    def chunk_source() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def vad(_tail: np.ndarray) -> np.ndarray:
        return np.array([0.0])

    listener = SilenceGatedListener(chunk_source=chunk_source, vad=vad)
    max_seconds = 1.0
    expected_calls = int(max_seconds * SAMPLE_RATE / CHUNK_SAMPLES)

    result = listener.listen(silence_seconds=0.5, max_seconds=max_seconds)

    assert result.size == 0
    assert calls == expected_calls


def test_listen_continuous_speech_cuts_at_max_seconds() -> None:
    """Caso 4: Habla continua -> corta en max_seconds, no antes.
    Si el bucle no respetara max_chunks, chunk_source fallaría al ser llamado
    más allá del límite. Mata M4.
    """
    max_seconds = 1.0
    expected_chunks = int(max_seconds * SAMPLE_RATE / CHUNK_SAMPLES)
    calls = 0

    def chunk_source() -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls > expected_chunks:
            raise StopIteration("chunk_source llamado más allá de max_chunks")
        return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def vad(_tail: np.ndarray) -> np.ndarray:
        return np.array([0.9])

    listener = SilenceGatedListener(chunk_source=chunk_source, vad=vad)
    result = listener.listen(silence_seconds=0.5, max_seconds=max_seconds)

    assert calls == expected_chunks
    assert result.size == expected_chunks * CHUNK_SAMPLES


def test_listen_vad_receives_context_tail_not_isolated_chunk() -> None:
    """Caso 5: El VAD recibe la cola de context_seconds, no el trozo suelto:
    con context_seconds=1.0, tras 40 trozos el array que llega al VAD mide 31 * 512.
    Mata M2.
    """
    tail_lengths: list[int] = []

    def chunk_source() -> np.ndarray:
        return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def vad(tail: np.ndarray) -> np.ndarray:
        tail_lengths.append(len(tail))
        return np.array([0.9])

    listener = SilenceGatedListener(
        chunk_source=chunk_source, vad=vad, context_seconds=1.0
    )
    # 40 trozos equivalen a 40 * 512 / 16000 = 1.28 s
    listener.listen(silence_seconds=1.0, max_seconds=40 * CHUNK_SAMPLES / SAMPLE_RATE)

    assert len(tail_lengths) == 40
    assert tail_lengths[0] == CHUNK_SAMPLES
    assert tail_lengths[-1] == 31 * CHUNK_SAMPLES


def test_write_wav_format_and_amplitude(tmp_path: Path) -> None:
    """Caso 6: write_wav crea un archivo que al releerse con wave tiene
    1 canal, 16 kHz, 16 bits y len(audio) frames; un valor 1.0 da 32767.
    Mata M5.
    """
    audio = np.array([1.0, -1.0, 0.0, 0.5], dtype=np.float32)
    wav_path = tmp_path / "salida.wav"

    ret_path = write_wav(audio, wav_path)
    assert ret_path == wav_path

    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16_000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == len(audio)
        frames = wf.readframes(len(audio))
        pcm = np.frombuffer(frames, dtype=np.int16)
        assert pcm[0] == 32767
        assert pcm[1] == -32767
        assert pcm[2] == 0
        assert pcm[3] == int(0.5 * 32767)


def test_check_capture_available_con_extra_instalado_devuelve_true() -> None:
    """I-110: Con el extra capture instalado, check_capture_available() retorna True."""
    pytest.importorskip("sounddevice", reason="requiere el extra 'capture' (sounddevice)")
    pytest.importorskip("faster_whisper", reason="requiere el extra 'capture' (faster_whisper)")
    assert check_capture_available() is True


def test_check_capture_available_sin_dependencias_devuelve_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-110: Al faltar dependencias, check_capture_available() retorna False."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert check_capture_available() is False


def test_check_capture_available_sin_faster_whisper_devuelve_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-110: Si falta faster_whisper, check_capture_available() retorna False."""
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert check_capture_available() is False

