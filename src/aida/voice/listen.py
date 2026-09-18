"""Escucha con corte por silencio.

`SilenceGatedListener.listen` acumula trozos de 512 muestras (16 kHz,
mono, float32) y termina cuando, **después de que empezó el habla**, pasan
`silence_seconds` sin ella — o al llegar a `max_seconds`, pase lo que
pase. El silencio inicial no cuenta: la persona puede tardar en empezar.

El detector (`vad`) es un llamable que recibe un array y devuelve una
probabilidad de habla por ventana de 512 muestras. Se le pasa la cola de
`context_seconds` del búfer, no el último trozo suelto: el VAD de
`faster-whisper` reconstruye su contexto en cada llamada, y un trozo de
32 ms sin contexto es casi ruido para él.

Sin azar, sin reloj: el resultado es función del flujo de trozos.
"""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 512


class SilenceGatedListener:
    def __init__(
        self,
        *,
        chunk_source: Callable[[], np.ndarray],
        vad: Callable[[np.ndarray], np.ndarray],
        speech_threshold: float = 0.5,
        context_seconds: float = 1.0,
    ) -> None:
        self._chunk_source = chunk_source
        self._vad = vad
        self._speech_threshold = speech_threshold
        self._context_chunks = max(1, int(context_seconds * SAMPLE_RATE / CHUNK_SAMPLES))

    def listen(self, *, silence_seconds: float, max_seconds: float) -> np.ndarray:
        """Audio desde el primer trozo hasta el corte. Array vacío si nunca
        hubo habla antes de `max_seconds`."""
        chunks: list[np.ndarray] = []
        max_chunks = int(max_seconds * SAMPLE_RATE / CHUNK_SAMPLES)
        silence_limit = int(silence_seconds * SAMPLE_RATE / CHUNK_SAMPLES)
        speech_started = False
        silent_chunks = 0

        while len(chunks) < max_chunks:
            chunks.append(self._chunk_source())
            tail = np.concatenate(chunks[-self._context_chunks :])
            is_speech = float(self._vad(tail)[-1]) >= self._speech_threshold
            if is_speech:
                speech_started = True
                silent_chunks = 0
            elif speech_started:
                silent_chunks += 1
                if silent_chunks >= silence_limit:
                    break

        if not speech_started:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


def write_wav(audio: np.ndarray, path: Path) -> Path:
    """float32 en [-1, 1] → WAV int16 mono 16 kHz. Es lo que
    `LocalWhisperSTT.transcribe` sabe leer sin tocar `capture.py`."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return path


class MicrophoneSource:
    """Origen real de trozos: el micrófono por defecto vía `sounddevice`.
    Import perezoso — `sounddevice` es del extra `capture`, opcional."""

    def __init__(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES
        )

    def __enter__(self) -> MicrophoneSource:
        self._stream.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stream.stop()
        self._stream.close()

    def __call__(self) -> np.ndarray:
        data, _overflowed = self._stream.read(CHUNK_SAMPLES)
        return np.asarray(data[:, 0], dtype=np.float32)


def vad_from_faster_whisper() -> Callable[[np.ndarray], np.ndarray]:
    """El Silero VAD que ya trae `faster-whisper`. Import perezoso, mismo
    motivo que arriba."""
    from faster_whisper.vad import get_vad_model

    model = get_vad_model()
    return lambda audio: np.asarray(model(audio))


def check_capture_available() -> bool:
    """Comprueba si las dependencias del extra 'capture' están instaladas y disponibles."""
    try:
        import faster_whisper  # noqa: F401
        import sounddevice  # noqa: F401
    except ImportError:
        return False
    return True
