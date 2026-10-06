"""Llevar cualquier audio al formato que todo el vertical ya sabe leer.

El navegador graba en webm u ogg (opus), el teléfono en m4a, una grabadora en mp3.
La nube rechazó un m4a de teléfono con cualquier etiqueta (`I-140`), y cada formato
nuevo sería otra forma de fallar en silencio. Así que todo lo que entra por la
aplicación se convierte aquí a lo mismo que la grabación de referencia: PCM de 16
bits, 16 kHz, mono. Es el formato que la nube acepta y el que el modelo local
remuestrea de todas formas.

`av` (PyAV) llega con el extra de captura —lo trae `faster-whisper`—, así que se
importa dentro: sin el extra, este módulo carga y solo falla quien convierte.
"""

from __future__ import annotations

import wave
from pathlib import Path

RATE = 16000


def to_wav(source: Path, target: Path) -> float:
    """Convierte `source` (cualquier formato que `av` sepa abrir) a WAV PCM de 16
    bits, `RATE` Hz y un canal, en `target`. Devuelve la duración en segundos: es
    lo que permite comprobar que no se perdió audio por el camino."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    muestras = 0
    with av.open(str(source)) as entrada, wave.open(str(target), "wb") as salida:
        salida.setnchannels(1)
        salida.setsampwidth(2)
        salida.setframerate(RATE)
        for frame in entrada.decode(audio=0):
            for convertido in resampler.resample(frame):
                salida.writeframes(convertido.to_ndarray().tobytes())
                muestras += convertido.samples
        for convertido in resampler.resample(None):
            salida.writeframes(convertido.to_ndarray().tobytes())
            muestras += convertido.samples
    return muestras / RATE
