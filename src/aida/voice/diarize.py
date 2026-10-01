"""Diarización: quién habla y cuándo, y su fusión con una transcripción por
segmentos. No sabe qué son los hablantes — solo produce identificadores
anónimos (`SPEAKER_00`, `SPEAKER_01`, ...) y los alinea por tiempo con los
segmentos de una transcripción ya hecha. Asignar un rol de dominio a cada
identificador es trabajo de quien use este módulo, no de este módulo.

Directorio exento del guardián de imports de modelo
(`tests/test_agnosticism.py`, `EXEMPT_DIR_NAMES`), como `cognition/`,
`intent/` y `arms/*/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """Un tramo de una transcripción, con marca de tiempo cuando el proveedor
    la da. Genérico: no es propiedad de ningún STT concreto, cualquier motor
    de voz lo produce.

    `start`/`end` admiten `None`: algunos modelos de nube devuelven un solo
    bloque de texto sin tiempos. No se inventan tiempos plausibles — un `0.0`
    se confundiría con un dato real—, así que quien necesite tiempos (la
    diarización) tiene que fallar explícitamente al encontrarse con `None`,
    no emparejar contra un cero."""

    start: float | None
    end: float | None
    text: str


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class DiarizationError(RuntimeError):
    """El audio no se pudo diarizar."""


class LocalPyannoteDiarizer:
    """Envuelve un `pyannote.audio.Pipeline` ya cargado — igual criterio que
    `LocalWhisperSTT`: cargar el pipeline es lento y con estado, un test lo
    sustituye por un doble sin descargar nada."""

    def __init__(self, pipeline: object) -> None:
        self._pipeline: Any = pipeline

    def diarize(self, audio_path: Path) -> tuple[SpeakerTurn, ...]:
        try:
            annotation = self._pipeline(str(audio_path))
            return tuple(
                SpeakerTurn(start=segment.start, end=segment.end, speaker=label)
                for segment, _track, label in annotation.itertracks(yield_label=True)
            )
        except Exception as exc:
            raise DiarizationError(f"no se pudo diarizar {audio_path}: {exc}") from exc


def merge_by_speaker(
    segments: tuple[TranscriptSegment, ...], turns: tuple[SpeakerTurn, ...]
) -> tuple[tuple[str, str], ...]:
    """Por cada segmento de transcripción, el hablante del turno con más
    solapamiento de tiempo. Aproximado a nivel de segmento, no de palabra —
    decisión de alcance §1.3: es lo proporcional a lo que hace falta hoy.

    Exige tiempos reales en **todos** los segmentos: sin ellos no hay
    solapamiento que calcular, y tratar `None` como `0.0` emparejaría
    cualquier cosa con el primer turno — una etiqueta de hablante inventada
    en un registro clínico es peor que no tener ninguna (plan 42)."""
    pairs: list[tuple[str, str]] = []
    for seg in segments:
        if seg.start is None or seg.end is None:
            raise DiarizationError(
                "la diarización necesita segmentos con marcas de tiempo, y llegó uno sin ellas"
            )
        seg_start, seg_end = seg.start, seg.end
        best = max(
            turns,
            key=lambda t: max(0.0, min(seg_end, t.end) - max(seg_start, t.start)),
            default=None,
        )
        if best is not None:
            overlap = max(0.0, min(seg_end, best.end) - max(seg_start, best.start))
            speaker = best.speaker if overlap > 0.0 else "DESCONOCIDO"
        else:
            speaker = "DESCONOCIDO"
        pairs.append((speaker, seg.text))
    return tuple(pairs)


def build_local_diarizer() -> LocalPyannoteDiarizer:
    """Construye el diarizador local. Vive aquí y no en `run.py` por la
    misma razón que `build_local_capture` vive en `arms/health/`: es la
    costura declarada hacia el runtime de modelo."""
    import inspect
    import os

    token = os.environ.get("AIDA_HF_TOKEN")
    if not token:
        raise DiarizationError("falta AIDA_HF_TOKEN para el modelo de diarización")

    from pyannote.audio import Pipeline

    sig = inspect.signature(Pipeline.from_pretrained)
    if "token" in sig.parameters:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=token
        )
    elif "use_auth_token" in sig.parameters:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    return LocalPyannoteDiarizer(pipeline)
