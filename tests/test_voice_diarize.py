"""Pruebas unitarias de diarización y alineación de hablantes (voice/diarize.py).

Cubre §5.1 de 42-plan-diarizacion.md:
- Dobles sin pyannote real.
- merge_by_speaker con segmento a caballo asignado por mayor solapamiento.
- LocalPyannoteDiarizer.diarize mapeando itertracks a SpeakerTurn.
- LocalPyannoteDiarizer.diarize capturando excepciones y relanzando DiarizationError.
- build_local_diarizer sin AIDA_HF_TOKEN lanzando DiarizationError clara.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aida.voice.diarize import (
    DiarizationError,
    LocalPyannoteDiarizer,
    SpeakerTurn,
    TranscriptSegment,
    build_local_diarizer,
    merge_by_speaker,
)

# ---------------------------------------------------------------------------
# Dobles de prueba para LocalPyannoteDiarizer (sin dependencias reales)
# ---------------------------------------------------------------------------


class _DummySegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _DummyAnnotation:
    def __init__(self, tracks: list[tuple[_DummySegment, str, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False) -> Any:
        assert yield_label is True
        return iter(self._tracks)


class _DummyPipeline:
    def __init__(self, tracks: list[tuple[_DummySegment, str, str]]) -> None:
        self.tracks = tracks
        self.called_with: str | None = None

    def __call__(self, audio_path: str) -> _DummyAnnotation:
        self.called_with = audio_path
        return _DummyAnnotation(self.tracks)


class _FailingPipeline:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def __call__(self, audio_path: str) -> Any:
        raise self.exc


# ---------------------------------------------------------------------------
# Pruebas de merge_by_speaker (§5.1)
# ---------------------------------------------------------------------------


def test_merge_by_speaker_asigna_segmento_a_caballo_por_mayor_solapamiento() -> None:
    """Dos turnos:
    - t1: [0.0, 5.0, SPEAKER_00]
    - t2: [5.0, 10.0, SPEAKER_01]

    Tres segmentos:
    - seg1: [1.0, 3.0] claramente en t1 (solape 2.0s con t1, 0.0s con t2) -> SPEAKER_00
    - seg2: [6.0, 8.0] claramente en t2 (solape 0.0s con t1, 2.0s con t2) -> SPEAKER_01
    - seg3: [4.0, 7.0] a caballo:
      solape t1: [4.0, 5.0] = 1.0s
      solape t2: [5.0, 7.0] = 2.0s
      -> 2.0s > 1.0s, debe asignarse a SPEAKER_01.
    """
    t1 = SpeakerTurn(start=0.0, end=5.0, speaker="SPEAKER_00")
    t2 = SpeakerTurn(start=5.0, end=10.0, speaker="SPEAKER_01")
    turns = (t1, t2)

    seg1 = TranscriptSegment(start=1.0, end=3.0, text="Buenos días doctor")
    seg2 = TranscriptSegment(start=6.0, end=8.0, text="Cuénteme qué le ocurre")
    seg3 = TranscriptSegment(start=4.0, end=7.0, text="Me duele el pecho desde ayer")
    segments = (seg1, seg2, seg3)

    pairs = merge_by_speaker(segments, turns)

    assert pairs == (
        ("SPEAKER_00", "Buenos días doctor"),
        ("SPEAKER_01", "Cuénteme qué le ocurre"),
        ("SPEAKER_01", "Me duele el pecho desde ayer"),
    )


def test_merge_by_speaker_sin_turnos_o_sin_solapamiento_marca_desconocido() -> None:
    """Segmentos fuera de rango o con turns vacíos se marcan como DESCONOCIDO."""
    seg = TranscriptSegment(start=12.0, end=15.0, text="Ruido de fondo")
    turn = SpeakerTurn(start=0.0, end=5.0, speaker="SPEAKER_00")

    # Caso sin solapamiento
    pairs_no_overlap = merge_by_speaker((seg,), (turn,))
    assert pairs_no_overlap == (("DESCONOCIDO", "Ruido de fondo"),)

    # Caso sin turnos
    pairs_no_turns = merge_by_speaker((seg,), ())
    assert pairs_no_turns == (("DESCONOCIDO", "Ruido de fondo"),)


# ---------------------------------------------------------------------------
# Pruebas de LocalPyannoteDiarizer (§5.1)
# ---------------------------------------------------------------------------


def test_local_pyannote_diarizer_diarize_mapea_a_speaker_turns() -> None:
    """Doble de pipeline cuyo itertracks devuelve tuplas fijas -> verifica mapeo a SpeakerTurn."""
    dummy_tracks = [
        (_DummySegment(start=0.0, end=4.5), "track_0", "SPEAKER_00"),
        (_DummySegment(start=4.5, end=9.2), "track_1", "SPEAKER_01"),
    ]
    pipeline = _DummyPipeline(dummy_tracks)
    diarizer = LocalPyannoteDiarizer(pipeline)

    audio_path = Path("/tmp/consulta_test.wav")
    turns = diarizer.diarize(audio_path)

    assert pipeline.called_with == str(audio_path)
    assert turns == (
        SpeakerTurn(start=0.0, end=4.5, speaker="SPEAKER_00"),
        SpeakerTurn(start=4.5, end=9.2, speaker="SPEAKER_01"),
    )


def test_local_pyannote_diarizer_pipeline_exception_relanza_diarization_error() -> None:
    """Cuando el pipeline lanza una excepción, se captura y relanza como DiarizationError."""
    pipeline = _FailingPipeline(RuntimeError("CUDA out of memory"))
    diarizer = LocalPyannoteDiarizer(pipeline)

    audio_path = Path("/tmp/audio_fallido.wav")
    pattern = r"no se pudo diarizar /tmp/audio_fallido\.wav: CUDA out of memory"
    with pytest.raises(DiarizationError, match=pattern) as exc_info:
        diarizer.diarize(audio_path)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Pruebas de build_local_diarizer (§5.1)
# ---------------------------------------------------------------------------


def test_build_local_diarizer_sin_token_lanza_diarization_error_sin_importar_pyannote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_local_diarizer sin AIDA_HF_TOKEN en entorno levanta DiarizationError clara."""
    monkeypatch.delenv("AIDA_HF_TOKEN", raising=False)

    with pytest.raises(
        DiarizationError, match="falta AIDA_HF_TOKEN para el modelo de diarización"
    ):
        build_local_diarizer()
