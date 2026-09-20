"""Pruebas para el caso de consulta clínica con diarización (arms/health/consultation.py).

Cubre §5.2 de 42-plan-diarizacion.md:
- Camino feliz, 2 hablantes: doble de STT y doble de diarizer asignan profesional y
  paciente cronológicamente.
- 3 hablantes: levanta DiarizationError y NO emite CATALOG_MATCHED en el registro de eventos.
- diarizer=None (por defecto): usa stt.transcribe, NUNCA llama a transcribe_segments,
  no emite SPEAKERS_DIARIZED.
- Wording sin etiquetas (professional_label=None o patient_label=None) con diarizer activo:
  levanta ValueError explícito.
- Evento SPEAKERS_DIARIZED: se emite con data={"speakers": "2"} antes de CATALOG_MATCHED.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from cryptography.fernet import Fernet

from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT
from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.consultation import Consultation, SummaryWording
from aida.arms.health.match import CatalogIndex, build_index
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.voice.diarize import (
    DiarizationError,
    LocalPyannoteDiarizer,
    SpeakerTurn,
    TranscriptSegment,
)

# ---------------------------------------------------------------------------
# Dobles de prueba para STT, OCR y Diarización
# ---------------------------------------------------------------------------


class _SpyWhisperSTT(LocalWhisperSTT):
    """Doble espía de LocalWhisperSTT que no carga modelos reales y cuenta llamadas."""

    def __init__(
        self,
        *,
        segments: tuple[TranscriptSegment, ...] = (),
        full_text: str = "",
    ) -> None:
        super().__init__(model=object())
        self._segments = segments
        self._full_text = full_text
        self.transcribe_calls = 0
        self.transcribe_segments_calls = 0

    def transcribe(self, audio_path: Path) -> str:
        self.transcribe_calls += 1
        return self._full_text

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        self.transcribe_segments_calls += 1
        return self._segments


class _DummyPyannoteDiarizer(LocalPyannoteDiarizer):
    """Doble de LocalPyannoteDiarizer con turnos prefijados y contador de llamadas."""

    def __init__(self, turns: tuple[SpeakerTurn, ...]) -> None:
        super().__init__(pipeline=object())
        self._turns = turns
        self.diarize_calls = 0

    def diarize(self, audio_path: Path) -> tuple[SpeakerTurn, ...]:
        self.diarize_calls += 1
        return self._turns


class _DummyOCR(LocalTesseractOCR):
    """Doble mínimo de LocalTesseractOCR."""

    def __init__(self) -> None:
        super().__init__(pytesseract_module=object())

    def read(self, document_path: Path) -> str:
        return "Historia previa sin antecedentes de interés"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_wording() -> SummaryWording:
    return SummaryWording(
        system_prompt="Organiza en Markdown la información de una consulta médica:",
        said_template="Dijo: {}",
        history_template="Historia: {}",
        matched_template="CIE-10: {}",
        no_match_text="Sin códigos",
        professional_label="MÉDICO",
        patient_label="PACIENTE",
    )


@pytest.fixture
def sample_catalog_index() -> CatalogIndex:
    return build_index(
        (
            DiagnosticCode(
                code="I10",
                description="Hipertensión esencial (primaria)",
                level=3,
                source="ambas",
            ),
        )
    )


@pytest.fixture
def sample_cache(tmp_path: Path) -> ConsultationCache:
    return ConsultationCache(tmp_path, key=Fernet.generate_key(), ttl_seconds=3600)


# ---------------------------------------------------------------------------
# Pruebas §5.2
# ---------------------------------------------------------------------------


def test_consultation_camino_feliz_dos_hablantes(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """Camino feliz 2 hablantes:
    - transcribe_segments y diarizer.diarize proporcionan datos de 2 hablantes.
    - run_with_history genera transcript con MÉDICO: ... y PACIENTE: ...
    - Asigna al primer hablante el rol profesional.
    - Emite evento SPEAKERS_DIARIZED con data={"speakers": "2"} antes de CATALOG_MATCHED.
    """
    segments = (
        TranscriptSegment(start=0.0, end=3.0, text="Buenos días, ¿qué síntomas presenta?"),
        TranscriptSegment(start=3.0, end=7.0, text="Tengo dolor de cabeza y mareos frecuentes."),
        TranscriptSegment(start=7.0, end=10.0, text="Vamos a medirle la presión arterial."),
    )
    turns = (
        SpeakerTurn(start=0.0, end=3.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=3.0, end=7.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=7.0, end=10.0, speaker="SPEAKER_00"),
    )

    stt = _SpyWhisperSTT(segments=segments)
    diarizer = _DummyPyannoteDiarizer(turns=turns)
    ocr = _DummyOCR()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=diarizer,
    )

    owner = AgentId("medico_general")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/consulta_audio.wav"),
        history_text="Paciente hipertenso conocido.",
    )

    expected_transcript = (
        "MÉDICO: Buenos días, ¿qué síntomas presenta?\n"
        "PACIENTE: Tengo dolor de cabeza y mareos frecuentes.\n"
        "MÉDICO: Vamos a medirle la presión arterial."
    )
    assert result.transcript == expected_transcript
    assert stt.transcribe_segments_calls == 1
    assert stt.transcribe_calls == 0
    assert diarizer.diarize_calls == 1

    # Verificación de eventos
    events = list(log)
    kinds = [e.kind for e in events]
    assert EventKind.SPEAKERS_DIARIZED in kinds
    assert EventKind.CATALOG_MATCHED in kinds

    # Orden: SPEAKERS_DIARIZED debe preceder a CATALOG_MATCHED
    idx_diarized = kinds.index(EventKind.SPEAKERS_DIARIZED)
    idx_catalog = kinds.index(EventKind.CATALOG_MATCHED)
    assert idx_diarized < idx_catalog

    # Contenido del evento SPEAKERS_DIARIZED
    ev_diarized = events[idx_diarized]
    assert ev_diarized.owner == owner
    assert ev_diarized.tick == 0
    assert ev_diarized.data == {"speakers": "2"}


def test_consultation_tres_hablantes_lanza_diarization_error_y_no_emite_catalog_matched(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """3 hablantes:
    - Diarizer devuelve 3 hablantes distintos -> lanza DiarizationError.
    - Verifica que la tubería se detiene y NO se emite CATALOG_MATCHED en el event log.
    """
    segments = (
        TranscriptSegment(start=0.0, end=2.0, text="Buenos días."),
        TranscriptSegment(start=2.0, end=4.0, text="Hola doctor."),
        TranscriptSegment(start=4.0, end=6.0, text="Vengo a acompañarlo hoy."),
    )
    turns = (
        SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.0, end=4.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=4.0, end=6.0, speaker="SPEAKER_02"),
    )

    stt = _SpyWhisperSTT(segments=segments)
    diarizer = _DummyPyannoteDiarizer(turns=turns)
    ocr = _DummyOCR()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=diarizer,
    )

    owner = AgentId("medico_general")
    with pytest.raises(
        DiarizationError, match=r"se detectaron 3 hablantes distintos, se esperaban 2"
    ):
        consultation.run_with_history(
            owner=owner,
            audio_path=Path("/tmp/consulta_audio.wav"),
            history_text="Sin antecedentes.",
        )

    # Verificar que NO se emitió CATALOG_MATCHED ni SPEAKERS_DIARIZED
    emitted_kinds = [e.kind for e in log]
    assert EventKind.CATALOG_MATCHED not in emitted_kinds
    assert EventKind.SPEAKERS_DIARIZED not in emitted_kinds
    assert EventKind.INPUT_OBSERVED not in emitted_kinds


def test_consultation_sin_diarizador_comportamiento_por_defecto(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """diarizer=None (por defecto):
    - Usa stt.transcribe, NUNCA transcribe_segments.
    - No emite ningún evento SPEAKERS_DIARIZED.
    - Comportamiento idéntico al actual.
    """
    full_audio_text = "Paciente masculino con dolor torácico."
    stt = _SpyWhisperSTT(full_text=full_audio_text)
    ocr = _DummyOCR()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
    )

    owner = AgentId("medico_general")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/consulta_audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert result.transcript == full_audio_text
    assert stt.transcribe_calls == 1
    assert stt.transcribe_segments_calls == 0

    emitted_kinds = [e.kind for e in log]
    assert EventKind.SPEAKERS_DIARIZED not in emitted_kinds
    assert EventKind.CATALOG_MATCHED in emitted_kinds


def test_consultation_wording_sin_etiquetas_con_diarizador_lanza_value_error(
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """Wording sin las etiquetas nuevas (professional_label=None o patient_label=None) con diarizer:
    Lanza ValueError explícito antes de procesar, no un AttributeError.
    """
    diarizer = _DummyPyannoteDiarizer(turns=())
    stt = _SpyWhisperSTT()
    ocr = _DummyOCR()
    log = EventLog(window=100)

    # Ambos None
    wording_ambos_none = SummaryWording(
        system_prompt="Resumen:",
        said_template="Dijo: {}",
        history_template="Historia: {}",
        matched_template="CIE-10: {}",
        no_match_text="Sin códigos",
        professional_label=None,
        patient_label=None,
    )
    consultation1 = Consultation(
        wording=wording_ambos_none,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=diarizer,
    )
    with pytest.raises(
        ValueError, match="diarización activa pero wording no trae professional_label/patient_label"
    ):
        consultation1.run_with_history(
            owner=AgentId("medico_general"),
            audio_path=Path("/tmp/consulta_audio.wav"),
            history_text="Sin antecedentes.",
        )

    # Solo patient_label None
    wording_paciente_none = SummaryWording(
        system_prompt="Resumen:",
        said_template="Dijo: {}",
        history_template="Historia: {}",
        matched_template="CIE-10: {}",
        no_match_text="Sin códigos",
        professional_label="MÉDICO",
        patient_label=None,
    )
    consultation2 = Consultation(
        wording=wording_paciente_none,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=diarizer,
    )
    with pytest.raises(
        ValueError, match="diarización activa pero wording no trae professional_label/patient_label"
    ):
        consultation2.run_with_history(
            owner=AgentId("medico_general"),
            audio_path=Path("/tmp/consulta_audio.wav"),
            history_text="Sin antecedentes.",
        )


# ---------------------------------------------------------------------------
# Pruebas 43-fix-diarizacion.md: I-115, I-116, I-117, I-118
# ---------------------------------------------------------------------------


def test_consultation_segmento_espacios_fuera_de_turno_ignorado_sin_diarization_error(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """I-115 (§2): Dos hablantes reales + un segmento de solo espacios fuera de turno
    (mapeado a 'DESCONOCIDO' por solapamiento 0).
    run_with_history funciona y produce las dos etiquetas sin DiarizationError.
    """
    segments = (
        TranscriptSegment(start=0.0, end=3.0, text="Buenos días, ¿cómo se siente?"),
        TranscriptSegment(start=3.0, end=6.0, text="Mejor que ayer, doctor."),
        TranscriptSegment(start=10.0, end=12.0, text="   "),
    )
    turns = (
        SpeakerTurn(start=0.0, end=3.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=3.0, end=6.0, speaker="SPEAKER_01"),
    )

    stt = _SpyWhisperSTT(segments=segments)
    diarizer = _DummyPyannoteDiarizer(turns=turns)
    ocr = _DummyOCR()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=diarizer,
    )

    owner = AgentId("medico_general")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/consulta_audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert "MÉDICO: Buenos días, ¿cómo se siente?" in result.transcript
    assert "PACIENTE: Mejor que ayer, doctor." in result.transcript
    events = list(log)
    diarized_events = [e for e in events if e.kind == EventKind.SPEAKERS_DIARIZED]
    assert len(diarized_events) == 1
    assert diarized_events[0].data == {"speakers": "2"}


def test_consultation_segmento_texto_real_fuera_de_turno_lanza_diarization_error(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """I-115 (§2): Dos hablantes reales + un segmento con texto real fuera de turno
    (mapeado a 'DESCONOCIDO' por solapamiento 0).
    Lanza DiarizationError indicando que se detectaron 3 hablantes distintos.
    """
    segments = (
        TranscriptSegment(start=0.0, end=3.0, text="Buenos días, ¿cómo se siente?"),
        TranscriptSegment(start=3.0, end=6.0, text="Mejor que ayer, doctor."),
        TranscriptSegment(start=10.0, end=12.0, text="¿Podemos pasar nosotros también?"),
    )
    turns = (
        SpeakerTurn(start=0.0, end=3.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=3.0, end=6.0, speaker="SPEAKER_01"),
    )

    stt = _SpyWhisperSTT(segments=segments)
    diarizer = _DummyPyannoteDiarizer(turns=turns)
    ocr = _DummyOCR()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=diarizer,
    )

    owner = AgentId("medico_general")
    with pytest.raises(
        DiarizationError, match=r"se detectaron 3 hablantes distintos, se esperaban 2"
    ):
        consultation.run_with_history(
            owner=owner,
            audio_path=Path("/tmp/consulta_audio.wav"),
            history_text="Sin antecedentes.",
        )

    emitted_kinds = [e.kind for e in log]
    assert EventKind.CATALOG_MATCHED not in emitted_kinds
    assert EventKind.SPEAKERS_DIARIZED not in emitted_kinds


def test_consultation_monologo_un_hablante_emite_speakers_diarized_uno(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """I-116 (§3): Monólogo (1 solo hablante en los turnos y segmentos).
    run_with_history tiene éxito y emite EventKind.SPEAKERS_DIARIZED con data={"speakers": "1"}.
    """
    segments = (
        TranscriptSegment(start=0.0, end=3.0, text="Dictado de informe clínico matutino."),
        TranscriptSegment(start=3.0, end=6.0, text="Paciente sin hallazgos patológicos."),
    )
    turns = (
        SpeakerTurn(start=0.0, end=3.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=3.0, end=6.0, speaker="SPEAKER_00"),
    )

    stt = _SpyWhisperSTT(segments=segments)
    diarizer = _DummyPyannoteDiarizer(turns=turns)
    ocr = _DummyOCR()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=diarizer,
    )

    owner = AgentId("medico_general")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/consulta_audio.wav"),
        history_text="Sin antecedentes previos.",
    )

    expected_transcript = (
        "MÉDICO: Dictado de informe clínico matutino.\n"
        "MÉDICO: Paciente sin hallazgos patológicos."
    )
    assert result.transcript == expected_transcript

    events = list(log)
    diarized_events = [e for e in events if e.kind == EventKind.SPEAKERS_DIARIZED]
    assert len(diarized_events) == 1
    assert diarized_events[0].data == {"speakers": "1"}


def test_default_summary_wording_falta_etiqueta_lanza_value_error(
    tmp_path: Path,
) -> None:
    """I-117 (§4): default_summary_wording(path) con un archivo YAML al que le falta
    professional_label o patient_label lanza ValueError mencionando la clave faltante.
    """
    base_data = {
        "system_prompt": "Organiza la consulta:",
        "said_template": "Dijo: {}",
        "history_template": "Historia: {}",
        "matched_template": "CIE-10: {}",
        "no_match_text": "Sin códigos",
        "patient_label": "PACIENTE",
    }
    # Falta professional_label
    wording_file_sin_prof = tmp_path / "wording_sin_prof.yaml"
    wording_file_sin_prof.write_text(yaml.safe_dump(base_data), encoding="utf-8")

    with pytest.raises(ValueError, match=r"professional_label"):
        default_summary_wording(wording_file_sin_prof)

    # Falta patient_label
    base_data_sin_pat = {
        "system_prompt": "Organiza la consulta:",
        "said_template": "Dijo: {}",
        "history_template": "Historia: {}",
        "matched_template": "CIE-10: {}",
        "no_match_text": "Sin códigos",
        "professional_label": "MÉDICO",
    }
    wording_file_sin_pat = tmp_path / "wording_sin_pat.yaml"
    wording_file_sin_pat.write_text(yaml.safe_dump(base_data_sin_pat), encoding="utf-8")

    with pytest.raises(ValueError, match=r"patient_label"):
        default_summary_wording(wording_file_sin_pat)


def test_diarize_and_label_con_diarizador_none_lanza_runtime_error(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """I-118 (§5): _diarize_and_label invocado con diarizer=None lanza
    RuntimeError('diarizador no configurado').
    """
    stt = _SpyWhisperSTT()
    ocr = _DummyOCR()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
    )

    owner = AgentId("medico_general")
    with pytest.raises(RuntimeError, match=r"diarizador no configurado"):
        consultation._diarize_and_label(owner, Path("/tmp/consulta_audio.wav"))

