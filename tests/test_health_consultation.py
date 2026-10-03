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

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.fernet import Fernet

from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT
from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.consultation import Consultation, PreparedConsultation, SummaryWording
from aida.arms.health.extract import build_health_extractor
from aida.arms.health.history import HistoryLayout, HistorySection
from aida.arms.health.lexicon import Lexicon
from aida.arms.health.match import CatalogIndex, CodableText, build_index
from aida.arms.health.metrics import ReviewMetricsStore, review_record
from aida.arms.health.model import ClinicalFinding
from aida.arms.health.negation import (
    load_negation,
    load_retraction,
    load_third_party,
)
from aida.arms.health.note import ClinicalNote, NoteSettings, RenderedNote
from aida.arms.health.review import Answer, ItemKind, ReviewItem, Verdict
from aida.arms.health.sections import SectionSpec, load_sections
from aida.arms.health.segment import ClinicalSegmenter
from aida.arms.health.wording import default_summary_wording
from aida.cognition.briefing import SpokenReply
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.engine.ports import CognitionProvider
from aida.voice.diarize import (
    DiarizationError,
    LocalPyannoteDiarizer,
    SpeakerTurn,
    TranscriptSegment,
)


def _codables(*textos: str) -> tuple[CodableText, ...]:
    return tuple(CodableText(query=t, source=t) for t in textos)


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


class _SpyCognitionProvider(CognitionProvider):
    """Proveedor espía que cuenta llamadas a think y devuelve una respuesta prefijada."""

    def __init__(self, response: str = "Resumen de prueba") -> None:
        self.response = response
        self.think_calls = 0

    def think(self, briefing: str, schema: type) -> Any:
        self.think_calls += 1
        return SpokenReply(text=self.response)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_wording_review() -> SummaryWording:
    return SummaryWording(
        system_prompt="Organiza en Markdown la información de una consulta médica:",
        said_template="Dijo: {}",
        history_template="Historia: {}",
        matched_template="CIE-10: {}",
        no_match_text="Sin códigos",
        professional_label="MÉDICO",
        patient_label="PACIENTE",
        section_template="### {}\n{}",
        confirmed_template="Códigos confirmados por el profesional: {}",
        no_confirmed_text="El profesional no confirmó ningún código.",
        history_codes_template="Códigos que ya constaban en la historia: {}",
        known_code_suffix="ya en la historia",
    )


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
        "MÉDICO: Dictado de informe clínico matutino.\nMÉDICO: Paciente sin hallazgos patológicos."
    )
    assert result.transcript == expected_transcript

    events = list(log)
    diarized_events = [e for e in events if e.kind == EventKind.SPEAKERS_DIARIZED]
    assert len(diarized_events) == 1
    assert diarized_events[0].data == {"speakers": "1"}


def test_default_summary_wording_falta_etiqueta_lanza_value_error(
    tmp_path: Path,
) -> None:
    """I-117 (§4), I-119: default_summary_wording(path) con un archivo YAML al que le falta
    professional_label, patient_label o section_template lanza ValueError
    mencionando la clave faltante.
    """
    base_data = {
        "system_prompt": "Organiza la consulta:",
        "said_template": "Dijo: {}",
        "history_template": "Historia: {}",
        "matched_template": "CIE-10: {}",
        "no_match_text": "Sin códigos",
        "patient_label": "PACIENTE",
        "section_template": "## {title}\n{content}",
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
        "section_template": "## {title}\n{content}",
    }
    wording_file_sin_pat = tmp_path / "wording_sin_pat.yaml"
    wording_file_sin_pat.write_text(yaml.safe_dump(base_data_sin_pat), encoding="utf-8")

    with pytest.raises(ValueError, match=r"patient_label"):
        default_summary_wording(wording_file_sin_pat)

    # Falta section_template
    base_data_sin_sec = {
        "system_prompt": "Organiza la consulta:",
        "said_template": "Dijo: {}",
        "history_template": "Historia: {}",
        "matched_template": "CIE-10: {}",
        "no_match_text": "Sin códigos",
        "professional_label": "MÉDICO",
        "patient_label": "PACIENTE",
    }
    wording_file_sin_sec = tmp_path / "wording_sin_sec.yaml"
    wording_file_sin_sec.write_text(yaml.safe_dump(base_data_sin_sec), encoding="utf-8")

    with pytest.raises(ValueError, match=r"section_template"):
        default_summary_wording(wording_file_sin_sec)


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


# ---------------------------------------------------------------------------
# Pruebas §6.3: Segmentación clínica (44-plan-segmentacion-clinica.md)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_wording_sections() -> SummaryWording:
    return SummaryWording(
        system_prompt="Organiza en Markdown la información de una consulta médica:",
        said_template="Dijo: {}",
        history_template="Historia: {}",
        matched_template="CIE-10: {}",
        no_match_text="Sin códigos",
        professional_label="MÉDICO",
        patient_label="PACIENTE",
        section_template="### {}\n{}",
    )


def test_consultation_segmenter_camino_feliz_sin_diarizador(
    sample_wording_sections: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§6.3: Camino feliz, sin diarizador:
    - Doble de transcribe_segments devuelve 3 segmentos de secciones distintas.
    - segmenter activo.
    - run_with_history agrupa por sección en el orden del catálogo usando section_template.
    - transcribe_segments es llamado 1 vez, transcribe 0 veces.
    """
    segments = (
        TranscriptSegment(start=0.0, end=3.0, text="consulta hoy por fiebre persistente"),
        TranscriptSegment(start=3.0, end=6.0, text="se receta paracetamol cada 8 horas"),
        TranscriptSegment(start=6.0, end=9.0, text="se solicita ecografía abdominal"),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
    )

    owner = AgentId("medico_general")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/consulta_audio.wav"),
        history_text="Paciente sin antecedentes de interés.",
    )

    # Verifica llamadas a STT
    assert stt.transcribe_segments_calls == 1
    assert stt.transcribe_calls == 0

    # Transcript une con saltos de línea sin prefijo de hablante
    expected_transcript = (
        "consulta hoy por fiebre persistente\n"
        "se receta paracetamol cada 8 horas\n"
        "se solicita ecografía abdominal"
    )
    assert result.transcript == expected_transcript

    # Resumen estructurado por secciones
    assert "### Motivo de consulta\nconsulta hoy por fiebre persistente" in result.summary_markdown
    assert "### Medicamentos\nse receta paracetamol cada 8 horas" in result.summary_markdown
    assert "### Órdenes\nse solicita ecografía abdominal" in result.summary_markdown
    assert "Dijo:" not in result.summary_markdown

    # Orden del catálogo conservado
    pos_motivo = result.summary_markdown.index("### Motivo de consulta")
    pos_med = result.summary_markdown.index("### Medicamentos")
    pos_ord = result.summary_markdown.index("### Órdenes")
    assert pos_motivo < pos_med < pos_ord


def test_consultation_con_diarizador_y_segmentador_juntos(
    sample_wording_sections: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§6.3: Con diarizador y segmentador juntos:
    - transcribe_segments y diarizer.diarize producen líneas con prefijo de hablante
      ("MÉDICO: ..." o "PACIENTE: ...").
    - Confirma que las líneas etiquetadas igual se clasifican por su contenido según el catálogo.
    """
    segments = (
        TranscriptSegment(
            start=0.0, end=3.0, text="Buenos días, el motivo de consulta es revisión rutinaria."
        ),
        TranscriptSegment(start=3.0, end=6.0, text="Tengo fiebre alta desde hace dos semanas."),
        TranscriptSegment(start=6.0, end=9.0, text="De acuerdo, se receta reposo y antipiréticos."),
    )
    turns = (
        SpeakerTurn(start=0.0, end=3.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=3.0, end=6.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=6.0, end=9.0, speaker="SPEAKER_00"),
    )

    stt = _SpyWhisperSTT(segments=segments)
    diarizer = _DummyPyannoteDiarizer(turns=turns)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=diarizer,
        segmenter=segmenter,
    )

    owner = AgentId("medico_general")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/consulta_audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert "MÉDICO: Buenos días, el motivo de consulta es revisión rutinaria." in result.transcript
    assert "PACIENTE: Tengo fiebre alta desde hace dos semanas." in result.transcript
    assert "MÉDICO: De acuerdo, se receta reposo y antipiréticos." in result.transcript

    # Clasificación por contenido a pesar del prefijo MÉDICO / PACIENTE
    assert (
        "### Motivo de consulta\nMÉDICO: Buenos días, el motivo de consulta es revisión rutinaria."
        in result.summary_markdown
    )
    assert (
        "### Medicamentos\nMÉDICO: De acuerdo, se receta reposo y antipiréticos."
        in result.summary_markdown
    )
    # Enfermedad actual ("desde hace dos semanas")
    assert (
        "### Enfermedad actual\nPACIENTE: Tengo fiebre alta desde hace dos semanas."
        in result.summary_markdown
    )


def test_consultation_segmenter_none_por_defecto_no_llama_transcribe_segments(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§6.3: segmenter=None (por defecto):
    - Comportamiento idéntico al original: llama a stt.transcribe, no llama a transcribe_segments,
      no emite TEXT_SEGMENTED, y usa said_template.
    """
    texto = "Paciente acude por cefalea generalizada."
    stt = _SpyWhisperSTT(full_text=texto)
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
        segmenter=None,
    )

    owner = AgentId("medico_general")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/consulta_audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert stt.transcribe_calls == 1
    assert stt.transcribe_segments_calls == 0
    assert result.transcript == texto
    assert f"Dijo: {texto}" in result.summary_markdown

    emitted_kinds = [e.kind for e in log]
    assert EventKind.TEXT_SEGMENTED not in emitted_kinds


def test_consultation_wording_sin_section_template_con_segmentador_lanza_value_error(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§6.3: Wording sin section_template (section_template=None) + segmentador activo:
    lanza ValueError explícito antes de procesar el resumen.
    """
    assert sample_wording.section_template is None

    stt = _SpyWhisperSTT(segments=(TranscriptSegment(0.0, 2.0, "consulta hoy por dolor"),))
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
    )

    owner = AgentId("medico_general")
    with pytest.raises(
        ValueError, match=r"segmentación activa pero wording no trae section_template"
    ):
        consultation.run_with_history(
            owner=owner,
            audio_path=Path("/tmp/consulta_audio.wav"),
            history_text="Sin antecedentes.",
        )


def test_consultation_evento_text_segmented_orden_y_datos(
    sample_wording_sections: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§6.3: Evento TEXT_SEGMENTED:
    - Se emite con las secciones correctas en data={"sections": ...}.
    - Orden §2.3: tras INPUT_OBSERVED del transcript, antes del INPUT_OBSERVED de la historia.
    """
    segments = (
        TranscriptSegment(start=0.0, end=3.0, text="consulta hoy por mareos"),
        TranscriptSegment(start=3.0, end=6.0, text="se receta reposo"),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
    )

    owner = AgentId("medico_general")
    consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/consulta_audio.wav"),
        history_text="Historia previa.",
    )

    events = list(log)
    kinds = [e.kind for e in events]

    assert EventKind.TEXT_SEGMENTED in kinds
    idx_seg = kinds.index(EventKind.TEXT_SEGMENTED)

    # El evento anterior debe ser INPUT_OBSERVED (del transcript)
    ev_anterior = events[idx_seg - 1]
    assert ev_anterior.kind == EventKind.INPUT_OBSERVED
    assert ev_anterior.content == "consulta hoy por mareos\nse receta reposo"

    # Tras TEXT_SEGMENTED viene MODEL_BUILT (Plan 45 §5.2)
    ev_model = events[idx_seg + 1]
    assert ev_model.kind == EventKind.MODEL_BUILT
    assert ev_model.owner == owner
    assert ev_model.tick == 0
    assert ev_model.data == {"findings": "2"}

    # El evento posterior debe ser INPUT_OBSERVED (de la historia)
    ev_posterior = events[idx_seg + 2]
    assert ev_posterior.kind == EventKind.INPUT_OBSERVED
    assert ev_posterior.content == "Historia previa."

    # Tras INPUT_OBSERVED de la historia viene CATALOG_MATCHED
    assert kinds[idx_seg + 3] == EventKind.CATALOG_MATCHED

    # Secuencia completa requerida por §5.2:
    # INPUT_OBSERVED (transcript) -> TEXT_SEGMENTED -> MODEL_BUILT ->
    # INPUT_OBSERVED (historia) -> CATALOG_MATCHED
    expected_sequence = [
        EventKind.INPUT_OBSERVED,
        EventKind.TEXT_SEGMENTED,
        EventKind.MODEL_BUILT,
        EventKind.INPUT_OBSERVED,
        EventKind.CATALOG_MATCHED,
    ]
    assert kinds[idx_seg - 1 : idx_seg + 4] == expected_sequence

    # Contenido del evento TEXT_SEGMENTED
    ev_seg = events[idx_seg]
    assert ev_seg.owner == owner
    assert ev_seg.tick == 0
    assert ev_seg.data == {"sections": "motivo_consulta,medicamentos"}


def test_consultation_con_segmentador_arma_encounter_y_emite_model_built(
    sample_wording_sections: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§5.2: Con segmentador activo:
    - result.encounter is not None
    - findings coincide exactamente con lo que produjo segmenter.segment(transcript)
    - MODEL_BUILT emitido con data["findings"] igual a la suma real de líneas
      (3 líneas en 2 secciones)
    """
    segments = (
        TranscriptSegment(start=0.0, end=2.0, text="consulta hoy por cefalea"),
        TranscriptSegment(start=2.0, end=4.0, text="el motivo de consulta incluye mareos"),
        TranscriptSegment(start=4.0, end=6.0, text="se receta paracetamol cada 8 horas"),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    sections = load_sections()
    segmenter = ClinicalSegmenter(sections)

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
    )

    owner = AgentId("medico_test")
    transcript_text = (
        "consulta hoy por cefalea\n"
        "el motivo de consulta incluye mareos\n"
        "se receta paracetamol cada 8 horas"
    )
    grouped = segmenter.segment(transcript_text)
    # grouped tiene 2 secciones: motivo_consulta con 2 líneas, medicamentos con 1 línea
    assert len(grouped) == 2
    assert len(grouped[0][1]) == 2
    assert len(grouped[1][1]) == 1

    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert result.encounter is not None
    assert result.encounter.owner == owner

    # Coincidencia exacta con lo producido por segmenter.segment(transcript)
    expected_findings_keys = [spec.section_id for spec, _ in grouped]
    assert list(result.encounter.findings.keys()) == expected_findings_keys
    for spec, lines in grouped:
        finding_texts = [f.text for f in result.encounter.findings[spec.section_id]]
        assert finding_texts == list(lines)
        for f in result.encounter.findings[spec.section_id]:
            assert f.section_id == spec.section_id

    # MODEL_BUILT con conteo real de líneas (3), no número de secciones (2)
    model_events = [e for e in log if e.kind == EventKind.MODEL_BUILT]
    assert len(model_events) == 1
    ev = model_events[0]
    assert ev.owner == owner
    assert ev.tick == 0
    assert ev.data == {"findings": "3"}


def test_consultation_sin_segmentador_no_arma_encounter_ni_emite_model_built(
    sample_wording: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§5.2: Sin segmentador (por defecto):
    - result.encounter is None
    - no se emite MODEL_BUILT
    """
    segments = (TranscriptSegment(start=0.0, end=2.0, text="consulta hoy por cefalea"),)
    stt = _SpyWhisperSTT(segments=segments)
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
        segmenter=None,
    )

    owner = AgentId("medico_test")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert result.encounter is None
    assert not any(e.kind == EventKind.MODEL_BUILT for e in log)


# ---------------------------------------------------------------------------
# Pruebas §7.5 — Integración con Extracción Determinista
# ---------------------------------------------------------------------------


def test_consultation_extractor_y_segmentador_activos(
    sample_wording_sections: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§7.5: Extractor + segmentador activos:
    - Los findings del encounter final llevan los campos completados
      (duration, laterality, reported_by).
    - Se emite FIELDS_EXTRACTED con los tipos correctos en orden alfabético.
    """
    segments = (
        TranscriptSegment(start=0.0, end=2.0, text="buenos días, ¿cuál es el motivo de consulta?"),
        TranscriptSegment(
            start=2.0,
            end=5.0,
            text="consulta por dolor en rodilla derecha desde hace dos semanas",
        ),
        TranscriptSegment(start=5.0, end=7.0, text="se receta paracetamol cada 8 horas"),
    )
    turns = (
        SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.0, end=5.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=5.0, end=7.0, speaker="SPEAKER_00"),
    )
    stt = _SpyWhisperSTT(segments=segments)
    diarizer = _DummyPyannoteDiarizer(turns=turns)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=diarizer,
        segmenter=segmenter,
        extractor=extractor,
    )

    owner = AgentId("dra_martinez")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert result.encounter is not None
    # Verificar que los findings tienen campos extraídos
    finding_paciente = next(
        f
        for items in result.encounter.findings.values()
        for f in items
        if "rodilla derecha" in f.text
    )
    assert finding_paciente.duration == "2 semanas"
    assert finding_paciente.laterality == "derecho"
    assert finding_paciente.reported_by == "PACIENTE"

    # Verificar el evento FIELDS_EXTRACTED
    fields_events = [e for e in log if e.kind == EventKind.FIELDS_EXTRACTED]
    assert len(fields_events) == 1
    ev = fields_events[0]
    assert ev.owner == owner
    assert ev.tick == 0
    # Orden alfabético estricto
    campos = ev.data.get("fields")
    assert isinstance(campos, str)
    assert set(campos.split(",")) == {"duration", "laterality", "reported_by"}
    assert campos == "duration,laterality,reported_by"
    assert campos == ",".join(sorted(campos.split(",")))


def test_consultation_extractor_sin_segmentador_lanza_value_error_sin_tocar_audio(
    sample_wording_sections: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§7.5: Extractor sin segmentador:
    - ValueError antes de tocar audio
    - Un doble de STT/OCR que levanta excepción confirma que no se llamó.
    """

    class _FailingSTT(LocalWhisperSTT):
        def __init__(self) -> None:
            super().__init__(model=object())

        def transcribe(self, audio_path: Path) -> str:
            raise AssertionError("STT.transcribe no debió ser llamado")

        def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
            raise AssertionError("STT.transcribe_segments no debió ser llamado")

    class _FailingOCR(LocalTesseractOCR):
        def __init__(self) -> None:
            super().__init__(pytesseract_module=object())

        def read(self, document_path: Path) -> str:
            raise AssertionError("OCR.read no debió ser llamado")

    log = EventLog(window=100)
    extractor = build_health_extractor()

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=_FailingSTT(),
        ocr=_FailingOCR(),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=None,
        extractor=extractor,
    )

    owner = AgentId("medico_test")
    with pytest.raises(ValueError, match=r"extracción activa pero no hay segmentador configurado"):
        consultation.run_with_history(
            owner=owner,
            audio_path=Path("/tmp/audio.wav"),
            history_text="Sin antecedentes.",
        )


def test_consultation_sin_extractor_campos_en_none_y_sin_fields_extracted(
    sample_wording_sections: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§7.5: Sin extractor (por defecto):
    - Cero cambio sobre el comportamiento de plan 45 más allá de la nueva forma de to_dict()
    - Los tres campos nuevos de ClinicalFinding quedan en None por defecto
    - No se emite FIELDS_EXTRACTED
    """
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="consulta por dolor en rodilla derecha desde hace dos semanas",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
        extractor=None,
    )

    owner = AgentId("medico_test")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert result.encounter is not None
    for items in result.encounter.findings.values():
        for f in items:
            assert f.duration is None
            assert f.laterality is None
            assert f.reported_by is None

    assert not any(e.kind == EventKind.FIELDS_EXTRACTED for e in log)


def test_consultation_sin_diarizacion_con_extractor_reported_by_none(
    sample_wording_sections: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§7.5: Sin diarización pero con extractor:
    - reported_by queda None en todos los hallazgos
    - duración y lateralidad sí se extraen
    - FIELDS_EXTRACTED se emite solo con 'duration,laterality'
    """
    segments = (
        TranscriptSegment(
            start=0.0,
            end=3.0,
            text="consulta por dolor en rodilla derecha hace 10 días",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )

    owner = AgentId("medico_test")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert result.encounter is not None
    motivo_findings = result.encounter.findings["motivo_consulta"]
    assert len(motivo_findings) >= 1
    finding = motivo_findings[0]
    assert finding.duration == "10 días"
    assert finding.laterality == "derecho"
    assert finding.reported_by is None

    fields_events = [e for e in log if e.kind == EventKind.FIELDS_EXTRACTED]
    assert len(fields_events) == 1
    ev = fields_events[0]
    assert ev.data == {"fields": "duration,laterality"}


def test_consultation_extractor_sin_aciertos_emite_fields_extracted_con_data_vacia(
    sample_wording_sections: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """§4 de 48-fix-extraccion.md (I-123):
    Con transcripción de líneas sin duración ni lateralidad y sin diarización
    (p. ej. 'se recomienda reposo'): verifica que FIELDS_EXTRACTED se emite con data == {}.
    """
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="se recomienda reposo",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )

    owner = AgentId("medico_test")
    result = consultation.run_with_history(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
    )

    assert result.encounter is not None
    for items in result.encounter.findings.values():
        for f in items:
            assert f.duration is None
            assert f.laterality is None
            assert f.reported_by is None

    fields_events = [e for e in log if e.kind == EventKind.FIELDS_EXTRACTED]
    assert len(fields_events) == 1
    ev = fields_events[0]
    assert ev.owner == owner
    assert ev.tick == 0
    assert ev.data == {}


# ---------------------------------------------------------------------------
# Pruebas §B.2: Revisión del profesional (51-plan-revision-profesional.md)
# ---------------------------------------------------------------------------


def test_consultation_prepare_sin_revision_emite_mismos_eventos_sin_llm_ni_cache(
    sample_wording_review: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 9 (§B.2): Consulta con segmentador y extractor, sin diarizador.
    run_with_history emite [INPUT_OBSERVED, TEXT_SEGMENTED, MODEL_BUILT,
    FIELDS_EXTRACTED, INPUT_OBSERVED, CATALOG_MATCHED].
    prepare emite la misma secuencia, el proveedor espía queda con think_calls == 0
    y cache.load(owner=...) is None.
    """
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="consulta por dolor en rodilla derecha desde hace dos semanas",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()

    # Instancia 1: run_with_history
    log1 = EventLog(window=100)
    spy_provider1 = _SpyCognitionProvider()
    c1 = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=spy_provider1,
        cache=sample_cache,
        log=log1,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )
    owner1 = AgentId("doc_run")
    c1.run_with_history(
        owner=owner1,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión esencial.",
    )
    kinds_run = [e.kind for e in log1]
    expected_sequence = [
        EventKind.INPUT_OBSERVED,
        EventKind.TEXT_SEGMENTED,
        EventKind.MODEL_BUILT,
        EventKind.FIELDS_EXTRACTED,
        EventKind.INPUT_OBSERVED,
        EventKind.CATALOG_MATCHED,
    ]
    assert kinds_run == expected_sequence
    assert spy_provider1.think_calls == 1
    assert sample_cache.load(owner=owner1) is not None

    # Instancia 2: prepare
    log2 = EventLog(window=100)
    spy_provider2 = _SpyCognitionProvider()
    c2 = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=spy_provider2,
        cache=sample_cache,
        log=log2,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )
    owner2 = AgentId("doc_prep")
    prepared = c2.prepare(
        owner=owner2,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión esencial.",
    )
    kinds_prep = [e.kind for e in log2]
    assert kinds_prep == expected_sequence
    assert spy_provider2.think_calls == 0
    assert sample_cache.load(owner=owner2) is None
    assert prepared.encounter is not None
    assert len(prepared.matches) > 0


def test_consultation_review_redacta_sin_modelo_con_codigos_confirmados_o_rechazados(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 10 (§B.2): review() construye el resumen sin llamar al modelo (think_calls == 0).
    Con códigos confirmados, usa confirmed_template y lista solo los confirmados;
    result.matched_codes contiene solo los confirmados; y cache.load(...) coincide con
    result.summary_markdown.
    Con todos los códigos rechazados, el resumen lleva no_confirmed_text.
    """
    catalog = build_index(
        (
            DiagnosticCode("I10", "Hipertensión esencial (primaria)", 3, "ambas"),
            DiagnosticCode("R51", "Cefalea", 3, "ambas"),
        )
    )
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="consulta por dolor de cabeza",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()

    # Parte A: confirma I10, rechaza R51
    log_a = EventLog(window=100)
    spy_provider_a = _SpyCognitionProvider()
    c_a = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=spy_provider_a,
        cache=sample_cache,
        log=log_a,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )
    owner_a = AgentId("doc_rev_a")
    prepared_a = c_a.prepare(
        owner=owner_a,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión esencial y cefalea.",
    )

    def reviewer_a(item: ReviewItem) -> Answer:
        if item.ref == "R51":
            return Answer(Verdict.REJECTED)
        return Answer(Verdict.CONFIRMED)

    result_a = c_a.review(prepared_a, reviewer=reviewer_a)
    assert spy_provider_a.think_calls == 0
    assert result_a.matched_codes == ("I10",)
    assert (
        "Códigos que ya constaban en la historia: I10 (Hipertensión esencial (primaria))"
        in result_a.summary_markdown
    )
    assert "Códigos confirmados por el profesional:" not in result_a.summary_markdown
    assert "R51" not in result_a.summary_markdown
    assert "El profesional no confirmó ningún código." not in result_a.summary_markdown
    assert sample_cache.load(owner=owner_a) == result_a.summary_markdown

    # Parte B: rechaza todos los códigos
    log_b = EventLog(window=100)
    spy_provider_b = _SpyCognitionProvider()
    c_b = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=spy_provider_b,
        cache=sample_cache,
        log=log_b,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )
    owner_b = AgentId("doc_rev_b")
    prepared_b = c_b.prepare(
        owner=owner_b,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión esencial y cefalea.",
    )

    def reviewer_b(item: ReviewItem) -> Answer:
        if item.allowed == (Verdict.CONFIRMED, Verdict.REJECTED):
            return Answer(Verdict.REJECTED)
        return Answer(Verdict.CONFIRMED)

    result_b = c_b.review(prepared_b, reviewer=reviewer_b)
    assert spy_provider_b.think_calls == 0
    assert result_b.matched_codes == ()
    assert "El profesional no confirmó ningún código." in result_b.summary_markdown
    assert "Códigos confirmados por el profesional:" not in result_b.summary_markdown
    assert "Códigos que ya constaban en la historia:" not in result_b.summary_markdown
    assert sample_cache.load(owner=owner_b) == result_b.summary_markdown


def test_consultation_review_wording_incompleto_falla_antes_de_preguntar(
    sample_wording_review: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 11 (§B.2): review() con SummaryWording sin confirmed_template da ValueError
    y el revisor espía registra 0 llamadas.
    """
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="consulta por dolor en rodilla derecha",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()

    c_prep = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )
    prepared = c_prep.prepare(
        owner=AgentId("doc_incomplete"),
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente hipertenso conocido.",
    )

    wording_incompleto = replace(sample_wording_review, confirmed_template=None)
    c_bad = Consultation(
        wording=wording_incompleto,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )

    calls = 0

    def reviewer_spy(item: ReviewItem) -> Answer:
        nonlocal calls
        calls += 1
        return Answer(Verdict.CONFIRMED)

    with pytest.raises(ValueError, match="confirmed_template"):
        c_bad.review(prepared, reviewer=reviewer_spy)

    assert calls == 0


def test_consultation_review_cortada_propaga_y_no_guarda_en_cache(
    sample_wording_review: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 12 (§B.2): review() con revisor que levanta EOFError:
    se propaga y cache.load(owner=...) is None.
    """
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="consulta por dolor en rodilla derecha",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()

    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )
    owner = AgentId("doc_cut")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente hipertenso conocido.",
    )

    def cut_reviewer(item: ReviewItem) -> Answer:
        raise EOFError("corte inesperado")

    with pytest.raises(EOFError):
        consultation.review(prepared, reviewer=cut_reviewer)

    assert sample_cache.load(owner=owner) is None


def test_consultation_refine_edited_recalcula_con_extractor_y_preserva_reported_by(
    sample_wording_review: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 13 (§B.2): _refine_edited:
    - Con extractor: sobre un ClinicalFinding con reported_by="PACIENTE",
      el texto «me duele el ojo izquierdo desde hace tres días» da laterality='izquierdo',
      duration='3 días' y reported_by="PACIENTE".
    - Sin extractor: solo cambia text.
    """
    stt = _SpyWhisperSTT()
    ocr = _DummyOCR()

    # Con extractor
    c_with_extractor = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        diarizer=None,
        segmenter=None,
        extractor=build_health_extractor(),
    )
    f1 = ClinicalFinding(
        text="texto previo",
        section_id="motivo_consulta",
        reported_by="PACIENTE",
    )
    refined1 = c_with_extractor._refine_edited(
        f1,
        "me duele el ojo izquierdo desde hace tres días",
    )
    assert refined1.text == "me duele el ojo izquierdo desde hace tres días"
    assert refined1.duration == "3 días"
    assert refined1.laterality == "izquierdo"
    assert refined1.reported_by == "PACIENTE"

    # Sin extractor
    c_without_extractor = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        diarizer=None,
        segmenter=None,
        extractor=None,
    )
    f2 = ClinicalFinding(
        text="texto previo",
        section_id="motivo_consulta",
        duration="10 días",
        laterality="derecho",
        reported_by="PACIENTE",
    )
    refined2 = c_without_extractor._refine_edited(f2, "nuevo texto sin extractor")
    assert refined2.text == "nuevo texto sin extractor"
    assert refined2.duration == "10 días"
    assert refined2.laterality == "derecho"
    assert refined2.reported_by == "PACIENTE"


def test_consultation_review_sin_segmentador_lanza_value_error(
    sample_wording_review: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 14 (§B.2): review() con PreparedConsultation de encounter=None da ValueError."""
    stt = _SpyWhisperSTT()
    ocr = _DummyOCR()
    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        diarizer=None,
        segmenter=None,
        extractor=None,
    )
    prepared_sin_encounter = PreparedConsultation(
        owner=AgentId("doc_no_encounter"),
        transcript="transcripción",
        history="historia",
        matches=(),
        grouped_sections=None,
        encounter=None,
    )

    with pytest.raises(ValueError, match="revisión activa pero no hay segmentador configurado"):
        consultation.review(
            prepared_sin_encounter,
            reviewer=lambda item: Answer(Verdict.CONFIRMED),
        )


def test_default_summary_wording_faltan_claves_revision_y_pack_real(tmp_path: Path) -> None:
    """Caso 15 (§B.2): default_summary_wording(path) con un YAML al que le falta
    confirmed_template da ValueError que la nombra; lo mismo con no_confirmed_text.
    default_summary_wording() (el pack real) carga los dos campos con valor no vacío.
    """
    base_dict = {
        "system_prompt": "Organiza:",
        "said_template": "Dijo: {}",
        "history_template": "Historia: {}",
        "matched_template": "CIE: {}",
        "no_match_text": "Sin códigos",
        "professional_label": "MÉDICO",
        "patient_label": "PACIENTE",
        "section_template": "### {}\n{}",
        "confirmed_template": "Códigos confirmados: {}",
        "no_confirmed_text": "Sin confirmados",
    }

    # Falta confirmed_template
    dict_no_confirmed = dict(base_dict)
    del dict_no_confirmed["confirmed_template"]
    path_no_confirmed = tmp_path / "wording_no_confirmed.yaml"
    path_no_confirmed.write_text(yaml.dump(dict_no_confirmed), encoding="utf-8")

    with pytest.raises(ValueError, match="confirmed_template"):
        default_summary_wording(path_no_confirmed)

    # Falta no_confirmed_text
    dict_no_confirmed_text = dict(base_dict)
    del dict_no_confirmed_text["no_confirmed_text"]
    path_no_confirmed_text = tmp_path / "wording_no_confirmed_text.yaml"
    path_no_confirmed_text.write_text(yaml.dump(dict_no_confirmed_text), encoding="utf-8")

    with pytest.raises(ValueError, match="no_confirmed_text"):
        default_summary_wording(path_no_confirmed_text)

    # Pack real
    real_wording = default_summary_wording()
    assert real_wording.confirmed_template is not None
    assert len(real_wording.confirmed_template.strip()) > 0
    assert real_wording.no_confirmed_text is not None
    assert len(real_wording.no_confirmed_text.strip()) > 0


def test_consultation_review_completa_registra_metricas_con_reloj_inyectado(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 7 (§B.2): review() con review_metrics sobre tmp_path y un reloj doble
    que devuelve 10.0 y después 52.5: el store tiene un registro con completed=True,
    seconds=42.5, y conteos iguales a los que da review_record sobre los ITEM_REVIEWED del log.
    """
    times = iter([10.0, 52.5])
    metrics_path = tmp_path / "revisiones.jsonl"
    store = ReviewMetricsStore(metrics_path)

    catalog = build_index((DiagnosticCode("I10", "Hipertensión esencial (primaria)", 3, "ambas"),))
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="consulta por dolor de cabeza",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
        review_metrics=store,
        clock=lambda: next(times),
    )
    owner = AgentId("doc_full_metrics")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión esencial.",
    )

    def sample_reviewer(item: ReviewItem) -> Answer:
        return Answer(Verdict.CONFIRMED)

    consultation.review(prepared, reviewer=sample_reviewer)

    records = store.read()
    assert len(records) == 1
    r = records[0]
    assert r.completed is True
    assert r.seconds == 42.5

    expected_rec = review_record(
        [e for e in log if e.kind is EventKind.ITEM_REVIEWED],
        completed=True,
        seconds=42.5,
    )
    assert r.findings == expected_rec.findings
    assert r.codes == expected_rec.codes
    assert r.sections == expected_rec.sections


def test_consultation_review_cortada_con_eof_error_registra_incompleta_y_no_cachea(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 8 (§B.2): Cortada con EOFError en el tercer ítem:
    la excepción se propaga, el store tiene un registro con completed=False
    y dos veredictos, y la caché no tiene nada para ese owner.
    """
    times = iter([5.0, 15.0])
    metrics_path = tmp_path / "revisiones_eof.jsonl"
    store = ReviewMetricsStore(metrics_path)

    catalog = build_index((DiagnosticCode("I10", "Hipertensión esencial (primaria)", 3, "ambas"),))
    segments = (
        TranscriptSegment(start=0.0, end=1.0, text="consulta hoy por dolor de cabeza"),
        TranscriptSegment(start=1.0, end=2.0, text="examen fisico con dolor a la palpacion"),
        TranscriptSegment(start=2.0, end=3.0, text="se receta reposo y medicacion"),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
        review_metrics=store,
        clock=lambda: next(times),
    )
    owner = AgentId("doc_eof_cut")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión esencial.",
    )

    call_count = 0

    def cut_reviewer(item: ReviewItem) -> Answer:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise EOFError("corte en tercer ítem")
        return Answer(Verdict.CONFIRMED)

    with pytest.raises(EOFError, match="corte en tercer ítem"):
        consultation.review(prepared, reviewer=cut_reviewer)

    assert sample_cache.load(owner=owner) is None

    records = store.read()
    assert len(records) == 1
    r = records[0]
    assert r.completed is False
    assert r.seconds == 10.0
    total_veredictos = sum(r.findings.values()) + sum(r.codes.values())
    assert total_veredictos == 2


def test_consultation_review_cortada_con_keyboard_interrupt_registra_incompleta_y_no_cachea(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 9 (§B.2): Cortada con KeyboardInterrupt en el tercer ítem:
    igual que el 8 (la excepción se propaga, completed=False, 2 veredictos, sin caché).
    """
    times = iter([1.0, 8.0])
    metrics_path = tmp_path / "revisiones_sigint.jsonl"
    store = ReviewMetricsStore(metrics_path)

    catalog = build_index((DiagnosticCode("I10", "Hipertensión esencial (primaria)", 3, "ambas"),))
    segments = (
        TranscriptSegment(start=0.0, end=1.0, text="consulta hoy por dolor de cabeza"),
        TranscriptSegment(start=1.0, end=2.0, text="examen fisico con dolor a la palpacion"),
        TranscriptSegment(start=2.0, end=3.0, text="se receta reposo y medicacion"),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
        review_metrics=store,
        clock=lambda: next(times),
    )
    owner = AgentId("doc_sigint_cut")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión esencial.",
    )

    call_count = 0

    def cut_reviewer(item: ReviewItem) -> Answer:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise KeyboardInterrupt("corte por señal")
        return Answer(Verdict.CONFIRMED)

    with pytest.raises(KeyboardInterrupt):
        consultation.review(prepared, reviewer=cut_reviewer)

    assert sample_cache.load(owner=owner) is None

    records = store.read()
    assert len(records) == 1
    r = records[0]
    assert r.completed is False
    assert r.seconds == 7.0
    total_veredictos = sum(r.findings.values()) + sum(r.codes.values())
    assert total_veredictos == 2


def test_consultation_review_wording_incompleto_no_crea_archivo_ni_abre_medicion(
    sample_wording_review: SummaryWording,
    sample_catalog_index: CatalogIndex,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 10 (§B.2): Wording incompleto (sin confirmed_template):
    review() da ValueError y el store queda vacío (el archivo ni existe).
    """
    wording_incompleto = replace(sample_wording_review, confirmed_template=None)
    metrics_path = tmp_path / "no_existe_metrics.jsonl"
    store = ReviewMetricsStore(metrics_path)

    consultation = Consultation(
        wording=wording_incompleto,
        stt=_SpyWhisperSTT(),
        ocr=_DummyOCR(),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        diarizer=None,
        segmenter=ClinicalSegmenter(load_sections()),
        extractor=build_health_extractor(),
        review_metrics=store,
    )
    owner = AgentId("doc_bad_wording")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Historia de prueba",
    )

    with pytest.raises(ValueError, match="confirmed_template"):
        consultation.review(prepared, reviewer=lambda item: Answer(Verdict.CONFIRMED))

    assert not metrics_path.exists()
    assert store.read() == ()


def test_consultation_dos_revisiones_seguidas_mismo_log_segundo_registro_aislado(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 11 (§B.2): Dos revisiones seguidas con la misma Consultation (mismo log):
    el segundo registro solo cuenta sus propios veredictos.
    """
    clock_values = [0.0, 10.0, 20.0, 35.0]
    clock_iter = iter(clock_values)

    metrics_path = tmp_path / "dos_revisiones.jsonl"
    store = ReviewMetricsStore(metrics_path)

    catalog = build_index((DiagnosticCode("I10", "Hipertensión esencial (primaria)", 3, "ambas"),))
    segments = (TranscriptSegment(start=0.0, end=1.0, text="consulta por dolor de cabeza"),)
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
        review_metrics=store,
        clock=lambda: next(clock_iter),
    )

    # Primera consulta
    owner1 = AgentId("doc_seq_1")
    prep1 = consultation.prepare(
        owner=owner1,
        audio_path=Path("/tmp/a1.wav"),
        history_text="Historia con hipertensión esencial",
    )
    consultation.review(prep1, reviewer=lambda item: Answer(Verdict.CONFIRMED))

    # Segunda consulta
    owner2 = AgentId("doc_seq_2")
    prep2 = consultation.prepare(
        owner=owner2,
        audio_path=Path("/tmp/a2.wav"),
        history_text="Historia con hipertensión esencial",
    )
    consultation.review(prep2, reviewer=lambda item: Answer(Verdict.REJECTED))

    records = store.read()
    assert len(records) == 2
    r1, r2 = records
    assert r1.completed is True
    assert r1.seconds == 10.0
    assert r2.completed is True
    assert r2.seconds == 15.0

    # r1 tuvo confirmados, r2 tuvo rechazados. r2 no debe tener confirmados de r1!
    assert r1.findings.get("confirmed", 0) > 0
    assert r1.findings.get("rejected", 0) == 0
    assert r2.findings.get("confirmed", 0) == 0
    assert r2.findings.get("rejected", 0) > 0
    total_r2 = sum(r2.findings.values()) + sum(r2.codes.values())
    total_r1 = sum(r1.findings.values()) + sum(r1.codes.values())
    assert total_r2 == total_r1


def test_consultation_review_sin_store_no_cambia_nada_y_journal_identico(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 12 (§B.2): Con review_metrics=None no se crea ningún archivo;
    y con los mismos dobles, list(log.to_jsonl()) tras review() es idéntico con store y sin store.
    """
    catalog = build_index((DiagnosticCode("I10", "Hipertensión esencial (primaria)", 3, "ambas"),))
    segments = (TranscriptSegment(start=0.0, end=1.0, text="consulta por dolor de cabeza"),)

    owner = AgentId("same_owner_for_journal")

    # Ejecución A: con store
    log_a = EventLog(window=100)
    store_path = tmp_path / "metrics_comp.jsonl"
    store = ReviewMetricsStore(store_path)
    c_a = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=segments),
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log_a,
        diarizer=None,
        segmenter=ClinicalSegmenter(load_sections()),
        extractor=build_health_extractor(),
        review_metrics=store,
        clock=lambda: 10.0,
    )
    p_a = c_a.prepare(
        owner=owner,
        audio_path=Path("/tmp/a.wav"),
        history_text="Historia con hipertensión esencial",
    )
    c_a.review(p_a, reviewer=lambda item: Answer(Verdict.CONFIRMED))

    # Ejecución B: sin store (review_metrics=None)
    log_b = EventLog(window=100)
    c_b = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=segments),
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log_b,
        diarizer=None,
        segmenter=ClinicalSegmenter(load_sections()),
        extractor=build_health_extractor(),
        review_metrics=None,
    )
    p_b = c_b.prepare(
        owner=owner,
        audio_path=Path("/tmp/a.wav"),
        history_text="Historia con hipertensión esencial",
    )
    c_b.review(p_b, reviewer=lambda item: Answer(Verdict.CONFIRMED))

    assert not (tmp_path / "sin_store_file.jsonl").exists()
    assert list(log_a.to_jsonl()) == list(log_b.to_jsonl())


# ---------------------------------------------------------------------------
# Plan 59: Textos codificables y exclusión de medicación (§B.3)
# ---------------------------------------------------------------------------


def test_consultation_codable_texts_con_y_sin_segmentacion(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 12 (§B.3): _codable_texts con secciones devuelve solo las de secciones
    codificables más la historia (sin concatenar en un solo texto); sin segmentación,
    devuelve las líneas de transcripción y de historia."""
    catalog = build_index(())
    consultation = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=()),
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
    )

    sec_motivo = SectionSpec(
        section_id="motivo", label="Motivo", trigger_phrases=(), codable=True, in_note=True
    )
    sec_plan = SectionSpec(
        section_id="plan", label="Plan", trigger_phrases=(), codable=False, in_note=True
    )
    grouped = (
        (sec_motivo, ("dolor agudo", "cefalea")),
        (sec_plan, ("ibuprofeno cada 8h",)),
    )

    # Con segmentación: filtra la sección no codificable y devuelve dos tuplas: (hoy, historia)
    hoy_con_seg, hist_con_seg = consultation._codable_texts(
        "transcripcion original", "hipertensión previa\n\ndiabetes", grouped
    )
    assert hoy_con_seg == _codables("dolor agudo", "cefalea")
    assert hist_con_seg == _codables("hipertensión previa", "diabetes")

    # Sin segmentación: devuelve cada línea no vacía de transcripción y de historia por separado
    hoy_sin_seg, hist_sin_seg = consultation._codable_texts(
        "dolor de pecho\n\ndisnea", "hipertensión previa\n", None
    )
    assert hoy_sin_seg == _codables("dolor de pecho", "disnea")
    assert hist_sin_seg == _codables("hipertensión previa")


def test_consultation_codable_texts_con_lexico_amplia_hoy_e_historia(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 9 (§B.2): _codable_texts con léxico añade los términos clínicos
    detrás de lo hablado y también detrás de la historia."""
    lexicon: Lexicon = (
        ("azucar", "diabetes mellitus"),
        ("tension alta", "hipertensión esencial"),
    )
    consultation = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=()),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        lexicon=lexicon,
    )

    sec_motivo = SectionSpec(
        section_id="motivo", label="Motivo", trigger_phrases=(), codable=True, in_note=True
    )
    grouped = ((sec_motivo, ("sufro de tension alta",)),)

    hoy, historia = consultation._codable_texts(
        "transcripcion", "antecedente de azucar en sangre", grouped
    )
    # Lo hablado lleva su término clínico traducido detrás
    assert hoy == (
        CodableText(query="sufro de tension alta", source="sufro de tension alta"),
        CodableText(query="hipertensión esencial", source="sufro de tension alta"),
    )
    # La historia también lleva su término clínico traducido detrás
    assert historia == (
        CodableText(
            query="antecedente de azucar en sangre", source="antecedente de azucar en sangre"
        ),
        CodableText(query="diabetes mellitus", source="antecedente de azucar en sangre"),
    )


def test_consultation_sin_lexico_mantiene_comportamiento_plan_62(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 10 (§B.2): Con lexicon=() (o por defecto) _codable_texts no añade
    equivalentes y se comporta idéntico al plan 62."""
    c_defecto = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=()),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=EventLog(),
    )
    c_vacio = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=()),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        lexicon=(),
    )

    sec = SectionSpec(section_id="m", label="M", trigger_phrases=(), codable=True, in_note=True)
    grouped = ((sec, ("tengo azucar y tension alta",)),)
    historia_txt = "azucar descompensada"

    hoy1, hist1 = c_defecto._codable_texts("t", historia_txt, grouped)
    hoy2, hist2 = c_vacio._codable_texts("t", historia_txt, grouped)

    assert hoy1 == _codables("tengo azucar y tension alta")
    assert hist1 == _codables("azucar descompensada")
    assert hoy1 == hoy2
    assert hist1 == hist2


def test_consultation_codable_texts_no_repite_terminos_clinicos(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 11 (§B.2): Los términos clínicos añadidos no se repiten aunque
    múltiples hallazgos o líneas hagan referencia al mismo concepto."""
    lexicon: Lexicon = (
        ("azucar", "diabetes mellitus"),
        ("diabetes", "diabetes mellitus"),
    )
    consultation = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=()),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        lexicon=lexicon,
    )

    sec = SectionSpec(section_id="m", label="M", trigger_phrases=(), codable=True, in_note=True)
    grouped = ((sec, ("tengo problemas con el azucar", "control de diabetes")),)
    historia_txt = "azucar alta\ndiabetes previa"

    hoy, historia = consultation._codable_texts("t", historia_txt, grouped)

    # Solo debe haber una aparición del término clínico
    assert hoy == (
        CodableText(query="tengo problemas con el azucar", source="tengo problemas con el azucar"),
        CodableText(query="control de diabetes", source="control de diabetes"),
        CodableText(query="diabetes mellitus", source="tengo problemas con el azucar"),
    )
    assert historia == (
        CodableText(query="azucar alta", source="azucar alta"),
        CodableText(query="diabetes previa", source="diabetes previa"),
        CodableText(query="diabetes mellitus", source="azucar alta"),
    )


def test_consultation_review_separa_codigos_por_origen_y_both_en_ambas(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 6 (§B.2): El resumen tras revisar: los confirmados de hoy en una línea,
    los que ya constaban en otra, y un código 'both' en las dos. Sin ninguno
    confirmado, no_confirmed_text y ninguna de las dos líneas."""
    catalog = build_index(
        (
            DiagnosticCode("T01", "asma bronquial", 3, "ambas"),
            DiagnosticCode("H01", "hipertensión esencial", 3, "ambas"),
            DiagnosticCode("B01", "diabetes mellitus", 3, "ambas"),
        )
    )
    # T01 solo en transcripción ("asma bronquial") -> TODAY
    # H01 solo en historia ("hipertensión esencial") -> HISTORY
    # B01 en ambos ("diabetes mellitus") -> BOTH
    transcript_text = "motivo de consulta: asma bronquial y diabetes mellitus"
    segments = (TranscriptSegment(start=0.0, end=2.0, text=transcript_text),)
    full_text = transcript_text
    stt = _SpyWhisperSTT(segments=segments, full_text=full_text)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()

    # Escenario 1: Todos confirmados
    log1 = EventLog(window=100)
    c1 = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log1,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )
    owner1 = AgentId("doc_origen_1")
    prep1 = c1.prepare(
        owner=owner1,
        audio_path=Path("/tmp/audio.wav"),
        history_text="hipertensión esencial\ndiabetes mellitus",
    )
    res1 = c1.review(prep1, reviewer=lambda item: Answer(Verdict.CONFIRMED))

    assert "Códigos confirmados por el profesional: " in res1.summary_markdown
    assert "Códigos que ya constaban en la historia: " in res1.summary_markdown
    assert "El profesional no confirmó ningún código." not in res1.summary_markdown

    # Extraer las dos líneas del resumen
    lineas = res1.summary_markdown.splitlines()
    linea_hoy = next(line for line in lineas if "Códigos confirmados por el profesional:" in line)
    linea_hist = next(line for line in lineas if "Códigos que ya constaban en la historia:" in line)

    # T01 (today) solo en linea_hoy
    assert "T01" in linea_hoy
    assert "T01" not in linea_hist

    # H01 (history) solo en linea_hist
    assert "H01" not in linea_hoy
    assert "H01" in linea_hist

    # B01 (both) DEBE estar en las DOS líneas
    assert "B01" in linea_hoy
    assert "B01" in linea_hist

    # Escenario 2: Todos los códigos rechazados
    log2 = EventLog(window=100)
    c2 = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log2,
        diarizer=None,
        segmenter=segmenter,
        extractor=extractor,
    )
    owner2 = AgentId("doc_origen_2")
    prep2 = c2.prepare(
        owner=owner2,
        audio_path=Path("/tmp/audio.wav"),
        history_text="hipertensión esencial\ndiabetes mellitus",
    )

    def rechaza_codigos(item: ReviewItem) -> Answer:
        if item.kind == ItemKind.CODE:
            return Answer(Verdict.REJECTED)
        return Answer(Verdict.CONFIRMED)

    res2 = c2.review(prep2, reviewer=rechaza_codigos)
    assert "El profesional no confirmó ningún código." in res2.summary_markdown
    assert "Códigos confirmados por el profesional:" not in res2.summary_markdown
    assert "Códigos que ya constaban en la historia:" not in res2.summary_markdown


def test_consultation_con_segmentador_excluye_codigos_de_medicacion(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 13 (§B.3): una consulta completa con segmentador excluye códigos
    provenientes de secciones no codificables (p.ej. medicamentos)."""
    # Catálogo con código para motivo (cefalea) y código para medicamento (amoxicilina)
    catalog = build_index(
        (
            DiagnosticCode("R51", "cefalea", 3, "ambas"),
            DiagnosticCode("T88.7", "amoxicilina", 3, "ambas"),
        )
    )
    # Segmentos que el segmentador real clasifica en motivo_consulta y medicamentos
    segments = (
        TranscriptSegment(start=0.0, end=2.0, text="consulta hoy por cefalea"),
        TranscriptSegment(start=2.0, end=4.0, text="se receta amoxicilina"),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        segmenter=segmenter,
        extractor=build_health_extractor(),
        match_threshold=0.3,
    )
    owner = AgentId("doc_med_excl")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="",
    )
    codigos_cotejados = [m.diagnostic.code for m in prepared.matches]
    # R51 (de motivo_consulta, codable=True) debe aparecer
    assert "R51" in codigos_cotejados
    # T88.7 (de medicamentos, codable=False) NO debe aparecer
    assert "T88.7" not in codigos_cotejados


def test_consultation_codable_texts_con_history_layout_filtra_solo_lineas_codificables(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 8 (§B.2): _codable_texts con history_layout: la historia aporta
    solo sus líneas codificables; el lado hablado no cambia."""
    layout = HistoryLayout(
        sections=(
            HistorySection(heading="ANTECEDENTES PERSONALES", codable=True),
            HistorySection(heading="ANTECEDENTES FAMILIARES", codable=False),
        ),
        ignore=("FECHA:",),
    )
    consultation = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=()),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        history_layout=layout,
    )
    sec = SectionSpec(section_id="m", label="M", trigger_phrases=(), codable=True, in_note=True)
    grouped = ((sec, ("dolor agudo de cabeza",)),)
    historia = """FECHA: 2026-09-24
ANTECEDENTES PERSONALES
Hipertensión esencial
ANTECEDENTES FAMILIARES
Padre con diabetes
"""
    hoy, hist = consultation._codable_texts("transcripcion", historia, grouped)
    assert hoy == _codables("dolor agudo de cabeza")
    assert hist == _codables("Hipertensión esencial")


def test_consultation_sin_history_layout_mantiene_comportamiento_previo(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 9 (§B.2): Con history_layout=None (o por defecto), la historia
    se divide línea a línea idéntico a antes de esta unidad."""
    c_defecto = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=()),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=EventLog(),
    )
    c_none = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(segments=()),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        history_layout=None,
    )
    historia = """Centro Médico X
Sexo: Femenino
ANTECEDENTES PERSONALES
Diabetes tipo 2
"""
    sec = SectionSpec(section_id="m", label="M", trigger_phrases=(), codable=True, in_note=True)
    grouped = ((sec, ("habla de hoy",)),)

    hoy1, hist1 = c_defecto._codable_texts("t", historia, grouped)
    hoy2, hist2 = c_none._codable_texts("t", historia, grouped)

    assert hist1 == _codables(
        "Centro Médico X",
        "Sexo: Femenino",
        "ANTECEDENTES PERSONALES",
        "Diabetes tipo 2",
    )
    assert hist1 == hist2
    assert hoy1 == hoy2


def test_consultation_review_resumen_conserva_historia_entera(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 10 (§B.2): El resumen tras revisar sigue llevando la historia entera:
    filtrar es para cotejar, no para redactar."""
    layout = HistoryLayout(
        sections=(
            HistorySection(heading="ANTECEDENTES PERSONALES", codable=True),
            HistorySection(heading="ANTECEDENTES FAMILIARES", codable=False),
        ),
        ignore=("PAGINA",),
    )
    segments = (TranscriptSegment(start=0.0, end=2.0, text="consulta por dolor de cabeza"),)
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        segmenter=segmenter,
        extractor=build_health_extractor(),
        history_layout=layout,
    )
    owner = AgentId("doc_hist_full")
    historia_entera = """Centro Médico Demo
PAGINA 1 DE 1
ANTECEDENTES PERSONALES
Hipertensión arterial
ANTECEDENTES FAMILIARES
Madre con infarto
"""
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text=historia_entera,
    )
    result = consultation.review(prepared, reviewer=lambda item: Answer(Verdict.CONFIRMED))

    # El extracto de historia y el resumen final deben contener la historia completa
    assert result.history_excerpt == historia_entera
    assert "Centro Médico Demo" in result.summary_markdown
    assert "PAGINA 1 DE 1" in result.summary_markdown
    assert "ANTECEDENTES FAMILIARES" in result.summary_markdown
    assert "Madre con infarto" in result.summary_markdown


# ---------------------------------------------------------------------------
# Pruebas B.5 & B.7: Negación clínica y evidencia (68-plan-negacion-evidencia.md)
# ---------------------------------------------------------------------------


def test_consultation_frase_negada_no_propone_codigo(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 25 (§B.5): Una frase negada no propone su código; con frase afirmada sí."""
    catalog = build_index((DiagnosticCode("R07", "dolor toracico", 3, "test"),))
    neg_cat = load_negation()

    # Frase negada: no propone
    stt_neg = _SpyWhisperSTT(full_text="No refiere dolor toracico")
    c_neg = Consultation(
        wording=sample_wording_review,
        stt=stt_neg,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        negation=neg_cat,
        match_threshold=0.3,
    )
    prep_neg = c_neg.prepare(
        owner=AgentId("doc_neg"), audio_path=Path("/tmp/a.wav"), history_text=""
    )
    assert prep_neg.matches == ()

    # Frase afirmada: sí propone
    stt_aff = _SpyWhisperSTT(full_text="Refiere dolor toracico")
    c_aff = Consultation(
        wording=sample_wording_review,
        stt=stt_aff,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        negation=neg_cat,
        match_threshold=0.3,
    )
    prep_aff = c_aff.prepare(
        owner=AgentId("doc_aff"), audio_path=Path("/tmp/a.wav"), history_text=""
    )
    assert len(prep_aff.matches) == 1
    assert prep_aff.matches[0].diagnostic.code == "R07"


def test_consultation_frase_mixta_propone_solo_lo_afirmado(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 26 (§B.5): Frase mixta conserva el diagnóstico afirmado y descarta el negado."""
    catalog = build_index(
        (
            DiagnosticCode("R51", "dolor de cabeza", 3, "test"),
            DiagnosticCode("R50", "fiebre", 3, "test"),
        )
    )
    stt = _SpyWhisperSTT(full_text="Tiene dolor de cabeza pero no fiebre")
    c = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        negation=load_negation(),
        match_threshold=0.3,
    )
    prep = c.prepare(owner=AgentId("doc_mix"), audio_path=Path("/tmp/a.wav"), history_text="")
    codigos = {m.diagnostic.code for m in prep.matches}
    assert "R51" in codigos
    assert "R50" not in codigos


def test_consultation_negacion_se_aplica_a_historia(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 27 (§B.5): La negación se aplica también a las líneas de historia clínica."""
    catalog = build_index((DiagnosticCode("E11", "diabetes mellitus", 3, "test"),))
    stt = _SpyWhisperSTT(full_text="")
    c = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        negation=load_negation(),
        match_threshold=0.3,
    )
    prep = c.prepare(
        owner=AgentId("doc_hist_neg"),
        audio_path=Path("/tmp/a.wav"),
        history_text="Sin antecedentes de diabetes",
    )
    assert not any(m.diagnostic.code == "E11" for m in prep.matches)


def test_consultation_lexico_no_cuela_lo_negado(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 28 (§B.5): El léxico no reintroduce términos negados."""
    catalog = build_index((DiagnosticCode("E11", "diabetes mellitus", 3, "test"),))
    lexicon: Lexicon = (("azucar", "diabetes mellitus"),)
    stt = _SpyWhisperSTT(full_text="No sufro del azucar")
    c = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        lexicon=lexicon,
        negation=load_negation(),
        match_threshold=0.3,
    )
    prep = c.prepare(owner=AgentId("doc_lex_neg"), audio_path=Path("/tmp/a.wav"), history_text="")
    assert not any(m.diagnostic.code == "E11" for m in prep.matches)


def test_consultation_evidencia_en_review_item(
    sample_cache: ConsultationCache,
) -> None:
    """Caso 29 (§B.5): La evidencia llega al ReviewItem si hay plantilla;
    si no, solo descripción."""
    catalog = build_index((DiagnosticCode("R07", "dolor toracico", 3, "test"),))
    segments = (TranscriptSegment(0.0, 1.0, "consulta hoy por dolor toracico pero no fiebre"),)
    stt = _SpyWhisperSTT(segments=segments)

    # Con plantilla
    wording_con_ev = default_summary_wording()
    c_con = Consultation(
        wording=wording_con_ev,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        negation=load_negation(),
        segmenter=ClinicalSegmenter(load_sections()),
        extractor=build_health_extractor(),
        match_threshold=0.3,
    )
    prep_con = c_con.prepare(
        owner=AgentId("doc_ev_1"), audio_path=Path("/tmp/a.wav"), history_text=""
    )
    items_con: list[ReviewItem] = []

    def _rev_con(item: ReviewItem) -> Answer:
        items_con.append(item)
        return Answer(Verdict.CONFIRMED)

    c_con.review(prep_con, reviewer=_rev_con)
    item_cod_con = next(it for it in items_con if it.kind == ItemKind.CODE)
    assert "dicho: «consulta hoy por dolor toracico pero no fiebre»" in item_cod_con.text

    # Sin plantilla
    wording_sin_ev = replace(wording_con_ev, evidence_template=None)
    c_sin = Consultation(
        wording=wording_sin_ev,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        negation=load_negation(),
        segmenter=ClinicalSegmenter(load_sections()),
        extractor=build_health_extractor(),
        match_threshold=0.3,
    )
    prep_sin = c_sin.prepare(
        owner=AgentId("doc_ev_2"), audio_path=Path("/tmp/a.wav"), history_text=""
    )
    items_sin: list[ReviewItem] = []

    def _rev_sin(item: ReviewItem) -> Answer:
        items_sin.append(item)
        return Answer(Verdict.CONFIRMED)

    c_sin.review(prep_sin, reviewer=_rev_sin)
    item_cod_sin = next(it for it in items_sin if it.kind == ItemKind.CODE)
    assert item_cod_sin.text == "dolor toracico"


def test_consultation_hallazgo_enteramente_negado_marcado_mixto_no(
    sample_cache: ConsultationCache,
) -> None:
    """Caso 30 (§B.5): El hallazgo enteramente negado se marca con sufijo; el mixto no."""
    wording = default_summary_wording()
    # Dos líneas: una enteramente negada y una mixta
    segments = (
        TranscriptSegment(0.0, 1.0, "No refiere fiebre"),
        TranscriptSegment(1.0, 2.0, "Tiene dolor de cabeza pero no náuseas"),
    )
    stt = _SpyWhisperSTT(segments=segments)
    c = Consultation(
        wording=wording,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        negation=load_negation(),
        extractor=build_health_extractor(),
        segmenter=ClinicalSegmenter(load_sections()),
    )
    prep = c.prepare(owner=AgentId("doc_find_neg"), audio_path=Path("/tmp/a.wav"), history_text="")
    items: list[ReviewItem] = []

    def _rev(item: ReviewItem) -> Answer:
        items.append(item)
        return Answer(Verdict.CONFIRMED)

    c.review(prep, reviewer=_rev)
    find_items = [it for it in items if it.kind == ItemKind.FINDING]

    item_fiebre = next(it for it in find_items if "No refiere fiebre" in it.text)
    assert item_fiebre.text == f"No refiere fiebre · {wording.negated_suffix}"

    item_cabeza = next(it for it in find_items if "Tiene dolor de cabeza" in it.text)
    assert item_cabeza.text == "Tiene dolor de cabeza pero no náuseas"
    assert wording.negated_suffix is not None
    assert wording.negated_suffix not in item_cabeza.text


def test_consultation_fields_extracted_incluye_negated(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 31 (§B.5): FIELDS_EXTRACTED incluye negated en fields en orden alfabético."""
    stt = _SpyWhisperSTT(
        segments=(
            TranscriptSegment(
                0.0, 1.0, "MÉDICO: no refiere dolor en ojo derecho desde hace 2 semanas"
            ),
        )
    )
    log = EventLog()
    c = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=sample_cache,
        log=log,
        negation=load_negation(),
        extractor=build_health_extractor(),
        segmenter=ClinicalSegmenter(load_sections()),
    )
    prep = c.prepare(owner=AgentId("doc_evt_neg"), audio_path=Path("/tmp/a.wav"), history_text="")
    evt = next(e for e in log if e.kind == EventKind.FIELDS_EXTRACTED)
    assert evt.data["fields"] == "duration,laterality,negated,reported_by"
    assert prep.encounter is not None
    findings = [f for items in prep.encounter.findings.values() for f in items]
    assert any(f.negated is not None for f in findings)


def test_consultation_sin_negation_inyectado_default_no_cambia_nada(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 32 (§B.5): Sin negation inyectado (default) produce mismos códigos
    para consulta sin negaciones."""
    catalog = build_index((DiagnosticCode("R51", "cefalea aguda", 3, "test"),))
    stt = _SpyWhisperSTT(full_text="cefalea aguda intensa")
    c_default = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        match_threshold=0.3,
    )
    c_explicit = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        negation=load_negation(),
        match_threshold=0.3,
    )
    p1 = c_default.prepare(
        owner=AgentId("doc_def_1"), audio_path=Path("/tmp/a.wav"), history_text=""
    )
    p2 = c_explicit.prepare(
        owner=AgentId("doc_def_2"), audio_path=Path("/tmp/a.wav"), history_text=""
    )
    assert [m.diagnostic.code for m in p1.matches] == [m.diagnostic.code for m in p2.matches]
    assert [m.score for m in p1.matches] == [m.score for m in p2.matches]


def test_default_summary_wording_claves_obligatorias(tmp_path: Path) -> None:
    """Caso 36 (§B.7): Faltan evidence_template o negated_suffix levanta
    ValueError que las nombra."""
    base_dict = {
        "system_prompt": "Organiza:",
        "said_template": "Dijo: {}",
        "history_template": "Historia: {}",
        "matched_template": "CIE: {}",
        "no_match_text": "Sin códigos",
        "professional_label": "MÉDICO",
        "patient_label": "PACIENTE",
        "section_template": "### {}\n{}",
        "confirmed_template": "Códigos confirmados: {}",
        "no_confirmed_text": "Sin confirmados",
        "history_codes_template": "Hist: {}",
        "known_code_suffix": "previo",
        "evidence_template": "{} · dicho: «{}»",
        "negated_suffix": "negado",
    }

    # Falta evidence_template
    d1 = dict(base_dict)
    del d1["evidence_template"]
    p1 = tmp_path / "w1.yaml"
    p1.write_text(yaml.dump(d1), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_template"):
        default_summary_wording(p1)

    # Falta negated_suffix
    d2 = dict(base_dict)
    del d2["negated_suffix"]
    p2 = tmp_path / "w2.yaml"
    p2.write_text(yaml.dump(d2), encoding="utf-8")
    with pytest.raises(ValueError, match="negated_suffix"):
        default_summary_wording(p2)

    # No es cadena
    d3 = dict(base_dict)
    d3["evidence_template"] = 123  # type: ignore[assignment]
    p3 = tmp_path / "w3.yaml"
    p3.write_text(yaml.dump(d3), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_template"):
        default_summary_wording(p3)


def test_default_summary_wording_pack_real_las_trae() -> None:
    """Caso 37 (§B.7): El pack real trae evidence_template y negated_suffix en SummaryWording."""
    w = default_summary_wording()
    assert w.evidence_template == "{} · dicho: «{}»"
    assert w.negated_suffix == "negado"


# ---------------------------------------------------------------------------
# Plan 69: Catálogo de referencia en consulta (§B.4)
# ---------------------------------------------------------------------------


def test_consultation_catalogo_referencia_propone_codigos_cuatro_caracteres(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 16 (§B.4): Una consulta con el catálogo de referencia propone códigos
    de exactamente cuatro caracteres, sin códigos de 3 caracteres ni rangos con guion."""
    catalog = build_index(
        (
            DiagnosticCode("I10X", "HIPERTENSION ESENCIAL", 3, "minsalud"),
            DiagnosticCode("E140", "DIABETES MELLITUS CON COMA", 3, "minsalud"),
            DiagnosticCode("R51X", "CEFALEA", 3, "minsalud"),
        )
    )
    stt = _SpyWhisperSTT(full_text="Refiere cefalea e hipertensión esencial")
    c = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        match_threshold=0.3,
    )
    prep = c.prepare(owner=AgentId("doc_ref_test"), audio_path=Path("/tmp/a.wav"), history_text="")
    assert len(prep.matches) > 0
    for match in prep.matches:
        code = match.diagnostic.code
        assert len(code) == 4, f"Código {code} no tiene 4 caracteres"
        assert "-" not in code, f"Código {code} contiene guion de rango"


# ---------------------------------------------------------------------------
# Plan 70: Retractación de negaciones en la consulta (§B.4)
# ---------------------------------------------------------------------------


def test_consultation_pareja_enumeracion_negacion_no_propone_codigo(
    sample_wording_review: SummaryWording, sample_cache: ConsultationCache
) -> None:
    """Caso 17 (§B.4): Transcripción de dos líneas y un catálogo de prueba con ese
    diagnóstico: sin regla sale el código; con regla, no."""
    catalog = build_index((DiagnosticCode("R002", "PALPITACIONES", 3, "minsalud"),))
    stt = _SpyWhisperSTT(full_text="Drolar en el pecho, falta de aire, palpitaciones.\nNo, eso no.")
    # Sin regla sale el código
    c_sin = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        match_threshold=0.3,
        negation=load_negation(),
        retraction=None,
    )
    prep_sin = c_sin.prepare(owner=AgentId("p"), audio_path=Path("/tmp/a.wav"), history_text="")
    assert any(m.diagnostic.code == "R002" for m in prep_sin.matches)

    # Con regla no sale
    c_con = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        match_threshold=0.3,
        negation=load_negation(),
        retraction=load_retraction(),
    )
    prep_con = c_con.prepare(owner=AgentId("p"), audio_path=Path("/tmp/a.wav"), history_text="")
    assert not any(m.diagnostic.code == "R002" for m in prep_con.matches)


def test_consultation_retractacion_opera_a_traves_de_secciones_distintas(
    sample_wording_review: SummaryWording, sample_cache: ConsultationCache
) -> None:
    """Caso 18 (§B.4): La retractación se calcula sobre el orden del transcript,
    no sobre las secciones. Si caen en secciones distintas, la regla sigue funcionando."""
    catalog = build_index((DiagnosticCode("R002", "PALPITACIONES", 3, "minsalud"),))
    texto = "Drolar en el pecho, falta de aire, palpitaciones.\nNo, eso no."
    segs = (
        TranscriptSegment(0.0, 1.0, "Drolar en el pecho, falta de aire, palpitaciones."),
        TranscriptSegment(1.0, 2.0, "No, eso no."),
    )
    stt = _SpyWhisperSTT(segments=segs, full_text=texto)

    class DummySegmenter:
        def segment(self, transcript_text: str) -> tuple[tuple[SectionSpec, tuple[str, ...]], ...]:
            s1 = SectionSpec("motivo", "Motivo", (), codable=True, in_note=True)
            s2 = SectionSpec("examen", "Examen", (), codable=True, in_note=True)
            return (
                (s1, ("Drolar en el pecho, falta de aire, palpitaciones.",)),
                (s2, ("No, eso no.",)),
            )

    c = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        match_threshold=0.3,
        segmenter=DummySegmenter(),  # type: ignore[arg-type]
        negation=load_negation(),
        retraction=load_retraction(),
    )
    prep = c.prepare(owner=AgentId("p"), audio_path=Path("/tmp/a.wav"), history_text="")
    assert not any(m.diagnostic.code == "R002" for m in prep.matches)


def test_consultation_historia_no_se_retracta(
    sample_wording_review: SummaryWording, sample_cache: ConsultationCache
) -> None:
    """Caso 19 (§B.4): La historia clínica no se retracta; las líneas con la misma
    forma se cotejan enteras."""
    catalog = build_index((DiagnosticCode("R002", "PALPITACIONES", 3, "minsalud"),))
    stt = _SpyWhisperSTT(full_text="Drolar en el pecho, falta de aire, palpitaciones.\nNo, eso no.")
    historia = "Drolar en el pecho, falta de aire, palpitaciones."
    c = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        match_threshold=0.3,
        negation=load_negation(),
        retraction=load_retraction(),
    )
    prep = c.prepare(owner=AgentId("p"), audio_path=Path("/tmp/a.wav"), history_text=historia)
    assert any(m.diagnostic.code == "R002" for m in prep.matches)


def test_consultation_sin_retraction_inyectada_nada_cambia(
    sample_wording_review: SummaryWording, sample_cache: ConsultationCache
) -> None:
    """Caso 20 (§B.4): Sin retraction inyectada (None), nada cambia y salen los mismos códigos."""
    catalog = build_index((DiagnosticCode("R002", "PALPITACIONES", 3, "minsalud"),))
    stt = _SpyWhisperSTT(full_text="Drolar en el pecho, falta de aire, palpitaciones.\nNo, eso no.")
    c = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        match_threshold=0.3,
        negation=load_negation(),
        retraction=None,
    )
    prep = c.prepare(owner=AgentId("p"), audio_path=Path("/tmp/a.wav"), history_text="")
    assert any(m.diagnostic.code == "R002" for m in prep.matches)


def test_consultation_hallazgo_retractado_permanece_en_modelo_clinico(
    sample_wording_review: SummaryWording, sample_cache: ConsultationCache
) -> None:
    """Caso 21 (§B.4): El hallazgo retractado permanece en encounter.findings."""
    catalog = build_index((DiagnosticCode("R002", "PALPITACIONES", 3, "minsalud"),))
    texto = "Drolar en el pecho, falta de aire, palpitaciones.\nNo, eso no."
    segs = (
        TranscriptSegment(0.0, 1.0, "Drolar en el pecho, falta de aire, palpitaciones."),
        TranscriptSegment(1.0, 2.0, "No, eso no."),
    )
    stt = _SpyWhisperSTT(segments=segs, full_text=texto)

    class DummySegmenter:
        def segment(self, transcript_text: str) -> tuple[tuple[SectionSpec, tuple[str, ...]], ...]:
            s1 = SectionSpec("motivo", "Motivo", (), codable=True, in_note=True)
            return (
                (
                    s1,
                    (
                        "Drolar en el pecho, falta de aire, palpitaciones.",
                        "No, eso no.",
                    ),
                ),
            )

    c = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(),
        match_threshold=0.3,
        segmenter=DummySegmenter(),  # type: ignore[arg-type]
        negation=load_negation(),
        retraction=load_retraction(),
    )
    prep = c.prepare(owner=AgentId("p"), audio_path=Path("/tmp/a.wav"), history_text="")
    assert prep.encounter is not None
    frases_modelo = [f.text for items in prep.encounter.findings.values() for f in items]
    assert "Drolar en el pecho, falta de aire, palpitaciones." in frases_modelo


def test_consultation_review_elapsed_seconds_externo_registra_tiempo_pasado(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 21 (§B.3): review(..., elapsed_seconds=42.5) registra 42.5 en el store
    independientemente del reloj del bucle.
    """
    times = iter([10.0, 11.0, 12.0])
    metrics_path = tmp_path / "revisiones_elapsed.jsonl"
    store = ReviewMetricsStore(metrics_path)

    catalog = build_index((DiagnosticCode("I10", "Hipertensión esencial (primaria)", 3, "ambas"),))
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="consulta por dolor de cabeza",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log,
        segmenter=segmenter,
        extractor=extractor,
        review_metrics=store,
        clock=lambda: next(times),
    )
    owner = AgentId("doc_external_elapsed")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión esencial.",
    )

    def sample_reviewer(item: ReviewItem) -> Answer:
        return Answer(Verdict.CONFIRMED)

    consultation.review(prepared, reviewer=sample_reviewer, elapsed_seconds=42.5)

    records = store.read()
    assert len(records) == 1
    assert records[0].completed is True
    assert records[0].seconds == 42.5


def test_consultation_review_sin_elapsed_seconds_mide_con_reloj_del_bucle(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 22 (§B.3): Sin elapsed_seconds (o None), se mide con el reloj del bucle
    (comportamiento anterior intacto).
    """
    times = iter([100.0, 125.0])
    metrics_path = tmp_path / "revisiones_clock.jsonl"
    store = ReviewMetricsStore(metrics_path)

    catalog = build_index((DiagnosticCode("I10", "Hipertensión esencial (primaria)", 3, "ambas"),))
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="consulta por dolor de cabeza",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log,
        segmenter=segmenter,
        extractor=extractor,
        review_metrics=store,
        clock=lambda: next(times),
    )
    owner = AgentId("doc_clock_elapsed")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión esencial.",
    )

    def sample_reviewer(item: ReviewItem) -> Answer:
        return Answer(Verdict.CONFIRMED)

    consultation.review(prepared, reviewer=sample_reviewer, elapsed_seconds=None)

    records = store.read()
    assert len(records) == 1
    assert records[0].completed is True
    assert records[0].seconds == 25.0


def test_consultation_review_cortada_registra_elapsed_seconds(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 23 (§B.3): Si el reviewer levanta (revisión cortada), el registro
    incompleto lleva el elapsed_seconds pasado.
    """
    metrics_path = tmp_path / "revisiones_aborted_elapsed.jsonl"
    store = ReviewMetricsStore(metrics_path)

    catalog = build_index((DiagnosticCode("I10", "Hipertensión esencial (primaria)", 3, "ambas"),))
    segments = (
        TranscriptSegment(
            start=0.0,
            end=2.0,
            text="consulta por dolor de cabeza",
        ),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    segmenter = ClinicalSegmenter(load_sections())
    extractor = build_health_extractor()
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording_review,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=log,
        segmenter=segmenter,
        extractor=extractor,
        review_metrics=store,
        clock=lambda: 10.0,
    )
    owner = AgentId("doc_aborted_elapsed")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Paciente con hipertensión.",
    )

    def failing_reviewer(item: ReviewItem) -> Answer:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        consultation.review(prepared, reviewer=failing_reviewer, elapsed_seconds=77.5)

    records = store.read()
    assert len(records) == 1
    assert records[0].completed is False
    assert records[0].seconds == 77.5


def test_consultation_terceros_no_produce_codigo_ni_por_lexico(
    sample_wording_sections: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 6 (§B.2, M2): Una frase de tercero no produce código, ni por el léxico.
    «Mi padre murió de un infarto» con léxico que traduce «infarto» no produce I214."""
    code_infarto = DiagnosticCode("I214", "Infarto subendocárdico agudo del miocardio", 3, "ambas")
    catalog_index = build_index((code_infarto,))
    lexicon: Lexicon = (("infarto", "infarto subendocárdico agudo del miocardio"),)

    segments = (
        TranscriptSegment(start=0.0, end=3.0, text="Mi padre murió de un infarto a los 60."),
    )
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())

    # Con filtro de terceros: NO sale el código
    consultation_con_tp = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        segmenter=segmenter,
        lexicon=lexicon,
        third_party=load_third_party(),
    )
    res_con_tp = consultation_con_tp.run_with_history(
        owner=AgentId("doc_tp"),
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
    )
    assert "I214" not in res_con_tp.matched_codes

    # Sin filtro de terceros (third_party=None): SÍ sale el código
    consultation_sin_tp = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=catalog_index,
        provider=None,
        cache=sample_cache,
        log=log,
        segmenter=segmenter,
        lexicon=lexicon,
        third_party=None,
    )
    res_sin_tp = consultation_sin_tp.run_with_history(
        owner=AgentId("doc_no_tp"),
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
    )
    assert "I214" in res_sin_tp.matched_codes


def test_consultation_filtro_terceros_actua_antes_del_lexico(
    sample_wording_sections: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 7 (§B.2, M1): El filtro de terceros actúa ANTES del léxico:
    si actuara después, clinical_terms ya habría generado el término clínico
    y el código se colaría."""
    lexicon: Lexicon = (("infarto", "infarto agudo de miocardio"),)
    code_infarto = DiagnosticCode("I214", "Infarto agudo de miocardio", 3, "ambas")
    catalog_index = build_index((code_infarto,))

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=_SpyWhisperSTT(),
        ocr=_DummyOCR(),
        catalog_index=catalog_index,
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        lexicon=lexicon,
        third_party=load_third_party(),
    )
    # _affirmed debe descartar antes de traducir, devolviendo tupla vacía
    afirmados = consultation._affirmed(("Mi padre murió de un infarto a los 60.",))
    assert len(afirmados) == 0


def test_consultation_frase_de_tercero_sigue_en_la_nota(
    sample_wording_sections: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 8 (§B.2, M3): La frase de tercero sigue en la nota:
    grouped_sections y el modelo clínico la conservan íntegramente (§3.1)."""
    frase_familiar = "Mi padre murió de un infarto a los 60."
    segments = (TranscriptSegment(start=0.0, end=3.0, text=frase_familiar),)
    stt = _SpyWhisperSTT(segments=segments)
    ocr = _DummyOCR()
    log = EventLog(window=100)
    segmenter = ClinicalSegmenter(load_sections())

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=stt,
        ocr=ocr,
        catalog_index=CatalogIndex(entries=(), stopwords=frozenset()),
        provider=None,
        cache=sample_cache,
        log=log,
        segmenter=segmenter,
        third_party=load_third_party(),
    )
    res = consultation.run_with_history(
        owner=AgentId("doc_note_tp"),
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
    )
    # La frase debe estar en el transcript, en el modelo clínico y en la nota/resumen
    assert frase_familiar in res.transcript
    assert res.encounter is not None
    hallazgos_textos = [
        finding.text
        for findings in res.encounter.findings.values()
        for finding in findings
    ]
    assert frase_familiar in hallazgos_textos
    assert frase_familiar in res.summary_markdown


def test_consultation_sin_terceros_inyectado_conserva_comportamiento(
    sample_wording_sections: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 9 (§B.2): Sin catálogo inyectado (third_party=None), nada cambia:
    el comportamiento previo se conserva."""
    consultation = Consultation(
        wording=sample_wording_sections,
        stt=_SpyWhisperSTT(),
        ocr=_DummyOCR(),
        catalog_index=CatalogIndex(entries=(), stopwords=frozenset()),
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        third_party=None,
    )
    afirmados = consultation._affirmed(("Mi padre murió de un infarto a los 60.",))
    assert len(afirmados) > 0
    assert any("Mi padre murió de un infarto" in c.query for c in afirmados)


# ===========================================================================
# Caso del Plan 80 (§B.3): Cableado de la conversación a la nota
# ===========================================================================


def test_render_note_pasa_transcripcion_incluyendo_otros_comentarios(
    sample_wording_sections: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 11 (§B.3): render_note pasa la transcripción a build_note; el briefing
    contiene una línea de prepared.transcript que la segmentación mandó a otros_comentarios."""
    class _SpyNoteProvider:
        def __init__(self) -> None:
            self.captured_briefing: str | None = None

        def think(self, briefing: str, schema: type) -> object | None:
            self.captured_briefing = briefing
            return ClinicalNote(fields={"motivo_consulta": "Motivo redactado."})

    spy_provider = _SpyNoteProvider()
    note_settings = NoteSettings(
        system_prompt="Redacta la nota de una consulta.",
        provider="test",
        provider_model="test-model",
        field_heading="### {}",
        provenance_template="_{}_",
        fallback_notice="",
        edited_template="_{}_",
    )

    consultation = Consultation(
        wording=sample_wording_sections,
        stt=_SpyWhisperSTT(),
        ocr=_DummyOCR(),
        catalog_index=CatalogIndex(entries=(), stopwords=frozenset()),
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        note_settings=note_settings,
        note_providers=(("test_provider", spy_provider),),
    )

    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    spec_otros = SectionSpec(
        "otros_comentarios", "Otros comentarios", (), codable=False, in_note=False
    )
    linea_otros = "Y el omeprazol para el estómago, porque tengo reflujo."

    prepared = PreparedConsultation(
        owner=AgentId("doc_render_note"),
        transcript=f"Tengo dolor de cabeza.\n{linea_otros}",
        history="",
        matches=(),
        grouped_sections=(
            (spec_motivo, ("Tengo dolor de cabeza.",)),
            (spec_otros, (linea_otros,)),
        ),
        encounter=None,
    )

    rendered = consultation.render_note(prepared)
    assert rendered.redacted is True
    assert spy_provider.captured_briefing is not None

    briefing = spy_provider.captured_briefing
    hechos_parte = briefing.split("Conversación completa, en orden:")[0]
    assert linea_otros not in hechos_parte
    assert linea_otros in briefing


# ---------------------------------------------------------------------------
# §B.6 — Cableado (Casos 25 y 26)
# ---------------------------------------------------------------------------


def test_consultation_review_note_y_edits_van_juntos(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
) -> None:
    """Caso 25 (§B.6): Consultation.review con note y sin note_edits (o al revés) -> ValueError."""
    consultation = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(),
        ocr=_DummyOCR(),
        catalog_index=CatalogIndex(entries=(), stopwords=frozenset()),
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        segmenter=ClinicalSegmenter(load_sections()),
        extractor=build_health_extractor(),
    )
    prepared = consultation.prepare(
        owner=AgentId("doc_note_edits"),
        audio_path=Path("/tmp/audio.wav"),
        history_text="",
    )
    spec_a = SectionSpec("a", "A", (), True, True)
    note = RenderedNote(fields=((spec_a, "uno"),), redacted=True)

    def dummy_reviewer(item: ReviewItem) -> Answer:
        return Answer(Verdict.CONFIRMED)

    # Con note pero sin note_edits
    with pytest.raises(
        ValueError, match="la nota y sus ediciones van juntas, o ninguna de las dos"
    ):
        consultation.review(prepared, reviewer=dummy_reviewer, note=note, note_edits=None)

    # Con note_edits pero sin note
    with pytest.raises(
        ValueError, match="la nota y sus ediciones van juntas, o ninguna de las dos"
    ):
        consultation.review(
            prepared, reviewer=dummy_reviewer, note=None, note_edits={"a": "uno"}
        )


def test_consultation_review_eventos_de_campos_caen_en_review_metrics_store(
    sample_wording_review: SummaryWording,
    sample_cache: ConsultationCache,
    tmp_path: Path,
) -> None:
    """Caso 26 (§B.6, M11): Los eventos de los campos caen en el registro de la revisión:
    tras Consultation.review(..., note=, note_edits=) con un campo editado, el ReviewRecord
    guardado en ReviewMetricsStore trae ese campo en fields."""
    metrics_path = tmp_path / "revisiones_pantalla.jsonl"
    store = ReviewMetricsStore(metrics_path)
    spec_motivo = SectionSpec("motivo", "Motivo de consulta", (), True, True)
    spec_plan = SectionSpec("plan", "Plan de manejo", (), True, True)
    note = RenderedNote(
        fields=((spec_motivo, "Cefalea"), (spec_plan, "Ibuprofeno")),
        redacted=True,
    )

    catalog = build_index(
        (DiagnosticCode("I10", "Hipertensión esencial", 3, "ambas"),)
    )
    consultation = Consultation(
        wording=sample_wording_review,
        stt=_SpyWhisperSTT(),
        ocr=_DummyOCR(),
        catalog_index=catalog,
        provider=None,
        cache=sample_cache,
        log=EventLog(window=100),
        segmenter=ClinicalSegmenter(load_sections()),
        extractor=build_health_extractor(),
        review_metrics=store,
    )
    prepared = consultation.prepare(
        owner=AgentId("doc_screen_metrics"),
        audio_path=Path("/tmp/audio.wav"),
        history_text="",
    )

    def dummy_reviewer(item: ReviewItem) -> Answer:
        return Answer(Verdict.CONFIRMED)

    result = consultation.review(
        prepared,
        reviewer=dummy_reviewer,
        review_findings=False,
        note=note,
        note_edits={"motivo": "Cefalea", "plan": "Ibuprofeno 400mg"},
    )

    assert result.note is not None
    assert result.note.edited == ("plan",)

    records = store.read()
    assert len(records) == 1
    r = records[0]
    assert r.completed is True
    # Verificamos que fields contiene el campo editado y el confirmado
    assert "plan" in r.fields
    assert r.fields["plan"] == {"edited": 1}
    assert "motivo" in r.fields
    assert r.fields["motivo"] == {"confirmed": 1}



