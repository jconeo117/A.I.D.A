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
from aida.arms.health.extract import build_health_extractor
from aida.arms.health.match import CatalogIndex, build_index
from aida.arms.health.sections import load_sections
from aida.arms.health.segment import ClinicalSegmenter
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
            start=0.0, end=3.0, text="Buenos días, ¿cuál es el motivo de consulta hoy?"
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

    assert "MÉDICO: Buenos días, ¿cuál es el motivo de consulta hoy?" in result.transcript
    assert "PACIENTE: Tengo fiebre alta desde hace dos semanas." in result.transcript
    assert "MÉDICO: De acuerdo, se receta reposo y antipiréticos." in result.transcript

    # Clasificación por contenido a pesar del prefijo MÉDICO / PACIENTE
    assert (
        "### Motivo de consulta\nMÉDICO: Buenos días, ¿cuál es el motivo de consulta hoy?"
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
    segments = (
        TranscriptSegment(start=0.0, end=2.0, text="consulta hoy por cefalea"),
    )
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
    with pytest.raises(
        ValueError, match=r"extracción activa pero no hay segmentador configurado"
    ):
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





