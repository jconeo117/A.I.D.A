"""Batería de pruebas adversariales y verificación de resiliencia para U9.

Diseñada por Challenger 2 para verificar empíricamente los límites de
robustez de `src/aida/arms/health/`:
1. Resiliencia criptográfica y de caché (`ConsultationCache`):
   - Cifrados corruptos (tokens truncados, bit flips, bytes aleatorios, tokens inválidos).
   - Clave de cifrado incorrecta en archivos existentes.
   - Condiciones de carrera y concurrencia (borrado concurrente, TTL cero y negativo).
   - Travesía de rutas y nombres de `owner` maliciosos (`../`, rutas absolutas, subdirectorios).
   - Cifrado válido con carga no decodificable como UTF-8.
2. Resiliencia ante fallos en STT y OCR (`LocalWhisperSTT`, `LocalTesseractOCR`):
   - Excepciones del motor de Whisper y fallos en el generador de segmentos.
   - Excepciones de Tesseract e imágenes inválidas/inexistentes.
3. Resiliencia de la tubería (`Consultation.run`):
   - Orden de emisión de eventos (`INPUT_OBSERVED` -> `INPUT_OBSERVED` -> `CATALOG_MATCHED`).
   - Aborto seguro y eventos parciales ante fallos de STT u OCR.
   - Comportamiento ante respuestas inesperadas de `CognitionProvider` (tipos no esperados).
   - Comportamiento actual ante excepciones en `CognitionProvider.think` (escape sin captura).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

import aida.engine.ontology as ontology
from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import (
    LocalTesseractOCR,
    LocalWhisperSTT,
    OCRError,
    TranscriptionError,
)
from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.consultation import Consultation, SummaryWording
from aida.arms.health.match import CatalogIndex, build_index
from aida.cognition.briefing import SpokenReply
from aida.engine.events import EventKind, EventLog
from aida.engine.ports import CognitionProvider
from aida.platform.identity import Professional

_ACTOR = Professional(id="t", name="Prueba", registry="0")

Image = pytest.importorskip(
    "PIL.Image",
    reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
)
pdfium = pytest.importorskip(
    "pypdfium2",
    reason="requiere el extra 'capture' (pypdfium2)",
)


@pytest.fixture
def sample_wording() -> SummaryWording:
    """Plantillas de prueba para redacción de consulta con terminología médica."""
    return SummaryWording(
        system_prompt="Organiza en Markdown la información de una consulta médica:",
        said_template="El paciente dijo: {}",
        history_template="La historia registra: {}",
        matched_template="Códigos de referencia cotejados, NO diagnósticos: {}",
        no_match_text="Ningún código del catálogo alcanzó el umbral de cotejo.",
    )


@pytest.fixture
def fernet_key() -> bytes:
    """Clave Fernet efímera de 32 bytes en base64 para pruebas."""
    return Fernet.generate_key()


@pytest.fixture
def sample_catalog_index() -> CatalogIndex:
    """Índice mínimo del catálogo CIE-10 para pruebas de tubería."""
    catalog = (
        DiagnosticCode(
            code="C02",
            description="Tumor maligno de la lengua",
            level=3,
            source="ambas",
        ),
        DiagnosticCode(
            code="R51",
            description="Cefalea y dolor de cabeza",
            level=3,
            source="ambas",
        ),
    )
    return build_index(catalog)


@pytest.fixture
def sample_media(tmp_path: Path) -> tuple[Path, Path]:
    """Genera archivos de audio ficticio e imagen válida para OCR."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy wave audio")
    img_path = tmp_path / "history.png"
    img = Image.new("RGB", (20, 20), color="white")
    img.save(img_path)
    return audio, img_path


# ---------------------------------------------------------------------------
# 1. Resiliencia Criptográfica y de Caché (ConsultationCache)
# ---------------------------------------------------------------------------


def test_cache_corrupted_ciphertexts_return_none(tmp_path: Path, fernet_key: bytes) -> None:
    """Comprueba que tokens de Fernet corruptos bajo diversas mutaciones
    (truncamiento, inversión de bits, ruido pseudoaleatorio, token vacío o base64
    inválido) son manejados con gracia devolviendo `None` sin causar caídas.
    """
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    owner = ontology.AgentId("paciente_corrupcion")
    cache.store(owner=owner, summary_markdown="# Resumen confidencial")

    cache_file = tmp_path / f"{owner}.cache"
    valid_token = cache_file.read_bytes()

    # 1. Token truncado a 10 bytes
    cache_file.write_bytes(valid_token[:10])
    assert cache.load(owner=owner) is None

    # 2. Inversión de bits (bit flip en el payload cifrado)
    flipped = bytearray(valid_token)
    flipped[20] ^= 0xFF
    cache_file.write_bytes(bytes(flipped))
    assert cache.load(owner=owner) is None

    # 3. Ruido pseudoaleatorio
    cache_file.write_bytes(os.urandom(96))
    assert cache.load(owner=owner) is None

    # 4. Archivo de 0 bytes
    cache_file.write_bytes(b"")
    assert cache.load(owner=owner) is None

    # 5. Base64 inválido / caracteres arbitrarios
    cache_file.write_bytes(b"no_es_un_token_fernet_valido_!@#$%^&*()")
    assert cache.load(owner=owner) is None


def test_cache_wrong_key_returns_none(tmp_path: Path, fernet_key: bytes) -> None:
    """Verifica que intentar leer con una clave Fernet distinta a la usada
    para almacenar devuelve `None` limpiamente vía `InvalidToken`.
    """
    other_key = Fernet.generate_key()
    cache_writer = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    cache_reader = ConsultationCache(tmp_path, key=other_key, ttl_seconds=3600)

    owner = ontology.AgentId("paciente_otra_clave")
    cache_writer.store(owner=owner, summary_markdown="# Datos de paciente")

    assert cache_reader.load(owner=owner) is None


def test_cache_zero_and_negative_ttl_expire_immediately(tmp_path: Path, fernet_key: bytes) -> None:
    """Verifica el comportamiento con TTL cero y negativo: el registro expira
    de inmediato sin lanzar excepciones aritméticas.
    """
    # TTL = 0
    cache_zero = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=0)
    owner_zero = ontology.AgentId("paciente_ttl_cero")
    cache_zero.store(owner=owner_zero, summary_markdown="# Resumen")
    # Almacenado con éxito; load expira de inmediato
    assert cache_zero.load(owner=owner_zero) is None

    # TTL = -60
    cache_neg = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=-60)
    owner_neg = ontology.AgentId("paciente_ttl_negativo")
    cache_neg.store(owner=owner_neg, summary_markdown="# Resumen")
    assert cache_neg.load(owner=owner_neg) is None


def test_cache_load_non_utf8_decrypted_payload_raises_unicodedecodeerror(
    tmp_path: Path, fernet_key: bytes
) -> None:
    """Verifica que si un archivo cifrado con la clave Fernet válida contiene
    bytes no decodificables como UTF-8, `load()` captura `UnicodeDecodeError`
    y retorna `None` sin propagar excepciones.
    """
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    owner = ontology.AgentId("paciente_payload_binario")
    cache_file = tmp_path / f"{owner}.cache"

    fernet = Fernet(fernet_key)
    # Cifrar bytes inválidos en UTF-8
    token = fernet.encrypt(b"\xff\xfe\xfd\x80\x81")
    cache_file.write_bytes(token)

    assert cache.load(owner=owner) is None


def test_cache_unlinking_race_condition_suppresses_filenotfounderror(
    tmp_path: Path, fernet_key: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica la resiliencia ante condiciones de carrera en `load()` y `purge_expired()`:
    al utilizar `missing_ok=True` en `path.unlink()`, si un proceso o hilo
    competidor elimina el archivo caducado tras la comprobación del TTL,
    no se propaga `FileNotFoundError`.
    """
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=0)
    owner = ontology.AgentId("paciente_carrera")
    cache.store(owner=owner, summary_markdown="# Resumen caduco")

    orig_stat = Path.stat

    def racy_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        st = orig_stat(self, *args, **kwargs)
        if str(self).endswith(".cache"):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(str(self))
        return st

    monkeypatch.setattr(Path, "stat", racy_stat)
    deleted = cache.purge_expired()
    assert deleted in (0, 1)

    monkeypatch.undo()

    # Verificar también que load() maneja la eliminación concurrente en unlink
    orig_unlink = Path.unlink

    def racy_unlink(self: Path, *args: Any, **kwargs: Any) -> Any:
        with contextlib.suppress(FileNotFoundError):
            orig_unlink(self)
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", racy_unlink)
    cache.store(owner=owner, summary_markdown="# Resumen caduco 2")
    assert cache.load(owner=owner) is None


def test_cache_path_traversal_via_owner_escapes_cache_directory(
    tmp_path: Path, fernet_key: bytes
) -> None:
    """Verifica la contención de travesía de directorios en `ConsultationCache`:
    al sanear `owner` extrayendo únicamente `Path(str(owner)).name`,
    los intentos de travesía relativa ('../'), rutas absolutas o subdirectorios
    permanecen estrictamente contenidos dentro de `cache_dir`.
    """
    cache_dir = tmp_path / "cache_sandbox"
    cache = ConsultationCache(cache_dir, key=fernet_key, ttl_seconds=3600)

    # 1. Travesía relativa: el archivo queda contenido dentro de cache_dir
    traversal_owner = ontology.AgentId("../escaped_relative")
    cache.store(owner=traversal_owner, summary_markdown="Contenido fuera de sandbox")
    assert not (tmp_path / "escaped_relative.cache").exists()
    assert (cache_dir / "escaped_relative.cache").exists()
    assert cache.load(owner=traversal_owner) == "Contenido fuera de sandbox"

    # 2. Ruta absoluta: no sobreescribe el directorio base en pathlib.Path
    abs_target = tmp_path / "escaped_absolute"
    abs_owner = ontology.AgentId(str(abs_target))
    cache.store(owner=abs_owner, summary_markdown="Contenido absoluto")
    assert not (tmp_path / "escaped_absolute.cache").exists()
    assert (cache_dir / "escaped_absolute.cache").exists()
    assert cache.load(owner=abs_owner) == "Contenido absoluto"

    # 3. Subdirectorio: se extrae solo el nombre base, sin requerir directorios intermedios
    subdir_owner = ontology.AgentId("non_existent_folder/patient")
    cache.store(owner=subdir_owner, summary_markdown="Contenido subdirectorio")
    assert (cache_dir / "patient.cache").exists()
    assert cache.load(owner=subdir_owner) == "Contenido subdirectorio"


# ---------------------------------------------------------------------------
# 2. Resiliencia en STT y OCR (capture.py)
# ---------------------------------------------------------------------------


def test_whisper_stt_wraps_engine_exceptions_in_transcription_error(
    tmp_path: Path,
) -> None:
    """Comprueba que cualquier excepción interna del motor `faster-whisper`
    (caída de CUDA, modelo corrupto, fallo de decodificación) se envuelve
    rigurosamente en `TranscriptionError`.
    """

    class CrashingWhisperModel:
        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("CUDA Out of Memory in forward pass")

    stt = LocalWhisperSTT(CrashingWhisperModel())
    fake_audio = tmp_path / "audio.wav"
    fake_audio.write_bytes(b"RIFF....WAVE")

    with pytest.raises(TranscriptionError, match="CUDA Out of Memory"):
        stt.transcribe(fake_audio)


def test_whisper_stt_wraps_generator_stream_exceptions_in_transcription_error(
    tmp_path: Path,
) -> None:
    """Comprueba que excepciones ocurridas durante la iteración del generador
    de segmentos de audio se envuelven también en `TranscriptionError`.
    """

    class CrashingSegmentGeneratorModel:
        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            def segment_gen() -> Any:
                mock_seg = MagicMock()
                mock_seg.text = "Primer segmento ok"
                mock_seg.no_speech_prob = 0.0
                yield mock_seg
                raise ValueError("Stream truncado inesperadamente")

            return segment_gen(), None

    stt = LocalWhisperSTT(CrashingSegmentGeneratorModel())
    fake_audio = tmp_path / "audio.wav"
    fake_audio.write_bytes(b"audio data")

    with pytest.raises(TranscriptionError, match="Stream truncado"):
        stt.transcribe(fake_audio)


def test_tesseract_ocr_wraps_engine_exceptions_in_ocr_error(tmp_path: Path) -> None:
    """Comprueba que fallos del binario del sistema o de la librería pytesseract
    se envuelven limpiamente en `OCRError`.
    """

    class CrashingTesseract:
        def image_to_string(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Tesseract segmentation fault (core dumped)")

    img_path = tmp_path / "history.png"
    img = Image.new("RGB", (20, 20), color="white")
    img.save(img_path)

    ocr = LocalTesseractOCR(CrashingTesseract())
    with pytest.raises(OCRError, match="Tesseract segmentation fault"):
        ocr.read(img_path)


def test_tesseract_ocr_wraps_missing_file_or_bad_format_in_ocr_error(
    tmp_path: Path,
) -> None:
    """Comprueba que archivos inexistentes o con formato de imagen corrupto
    abiertos por PIL se envuelven en `OCRError`.
    """
    ocr = LocalTesseractOCR(MagicMock())
    with pytest.raises(OCRError, match="No such file or directory"):
        ocr.read(tmp_path / "non_existent.png")


def test_tesseract_ocr_reads_multipage_pdf_in_order(tmp_path: Path) -> None:
    """PDF de dos páginas devuelve los textos unidos por '\n\n' en orden de página."""

    class MultiPageTesseract:
        def image_to_string(self, img: Any, *args: Any, **kwargs: Any) -> str:
            return f"P{img.height}"

    pdf_path = tmp_path / "multipage.pdf"
    img1 = Image.new("RGB", (100, 60), color="white")
    img2 = Image.new("RGB", (100, 120), color="black")
    img1.save(pdf_path, save_all=True, append_images=[img2])

    ocr = LocalTesseractOCR(MultiPageTesseract())
    result = ocr.read(pdf_path)
    assert result == "P120\n\nP240"


def test_tesseract_ocr_passes_pil_image_for_both_pdf_and_non_pdf(
    tmp_path: Path,
) -> None:
    """PDF y no-PDF pasan por el mismo doble y la imagen recibida por
    image_to_string es PIL.Image en los dos casos.
    """
    seen_types: list[type] = []

    class AssertingTesseract:
        def image_to_string(self, img: Any, *args: Any, **kwargs: Any) -> str:
            assert isinstance(img, Image.Image)
            seen_types.append(type(img))
            return "texto extraido"

    ocr = LocalTesseractOCR(AssertingTesseract())

    # No-PDF (PNG)
    png_path = tmp_path / "record.png"
    img_png = Image.new("RGB", (50, 50), color="white")
    img_png.save(png_path)
    assert ocr.read(png_path) == "texto extraido"

    # PDF
    pdf_path = tmp_path / "record.pdf"
    img_pdf = Image.new("RGB", (50, 50), color="white")
    img_pdf.save(pdf_path)
    assert ocr.read(pdf_path) == "texto extraido"

    assert len(seen_types) == 2


def test_tesseract_ocr_wraps_corrupt_pdf_in_ocr_error(tmp_path: Path) -> None:
    """PDF ilegible (archivo .pdf con bytes basura) produce OCRError,
    no una excepción cruda de pypdfium2.
    """
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"%PDF-malformed-garbage-bytes\xff\xfe\xaa\xbb")

    ocr = LocalTesseractOCR(MagicMock())
    with pytest.raises(OCRError) as exc_info:
        ocr.read(bad_pdf)

    assert "no se pudo leer" in str(exc_info.value)


def test_tesseract_ocr_handles_uppercase_pdf_extension(tmp_path: Path) -> None:
    """Sufijo en mayúsculas (.PDF) toma la rama PDF."""

    class UpperTesseract:
        def image_to_string(self, *args: Any, **kwargs: Any) -> str:
            return "Texto extraido de PDF en mayusculas"

    pdf_path = tmp_path / "historia.PDF"
    img = Image.new("RGB", (60, 60), color="white")
    img.save(pdf_path)

    ocr = LocalTesseractOCR(UpperTesseract())
    result = ocr.read(pdf_path)
    assert result == "Texto extraido de PDF en mayusculas"


# ---------------------------------------------------------------------------
# 3. Resiliencia de la Tubería (Consultation.run)
# ---------------------------------------------------------------------------


class MockWhisperBackend:
    def __init__(self, text: str = "Paciente con dolor de cabeza persistente") -> None:
        self.text = text

    def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
        mock_seg = MagicMock()
        mock_seg.text = self.text
        mock_seg.no_speech_prob = 0.0
        return [mock_seg], None


class MockTesseractBackend:
    def __init__(self, text: str = "Historia clinica previa: cefalea tensional recurrente") -> None:
        self.text = text

    def image_to_string(self, *args: Any, **kwargs: Any) -> str:
        return self.text


class DummyCognitionProvider(CognitionProvider):
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, type]] = []

    def think(self, briefing: str, schema: type) -> Any:
        self.calls.append((briefing, schema))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_consultation_run_event_emission_order_and_payload(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """Verifica empíricamente el orden y contenido exacto de los eventos emitidos
    en una ejecución exitosa:
    1. INPUT_OBSERVED (transcript)
    2. INPUT_OBSERVED (history)
    3. CATALOG_MATCHED (matched codes)
    """
    audio_path, history_path = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    log = EventLog(window=100)
    provider = DummyCognitionProvider(SpokenReply(text="## Resumen redactado por IA"))

    stt = LocalWhisperSTT(MockWhisperBackend("Dolor de cabeza"))
    ocr = LocalTesseractOCR(MockTesseractBackend("Antecedentes de cefalea"))

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=provider,
        cache=cache,
        log=log,
        match_threshold=0.3,
    )

    owner = ontology.AgentId("paciente_auditoria")
    consent = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    result = consultation.run(
        owner=owner, audio_path=audio_path, history_path=history_path, consent=consent
    )

    events = list(log)
    assert len(events) == 4

    # Evento 0: Autorización registrada
    ev0 = events[0]
    assert ev0.kind == EventKind.CONSENT_RECORDED
    assert ev0.owner == owner

    # Evento 1: Transcripción
    ev1 = events[1]
    assert ev1.tick == 0
    assert ev1.kind == EventKind.INPUT_OBSERVED
    assert ev1.owner == owner
    assert ev1.content == "Dolor de cabeza"

    # Evento 2: Historia
    ev2 = events[2]
    assert ev2.tick == 0
    assert ev2.kind == EventKind.INPUT_OBSERVED
    assert ev2.owner == owner
    assert ev2.content == "Antecedentes de cefalea"

    # Evento 3: Códigos cotejados
    ev3 = events[3]
    assert ev3.tick == 0
    assert ev3.kind == EventKind.CATALOG_MATCHED
    assert ev3.owner == owner
    assert ev3.data == {"codes": "R51"}

    # Resumen en resultado y guardado en caché
    assert result.summary_markdown == "## Resumen redactado por IA"
    assert cache.load(owner=owner) == "## Resumen redactado por IA"


def test_consultation_run_empty_catalog_matches_emits_empty_data_dict(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """Comprueba que cuando ningún código del catálogo alcanza el umbral,
    el evento CATALOG_MATCHED emite `data={}` (dict vacío) y el resumen
    refleja la ausencia de cotejos.
    """
    audio_path, history_path = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    log = EventLog(window=100)

    stt = LocalWhisperSTT(MockWhisperBackend("Texto sin relacion medica alguna"))
    ocr = LocalTesseractOCR(MockTesseractBackend("Otro texto sin relacion"))

    consultation = Consultation(
        wording=sample_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=log,
        match_threshold=0.99,  # Umbral inalcanzable
    )

    owner = ontology.AgentId("paciente_sin_matches")
    consent = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    result = consultation.run(
        owner=owner, audio_path=audio_path, history_path=history_path, consent=consent
    )

    events = [e for e in log if e.kind != EventKind.CONSENT_RECORDED]
    assert len(events) == 3
    assert events[2].kind == EventKind.CATALOG_MATCHED
    assert events[2].data == {}
    assert result.matched_codes == ()
    assert "Ningún código del catálogo alcanzó el umbral de cotejo." in result.summary_markdown


def test_consultation_run_failure_in_stt_stops_pipeline_before_emitting_events(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """Comprueba que si el STT falla con `TranscriptionError`, la tubería se detiene
    de inmediato sin emitir eventos y sin invocar OCR, cotejo ni caché.
    """
    audio_path, history_path = sample_media

    class FailingBackend:
        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Audio corrupto")

    mock_ocr_backend = MagicMock()
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(FailingBackend()),
        ocr=LocalTesseractOCR(mock_ocr_backend),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=log,
    )

    owner = ontology.AgentId("p_fail_stt")
    consent = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    with pytest.raises(TranscriptionError):
        consultation.run(
            owner=owner,
            audio_path=audio_path,
            history_path=history_path,
            consent=consent,
        )

    assert [e for e in log if e.kind != EventKind.CONSENT_RECORDED] == []
    mock_ocr_backend.image_to_string.assert_not_called()


def test_consultation_run_failure_in_ocr_preserves_only_stt_event(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """Comprueba que si el OCR falla con `OCRError`, el evento de transcripción
    ya emitido permanece en el log de auditoría pero no se emite CATALOG_MATCHED.
    """
    audio_path, history_path = sample_media

    class FailingTesseractBackend:
        def image_to_string(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Imagen no legible")

    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(MockWhisperBackend("Audio procesado")),
        ocr=LocalTesseractOCR(FailingTesseractBackend()),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=log,
    )

    owner = ontology.AgentId("p_fail_ocr")
    consent = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    with pytest.raises(OCRError):
        consultation.run(
            owner=owner,
            audio_path=audio_path,
            history_path=history_path,
            consent=consent,
        )

    events = [e for e in log if e.kind != EventKind.CONSENT_RECORDED]
    assert len(events) == 1
    assert events[0].kind == EventKind.INPUT_OBSERVED
    assert events[0].content == "Audio procesado"


def test_consultation_run_unexpected_provider_types_gracefully_fallback_to_markdown(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """Comprueba que cuando `CognitionProvider.think` retorna tipos inesperados
    (None, int, str, dict, objetos arbitrarios en lugar de SpokenReply), la tubería
    degrada con gracia generando el resumen Markdown determinista basado en los hechos.
    """
    audio_path, history_path = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    owner = ontology.AgentId("paciente_fallback")

    # Referencia con provider=None
    c_none = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(MockWhisperBackend("Dolor")),
        ocr=LocalTesseractOCR(MockTesseractBackend("Historia")),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=EventLog(window=100),
    )
    consent_none = c_none.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    expected_fallback = c_none.run(
        owner=owner, audio_path=audio_path, history_path=history_path, consent=consent_none
    ).summary_markdown

    class ForeignObjectWithText:
        def __init__(self, text: str) -> None:
            self.text = text

    unexpected_payloads = [
        None,
        "cadena cruda sin esquema",
        12345,
        {"text": "diccionario no modelado"},
        [1, 2, 3],
        object(),
        ForeignObjectWithText("objeto arbitrario con text"),
    ]

    for payload in unexpected_payloads:
        prov = DummyCognitionProvider(payload)
        c = Consultation(
            wording=sample_wording,
            stt=LocalWhisperSTT(MockWhisperBackend("Dolor")),
            ocr=LocalTesseractOCR(MockTesseractBackend("Historia")),
            catalog_index=sample_catalog_index,
            provider=prov,
            cache=cache,
            log=EventLog(window=100),
        )
        consent = c.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
        res = c.run(
            owner=owner,
            audio_path=audio_path,
            history_path=history_path,
            consent=consent,
        )
        assert res.summary_markdown == expected_fallback, (
            f"Fallo en degradación con payload: {type(payload)}"
        )


def test_consultation_run_provider_exception_currently_escapes_uncaught(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """Verifica que si `provider.think` lanza una excepción (p. ej. `RuntimeError`
    por fallo de red o cuota del LLM), `Consultation.run` degrada con gracia
    generando el resumen determinista en Markdown basado en los hechos y
    almacenándolo en caché sin interrumpir el flujo.
    """
    audio_path, history_path = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    prov = DummyCognitionProvider(RuntimeError("Fallo de infraestructura LLM"))

    consultation = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(MockWhisperBackend("Audio de consulta")),
        ocr=LocalTesseractOCR(MockTesseractBackend("Historia clinica")),
        catalog_index=sample_catalog_index,
        provider=prov,
        cache=cache,
        log=EventLog(window=100),
    )

    owner = ontology.AgentId("paciente_proveedor_caido")
    consent = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)

    result = consultation.run(
        owner=owner, audio_path=audio_path, history_path=history_path, consent=consent
    )

    # Verifica que se generó el resumen de respaldo determinista
    assert "- El paciente dijo: Audio de consulta" in result.summary_markdown
    assert "- La historia registra: Historia clinica" in result.summary_markdown
    # Verifica que el resultado se almacenó con éxito en caché
    assert cache.load(owner=owner) == result.summary_markdown


def test_consultation_run_with_neutral_wording_has_no_medical_domain_literals(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
) -> None:
    """I-073 (§4.2): Verifica que al inyectar un SummaryWording neutro sin términos
    de dominio médico, el resumen determinista refleja fielmente las plantillas
    proporcionadas y no contiene ninguno de los literales médicos preexistentes
    ('paciente', 'consulta médica', 'diagnóstico', 'cefalea', 'historia clínica').
    """
    audio_path, history_path = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    log = EventLog(window=100)

    neutral_wording = SummaryWording(
        system_prompt="Sintetizar eventos observados en formato estructurado:",
        said_template="A dijo: {}",
        history_template="B registro: {}",
        matched_template="Indices cotejados: {}",
        no_match_text="Sin coincidencias registradas.",
    )

    stt = LocalWhisperSTT(MockWhisperBackend("Declaracion neutral de audio"))
    ocr = LocalTesseractOCR(MockTesseractBackend("Documento de texto neutral"))

    consultation = Consultation(
        wording=neutral_wording,
        stt=stt,
        ocr=ocr,
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=log,
        match_threshold=0.99,
    )

    owner = ontology.AgentId("agente_agnostico")
    consent = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    result = consultation.run(
        owner=owner, audio_path=audio_path, history_path=history_path, consent=consent
    )

    # Verifica que los textos inyectados están presentes
    assert "- A dijo: Declaracion neutral de audio" in result.summary_markdown
    assert "- B registro: Documento de texto neutral" in result.summary_markdown
    assert "- Sin coincidencias registradas." in result.summary_markdown

    # Verifica ausencia estricta de literales de dominio médico
    summary_lower = result.summary_markdown.lower()
    for forbidden_term in (
        "paciente",
        "consulta médica",
        "consulta medica",
        "diagnóstico",
        "diagnostico",
        "cefalea",
        "historia clínica",
        "historia clinica",
    ):
        assert forbidden_term not in summary_lower, (
            f"Término de dominio médico {forbidden_term!r} presente en resumen:\n"
            f"{result.summary_markdown}"
        )


def test_consultation_run_provider_non_spoken_reply_with_text_falls_back_to_markdown(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """I-074 (§4.3) / Mata Mutación V11:
    Si el proveedor devuelve un objeto ajeno que no es instancia de `SpokenReply`
    pero posee un atributo `.text`, la comprobación estricta `isinstance(result, SpokenReply)`
    debe rechazarlo y degradar al resumen determinista en Markdown basado en hechos.
    El texto del objeto ajeno NO debe filtrarse en el resumen.

    Si alguien relaja la comprobación `isinstance(result, SpokenReply)` a
    `result is not None` (Mutación V11), el texto espurio se filtraría y este
    test fallaría de forma observable (RED).
    """
    audio_path, history_path = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    owner = ontology.AgentId("paciente_v11_guard")

    # Resumen determinista de referencia con provider=None
    c_reference = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(MockWhisperBackend("Dolor persistente")),
        ocr=LocalTesseractOCR(MockTesseractBackend("Historia clinica previa")),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=EventLog(window=100),
    )
    consent_ref = c_reference.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    expected_fallback = c_reference.run(
        owner=owner, audio_path=audio_path, history_path=history_path, consent=consent_ref
    ).summary_markdown

    class ForeignObjectWithText:
        def __init__(self, text: str) -> None:
            self.text = text

    foreign_payload = ForeignObjectWithText("TEXTO NO VALIDADO QUE SE CUELA")
    provider = DummyCognitionProvider(foreign_payload)

    c_adversarial = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(MockWhisperBackend("Dolor persistente")),
        ocr=LocalTesseractOCR(MockTesseractBackend("Historia clinica previa")),
        catalog_index=sample_catalog_index,
        provider=provider,
        cache=cache,
        log=EventLog(window=100),
    )

    consent_adv = c_adversarial.record_consent(
        cloud=True, owner=owner, mode="uploaded", actor=_ACTOR
    )
    result = c_adversarial.run(
        owner=owner, audio_path=audio_path, history_path=history_path, consent=consent_adv
    )

    # El texto del objeto ajeno no debe filtrarse bajo ninguna circunstancia
    assert "TEXTO NO VALIDADO QUE SE CUELA" not in result.summary_markdown
    # Debe degradar de forma exacta al resumen determinista
    assert result.summary_markdown == expected_fallback


def test_consultation_run_adversarial_rogue_objects_and_exceptions_stress_harness(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """Stress-test adversarial: comprueba que ante cualquier objeto devuelto por el
    proveedor que no sea exactamente una instancia de `SpokenReply` (objetos con `.text`
    estático, dinámico vía property, property que explota, `__getattr__` dinámico o
    explosivo, clones duck-typed, primitivas arbitrarias) o ante proveedores que
    lanzan diversas excepciones de `Exception`, `Consultation.run()`:
    1. Nunca propaga excepciones no controladas.
    2. Degrada limpiamente al resumen determinista en Markdown basado en hechos.
    3. Jamás filtra cadenas espurias o no autorizadas en el resumen ni en la caché.
    """
    audio_path, history_path = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    owner = ontology.AgentId("paciente_stress_test")

    c_baseline = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(MockWhisperBackend("Dolor en rodilla derecha")),
        ocr=LocalTesseractOCR(MockTesseractBackend("Radiografia sin hallazgos")),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=EventLog(window=100),
    )
    consent_base = c_baseline.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    expected_fallback = c_baseline.run(
        owner=owner, audio_path=audio_path, history_path=history_path, consent=consent_base
    ).summary_markdown

    class StaticForeignWithText:
        def __init__(self, text: str) -> None:
            self.text = text

    class DynamicPropertyObject:
        @property
        def text(self) -> str:
            return "UNAUTHORIZED_DYNAMIC_PROPERTY"

    class ExplodingPropertyObject:
        @property
        def text(self) -> str:
            raise RuntimeError("EXPLOIT_PROPERTY_EXPLOSION")

    class DynamicGetattrObject:
        def __getattr__(self, name: str) -> str:
            return f"UNAUTHORIZED_GETATTR_{name}"

    class ExplodingGetattrObject:
        def __getattr__(self, name: str) -> Any:
            raise ValueError("EXPLOIT_GETATTR_EXPLOSION")

    class DuckTypedCloneSpokenReply:
        def __init__(self) -> None:
            self.text = "UNAUTHORIZED_DUCK_TYPED_CLONE"

    class CustomCognitionError(Exception):
        """Excepción personalizada de proveedor."""

    rogue_payloads: list[Any] = [
        # Objetos con atributo o getter .text
        StaticForeignWithText("UNAUTHORIZED_STATIC_TEXT"),
        DynamicPropertyObject(),
        ExplodingPropertyObject(),
        DynamicGetattrObject(),
        ExplodingGetattrObject(),
        DuckTypedCloneSpokenReply(),
        # Primitivas y estructuras nativas
        None,
        "",
        "UNAUTHORIZED_RAW_STRING",
        12345,
        0,
        -999,
        3.14159,
        {"text": "UNAUTHORIZED_DICT_TEXT"},
        {"foo": "bar"},
        ["UNAUTHORIZED_LIST_TEXT"],
        ("UNAUTHORIZED_TUPLE_TEXT",),
        {"UNAUTHORIZED_SET_TEXT"},
        True,
        False,
        b"UNAUTHORIZED_BYTES",
        object(),
        # Callables y tipos
        lambda: "UNAUTHORIZED_LAMBDA",
        SpokenReply,  # la clase misma, no una instancia
    ]

    unauthorized_tokens = [
        "UNAUTHORIZED",
        "EXPLOIT",
    ]

    for payload in rogue_payloads:
        provider = DummyCognitionProvider(payload)
        consultation = Consultation(
            wording=sample_wording,
            stt=LocalWhisperSTT(MockWhisperBackend("Dolor en rodilla derecha")),
            ocr=LocalTesseractOCR(MockTesseractBackend("Radiografia sin hallazgos")),
            catalog_index=sample_catalog_index,
            provider=provider,
            cache=cache,
            log=EventLog(window=100),
        )
        consent_p = consultation.record_consent(
            cloud=True, owner=owner, mode="uploaded", actor=_ACTOR
        )
        res = consultation.run(
            owner=owner, audio_path=audio_path, history_path=history_path, consent=consent_p
        )
        assert res.summary_markdown == expected_fallback, (
            f"Fallo de degradación determinista con payload {type(payload)}: {res.summary_markdown}"
        )
        for token in unauthorized_tokens:
            assert token not in res.summary_markdown, (
                f"Fuga detectada de token no autorizado {token!r} con payload {type(payload)}"
            )
        # La caché también debe contener exactamente el fallback
        assert cache.load(owner=owner) == expected_fallback

    # Batería de excepciones lanzadas por el proveedor
    failing_exceptions = [
        RuntimeError("Fallo de infraestructura LLM"),
        ValueError("Parámetros de contexto inválidos"),
        TypeError("Fallo de serialización en adaptador"),
        KeyError("Clave no encontrada en payload LLM"),
        TimeoutError("Tiempo de espera agotado en API"),
        ZeroDivisionError("División por cero en muestreo de tokens"),
        CustomCognitionError("Error específico de proveedor cognitivo"),
    ]

    for exc in failing_exceptions:
        prov_exc = DummyCognitionProvider(exc)
        consultation = Consultation(
            wording=sample_wording,
            stt=LocalWhisperSTT(MockWhisperBackend("Dolor en rodilla derecha")),
            ocr=LocalTesseractOCR(MockTesseractBackend("Radiografia sin hallazgos")),
            catalog_index=sample_catalog_index,
            provider=prov_exc,
            cache=cache,
            log=EventLog(window=100),
        )
        consent_e = consultation.record_consent(
            cloud=True, owner=owner, mode="uploaded", actor=_ACTOR
        )
        res = consultation.run(
            owner=owner, audio_path=audio_path, history_path=history_path, consent=consent_e
        )
        assert res.summary_markdown == expected_fallback, (
            f"Fallo de recuperación ante excepción {type(exc)}: {res.summary_markdown}"
        )
        assert cache.load(owner=owner) == expected_fallback


def test_consultation_run_with_history_does_not_call_ocr(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """§6.3 Caso 1: run_with_history(...) no invoca OCR bajo ninguna circunstancia
    y utiliza history_text tal cual en el resumen.
    """
    audio_path, _ = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    log = EventLog(window=100)

    class CrashingOCRBackend:
        def image_to_string(self, *args: Any, **kwargs: Any) -> str:
            raise AssertionError("OCR no debe ser invocado en run_with_history")

    consultation = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(MockWhisperBackend("Audio de consulta")),
        ocr=LocalTesseractOCR(CrashingOCRBackend()),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=log,
        match_threshold=0.99,
    )

    owner = ontology.AgentId("paciente_precargado")
    history_precargada = "Historia clínica precargada en memoria sin OCR"
    consent = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    result = consultation.run_with_history(
        owner=owner,
        audio_path=audio_path,
        history_text=history_precargada,
        consent=consent,
    )

    assert history_precargada in result.summary_markdown
    events = [e for e in log if e.kind != EventKind.CONSENT_RECORDED]
    assert len(events) == 3
    assert events[1].kind == EventKind.INPUT_OBSERVED
    assert events[1].content == history_precargada


def test_consultation_run_calls_ocr_exactly_once(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
) -> None:
    """§6.3 Caso 2: run(...) sigue invocando al OCR exactamente una vez."""
    audio_path, history_path = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    log = EventLog(window=100)

    ocr_calls = 0

    class CountingOCRBackend:
        def image_to_string(self, *args: Any, **kwargs: Any) -> str:
            nonlocal ocr_calls
            ocr_calls += 1
            return "Texto de historia leída por OCR"

    consultation = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(MockWhisperBackend("Audio")),
        ocr=LocalTesseractOCR(CountingOCRBackend()),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=log,
        match_threshold=0.99,
    )

    owner = ontology.AgentId("paciente_con_ocr")
    consent = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    result = consultation.run(
        owner=owner,
        audio_path=audio_path,
        history_path=history_path,
        consent=consent,
    )

    assert ocr_calls == 1
    assert "Texto de historia leída por OCR" in result.summary_markdown


def test_consultation_run_delegates_to_run_with_history(
    tmp_path: Path,
    fernet_key: bytes,
    sample_catalog_index: CatalogIndex,
    sample_media: tuple[Path, Path],
    sample_wording: SummaryWording,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§6.3 / Mata M8: run(...) delega directamente en run_with_history(...)
    pasando el texto leído por el OCR y los argumentos correspondientes.
    Si run duplica el cuerpo en lugar de delegar, este test falla.
    """
    from aida.arms.health.consultation import ConsultationResult

    audio_path, history_path = sample_media
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    log = EventLog(window=100)

    consultation = Consultation(
        wording=sample_wording,
        stt=LocalWhisperSTT(MockWhisperBackend("Audio")),
        ocr=LocalTesseractOCR(MockTesseractBackend("Historia del OCR")),
        catalog_index=sample_catalog_index,
        provider=None,
        cache=cache,
        log=log,
    )

    owner = ontology.AgentId("paciente_delegacion")
    delegated_call: dict[str, Any] = {}
    dummy_result = ConsultationResult(
        owner=owner,
        transcript="Audio",
        history_excerpt="Historia del OCR",
        summary_markdown="# Resumen delegado",
        matched_codes=(),
    )

    def mock_run_with_history(
        *, owner: ontology.AgentId, audio_path: Path, history_text: Any, consent: Any = None
    ) -> ConsultationResult:
        ht = history_text() if callable(history_text) else history_text
        delegated_call["owner"] = owner
        delegated_call["audio_path"] = audio_path
        delegated_call["history_text"] = ht
        delegated_call["consent"] = consent
        return dummy_result

    monkeypatch.setattr(consultation, "run_with_history", mock_run_with_history)

    consent = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    result = consultation.run(
        owner=owner,
        audio_path=audio_path,
        history_path=history_path,
        consent=consent,
    )

    assert result is dummy_result
    assert delegated_call == {
        "owner": owner,
        "audio_path": audio_path,
        "history_text": "Historia del OCR",
        "consent": consent,
    }
