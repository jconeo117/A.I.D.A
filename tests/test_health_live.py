"""Pruebas unitarias de la consulta en vivo (Plan 77b, §B.2).

Cubre los casos 3 a 11:
3. feed de PCM con habla y silencio: transcriptor recibe enunciado y snapshot muestra frases.
4. WAV de la sesión tiene exactamente las muestras enviadas fuera de pausa.
5. En pausa, feed no escribe nada y snapshot trae una línea pause.
6. edit_field deja el campo en mine; una redacción posterior no lo toca.
7. Redacción salta con >= 6 frases o tras 8 s de silencio; nunca dos a la vez.
8. set_code y rejected(): snapshot lo trae con included: False.
9. Transcriptor con excepción: offline pasa a True y grabación sigue; recupera a False.
10. Error al leer documento: history() es "" y el vivo sigue.
11. stop cierra el WAV y devuelve su ruta; feed después de stop no hace nada.
"""

from __future__ import annotations

import time
import wave
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from cryptography.fernet import Fernet

from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT
from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.consultation import Consultation, PreparedConsultation
from aida.arms.health.extract import build_health_extractor
from aida.arms.health.live import LiveSession, line_kind
from aida.arms.health.match import build_index
from aida.arms.health.note import ClinicalNote, RenderedNote
from aida.arms.health.sections import SectionSpec, load_sections
from aida.arms.health.segment import ClinicalSegmenter
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import EventLog
from aida.engine.ontology import AgentId
from aida.voice.diarize import TranscriptSegment


class _MockLiveTranscriber:
    def __init__(self, segments_seq: Sequence[Sequence[TranscriptSegment]] | None = None) -> None:
        self._seq = list(segments_seq) if segments_seq is not None else []
        self.calls: list[tuple[Path, str]] = []
        self.fail = False

    def transcribe_segments(
        self, audio_path: Path, *, context: str = ""
    ) -> tuple[TranscriptSegment, ...]:
        self.calls.append((audio_path, context))
        if self.fail:
            raise RuntimeError("fallo de red en transcripción")
        if self._seq:
            return tuple(self._seq.pop(0))
        return (TranscriptSegment(0.0, 1.0, "Consulta por dolor de cabeza."),)


def _fake_vad(audio: np.ndarray) -> np.ndarray:
    chunk = audio[-512:] if audio.size >= 512 else audio
    prob = 0.9 if np.max(np.abs(chunk)) > 0.05 else 0.0
    return np.array([prob], dtype=np.float32)


def _pcm_speech(chunks: int = 5) -> bytes:
    return (np.ones(512 * chunks, dtype=np.int16) * 10000).tobytes()


def _pcm_silence(chunks: int = 26) -> bytes:
    return np.zeros(512 * chunks, dtype=np.int16).tobytes()


def _crear_consultation(tmp_path: Path) -> Consultation:
    from aida.arms.health.note import load_note_settings

    catalog = build_index((DiagnosticCode("R51", "Cefalea", 3, "ambas"),))
    note_settings = load_note_settings()

    class _SpyNoteProvider:
        def think(self, briefing: str, schema: type) -> object | None:
            return ClinicalNote(fields={"motivo": "Cefalea según IA"})

    return Consultation(
        wording=default_summary_wording(),
        stt=LocalWhisperSTT(model=object()),
        ocr=LocalTesseractOCR(pytesseract_module=object()),
        catalog_index=catalog,
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=EventLog(window=100),
        segmenter=ClinicalSegmenter(load_sections()),
        extractor=build_health_extractor(),
        note_settings=note_settings,
        note_providers=(("test", _SpyNoteProvider()),),
    )


def test_live_session_feed_pcm_y_snapshot(tmp_path: Path) -> None:
    """Caso 3 (§B.2): feed de PCM con habla y silencio activa transcripción

    y las frases aparecen en snapshot.
    """
    consultation = _crear_consultation(tmp_path)
    transcriber = _MockLiveTranscriber()
    secciones = (SectionSpec("motivo", "Motivo", (), codable=True, in_note=True),)
    consent = consultation.record_consent(owner=AgentId("doc_01"), mode="recorded")
    session = LiveSession(
        consultation=consultation,
        owner="doc_01",
        note_sections=secciones,
        history_text=lambda: "",
        workdir=tmp_path,
        transcriber=transcriber,
        vad=_fake_vad,
        consent=consent,
    )
    session.start()
    try:
        session.feed(_pcm_speech(5))
        session.feed(_pcm_silence(26))
        # Esperar a que el oyente procese el enunciado
        deadline = time.monotonic() + 3.0
        snap = session.snapshot(0)
        while time.monotonic() < deadline and not snap["lines"]:
            time.sleep(0.05)
            snap = session.snapshot(0)

        assert len(transcriber.calls) >= 1
        assert len(snap["lines"]) >= 1
        assert snap["lines"][0]["text"] == "Consulta por dolor de cabeza."
    finally:
        session.stop()


def test_live_session_wav_muestras_exactas(tmp_path: Path) -> None:
    """Caso 4 (§B.2): El WAV de la sesión tiene exactamente las muestras

    enviadas fuera de pausa.
    """
    consultation = _crear_consultation(tmp_path)
    secciones = (SectionSpec("motivo", "Motivo", (), codable=True, in_note=True),)
    consent = consultation.record_consent(owner=AgentId("doc_01"), mode="recorded")
    session = LiveSession(
        consultation=consultation,
        owner="doc_01",
        note_sections=secciones,
        history_text=lambda: "",
        workdir=tmp_path,
        transcriber=None,
        vad=_fake_vad,
        consent=consent,
    )
    part1 = (np.ones(512 * 2, dtype=np.int16) * 111).tobytes()
    part2 = (np.ones(512 * 3, dtype=np.int16) * 222).tobytes()
    session.feed(part1)
    session.feed(part2)
    wav_path = session.stop()

    with wave.open(str(wav_path), "rb") as w:
        frames = w.readframes(w.getnframes())
        assert frames == part1 + part2


def test_live_session_pausa_no_escribe_pcm_y_snapshot_marca_pause(tmp_path: Path) -> None:
    """Caso 5 (§B.2, M2): En pausa, feed no escribe nada al WAV y snapshot

    trae una línea pause.
    """
    consultation = _crear_consultation(tmp_path)
    secciones = (SectionSpec("motivo", "Motivo", (), codable=True, in_note=True),)
    consent = consultation.record_consent(owner=AgentId("doc_01"), mode="recorded")
    session = LiveSession(
        consultation=consultation,
        owner="doc_01",
        note_sections=secciones,
        history_text=lambda: "",
        workdir=tmp_path,
        transcriber=None,
        vad=_fake_vad,
        consent=consent,
    )
    part1 = (np.ones(512 * 2, dtype=np.int16) * 10).tobytes()
    paused_part = (np.ones(512 * 4, dtype=np.int16) * 99).tobytes()
    part2 = (np.ones(512 * 2, dtype=np.int16) * 20).tobytes()

    session.feed(part1)
    session.pause(True)
    session.feed(paused_part)
    session.pause(False)
    session.feed(part2)

    snap = session.snapshot(0)
    lines = snap["lines"]
    assert any(line["kind"] == "pause" for line in lines)

    wav_path = session.stop()
    with wave.open(str(wav_path), "rb") as w:
        frames = w.readframes(w.getnframes())
        assert frames == part1 + part2
        assert paused_part not in frames


def test_live_session_edit_field_bloquea_campo_frente_a_redaccion(tmp_path: Path) -> None:
    """Caso 6 (§B.2, M3): edit_field deja el campo en 'mine' y una redacción

    posterior no lo toca.
    """
    consultation = _crear_consultation(tmp_path)
    spec_motivo = SectionSpec("motivo", "Motivo", (), codable=True, in_note=True)
    consent = consultation.record_consent(owner=AgentId("doc_01"), mode="recorded")
    session = LiveSession(
        consultation=consultation,
        owner="doc_01",
        note_sections=(spec_motivo,),
        history_text=lambda: "",
        workdir=tmp_path,
        transcriber=None,
        vad=_fake_vad,
        consent=consent,
    )
    # Médico edita el campo
    ok = session.edit_field("motivo", "Mi nota manual")
    assert ok is True
    assert session.mine() == {"motivo": "Mi nota manual"}

    snap = session.snapshot(0)
    f_motivo = next(f for f in snap["fields"] if f["id"] == "motivo")
    assert f_motivo["text"] == "Mi nota manual"
    assert f_motivo["state"] == "mine"

    # Redacción posterior intenta sobrescribir con IA
    preparado_mock = MagicMock(spec=PreparedConsultation)
    consultation.render_note = MagicMock(  # type: ignore[method-assign]
        return_value=RenderedNote(
            fields=((spec_motivo, "Texto redactado por IA"),),
            redacted=True,
        )
    )
    session._draft(preparado_mock)

    # El campo del médico se conserva íntegro
    snap2 = session.snapshot(0)
    f_motivo2 = next(f for f in snap2["fields"] if f["id"] == "motivo")
    assert f_motivo2["text"] == "Mi nota manual"
    assert f_motivo2["state"] == "mine"
    assert session.mine() == {"motivo": "Mi nota manual"}
    session.stop()


def test_live_session_redaccion_condiciones_y_no_concurrente(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caso 7 (§B.2, M4): La redacción salta con >= 6 frases nuevas o tras 8s

    sin hablar (reloj inyectado); nunca dos redacciones a la vez.
    """
    current_time = 1000.0

    def fake_clock() -> float:
        return current_time

    consultation = _crear_consultation(tmp_path)
    spec_motivo = SectionSpec("motivo", "Motivo", (), codable=True, in_note=True)
    consent = consultation.record_consent(owner=AgentId("doc_01"), mode="recorded")
    session = LiveSession(
        consultation=consultation,
        owner="doc_01",
        note_sections=(spec_motivo,),
        history_text=lambda: "",
        workdir=tmp_path,
        transcriber=None,
        vad=_fake_vad,
        consent=consent,
        clock=fake_clock,
    )
    # Hacemos que time.sleep en _watch sea ultra rápido para el test
    orig_sleep = time.sleep
    monkeypatch.setattr("aida.arms.health.live.time.sleep", lambda _: orig_sleep(0.01))

    draft_calls: list[PreparedConsultation | None] = []
    orig_draft = session._draft

    def spy_draft(prep: PreparedConsultation | None) -> None:
        draft_calls.append(prep)
        orig_draft(prep)

    session._draft = spy_draft  # type: ignore[assignment]
    session._prepared = MagicMock(spec=PreparedConsultation)

    def esperar_draft(esperado: int, timeout: float = 1.5) -> None:
        fin = time.time() + timeout
        while len(draft_calls) < esperado and time.time() < fin:
            orig_sleep(0.01)

    session.start()
    try:
        # 1. Agregamos 3 frases (< 6) y tiempo avanza solo 2s (< 8s)
        with session._lock:
            session._lines.extend([("line", f"Frase {i}") for i in range(3)])
            session._last_speech = current_time
        current_time += 2.0
        time.sleep(0.05)
        # NO debe haber redactado todavía
        assert len(draft_calls) == 0

        # 2. Ahora pasa el tiempo de silencio (>= 8s)
        current_time += 8.5
        esperar_draft(1)
        # Debe haber redactado por silencio
        assert len(draft_calls) == 1

        # 3. Agregamos 6 frases nuevas (>= 6) y tiempo avanza solo 0.1s (< 8s)
        with session._lock:
            session._lines.extend([("line", f"Frase extra {i}") for i in range(6)])
            session._last_speech = current_time
        current_time += 0.1
        esperar_draft(2)
        # Debe haber redactado de inmediato por cantidad de frases
        assert len(draft_calls) == 2

        # 4. Concurrencia: si _drafting es True, _watch no inicia otra redacción
        with session._lock:
            session._drafting = True
            session._lines.extend([("line", f"Frase concurrente {i}") for i in range(10)])
        current_time += 10.0
        time.sleep(0.05)
        assert len(draft_calls) == 2
    finally:
        session.stop()



def test_live_session_set_code_rejected_y_snapshot(tmp_path: Path) -> None:
    """Caso 8 (§B.2): set_code(c, False) y rejected() lo devuelve; snapshot

    lo reporta con included: False.
    """
    consultation = _crear_consultation(tmp_path)
    secciones = (SectionSpec("motivo", "Motivo", (), codable=True, in_note=True),)
    consent = consultation.record_consent(owner=AgentId("doc_01"), mode="recorded")
    session = LiveSession(
        consultation=consultation,
        owner="doc_01",
        note_sections=secciones,
        history_text=lambda: "",
        workdir=tmp_path,
        transcriber=None,
        vad=_fake_vad,
        consent=consent,
    )
    session._codes = (
        {
            "code": "R51",
            "description": "Cefalea",
            "origin": "today",
            "evidence": "dolor de cabeza",
        },
    )

    # Código inicialmente incluido
    snap0 = session.snapshot(0)
    assert snap0["codes"][0]["included"] is True

    # Descartar código
    session.set_code("R51", False)
    assert "R51" in session.rejected()

    snap1 = session.snapshot(0)
    assert snap1["codes"][0]["included"] is False

    # Volver a incluir
    session.set_code("R51", True)
    assert "R51" not in session.rejected()
    snap2 = session.snapshot(0)
    assert snap2["codes"][0]["included"] is True
    session.stop()


def test_live_session_transcriptor_error_offline_true_y_continua(tmp_path: Path) -> None:
    """Caso 9 (§B.2, M5): Si el transcriptor levanta excepción, offline pasa a True

    y la grabación sigue; con el siguiente enunciado bueno vuelve a False.
    """
    consultation = _crear_consultation(tmp_path)
    transcriber = _MockLiveTranscriber()
    secciones = (SectionSpec("motivo", "Motivo", (), codable=True, in_note=True),)
    consent = consultation.record_consent(owner=AgentId("doc_01"), mode="recorded")
    session = LiveSession(
        consultation=consultation,
        owner="doc_01",
        note_sections=secciones,
        history_text=lambda: "",
        workdir=tmp_path,
        transcriber=transcriber,
        vad=_fake_vad,
        consent=consent,
    )
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFF")

    # Transcriptor falla: no levanta excepción hacia afuera y offline=True
    transcriber.fail = True
    session._transcribe(dummy_wav)
    assert session.snapshot(0)["offline"] is True

    # Transcriptor se recupera: offline vuelve a False
    transcriber.fail = False
    session._transcribe(dummy_wav)
    assert session.snapshot(0)["offline"] is False
    assert len(session.snapshot(0)["lines"]) == 1
    session.stop()


def test_live_session_error_documento_historia_vacia_y_continua(tmp_path: Path) -> None:
    """Caso 10 (§B.2, M6): Si leer el documento levanta excepción, history() es ""

    y la sesión continúa normalmente.
    """
    consultation = _crear_consultation(tmp_path)

    def documento_roto() -> str:
        raise OSError("archivo inaccesible")

    secciones = (SectionSpec("motivo", "Motivo", (), codable=True, in_note=True),)
    consent = consultation.record_consent(owner=AgentId("doc_01"), mode="recorded")
    session = LiveSession(
        consultation=consultation,
        owner="doc_01",
        note_sections=secciones,
        history_text=documento_roto,
        workdir=tmp_path,
        transcriber=None,
        vad=_fake_vad,
        consent=consent,
    )
    session.start()
    time.sleep(0.05)
    assert session.history() == ""
    # La sesión puede recibir audio y detenerse con normalidad
    session.feed(_pcm_speech(2))
    wav_path = session.stop()
    assert wav_path.exists()


def test_live_session_stop_cierra_wav_y_feed_posterior_ignorado(tmp_path: Path) -> None:
    """Caso 11 (§B.2): stop cierra el WAV y devuelve su ruta; feed después de

    stop no hace nada.
    """
    consultation = _crear_consultation(tmp_path)
    secciones = (SectionSpec("motivo", "Motivo", (), codable=True, in_note=True),)
    consent = consultation.record_consent(owner=AgentId("doc_01"), mode="recorded")
    session = LiveSession(
        consultation=consultation,
        owner="doc_01",
        note_sections=secciones,
        history_text=lambda: "",
        workdir=tmp_path,
        transcriber=None,
        vad=_fake_vad,
        consent=consent,
    )
    initial_pcm = (np.ones(512 * 2, dtype=np.int16) * 44).tobytes()
    session.feed(initial_pcm)

    wav_path = session.stop()
    assert wav_path == session.audio_path
    assert wav_path.exists()

    # feed posterior no hace nada y no falla
    session.feed((np.ones(512 * 3, dtype=np.int16) * 55).tobytes())

    with wave.open(str(wav_path), "rb") as w:
        frames = w.readframes(w.getnframes())
        assert frames == initial_pcm


def test_line_kind() -> None:
    """line_kind clasifica con 'q' interrogaciones y 'a' respuestas."""
    assert line_kind("¿Qué síntomas tiene?") == "q"
    assert line_kind("Siento dolor de cabeza.") == "a"
