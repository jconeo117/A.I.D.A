"""Pruebas de la autorización registrada y bloqueante (Plan 82).

Cubre §B.2, §B.3 y §B.4 de .claude/workflow/82-plan-autorizacion.md:
§B.2 La puerta:
  5. transcribe, prepare, run, run_with_history con consent=None -> ConsentMissingError (0 STT).
  6. Autorización de otro dueño -> ConsentMissingError (0 STT).
  7. ConsentRecord forjado a mano -> ConsentMissingError (0 STT).
  8. Consent válido -> transcribe, 1 llamada STT.
  9. LiveSession sin autorización -> ConsentMissingError sin crear audio.wav.
  10. record_consent con modo desconocido -> ValueError, sin línea y sin evento.
§B.3 El registro:
  11. append encadena: primera prev con 64 ceros, segunda con digest anterior, digest coincide.
  12. Archivo con permisos 0o600 y recarga en nuevo ConsentLedger reconoce y sigue cadena.
  13. verify: íntegro (), línea borrada, campo cambiado, línea ilegible.
  14. Huella y versión exactas de consent.yaml para cada modo.
  15. ConsentTexts.line formato exacto.
  16. load_consent_texts valida claves obligatorias.
  17. record_consent emite CONSENT_RECORDED con mode, version y digest, sin content.
§B.4 La aplicación, la revisión, el export y la CLI:
  18. /empezar sin consent al subir y al grabar: 400 missing_consent, registro 0, STT 0.
  19. Con consent: registra uploaded/recorded, vivo recibe autorización y pasa al análisis final.
  20. Página de revisión lleva la línea de autorización; sin autorización no la lleva.
  21. Inicio lleva id="consent-box", textos del pack en CONSENT, sin consent_label.
  22. export_consultation con consent_line la escribe al final; sin ella no.
  23. CLI: --consultation sin --consent -> 1 y 0 llamadas STT; error de uso sale antes.
  24. CLI: --mic --history sin --consent -> 1 antes de abrir micrófono.
  25. CLI: --consents: lista y código 0 íntegra; 1 si rota.
  26. collect_review pasa consent_line a la página.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import stat
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
import yaml
from cryptography.fernet import Fernet

from aida.arms.health.app import (
    AppReady,
    _start_page,
    load_app_wording,
    serve,
)
from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT
from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.consent import (
    _GENESIS,
    ConsentLedger,
    ConsentMissingError,
    ConsentRecord,
    load_consent_texts,
)
from aida.arms.health.consultation import (
    Consultation,
    ConsultationResult,
)
from aida.arms.health.export import export_consultation
from aida.arms.health.live import LiveSession, LiveTranscriber
from aida.arms.health.match import build_index
from aida.arms.health.note import RenderedNote, load_note_settings
from aida.arms.health.review_ui import (
    _render_page,
    collect_review,
    history_view,
    load_ui_settings,
)
from aida.arms.health.sections import SectionSpec
from aida.arms.health.segment import build_health_segmenter
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.platform.identity import Professional, ProfileStore, workstation
from aida.platform.license import License, LicenseState
from aida.run import main
from aida.voice.diarize import TranscriptSegment
from tests.conftest import _ORIGINAL_HANDLE_REQUEST

_ACTOR = Professional(id="t", name="Prueba", registry="0")


@pytest.fixture(autouse=True)
def _permitir_servidor_app_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restaura el transporte real de httpx para comunicar con el servidor local."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


class _SpySTT(LocalWhisperSTT):
    def __init__(self) -> None:
        super().__init__(model=object())
        self.transcribe_calls = 0
        self.transcribe_segments_calls = 0

    def transcribe(self, audio_path: Path) -> str:
        self.transcribe_calls += 1
        return "El paciente refiere cefalea frontal."

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        self.transcribe_segments_calls += 1
        return (
            TranscriptSegment(0.0, 2.0, "Motivo de consulta: cefalea frontal pulsátil."),
            TranscriptSegment(2.0, 4.0, "Plan de manejo: paracetamol cada 8 horas."),
        )


class _DummyOCR(LocalTesseractOCR):
    def __init__(self) -> None:
        super().__init__(pytesseract_module=object())

    def read(self, path: Path) -> str:
        return "Historia clínica previa: cefalea."


def _crear_wav_minimo() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def _build_consultation(
    tmp_path: Path,
    stt: _SpySTT | None = None,
    log: EventLog | None = None,
    ledger: ConsentLedger | None = None,
) -> Consultation:
    return Consultation(
        wording=default_summary_wording(),
        stt=stt if stt is not None else _SpySTT(),
        ocr=_DummyOCR(),
        catalog_index=build_index((DiagnosticCode("R51X", "Cefalea", 4, "cat"),)),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=log if log is not None else EventLog(),
        consents=ledger,
    )


def _crear_app_ready(
    tmp_path: Path,
    stt: _SpySTT | None = None,
    ledger: ConsentLedger | None = None,
    live_stt: LiveTranscriber | None = None,
) -> AppReady:
    log = EventLog()
    consultation = Consultation(
        wording=default_summary_wording(),
        stt=stt if stt is not None else _SpySTT(),
        ocr=_DummyOCR(),
        catalog_index=build_index((DiagnosticCode("R51X", "Cefalea", 4, "cat"),)),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=log,
        consents=ledger,
        segmenter=build_health_segmenter(),
        note_settings=load_note_settings(),
    )
    return AppReady(
        consultation=consultation,
        reader=_DummyOCR(),
        log=log,
        cloud=True,
        writer="modelo-prueba",
        live_stt=live_stt,
    )


def _lanzar_servidor(
    build_fn: Callable[[], AppReady],
    **kwargs: Any,
) -> tuple[str, str, int, threading.Thread]:
    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    serve_kwargs = {
        "settings": load_ui_settings(),
        "wording": load_app_wording(),
        "open_browser": _abrir,
        "vad_factory": lambda: lambda audio: np.zeros(1, dtype=np.float32),
        **kwargs,
    }
    if "profiles" not in serve_kwargs:
        serve_kwargs["profiles"] = ProfileStore(Path(tempfile.mkdtemp()) / "profiles.json")
    if "license" not in serve_kwargs:
        serve_kwargs["license"] = LicenseState(
            status="valid",
            license=License(
                id="t",
                client="t",
                arms=("health",),
                issued=date.today(),
                expires=date.today() + timedelta(days=365),
            ),
        )
    hilo = threading.Thread(
        target=serve,
        args=(build_fn,),
        kwargs=serve_kwargs,
        daemon=True,
    )
    hilo.start()
    assert listo.wait(timeout=5.0)
    url = info["url"]
    partes = url.rsplit("/", 1)
    httpx.post(
        f"{url}/perfil",
        json={"name": "Dra. Laura Gómez", "registry": "52.341.889", "pin": "1234", "pin2": "1234"},
        timeout=5.0,
    )
    return url, partes[1], int(partes[0].split(":")[-1]), hilo


# ---------------------------------------------------------------------------
# §B.2 — La puerta
# ---------------------------------------------------------------------------


def test_consent_missing_en_transcribe_prepare_run_run_with_history_cero_llamadas_stt(
    tmp_path: Path,
) -> None:
    """5. transcribe, prepare, run y run_with_history con consent=None lanzan
    ConsentMissingError y el STT tiene 0 llamadas."""
    stt = _SpySTT()
    c = _build_consultation(tmp_path, stt=stt)
    owner = AgentId("doc_01")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(_crear_wav_minimo())
    hist = tmp_path / "hist.png"
    hist.write_bytes(b"dummy")

    with pytest.raises(ConsentMissingError):
        c.transcribe(owner=owner, audio_path=audio, consent=None)
    assert stt.transcribe_calls == 0

    with pytest.raises(ConsentMissingError):
        c.prepare(owner=owner, audio_path=audio, history_text="...", consent=None)
    assert stt.transcribe_calls == 0

    with pytest.raises(ConsentMissingError):
        c.run(owner=owner, audio_path=audio, history_path=hist, consent=None)
    assert stt.transcribe_calls == 0

    with pytest.raises(ConsentMissingError):
        c.run_with_history(owner=owner, audio_path=audio, history_text="...", consent=None)
    assert stt.transcribe_calls == 0


def test_consent_otro_dueno_lanza_consent_missing_error_cero_llamadas_stt(
    tmp_path: Path,
) -> None:
    """6. Con la autorización de otro dueño: ConsentMissingError, 0 llamadas."""
    stt = _SpySTT()
    c = _build_consultation(tmp_path, stt=stt)
    owner_bueno = AgentId("doc_autorizado")
    owner_ajeno = AgentId("doc_otro")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(_crear_wav_minimo())

    consent = c.record_consent(cloud=True, owner=owner_bueno, mode="uploaded", actor=_ACTOR)

    with pytest.raises(ConsentMissingError):
        c.transcribe(owner=owner_ajeno, audio_path=audio, consent=consent)
    assert stt.transcribe_calls == 0


def test_consent_record_forjado_a_mano_lanza_consent_missing_error_cero_llamadas_stt(
    tmp_path: Path,
) -> None:
    """7. Con un ConsentRecord construido a mano (no salido de record_consent),
    aunque el dueño coincida: ConsentMissingError, 0 llamadas."""
    stt = _SpySTT()
    c = _build_consultation(tmp_path, stt=stt)
    owner = AgentId("doc_forjador")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(_crear_wav_minimo())

    forjado = ConsentRecord(
        owner=owner,
        mode="uploaded",
        version="v1",
        text_sha256="abc",
        cloud=True,
        workstation=workstation(),
        at="2026-10-06T10:00:00-05:00",
        prev=_GENESIS,
        digest="invalido",
    )

    with pytest.raises(ConsentMissingError):
        c.transcribe(owner=owner, audio_path=audio, consent=forjado)
    assert stt.transcribe_calls == 0


def test_consent_valido_transcribe_llama_stt_una_vez(tmp_path: Path) -> None:
    """8. Con la buena: transcribe, 1 llamada."""
    stt = _SpySTT()
    c = _build_consultation(tmp_path, stt=stt)
    owner = AgentId("doc_valido")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(_crear_wav_minimo())

    consent = c.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    res = c.transcribe(owner=owner, audio_path=audio, consent=consent)

    assert "cefalea" in res
    assert stt.transcribe_calls == 1


def test_live_session_sin_consent_lanza_consent_missing_error_y_no_crea_wav(
    tmp_path: Path,
) -> None:
    """9. LiveSession sin autorización lanza ConsentMissingError y no crea audio.wav en workdir."""
    c = _build_consultation(tmp_path)
    owner = AgentId("doc_live")
    workdir = tmp_path / "sesion_live"
    workdir.mkdir(parents=True)

    with pytest.raises(ConsentMissingError):
        LiveSession(
            consultation=c,
            owner=str(owner),
            note_sections=(),
            history_text=lambda: "",
            workdir=workdir,
            transcriber=MagicMock(),
            vad=lambda pcm: pcm,
            consent=None,
        )

    assert not (workdir / "audio.wav").exists()


def test_record_consent_modo_desconocido_lanza_value_error_sin_linea_ni_evento(
    tmp_path: Path,
) -> None:
    """10. record_consent con un modo desconocido: ValueError, sin línea y sin evento."""
    log = EventLog()
    c = _build_consultation(tmp_path, log=log)
    owner = AgentId("doc_modo")

    with pytest.raises(ValueError, match="modo de autorización desconocido"):
        c.record_consent(cloud=True, owner=owner, mode="inventado", actor=_ACTOR)

    assert len(list(log)) == 0
    assert len(c._consents.records()) == 0


# ---------------------------------------------------------------------------
# §B.3 — El registro
# ---------------------------------------------------------------------------


def test_consent_ledger_append_encadena_prev_y_digest(tmp_path: Path) -> None:
    """11. append encadena: la primera prev es 64 ceros; cada una la huella de la anterior;
    digest recalculado coincide."""
    ledger = ConsentLedger()
    owner = AgentId("doc_chain")

    r1 = ledger.append(
        owner=owner,
        mode="uploaded",
        version="v1",
        text="Texto 1",
        cloud=True,
        workstation="ws1",
        at="2026-10-06T10:00:00-05:00",
    )
    assert r1.prev == _GENESIS
    assert (
        r1.digest
        == hashlib.sha256(
            json.dumps(
                {
                    "at": "2026-10-06T10:00:00-05:00",
                    "cloud": "true",
                    "mode": "uploaded",
                    "owner": "doc_chain",
                    "prev": _GENESIS,
                    "text_sha256": hashlib.sha256(b"Texto 1").hexdigest(),
                    "version": "v1",
                    "workstation": "ws1",
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )

    r2 = ledger.append(
        owner=owner,
        mode="recorded",
        version="v1",
        text="Texto 2",
        cloud=True,
        workstation="ws1",
        at="2026-10-06T10:05:00-05:00",
    )
    assert r2.prev == r1.digest
    assert r2.digest != r1.digest


def test_consent_ledger_con_archivo_permisos_0600_y_recarga(tmp_path: Path) -> None:
    """12. Con archivo: una línea por autorización, 0o600, y un ConsentLedger nuevo
    sobre el mismo archivo la reconoce (contains) y sigue la cadena."""
    archivo = tmp_path / "autorizaciones.jsonl"
    ledger1 = ConsentLedger(archivo)
    r1 = ledger1.append(
        owner=AgentId("doc_persist"),
        mode="uploaded",
        version="v1",
        text="Texto",
        cloud=True,
        workstation="ws",
        at="2026-10-06T10:00:00-05:00",
    )

    assert archivo.is_file()
    assert stat.S_IMODE(archivo.stat().st_mode) == 0o600
    lineas = archivo.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 1

    # Recarga desde disco
    ledger2 = ConsentLedger(archivo)
    assert ledger2.contains(r1)
    r2 = ledger2.append(
        owner=AgentId("doc_persist_2"),
        mode="recorded",
        version="v1",
        text="Texto",
        cloud=True,
        workstation="ws",
        at="2026-10-06T10:10:00-05:00",
    )
    assert r2.prev == r1.digest
    assert len(archivo.read_text(encoding="utf-8").splitlines()) == 2


def test_consent_ledger_verify_detecta_corrupciones(tmp_path: Path) -> None:
    """13. verify: íntegro => (); línea borrada => «no sigue a la anterior»;
    campo cambiado => «no corresponde a su huella»; línea ilegible => «no se puede leer»."""
    archivo = tmp_path / "autorizaciones.jsonl"
    ledger = ConsentLedger(archivo)

    # Crear 3 autorizaciones íntegras
    for i in range(1, 4):
        ledger.append(
            owner=AgentId(f"doc_{i}"),
            mode="uploaded",
            version="v1",
            text=f"Texto {i}",
            cloud=True,
            workstation="ws",
            at=f"2026-10-06T10:0{i}:00-05:00",
        )
    assert ledger.verify() == ()

    lineas_originales = archivo.read_text(encoding="utf-8").splitlines()

    # 1. Línea borrada (borramos la segunda)
    archivo.write_text(
        "\n".join([lineas_originales[0], lineas_originales[2]]) + "\n", encoding="utf-8"
    )
    problemas_borrada = ledger.verify()
    assert len(problemas_borrada) == 1
    assert "no sigue a la anterior" in problemas_borrada[0]

    # 2. Campo cambiado (modificamos el owner en la línea 3 sin cambiar digest)
    datos3 = json.loads(lineas_originales[2])
    datos3["owner"] = "doc_falsificado"
    archivo.write_text(
        "\n".join([lineas_originales[0], lineas_originales[1], json.dumps(datos3)]) + "\n",
        encoding="utf-8",
    )
    problemas_cambiada = ledger.verify()
    assert len(problemas_cambiada) == 1
    assert "no corresponde a su huella" in problemas_cambiada[0]

    # 3. Línea ilegible
    archivo.write_text(
        "\n".join([lineas_originales[0], "{ json roto", lineas_originales[2]]) + "\n",
        encoding="utf-8",
    )
    problemas_rota = ledger.verify()
    assert any("no se puede leer" in p for p in problemas_rota)


def test_consent_huella_del_texto_y_version_exactas_de_consent_yaml() -> None:
    """14. La huella del texto es la del texto exacto de consent.yaml para el modo,
    y la versión, la del archivo."""
    textos = load_consent_texts()
    assert textos.version == "v2"

    for modo in ("recorded", "uploaded"):
        texto_modo_cloud = textos.read_text(modo, cloud=True)
        huella_calculada_cloud = hashlib.sha256(texto_modo_cloud.encode("utf-8")).hexdigest()
        assert len(huella_calculada_cloud) == 64
        texto_modo_local = textos.read_text(modo, cloud=False)
        huella_calculada_local = hashlib.sha256(texto_modo_local.encode("utf-8")).hexdigest()
        assert len(huella_calculada_local) == 64
        assert huella_calculada_cloud != huella_calculada_local


def test_consent_texts_line_formato_exacto() -> None:
    """15. ConsentTexts.line da «Autorización registrada · grabada · 06/10/2026 10:32
    · texto v2 · con nube» para una autorización de esa fecha."""
    textos = load_consent_texts()
    rec = ConsentRecord(
        owner=AgentId("doc_test"),
        mode="recorded",
        version="v2",
        text_sha256="abc",
        cloud=True,
        workstation="ws",
        at="2026-10-06T10:32:00-05:00",
        prev=_GENESIS,
        digest="xyz",
    )
    linea = textos.line(rec)
    assert linea == "Autorización registrada · grabada · 06/10/2026 10:32 · texto v2 · con nube"


def test_load_consent_texts_valida_claves_obligatorias(tmp_path: Path) -> None:
    """16. load_consent_texts revienta si falta version, line_template o cualquiera de
    las cuatro claves."""
    base = {
        "version": "v2",
        "line_template": "Autorización registrada · {0} · {1} · texto {2} · {3}",
        "cloud_labels": {"true": "con nube", "false": "solo en este equipo"},
        "recorded": {
            "title": "T",
            "text": "X",
            "check": "C",
            "label": "grabada",
            "cloud": {"text": "XT", "check": "XC"},
        },
        "uploaded": {
            "title": "T",
            "text": "X",
            "check": "C",
            "label": "subida",
            "cloud": {"text": "XT", "check": "XC"},
        },
    }

    # Falta version
    d1 = dict(base)
    del d1["version"]
    p1 = tmp_path / "c1.yaml"
    p1.write_text(yaml.dump(d1), encoding="utf-8")
    with pytest.raises(ValueError, match="'version' debe ser una cadena no vacía"):
        load_consent_texts(p1)

    # Falta line_template
    d2 = dict(base)
    del d2["line_template"]
    p2 = tmp_path / "c2.yaml"
    p2.write_text(yaml.dump(d2), encoding="utf-8")
    with pytest.raises(ValueError, match="'line_template' debe ser una cadena no vacía"):
        load_consent_texts(p2)

    # Falta una clave en recorded
    d3 = dict(base)
    d3["recorded"] = {"title": "T", "text": "X", "check": "C"}  # falta label
    p3 = tmp_path / "c3.yaml"
    p3.write_text(yaml.dump(d3), encoding="utf-8")
    with pytest.raises(ValueError, match="'recorded' necesita"):
        load_consent_texts(p3)


def test_record_consent_emite_consent_recorded_sin_content(tmp_path: Path) -> None:
    """17. record_consent emite un CONSENT_RECORDED con mode, version y digest, y sin content."""
    log = EventLog()
    c = _build_consultation(tmp_path, log=log)
    owner = AgentId("doc_audit")

    rec = c.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)

    eventos = [e for e in log if e.kind == EventKind.CONSENT_RECORDED]
    assert len(eventos) == 1
    ev = eventos[0]
    assert ev.owner == owner
    assert ev.tick == 0
    assert ev.data == {
        "mode": "uploaded",
        "version": rec.version,
        "digest": rec.digest,
        "cloud": "true",
    }
    assert ev.content is None


# ---------------------------------------------------------------------------
# §B.4 — La aplicación, la revisión, el export y la CLI
# ---------------------------------------------------------------------------


def test_servidor_empezar_sin_consent_responde_400_missing_consent(tmp_path: Path) -> None:
    """18. /empezar sin consent al subir y al grabar: 400 con missing_consent,
    el registro sin líneas y el STT sin llamadas."""
    stt = _SpySTT()
    ledger = ConsentLedger(tmp_path / "autorizaciones.jsonl")
    ready = _crear_app_ready(tmp_path, stt=stt, ledger=ledger)
    url, _token, _, hilo = _lanzar_servidor(lambda: ready)

    audio_bytes = _crear_wav_minimo()
    wording = load_app_wording()
    try:
        # Subir audio y documento primero
        httpx.post(
            f"{url}/audio", content=audio_bytes, headers={"X-Name": "consulta.wav"}, timeout=5.0
        )
        httpx.post(
            f"{url}/documento",
            content=b"Historia previa",
            headers={"X-Name": "historia.txt"},
            timeout=5.0,
        )

        # 1. Al subir (recorded=False) sin consent
        r_subir = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-001", "consent": False, "recorded": False},
            timeout=5.0,
        )
        assert r_subir.status_code == 400
        assert r_subir.json()["error"] == wording["missing_consent"]
        assert len(ledger.records()) == 0
        assert stt.transcribe_calls == 0

        # 2. Al grabar (recorded=True) sin consent
        r_grabar = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-002", "consent": False, "recorded": True},
            timeout=5.0,
        )
        assert r_grabar.status_code == 400
        assert r_grabar.json()["error"] == wording["missing_consent"]
        assert len(ledger.records()) == 0
        assert stt.transcribe_calls == 0
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_empezar_con_consent_registra_y_pasa_a_vivo_o_analisis(
    tmp_path: Path,
) -> None:
    """19. Con consent: una línea con el modo correcto; al grabar, el vivo recibe esa
    misma autorización y el análisis final pasa con ella (_finish -> procesar)."""
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    # 1. Prueba de subida: registra mode="uploaded" y procesa
    stt_up = _SpySTT()
    ledger_up = ConsentLedger(tmp_path / "autorizaciones_up.jsonl")
    ready_up = _crear_app_ready(tmp_path / "up", stt=stt_up, ledger=ledger_up)
    url_up, _, _, hilo_up = _lanzar_servidor(lambda: ready_up)

    try:
        httpx.post(
            f"{url_up}/audio",
            content=_crear_wav_minimo(),
            headers={"X-Name": "consulta.wav"},
            timeout=5.0,
        )
        httpx.post(
            f"{url_up}/documento",
            content=b"Historia previa",
            headers={"X-Name": "historia.txt"},
            timeout=5.0,
        )
        r_subir = httpx.post(
            f"{url_up}/empezar",
            json={"owner": "HC-001", "consent": True, "recorded": False, "cloud": True},
            timeout=5.0,
        )
        assert r_subir.status_code == 200
        assert len(ledger_up.records()) == 1
        assert ledger_up.records()[0].mode == "uploaded"

        for _ in range(50):
            estado = httpx.get(f"{url_up}/estado", timeout=5.0).json()
            if estado.get("state") == "review":
                break
            time.sleep(0.05)
        assert estado.get("state") == "review"
    finally:
        httpx.post(f"{url_up}/salir", timeout=5.0)
        hilo_up.join(timeout=5.0)

    # 2. Prueba de grabación: registra mode="recorded", el vivo recibe la
    # autorización y termina procesando con ella
    stt_rec = _SpySTT()
    ledger_rec = ConsentLedger(tmp_path / "autorizaciones_rec.jsonl")
    ready_rec = _crear_app_ready(tmp_path / "rec", stt=stt_rec, ledger=ledger_rec)
    url_rec, _, _, hilo_rec = _lanzar_servidor(lambda: ready_rec)

    try:
        httpx.post(
            f"{url_rec}/documento",
            content=b"Historia previa",
            headers={"X-Name": "historia.txt"},
            timeout=5.0,
        )
        r_grabar = httpx.post(
            f"{url_rec}/empezar",
            json={"owner": "HC-002", "consent": True, "recorded": True, "cloud": True},
            timeout=5.0,
        )
        assert r_grabar.status_code == 200
        assert len(ledger_rec.records()) == 1
        assert ledger_rec.records()[0].mode == "recorded"

        # Terminar grabación para que _finish llame a procesar con vivo.consent
        r_term = httpx.post(f"{url_rec}/terminar", timeout=5.0)
        assert r_term.status_code == 200

        for _ in range(50):
            estado = httpx.get(f"{url_rec}/estado", timeout=5.0).json()
            if estado.get("state") == "review":
                break
            time.sleep(0.05)
        assert estado.get("state") == "review"
    finally:
        httpx.post(f"{url_rec}/salir", timeout=5.0)
        hilo_rec.join(timeout=5.0)


def test_review_ui_render_page_incluye_consent_line_si_se_pasa() -> None:
    """20. La página de revisión lleva la línea de la autorización; sin autorización
    en la sesión (_render_page directo), no la lleva."""
    spec = SectionSpec("motivo", "Motivo de consulta", (), True, True)
    note = RenderedNote(fields=((spec, "Cefalea"),), redacted=True)
    ui_settings = load_ui_settings()
    linea = "Autorización registrada · subida · 06/10/2026 10:32 · texto v1"

    # Con consent_line
    html_con = _render_page(
        matches=(),
        note=note,
        transcript="",
        history=history_view(None, ""),
        owner="HC-01",
        moment=datetime.now(),
        wording=ui_settings.wording,
        token="tok",
        consent_line=linea,
    )
    assert linea in html_con
    assert "i-shield" in html_con

    # Sin consent_line
    html_sin = _render_page(
        matches=(),
        note=note,
        transcript="",
        history=history_view(None, ""),
        owner="HC-01",
        moment=datetime.now(),
        wording=ui_settings.wording,
        token="tok",
        consent_line=None,
    )
    assert "Autorización registrada" not in html_sin


def test_start_page_contiene_consent_box_textos_y_no_consent_label(tmp_path: Path) -> None:
    """21. El inicio lleva id="consent-box", los dos textos del pack en CONSENT y la
    versión; no lleva consent_label."""
    wording = load_app_wording()
    consent_texts = load_consent_texts()
    ready = _crear_app_ready(tmp_path)

    html = _start_page(
        w=wording,
        brand="AIDA",
        token="tok",
        ready=ready,
        moment=datetime.now(),
        consent=consent_texts,
    )

    assert 'id="consent-box"' in html
    assert 'id="consent-check"' in html
    assert "consent_label" not in html
    assert consent_texts.version in html
    assert consent_texts.modes["uploaded"].title in html
    assert consent_texts.modes["recorded"].title in html


def test_export_consultation_con_consent_line_lo_escribe_al_final(tmp_path: Path) -> None:
    """22. export_consultation(..., consent_line=...) la escribe al final;
    sin ella, el archivo es igual que antes."""
    spec = SectionSpec("motivo", "Motivo de consulta", (), True, True)
    note = RenderedNote(fields=((spec, "Cefalea intensa"),), redacted=True)
    out_dir = tmp_path / "export"
    linea = "Autorización registrada · subida · 06/10/2026 10:32 · texto v1"

    res_con = ConsultationResult(
        owner=AgentId("HC-01"),
        transcript="",
        history_excerpt="",
        matched_codes=(),
        summary_markdown="- Cefalea",
        note=note,
    )

    # Con consent_line
    dest_con = out_dir / "resumen_con.md"
    archivos_con = export_consultation(
        res_con,
        destination=dest_con,
        moment=datetime.now(),
        include_model=False,
        consent_line=linea,
    )
    contenido_con = archivos_con.summary.read_text(encoding="utf-8")
    assert contenido_con.strip().endswith(linea)

    # Sin consent_line
    res_sin = ConsultationResult(
        owner=AgentId("HC-02"),
        transcript="",
        history_excerpt="",
        matched_codes=(),
        summary_markdown="- Cefalea",
        note=note,
    )
    dest_sin = out_dir / "resumen_sin.md"
    archivos_sin = export_consultation(
        res_sin,
        destination=dest_sin,
        moment=datetime.now(),
        include_model=False,
        consent_line=None,
    )
    contenido_sin = archivos_sin.summary.read_text(encoding="utf-8")
    assert "Autorización registrada" not in contenido_sin


def test_cli_consultation_sin_consent_devuelve_uno_sin_stt_y_error_uso_antes(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """23. CLI: --consultation sin --consent => 1, el mensaje, y el STT simulado
    sin llamadas. Un error de uso (--ui sin --review) sigue saliendo antes que la falta
    de --consent."""
    pytest.importorskip("faster_whisper", reason="requiere el extra 'capture'")
    stt_calls = 0

    class _SpyBackend:
        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal stt_calls
            stt_calls += 1
            return [], None

    monkeypatch.setattr("faster_whisper.WhisperModel", lambda *a, **kw: _SpyBackend())
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", Fernet.generate_key().decode())

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF dummy")
    hist = tmp_path / "h.png"
    hist.write_bytes(b"dummy")

    # 1. Sin --consent: código 1, mensaje, 0 llamadas STT
    codigo = main(["--consultation", "--audio", str(audio), "--history", str(hist)])
    captured = capsys.readouterr()
    assert codigo == 1
    assert "falta --consent" in captured.err
    assert stt_calls == 0

    # 2. Error de uso (--ui sin --review) antes de comprobar consent
    codigo_uso = main(["--consultation", "--ui", "--audio", str(audio), "--history", str(hist)])
    captured_uso = capsys.readouterr()
    assert codigo_uso == 2
    assert "--ui requiere --review" in captured_uso.err
    assert "falta --consent" not in captured_uso.err


def test_cli_mic_sin_consent_devuelve_uno_antes_de_abrir_microfono(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """24. --mic --history sin --consent => 1 antes de abrir el micrófono."""
    monkeypatch.setattr("aida.run._check_mic_dependencies", lambda: True)
    hist = tmp_path / "h.png"
    hist.write_bytes(b"dummy")

    codigo = main(
        [
            "--pack",
            "packs/health/consultorio/pack.yaml",
            "--converse",
            "--mic",
            "--history",
            str(hist),
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 1
    assert "falta --consent" in captured.err


def test_cli_consents_lista_cadena_integra_y_falla_si_rota(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """25. --consents: lista y código 0 con la cadena íntegra; 1 con una línea cambiada."""
    falso_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: falso_home)

    ruta_consents = falso_home / ".local" / "share" / "aida" / "autorizaciones.jsonl"
    ledger = ConsentLedger(ruta_consents)
    ledger.append(
        owner=AgentId("HC-0042"),
        mode="uploaded",
        version="v1",
        text="Texto",
        cloud=True,
        workstation="ws",
        at="2026-10-06T10:00:00-05:00",
    )
    ledger.append(
        owner=AgentId("HC-0042"),
        mode="recorded",
        version="v1",
        text="Texto",
        cloud=True,
        workstation="ws",
        at="2026-10-06T10:05:00-05:00",
    )

    # 1. Cadena íntegra -> código 0
    codigo_ok = main(["--consents"])
    cap_ok = capsys.readouterr()
    assert codigo_ok == 0
    assert "2 autorizaciones, cadena íntegra" in cap_ok.err
    assert "HC-0042" in cap_ok.out

    # 2. Línea cambiada -> código 1
    lineas = ruta_consents.read_text(encoding="utf-8").splitlines()
    datos2 = json.loads(lineas[1])
    datos2["owner"] = "HC-HACK"
    ruta_consents.write_text(lineas[0] + "\n" + json.dumps(datos2) + "\n", encoding="utf-8")

    codigo_roto = main(["--consents"])
    cap_roto = capsys.readouterr()
    assert codigo_roto == 1
    assert "su contenido no corresponde a su huella" in cap_roto.err


def test_collect_review_pasa_consent_line_a_pagina() -> None:
    """26. collect_review(..., consent_line=...) la pasa a la página."""
    sig = inspect.signature(collect_review)
    assert "consent_line" in sig.parameters

    spec = SectionSpec("motivo", "Motivo de consulta", (), True, True)
    note = RenderedNote(fields=((spec, "Cefalea"),), redacted=True)
    ui_settings = load_ui_settings()
    linea = "Autorización registrada · grabada · 06/10/2026 10:32 · texto v1"

    html = _render_page(
        matches=(),
        note=note,
        transcript="",
        history=history_view(None, ""),
        owner="HC-01",
        moment=datetime.now(),
        wording=ui_settings.wording,
        token="tok",
        consent_line=linea,
    )
    assert linea in html
