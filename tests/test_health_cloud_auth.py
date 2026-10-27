"""Tests de la autorización de nube, solo en este equipo y país de proveedores (Plan 85b).

Cubre §B.2:
1. La puerta: Consultation con stt y local_stt espías (cloud=False -> local,
   cloud=True -> stt, cloud=False + local_stt=None -> CloudNotAuthorizedError;
   ídem diarizador).
2. La redacción: con cloud=False, finish y render_note usan local_provider y
   local_note_providers; espía de nube no recibe llamadas. Sin locales, respaldo literal.
3. El registro: cloud en línea ("true"/"false"), auditoría y evento; línea v1
   escrita a mano sin cloud se lee True y verify() sigue vacío; huella con
   cloud=True vs False.
4. El país: check_cloud_countries pasa con "Estados Unidos" y "estados unidos" (plegado);
   revienta con "Irlanda", nombrando país y modo; _build_consultation revienta con
   un provider_countries sin el proveedor en uso.
5. La aplicación: con ready.cloud inicio tiene id="cloud", sin clave o con local_only no;
   /empezar sin cloud -> cloud=False y LiveSession con transcriber=None; con cloud: true
   pero ready.cloud falso -> cloud=False; sin cloud y sin modelo local -> 400 cloud_needs_local.
6. Ajustes: /solo-local con {"on": true} llama a save_local_only(True); página con
   local_only() verdadero muestra "Nada sale de este equipo" y filas cloud off; con
   falso, "Sale a Estados Unidos".
7. keys: save_local_only(True) crea el archivo 0600, local_only() es True; False lo borra.
8. _app_build: con local_only() verdadero y clave puesta: cloud=False, live_stt=None,
   local_only=True.
9. La CLI: sin --consent-cloud registro queda cloud=False; con ella, True.
   --consent-cloud sin --consent => 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from cryptography.fernet import Fernet

from aida.arms.health.app import (
    AppReady,
    AppTools,
    _settings_page,
    _start_page,
    load_app_wording,
)
from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalWhisperSTT
from aida.arms.health.consent import (
    _GENESIS,
    ConsentLedger,
    check_cloud_countries,
    load_consent_texts,
)
from aida.arms.health.consultation import (
    CloudNotAuthorizedError,
    Consultation,
    PreparedConsultation,
)
from aida.arms.health.match import build_index
from aida.arms.health.note import ClinicalNote, NoteSettings, load_note_settings
from aida.arms.health.sections import SectionSpec
from aida.arms.health.wording import default_summary_wording
from aida.cognition.briefing import SpokenReply
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.engine.ports import CognitionProvider
from aida.keys import local_only, save_local_only
from aida.platform.audit import AuditLog
from aida.resources import user_dir
from aida.run import _app_build, _build_consultation, main
from aida.voice.diarize import LocalPyannoteDiarizer, SpeakerTurn, TranscriptSegment
from tests.conftest import _ORIGINAL_HANDLE_REQUEST
from tests.test_health_consent import _ACTOR, _crear_wav_minimo, _DummyOCR, _lanzar_servidor
from tests.test_run import _crear_perfil_cli


@pytest.fixture(autouse=True)
def _permitir_transporte_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


class _SpySTT(LocalWhisperSTT):
    def __init__(self, label: str = "stt") -> None:
        super().__init__(model=object())
        self.label = label
        self.transcribe_calls = 0
        self.transcribe_segments_calls = 0

    def transcribe(self, audio_path: Path) -> str:
        self.transcribe_calls += 1
        return f"transcrito por {self.label}"

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        self.transcribe_segments_calls += 1
        return (TranscriptSegment(0.0, 1.0, f"transcrito por {self.label}"),)


class _SpyDiarizer(LocalPyannoteDiarizer):
    def __init__(self) -> None:
        super().__init__(pipeline=object())

    def diarize(self, audio_path: Path) -> tuple[SpeakerTurn, ...]:
        return (SpeakerTurn(0.0, 1.0, "SPEAKER_00"),)


class _SpyProvider(CognitionProvider):
    def __init__(self, label: str = "llm") -> None:
        self.label = label
        self.think_calls = 0

    def think(self, briefing: str, schema: type) -> Any:
        self.think_calls += 1
        return SpokenReply(text=f"resumen por {self.label}")


class _SpyNoteProvider:
    def __init__(self, label: str = "note_llm") -> None:
        self.label = label
        self.think_calls = 0

    def think(self, briefing: str, schema: type) -> object | None:
        self.think_calls += 1
        return ClinicalNote(fields={"motivo_consulta": f"nota por {self.label}"})


# ---------------------------------------------------------------------------
# §B.2.1 — La puerta
# ---------------------------------------------------------------------------


def test_puerta_transcriber_cloud_false_usa_local_y_stt_nube_cero_llamadas(tmp_path: Path) -> None:
    stt_nube = _SpySTT("nube")
    stt_local = _SpySTT("local")
    c = Consultation(
        wording=default_summary_wording(),
        stt=stt_nube,
        local_stt=stt_local,
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=EventLog(),
    )
    owner = AgentId("p1")
    audio = tmp_path / "a.wav"
    audio.write_bytes(_crear_wav_minimo())

    consent = c.record_consent(cloud=False, owner=owner, mode="uploaded", actor=_ACTOR)
    res = c.transcribe(owner=owner, audio_path=audio, consent=consent)

    assert "transcrito por local" in res
    assert stt_local.transcribe_calls == 1
    assert stt_nube.transcribe_calls == 0

    prep = c.prepare(owner=owner, audio_path=audio, history_text="", consent=consent)
    assert prep.cloud is False


def test_puerta_transcriber_cloud_true_usa_stt_nube_y_local_cero_llamadas(tmp_path: Path) -> None:
    stt_nube = _SpySTT("nube")
    stt_local = _SpySTT("local")
    c = Consultation(
        wording=default_summary_wording(),
        stt=stt_nube,
        local_stt=stt_local,
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=EventLog(),
    )
    owner = AgentId("p2")
    audio = tmp_path / "a.wav"
    audio.write_bytes(_crear_wav_minimo())

    consent = c.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    res = c.transcribe(owner=owner, audio_path=audio, consent=consent)

    assert "transcrito por nube" in res
    assert stt_nube.transcribe_calls == 1
    assert stt_local.transcribe_calls == 0


def test_puerta_transcriber_cloud_false_sin_local_lanza_cloud_not_authorized(
    tmp_path: Path,
) -> None:
    stt_nube = _SpySTT("nube")
    c = Consultation(
        wording=default_summary_wording(),
        stt=stt_nube,
        local_stt=None,
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=EventLog(),
    )
    owner = AgentId("p3")
    audio = tmp_path / "a.wav"
    audio.write_bytes(_crear_wav_minimo())

    consent = c.record_consent(cloud=False, owner=owner, mode="uploaded", actor=_ACTOR)
    with pytest.raises(CloudNotAuthorizedError, match="sin nube autorizada"):
        c.transcribe(owner=owner, audio_path=audio, consent=consent)
    assert stt_nube.transcribe_calls == 0


def test_puerta_diarizador_con_cloud_false_usa_local_stt(tmp_path: Path) -> None:
    stt_nube = _SpySTT("nube")
    stt_local = _SpySTT("local")
    wording = default_summary_wording()
    c = Consultation(
        wording=wording,
        stt=stt_nube,
        local_stt=stt_local,
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=EventLog(),
        diarizer=_SpyDiarizer(),
    )
    owner = AgentId("p_diar")
    audio = tmp_path / "a.wav"
    audio.write_bytes(_crear_wav_minimo())

    consent_false = c.record_consent(cloud=False, owner=owner, mode="uploaded", actor=_ACTOR)
    c.transcribe(owner=owner, audio_path=audio, consent=consent_false)
    assert stt_local.transcribe_segments_calls == 1
    assert stt_nube.transcribe_segments_calls == 0

    consent_true = c.record_consent(cloud=True, owner=owner, mode="uploaded", actor=_ACTOR)
    c.transcribe(owner=owner, audio_path=audio, consent=consent_true)
    assert stt_nube.transcribe_segments_calls == 1


# ---------------------------------------------------------------------------
# §B.2.2 — La redacción
# ---------------------------------------------------------------------------


def test_redaccion_cloud_false_usa_local_provider_y_local_note_providers(tmp_path: Path) -> None:
    p_nube = _SpyProvider("nube")
    p_local = _SpyProvider("local")
    note_nube = _SpyNoteProvider("nube")
    note_local = _SpyNoteProvider("local")

    settings = NoteSettings(
        system_prompt="s",
        provider="nube",
        provider_model="m",
        field_heading="### {}",
        provenance_template="_{}_",
        fallback_notice="",
        edited_template="_{}_",
    )
    sec = SectionSpec("motivo_consulta", "Motivo", (), True, True)

    c = Consultation(
        wording=default_summary_wording(),
        stt=_SpySTT(),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=p_nube,
        local_provider=p_local,
        note_settings=settings,
        note_providers=(("nube", note_nube),),
        local_note_providers=(("local", note_local),),
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=EventLog(),
    )

    prep = PreparedConsultation(
        owner=AgentId("p_redact"),
        transcript="consulta",
        history="",
        matches=(),
        grouped_sections=((sec, ("consulta",)),),
        encounter=None,
        cloud=False,
    )

    res = c.finish(prep)
    assert "resumen por local" in res.summary_markdown
    assert p_local.think_calls == 1
    assert p_nube.think_calls == 0

    note = c.render_note(prep)
    assert "nota por local" in note.fields[0][1]
    assert note_local.think_calls == 1
    assert note_nube.think_calls == 0


def test_redaccion_cloud_false_sin_locales_respaldo_literal(tmp_path: Path) -> None:
    settings = NoteSettings(
        system_prompt="s",
        provider="nube",
        provider_model="m",
        field_heading="### {}",
        provenance_template="_{}_",
        fallback_notice="Aviso: sin redacción",
        edited_template="_{}_",
    )
    sec = SectionSpec("motivo_consulta", "Motivo", (), True, True)

    c = Consultation(
        wording=default_summary_wording(),
        stt=_SpySTT(),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=_SpyProvider("nube"),
        local_provider=None,
        note_settings=settings,
        note_providers=(("nube", _SpyNoteProvider("nube")),),
        local_note_providers=(),
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=EventLog(),
    )

    prep = PreparedConsultation(
        owner=AgentId("p_lit"),
        transcript="dolor de cabeza",
        history="",
        matches=(),
        grouped_sections=((sec, ("dolor de cabeza",)),),
        encounter=None,
        cloud=False,
    )

    note = c.render_note(prep)
    assert "dolor de cabeza" in note.fields[0][1]
    assert note.redacted is False


# ---------------------------------------------------------------------------
# §B.2.3 — El registro
# ---------------------------------------------------------------------------


def test_registro_cloud_en_linea_auditoria_y_evento(tmp_path: Path) -> None:
    archivo = tmp_path / "consents.jsonl"
    ledger = ConsentLedger(archivo)
    log = EventLog()
    audit = AuditLog(tmp_path / "audit.jsonl", key=Fernet.generate_key())

    c = Consultation(
        wording=default_summary_wording(),
        stt=_SpySTT(),
        ocr=_DummyOCR(),
        catalog_index=build_index(()),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=log,
        consents=ledger,
        audit=audit,
    )

    rec_false = c.record_consent(cloud=False, owner=AgentId("o1"), mode="uploaded", actor=_ACTOR)
    assert rec_false.cloud is False

    lineas = archivo.read_text(encoding="utf-8").splitlines()
    assert '"cloud": "false"' in lineas[0]

    evt_false = next(e for e in log if e.kind == EventKind.CONSENT_RECORDED)
    assert evt_false.data["cloud"] == "false"

    audit_records = list(audit.lines())
    audit_data = json.loads(str(audit_records[0]["data"]))
    assert audit_data["cloud"] == "false"


def test_registro_linea_v1_escrita_a_mano_lee_true_y_verify_vacio(tmp_path: Path) -> None:
    archivo = tmp_path / "consents_v1.jsonl"
    linea_v1 = {
        "at": "2026-10-06T10:00:00-05:00",
        "mode": "uploaded",
        "owner": "doc_v1",
        "prev": _GENESIS,
        "text_sha256": hashlib.sha256(b"Texto v1").hexdigest(),
        "version": "v1",
        "workstation": "ws",
    }
    digest = hashlib.sha256(
        json.dumps(linea_v1, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    linea_v1["digest"] = digest
    archivo.write_text(json.dumps(linea_v1, ensure_ascii=False) + "\n", encoding="utf-8")

    ledger = ConsentLedger(archivo)
    records = ledger.records()
    assert len(records) == 1
    assert records[0].cloud is True
    assert ledger.verify() == ()


def test_registro_huella_con_cloud_true_vs_false() -> None:
    textos = load_consent_texts()
    modo = "uploaded"
    texto_false = textos.read_text(modo, cloud=False)
    texto_true = textos.read_text(modo, cloud=True)

    assert texto_false == textos.modes[modo].text
    assert texto_true == f"{textos.modes[modo].text} {textos.modes[modo].cloud.text}"
    assert (
        hashlib.sha256(texto_false.encode()).hexdigest()
        != hashlib.sha256(texto_true.encode()).hexdigest()
    )


# ---------------------------------------------------------------------------
# §B.2.4 — El país
# ---------------------------------------------------------------------------


def test_check_cloud_countries_pasa_con_plegado_y_revienta_con_irlanda() -> None:
    textos = load_consent_texts()
    check_cloud_countries(textos, ["Estados Unidos"])
    check_cloud_countries(textos, ["estados unidos"])

    with pytest.raises(ValueError, match=r"no nombra el país.*Irlanda"):
        check_cloud_countries(textos, ["Irlanda"])


def test_build_consultation_revienta_con_proveedor_sin_pais_declarado(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = Fernet.generate_key().decode()
    args = argparse.Namespace(
        llm="desconocido",
        review_metrics=None,
    )
    ajustes_sin_pais = NoteSettings(
        system_prompt="s",
        provider="desconocido",
        provider_model="m",
        field_heading="### {}",
        provenance_template="_{}_",
        fallback_notice="",
        edited_template="_{}_",
        provider_countries={},
    )
    monkeypatch.setattr("aida.arms.health.note.load_note_settings", lambda: ajustes_sin_pais)
    with pytest.raises(ValueError, match=r"no declara país en note.yaml"):
        _build_consultation(
            args,
            stt=_SpySTT(),
            ocr=_DummyOCR(),
            key=key,
        )


# ---------------------------------------------------------------------------
# §B.2.5 — La aplicación
# ---------------------------------------------------------------------------


def test_app_inicio_con_ready_cloud_muestra_casilla_cloud() -> None:
    wording = load_app_wording()
    textos = load_consent_texts()
    ready_con = AppReady(
        consultation=cast(Any, object()),
        reader=cast(Any, object()),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )
    html_con = _start_page(
        w=wording,
        brand="AIDA",
        token="tok",
        ready=ready_con,
        moment=datetime.now(),
        consent=textos,
    )
    assert 'id="cloud"' in html_con

    ready_sin = AppReady(
        consultation=cast(Any, object()),
        reader=cast(Any, object()),
        log=EventLog(),
        cloud=False,
        writer="modelo-prueba",
    )
    html_sin = _start_page(
        w=wording,
        brand="AIDA",
        token="tok",
        ready=ready_sin,
        moment=datetime.now(),
        consent=textos,
    )
    assert 'id="cloud"' not in html_sin


def test_app_empezar_sin_cloud_guarda_cloud_false_y_transcriber_none(tmp_path: Path) -> None:
    pytest.importorskip("av", reason="requiere capture")
    ledger = ConsentLedger(tmp_path / "consents.jsonl")
    stt_espia = _SpySTT()
    ready = AppReady(
        consultation=Consultation(
            wording=default_summary_wording(),
            stt=stt_espia,
            local_stt=stt_espia,
            ocr=_DummyOCR(),
            catalog_index=build_index(()),
            provider=None,
            cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
            log=EventLog(),
            consents=ledger,
            note_settings=load_note_settings(),
        ),
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="test",
        live_stt=cast(Any, object()),
        local_model=True,
    )
    url, _, _, hilo = _lanzar_servidor(lambda: ready)
    try:
        httpx.post(f"{url}/documento", content=b"hist", headers={"X-Name": "h.txt"}, timeout=5.0)
        r = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-001", "consent": True, "recorded": True, "cloud": False},
            timeout=5.0,
        )
        assert r.status_code == 200
        assert len(ledger.records()) == 1
        assert ledger.records()[0].cloud is False
        st = httpx.get(f"{url}/vivo", timeout=5.0).json()
        assert st.get("offline") is True
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_app_empezar_cloud_true_pero_ready_cloud_falso_no_abre_nube(tmp_path: Path) -> None:
    pytest.importorskip("av", reason="requiere capture")
    ledger = ConsentLedger(tmp_path / "consents.jsonl")
    stt_espia = _SpySTT()
    ready = AppReady(
        consultation=Consultation(
            wording=default_summary_wording(),
            stt=stt_espia,
            local_stt=stt_espia,
            ocr=_DummyOCR(),
            catalog_index=build_index(()),
            provider=None,
            cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
            log=EventLog(),
            consents=ledger,
            note_settings=load_note_settings(),
        ),
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=False,
        writer="test",
        local_model=True,
    )
    url, _, _, hilo = _lanzar_servidor(lambda: ready)
    try:
        httpx.post(f"{url}/documento", content=b"hist", headers={"X-Name": "h.txt"}, timeout=5.0)
        r = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-001b", "consent": True, "recorded": True, "cloud": True},
            timeout=5.0,
        )
        assert r.status_code == 200
        assert len(ledger.records()) == 1
        assert ledger.records()[0].cloud is False
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_app_empezar_sin_cloud_y_sin_modelo_local_responde_400(tmp_path: Path) -> None:
    ready = AppReady(
        consultation=cast(Any, object()),
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
        local_model=False,
    )
    wording = load_app_wording()
    url, _, _, hilo = _lanzar_servidor(lambda: ready)
    try:
        httpx.post(f"{url}/documento", content=b"hist", headers={"X-Name": "h.txt"}, timeout=5.0)
        httpx.post(
            f"{url}/audio",
            content=_crear_wav_minimo(),
            headers={"X-Name": "a.wav"},
            timeout=5.0,
        )
        r = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-002", "consent": True, "recorded": False, "cloud": False},
            timeout=5.0,
        )
        assert r.status_code == 400
        assert r.json()["error"] == wording["cloud_needs_local"]
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# §B.2.6 & §B.2.7 — Ajustes y keys
# ---------------------------------------------------------------------------


def test_ajustes_solo_local_endpoint_y_render_pagina(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    guardados: list[bool] = []
    tools = AppTools(
        cloud_key_source=lambda: "file",
        check_cloud_key=lambda k: (True, "200"),
        save_cloud_key=lambda k: None,
        model_name="small",
        model_total_mb=464,
        model_available=lambda: True,
        model_bytes=lambda: 0,
        download_model=lambda: None,
        reader_version=lambda: "5.3.0",
        cache_key_since=lambda: 1700000000.0,
        local_only=lambda: True,
        save_local_only=lambda v: guardados.append(v),
        destinations=(
            ("transcription", "OpenAI · Estados Unidos"),
            ("note", "openai · Estados Unidos"),
        ),
    )
    ready = AppReady(
        consultation=cast(Any, object()),
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=False,
        writer="modelo-prueba",
    )
    url, _, _, hilo = _lanzar_servidor(lambda: ready, tools=tools)
    try:
        r = httpx.post(f"{url}/solo-local", json={"on": True}, timeout=5.0)
        assert r.status_code == 200
        assert guardados == [True]

        # Comprobar render de la página con local_only=True
        wording = load_app_wording()
        html_local = _settings_page(
            w=wording,
            brand="AIDA",
            token="tok",
            tools=tools,
            moment=datetime.now(),
            download="idle",
        )
        assert wording["where_local"] in html_local
        assert 'class="cloud off"' in html_local

        # Comprobar render de la página con local_only=False
        tools_salida = AppTools(
            cloud_key_source=lambda: "file",
            check_cloud_key=lambda k: (True, "200"),
            save_cloud_key=lambda k: None,
            model_name="small",
            model_total_mb=464,
            model_available=lambda: True,
            model_bytes=lambda: 0,
            download_model=lambda: None,
            reader_version=lambda: "5.3.0",
            cache_key_since=lambda: 1700000000.0,
            local_only=lambda: False,
            save_local_only=lambda v: None,
            destinations=(
                ("transcription", "OpenAI · Estados Unidos"),
                ("note", "openai · Estados Unidos"),
            ),
        )
        html_salida = _settings_page(
            w=wording,
            brand="AIDA",
            token="tok",
            tools=tools_salida,
            moment=datetime.now(),
            download="idle",
        )
        assert wording["where_out"].format("Estados Unidos") in html_salida
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_keys_save_local_only_crea_0600_y_borra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    falso_config = tmp_path / "config"
    falso_config.mkdir(parents=True)
    monkeypatch.setattr("aida.keys.user_dir", lambda kind: falso_config)

    assert local_only() is False
    save_local_only(True)
    assert local_only() is True
    archivo = falso_config / "solo-local"
    assert archivo.is_file()
    assert sys.platform == "win32" or stat.S_IMODE(archivo.stat().st_mode) == 0o600

    save_local_only(False)
    assert local_only() is False
    assert not archivo.exists()


# ---------------------------------------------------------------------------
# §B.2.8 — _app_build
# ---------------------------------------------------------------------------


def test_app_build_con_local_only_y_clave_apaga_nube(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("faster_whisper", reason="requiere capture")
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("AIDA_STT_API_KEY", "sk-test")
    monkeypatch.setattr("aida.keys.local_only", lambda: True)
    monkeypatch.setattr("aida.run.local_only", lambda: True)
    monkeypatch.setattr("aida.arms.health.capture.local_model_available", lambda: True)
    # Ni el modelo ni Tesseract de este equipo: el test mide la decisión, no la instalación.
    monkeypatch.setattr(
        "aida.arms.health.capture.capture_diagnosis", lambda **_: (("ok", True),)
    )
    monkeypatch.setattr("faster_whisper.WhisperModel", lambda *a, **k: object())

    args = argparse.Namespace(llm=None)
    res = _app_build(args)
    assert isinstance(res, AppReady)
    assert res.cloud is False
    assert res.live_stt is None
    assert res.local_only is True


# ---------------------------------------------------------------------------
# §B.2.9 — La CLI
# ---------------------------------------------------------------------------


def test_cli_consent_cloud_registra_true_o_false(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("faster_whisper", reason="requiere capture")
    falso_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: falso_home)
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(
        "aida.arms.health.capture.build_local_capture",
        lambda: (_SpySTT(), _DummyOCR()),
    )

    perfil = _crear_perfil_cli(monkeypatch)

    audio = tmp_path / "a.wav"
    audio.write_bytes(_crear_wav_minimo())
    hist = tmp_path / "h.png"
    hist.write_bytes(b"dummy")

    # 1. Sin --consent-cloud
    codigo_sin = main(
        [
            "--consultation",
            "--audio",
            str(audio),
            "--history",
            str(hist),
            "--consent",
            "--as",
            perfil,
            "--llm",
            "ninguno",
        ]
    )
    assert codigo_sin == 0
    consents_path = user_dir("data") / "autorizaciones.jsonl"
    ledger = ConsentLedger(consents_path)
    records = ledger.records()
    assert len(records) >= 1
    assert records[-1].cloud is False

    # 2. Con --consent-cloud
    codigo_con = main(
        [
            "--consultation",
            "--audio",
            str(audio),
            "--history",
            str(hist),
            "--consent",
            "--consent-cloud",
            "--as",
            perfil,
            "--llm",
            "ninguno",
        ]
    )
    assert codigo_con == 0
    records2 = ConsentLedger(consents_path).records()
    assert len(records2) >= 2
    assert records2[-1].cloud is True


def test_cli_consent_cloud_sin_consent_termina_con_uno(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("faster_whisper", reason="requiere capture")
    monkeypatch.setattr("faster_whisper.WhisperModel", lambda *a, **k: object())
    audio = tmp_path / "a.wav"
    audio.write_bytes(_crear_wav_minimo())
    hist = tmp_path / "h.txt"
    hist.write_text("historia")
    codigo = main(
        [
            "--consultation",
            "--audio",
            str(audio),
            "--history",
            str(hist),
            "--consent-cloud",
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 1
    assert "falta --consent" in captured.err
