"""Pruebas de identidad del profesional y auditoría (Plan 83).

Cubre §B.2, §B.3, §B.4 y §B.5 de .claude/workflow/83-plan-identidad-auditoria.md:
§B.2 La cadena común (ledger.py):
  6. test_chained_log_append_encadena_y_rechaza_campos_protegidos
  7. test_chained_log_dos_instancias_alternadas_mantienen_cadena_integra
  8. test_chained_log_verify_detecta_borrada_cambiada_e_ilegible
  9. test_chained_log_compatibilidad_con_plan_82
§B.3 Identidad (identity.py):
  10. test_profile_store_create_guarda_0600_sin_pin_en_claro_y_valida_campos
  11. test_profile_store_verify_pin_fallos_y_bloqueo_con_reloj
  12. test_profile_store_professionals_ordenados_por_nombre_e_id
§B.4 Auditoría y la puerta (audit.py, consultation.py):
  13. test_consultation_review_sin_actor_lanza_identity_error_y_no_audita
  14. test_consultation_review_con_actor_audita_review_saved_y_show_recupera_textos
  15. test_audit_show_con_blob_alterado_lanza_value_error
  16. test_audit_record_sin_clave_o_sin_blobs_escribe_blob_vacio
  17. test_record_consent_con_actor_incluye_label_en_workstation_y_audita
  18. test_audit_record_con_clase_fuera_de_kinds_lanza_value_error
§B.5 La aplicación y la CLI (app.py, run.py):
  19. test_app_sin_identificarse_da_401_y_muestra_primer_perfil_o_quien
  20. test_app_post_perfil_fuera_de_first_profile_y_ajustes_da_409_y_desde_ajustes_audita
  21. test_app_entrar_con_pin_malo_da_400_con_intentos_restantes
  22. test_app_bloqueo_por_inactividad_en_start_y_no_en_review_y_desbloqueo
  23. test_app_cambiar_en_review_da_409_y_en_start_pasa_a_who_y_audita_cierre
  24. test_app_historia_vista_dos_veces_audita_una_sola_vez
  25. test_app_guardar_revision_audita_review_saved_con_profesional_de_sesion
  26. test_cli_as_y_audit_comandos
"""

from __future__ import annotations

import hashlib
import io
import json
import stat
import threading
import time
import wave
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from cryptography.fernet import Fernet

from aida.arms.health.app import (
    AppProblem,
    AppReady,
    AppTools,
    _Session,
    load_app_wording,
    serve,
)
from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT
from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.consent import ConsentLedger, load_consent_texts
from aida.arms.health.consultation import Consultation
from aida.arms.health.extract import build_health_extractor
from aida.arms.health.match import build_index
from aida.arms.health.note import load_note_settings
from aida.arms.health.review import Answer, Verdict
from aida.arms.health.review_ui import HistoryView, load_ui_settings
from aida.arms.health.segment import build_health_segmenter
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.platform.audit import KINDS, AuditLog
from aida.platform.identity import (
    IdentityError,
    Professional,
    ProfileStore,
    valid_pin,
)
from aida.platform.ledger import GENESIS, ChainedLog, chain_digest
from aida.platform.license import License, LicenseState
from aida.resources import user_dir
from aida.run import _audit, main
from aida.voice.diarize import TranscriptSegment
from tests.conftest import _ORIGINAL_HANDLE_REQUEST


@pytest.fixture(autouse=True)
def _permitir_servidor_app_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restaura el transporte real de httpx para comunicar con el servidor local."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


# ---------------------------------------------------------------------------
# Helpers para dobles y servidor de pruebas
# ---------------------------------------------------------------------------


class _DummySTT(LocalWhisperSTT):
    def __init__(self, segments: tuple[TranscriptSegment, ...] | None = None) -> None:
        super().__init__(model=object())
        self._segments = (
            segments
            if segments is not None
            else (TranscriptSegment(0.0, 2.0, "consulta hoy por dolor en rodilla izquierda"),)
        )

    def transcribe(self, path: Path) -> str:
        return " ".join(s.text for s in self._segments)

    def transcribe_segments(self, path: Path) -> tuple[TranscriptSegment, ...]:
        return self._segments


class _DummyOCR(LocalTesseractOCR):
    def __init__(self, text: str = "Historia clínica previa: dolor articular.") -> None:
        super().__init__(pytesseract_module=object())
        self._text = text

    def read(self, path: Path) -> str:
        return self._text


def _crear_wav_minimo() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def _dummy_vad(audio: np.ndarray) -> np.ndarray:
    return np.zeros((0, 2), dtype=np.int64)


def _crear_consultation(
    tmp_path: Path,
    *,
    audit: AuditLog | None = None,
    log: EventLog | None = None,
) -> tuple[Consultation, EventLog, AuditLog]:
    event_log = log if log is not None else EventLog()
    key = Fernet.generate_key()
    audit_log = (
        audit
        if audit is not None
        else AuditLog(
            tmp_path / "auditoria.jsonl",
            blobs=tmp_path / "blobs",
            key=key,
        )
    )
    consultation = Consultation(
        wording=default_summary_wording(),
        stt=_DummySTT(),
        ocr=_DummyOCR(),
        catalog_index=build_index((DiagnosticCode("M255", "Dolor en articulación", 4, "cat"),)),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=key),
        log=event_log,
        segmenter=build_health_segmenter(),
        extractor=build_health_extractor(),
        note_settings=load_note_settings(),
        consents=ConsentLedger(tmp_path / "autorizaciones.jsonl"),
        audit=audit_log,
        consent_texts=load_consent_texts(),
    )
    return consultation, event_log, audit_log


def _lanzar_servidor_identity(
    build_fn: Callable[[], AppReady | AppProblem],
    *,
    profiles: ProfileStore,
    audit: AuditLog,
    clock: Callable[[], float] = time.monotonic,
    tools: AppTools | None = None,
    identificar: bool = False,
    pin: str = "1234",
) -> tuple[str, str, int, threading.Thread, dict[str, Any]]:
    settings = load_ui_settings()
    wording = load_app_wording()
    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    serve_kwargs = {
        "settings": settings,
        "wording": wording,
        "open_browser": _abrir,
        "vad_factory": lambda: _dummy_vad,
        "clock": clock,
        "profiles": profiles,
        "audit": audit,
        "tools": tools,
    }
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
    assert listo.wait(timeout=5.0), "el servidor no abrió la URL en el tiempo límite"
    url = info["url"]
    partes = url.rsplit("/", 1)
    token = partes[1]
    puerto = int(partes[0].split(":")[-1])
    if identificar:
        r = httpx.post(
            f"{url}/perfil",
            json={
                "name": "Dra. Laura Gómez",
                "registry": "52.341.889",
                "pin": pin,
                "pin2": pin,
            },
            timeout=5.0,
        )
        assert r.status_code == 200, f"no se pudo autoidentificar: {r.text}"
    return url, token, puerto, hilo, info


# ---------------------------------------------------------------------------
# §B.2 — La cadena común (ledger.py)
# ---------------------------------------------------------------------------


def test_chained_log_append_encadena_y_rechaza_campos_protegidos(tmp_path: Path) -> None:
    """Caso 6 (§B.2): ChainedLog.append encadena (primera prev = GENESIS) y
    rechaza campos prev o digest del llamador."""
    archivo = tmp_path / "cadena.jsonl"
    log = ChainedLog(archivo)

    # 1. Primera línea: prev es GENESIS
    l1 = log.append({"accion": "inicio", "usuario": "ana"})
    assert l1["prev"] == GENESIS
    assert l1["digest"] == chain_digest({"accion": "inicio", "usuario": "ana", "prev": GENESIS})

    # 2. Segunda línea: prev es la huella de la anterior
    l2 = log.append({"accion": "paso2", "usuario": "ana"})
    assert l2["prev"] == l1["digest"]
    assert l2["digest"] == chain_digest({"accion": "paso2", "usuario": "ana", "prev": l1["digest"]})

    # 3. Rechaza campos protegidos 'prev' y 'digest'
    with pytest.raises(ValueError, match="'prev' y 'digest' los pone la cadena"):
        log.append({"prev": "trampa"})

    with pytest.raises(ValueError, match="'prev' y 'digest' los pone la cadena"):
        log.append({"digest": "trampa"})


def test_chained_log_dos_instancias_alternadas_mantienen_cadena_integra(tmp_path: Path) -> None:
    """Caso 7 (§B.2, M2): Dos instancias sobre el mismo archivo que añaden alternándose
    dejan una cadena íntegra (verify() == ()). Comprueba _last_digest()."""
    archivo = tmp_path / "concurrente.jsonl"
    instancia1 = ChainedLog(archivo)
    instancia2 = ChainedLog(archivo)

    l1 = instancia1.append({"origen": "instancia1_a"})
    l2 = instancia2.append({"origen": "instancia2_a"})
    l3 = instancia1.append({"origen": "instancia1_b"})
    l4 = instancia2.append({"origen": "instancia2_b"})

    assert l2["prev"] == l1["digest"]
    assert l3["prev"] == l2["digest"]
    assert l4["prev"] == l3["digest"]

    # Verificación en disco con una instancia limpia
    verificador = ChainedLog(archivo)
    assert verificador.verify() == ()
    assert len(verificador.lines()) == 4


def test_chained_log_verify_detecta_borrada_cambiada_e_ilegible(tmp_path: Path) -> None:
    """Caso 8 (§B.2): verify detecta línea borrada, cambiada e ilegible con
    sus mensajes literales exactos."""
    archivo = tmp_path / "corrupcion.jsonl"
    log = ChainedLog(archivo)
    for i in range(1, 4):
        log.append({"idx": str(i)})
    assert log.verify() == ()

    lineas_originales = archivo.read_text(encoding="utf-8").splitlines()

    # 1. Línea borrada (borramos la 2)
    archivo.write_text(
        "\n".join([lineas_originales[0], lineas_originales[2]]) + "\n",
        encoding="utf-8",
    )
    problemas_borrada = log.verify()
    assert any("línea 2: no sigue a la anterior" in p for p in problemas_borrada)

    # 2. Línea cambiada (modificamos el contenido de la 2 manteniendo digest antiguo)
    datos_l2 = json.loads(lineas_originales[1])
    datos_l2["idx"] = "modificado"
    lineas_cambiadas = [
        lineas_originales[0],
        json.dumps(datos_l2, ensure_ascii=False, sort_keys=True),
        lineas_originales[2],
    ]
    archivo.write_text("\n".join(lineas_cambiadas) + "\n", encoding="utf-8")
    problemas_cambiada = log.verify()
    assert any("línea 2: su contenido no corresponde a su huella" in p for p in problemas_cambiada)

    # 3. Línea ilegible (JSON roto en la 2)
    lineas_rotas = [lineas_originales[0], "{ json roto no valido", lineas_originales[2]]
    archivo.write_text("\n".join(lineas_rotas) + "\n", encoding="utf-8")
    problemas_ilegible = log.verify()
    assert any("línea 2: no se puede leer" in p for p in problemas_ilegible)


def test_chained_log_compatibilidad_con_plan_82(tmp_path: Path) -> None:
    """Caso 9 (§B.2): Compatibilidad con el formato de autorizaciones del Plan 82.
    Un archivo escrito con el formato de la 82 verifica íntegro y contains() reconoce."""
    archivo = tmp_path / "autorizaciones_plan82.jsonl"

    c1 = {
        "at": "2026-10-06T10:00:00-05:00",
        "mode": "uploaded",
        "owner": "paciente_82_01",
        "prev": GENESIS,
        "text_sha256": hashlib.sha256(b"Autorizo el procesamiento...").hexdigest(),
        "version": "v1",
        "workstation": "operador@equipo",
    }
    h1 = chain_digest(c1)
    l1 = {**c1, "digest": h1}

    c2 = {
        "at": "2026-10-06T10:15:00-05:00",
        "mode": "recorded",
        "owner": "paciente_82_02",
        "prev": h1,
        "text_sha256": hashlib.sha256("Autorizo la grabación...".encode()).hexdigest(),
        "version": "v1",
        "workstation": "operador@equipo",
    }
    h2 = chain_digest(c2)
    l2 = {**c2, "digest": h2}

    contenido = (
        json.dumps(l1, ensure_ascii=False, sort_keys=True)
        + "\n"
        + json.dumps(l2, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    archivo.write_text(contenido, encoding="utf-8")

    chained = ChainedLog(archivo)
    assert chained.verify() == ()
    assert chained.contains(h1)
    assert chained.contains(h2)

    consent_ledger = ConsentLedger(archivo)
    assert consent_ledger.verify() == ()
    recs = consent_ledger.records()
    assert len(recs) == 2
    assert consent_ledger.contains(recs[0])
    assert consent_ledger.contains(recs[1])


# ---------------------------------------------------------------------------
# §B.3 — Identidad (identity.py)
# ---------------------------------------------------------------------------


def test_profile_store_create_guarda_0600_sin_pin_en_claro_y_valida_campos(tmp_path: Path) -> None:
    """Caso 10 (§B.3, M5): create guarda 0o600, SIN el PIN en el archivo, con sal
    distinta en dos perfiles con el mismo PIN; valida pin_format, missing y duplicate."""
    archivo = tmp_path / "profiles.json"
    store = ProfileStore(archivo)

    p1 = store.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")
    p2 = store.create(name="Dr. Carlos Pérez", registry="41.234.567", pin="1234")

    # Permisos 0o600
    assert archivo.is_file()
    assert stat.S_IMODE(archivo.stat().st_mode) == 0o600

    # Sin PIN en claro en el archivo
    contenido = archivo.read_text(encoding="utf-8")
    assert "1234" not in contenido

    # Sales y huellas distintas para el mismo PIN
    datos = json.loads(contenido)
    assert datos[p1.id]["salt"] != datos[p2.id]["salt"]
    assert datos[p1.id]["pin"] != datos[p2.id]["pin"]

    # Validación de formato de PIN (4 a 8 dígitos)
    assert valid_pin("1234") is True
    assert valid_pin("12345678") is True
    assert valid_pin("123") is False
    assert valid_pin("123456789") is False
    assert valid_pin("abcd") is False

    with pytest.raises(IdentityError) as exc_pin_corto:
        store.create(name="Dra. Gómez", registry="99.999.999", pin="123")
    assert exc_pin_corto.value.code == "pin_format"

    with pytest.raises(IdentityError) as exc_pin_letras:
        store.create(name="Dra. Gómez", registry="99.999.999", pin="12ab")
    assert exc_pin_letras.value.code == "pin_format"

    # Validación de campos obligatorios (nombre o registro vacío/espacios)
    with pytest.raises(IdentityError) as exc_nom_vacio:
        store.create(name="   ", registry="99.999.999", pin="1234")
    assert exc_nom_vacio.value.code == "missing"

    with pytest.raises(IdentityError) as exc_reg_vacio:
        store.create(name="Dra. Gómez", registry="", pin="1234")
    assert exc_reg_vacio.value.code == "missing"

    # Validación de registro duplicado
    with pytest.raises(IdentityError) as exc_dup:
        store.create(name="Dra. Otra", registry="52.341.889", pin="5678")
    assert exc_dup.value.code == "duplicate"


def test_profile_store_verify_pin_fallos_y_bloqueo_con_reloj(tmp_path: Path) -> None:
    """Caso 11 (§B.3, M3, M4): verify con PIN bueno devuelve profesional; malo: bad_pin
    y attempts_left baja; al quinto: wait y wait_seconds > 0; tras 30 s permite de nuevo."""
    tiempo = 1000.0
    store = ProfileStore(tmp_path / "profiles.json", clock=lambda: tiempo)
    p = store.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")

    # PIN bueno
    prof = store.verify(p.id, "1234")
    assert prof.id == p.id
    assert prof.name == "Dra. Laura Gómez"
    assert prof.registry == "52.341.889"
    assert prof.label() == "Dra. Laura Gómez (52.341.889)"

    # Fallos 1 a 4: decrementan attempts_left
    for intento in range(1, 5):
        assert store.attempts_left(p.id) == 6 - intento
        with pytest.raises(IdentityError) as exc:
            store.verify(p.id, "9999")
        assert exc.value.code == "bad_pin"
        assert store.attempts_left(p.id) == 5 - intento

    # Fallo 5: salta bloqueo 'wait' y wait_seconds > 0
    with pytest.raises(IdentityError) as exc_quinto:
        store.verify(p.id, "9999")
    assert exc_quinto.value.code == "wait"
    assert store.wait_seconds(p.id) > 0.0

    # Intento durante el bloqueo (incluso con PIN bueno) es rechazado con 'wait'
    with pytest.raises(IdentityError) as exc_bloqueado:
        store.verify(p.id, "1234")
    assert exc_bloqueado.value.code == "wait"

    # Tras 30 segundos, vuelve a permitir
    tiempo += 30.1
    assert store.wait_seconds(p.id) == 0.0
    prof_ok = store.verify(p.id, "1234")
    assert prof_ok.id == p.id
    assert store.attempts_left(p.id) == 5


def test_profile_store_professionals_ordenados_por_nombre_e_id(tmp_path: Path) -> None:
    """Caso 12 (§B.3): professionals() devuelve los profesionales ordenados por
    nombre (casefold) e id, independientemente del orden de inserción en el archivo."""
    archivo = tmp_path / "profiles.json"
    store = ProfileStore(archivo)

    store.create(name="Zoe Méndez", registry="03", pin="1234")
    store.create(name="carlos Gómez", registry="01", pin="1234")
    store.create(name="Ana Ruiz", registry="02", pin="1234")

    nombres = [p.name for p in store.professionals()]
    assert nombres == ["Ana Ruiz", "carlos Gómez", "Zoe Méndez"]


# ---------------------------------------------------------------------------
# §B.4 — Auditoría y la puerta (audit.py, consultation.py)
# ---------------------------------------------------------------------------


def test_consultation_review_sin_actor_lanza_identity_error_y_no_audita(tmp_path: Path) -> None:
    """Caso 13 (§B.4, M1): Consultation.review sin actor lanza IdentityError ('no_actor')
    y NO emite ITEM_REVIEWED, no guarda en caché ni escribe en auditoría."""
    consultation, event_log, audit_log = _crear_consultation(tmp_path)
    owner = AgentId("paciente_puerta")
    actor = Professional(id="c1", name="Dr. Consent", registry="111")
    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Sin antecedentes.",
        consent=consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=actor),
    )

    with pytest.raises(IdentityError) as exc:
        consultation.review(
            prepared,
            reviewer=lambda item: Answer(Verdict.CONFIRMED),
            actor=None,
        )
    assert exc.value.code == "no_actor"

    # 0 eventos ITEM_REVIEWED
    assert not any(e.kind == EventKind.ITEM_REVIEWED for e in event_log)

    # Nada en caché
    cache_file = tmp_path / "cache" / f"{owner}.cache"
    assert not cache_file.exists()

    # Nada de review_saved en auditoría
    assert not any(linea["kind"] == "review_saved" for linea in audit_log.lines())


def test_consultation_review_con_actor_audita_review_saved_y_show_recupera_textos(
    tmp_path: Path,
) -> None:
    """Caso 14 (§B.4): Con actor, review registra review_saved con su nombre, códigos
    confirmados/quitados, campos editados y show(blob) devuelve draft y final con ediciones."""
    consultation, _, audit_log = _crear_consultation(tmp_path)
    owner = AgentId("paciente_revisado")
    actor = Professional(id="p_01", name="Dra. Laura Gómez", registry="52.341.889")

    prepared = consultation.prepare(
        owner=owner,
        audio_path=Path("/tmp/audio.wav"),
        history_text="Dolor de rodilla.",
        consent=consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=actor),
    )

    note = consultation.render_note(prepared)
    edits = {spec.section_id: texto for spec, texto in note.fields}
    primer_id = note.fields[0][0].section_id
    edits[primer_id] = "Texto final editado por la doctora."

    consultation.review(
        prepared,
        reviewer=lambda item: Answer(Verdict.CONFIRMED),
        actor=actor,
        note=note,
        note_edits=edits,
    )

    lineas_guardadas = [linea for linea in audit_log.lines() if linea["kind"] == "review_saved"]
    assert len(lineas_guardadas) == 1
    linea = lineas_guardadas[0]

    assert linea["actor"] == actor.id
    assert linea["actor_name"] == actor.label()
    data = json.loads(linea["data"])
    assert "codes_confirmed" in data
    assert "codes_removed" in data
    assert primer_id in data.get("fields_edited", [])

    assert linea["blob"] != ""
    textos = audit_log.show(linea["blob"])
    assert "draft" in textos
    assert "final" in textos
    final_dict = textos["final"]
    assert isinstance(final_dict, dict)
    assert final_dict[primer_id] == "Texto final editado por la doctora."


def test_audit_show_con_blob_alterado_lanza_value_error(tmp_path: Path) -> None:
    """Caso 15 (§B.4, M13): show() con un archivo blob alterado en disco lanza
    ValueError por discrepancia entre el contenido descifrado y su huella."""
    key = Fernet.generate_key()
    blobs_dir = tmp_path / "blobs"
    audit = AuditLog(tmp_path / "audit.jsonl", blobs=blobs_dir, key=key)
    actor = Professional(id="p1", name="Dr. Pérez", registry="123")

    linea = audit.record(
        "review_saved", actor=actor, texts={"draft": "original", "final": "original"}
    )
    blob = linea["blob"]
    assert blob != ""

    # Verificación íntegra inicial
    assert audit.show(blob)["draft"] == "original"

    # Alteramos el blob en disco cifrando contenido diferente con la misma clave
    texto_falso = json.dumps({"draft": "alterado"}, ensure_ascii=False, sort_keys=True).encode(
        "utf-8"
    )
    (blobs_dir / f"{blob}.bin").write_bytes(Fernet(key).encrypt(texto_falso))

    with pytest.raises(ValueError, match="no corresponde a su huella"):
        audit.show(blob)


def test_audit_record_sin_clave_o_sin_blobs_escribe_blob_vacio(tmp_path: Path) -> None:
    """Caso 16 (§B.4): Sin clave o sin carpeta de textos, record escribe la línea con blob vacío."""
    actor = Professional(id="p1", name="Dr. Uno", registry="1")
    # Sin clave ni carpeta
    audit1 = AuditLog(tmp_path / "audit1.jsonl", blobs=None, key=None)
    l1 = audit1.record("review_saved", actor=actor, texts={"draft": "algo"})
    assert l1["blob"] == ""

    # Sin clave pero con carpeta
    audit2 = AuditLog(tmp_path / "audit2.jsonl", blobs=tmp_path / "b", key=None)
    l2 = audit2.record("review_saved", actor=actor, texts={"draft": "algo"})
    assert l2["blob"] == ""


def test_record_consent_con_actor_incluye_label_en_workstation_y_audita(tmp_path: Path) -> None:
    """Caso 17 (§B.4, M12): record_consent con actor incluye actor.label() al inicio
    de workstation y escribe consent_recorded en auditoría con el digest."""
    consultation, _, audit_log = _crear_consultation(tmp_path)
    actor = Professional(id="p_99", name="Dra. Laura Gómez", registry="52.341.889")
    owner = AgentId("paciente_consentimiento")

    rec = consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=actor)

    # workstation empieza por el label del profesional
    assert rec.workstation.startswith(f"{actor.label()} @ ")

    # Se audita consent_recorded con el digest
    lineas = [item for item in audit_log.lines() if item["kind"] == "consent_recorded"]
    assert len(lineas) == 1
    linea_consent = lineas[0]
    assert linea_consent["actor"] == actor.id
    assert linea_consent["actor_name"] == actor.label()
    data = json.loads(linea_consent["data"])
    assert data["consent"] == rec.digest


def test_audit_record_con_clase_fuera_de_kinds_lanza_value_error(tmp_path: Path) -> None:
    """Caso 18 (§B.4): AuditLog.record con una clase no declarada en KINDS lanza ValueError."""
    actor = Professional(id="p1", name="Dr. Uno", registry="1")
    audit = AuditLog(tmp_path / "audit.jsonl")
    assert "review_saved" in KINDS
    with pytest.raises(ValueError, match="clase de hecho de auditoría desconocida"):
        audit.record("evento_inventado", actor=actor)


# ---------------------------------------------------------------------------
# §B.5 — La aplicación y la CLI (app.py, run.py)
# ---------------------------------------------------------------------------


def test_app_sin_identificarse_da_401_y_muestra_primer_perfil_o_quien(tmp_path: Path) -> None:
    """Caso 19 (§B.5, M8): Sin identificarse, GET /estado y POST /empezar dan 401;
    la página inicial es «Su perfil» si no hay perfiles y «¿Quién atiende?» si los hay."""
    profiles = ProfileStore(tmp_path / "profiles.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    ready = AppReady(
        consultation=_crear_consultation(tmp_path, audit=audit)[0],
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )

    # A. Sin perfiles en el almacén
    url, _, _, _, _ = _lanzar_servidor_identity(
        lambda: ready,
        profiles=profiles,
        audit=audit,
        identificar=False,
    )
    r_index = httpx.get(url, timeout=5.0)
    assert r_index.status_code == 200
    assert "Su perfil" in r_index.text

    r_estado = httpx.get(f"{url}/estado", timeout=5.0)
    assert r_estado.status_code == 401

    r_empezar = httpx.post(f"{url}/empezar", timeout=5.0)
    assert r_empezar.status_code == 401

    # B. Con un perfil ya existente en el almacén
    profiles.create(name="Dr. Juan López", registry="11.222.333", pin="1234")
    ready2 = AppReady(
        consultation=_crear_consultation(tmp_path / "t2", audit=audit)[0],
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )
    url2, _, _, _, _ = _lanzar_servidor_identity(
        lambda: ready2,
        profiles=profiles,
        audit=audit,
        identificar=False,
    )
    r_index2 = httpx.get(url2, timeout=5.0)
    assert r_index2.status_code == 200
    assert "¿Quién atiende?" in r_index2.text


def test_app_post_perfil_fuera_de_first_profile_y_ajustes_da_409_y_desde_ajustes_audita(
    tmp_path: Path,
) -> None:
    """Caso 20 (§B.5, M10): POST /perfil fuera de first_profile y sin Ajustes da 409.
    Desde Ajustes identificado crea perfil (200) y audita profile_created."""
    profiles = ProfileStore(tmp_path / "profiles.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    ready = AppReady(
        consultation=_crear_consultation(tmp_path, audit=audit)[0],
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )
    tools = AppTools(
        cloud_key_source=lambda: None,
        check_cloud_key=lambda c: (True, ""),
        save_cloud_key=lambda c: None,
        model_name="modelo-prueba",
        model_total_mb=100,
        model_available=lambda: True,
        model_bytes=lambda: 1000,
        download_model=lambda: None,
        reader_version=lambda: "1.0",
        cache_key_since=lambda: 0.0,
        local_only=lambda: False,
        save_local_only=lambda v: None,
        destinations=(),
    )

    url, _, _, _, _ = _lanzar_servidor_identity(
        lambda: ready,
        profiles=profiles,
        audit=audit,
        tools=tools,
        identificar=True,  # Crea el primer perfil y deja sesión en 'start'
    )

    # 1. En estado 'start': crear perfil da 409
    r_start = httpx.post(
        f"{url}/perfil",
        json={"name": "Dr. Segundo", "registry": "22.222.222", "pin": "1234", "pin2": "1234"},
        timeout=5.0,
    )
    assert r_start.status_code == 409

    # 2. Entrar a Ajustes
    r_ajustes = httpx.post(f"{url}/ajustes", timeout=5.0)
    assert r_ajustes.status_code == 200

    # 3. Desde Ajustes identificado: 200 y audita profile_created
    r_nuevo = httpx.post(
        f"{url}/perfil",
        json={"name": "Dr. Segundo", "registry": "22.222.222", "pin": "1234", "pin2": "1234"},
        timeout=5.0,
    )
    assert r_nuevo.status_code == 200

    lineas_perfil = [linea for linea in audit.lines() if linea["kind"] == "profile_created"]
    assert len(lineas_perfil) >= 2  # el primero y el segundo
    ultimo = lineas_perfil[-1]
    assert "Dr. Segundo" in ultimo["data"]
    assert ultimo["actor_name"] == "Dra. Laura Gómez (52.341.889)"


def test_app_entrar_con_pin_malo_da_400_con_intentos_restantes(tmp_path: Path) -> None:
    """Caso 21 (§B.5): entrar con PIN erróneo devuelve 400 y mensaje con intentos restantes."""
    profiles = ProfileStore(tmp_path / "profiles.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    p = profiles.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")
    ready = AppReady(
        consultation=_crear_consultation(tmp_path, audit=audit)[0],
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )

    url, _, _, _, _ = _lanzar_servidor_identity(
        lambda: ready,
        profiles=profiles,
        audit=audit,
        identificar=False,
    )

    r_malo = httpx.post(
        f"{url}/entrar",
        json={"profile": p.id, "pin": "0000"},
        timeout=5.0,
    )
    assert r_malo.status_code == 400
    assert "4 intentos" in r_malo.json().get("error", "")


def test_app_bloqueo_por_inactividad_en_start_y_no_en_review_y_desbloqueo(tmp_path: Path) -> None:
    """Caso 22 (§B.5, M6, M7): Pasado el plazo en start, vigilar bloquea (state locked)
    y audita locked; en review NO bloquea. Desbloquear con PIN bueno vuelve a start y
    audita unlocked."""
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    tiempo = 100.0
    profiles = ProfileStore(tmp_path / "profiles.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    ready = AppReady(
        consultation=_crear_consultation(tmp_path, audit=audit)[0],
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )

    url, _, _, _, _ = _lanzar_servidor_identity(
        lambda: ready,
        profiles=profiles,
        audit=audit,
        clock=lambda: tiempo,
        identificar=True,  # Identificada en 'start'
    )

    # 1. En 'start': transcurren 11 minutos (> 10 minutos de reposo)
    tiempo += 11 * 60
    r_act = httpx.post(f"{url}/actividad", timeout=5.0)
    assert r_act.status_code == 401

    r_estado = httpx.get(f"{url}/estado", timeout=5.0)
    assert r_estado.status_code == 401

    r_html = httpx.get(url, timeout=5.0)
    bloqueada_en_html = (
        "Bloqueada" in r_html.text
        or "pantalla-bloqueada" in r_html.text
        or "Desbloquear" in r_html.text
    )
    assert bloqueada_en_html

    assert any(linea["kind"] == "locked" for linea in audit.lines())

    # 2. Desbloquear con PIN bueno vuelve a 'start'
    r_desbloqueo = httpx.post(f"{url}/desbloquear", json={"pin": "1234"}, timeout=5.0)
    assert r_desbloqueo.status_code == 200
    assert any(linea["kind"] == "unlocked" for linea in audit.lines())

    r_estado_post = httpx.get(f"{url}/estado", timeout=5.0)
    assert r_estado_post.status_code == 200
    assert r_estado_post.json()["state"] == "start"

    # 3. En 'review': transcurren 11 minutos y NO bloquea
    httpx.post(
        f"{url}/audio",
        content=_crear_wav_minimo(),
        headers={"X-Name": "c.wav"},
        timeout=5.0,
    )
    httpx.post(
        f"{url}/documento",
        content=b"Historia previa",
        headers={"X-Name": "h.txt"},
        timeout=5.0,
    )
    httpx.post(
        f"{url}/empezar",
        json={"owner": "paciente_nobloq", "consent": True, "cloud": True},
        timeout=5.0,
    )
    for _ in range(30):
        time.sleep(0.05)
        if httpx.get(f"{url}/estado", timeout=5.0).json().get("state") == "review":
            break
    assert httpx.get(f"{url}/estado", timeout=5.0).json()["state"] == "review"

    # Avanza 11 minutos estando en review
    tiempo += 11 * 60
    assert httpx.get(f"{url}/estado", timeout=5.0).json()["state"] == "review"


def test_app_cambiar_en_review_da_409_y_en_start_pasa_a_who_y_audita_cierre(tmp_path: Path) -> None:
    """Caso 23 (§B.5): cambiar en review da 409; en start pasa a who y audita session_closed."""
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    profiles = ProfileStore(tmp_path / "profiles.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    ready = AppReady(
        consultation=_crear_consultation(tmp_path, audit=audit)[0],
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )

    url, _, _, _, _ = _lanzar_servidor_identity(
        lambda: ready,
        profiles=profiles,
        audit=audit,
        identificar=True,
    )

    # En 'start': cambiar pasa a 'who' y audita session_closed
    r_cambiar = httpx.post(f"{url}/cambiar", timeout=5.0)
    assert r_cambiar.status_code == 200
    assert any(linea["kind"] == "session_closed" for linea in audit.lines())
    assert httpx.get(f"{url}/estado", timeout=5.0).status_code == 401

    # Volvemos a entrar para probar desde review
    prof = profiles.professionals()[0]
    httpx.post(f"{url}/entrar", json={"profile": prof.id, "pin": "1234"}, timeout=5.0)

    httpx.post(
        f"{url}/audio",
        content=_crear_wav_minimo(),
        headers={"X-Name": "c.wav"},
        timeout=5.0,
    )
    httpx.post(
        f"{url}/documento",
        content=b"Historia previa",
        headers={"X-Name": "h.txt"},
        timeout=5.0,
    )
    httpx.post(
        f"{url}/empezar",
        json={"owner": "paciente_cambio", "consent": True, "cloud": True},
        timeout=5.0,
    )
    for _ in range(30):
        time.sleep(0.05)
        if httpx.get(f"{url}/estado", timeout=5.0).json().get("state") == "review":
            break
    assert httpx.get(f"{url}/estado", timeout=5.0).json()["state"] == "review"

    # En 'review': cambiar da 409
    r_cambiar_rev = httpx.post(f"{url}/cambiar", timeout=5.0)
    assert r_cambiar_rev.status_code == 409


def test_app_historia_vista_dos_veces_audita_una_sola_vez(tmp_path: Path) -> None:
    """Caso 24 (§B.5, M9): GET /historia dos veces en la misma consulta registra
    exactamente una sola línea history_viewed en la auditoría."""
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    profiles = ProfileStore(tmp_path / "profiles.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    ready = AppReady(
        consultation=_crear_consultation(tmp_path, audit=audit)[0],
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )

    url, _, _, _, _ = _lanzar_servidor_identity(
        lambda: ready,
        profiles=profiles,
        audit=audit,
        identificar=True,
    )

    httpx.post(
        f"{url}/audio",
        content=_crear_wav_minimo(),
        headers={"X-Name": "c.wav"},
        timeout=5.0,
    )
    httpx.post(
        f"{url}/documento",
        content=b"%PDF-1.4 Historia previa",
        headers={"X-Name": "historia.pdf"},
        timeout=5.0,
    )
    httpx.post(
        f"{url}/empezar",
        json={"owner": "paciente_historia", "consent": True, "cloud": True},
        timeout=5.0,
    )
    for _ in range(30):
        time.sleep(0.05)
        if httpx.get(f"{url}/estado", timeout=5.0).json().get("state") == "review":
            break
    assert httpx.get(f"{url}/estado", timeout=5.0).json()["state"] == "review"

    r1 = httpx.get(f"{url}/historia", timeout=5.0)
    assert r1.status_code == 200

    r2 = httpx.get(f"{url}/historia", timeout=5.0)
    assert r2.status_code == 200

    vistas = [linea for linea in audit.lines() if linea["kind"] == "history_viewed"]
    assert len(vistas) == 1
    assert vistas[0]["owner"] == "paciente_historia"


def test_app_guardar_revision_audita_review_saved_con_profesional_de_sesion(tmp_path: Path) -> None:
    """Caso 25 (§B.5, M11): Guardar la revisión en la aplicación audita review_saved
    asociado al profesional activo en la sesión."""
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    profiles = ProfileStore(tmp_path / "profiles.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    ready = AppReady(
        consultation=_crear_consultation(tmp_path, audit=audit)[0],
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )

    url, _, _, _, _ = _lanzar_servidor_identity(
        lambda: ready,
        profiles=profiles,
        audit=audit,
        identificar=True,
    )

    httpx.post(
        f"{url}/audio",
        content=_crear_wav_minimo(),
        headers={"X-Name": "c.wav"},
        timeout=5.0,
    )
    httpx.post(
        f"{url}/documento",
        content=b"Historia previa",
        headers={"X-Name": "h.txt"},
        timeout=5.0,
    )
    httpx.post(
        f"{url}/empezar",
        json={"owner": "paciente_guardar", "consent": True, "cloud": True},
        timeout=5.0,
    )
    for _ in range(30):
        time.sleep(0.05)
        if httpx.get(f"{url}/estado", timeout=5.0).json().get("state") == "review":
            break
    assert httpx.get(f"{url}/estado", timeout=5.0).json()["state"] == "review"

    # Guardar la revisión
    r_guardar = httpx.post(
        f"{url}/guardar",
        json={
            "answers": {"M255": {"verdict": "confirmed"}},
            "note": {"motivo_consulta": "dolor rodilla"},
            "elapsed_seconds": 2.0,
        },
        timeout=5.0,
    )
    assert r_guardar.status_code == 200

    lineas_rev = [linea for linea in audit.lines() if linea["kind"] == "review_saved"]
    assert len(lineas_rev) == 1
    assert lineas_rev[0]["actor_name"] == "Dra. Laura Gómez (52.341.889)"
    assert lineas_rev[0]["actor"] != ""


def test_cli_as_y_audit_comandos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Caso 26 (§B.5, M14): CLI: --review sin --as => 1; --as desconocido => 1;
    --audit con cadena rota => 1; --audit-show con prefijo no único => 1."""
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(
        "aida.arms.health.capture.build_local_capture",
        lambda: (_DummySTT(), _DummyOCR()),
    )
    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio")
    hist_path = tmp_path / "historia.txt"
    hist_path.write_text("Historia.")

    # 1. --review sin --as => 1
    cod1 = main(
        [
            "--consultation",
            "--segment",
            "--review",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            "pac_cli_no_as",
            "--consent",
        ]
    )
    cap1 = capsys.readouterr()
    assert cod1 == 1
    assert "aida: error: falta --as PERFIL: todo registro tiene autor" in cap1.err

    # 2. --as desconocido => 1
    store = ProfileStore(user_dir("config") / "profiles.json")
    store.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")

    cod2 = main(
        [
            "--consultation",
            "--segment",
            "--review",
            "--as",
            "00.000.000",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            "pac_cli_bad_as",
            "--consent",
        ]
    )
    cap2 = capsys.readouterr()
    assert cod2 == 1
    assert "no hay un perfil '00.000.000' en este equipo" in cap2.err

    # 3. --audit con cadena rota => 1
    _audit.cache_clear()
    audit_file = user_dir("data") / "auditoria.jsonl"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    l1 = {
        "at": "2026-01-01T10:00:00",
        "kind": "session_opened",
        "actor": "p1",
        "actor_name": "Doc",
        "workstation": "w",
        "owner": "",
        "data": "{}",
        "blob": "blob_uno_1234",
        "prev": GENESIS,
    }
    h1 = chain_digest(l1)
    l1["digest"] = h1
    l2 = {
        "at": "2026-01-01T10:05:00",
        "kind": "session_closed",
        "actor": "p1",
        "actor_name": "Doc",
        "workstation": "w",
        "owner": "",
        "data": "{}",
        "blob": "blob_dos_5678",
        "prev": "cadena_rota_no_coincide",
    }
    h2 = chain_digest(l2)
    l2["digest"] = h2
    audit_file.write_text(
        json.dumps(l1, ensure_ascii=False, sort_keys=True)
        + "\n"
        + json.dumps(l2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    cod3 = main(["--audit"])
    cap3 = capsys.readouterr()
    assert cod3 == 1
    assert "aida: error: línea 2: no sigue a la anterior" in cap3.err

    # 4. --audit-show con prefijo no único => 1
    _audit.cache_clear()
    cod4 = main(["--audit-show", "blob_"])
    cap4 = capsys.readouterr()
    assert cod4 == 1
    assert "aida: error: esa huella no identifica una sola línea" in cap4.err


# ---------------------------------------------------------------------------
# Plan 83b: Todo registro tiene autor (I-156) (§B.2)
# ---------------------------------------------------------------------------


def test_audit_record_sin_actor_lanza_identity_error_y_no_anade_linea(tmp_path: Path) -> None:
    """Caso 4 (§B.2, M1): AuditLog.record con actor=None lanza IdentityError('no_actor')
    y no añade línea nueva a la cadena."""
    audit_file = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_file)
    with pytest.raises(IdentityError) as exc:
        audit.record("session_opened", actor=None)  # type: ignore[arg-type]
    assert exc.value.code == "no_actor"
    assert not audit_file.exists() or len(list(audit.lines())) == 0


def test_record_consent_sin_actor_lanza_identity_error_sin_linea_ni_evento(tmp_path: Path) -> None:
    """Caso 5 (§B.2, M2): record_consent con actor=None lanza IdentityError('no_actor')
    y no escribe en consents ni emite CONSENT_RECORDED."""
    consultation, event_log, _ = _crear_consultation(tmp_path)
    owner = AgentId("doc_no_actor")
    with pytest.raises(IdentityError) as exc:
        consultation.record_consent(cloud=True, owner=owner, mode="uploaded", actor=None)  # type: ignore[arg-type]
    assert exc.value.code == "no_actor"
    assert len(consultation._consents.records()) == 0
    assert not any(e.kind == EventKind.CONSENT_RECORDED for e in event_log)


def test_cli_consultation_sin_as_devuelve_uno_sin_llamar_stt(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 6 (§B.2, M3): CLI --consultation sin --as sale con 1, mensaje de error y
    el STT simulado no recibe llamadas."""
    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(_crear_wav_minimo())
    hist_path = tmp_path / "historia.txt"
    hist_path.write_text("Historia clínica.")

    stt_calls = 0

    class _CountingSTT:
        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal stt_calls
            stt_calls += 1
            return []

        def transcribe_segments(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal stt_calls
            stt_calls += 1
            return ()

    monkeypatch.setattr(
        "aida.arms.health.capture.build_local_capture",
        lambda: (_CountingSTT(), _DummyOCR()),
    )

    codigo = main(
        [
            "--consultation",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            "pac_cli_sin_as",
            "--consent",
        ]
    )
    cap = capsys.readouterr()
    assert codigo == 1
    assert "aida: error: falta --as PERFIL: todo registro tiene autor" in cap.err
    assert stt_calls == 0


def test_cli_mic_history_sin_as_devuelve_uno_antes_de_abrir_microfono(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 6 (§B.2, M4): CLI --mic --history sin --as sale con 1 antes de abrir el micrófono."""
    pytest.importorskip("sounddevice", reason="requiere el extra 'capture' (sounddevice)")
    from tests.test_run import _pack_yaml_valido

    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")
    hist_path = tmp_path / "historia.txt"
    hist_path.write_text("Historia clínica.")

    mic_opened = False

    class _SpyMic:
        def __enter__(self) -> Any:
            nonlocal mic_opened
            mic_opened = True
            raise AssertionError("no debió abrir el micrófono")

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr("aida.voice.listen.MicrophoneSource", _SpyMic)
    monkeypatch.setattr(
        "aida.arms.health.capture.build_local_capture",
        lambda: (_DummySTT(), _DummyOCR()),
    )

    codigo = main(
        [
            "--pack",
            str(pack_file),
            "--converse",
            "--mic",
            "--history",
            str(hist_path),
            "--owner",
            "pac_mic_sin_as",
            "--consent",
        ]
    )
    cap = capsys.readouterr()
    assert codigo == 1
    assert "aida: error: falta --as PERFIL: todo registro tiene autor" in cap.err
    assert not mic_opened


def test_cli_as_con_export_auditoria_y_autorizacion_mismo_profesional(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 7 (§B.2): CLI con --as y --export registra la línea 'exported' y
    la autorización con el mismo profesional."""
    _audit.cache_clear()
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    store = ProfileStore(user_dir("config") / "profiles.json")
    prof = store.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")
    monkeypatch.setattr("getpass.getpass", lambda *args, **kwargs: "1234")

    monkeypatch.setattr(
        "aida.arms.health.capture.build_local_capture",
        lambda: (_DummySTT(), _DummyOCR()),
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(_crear_wav_minimo())
    hist_path = tmp_path / "historia.txt"
    hist_path.write_text("Historia clínica previa.")
    export_path = tmp_path / "salida.md"

    codigo = main(
        [
            "--consultation",
            "--as",
            prof.registry,
            "--segment",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            "pac_cli_export_mismo_prof",
            "--export",
            str(export_path),
            "--llm",
            "ninguno",
            "--consent",
        ]
    )
    assert codigo == 0

    consents_file = user_dir("data") / "autorizaciones.jsonl"
    assert consents_file.exists()
    consent_lines = [
        json.loads(line)
        for line in consents_file.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    assert len(consent_lines) > 0
    last_consent = consent_lines[-1]
    assert prof.label() in last_consent["workstation"]

    audit_file = user_dir("data") / "auditoria.jsonl"
    assert audit_file.exists()
    audit_lines = [
        json.loads(line)
        for line in audit_file.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    exported_lines = [line for line in audit_lines if line.get("kind") == "exported"]
    assert len(exported_lines) == 1
    exp = exported_lines[0]
    assert exp["actor"] == prof.id
    assert exp["actor_name"] == prof.label()

    assert exp["actor_name"] in last_consent["workstation"]


def test_app_actor_sin_profesional_en_estado_que_registra_falla_sin_auditar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 8 (§B.2, M5): actor() de serve con session.professional a None en un estado
    que registra (forzado en el test), la petición falla sin escribir en la auditoría."""
    profiles = ProfileStore(tmp_path / "profiles.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    ready = AppReady(
        consultation=_crear_consultation(tmp_path, audit=audit)[0],
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="modelo-prueba",
    )

    captured_sessions: list[_Session] = []
    orig_init = _Session.__init__

    def spy_init(self: _Session, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        captured_sessions.append(self)

    monkeypatch.setattr(_Session, "__init__", spy_init)

    record_calls: list[Any] = []
    orig_record = audit.record

    def spy_record(*args: Any, **kwargs: Any) -> Any:
        record_calls.append(kwargs)
        return orig_record(*args, **kwargs)

    monkeypatch.setattr(audit, "record", spy_record)

    url, _, _, _, _ = _lanzar_servidor_identity(
        lambda: ready,
        profiles=profiles,
        audit=audit,
        identificar=True,
    )

    assert len(captured_sessions) >= 1
    session = captured_sessions[0]
    prof_real = session.professional
    # Forzar session.professional = None en un estado que registra (review)
    session.state = "review"
    session.view = HistoryView(
        content_type="text/plain", data=b"historia original", text="historia original"
    )
    session.history_seen = False

    class _FlapProf:
        def __init__(self, real: Any) -> None:
            self.real = real

        def __get__(self, obj: Any, objtype: Any = None) -> Any:
            import traceback

            for frame in traceback.extract_stack():
                if frame.name == "actor":
                    return None
            return self.real

        def __set__(self, obj: Any, val: Any) -> None:
            pass

    monkeypatch.setattr(_Session, "professional", _FlapProf(prof_real))

    audit_lines_before = len(list(audit.lines()))
    record_calls.clear()

    # Petición a /historia que audita history_viewed llamando a actor()
    # Con IdentityError en actor(), el servidor cierra la conexión y httpx lanza error
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{url}/historia", timeout=5.0)

    # auditoria.record no debe haber sido llamado
    assert len(record_calls) == 0
    # No se escribe nada en la auditoría
    assert len(list(audit.lines())) == audit_lines_before
