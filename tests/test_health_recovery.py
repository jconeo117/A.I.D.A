"""Tests de recuperación de grabaciones interrumpidas (Plan 87).

Cubre §B.2:
1. sealed:
   - read_appended con la última línea truncada a mano devuelve lo anterior;
   - con una línea del medio dañada, InvalidToken;
   - append llama a os.fsync (espía).
2. recovery:
   - ida y vuelta de write_marker -> interrupted;
   - el archivo no contiene el paciente en claro;
   - con otra clave, interrupted no la lista, pero expired sí la cuenta al pasar del plazo;
   - purge_unmarked borra una carpeta sin metadatos y respeta una con ellos, aunque sean ilegibles.
3. Arranque: serve con una carpeta de grabación con metadatos y otra sin ellos en sessions=
   => tras arrancar, solo queda la primera. Los aida-sesion-* de /tmp también se borran.
4. Quién la ve: dos perfiles; la grabación es del primero. Al entrar el segundo, el inicio no
   tiene id="recover", y /recuperar y /descartar con ese id dan 404. Al entrar el primero, la
   tarjeta está con el paciente y la duración.
5. Recuperar:
   - con una Consultation de prueba, /recuperar llama a procesar con RECORDING_NAME y la
     autorización de la huella, y llega a review;
   - deja recording_recovered con autor;
   - si la huella ya no está en el registro, 409 recover_gone y la carpeta sigue;
   - con licencia caducada, 403.
6. Descartar: borra la carpeta y deja recording_discarded con autor.
7. Plazo: con interrupted_hours=0 y dos grabaciones (una de cada profesional), al entrar
   cualquiera se borran las dos, con dos recording_purged firmadas por quien entra. Con None,
   no se borra nada.
8. Sin clave del equipo (seal_key=None): grabar no escribe sesion.json.sealed.
9. load_retention revienta sin interrupted_hours o con 0.
10. user_dir("data") respeta AIDA_DATA_DIR.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from cryptography.fernet import Fernet, InvalidToken

from aida.arms.health.app import (
    _WORKDIR_PREFIX,
    AppReady,
    load_app_wording,
)
from aida.arms.health.cache import ConsultationCache
from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.consent import ConsentLedger
from aida.arms.health.consultation import Consultation
from aida.arms.health.extract import build_health_extractor
from aida.arms.health.live import RECORDING_NAME
from aida.arms.health.match import build_index
from aida.arms.health.note import load_note_settings
from aida.arms.health.recovery import (
    MARKER_NAME,
    expired,
    interrupted,
    purge_unmarked,
    write_marker,
)
from aida.arms.health.retention import load_retention
from aida.arms.health.segment import build_health_segmenter
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import EventLog
from aida.engine.ontology import AgentId
from aida.platform.audit import AuditLog
from aida.platform.identity import ProfileStore
from aida.platform.license import License, LicenseState
from aida.platform.sealed import SealedAppender, read_appended, seal_bytes
from aida.resources import user_dir
from tests.conftest import _ORIGINAL_HANDLE_REQUEST
from tests.test_health_app import _crear_app_ready, _DummyOCR, _DummySTT, _lanzar_servidor_app


@pytest.fixture(autouse=True)
def _permitir_transporte_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


def _crear_pcm_minimo() -> bytes:
    """1600 muestras a 16 bits = 3200 bytes de PCM puro a 16 kHz."""
    return b"\x00\x00" * 1600


# ---------------------------------------------------------------------------
# §B.2.1 — sealed
# ---------------------------------------------------------------------------


def test_sealed_read_appended_tolera_ultima_linea_truncada(tmp_path: Path) -> None:
    path = tmp_path / "appended.sealed"
    key = Fernet.generate_key()
    appender = SealedAppender(path, key)
    appender.append(b"primer trozo de audio")
    appender.append(b"segundo trozo de audio")

    # Truncamos los últimos 15 bytes del archivo (dañando la última línea)
    contenido = path.read_bytes()
    path.write_bytes(contenido[:-15])

    recuperado = read_appended(path, key)
    assert recuperado == b"primer trozo de audio"


def test_sealed_read_appended_linea_intermedia_danada_falla(tmp_path: Path) -> None:
    path = tmp_path / "appended.sealed"
    key = Fernet.generate_key()
    appender = SealedAppender(path, key)
    appender.append(b"trozo 1")
    appender.append(b"trozo 2")
    appender.append(b"trozo 3")

    # Corrompemos la línea 2 (intermedia)
    lineas = [linea for linea in path.read_bytes().splitlines() if linea.strip()]
    assert len(lineas) == 3
    lineas[1] = lineas[1][:-6] + b"XXXXXX"
    path.write_bytes(b"\n".join(lineas) + b"\n")

    with pytest.raises(InvalidToken):
        read_appended(path, key)


def test_sealed_append_llama_fsync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "fsync.sealed"
    key = Fernet.generate_key()
    appender = SealedAppender(path, key)

    fsync_llamadas: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsync_llamadas.append(fd))

    appender.append(b"trozo de datos")
    assert len(fsync_llamadas) == 1


# ---------------------------------------------------------------------------
# §B.2.2 — recovery
# ---------------------------------------------------------------------------


def test_recovery_write_marker_e_interrupted_ida_y_vuelta(tmp_path: Path) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    folder = root / "aida-sesion-test1"
    folder.mkdir()
    key = Fernet.generate_key()
    (folder / RECORDING_NAME).write_bytes(b"dummy")

    ahora = datetime.now().astimezone().replace(microsecond=0)
    write_marker(
        folder,
        owner="HC-0042",
        consent_digest="dig-abc",
        professional="prof-laura",
        started=ahora,
        document="documento.pdf.sealed",
        key=key,
    )

    interrumpidas = interrupted(root, key)
    assert len(interrumpidas) == 1
    item = interrumpidas[0]
    assert item.folder == folder
    assert item.owner == "HC-0042"
    assert item.consent_digest == "dig-abc"
    assert item.professional == "prof-laura"
    assert item.document == "documento.pdf.sealed"
    assert item.started.isoformat(timespec="seconds") == ahora.isoformat(timespec="seconds")
    assert item.duration().total_seconds() >= 0
    assert item.expires(72) == ahora + timedelta(hours=72)


def test_recovery_archivo_marcador_no_contiene_paciente_en_claro(tmp_path: Path) -> None:
    folder = tmp_path / "sesion"
    folder.mkdir()
    key = Fernet.generate_key()
    write_marker(
        folder,
        owner="HC-SECRETO",
        consent_digest="d",
        professional="prof-secreto",
        started=datetime.now().astimezone(),
        document="doc.pdf.sealed",
        key=key,
    )
    marcador_bytes = (folder / MARKER_NAME).read_bytes()
    assert b"HC-SECRETO" not in marcador_bytes
    assert b"prof-secreto" not in marcador_bytes


def test_recovery_otra_clave_no_lista_pero_expired_si_cuenta(tmp_path: Path) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    folder = root / "aida-sesion-c1"
    folder.mkdir()
    key = Fernet.generate_key()
    (folder / RECORDING_NAME).write_bytes(b"rec")
    ahora = datetime.now().astimezone()
    write_marker(
        folder,
        owner="HC-1",
        consent_digest="d",
        professional="p",
        started=ahora,
        document="doc.pdf.sealed",
        key=key,
    )

    otra_clave = Fernet.generate_key()
    assert interrupted(root, otra_clave) == ()

    # Expired sí la encuentra aunque no descifre el marcador (usa la fecha del archivo)
    futuro = ahora + timedelta(hours=1)
    vencidas = expired(root, otra_clave, hours=0, now=futuro)
    assert folder in vencidas


def test_recovery_purge_unmarked_borra_sin_marcador_y_respeta_con_marcador(tmp_path: Path) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    c_sin = root / "aida-sesion-sin-marcador"
    c_sin.mkdir()
    (c_sin / "audio.raw").write_bytes(b"subida a medias")

    c_con_ilegible = root / "aida-sesion-con-marcador-ilegible"
    c_con_ilegible.mkdir()
    (c_con_ilegible / MARKER_NAME).write_bytes(b"no es fernet valido")

    purge_unmarked(root)
    assert not c_sin.exists()
    assert c_con_ilegible.exists()


# ---------------------------------------------------------------------------
# §B.2.3 — Arranque
# ---------------------------------------------------------------------------


def test_arranque_serve_limpia_sin_marcador_y_temporales(tmp_path: Path) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    c_sin = root / "aida-sesion-huerfana"
    c_sin.mkdir()
    (c_sin / "datos.tmp").write_bytes(b"incompleto")

    c_con = root / "aida-sesion-valida"
    c_con.mkdir()
    key = Fernet.generate_key()
    write_marker(
        c_con,
        owner="PAC-01",
        consent_digest="d",
        professional="prof1",
        started=datetime.now().astimezone(),
        document="d.pdf.sealed",
        key=key,
    )
    (c_con / RECORDING_NAME).write_bytes(b"audio")

    # Archivo huérfano en /tmp
    tmp_sesion = Path(tempfile.gettempdir()) / f"{_WORKDIR_PREFIX}tmp-test-{uuid.uuid4()}"
    tmp_sesion.mkdir(parents=True, exist_ok=True)
    (tmp_sesion / "audio.wav").write_bytes(b"antiguo")

    ready = _crear_app_ready(tmp_path)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, sessions=root)
    try:
        httpx.get(url, timeout=5.0)
        assert not c_sin.exists()
        assert c_con.exists()
        assert not tmp_sesion.exists()
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# §B.2.4 — Quién la ve
# ---------------------------------------------------------------------------


def test_quien_la_ve_dos_perfiles(tmp_path: Path) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    folder = root / "aida-sesion-prof1"
    folder.mkdir()
    key = Fernet.generate_key()
    (folder / RECORDING_NAME).write_bytes(b"rec")
    (folder / "hist.pdf.sealed").write_bytes(b"hist")

    profiles = ProfileStore(tmp_path / "profiles.json")
    p1 = profiles.create(name="Dra. Uno", registry="REG-001", pin="1111")
    p2 = profiles.create(name="Dr. Dos", registry="REG-002", pin="2222")

    ahora = datetime.now().astimezone() - timedelta(minutes=5)
    write_marker(
        folder,
        owner="HC-PACIENTE-UNO",
        consent_digest="dig-01",
        professional=p1.id,
        started=ahora,
        document="hist.pdf.sealed",
        key=key,
    )

    ready = _crear_app_ready(tmp_path)
    ready = AppReady(
        consultation=ready.consultation,
        reader=ready.reader,
        log=ready.log,
        cloud=ready.cloud,
        writer=ready.writer,
        live_stt=ready.live_stt,
        local_model=ready.local_model,
        seal_key=key,
    )

    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        sessions=root,
        profiles=profiles,
        identificar=False,
    )
    try:
        # 1. Entra Dr. Dos (p2): no ve la tarjeta de Dra. Uno
        r_entrar2 = httpx.post(f"{url}/entrar", json={"profile": p2.id, "pin": "2222"}, timeout=5.0)
        assert r_entrar2.status_code == 200

        r_inicio2 = httpx.get(url, timeout=5.0)
        assert r_inicio2.status_code == 200
        assert 'id="recover"' not in r_inicio2.text

        r_rec2 = httpx.post(f"{url}/recuperar", json={"id": folder.name}, timeout=5.0)
        assert r_rec2.status_code == 404

        r_desc2 = httpx.post(f"{url}/descartar", json={"id": folder.name}, timeout=5.0)
        assert r_desc2.status_code == 404

        # 2. Cambia a Dra. Uno (p1): sí ve la tarjeta con el paciente
        httpx.post(f"{url}/cambiar", timeout=5.0)
        r_entrar1 = httpx.post(f"{url}/entrar", json={"profile": p1.id, "pin": "1111"}, timeout=5.0)
        assert r_entrar1.status_code == 200

        r_inicio1 = httpx.get(url, timeout=5.0)
        assert r_inicio1.status_code == 200
        assert 'id="recover"' in r_inicio1.text
        assert "HC-PACIENTE-UNO" in r_inicio1.text
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# §B.2.5 — Recuperar
# ---------------------------------------------------------------------------


def test_recuperar_flujo_completo_con_consulta_y_auditoria(tmp_path: Path) -> None:
    pytest.importorskip("av", reason="requiere capture")
    root = tmp_path / "sesiones"
    root.mkdir()
    folder = root / "aida-sesion-rec"
    folder.mkdir()
    key = Fernet.generate_key()

    # Guardar audio en RECORDING_NAME con SealedAppender
    appender = SealedAppender(folder / RECORDING_NAME, key)
    appender.append(_crear_pcm_minimo())

    # Guardar documento sellado
    doc_path = folder / "historia.pdf.sealed"
    seal_bytes(doc_path, b"%PDF-1.4 prueba", key)

    profiles = ProfileStore(tmp_path / "profiles.json")
    prof = profiles.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")

    # Registro de consentimiento en consultation
    ledger = ConsentLedger(tmp_path / "consents.jsonl")
    stt = _DummySTT()
    c = Consultation(
        wording=default_summary_wording(),
        stt=stt,
        local_stt=stt,
        ocr=_DummyOCR(),
        catalog_index=build_index((DiagnosticCode("I10X", "Hipertensión esencial", 4, "cat"),)),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=EventLog(),
        consents=ledger,
        segmenter=build_health_segmenter(),
        extractor=build_health_extractor(),
        note_settings=load_note_settings(),
    )
    registro = c.record_consent(
        owner=AgentId("HC-REC-01"),
        mode="recorded",
        actor=prof,
        cloud=True,
    )

    write_marker(
        folder,
        owner="HC-REC-01",
        consent_digest=registro.digest,
        professional=prof.id,
        started=datetime.now().astimezone(),
        document="historia.pdf.sealed",
        key=key,
    )

    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)

    ready = AppReady(
        consultation=c,
        reader=_DummyOCR(),
        log=EventLog(),
        cloud=True,
        writer="test",
        seal_key=key,
    )

    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        sessions=root,
        profiles=profiles,
        audit=audit,
        identificar=False,
    )
    try:
        r_entrar = httpx.post(
            f"{url}/entrar", json={"profile": prof.id, "pin": "1234"}, timeout=5.0
        )
        assert r_entrar.status_code == 200

        r_rec = httpx.post(f"{url}/recuperar", json={"id": folder.name}, timeout=5.0)
        assert r_rec.status_code == 200

        # Esperar a que pase a review
        for _ in range(50):
            st = httpx.get(f"{url}/estado", timeout=5.0).json()
            if st.get("state") == "review":
                break
            time.sleep(0.1)
        assert st.get("state") == "review"

        lineas_aud = list(audit.lines())
        rec_evts = [linea for linea in lineas_aud if linea.get("kind") == "recording_recovered"]
        assert len(rec_evts) == 1
        assert rec_evts[0]["owner"] == "HC-REC-01"
        assert rec_evts[0]["actor"] == prof.id
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_recuperar_huella_no_esta_en_registro_da_409_recover_gone_y_carpeta_sigue(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    folder = root / "aida-sesion-rec2"
    folder.mkdir()
    key = Fernet.generate_key()
    (folder / RECORDING_NAME).write_bytes(b"rec")
    (folder / "doc.pdf.sealed").write_bytes(b"doc")

    profiles = ProfileStore(tmp_path / "profiles.json")
    prof = profiles.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")

    write_marker(
        folder,
        owner="HC-REC-02",
        consent_digest="huella_que_no_existe_en_el_ledger",
        professional=prof.id,
        started=datetime.now().astimezone(),
        document="doc.pdf.sealed",
        key=key,
    )

    ready = _crear_app_ready(tmp_path)
    ready = AppReady(
        consultation=ready.consultation,
        reader=ready.reader,
        log=ready.log,
        cloud=ready.cloud,
        writer=ready.writer,
        seal_key=key,
    )
    wording = load_app_wording()

    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        sessions=root,
        profiles=profiles,
        identificar=False,
    )
    try:
        r_entrar = httpx.post(
            f"{url}/entrar", json={"profile": prof.id, "pin": "1234"}, timeout=5.0
        )
        assert r_entrar.status_code == 200

        r_rec = httpx.post(f"{url}/recuperar", json={"id": folder.name}, timeout=5.0)
        assert r_rec.status_code == 409
        assert r_rec.json()["error"] == wording["recover_gone"]
        # La carpeta sigue existiendo
        assert folder.exists()
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_recuperar_con_licencia_caducada_da_403(tmp_path: Path) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    folder = root / "aida-sesion-rec3"
    folder.mkdir()
    key = Fernet.generate_key()
    (folder / RECORDING_NAME).write_bytes(b"rec")

    profiles = ProfileStore(tmp_path / "profiles.json")
    prof = profiles.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")

    write_marker(
        folder,
        owner="HC-REC-03",
        consent_digest="d",
        professional=prof.id,
        started=datetime.now().astimezone(),
        document="doc.pdf.sealed",
        key=key,
    )

    lic_caducada = LicenseState(
        status="expired",
        license=License(
            id="t",
            client="t",
            arms=("health",),
            issued=date.today() - timedelta(days=400),
            expires=date.today() - timedelta(days=35),
        ),
    )

    ready = _crear_app_ready(tmp_path)
    ready = AppReady(
        consultation=ready.consultation,
        reader=ready.reader,
        log=ready.log,
        cloud=ready.cloud,
        writer=ready.writer,
        seal_key=key,
    )

    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        sessions=root,
        profiles=profiles,
        license=lic_caducada,
        identificar=False,
    )
    try:
        r_entrar = httpx.post(
            f"{url}/entrar", json={"profile": prof.id, "pin": "1234"}, timeout=5.0
        )
        assert r_entrar.status_code == 200

        r_rec = httpx.post(f"{url}/recuperar", json={"id": folder.name}, timeout=5.0)
        assert r_rec.status_code == 403
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# §B.2.6 — Descartar
# ---------------------------------------------------------------------------


def test_descartar_borra_carpeta_y_registra_auditoria(tmp_path: Path) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    folder = root / "aida-sesion-drop"
    folder.mkdir()
    key = Fernet.generate_key()
    (folder / RECORDING_NAME).write_bytes(b"rec")

    profiles = ProfileStore(tmp_path / "profiles.json")
    prof = profiles.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")

    write_marker(
        folder,
        owner="HC-DROP",
        consent_digest="d",
        professional=prof.id,
        started=datetime.now().astimezone(),
        document="doc.pdf.sealed",
        key=key,
    )

    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)

    ready = _crear_app_ready(tmp_path)
    ready = AppReady(
        consultation=ready.consultation,
        reader=ready.reader,
        log=ready.log,
        cloud=ready.cloud,
        writer=ready.writer,
        seal_key=key,
    )

    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        sessions=root,
        profiles=profiles,
        audit=audit,
        identificar=False,
    )
    try:
        r_entrar = httpx.post(
            f"{url}/entrar", json={"profile": prof.id, "pin": "1234"}, timeout=5.0
        )
        assert r_entrar.status_code == 200

        r_drop = httpx.post(f"{url}/descartar", json={"id": folder.name}, timeout=5.0)
        assert r_drop.status_code == 200
        assert r_drop.json() == {"ok": True}
        assert not folder.exists()

        lineas = list(audit.lines())
        drop_evts = [linea for linea in lineas if linea.get("kind") == "recording_discarded"]
        assert len(drop_evts) == 1
        assert drop_evts[0]["owner"] == "HC-DROP"
        assert drop_evts[0]["actor"] == prof.id
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# §B.2.7 — Plazo
# ---------------------------------------------------------------------------


def test_plazo_expired_con_interrupted_hours_cero_borra_ambas_al_entrar(tmp_path: Path) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    f1 = root / "aida-sesion-p1"
    f1.mkdir()
    f2 = root / "aida-sesion-p2"
    f2.mkdir()
    key = Fernet.generate_key()
    (f1 / RECORDING_NAME).write_bytes(b"rec1")
    (f2 / RECORDING_NAME).write_bytes(b"rec2")

    profiles = ProfileStore(tmp_path / "profiles.json")
    p1 = profiles.create(name="Dra. Uno", registry="REG-1", pin="1111")
    p2 = profiles.create(name="Dr. Dos", registry="REG-2", pin="2222")

    pasado = datetime.now().astimezone() - timedelta(minutes=10)
    write_marker(
        f1,
        owner="PAC-1",
        consent_digest="d1",
        professional=p1.id,
        started=pasado,
        document="d1.sealed",
        key=key,
    )
    write_marker(
        f2,
        owner="PAC-2",
        consent_digest="d2",
        professional=p2.id,
        started=pasado,
        document="d2.sealed",
        key=key,
    )

    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)

    ready = _crear_app_ready(tmp_path)
    ready = AppReady(
        consultation=ready.consultation,
        reader=ready.reader,
        log=ready.log,
        cloud=ready.cloud,
        writer=ready.writer,
        seal_key=key,
    )

    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        sessions=root,
        profiles=profiles,
        audit=audit,
        interrupted_hours=0,
        identificar=False,
    )
    try:
        # Entra Dra. Uno -> al entrar, retener() purga ambas grabaciones vencidas
        r_entrar = httpx.post(
            f"{url}/entrar", json={"profile": p1.id, "pin": "1111"}, timeout=5.0
        )
        assert r_entrar.status_code == 200

        assert not f1.exists()
        assert not f2.exists()

        lineas = list(audit.lines())
        purged_evts = [linea for linea in lineas if linea.get("kind") == "recording_purged"]
        assert len(purged_evts) == 2
        # Ambas firmadas por Dra. Uno (quien entró)
        assert all(e["actor"] == p1.id for e in purged_evts)
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_plazo_interrupted_hours_none_no_borra_nada(tmp_path: Path) -> None:
    root = tmp_path / "sesiones"
    root.mkdir()
    f1 = root / "aida-sesion-guardar"
    f1.mkdir()
    key = Fernet.generate_key()
    (f1 / RECORDING_NAME).write_bytes(b"rec")

    profiles = ProfileStore(tmp_path / "profiles.json")
    p1 = profiles.create(name="Dra. Uno", registry="REG-1", pin="1111")

    write_marker(
        f1,
        owner="PAC-X",
        consent_digest="d",
        professional=p1.id,
        started=datetime.now().astimezone() - timedelta(days=10),
        document="d.sealed",
        key=key,
    )

    ready = _crear_app_ready(tmp_path)
    ready = AppReady(
        consultation=ready.consultation,
        reader=ready.reader,
        log=ready.log,
        cloud=ready.cloud,
        writer=ready.writer,
        seal_key=key,
    )

    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        sessions=root,
        profiles=profiles,
        interrupted_hours=None,
        identificar=True,
    )
    try:
        assert f1.exists()
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# §B.2.8 — Sin clave del equipo (seal_key=None)
# ---------------------------------------------------------------------------


def test_grabar_sin_seal_key_no_escribe_marcador(tmp_path: Path) -> None:
    pytest.importorskip("av", reason="requiere capture")
    root = tmp_path / "sesiones"
    root.mkdir()

    ready = _crear_app_ready(tmp_path)
    assert ready.seal_key is None

    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        sessions=root,
        identificar=True,
    )
    try:
        httpx.post(f"{url}/documento", content=b"hist", headers={"X-Name": "h.txt"}, timeout=5.0)
        r_start = httpx.post(
            f"{url}/empezar",
            json={"owner": "PAC-NO-KEY", "consent": True, "recorded": True, "cloud": False},
            timeout=5.0,
        )
        assert r_start.status_code == 200

        carpetas = list(root.glob(f"{_WORKDIR_PREFIX}*"))
        assert len(carpetas) == 1
        sesion_dir = carpetas[0]
        assert not (sesion_dir / MARKER_NAME).exists()
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_grabar_con_seal_key_escribe_marcador(tmp_path: Path) -> None:
    pytest.importorskip("av", reason="requiere capture")
    root = tmp_path / "sesiones"
    root.mkdir()
    key = Fernet.generate_key()

    ready_base = _crear_app_ready(tmp_path)
    ready = AppReady(
        consultation=ready_base.consultation,
        reader=ready_base.reader,
        log=ready_base.log,
        cloud=ready_base.cloud,
        writer=ready_base.writer,
        seal_key=key,
    )

    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        sessions=root,
        identificar=True,
    )
    try:
        httpx.post(f"{url}/documento", content=b"hist", headers={"X-Name": "h.txt"}, timeout=5.0)
        r_start = httpx.post(
            f"{url}/empezar",
            json={"owner": "PAC-CON-KEY", "consent": True, "recorded": True, "cloud": False},
            timeout=5.0,
        )
        assert r_start.status_code == 200

        carpetas = list(root.glob(f"{_WORKDIR_PREFIX}*"))
        assert len(carpetas) == 1
        sesion_dir = carpetas[0]
        assert (sesion_dir / MARKER_NAME).exists()
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# §B.2.9 — load_retention
# ---------------------------------------------------------------------------


def test_load_retention_revienta_sin_interrupted_hours_o_con_cero(tmp_path: Path) -> None:
    # 1. Sin interrupted_hours
    p_sin = tmp_path / "retention_sin.yaml"
    p_sin.write_text(
        "audit_texts_days: 30\ntranscripts_hours: 24\nsummaries_hours: 48\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_retention(p_sin)

    # 2. Con interrupted_hours = 0
    p_cero = tmp_path / "retention_cero.yaml"
    p_cero.write_text(
        "audit_texts_days: 30\ntranscripts_hours: 24\nsummaries_hours: 48\ninterrupted_hours: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_retention(p_cero)

    # 3. Válido con 72
    p_ok = tmp_path / "retention_ok.yaml"
    p_ok.write_text(
        "audit_texts_days: 30\ntranscripts_hours: 24\nsummaries_hours: 48\ninterrupted_hours: 72\n",
        encoding="utf-8",
    )
    ret = load_retention(p_ok)
    assert ret.interrupted_hours == 72


# ---------------------------------------------------------------------------
# §B.2.10 — user_dir("data")
# ---------------------------------------------------------------------------


def test_user_dir_data_respeta_aida_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom_data = tmp_path / "mi_data_dir"
    monkeypatch.setenv("AIDA_DATA_DIR", str(custom_data))
    assert user_dir("data") == custom_data
