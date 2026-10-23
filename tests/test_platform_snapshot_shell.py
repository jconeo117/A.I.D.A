"""Tests de las rutas de copia de seguridad en el armazón (78d §B.1)."""

from __future__ import annotations

import base64
import contextlib
import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from aida.arms.health.app import load_app_wording
from aida.platform.audit import AuditLog
from aida.platform.identity import ProfileStore
from aida.platform.license import License, LicenseState
from aida.platform.snapshot import (
    MAGIC,
    create_snapshot,
    pending_restore,
    restore_snapshot,
)
from aida.platform.web.pages import JS_SNAPSHOT, snapshot_block
from aida.platform.web.shell import ArmApp, run_shell
from aida.platform.web.wording import load_shell_wording
from tests.conftest import _ORIGINAL_HANDLE_REQUEST
from tests.test_platform_web import DummyArm


@pytest.fixture(autouse=True)
def _permitir_servidor_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permite conexiones HTTP locales durante los tests del armazón."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


def _lanzar_armazon_con_config(
    arm: ArmApp,
    tmp_path: Path,
    config_dir: Path,
    *,
    profiles: ProfileStore | None = None,
    audit: AuditLog | None = None,
    license_state: LicenseState | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = datetime.now,
    today: Callable[[], date] = date.today,
) -> tuple[str, threading.Thread, ProfileStore, AuditLog]:
    wording = load_shell_wording()
    store = profiles if profiles is not None else ProfileStore(config_dir / "profiles.json")
    auditoria = audit if audit is not None else AuditLog(tmp_path / "audit.jsonl")
    lic = (
        license_state
        if license_state is not None
        else LicenseState(
            status="valid",
            license=License(
                id="test-lic",
                client="Clínica Prueba",
                arms=(arm.id,),
                issued=today(),
                expires=today() + timedelta(days=30),
            ),
            days_left=30,
        )
    )
    lock = threading.Lock()
    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    hilo = threading.Thread(
        target=run_shell,
        args=([arm],),
        kwargs={
            "brand": "AIDA Test",
            "wording": wording,
            "profiles": store,
            "audit": auditoria,
            "license": lic,
            "license_path": config_dir / "licencia.json",
            "lock": lock,
            "open_browser": _abrir,
            "clock": clock,
            "now": now,
            "today": today,
            "config_dir": config_dir,
        },
        daemon=True,
    )
    hilo.start()
    assert listo.wait(timeout=5.0), "el armazón no abrió la URL en 5s"
    return info["url"], hilo, store, auditoria


def test_restaurar_sin_perfiles_y_sin_entrar_200_y_cierra(tmp_path: Path) -> None:
    # Arrange
    src_config = tmp_path / "src_config"
    src_config.mkdir()
    (src_config / "test.txt").write_text("restaurado-ok", encoding="utf-8")
    now_dt = datetime(2026, 10, 23, 10, 0, 0)
    frase = "frase-de-prueba-123"
    snap = create_snapshot(src_config, frase, now=now_dt, version="0.1.0")
    b64_data = base64.b64encode(snap.data).decode("ascii")

    target_config = tmp_path / "target_config"
    target_config.mkdir()
    url, hilo, _, _ = _lanzar_armazon_con_config(DummyArm(), tmp_path, target_config)

    # Act
    try:
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            resp = client.post(
                f"{url}/copia-restaurar",
                json={"file": b64_data, "passphrase": frase, "confirm": False},
            )

        # Assert
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        hilo.join(timeout=1.0)
        assert not hilo.is_alive()
        assert (target_config / "test.txt").read_text(encoding="utf-8") == "restaurado-ok"
    finally:
        if hilo.is_alive():
            with (
                contextlib.suppress(Exception),
                httpx.Client(trust_env=False, timeout=5.0) as client,
            ):
                client.post(f"{url}/salir")
            hilo.join(timeout=1.0)


def test_restaurar_con_perfiles_sin_entrar_401(tmp_path: Path) -> None:
    # Arrange
    cfg = tmp_path / "config"
    cfg.mkdir()
    store = ProfileStore(cfg / "profiles.json")
    store.create(name="Dra. Uno", registry="REG-1", pin="1234")
    url, hilo, _, _ = _lanzar_armazon_con_config(DummyArm(), tmp_path, cfg, profiles=store)

    try:
        # Act
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            resp = client.post(
                f"{url}/copia-restaurar",
                json={"file": "YWJj", "passphrase": "123456789012", "confirm": True},
            )
        # Assert
        assert resp.status_code == 401
        assert resp.json() == {"error": ""}
        assert hilo.is_alive()
    finally:
        with contextlib.suppress(Exception), httpx.Client(trust_env=False, timeout=5.0) as client:
            client.post(f"{url}/salir")
        hilo.join(timeout=5.0)


def test_restaurar_con_perfiles_dentro_exige_confirmacion(tmp_path: Path) -> None:
    # Arrange
    src_config = tmp_path / "src_config"
    src_config.mkdir()
    (src_config / "dato.txt").write_text("dato-nuevo", encoding="utf-8")
    now_dt = datetime(2026, 10, 23, 10, 0, 0)
    frase = "frase-de-prueba-123"
    snap = create_snapshot(src_config, frase, now=now_dt, version="0.1.0")
    b64_data = base64.b64encode(snap.data).decode("ascii")

    cfg = tmp_path / "config"
    cfg.mkdir()
    store = ProfileStore(cfg / "profiles.json")
    prof = store.create(name="Dra. Uno", registry="REG-1", pin="1234")
    url, hilo, _, _ = _lanzar_armazon_con_config(DummyArm(), tmp_path, cfg, profiles=store)
    wording = load_shell_wording()

    try:
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            # Identificarse
            resp_login = client.post(f"{url}/entrar", json={"profile": prof.id, "pin": "1234"})
            assert resp_login.status_code == 200

            # Act 1: sin confirmación -> 400
            resp_no_conf = client.post(
                f"{url}/copia-restaurar",
                json={"file": b64_data, "passphrase": frase, "confirm": False},
            )
            assert resp_no_conf.status_code == 400
            assert resp_no_conf.json()["error"] == wording["snapshot_confirm_needed"]
            assert hilo.is_alive()

            # Act 2: con confirmación -> 200 y cierra
            resp_conf = client.post(
                f"{url}/copia-restaurar",
                json={"file": b64_data, "passphrase": frase, "confirm": True},
            )
            assert resp_conf.status_code == 200
            assert resp_conf.json() == {"ok": True}

        hilo.join(timeout=5.0)
        assert not hilo.is_alive()
    finally:
        if hilo.is_alive():
            with (
                contextlib.suppress(Exception),
                httpx.Client(trust_env=False, timeout=5.0) as client,
            ):
                client.post(f"{url}/salir")
            hilo.join(timeout=5.0)


def test_crear_copia_exige_estar_identificado(tmp_path: Path) -> None:
    # Arrange
    cfg = tmp_path / "config"
    cfg.mkdir()
    url, hilo, _, _ = _lanzar_armazon_con_config(DummyArm(), tmp_path, cfg)

    try:
        # Act
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            resp = client.post(
                f"{url}/copia-crear",
                json={"passphrase": "123456789012", "passphrase2": "123456789012"},
            )
        # Assert
        assert resp.status_code == 401
    finally:
        with contextlib.suppress(Exception), httpx.Client(trust_env=False, timeout=5.0) as client:
            client.post(f"{url}/salir")
        hilo.join(timeout=5.0)


def test_crear_copia_frases_distintas_400(tmp_path: Path) -> None:
    # Arrange
    cfg = tmp_path / "config"
    cfg.mkdir()
    store = ProfileStore(cfg / "profiles.json")
    prof = store.create(name="Dra. Uno", registry="REG-1", pin="1234")
    url, hilo, _, _ = _lanzar_armazon_con_config(DummyArm(), tmp_path, cfg, profiles=store)
    wording = load_shell_wording()

    try:
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            client.post(f"{url}/entrar", json={"profile": prof.id, "pin": "1234"})
            # Act
            resp = client.post(
                f"{url}/copia-crear",
                json={"passphrase": "123456789012", "passphrase2": "otra-frase-distinta"},
            )
            # Assert
            assert resp.status_code == 400
            assert resp.json()["error"] == wording["snapshot_error_mismatch"]
    finally:
        with contextlib.suppress(Exception), httpx.Client(trust_env=False, timeout=5.0) as client:
            client.post(f"{url}/salir")
        hilo.join(timeout=5.0)


def test_crear_copia_audita_snapshot_created(tmp_path: Path) -> None:
    # Arrange
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "clave.txt").write_text("secreto", encoding="utf-8")
    store = ProfileStore(cfg / "profiles.json")
    prof = store.create(name="Dra. Uno", registry="REG-1", pin="1234")
    auditoria = AuditLog(tmp_path / "audit.jsonl")
    url, hilo, _, _ = _lanzar_armazon_con_config(
        DummyArm(), tmp_path, cfg, profiles=store, audit=auditoria
    )

    try:
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            client.post(f"{url}/entrar", json={"profile": prof.id, "pin": "1234"})
            # Act
            resp = client.post(
                f"{url}/copia-crear",
                json={"passphrase": "123456789012", "passphrase2": "123456789012"},
            )
            # Assert
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/octet-stream"
            assert resp.content.startswith(MAGIC)

            lineas = [x for x in auditoria.lines() if x["kind"] == "snapshot_created"]
            assert len(lineas) == 1
            assert lineas[0]["actor"] == prof.id
            assert "files" in lineas[0]["data"]
    finally:
        with contextlib.suppress(Exception), httpx.Client(trust_env=False, timeout=5.0) as client:
            client.post(f"{url}/salir")
        hilo.join(timeout=5.0)


def test_al_entrar_tras_restaurar_se_audita_una_vez(tmp_path: Path) -> None:
    # Arrange
    src_cfg = tmp_path / "src_config"
    src_cfg.mkdir()
    store_src = ProfileStore(src_cfg / "profiles.json")
    prof = store_src.create(name="Dra. Uno", registry="REG-1", pin="1234")
    now_dt = datetime(2026, 10, 23, 10, 0, 0)
    frase = "frase-segura-1234"
    snap = create_snapshot(src_cfg, frase, now=now_dt, version="0.1.0")

    target_cfg = tmp_path / "target_config"
    restore_snapshot(target_cfg, snap.data, frase, now=now_dt)

    auditoria = AuditLog(tmp_path / "audit.jsonl")
    url, hilo, _, _ = _lanzar_armazon_con_config(
        DummyArm(), tmp_path, target_cfg, audit=auditoria
    )

    try:
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            # Primer login -> audita snapshot_restored y limpia pendiente
            resp1 = client.post(f"{url}/entrar", json={"profile": prof.id, "pin": "1234"})
            assert resp1.status_code == 200

            lineas = [x for x in auditoria.lines() if x["kind"] == "snapshot_restored"]
            assert len(lineas) == 1
            assert lineas[0]["actor"] == prof.id
            assert pending_restore(target_cfg) is None

            # Cambiar de profesional y volver a entrar -> no vuelve a auditar
            resp_cambiar = client.post(f"{url}/cambiar")
            assert resp_cambiar.status_code == 200
            resp2 = client.post(f"{url}/entrar", json={"profile": prof.id, "pin": "1234"})
            assert resp2.status_code == 200

            lineas2 = [x for x in auditoria.lines() if x["kind"] == "snapshot_restored"]
            assert len(lineas2) == 1
    finally:
        with contextlib.suppress(Exception), httpx.Client(trust_env=False, timeout=5.0) as client:
            client.post(f"{url}/salir")
        hilo.join(timeout=5.0)


def test_primera_pantalla_trae_restaurar_sin_guardar(tmp_path: Path) -> None:
    # Arrange
    cfg = tmp_path / "config"
    cfg.mkdir()
    url, hilo, _, _ = _lanzar_armazon_con_config(DummyArm(), tmp_path, cfg)

    try:
        # Act
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            resp = client.get(url)
        # Assert
        assert resp.status_code == 200
        assert 'id="snapshot-card"' in resp.text
        assert 'id="snapshot-restore"' in resp.text
        assert 'id="snapshot-create"' not in resp.text
        assert 'class="first-snapshot"' in resp.text
    finally:
        with contextlib.suppress(Exception), httpx.Client(trust_env=False, timeout=5.0) as client:
            client.post(f"{url}/salir")
        hilo.join(timeout=5.0)


def test_ajustes_de_salud_trae_la_tarjeta_completa() -> None:
    # Arrange
    wording = load_shell_wording()
    w_salud = load_app_wording()

    # Act
    card_html = snapshot_block(wording, create=True)
    card_html_salud = snapshot_block(w_salud, create=True)

    # Assert
    for card in (card_html, card_html_salud):
        assert 'id="snapshot-card"' in card
        assert 'id="snapshot-create"' in card
        assert 'id="snapshot-restore"' in card
        assert 'id="snapshot-confirm"' in card
    assert "copia-crear" in JS_SNAPSHOT
    assert "copia-restaurar" in JS_SNAPSHOT
