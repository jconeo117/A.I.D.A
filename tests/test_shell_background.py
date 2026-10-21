from __future__ import annotations

import contextlib
from pathlib import Path

import httpx
import pytest

from aida.platform.audit import AuditLog
from aida.platform.identity import ProfileStore
from tests.conftest import _ORIGINAL_HANDLE_REQUEST
from tests.test_platform_web import DummyArm, _lanzar_armazon


@pytest.fixture(autouse=True)
def _permitir_servidor_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


def test_peticion_de_fondo_no_cuenta_como_uso(tmp_path: Path) -> None:
    arm = DummyArm()
    store = ProfileStore(tmp_path / "profiles.json")
    prof = store.create(name="Dra. Test", registry="REG-TEST", pin="1111")
    auditoria = AuditLog(tmp_path / "audit.jsonl")

    reloj = [1000.0]

    def _clock() -> float:
        return reloj[0]

    url, hilo, _, _ = _lanzar_armazon(
        arm, tmp_path, profiles=store, audit=auditoria, clock=_clock
    )
    try:
        with httpx.Client(trust_env=False) as client:
            # Identificarse
            r_login = client.post(
                f"{url}/entrar", json={"profile": prof.id, "pin": "1111"}, timeout=5.0
            )
            assert r_login.status_code == 200

            # Arm en reposo
            arm.set_at_rest(True)

            # Serie de peticiones de fondo antes de vencer el plazo (10 min = 600 s)
            reloj[0] += 300.0  # +5 min (reloj=1300)
            r1 = client.get(url, headers={"X-AIDA-Background": "1"}, timeout=5.0)
            assert r1.status_code == 200
            assert "AIDA está bloqueada" not in r1.text

            reloj[0] += 200.0  # +3m20s (reloj=1500; han pasado 500s desde entrar)
            r2 = client.get(url, headers={"X-AIDA-Background": "1"}, timeout=5.0)
            assert r2.status_code == 200
            assert "AIDA está bloqueada" not in r2.text

            # Avanzamos hasta que transcurran > 600 s desde el último uso real
            reloj[0] += 150.0  # 650 s desde entrar, pero solo 150s desde la última de fondo
            r3 = client.get(url, headers={"X-AIDA-Background": "1"}, timeout=5.0)
            assert r3.status_code == 200
            assert "AIDA está bloqueada" in r3.text

            # Sesión quedó bloqueada
            bloqueos = [x for x in auditoria.lines() if x["kind"] == "locked"]
            assert len(bloqueos) == 1
    finally:
        with contextlib.suppress(Exception), httpx.Client(trust_env=False) as client:
            client.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_peticion_normal_si_cuenta_como_uso(tmp_path: Path) -> None:
    arm = DummyArm()
    store = ProfileStore(tmp_path / "profiles.json")
    prof = store.create(name="Dra. Test", registry="REG-TEST", pin="1111")
    auditoria = AuditLog(tmp_path / "audit.jsonl")

    reloj = [1000.0]

    def _clock() -> float:
        return reloj[0]

    url, hilo, _, _ = _lanzar_armazon(
        arm, tmp_path, profiles=store, audit=auditoria, clock=_clock
    )
    try:
        with httpx.Client(trust_env=False) as client:
            # Identificarse
            r_login = client.post(
                f"{url}/entrar", json={"profile": prof.id, "pin": "1111"}, timeout=5.0
            )
            assert r_login.status_code == 200

            arm.set_at_rest(True)

            # Petición normal a los 500 s (actualiza last_activity a 1500)
            reloj[0] += 500.0
            r_normal = client.get(url, timeout=5.0)
            assert r_normal.status_code == 200
            assert "AIDA está bloqueada" not in r_normal.text

            # A los 700 s desde entrar (reloj=1700), pero solo 200 s desde la petición normal:
            reloj[0] += 200.0
            r_despues = client.get(url, timeout=5.0)
            assert r_despues.status_code == 200
            # Sigue activa, NO se ha bloqueado porque la petición normal contó como uso
            assert "AIDA está bloqueada" not in r_despues.text
            assert not any(x["kind"] == "locked" for x in auditoria.lines())
    finally:
        with contextlib.suppress(Exception), httpx.Client(trust_env=False) as client:
            client.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)
