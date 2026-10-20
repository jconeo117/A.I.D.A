from __future__ import annotations

import contextlib
import threading
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from aida.platform.audit import AuditLog
from aida.platform.identity import Professional, ProfileStore
from aida.platform.license import LicenseState
from aida.platform.web.shell import ArmApp, Response, ShellContext, json_response, run_shell
from aida.platform.web.wording import load_shell_wording
from tests.conftest import _ORIGINAL_HANDLE_REQUEST


@pytest.fixture(autouse=True)
def _permitir_servidor_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permite conexiones HTTP locales durante los tests del lanzador."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


class MockArm(ArmApp):
    def __init__(self, arm_id: str, name: str, at_rest_flag: bool = True) -> None:
        self.id = arm_id
        self.name = name
        self.en_reposo = at_rest_flag
        self.resets = 0

    def at_rest(self) -> bool:
        return self.en_reposo

    def in_settings(self) -> bool:
        return False

    def reset(self) -> None:
        self.resets += 1

    def rebuild(self) -> None:
        self.reset()

    def on_identified(self, actor: Professional) -> None:
        pass

    def on_unlocked(self) -> None:
        pass

    def settings_cards(self, ctx: ShellContext) -> str:
        return ""

    def readonly_items(self, ctx: ShellContext) -> tuple[str, ...]:
        return ()

    def page(self, ctx: ShellContext) -> str:
        return f"<html><body><h1>Pantalla de {self.name}</h1></body></html>"

    def get(self, path: str, ctx: ShellContext) -> Response | None:
        if path == "ping":
            return json_response(200, {"brazo": self.id})
        return None

    def post(
        self, path: str, body: bytes | None, headers: Mapping[str, str], ctx: ShellContext
    ) -> Response | None:
        if path == "do":
            return json_response(200, {"ok": True, "brazo": self.id})
        return None


def _lanzar_shell(tmp_path: Path, arms: list[ArmApp]) -> tuple[str, threading.Thread]:
    store = ProfileStore(tmp_path / "profiles.json")
    p = store.create(name="Dra. Gómez", registry="12345", pin="1234")
    auditoria = AuditLog(tmp_path / "audit.jsonl")
    lock = threading.Lock()
    info: dict[str, str] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    hilo = threading.Thread(
        target=run_shell,
        args=(arms,),
        kwargs={
            "brand": "AIDA",
            "wording": load_shell_wording(),
            "profiles": store,
            "audit": auditoria,
            "license": LicenseState(status="active"),
            "license_path": None,
            "lock": lock,
            "open_browser": _abrir,
        },
        daemon=True,
    )
    hilo.start()
    listo.wait(timeout=5.0)
    info["profile_id"] = p.id
    return info["url"], hilo


def test_lanzador_con_dos_brazos(tmp_path: Path) -> None:
    arm_salud = MockArm("health", "Consultorio")
    arm_agenda = MockArm("agenda", "Agenda")
    url, hilo = _lanzar_shell(tmp_path, [arm_salud, arm_agenda])

    try:
        # Recuperar perfil creado
        store = ProfileStore(tmp_path / "profiles.json")
        prof = store.professionals()[0]
        r_entrar = httpx.post(
            f"{url}/entrar", json={"profile": prof.id, "pin": "1234"}, timeout=5.0
        )
        assert r_entrar.status_code == 200

        r_page = httpx.get(url, timeout=5.0)
        assert r_page.status_code == 200
        assert 'id="launcher"' in r_page.text
        idx_agenda = r_page.text.find('data-arm="agenda"')
        idx_health = r_page.text.find('data-arm="health"')
        assert idx_agenda != -1 and idx_health != -1
        assert idx_agenda < idx_health

        r_arm_route = httpx.post(f"{url}/do", timeout=5.0)
        assert r_arm_route.status_code == 409

        r_unk = httpx.post(f"{url}/brazo", json={"id": "fantasma"}, timeout=5.0)
        assert r_unk.status_code == 404

        r_open = httpx.post(f"{url}/brazo", json={"id": "agenda"}, timeout=5.0)
        assert r_open.status_code == 200

        r_do = httpx.post(f"{url}/do", timeout=5.0)
        assert r_do.status_code == 200
        assert r_do.json()["brazo"] == "agenda"

        arm_agenda.en_reposo = False
        r_back_busy = httpx.post(f"{url}/lanzador", timeout=5.0)
        assert r_back_busy.status_code == 409

        arm_agenda.en_reposo = True
        r_back_ok = httpx.post(f"{url}/lanzador", timeout=5.0)
        assert r_back_ok.status_code == 200
        assert arm_agenda.resets == 1

        r_launcher_again = httpx.get(url, timeout=5.0)
        assert 'id="launcher"' in r_launcher_again.text
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_unico_brazo_no_usa_lanzador(tmp_path: Path) -> None:
    arm_unico = MockArm("health", "Consultorio")
    url, hilo = _lanzar_shell(tmp_path, [arm_unico])

    try:
        store = ProfileStore(tmp_path / "profiles.json")
        prof = store.professionals()[0]
        r_entrar = httpx.post(
            f"{url}/entrar", json={"profile": prof.id, "pin": "1234"}, timeout=5.0
        )
        assert r_entrar.status_code == 200

        r_page = httpx.get(url, timeout=5.0)
        assert r_page.status_code == 200
        assert "Pantalla de Consultorio" in r_page.text
        assert 'id="launcher"' not in r_page.text
        assert 'id="to-launcher"' not in r_page.text

        r_lanz = httpx.post(f"{url}/lanzador", timeout=5.0)
        assert r_lanz.status_code == 404
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)
