"""Tests del armazón web común de la plataforma (Plan 101a §B.2).

Verifica el contrato entre el armazón de la plataforma y un brazo desacoplado:
1. Flujo de identidad: primer perfil, quién atiende, entrada con PIN y auditoría.
2. Bloqueo por inactividad: consulta a `arm.at_rest()`.
3. «Cambiar de profesional»: bloqueo ante brazo ocupado y cierre de sesión en reposo.
4. Despacho de rutas: 404 para desconocidas, 401 antes de identificar, paths limpios sin token.
5. Licencia: pantallas de carga, solo lectura y pase al brazo según vigencia.
6. Ajustes: integración del marco común con las tarjetas del brazo.
7. Textos: validación de shell y health wording, detección de claves duplicadas.
8. Independencia: verificación por AST de que platform/web no importa aida.arms.
"""

from __future__ import annotations

import ast
import contextlib
import threading
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from aida.arms.health.app import load_app_wording
from aida.platform.audit import AuditLog
from aida.platform.identity import Professional, ProfileStore
from aida.platform.license import License, LicenseState
from aida.platform.web.pages import (
    license_page,
    readonly_page,
    settings_frame,
)
from aida.platform.web.shell import ArmApp, Response, ShellContext, json_response, run_shell
from aida.platform.web.wording import load_shell_wording
from tests.conftest import _ORIGINAL_HANDLE_REQUEST


@pytest.fixture(autouse=True)
def _permitir_servidor_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permite conexiones HTTP locales durante los tests del armazón."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


class DummyArm:
    """Brazo mínimo de prueba desacoplado de la aplicación de salud."""

    def __init__(self, arm_id: str = "dummy", name: str = "Brazo Prueba") -> None:
        self.id = arm_id
        self.name = name
        self.page_marker = "<html><body><h1>Brazo Prueba Activo</h1></body></html>"
        self._at_rest = True
        self._in_settings = False
        self.reset_called = 0
        self.rebuild_called = 0
        self.on_identified_calls: list[Professional] = []
        self.on_unlocked_called = 0
        self.get_calls: list[tuple[str, ShellContext]] = []
        self.post_calls: list[tuple[str, bytes | None, Mapping[str, str], ShellContext]] = []

    def page(self, ctx: ShellContext) -> str:
        if ctx.license.status in ("missing", "invalid"):
            return license_page(
                ctx.wording, ctx.brand, ctx.token, ctx.license, ctx.now, ctx.professional
            )
        if not ctx.license.allows(self.id) and self.at_rest():
            return readonly_page(
                ctx.wording,
                ctx.brand,
                ctx.token,
                ctx.license,
                ctx.now,
                ctx.professional,
                self.name,
                self.readonly_items(ctx),
            )
        return self.page_marker

    def get(self, path: str, ctx: ShellContext) -> Response | None:
        self.get_calls.append((path, ctx))
        if path == "ping":
            return json_response(200, {"pong": True})
        return None

    def post(
        self, path: str, body: bytes | None, headers: Mapping[str, str], ctx: ShellContext
    ) -> Response | None:
        self.post_calls.append((path, body, headers, ctx))
        if path == "accion-brazo":
            return json_response(200, {"ejecutado": True})
        return None

    def at_rest(self) -> bool:
        return self._at_rest

    def set_at_rest(self, value: bool) -> None:
        self._at_rest = value

    def in_settings(self) -> bool:
        return self._in_settings

    def reset(self) -> None:
        self.reset_called += 1

    def rebuild(self) -> None:
        self.rebuild_called += 1

    def on_identified(self, actor: Professional) -> None:
        self.on_identified_calls.append(actor)

    def on_unlocked(self) -> None:
        self.on_unlocked_called += 1

    def settings_cards(self, ctx: ShellContext) -> str:
        return '<div id="dummy-cards"><h3>Tarjetas Propias del Brazo</h3></div>'

    def readonly_items(self, ctx: ShellContext) -> tuple[str, ...]:
        return ("Ítem lectura A", "Ítem lectura B")


def _lanzar_armazon(
    arm: ArmApp,
    tmp_path: Path,
    *,
    profiles: ProfileStore | None = None,
    audit: AuditLog | None = None,
    license_state: LicenseState | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = datetime.now,
    today: Callable[[], date] = date.today,
) -> tuple[str, threading.Thread, ProfileStore, AuditLog]:
    wording = load_shell_wording()
    store = profiles if profiles is not None else ProfileStore(tmp_path / "profiles.json")
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
            "license_path": tmp_path / "licencia.json",
            "lock": lock,
            "open_browser": _abrir,
            "clock": clock,
            "now": now,
            "today": today,
        },
        daemon=True,
    )
    hilo.start()
    assert listo.wait(timeout=5.0), "el armazón no abrió la URL en 5s"
    return info["url"], hilo, store, auditoria


# ---------------------------------------------------------------------------
# B.2 §1 — El armazón con un brazo de prueba (sin salud)
# ---------------------------------------------------------------------------


def test_armazon_con_brazo_prueba_flujo_identificacion(tmp_path: Path) -> None:
    arm = DummyArm()
    store = ProfileStore(tmp_path / "profiles.json")
    auditoria = AuditLog(tmp_path / "audit.jsonl")

    # 1. Sin perfiles => first_profile
    url, hilo, _, _ = _lanzar_armazon(arm, tmp_path, profiles=store, audit=auditoria)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert "Su perfil" in resp.text

        # Crear primer perfil => entra directamente a la página del brazo
        resp_p = httpx.post(
            f"{url}/perfil",
            json={
                "name": "Dra. Primero",
                "registry": "REG-001",
                "pin": "1234",
                "pin2": "1234",
            },
            timeout=5.0,
        )
        assert resp_p.status_code == 200
        assert arm.page_marker in httpx.get(url, timeout=5.0).text
        assert len(arm.on_identified_calls) == 1
        assert arm.on_identified_calls[0].name == "Dra. Primero"

        opened = [x for x in auditoria.lines() if x["kind"] == "session_opened"]
        assert len(opened) == 1
        assert opened[0]["actor_name"] == "Dra. Primero (REG-001)"
        assert opened[0]["actor"] == arm.on_identified_calls[0].id
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)

    # 2. Con perfiles existentes => who y luego PIN correcto => página del brazo
    arm2 = DummyArm()
    store2 = ProfileStore(tmp_path / "profiles2.json")
    prof = store2.create(name="Dr. Segundo", registry="REG-002", pin="4321")
    auditoria2 = AuditLog(tmp_path / "audit2.jsonl")

    url2, hilo2, _, _ = _lanzar_armazon(arm2, tmp_path, profiles=store2, audit=auditoria2)
    try:
        resp_who = httpx.get(url2, timeout=5.0)
        assert resp_who.status_code == 200
        assert "¿Quién atiende?" in resp_who.text
        assert "Dr. Segundo" in resp_who.text

        # Entrar con PIN correcto
        resp_entrar = httpx.post(
            f"{url2}/entrar",
            json={"profile": prof.id, "pin": "4321"},
            timeout=5.0,
        )
        assert resp_entrar.status_code == 200
        assert arm2.page_marker in httpx.get(url2, timeout=5.0).text
        assert len(arm2.on_identified_calls) == 1
        assert arm2.on_identified_calls[0].id == prof.id

        opened2 = [x for x in auditoria2.lines() if x["kind"] == "session_opened"]
        assert len(opened2) == 1
        assert opened2[0]["actor"] == prof.id
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url2}/salir", timeout=5.0)
        hilo2.join(timeout=5.0)


# ---------------------------------------------------------------------------
# B.2 §2 — El bloqueo pregunta al brazo
# ---------------------------------------------------------------------------


def test_bloqueo_pregunta_al_brazo_en_reposo_y_ocupado(tmp_path: Path) -> None:
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
        httpx.post(f"{url}/entrar", json={"profile": prof.id, "pin": "1111"}, timeout=5.0)

        # 1. Con at_rest() falso => pasado el plazo NO bloquea
        arm.set_at_rest(False)
        reloj[0] += 15 * 60  # Pasaron 15 minutos (> idle_lock_minutes=10)
        resp_ocupado = httpx.get(url, timeout=5.0)
        assert resp_ocupado.status_code == 200
        assert arm.page_marker in resp_ocupado.text
        assert "AIDA está bloqueada" not in resp_ocupado.text
        assert arm.reset_called == 0
        assert not any(x["kind"] == "locked" for x in auditoria.lines())

        # 2. Con at_rest() verdadero => pasado el plazo bloquea, llama a reset() y audita locked
        arm.set_at_rest(True)
        reloj[0] += 15 * 60
        resp_reposo = httpx.get(url, timeout=5.0)
        assert resp_reposo.status_code == 200
        assert "AIDA está bloqueada" in resp_reposo.text
        assert arm.reset_called >= 1
        bloqueos = [x for x in auditoria.lines() if x["kind"] == "locked"]
        assert len(bloqueos) == 1
        assert bloqueos[0]["actor"] == prof.id
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# B.2 §3 — «Cambiar»
# ---------------------------------------------------------------------------


def test_cambiar_profesional_brazo_ocupado_y_reposo(tmp_path: Path) -> None:
    arm = DummyArm()
    store = ProfileStore(tmp_path / "profiles.json")
    prof = store.create(name="Dra. Test", registry="REG-TEST", pin="1111")
    auditoria = AuditLog(tmp_path / "audit.jsonl")

    url, hilo, _, _ = _lanzar_armazon(arm, tmp_path, profiles=store, audit=auditoria)
    try:
        httpx.post(f"{url}/entrar", json={"profile": prof.id, "pin": "1111"}, timeout=5.0)

        # 1. Ocupado => 409
        arm.set_at_rest(False)
        resp_409 = httpx.post(f"{url}/cambiar", timeout=5.0)
        assert resp_409.status_code == 409

        # 2. En reposo => 200, página who y session_closed en auditoría
        arm.set_at_rest(True)
        resp_ok = httpx.post(f"{url}/cambiar", timeout=5.0)
        assert resp_ok.status_code == 200
        assert arm.reset_called >= 1

        resp_page = httpx.get(url, timeout=5.0)
        assert resp_page.status_code == 200
        assert "¿Quién atiende?" in resp_page.text

        cerrados = [x for x in auditoria.lines() if x["kind"] == "session_closed"]
        assert len(cerrados) == 1
        assert cerrados[0]["actor"] == prof.id
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# B.2 §4 — Rutas
# ---------------------------------------------------------------------------


def test_rutas_desconocidas_sin_identificar_y_sin_prefijo_token(tmp_path: Path) -> None:
    arm = DummyArm()
    store = ProfileStore(tmp_path / "profiles.json")
    prof = store.create(name="Dr. Rutas", registry="REG-RUTAS", pin="1111")

    url, hilo, _, _ = _lanzar_armazon(arm, tmp_path, profiles=store)
    try:
        # 1. Sin identificarse => 401 y el brazo NO recibe la llamada
        r_no_auth_get = httpx.get(f"{url}/ping", timeout=5.0)
        assert r_no_auth_get.status_code == 401
        assert len(arm.get_calls) == 0

        r_no_auth_post = httpx.post(f"{url}/accion-brazo", timeout=5.0)
        assert r_no_auth_post.status_code == 401
        assert len(arm.post_calls) == 0

        # Identificarse
        httpx.post(f"{url}/entrar", json={"profile": prof.id, "pin": "1111"}, timeout=5.0)

        # 2. Ruta que el brazo no conoce => 404
        r_404_get = httpx.get(f"{url}/desconocida", timeout=5.0)
        assert r_404_get.status_code == 404

        r_404_post = httpx.post(f"{url}/accion-desconocida", timeout=5.0)
        assert r_404_post.status_code == 404

        # 3. El path llega sin el token al brazo
        r_ok_get = httpx.get(f"{url}/ping", timeout=5.0)
        assert r_ok_get.status_code == 200
        assert arm.get_calls[-1][0] == "ping"

        r_ok_post = httpx.post(f"{url}/accion-brazo", timeout=5.0)
        assert r_ok_post.status_code == 200
        assert arm.post_calls[-1][0] == "accion-brazo"
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# B.2 §5 — Licencia
# ---------------------------------------------------------------------------


def test_licencia_ausente_caducada_y_vigente(tmp_path: Path) -> None:
    # 1. Licencia ausente => pantalla de cargarla
    arm_miss = DummyArm()
    store_miss = ProfileStore(tmp_path / "prof_miss.json")
    p1 = store_miss.create(name="Dra. L1", registry="REG-L1", pin="1111")
    url1, hilo1, _, _ = _lanzar_armazon(
        arm_miss, tmp_path, profiles=store_miss, license_state=LicenseState(status="missing")
    )
    try:
        httpx.post(f"{url1}/entrar", json={"profile": p1.id, "pin": "1111"}, timeout=5.0)
        resp1 = httpx.get(url1, timeout=5.0)
        assert resp1.status_code == 200
        assert "AIDA no tiene una licencia válida" in resp1.text
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url1}/salir", timeout=5.0)
        hilo1.join(timeout=5.0)

    # 2. Caducada y en reposo => solo lectura con readonly_items del brazo
    arm_cad = DummyArm()
    store_cad = ProfileStore(tmp_path / "prof_cad.json")
    p2 = store_cad.create(name="Dra. L2", registry="REG-L2", pin="1111")
    lic_cad = License(
        id="cad",
        client="Prueba",
        arms=(arm_cad.id,),
        issued=date.today() - timedelta(days=40),
        expires=date.today() - timedelta(days=1),
    )
    url2, hilo2, _, _ = _lanzar_armazon(
        arm_cad,
        tmp_path,
        profiles=store_cad,
        license_state=LicenseState(status="expired", license=lic_cad, days_left=-1),
    )
    try:
        httpx.post(f"{url2}/entrar", json={"profile": p2.id, "pin": "1111"}, timeout=5.0)
        resp2 = httpx.get(url2, timeout=5.0)
        assert resp2.status_code == 200
        assert "Solo lectura" in resp2.text
        assert "Ítem lectura A" in resp2.text
        assert "Ítem lectura B" in resp2.text
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url2}/salir", timeout=5.0)
        hilo2.join(timeout=5.0)

    # 3. Vigente => la del brazo
    arm_vig = DummyArm()
    store_vig = ProfileStore(tmp_path / "prof_vig.json")
    p3 = store_vig.create(name="Dra. L3", registry="REG-L3", pin="1111")
    url3, hilo3, _, _ = _lanzar_armazon(arm_vig, tmp_path, profiles=store_vig)
    try:
        httpx.post(f"{url3}/entrar", json={"profile": p3.id, "pin": "1111"}, timeout=5.0)
        resp3 = httpx.get(url3, timeout=5.0)
        assert resp3.status_code == 200
        assert arm_vig.page_marker in resp3.text
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url3}/salir", timeout=5.0)
        hilo3.join(timeout=5.0)


# ---------------------------------------------------------------------------
# B.2 §6 — Ajustes
# ---------------------------------------------------------------------------


def test_ajustes_marco_plataforma_y_tarjetas_del_brazo() -> None:
    wording = load_shell_wording()
    prof = Professional(id="p1", name="Dra. Laura Gómez", registry="52.341.889")
    lic = LicenseState(
        status="valid",
        license=License(
            id="lic-1",
            client="Centro Aurora",
            arms=("dummy",),
            issued=date.today(),
            expires=date.today() + timedelta(days=30),
        ),
        days_left=30,
    )
    ctx = ShellContext(
        token="tok-prueba",
        brand="AIDA Test",
        now=datetime.now(),
        professional=prof,
        license=lic,
        audit=AuditLog(),
        wording=wording,
        lock=threading.Lock(),
        profiles=(prof,),
    )
    arm = DummyArm()
    tarjetas_brazo = arm.settings_cards(ctx)

    html = settings_frame(
        wording,
        "AIDA Test",
        "tok-prueba",
        datetime.now(),
        cards=tarjetas_brazo,
        script="",
        professional=prof,
        license=lic,
    )
    assert wording["title_settings"] in html
    assert "Tarjetas Propias del Brazo" in html  # Tarjeta del brazo
    assert wording["back_label"] in html  # Volver al inicio


# ---------------------------------------------------------------------------
# B.2 §7 — Textos
# ---------------------------------------------------------------------------


def test_textos_carga_validacion_y_claves_duplicadas(tmp_path: Path) -> None:
    # 1. load_shell_wording revienta si falta una clave
    yaml_incompleto = tmp_path / "shell_incompleto.yaml"
    yaml_incompleto.write_text("shell:\n  who_title: Incompleto\n", encoding="utf-8")
    with pytest.raises(ValueError, match="faltan claves"):
        load_shell_wording(yaml_incompleto)

    # 2. load_app_wording de salud resuelve una clave de la plataforma y una suya
    w_salud = load_app_wording()
    assert w_salud["who_title"] == "¿Quién atiende?"
    assert w_salud["title_start"] != ""

    # 3. Una clave en los dos archivos => error de carga
    yaml_duplicado = tmp_path / "health_dup.yaml"
    # Cargamos el original de consultorio y le inyectamos una clave de plataforma
    original_consultorio = Path("packs/health/consultorio/ui.yaml").read_text(encoding="utf-8")
    datos = yaml.safe_load(original_consultorio)
    datos["app"]["who_title"] = "Duplicada"
    yaml_duplicado.write_text(yaml.dump(datos), encoding="utf-8")

    with pytest.raises(ValueError, match="claves de plataforma en la sección 'app'"):
        load_app_wording(yaml_duplicado)


# ---------------------------------------------------------------------------
# B.2 §8 — Independencia
# ---------------------------------------------------------------------------


def test_independencia_platform_web_no_importa_aida_arms() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "src" / "aida" / "platform" / "web"
    sources = list(web_dir.glob("*.py"))
    assert sources, f"no se encontraron fuentes en {web_dir}"

    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("aida.arms"), (
                        f"{source.name}:{node.lineno} importa {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("aida.arms"), (
                    f"{source.name}:{node.lineno} importa de {node.module}"
                )
