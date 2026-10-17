"""Tests de la plataforma: núcleo, brazos y licencia (Plan 100).

Cubre §B.2 de .claude/workflow/100-plan-plataforma.md:
3. read_license: buena, fronteras de caducidad, ayer, manipulada, otra clave, json roto,
   PUBLIC_KEY=None.
4. allows y licensed: fuera de licencia, caducada, missing, invalid.
5. canonical: determinismo con claves en distinto orden.
6. discover: orden por id, validación de manifest (app sin : o faltante), entry() importa.
7. El núcleo no importa brazos: análisis AST de src/aida/platform/.
8. La aplicación: sin licencia (403/licencia), caducada (solo lectura, 403, ajustes 200),
   expiring (aviso cabecera), POST /licencia rechaza inválida/caducada sin pisar, acepta
   buena con 0o600 y auditoría.
9. La CLI: --consultation y --mic sin licencia (1, missing, 0 STT) y caducada (1, expired).
10. _open_platform: abre salud con licencia de prueba; falla con 1 sin brazos instalados.
11. tools/licencias.py: keygen escribe privada 0600 y pública en MODULO redirigido,
    se niega en RAIZ y no pisa; sign + show dan valid.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import importlib
import json
import os
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aida import run
from aida.arms.health.app import AppWording, load_app_wording, serve
from aida.arms.health.review_ui import load_ui_settings
from aida.platform import _public_key
from aida.platform.arms import ArmManifest, discover, load_manifest
from aida.platform.audit import AuditLog
from aida.platform.identity import Professional, ProfileStore
from aida.platform.license import (
    WARN_DAYS,
    License,
    LicenseState,
    canonical,
    read_license,
)
from tests.conftest import _ORIGINAL_HANDLE_REQUEST
from tests.test_health_app import _crear_app_ready, _dummy_vad
from tests.test_run import _crear_perfil_cli, _pack_yaml_valido

licencias: Any = importlib.import_module("tools.licencias")


@pytest.fixture(autouse=True)
def _permitir_servidor_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permite conexiones HTTP locales durante los tests de la aplicación."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


def _firmar_licencia(
    privada: Any,
    *,
    id_lic: str = "lic-1",
    client: str = "Centro Médico Aurora",
    arms: tuple[str, ...] = ("health",),
    issued: date | None = None,
    expires: date,
) -> str:
    payload: dict[str, object] = {
        "id": id_lic,
        "client": client,
        "arms": list(arms),
        "issued": (issued or date.today()).isoformat(),
        "expires": expires.isoformat(),
    }
    firma = privada.sign(canonical(payload))
    return json.dumps(
        {
            "payload": payload,
            "signature": base64.b64encode(firma).decode(),
        }
    )


# ---------------------------------------------------------------------------
# §3 — read_license
# ---------------------------------------------------------------------------


def test_read_license_valida_y_fronteras_caducidad(
    _par_de_prueba: tuple[str, object],
) -> None:
    pub_b64, priv = _par_de_prueba
    hoy = date.today()

    # Buena (365 días)
    texto_buena = _firmar_licencia(priv, expires=hoy + timedelta(days=365))
    st = read_license(texto_buena, today=hoy, public_key=pub_b64)
    assert st.status == "valid"
    assert st.days_left == 365
    assert st.license is not None
    assert st.license.client == "Centro Médico Aurora"

    # A WARN_DAYS (15) días => expiring
    texto_warn = _firmar_licencia(priv, expires=hoy + timedelta(days=WARN_DAYS))
    st_warn = read_license(texto_warn, today=hoy, public_key=pub_b64)
    assert st_warn.status == "expiring"
    assert st_warn.days_left == WARN_DAYS

    # A WARN_DAYS + 1 (16) días => valid
    texto_warn_plus = _firmar_licencia(priv, expires=hoy + timedelta(days=WARN_DAYS + 1))
    st_plus = read_license(texto_warn_plus, today=hoy, public_key=pub_b64)
    assert st_plus.status == "valid"
    assert st_plus.days_left == WARN_DAYS + 1


def test_read_license_caducada_ayer(_par_de_prueba: tuple[str, object]) -> None:
    pub_b64, priv = _par_de_prueba
    hoy = date.today()

    # Vencida ayer => expired con days_left < 0
    texto_cad = _firmar_licencia(priv, expires=hoy - timedelta(days=1))
    st_cad = read_license(texto_cad, today=hoy, public_key=pub_b64)
    assert st_cad.status == "expired"
    assert st_cad.days_left == -1
    assert st_cad.days_left < 0
    assert st_cad.license is not None


def test_read_license_campo_cambiado_y_otra_clave(
    _par_de_prueba: tuple[str, object],
) -> None:
    pub_b64, priv = _par_de_prueba
    hoy = date.today()

    # Un campo cambiado (tampered) => invalid
    texto_buena = _firmar_licencia(priv, expires=hoy + timedelta(days=30))
    datos = json.loads(texto_buena)
    datos["payload"]["expires"] = (hoy + timedelta(days=999)).isoformat()
    st_mod = read_license(json.dumps(datos), today=hoy, public_key=pub_b64)
    assert st_mod.status == "invalid"
    assert st_mod.license is None

    # Firmada con otra clave => invalid
    otra_priv = Ed25519PrivateKey.generate()
    texto_otra = _firmar_licencia(otra_priv, expires=hoy + timedelta(days=30))
    st_otra = read_license(texto_otra, today=hoy, public_key=pub_b64)
    assert st_otra.status == "invalid"


def test_read_license_json_roto_o_sin_firma(_par_de_prueba: tuple[str, object]) -> None:
    pub_b64, _ = _par_de_prueba
    hoy = date.today()

    assert read_license("no-es-json", today=hoy, public_key=pub_b64).status == "invalid"
    assert read_license("{}", today=hoy, public_key=pub_b64).status == "invalid"
    assert (
        read_license(
            json.dumps({"payload": {"id": "1"}, "signature": "abc"}),
            today=hoy,
            public_key=pub_b64,
        ).status
        == "invalid"
    )
    assert (
        read_license(
            json.dumps({"payload": "no-es-dict", "signature": "abc"}),
            today=hoy,
            public_key=pub_b64,
        ).status
        == "invalid"
    )


def test_read_license_public_key_none(
    monkeypatch: pytest.MonkeyPatch, _par_de_prueba: tuple[str, object]
) -> None:
    _, priv = _par_de_prueba
    hoy = date.today()
    texto = _firmar_licencia(priv, expires=hoy + timedelta(days=30))

    # PUBLIC_KEY = None => invalid siempre
    monkeypatch.setattr(_public_key, "PUBLIC_KEY", None)
    st = read_license(texto, today=hoy, public_key=None)
    assert st.status == "invalid"


# ---------------------------------------------------------------------------
# §4 — allows y licensed
# ---------------------------------------------------------------------------


def test_license_state_allows_and_licensed() -> None:
    hoy = date.today()
    lic_h = License(
        id="1", client="C", arms=("health",), issued=hoy, expires=hoy + timedelta(days=10)
    )

    # Brazo fuera de la licencia => ni allows ni licensed
    lic_otro = License(
        id="2", client="C", arms=("dental",), issued=hoy, expires=hoy + timedelta(days=10)
    )
    st_otro = LicenseState(status="valid", license=lic_otro)
    assert st_otro.allows("health") is False
    assert st_otro.licensed("health") is False

    # Caducada => licensed sí, allows no
    lic_cad = License(
        id="3",
        client="C",
        arms=("health",),
        issued=hoy - timedelta(days=20),
        expires=hoy - timedelta(days=1),
    )
    st_cad = LicenseState(status="expired", license=lic_cad, days_left=-1)
    assert st_cad.licensed("health") is True
    assert st_cad.allows("health") is False

    # missing => ninguno
    st_missing = LicenseState(status="missing")
    assert st_missing.allows("health") is False
    assert st_missing.licensed("health") is False

    # invalid => ninguno
    st_inv = LicenseState(status="invalid")
    assert st_inv.allows("health") is False
    assert st_inv.licensed("health") is False

    # valid / expiring => ambos True
    st_val = LicenseState(status="valid", license=lic_h, days_left=10)
    assert st_val.allows("health") is True
    assert st_val.licensed("health") is True

    st_exp = LicenseState(status="expiring", license=lic_h, days_left=5)
    assert st_exp.allows("health") is True
    assert st_exp.licensed("health") is True


# ---------------------------------------------------------------------------
# §5 — canonical
# ---------------------------------------------------------------------------


def test_canonical_determinismo() -> None:
    d1 = {"z": 10, "a": [3, 2, 1], "m": {"nested_z": 1, "nested_a": 2}}
    d2 = {"m": {"nested_a": 2, "nested_z": 1}, "a": [3, 2, 1], "z": 10}
    assert canonical(d1) == canonical(d2)


# ---------------------------------------------------------------------------
# §6 — discover
# ---------------------------------------------------------------------------


def test_discover_orden_por_id_y_manifiestos(tmp_path: Path) -> None:
    packs = tmp_path / "packs"
    arm_b = packs / "brazo_b"
    arm_b.mkdir(parents=True)
    (arm_b / "arm.yaml").write_text(
        'id: "z_salud"\nname: "Aida Z"\ndescription: "Desc"\nversion: "1.0"\napp: "math:sqrt"\n',
        encoding="utf-8",
    )

    arm_a = packs / "brazo_a"
    arm_a.mkdir(parents=True)
    (arm_a / "arm.yaml").write_text(
        'id: "a_salud"\nname: "Zeta Arm"\ndescription: "Desc"\nversion: "1.0"\napp: "math:sin"\n',
        encoding="utf-8",
    )

    brazos = discover(tmp_path)
    assert len(brazos) == 2
    assert [b.id for b in brazos] == ["a_salud", "z_salud"]


def test_load_manifest_validaciones(tmp_path: Path) -> None:
    # Sin app
    p1 = tmp_path / "sin_app.yaml"
    p1.write_text('id: "x"\nname: "X"\ndescription: "D"\nversion: "1.0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="necesita"):
        load_manifest(p1)

    # Sin ':' en app
    p2 = tmp_path / "sin_dos_puntos.yaml"
    p2.write_text(
        'id: "x"\nname: "X"\ndescription: "D"\nversion: "1.0"\napp: "modulo_solo"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'app' debe ser 'modulo:funcion'"):
        load_manifest(p2)


def test_arm_manifest_entry() -> None:
    import math

    manif = ArmManifest(id="h", name="H", description="D", version="1", app="math:cos")
    assert manif.entry() is math.cos


# ---------------------------------------------------------------------------
# §7 — El núcleo no importa brazos
# ---------------------------------------------------------------------------


def test_nucleo_no_importa_brazos() -> None:
    platform_dir = Path(__file__).resolve().parents[1] / "src" / "aida" / "platform"
    archivos = list(platform_dir.glob("*.py"))
    assert archivos, "no se encontraron archivos en aida/platform"

    for py_file in archivos:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("aida.arms"), (
                        f"{py_file.name} importa {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                modulo = node.module or ""
                assert not modulo.startswith("aida.arms"), (
                    f"{py_file.name} importa {modulo}"
                )


# ---------------------------------------------------------------------------
# §8 — La aplicación
# ---------------------------------------------------------------------------


def _lanzar_servidor_prueba(
    tmp_path: Path,
    *,
    license_state: LicenseState,
    license_file: Path | None = None,
    wording: AppWording | None = None,
    profiles: ProfileStore | None = None,
    audit: AuditLog | None = None,
) -> tuple[str, threading.Thread, ProfileStore, AuditLog]:
    ajustes = load_ui_settings()
    palabras = wording if wording is not None else load_app_wording()
    ready_instance = _crear_app_ready(tmp_path)
    store = profiles or ProfileStore(tmp_path / "profiles.json")
    store.create(name="Dra. Laura Gómez", registry="52.341.889", pin="1234")
    auditoria = audit or AuditLog(tmp_path / "audit.jsonl")

    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    lic_path = license_file or (tmp_path / "licencia.json")

    hilo = threading.Thread(
        target=serve,
        args=(lambda: ready_instance,),
        kwargs={
            "settings": ajustes,
            "wording": palabras,
            "open_browser": _abrir,
            "vad_factory": lambda: _dummy_vad,
            "tools": run._app_tools(),
            "profiles": store,
            "audit": auditoria,
            "license": license_state,
            "license_path": lic_path,
            "arm_id": "health",
            "arm_name": "Consultorio",
        },
        daemon=True,
    )
    hilo.start()
    assert listo.wait(timeout=5.0), "el servidor no abrió la URL en 5s"
    url = info["url"]

    # Identificarse como Dra. Laura Gómez
    prof_id = store.professionals()[0].id
    resp = httpx.post(f"{url}/entrar", json={"profile": prof_id, "pin": "1234"}, timeout=5.0)
    assert resp.status_code == 200

    return url, hilo, store, auditoria


def test_app_sin_licencia_muestra_pagina_y_empezar_403(tmp_path: Path) -> None:
    url, hilo, _, _ = _lanzar_servidor_prueba(
        tmp_path, license_state=LicenseState(status="missing")
    )
    try:
        # GET / muestra la pantalla de Licencia
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert "<title>Licencia</title>" in resp.text

        # /empezar devuelve 403
        r_emp = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-0042", "recorded": False, "consent": True},
            timeout=5.0,
        )
        assert r_emp.status_code == 403
        assert "La licencia no permite empezar consultas nuevas" in r_emp.text
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_app_licencia_caducada_solo_lectura_empezar_403_ajustes_200(
    tmp_path: Path,
) -> None:
    hoy = date.today()
    lic_cad = License(
        id="cad",
        client="Centro Médico Aurora",
        arms=("health",),
        issued=hoy - timedelta(days=40),
        expires=hoy - timedelta(days=1),
    )
    url, hilo, _, _ = _lanzar_servidor_prueba(
        tmp_path,
        license_state=LicenseState(status="expired", license=lic_cad, days_left=-1),
    )
    try:
        # GET / muestra pantalla en solo lectura
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert "<title>Consultorio</title>" in resp.text
        assert "Solo lectura" in resp.text

        # /empezar devuelve 403
        r_emp = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-0042", "recorded": False, "consent": True},
            timeout=5.0,
        )
        assert r_emp.status_code == 403

        # /ajustes devuelve 200
        r_aj = httpx.post(f"{url}/ajustes", content=b"", timeout=5.0)
        assert r_aj.status_code == 200
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_app_licencia_expiring_aviso_cabecera(tmp_path: Path) -> None:
    hoy = date.today()
    lic_exp = License(
        id="exp",
        client="Centro Médico Aurora",
        arms=("health",),
        issued=hoy - timedelta(days=10),
        expires=hoy + timedelta(days=9),
    )
    url, hilo, _, _ = _lanzar_servidor_prueba(
        tmp_path,
        license_state=LicenseState(status="expiring", license=lic_exp, days_left=9),
    )
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert "head-warn" in resp.text
        assert "9 días" in resp.text
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_app_post_licencia(
    tmp_path: Path, _par_de_prueba: tuple[str, object]
) -> None:
    _, priv = _par_de_prueba
    hoy = date.today()
    lic_path = tmp_path / "config" / "licencia.json"

    url, hilo, _, auditoria = _lanzar_servidor_prueba(
        tmp_path,
        license_state=LicenseState(status="missing"),
        license_file=lic_path,
    )
    try:
        # 1. Inválida / manipulada => 400 y el archivo no se crea
        resp_inv = httpx.post(f"{url}/licencia", content=b"cuerpo-invalido", timeout=5.0)
        assert resp_inv.status_code == 400
        assert not lic_path.exists()

        # 2. Caducada => 400 y el archivo no se crea
        texto_cad = _firmar_licencia(priv, expires=hoy - timedelta(days=5))
        resp_cad = httpx.post(
            f"{url}/licencia", content=texto_cad.encode("utf-8"), timeout=5.0
        )
        assert resp_cad.status_code == 400
        assert not lic_path.exists()

        # 3. Buena => 200, archivo escrito con 0o600 y auditoría con el profesional
        texto_buena = _firmar_licencia(priv, expires=hoy + timedelta(days=60))
        resp_ok = httpx.post(
            f"{url}/licencia", content=texto_buena.encode("utf-8"), timeout=5.0
        )
        assert resp_ok.status_code == 200
        assert lic_path.exists()
        assert (lic_path.stat().st_mode & 0o777) == 0o600

        # Verificación en auditoría
        lineas = [x for x in auditoria.lines() if x["kind"] == "license_loaded"]
        assert len(lineas) == 1
        assert "Dra. Laura Gómez" in lineas[0]["actor_name"]
        assert "52.341.889" in lineas[0]["actor_name"]
        assert json.loads(lineas[0]["data"])["client"] == "Centro Médico Aurora"
    finally:
        with contextlib.suppress(Exception):
            httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


# ---------------------------------------------------------------------------
# §9 — La CLI
# ---------------------------------------------------------------------------


def test_cli_consultation_licencia_requerida(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _par_de_prueba: tuple[str, object],
) -> None:
    _, priv = _par_de_prueba
    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")
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
        lambda: (_CountingSTT(), MagicMock()),
    )
    _crear_perfil_cli(monkeypatch)

    lic_config = Path(os.environ["AIDA_CONFIG_DIR"]) / "licencia.json"

    # 1. Sin licencia => exit 1, (missing), 0 llamadas a STT
    if lic_config.exists():
        lic_config.unlink()

    codigo = run.main(
        [
            "--consultation",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            "pac_lic_cli",
            "--consent",
            "--as",
            "52.341.889",
        ]
    )
    assert codigo == 1
    cap = capsys.readouterr()
    assert "(missing)" in cap.err
    assert stt_calls == 0

    # 2. Caducada => exit 1, (expired), 0 llamadas a STT
    texto_cad = _firmar_licencia(priv, expires=date.today() - timedelta(days=2))
    lic_config.write_text(texto_cad, encoding="utf-8")

    codigo_cad = run.main(
        [
            "--consultation",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            "pac_lic_cli",
            "--consent",
            "--as",
            "52.341.889",
        ]
    )
    assert codigo_cad == 1
    cap_cad = capsys.readouterr()
    assert "(expired)" in cap_cad.err
    assert stt_calls == 0


def test_cli_mic_history_licencia_requerida(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _par_de_prueba: tuple[str, object],
) -> None:
    _, priv = _par_de_prueba
    monkeypatch.setattr(run, "_check_mic_dependencies", lambda: True)

    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")
    hist_path = tmp_path / "historia.txt"
    hist_path.write_text("Historia clínica.")
    _crear_perfil_cli(monkeypatch)

    lic_config = Path(os.environ["AIDA_CONFIG_DIR"]) / "licencia.json"

    # 1. Sin licencia => exit 1, (missing)
    if lic_config.exists():
        lic_config.unlink()

    codigo = run.main(
        [
            "--pack",
            str(pack_file),
            "--converse",
            "--mic",
            "--history",
            str(hist_path),
            "--owner",
            "pac_mic_lic",
            "--consent",
            "--as",
            "52.341.889",
        ]
    )
    assert codigo == 1
    cap = capsys.readouterr()
    assert "(missing)" in cap.err

    # 2. Caducada => exit 1, (expired)
    texto_cad = _firmar_licencia(priv, expires=date.today() - timedelta(days=2))
    lic_config.write_text(texto_cad, encoding="utf-8")

    codigo_cad = run.main(
        [
            "--pack",
            str(pack_file),
            "--converse",
            "--mic",
            "--history",
            str(hist_path),
            "--owner",
            "pac_mic_lic",
            "--consent",
            "--as",
            "52.341.889",
        ]
    )
    assert codigo_cad == 1
    cap_cad = capsys.readouterr()
    assert "(expired)" in cap_cad.err


# ---------------------------------------------------------------------------
# §10 — _open_platform
# ---------------------------------------------------------------------------


def test_open_platform_sin_brazos(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aida.platform.arms.discover", lambda: ())
    parser = run._construir_parser()
    args = parser.parse_args(["--app"])
    code = run._open_platform(args)
    assert code == 1
    assert "no hay ningún brazo instalado" in capsys.readouterr().err


def test_open_platform_con_licencia(monkeypatch: pytest.MonkeyPatch) -> None:
    llamadas: list[Any] = []
    fake_arm = ArmManifest(
        id="health",
        name="Salud",
        description="Desc",
        version="1",
        app="aida.run:launch_health",
    )

    def _simulado(args: Any, estado: Any, arm: Any) -> int:
        llamadas.append((args, estado, arm))
        return 77

    monkeypatch.setattr(run, "launch_health", _simulado)
    monkeypatch.setattr("aida.platform.arms.discover", lambda: (fake_arm,))

    parser = run._construir_parser()
    args = parser.parse_args(["--app"])
    code = run._open_platform(args)
    assert code == 77
    assert len(llamadas) == 1
    assert llamadas[0][2].id == "health"


# ---------------------------------------------------------------------------
# §11 — tools/licencias.py
# ---------------------------------------------------------------------------


def test_tools_licencias_keygen_sign_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod_redir = tmp_path / "_public_key.py"
    monkeypatch.setattr(licencias, "MODULO", mod_redir)

    # 1. keygen se niega dentro de RAIZ
    dentro_raiz = licencias.RAIZ / "tmp_claves_test"
    res_dentro = licencias.keygen(dentro_raiz)
    assert res_dentro == 1
    err = capsys.readouterr().err
    assert "la clave privada no se guarda dentro del repositorio" in err

    # 2. keygen escribe privada 0600 y publica en MODULO
    dir_claves = tmp_path / "mis_claves"
    res_ok = licencias.keygen(dir_claves)
    assert res_ok == 0
    priv = dir_claves / "aida-licencias.key"
    assert priv.exists()
    assert (priv.stat().st_mode & 0o777) == 0o600
    assert mod_redir.exists()
    contenido_mod = mod_redir.read_text(encoding="utf-8")
    assert "PUBLIC_KEY" in contenido_mod

    # 3. keygen se niega a pisar
    res_pisar = licencias.keygen(dir_claves)
    assert res_pisar == 1
    err_pisar = capsys.readouterr().err
    assert "no se pisa" in err_pisar

    # 4. sign genera archivo de licencia
    lic_file = tmp_path / "mi_licencia.json"
    res_sign = licencias.sign(
        priv,
        client="Hospital Central",
        arms="health",
        expires=(date.today() + timedelta(days=60)).isoformat(),
        out=lic_file,
    )
    assert res_sign == 0
    assert lic_file.exists()

    # 5. show => valid
    m = re.search(r'PUBLIC_KEY: str \| None = "([^"]+)"', contenido_mod)
    assert m is not None
    pub_key_b64 = m.group(1)
    monkeypatch.setattr("aida.platform._public_key.PUBLIC_KEY", pub_key_b64)

    capsys.readouterr()  # limpiar stdout/stderr
    res_show = licencias.show(lic_file)
    assert res_show == 0
    out_show = capsys.readouterr().out
    assert "estado: valid" in out_show
    assert "Hospital Central" in out_show


def test_audit_purge_texts(tmp_path: Path) -> None:
    """B.2 §6: AuditLog.purge_texts: con dos revisiones guardadas y un corte posterior,
    borra los dos textos, deja UNA línea texts_purged con count = 2 y el autor;
    las líneas originales siguen; un corte anterior a todo => 0 y NINGUNA línea nueva;
    show de un borrado => ValueError con 'plazo de retención'.
    """
    key = Fernet.generate_key()
    blobs_dir = tmp_path / "blobs"
    log_path = tmp_path / "audit.jsonl"
    actor = Professional(id="prof_1", name="Dra. Gómez", registry="52.341.889")
    now_dt = datetime(2026, 10, 16, 12, 0, 0).astimezone()

    audit = AuditLog(log_path, blobs=blobs_dir, key=key, now=lambda: now_dt)

    l1 = audit.record("review_saved", actor=actor, texts={"nota": "primera consulta"})
    l2 = audit.record("review_saved", actor=actor, texts={"nota": "segunda consulta"})
    blob1, blob2 = l1["blob"], l2["blob"]
    assert blob1 and blob2
    assert (blobs_dir / f"{blob1}.bin").is_file()
    assert (blobs_dir / f"{blob2}.bin").is_file()
    assert audit.show(blob1) == {"nota": "primera consulta"}

    corte_anterior = now_dt - timedelta(days=1)
    borrados_0 = audit.purge_texts(corte_anterior, actor=actor)
    assert borrados_0 == 0
    assert len(audit.lines()) == 2
    assert not any(linea["kind"] == "texts_purged" for linea in audit.lines())

    corte_posterior = now_dt + timedelta(minutes=1)
    borrados_2 = audit.purge_texts(corte_posterior, actor=actor)
    assert borrados_2 == 2
    assert not (blobs_dir / f"{blob1}.bin").exists()
    assert not (blobs_dir / f"{blob2}.bin").exists()

    lineas = audit.lines()
    assert len(lineas) == 3
    assert lineas[0]["kind"] == "review_saved"
    assert lineas[1]["kind"] == "review_saved"
    assert lineas[2]["kind"] == "texts_purged"
    assert lineas[2]["actor"] == actor.id
    datos_purga = json.loads(lineas[2]["data"])
    assert datos_purga["count"] == "2"

    with pytest.raises(ValueError, match="ese texto se borró al cumplir su plazo de retención"):
        audit.show(blob1)


def test_app_retention_days_en_serve(tmp_path: Path) -> None:
    """B.2 §7: serve(…, retention_days=0): al entrar con PIN, el borrado corre
    y lo firma ese profesional; con retention_days=None, no corre.
    """
    key = Fernet.generate_key()
    blobs = tmp_path / "blobs"
    audit_file = tmp_path / "audit.jsonl"
    actor_previo = Professional(id="prev", name="Previo", registry="000")
    audit = AuditLog(audit_file, blobs=blobs, key=key)
    l1 = audit.record("review_saved", actor=actor_previo, texts={"dato": "viejo"})
    huella_previa = l1["blob"]
    assert (blobs / f"{huella_previa}.bin").exists()

    store = ProfileStore(tmp_path / "profiles.json")
    store.create(name="Dr. Juan", registry="12345", pin="9999")

    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir1(u: str) -> None:
        info["url"] = u
        listo.set()

    ready = _crear_app_ready(tmp_path)
    lic_valida = LicenseState(
        status="valid",
        license=License("l", "c", ("health",), date.today(), date.today()),
    )
    hilo = threading.Thread(
        target=serve,
        args=(lambda: ready,),
        kwargs={
            "settings": load_ui_settings(),
            "wording": load_app_wording(),
            "open_browser": _abrir1,
            "vad_factory": lambda: _dummy_vad,
            "profiles": store,
            "audit": audit,
            "license": lic_valida,
            "retention_days": 0,
        },
        daemon=True,
    )
    hilo.start()
    listo.wait(timeout=5.0)
    url = info["url"]
    try:
        prof_id = store.professionals()[0].id
        r_entrar = httpx.post(
            f"{url}/entrar", json={"profile": prof_id, "pin": "9999"}, timeout=5.0
        )
        assert r_entrar.status_code == 200

        assert not (blobs / f"{huella_previa}.bin").exists()
        lineas_purged = [linea for linea in audit.lines() if linea["kind"] == "texts_purged"]
        assert len(lineas_purged) == 1
        assert lineas_purged[0]["actor"] == prof_id
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)

    # Con retention_days=None no corre
    blobs2 = tmp_path / "blobs2"
    audit2 = AuditLog(tmp_path / "audit2.jsonl", blobs=blobs2, key=key)
    l2 = audit2.record("review_saved", actor=actor_previo, texts={"dato": "viejo2"})
    h2 = l2["blob"]
    assert (blobs2 / f"{h2}.bin").exists()

    store2 = ProfileStore(tmp_path / "profiles2.json")
    store2.create(name="Dr. Pedro", registry="67890", pin="1111")
    info2: dict[str, Any] = {}
    listo2 = threading.Event()

    def _abrir2(u: str) -> None:
        info2["url"] = u
        listo2.set()

    hilo2 = threading.Thread(
        target=serve,
        args=(lambda: ready,),
        kwargs={
            "settings": load_ui_settings(),
            "wording": load_app_wording(),
            "open_browser": _abrir2,
            "vad_factory": lambda: _dummy_vad,
            "profiles": store2,
            "audit": audit2,
            "license": lic_valida,
            "retention_days": None,
        },
        daemon=True,
    )
    hilo2.start()
    listo2.wait(timeout=5.0)
    url2 = info2["url"]
    try:
        prof_id2 = store2.professionals()[0].id
        r_entrar2 = httpx.post(
            f"{url2}/entrar", json={"profile": prof_id2, "pin": "1111"}, timeout=5.0
        )
        assert r_entrar2.status_code == 200

        assert (blobs2 / f"{h2}.bin").exists()
        assert not any(linea["kind"] == "texts_purged" for linea in audit2.lines())
    finally:
        httpx.post(f"{url2}/salir", timeout=5.0)
        hilo2.join(timeout=5.0)


def test_tools_licencias_show_archivo_inexistente(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """B.2 §10: tools/licencias.py show con un archivo que no existe => 1 y 'no existe'."""
    archivo_inexistente = tmp_path / "no_existe_para_nada.json"
    rc = licencias.show(archivo_inexistente)
    assert rc == 1
    err = capsys.readouterr().err
    assert f"error: no existe {archivo_inexistente}" in err

