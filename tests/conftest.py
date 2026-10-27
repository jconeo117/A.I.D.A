"""Lo que vale para toda la batería.

La batería no sale a la red. Hasta el plan 79 su resultado dependía de qué
hubiera encendido en la máquina: con un Ollama local en marcha, cuatro tests de
`test_run.py` recibían prosa de un modelo donde esperaban el respaldo literal, y
la batería pasaba de 50 s a 18 min. Limpiar variables de entorno no bastaba:
borrar `OLLAMA_HOST` apunta justo al Ollama de `localhost`.

Se corta en el transporte real de `httpx` y no más arriba: los tests que simulan
un proveedor con `httpx.MockTransport` no pasan por él y siguen funcionando. Y
un `ConnectError` es lo que los proveedores ya tratan como «no responde».
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

_ORIGINAL_HANDLE_REQUEST = httpx.HTTPTransport.handle_request


@pytest.fixture(autouse=True)
def _sin_red(monkeypatch: pytest.MonkeyPatch) -> None:
    def _rechazar(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"la batería no sale a la red: {request.url}", request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _rechazar)


@pytest.fixture(autouse=True)
def _sin_claves_del_equipo(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """La batería no lee las claves de quien la corre (plan 78). Desde que AIDA las
    guarda en `~/.config/aida`, un equipo con `stt.key` hacía que los tests que
    esperan «sin clave de nube» la encontraran; y sin clave de caché, AIDA la
    crearía en la carpeta real."""
    data_dir = tmp_path_factory.mktemp("data-aida")
    repo_root = Path(__file__).resolve().parents[1]
    (data_dir / "packs").symlink_to(repo_root / "packs")
    (data_dir / "data").symlink_to(repo_root / "data")
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(tmp_path_factory.mktemp("config-aida")))
    monkeypatch.setenv("AIDA_DATA_DIR", str(data_dir))
    # En Windows `user_dir` lee estas dos del entorno, y un test que cambia `Path.home`
    # seguiría escribiendo en las carpetas reales de quien corre la batería (78b-fix).
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)


@pytest.fixture(scope="session")
def _par_de_prueba() -> tuple[str, object]:
    """Una pareja Ed25519 solo para la batería (plan 100): la clave pública de verdad
    la escribe `tools/licencias.py keygen`, y no hay modo desarrollo que la salte."""
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    privada = Ed25519PrivateKey.generate()
    publica = privada.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return base64.b64encode(publica).decode(), privada


@pytest.fixture(autouse=True)
def _licencia_de_prueba(
    monkeypatch: pytest.MonkeyPatch,
    _sin_claves_del_equipo: None,
    _par_de_prueba: tuple[str, object],
) -> None:
    """Toda la batería corre con una licencia de salud vigente, firmada con la clave de
    prueba, en la carpeta de configuración temporal del test. Un test que quiera otra
    situación (sin licencia, caducada) la sobrescribe."""
    import base64
    import json
    import os
    from datetime import date, timedelta
    from pathlib import Path

    from aida.platform import _public_key
    from aida.platform.license import canonical

    publica, privada = _par_de_prueba
    monkeypatch.setattr(_public_key, "PUBLIC_KEY", publica)
    payload: dict[str, object] = {
        "id": "prueba",
        "client": "Batería",
        "arms": ["health"],
        "issued": date.today().isoformat(),
        "expires": (date.today() + timedelta(days=365)).isoformat(),
    }
    firma = privada.sign(canonical(payload))  # type: ignore[attr-defined]
    destino = Path(os.environ["AIDA_CONFIG_DIR"]) / "licencia.json"
    destino.write_text(
        json.dumps({"payload": payload, "signature": base64.b64encode(firma).decode()}),
        encoding="utf-8",
    )
