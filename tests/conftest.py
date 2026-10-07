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
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(tmp_path_factory.mktemp("config-aida")))
