"""Verificación de aislamiento de red de la suite de pruebas (§B.8, Caso 15)."""

from __future__ import annotations

import httpx
import pytest


def test_conftest_sin_red_bloquea_peticiones_http() -> None:
    """Caso 15 (§B.8): httpx.Client().get(...) lanza httpx.ConnectError con el
    mensaje de la batería no sale a la red."""
    with pytest.raises(httpx.ConnectError) as exc_info:
        httpx.Client().get("http://127.0.0.1:1/")
    assert "la batería no sale a la red" in str(exc_info.value)
