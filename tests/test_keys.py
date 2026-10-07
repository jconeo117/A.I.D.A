"""Tests de gestión y validación de claves (plan 78, §B.2).

Cubre:
6. cache_key() sin variable crea cache.key con permisos 0o600 y es idempotente.
7. cache_key(create=False) con carpeta vacía devuelve None sin crear archivo.
8. AIDA_CONSULT_CACHE_KEY tiene prioridad absoluta sobre el archivo; cache_key_created() es None.
9. cloud_key() y cloud_key_source(): AIDA_STT_API_KEY > OPENAI_API_KEY > stt.key > None.
10. save_cloud_key limpia espacios y guarda stt.key con permisos 0o600.
11. check_cloud_key con httpx.get simulado: 200, 401, ConnectError; comprueba cabecera
    Authorization y ruta /v1/models.
12. build_provider("openai", ...) construye el proveedor leyendo la clave solo desde stt.key.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from cryptography.fernet import Fernet

from aida.cognition.providers import OpenAIProvider, build_provider
from aida.keys import (
    cache_key,
    cache_key_created,
    check_cloud_key,
    cloud_key,
    cloud_key_source,
    save_cloud_key,
)


def test_cache_key_crea_archivo_con_permisos_y_reutiliza(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 6 (§B.2, M4): cache_key() sin variable y con la carpeta vacía devuelve
    una clave Fernet válida, crea cache.key con permisos 0o600, y una segunda llamada
    devuelve exactamente la misma clave.
    """
    config_dir = tmp_path / "config"
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("AIDA_CONSULT_CACHE_KEY", raising=False)

    key1 = cache_key()
    assert key1 is not None
    # Debe ser una clave Fernet válida
    Fernet(key1.encode())

    archivo = config_dir / "cache.key"
    assert archivo.is_file()
    assert archivo.read_text(encoding="utf-8").strip() == key1
    assert (archivo.stat().st_mode & 0o777) == 0o600

    # Segunda llamada devuelve exactamente la misma clave sin regenerar
    key2 = cache_key()
    assert key2 == key1


def test_cache_key_create_false_devuelve_none_y_no_crea(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 7 (§B.2): cache_key(create=False) con la carpeta vacía devuelve None
    y no crea ningún archivo.
    """
    config_dir = tmp_path / "config"
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("AIDA_CONSULT_CACHE_KEY", raising=False)

    key = cache_key(create=False)
    assert key is None
    assert not (config_dir / "cache.key").exists()


def test_cache_key_prioridad_entorno_sobre_archivo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 8 (§B.2): Con AIDA_CONSULT_CACHE_KEY fijada, cache_key() devuelve la
    variable aunque exista archivo en disco, y cache_key_created() es None.
    """
    config_dir = tmp_path / "config"
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))
    archivo = config_dir / "cache.key"
    config_dir.mkdir(parents=True, exist_ok=True)
    archivo.write_text("clave-en-archivo\n", encoding="utf-8")

    clave_env = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", clave_env)

    assert cache_key() == clave_env
    assert cache_key_created() is None


def test_cloud_key_prioridades_y_origenes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 9 (§B.2, M5): cloud_key(): AIDA_STT_API_KEY > OPENAI_API_KEY > stt.key > None;
    y cloud_key_source() reporta 'env', 'env', 'file', None correspondientemente.
    """
    config_dir = tmp_path / "config"
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "stt.key").write_text("sk-archivo\n", encoding="utf-8")

    # 1. Con AIDA_STT_API_KEY (manda sobre todo)
    monkeypatch.setenv("AIDA_STT_API_KEY", "sk-stt-env")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")
    assert cloud_key() == "sk-stt-env"
    assert cloud_key_source() == "env"

    # 2. Con OPENAI_API_KEY (manda sobre archivo)
    monkeypatch.delenv("AIDA_STT_API_KEY", raising=False)
    assert cloud_key() == "sk-openai-env"
    assert cloud_key_source() == "env"

    # 3. Solo con archivo
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cloud_key() == "sk-archivo"
    assert cloud_key_source() == "file"

    # 4. Sin nada
    (config_dir / "stt.key").unlink()
    assert cloud_key() is None
    assert cloud_key_source() is None


def test_save_cloud_key_guarda_con_permisos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 10 (§B.2): save_cloud_key('  sk-x \n') guarda 'sk-x' con 0o600,
    y cloud_key() lo lee correctamente.
    """
    config_dir = tmp_path / "config"
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("AIDA_STT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    save_cloud_key("  sk-x \n")
    archivo = config_dir / "stt.key"
    assert archivo.is_file()
    assert archivo.read_text(encoding="utf-8") == "sk-x\n"
    assert (archivo.stat().st_mode & 0o777) == 0o600
    assert cloud_key() == "sk-x"


def test_check_cloud_key_simulacion_y_cabeceras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 11 (§B.2, M6): check_cloud_key con httpx.get simulado devuelve (True, '200')
    ante 200, (False, '401') ante 401, y (False, 'ConnectError') ante ConnectError.
    Comprueba que la clave viaja en Authorization y no en la URL, y que la URL
    termina en /v1/models.
    """
    peticiones: list[tuple[str, dict[str, str]]] = []

    def fake_get_200(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        peticiones.append((url, headers))
        request = httpx.Request("GET", url, headers=headers)
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx, "get", fake_get_200)
    sirve, motivo = check_cloud_key("  sk-clave-secreta  ", base_url="https://api.openai.com")
    assert sirve is True
    assert motivo == "200"
    assert len(peticiones) == 1
    url_pedida, headers_pedidos = peticiones[0]
    assert url_pedida == "https://api.openai.com/v1/models"
    assert "sk-clave-secreta" not in url_pedida
    assert headers_pedidos.get("Authorization") == "Bearer sk-clave-secreta"

    # Simulación 401
    def fake_get_401(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        request = httpx.Request("GET", url, headers=headers)
        return httpx.Response(401, request=request)

    monkeypatch.setattr(httpx, "get", fake_get_401)
    sirve_401, motivo_401 = check_cloud_key("sk-invalida", base_url="https://api.openai.com")
    assert sirve_401 is False
    assert motivo_401 == "401"

    # Simulación ConnectError
    def fake_get_error(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        request = httpx.Request("GET", url, headers=headers)
        raise httpx.ConnectError("fallo de red", request=request)

    monkeypatch.setattr(httpx, "get", fake_get_error)
    sirve_err, motivo_err = check_cloud_key("sk-red", base_url="https://api.openai.com")
    assert sirve_err is False
    assert motivo_err == "ConnectError"


def test_build_provider_openai_lee_stt_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 12 (§B.2): build_provider('openai', ...) con la clave solo en stt.key
    construye OpenAIProvider exitosamente.
    """
    config_dir = tmp_path / "config"
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("AIDA_STT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "stt.key").write_text("sk-guardada\n", encoding="utf-8")

    provider = build_provider("openai")
    assert isinstance(provider, OpenAIProvider)
