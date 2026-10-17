"""Las claves de AIDA: dónde se leen, dónde se guardan, y la de la caché creada sola.

Antes vivían solo en variables de entorno, que hay que poner a mano en cada terminal:
imposible para quien abre AIDA con un icono (plan 78). Ahora, por orden: la variable
de entorno (si existe, manda: es lo que usa el desarrollo y la batería) y, si no, un
archivo en la carpeta de configuración del usuario, legible solo por él.

Límite que queda escrito: la clave de la caché vive en la misma máquina que la caché.
Protege la caché copiada sola, no la carpeta del usuario entera. El llavero del
sistema queda para después.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from aida.resources import user_dir

_CACHE_ENV = "AIDA_CONSULT_CACHE_KEY"
_CLOUD_ENV = ("AIDA_STT_API_KEY", "OPENAI_API_KEY")
_CACHE_FILE = "cache.key"
_CLOUD_FILE = "stt.key"
_LOCAL_ONLY_FILE = "solo-local"


def _write_private(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    # En Windows los permisos los da la carpeta del perfil: chmod no aplica.
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def cache_key(*, create: bool = True) -> str | None:
    """La clave Fernet de la caché. La primera vez, si no hay ninguna y `create`, la
    genera y la guarda: el usuario no tiene por qué saber qué es una clave Fernet."""
    declarada = os.environ.get(_CACHE_ENV)
    if declarada:
        return declarada
    ruta = user_dir("config") / _CACHE_FILE
    if ruta.is_file():
        return ruta.read_text(encoding="utf-8").strip() or None
    if not create:
        return None
    from cryptography.fernet import Fernet

    nueva = Fernet.generate_key().decode()
    _write_private(ruta, nueva)
    return nueva


def cache_key_created() -> float | None:
    """Cuándo se guardó la clave de la caché, para que Ajustes lo diga. `None` si viene
    del entorno o aún no existe."""
    if os.environ.get(_CACHE_ENV):
        return None
    ruta = user_dir("config") / _CACHE_FILE
    return ruta.stat().st_mtime if ruta.is_file() else None


def cloud_key() -> str | None:
    """La clave de la nube: la misma credencial transcribe y redacta. Una sola puerta
    para leerla: antes la leían tres sitios con criterios distintos."""
    for nombre in _CLOUD_ENV:
        valor = os.environ.get(nombre)
        if valor:
            return valor
    ruta = user_dir("config") / _CLOUD_FILE
    if ruta.is_file():
        return ruta.read_text(encoding="utf-8").strip() or None
    return None


def cloud_key_source() -> str | None:
    """`env`, `file` o `None`: Ajustes lo enseña, porque una clave del entorno pisa la
    guardada y, sin decirlo, «Guardar» parecería no hacer nada."""
    if any(os.environ.get(nombre) for nombre in _CLOUD_ENV):
        return "env"
    return "file" if cloud_key() is not None else None


def save_cloud_key(value: str) -> None:
    _write_private(user_dir("config") / _CLOUD_FILE, value.strip())


def local_only() -> bool:
    """«Procesar solo en este equipo» (plan 85b). Un archivo y no una variable: lo cambia
    el profesional desde Ajustes y tiene que sobrevivir a cerrar AIDA."""
    return (user_dir("config") / _LOCAL_ONLY_FILE).is_file()


def save_local_only(value: bool) -> None:
    ruta = user_dir("config") / _LOCAL_ONLY_FILE
    if value:
        _write_private(ruta, "")
    else:
        ruta.unlink(missing_ok=True)


def check_cloud_key(value: str, *, base_url: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Prueba la clave con la petición más barata que hay (listar modelos), antes de
    gastarla en una consulta: `I-154` estuvo tres unidades sin verse porque nada la
    probaba. Devuelve si sirve y un motivo corto: el código HTTP o el fallo de red."""
    import httpx

    try:
        respuesta = httpx.get(
            f"{base_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {value.strip()}"},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return False, type(exc).__name__
    if respuesta.status_code == 200:
        return True, "200"
    return False, str(respuesta.status_code)
