"""Una sola AIDA abierta por usuario (plan 78b).

Empaquetada, AIDA no tiene ventana: si se cierra la pestaña sin «Salir», el proceso sigue
vivo y no se ve. Abrir el icono otra vez arrancaría una segunda copia, y las dos añadirían
a los mismos registros encadenados y a las mismas bases: la cadena se bifurca y deja de
verificarse. La segunda copia, en vez de eso, abre el navegador en la sesión de la primera
y termina.

El bloqueo es del sistema operativo sobre un archivo abierto, no la existencia del
archivo: si AIDA muere de golpe, el sistema lo suelta y la siguiente arranca. Un testigo
sin bloqueo dejaría AIDA sin abrir después de un apagón.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO

LOCK_NAME = "aida.lock"
URL_NAME = "aida.url"


class InstanceLock:
    def __init__(self, folder: Path, handle: IO[bytes]) -> None:
        self._folder = folder
        self._handle = handle

    def record(self, url: str) -> None:
        """La dirección de la sesión viva, aparte del archivo bloqueado: en Windows la zona
        bloqueada no se puede leer. Se sustituye entera para que nadie lea media."""
        destino = self._folder / URL_NAME
        temporal = destino.with_suffix(".tmp")
        temporal.write_text(url, encoding="utf-8")
        os.replace(temporal, destino)

    def release(self) -> None:
        (self._folder / URL_NAME).unlink(missing_ok=True)
        self._handle.close()


def _lock(handle: IO[bytes]) -> bool:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def acquire(folder: Path) -> InstanceLock | None:
    """`None` si otra AIDA ya la tiene. Al conseguirla se borra la dirección que dejó una
    copia muerta: la segunda copia solo debe abrir la de una viva."""
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / LOCK_NAME).open("a+b")
    if not _lock(handle):
        handle.close()
        return None
    (folder / URL_NAME).unlink(missing_ok=True)
    return InstanceLock(folder, handle)


def running_url(folder: Path) -> str | None:
    try:
        return (folder / URL_NAME).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
