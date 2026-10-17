"""Archivos sellados: lo que se guarda en disco durante una sesión, cifrado (plan 85a).

La Ley 1581 pide medidas de seguridad para los datos sensibles, y el audio de quien es
grabado lo es. Antes, lo subido y lo grabado vivían en claro en la carpeta de la sesión
mientras duraba. Ahora se escriben **sellados** con la clave del equipo, y solo se
descifran a un temporal propio mientras alguien los procesa: el convertidor de audio, el
transcriptor y el lector de documentos trabajan con archivos.

Se sella con la clave de la caché, no con una de sesión que muera con el proceso: así
una grabación interrumpida se podrá recuperar (87). El límite es el mismo que el de la
caché: quien tenga el equipo y su clave puede descifrarlo.

La grabación se sella por trozos, un token por línea: si el proceso cae a mitad, lo
escrito hasta ese momento sigue siendo legible.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

PLAIN_PREFIX = "aida-claro-"
"""Prefijo de los temporales en claro. Al arrancar se borra todo lo que lo lleve: si el
proceso cayó a mitad de procesar, el claro no sobrevive al siguiente arranque."""


def _fernet(key: bytes):  # type: ignore[no-untyped-def]
    from cryptography.fernet import Fernet

    return Fernet(key)


def _private(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def seal_bytes(path: Path, data: bytes, key: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_fernet(key).encrypt(data))
    _private(path)


def open_bytes(path: Path, key: bytes) -> bytes:
    return bytes(_fernet(key).decrypt(path.read_bytes()))


class SealedAppender:
    """Un archivo sellado que crece: cada `append` es un token en su línea."""

    def __init__(self, path: Path, key: bytes) -> None:
        self._path = path
        self._fernet = _fernet(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        _private(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, data: bytes) -> None:
        if not data:
            return
        with self._path.open("ab") as archivo:
            archivo.write(self._fernet.encrypt(data) + b"\n")
            archivo.flush()
            # Un apagón pierde como mucho el último trozo (87).
            os.fsync(archivo.fileno())


def read_appended(path: Path, key: bytes) -> bytes:
    """Todo lo añadido, descifrado y en orden. Una última línea que no descifra es un
    corte a mitad de escritura: se ignora, y lo anterior sigue siendo la grabación. Una
    rota en medio no es un corte sino daño, y sigue reventando."""
    from cryptography.fernet import InvalidToken

    fernet = _fernet(key)
    lineas = [linea for linea in path.read_bytes().splitlines() if linea.strip()]
    partes: list[bytes] = []
    for indice, linea in enumerate(lineas):
        try:
            partes.append(fernet.decrypt(linea))
        except InvalidToken:
            if indice != len(lineas) - 1:
                raise
    return b"".join(partes)


def plain_name(path: Path) -> str:
    """El nombre con el que se descifra: el del sellado sin `.sealed`, para que quien lo
    lea sepa por la extensión qué es (un PDF, una imagen, un audio)."""
    return path.name.removesuffix(".sealed")


@contextlib.contextmanager
def plain_directory() -> Iterator[Path]:
    """Una carpeta temporal propia para el claro, borrada al salir pase lo que pase."""
    carpeta = Path(tempfile.mkdtemp(prefix=PLAIN_PREFIX))
    try:
        yield carpeta
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)


@contextlib.contextmanager
def plain_copy(path: Path, key: bytes) -> Iterator[Path]:
    """El archivo descifrado, en claro solo mientras dura el `with`."""
    with plain_directory() as carpeta:
        destino = carpeta / plain_name(path)
        destino.write_bytes(open_bytes(path, key))
        _private(destino)
        yield destino


def purge_plain_leftovers() -> None:
    """Lo que un proceso caído dejó en claro: se borra al arrancar."""
    for resto in sorted(Path(tempfile.gettempdir()).glob(f"{PLAIN_PREFIX}*")):
        shutil.rmtree(resto, ignore_errors=True)
