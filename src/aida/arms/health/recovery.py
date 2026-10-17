"""Las grabaciones que AIDA dejó sin terminar, y lo que hace falta para recuperarlas (87).

Antes, la sesión vivía en `/tmp` —memoria en Linux— y el arranque la borraba: una
consulta grabada a medias se perdía con un corte de luz. Ahora vive en la carpeta de
datos, y junto al audio queda un marcador sellado con quién la grabó, con qué
autorización y cuándo empezó. Sin el marcador, la carpeta es una subida a medias y se
borra al arrancar; con él, solo la borra el plazo.

El marcador va sellado con la clave del equipo y no con una de sesión: la de sesión
muere con el proceso, y el propósito entero es que la siguiente ejecución la lea.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from cryptography.fernet import InvalidToken

from aida.arms.health.live import RECORDING_NAME
from aida.platform.sealed import open_bytes, seal_bytes
from aida.resources import user_dir

MARKER_NAME = "sesion.json.sealed"
_VERSION = 1


def sessions_root() -> Path:
    """`user_dir("data")/"sesiones"`. Fuera de /tmp: en Linux es memoria y un reinicio se
    llevaría la grabación (87)."""
    return user_dir("data") / "sesiones"


@dataclass(frozen=True, slots=True)
class Interrupted:
    folder: Path
    owner: str
    consent_digest: str
    professional: str
    started: datetime
    document: str
    recorded_until: datetime
    """La fecha del archivo de la grabación: lo último que se escribió."""

    def duration(self) -> timedelta:
        return self.recorded_until - self.started

    def expires(self, hours: int) -> datetime:
        return self.started + timedelta(hours=hours)


def write_marker(
    folder: Path,
    *,
    owner: str,
    consent_digest: str,
    professional: str,
    started: datetime,
    document: str,
    key: bytes,
) -> None:
    datos = {
        "consent": consent_digest,
        "document": document,
        "owner": owner,
        "professional": professional,
        "started": started.isoformat(timespec="seconds"),
        "version": _VERSION,
    }
    seal_bytes(folder / MARKER_NAME, json.dumps(datos, sort_keys=True).encode("utf-8"), key)


def _leer_marcador(folder: Path, key: bytes) -> dict[str, object] | None:
    """`None` si el marcador no se puede leer con esta clave o no tiene la forma esperada:
    la carpeta sigue ahí —se respeta—, pero no se puede ofrecer."""
    try:
        datos = json.loads(open_bytes(folder / MARKER_NAME, key))
    except (InvalidToken, ValueError, OSError):
        return None
    return datos if isinstance(datos, dict) else None


def _fecha_archivo(folder: Path) -> datetime:
    """Lo último que se escribió en la sesión: la grabación, o el marcador si aún no hay
    audio. Es el reloj del plazo cuando el marcador no se puede leer."""
    grabacion = folder / RECORDING_NAME
    origen = grabacion if grabacion.is_file() else folder / MARKER_NAME
    return datetime.fromtimestamp(origen.stat().st_mtime).astimezone()


def _carpetas_con_marcador(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    return tuple(
        carpeta
        for carpeta in sorted(root.iterdir())
        if carpeta.is_dir() and (carpeta / MARKER_NAME).is_file()
    )


def interrupted(root: Path, key: bytes) -> tuple[Interrupted, ...]:
    """Las que tienen metadatos legibles y grabación, ordenadas por `started`."""
    recuperables: list[Interrupted] = []
    for carpeta in _carpetas_con_marcador(root):
        datos = _leer_marcador(carpeta, key)
        grabacion = carpeta / RECORDING_NAME
        if datos is None or not grabacion.is_file():
            continue
        recuperables.append(
            Interrupted(
                folder=carpeta,
                owner=str(datos["owner"]),
                consent_digest=str(datos["consent"]),
                professional=str(datos["professional"]),
                started=datetime.fromisoformat(str(datos["started"])),
                document=str(datos["document"]),
                recorded_until=_fecha_archivo(carpeta),
            )
        )
    return tuple(sorted(recuperables, key=lambda i: (i.started, i.folder.name)))


def purge_unmarked(root: Path) -> None:
    """Al arrancar: borra las carpetas sin metadatos (subidas a medias)."""
    for carpeta in root.iterdir() if root.is_dir() else ():
        if carpeta.is_dir() and not (carpeta / MARKER_NAME).is_file():
            shutil.rmtree(carpeta, ignore_errors=True)


def started_at(folder: Path, key: bytes) -> tuple[datetime, bool]:
    """Cuándo empezó la grabación y si el marcador se pudo leer. Sin lectura, la fecha
    es la del archivo: el plazo sigue corriendo aunque no se sepa de quién era."""
    datos = _leer_marcador(folder, key)
    if datos is None:
        return _fecha_archivo(folder), False
    return datetime.fromisoformat(str(datos["started"])), True


def expired(root: Path, key: bytes, hours: int, now: datetime) -> tuple[Path, ...]:
    """Las carpetas con metadatos (legibles o no) que pasan del plazo, en orden de nombre."""
    vencidas: list[Path] = []
    for carpeta in _carpetas_con_marcador(root):
        empieza, _legible = started_at(carpeta, key)
        if empieza + timedelta(hours=hours) <= now:
            vencidas.append(carpeta)
    return tuple(vencidas)
