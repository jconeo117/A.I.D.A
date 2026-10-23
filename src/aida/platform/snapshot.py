"""La copia de la configuración (78d): claves, perfiles, licencia y ajustes, cifrados con una
frase.

Por qué existe: todo lo que AIDA guarda va sellado con la clave del equipo, y esa clave vive en
la carpeta de configuración. Perder la carpeta es perder todo lo sellado. La copia es pequeña
(pocos KB) para que quepa en cualquier sitio, y no sirve sin la frase.

Por qué así:
- Se valida entera en memoria antes de tocar el disco: una copia manipulada no puede escribir
  fuera de la carpeta.
- La carpeta que había no se borra: se aparta a una hermana con fecha.
- La auditoría la firma el primer profesional que entra después (`restore-pending.json`),
  porque la auditoría exige autor y restaurar puede hacerse antes de que exista ninguno.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"AIDACOPIA1\n"
SALT_BYTES = 16
MIN_PASSPHRASE = 12
MAX_BYTES = 5 * 1024 * 1024
PENDING = "restore-pending.json"


class SnapshotError(Exception):
    """`reason`: `weak`, `empty`, `too_large`, `bad_file`, `bad_passphrase` o `unsafe_path`. El
    texto lo pone el pack (`snapshot_error_<reason>`)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Snapshot:
    data: bytes
    files: int


def _fernet(passphrase: str, salt: bytes) -> Fernet:
    clave = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase.encode("utf-8"))
    return Fernet(base64.urlsafe_b64encode(clave))


def _collect(config_dir: Path) -> dict[str, bytes]:
    """Todo archivo normal de la carpeta, en orden de ruta. Sin enlaces ni el aviso pendiente."""
    archivos: dict[str, bytes] = {}
    if not config_dir.is_dir():
        return archivos
    for ruta in sorted(config_dir.rglob("*")):
        if ruta.is_symlink() or not ruta.is_file():
            continue
        relativa = ruta.relative_to(config_dir).as_posix()
        if relativa == PENDING:
            continue
        archivos[relativa] = ruta.read_bytes()
    return archivos


def create_snapshot(config_dir: Path, passphrase: str, *, now: datetime, version: str) -> Snapshot:
    if len(passphrase) < MIN_PASSPHRASE:
        raise SnapshotError("weak")
    archivos = _collect(config_dir)
    if not archivos:
        raise SnapshotError("empty")
    if sum(len(v) for v in archivos.values()) > MAX_BYTES:
        raise SnapshotError("too_large")
    contenido = json.dumps(
        {
            "format": 1,
            "created": now.isoformat(timespec="seconds"),
            "aida_version": version,
            "files": {k: base64.b64encode(v).decode("ascii") for k, v in archivos.items()},
        },
        sort_keys=True,
    ).encode("utf-8")
    sal = os.urandom(SALT_BYTES)
    return Snapshot(
        data=MAGIC + sal + _fernet(passphrase, sal).encrypt(contenido), files=len(archivos)
    )


def _safe(relativa: str) -> bool:
    """Una ruta relativa, en barras normales, sin subir de nivel y sin unidad de Windows."""
    if not relativa or "\\" in relativa or ":" in relativa:
        return False
    partes = PurePosixPath(relativa)
    return not partes.is_absolute() and ".." not in partes.parts


def _read(data: bytes, passphrase: str) -> tuple[dict[str, bytes], dict[str, str]]:
    if len(data) > 2 * MAX_BYTES or not data.startswith(MAGIC):
        raise SnapshotError("bad_file")
    sal = data[len(MAGIC) : len(MAGIC) + SALT_BYTES]
    token = data[len(MAGIC) + SALT_BYTES :]
    if len(sal) != SALT_BYTES or not token:
        raise SnapshotError("bad_file")
    try:
        claro = _fernet(passphrase, sal).decrypt(token)
    except InvalidToken as exc:
        raise SnapshotError("bad_passphrase") from exc
    try:
        contenido = json.loads(claro)
        archivos = {
            str(k): base64.b64decode(str(v), validate=True)
            for k, v in dict(contenido["files"]).items()
        }
        datos = {
            "created": str(contenido["created"]),
            "aida_version": str(contenido["aida_version"]),
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise SnapshotError("bad_file") from exc
    if not all(_safe(k) for k in archivos):
        raise SnapshotError("unsafe_path")
    if sum(len(v) for v in archivos.values()) > MAX_BYTES:
        raise SnapshotError("too_large")
    return archivos, datos


def restore_snapshot(
    config_dir: Path, data: bytes, passphrase: str, *, now: datetime
) -> Path | None:
    """Devuelve la carpeta apartada, o `None` si no había nada que apartar."""
    archivos, datos = _read(data, passphrase)
    apartada: Path | None = None
    if config_dir.exists() and any(config_dir.iterdir()):
        apartada = config_dir.with_name(f"{config_dir.name}.antes-{now.strftime('%Y%m%d-%H%M%S')}")
        config_dir.replace(apartada)
    config_dir.mkdir(parents=True, exist_ok=True)
    for relativa, contenido in archivos.items():
        destino = config_dir / PurePosixPath(relativa)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(contenido)
        with contextlib.suppress(OSError):
            destino.chmod(0o600)
    aviso = {
        "restored_at": now.isoformat(timespec="seconds"),
        "created": datos["created"],
        "aida_version": datos["aida_version"],
        "files": str(len(archivos)),
        "previous": apartada.name if apartada is not None else "",
    }
    (config_dir / PENDING).write_text(json.dumps(aviso, sort_keys=True), encoding="utf-8")
    return apartada


def pending_restore(config_dir: Path) -> dict[str, str] | None:
    ruta = config_dir / PENDING
    if not ruta.is_file():
        return None
    try:
        return {str(k): str(v) for k, v in dict(json.loads(ruta.read_text("utf-8"))).items()}
    except (ValueError, TypeError):
        return {"previous": "", "files": "", "created": "", "aida_version": "", "restored_at": ""}


def clear_pending(config_dir: Path) -> None:
    (config_dir / PENDING).unlink(missing_ok=True)
