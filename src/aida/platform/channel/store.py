"""Lo que entra y sale por el canal, en el equipo (102a). Teléfono, nombre y texto, sellados
con la clave del equipo; el teléfono, además, con una huella HMAC de clave derivada para
encontrar una conversación sin descifrar todas (como la agenda, con otra derivación).

`wamid UNIQUE` es la única puerta contra repetidos: el relé entrega al menos una vez."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from cryptography.fernet import Fernet

_PERSON_INDEX = b"aida-channel-idx"


class Direction(StrEnum):
    IN = "in"
    OUT = "out"


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    wamid: str
    phone: str
    name: str
    direction: Direction
    kind: str
    text: str
    at: datetime
    by: str
    """Quién envió, si salió del equipo; vacío si entró."""


class ChannelStore(Protocol):
    def add_inbound(
        self, *, wamid: str, phone: str, name: str, kind: str, text: str, at: datetime
    ) -> Message | None:
        """`None` si ese `wamid` ya estaba."""
        ...
    def add_outbound(
        self, *, wamid: str, phone: str, text: str, at: datetime, by: str
    ) -> Message: ...
    def messages(self, phone: str) -> tuple[Message, ...]:
        """Las de ese teléfono, por `at` y, a igualdad, por `id`."""
        ...
    def last_inbound(self, phone: str) -> datetime | None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY,
  wamid TEXT NOT NULL UNIQUE,
  phone_index TEXT NOT NULL,
  phone BLOB NOT NULL,
  name BLOB NOT NULL,
  direction TEXT NOT NULL,
  kind TEXT NOT NULL,
  text BLOB NOT NULL,
  at TEXT NOT NULL,
  by TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_phone ON messages(phone_index, id);
"""


def normalize_number(phone: str) -> str:
    return re.sub(r"\D", "", phone)


class SqliteChannelStore:
    def __init__(self, path: Path, key: bytes) -> None:
        self._fernet = Fernet(key)
        self._clave_indice = hashlib.blake2b(key, person=_PERSON_INDEX, digest_size=32).digest()
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not path.exists()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()
        if nuevo:
            with contextlib.suppress(OSError):
                os.chmod(path, 0o600)

    def _index(self, phone: str) -> str:
        return hmac.new(
            self._clave_indice, normalize_number(phone).encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _seal(self, texto: str) -> bytes:
        return self._fernet.encrypt(texto.encode("utf-8"))

    def _row(self, fila: sqlite3.Row) -> Message:
        abrir = self._fernet.decrypt
        return Message(
            id=int(fila["id"]),
            wamid=str(fila["wamid"]),
            phone=abrir(fila["phone"]).decode("utf-8"),
            name=abrir(fila["name"]).decode("utf-8"),
            direction=Direction(str(fila["direction"])),
            kind=str(fila["kind"]),
            text=abrir(fila["text"]).decode("utf-8"),
            at=datetime.fromisoformat(str(fila["at"])),
            by=str(fila["by"]),
        )

    def _insert(
        self, *, wamid: str, phone: str, name: str, direction: Direction, kind: str,
        text: str, at: datetime, by: str,
    ) -> Message | None:
        numero = normalize_number(phone)
        with self._lock:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO messages (wamid, phone_index, phone, name, direction,"
                " kind, text, at, by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    wamid, self._index(numero), self._seal(numero), self._seal(name),
                    direction.value, kind, self._seal(text), at.isoformat(), by,
                ),
            )
            self._db.commit()
            if cursor.rowcount == 0:
                return None
            fila = self._db.execute(
                "SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return self._row(fila)

    def add_inbound(
        self, *, wamid: str, phone: str, name: str, kind: str, text: str, at: datetime
    ) -> Message | None:
        return self._insert(
            wamid=wamid, phone=phone, name=name, direction=Direction.IN, kind=kind,
            text=text, at=at, by="",
        )

    def add_outbound(
        self, *, wamid: str, phone: str, text: str, at: datetime, by: str
    ) -> Message:
        guardado = self._insert(
            wamid=wamid, phone=phone, name="", direction=Direction.OUT, kind="text",
            text=text, at=at, by=by,
        )
        if guardado is None:
            raise ValueError(f"wamid repetido al enviar: {wamid}")
        return guardado

    def messages(self, phone: str) -> tuple[Message, ...]:
        with self._lock:
            filas = self._db.execute(
                "SELECT * FROM messages WHERE phone_index = ?", (self._index(phone),)
            ).fetchall()
        return tuple(sorted((self._row(f) for f in filas), key=lambda m: (m.at, m.id)))

    def last_inbound(self, phone: str) -> datetime | None:
        # Se compara en Python: las fechas ISO con distinta zona no se ordenan como texto.
        with self._lock:
            filas = self._db.execute(
                "SELECT at FROM messages WHERE phone_index = ? AND direction = ?",
                (self._index(phone), Direction.IN.value),
            ).fetchall()
        fechas = [datetime.fromisoformat(str(f["at"])) for f in filas]
        return max(fechas) if fechas else None
