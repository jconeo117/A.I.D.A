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


@dataclass(frozen=True, slots=True)
class Conversation:
    phone: str
    name: str
    """El último nombre de perfil que dio el contacto; vacío si nunca dio uno."""
    last: Message
    unread: int


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
    def conversations(self) -> tuple[Conversation, ...]:
        """Una por teléfono; la de actividad más reciente primero y, a igualdad, la de `id`
        mayor."""
        ...
    def mark_read(self, phone: str, *, up_to: int, by: str, at: datetime) -> bool:
        """Leída hasta `up_to` para todo el mostrador. No retrocede. `False` si no hay
        conversación con ese teléfono."""
        ...


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
CREATE TABLE IF NOT EXISTS reads (
  phone_index TEXT PRIMARY KEY,
  last_read INTEGER NOT NULL,
  by TEXT NOT NULL,
  at TEXT NOT NULL
);
"""


def normalize_number(phone: str) -> str:
    return re.sub(r"\D", "", phone)


def to_whatsapp(phone: str, country: str) -> str:
    """El número como lo escribe Meta: con el prefijo del país. Una sola puerta: la usan el
    índice del almacén y el envío, y con dos copias una conversación se partiría en dos."""
    digitos = normalize_number(phone)
    if country and digitos and not digitos.startswith(country):
        return country + digitos
    return digitos


def national(phone: str, country: str) -> str:
    """Sin el prefijo: como lo guarda la agenda si el mostrador no lo tecleó."""
    digitos = normalize_number(phone)
    if country and digitos.startswith(country):
        return digitos[len(country) :]
    return digitos


class SqliteChannelStore:
    def __init__(self, path: Path, key: bytes, *, country: str = "") -> None:
        self._country = country
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
            self._clave_indice,
            to_whatsapp(phone, self._country).encode("utf-8"),
            hashlib.sha256,
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
        numero = to_whatsapp(phone, self._country)
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

    def conversations(self) -> tuple[Conversation, ...]:
        # Descifra todo: a la escala de un mostrador cabe (la 106 lo resolverá en su base).
        with self._lock:
            filas = self._db.execute("SELECT * FROM messages").fetchall()
            leidas = {
                str(f["phone_index"]): int(f["last_read"])
                for f in self._db.execute("SELECT phone_index, last_read FROM reads").fetchall()
            }
            grupos: dict[str, list[Message]] = {}
            for f in filas:
                grupos.setdefault(str(f["phone_index"]), []).append(self._row(f))
        resultado: list[Conversation] = []
        for indice in sorted(grupos):
            lista = sorted(grupos[indice], key=lambda m: (m.at, m.id))
            ultima = lista[-1]
            hasta = leidas.get(indice, 0)
            resultado.append(
                Conversation(
                    phone=ultima.phone,
                    name=next(
                        (m.name for m in reversed(lista) if m.direction == Direction.IN and m.name),
                        "",
                    ),
                    last=ultima,
                    unread=sum(
                        1 for m in lista if m.direction == Direction.IN and m.id > hasta
                    ),
                )
            )
        return tuple(sorted(resultado, key=lambda c: (c.last.at, c.last.id), reverse=True))

    def mark_read(self, phone: str, *, up_to: int, by: str, at: datetime) -> bool:
        indice = self._index(phone)
        with self._lock:
            existe = self._db.execute(
                "SELECT 1 FROM messages WHERE phone_index = ? LIMIT 1", (indice,)
            ).fetchone()
            if existe is None:
                return False
            self._db.execute(
                "INSERT INTO reads (phone_index, last_read, by, at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(phone_index) DO UPDATE SET last_read = excluded.last_read,"
                " by = excluded.by, at = excluded.at"
                " WHERE excluded.last_read > reads.last_read",
                (indice, up_to, by, at.isoformat()),
            )
            self._db.commit()
            return True
