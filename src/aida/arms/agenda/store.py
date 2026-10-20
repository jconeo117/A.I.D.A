"""El almacenamiento de la agenda, detrás de un `Protocol` (101b).

El nombre y el teléfono de un contacto van sellados con la clave del equipo: son datos de
una persona. El teléfono, además, lleva una huella HMAC para encontrarlo exacto sin tener que
descifrar todos los contactos, y esa huella usa una clave derivada, no la que sella: quien
tenga una no puede usar la otra.

Las fechas, el servicio, el recurso y el estado van en claro: sin ellos no se puede pintar ni
buscar un hueco.
"""

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

_PERSONA_INDICE = b"aida-agenda-idx"


class AppointmentState(StrEnum):
    BOOKED = "booked"
    CANCELLED = "cancelled"
    ATTENDED = "attended"
    NO_SHOW = "no_show"


@dataclass(frozen=True, slots=True)
class Appointment:
    id: int
    client: int
    service: str
    resource: str
    start: datetime
    minutes: int
    state: AppointmentState
    created_by: str
    updated_by: str


@dataclass(frozen=True, slots=True)
class Client:
    id: int
    name: str
    phone: str


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    phone: str
    direction: str
    text: str
    at: datetime
    kind: str


class AgendaStore(Protocol):
    def appointments_for_client(self, client: int) -> tuple[Appointment, ...]: ...
    def add_message(
        self, *, phone: str, direction: str, text: str, at: datetime, kind: str
    ) -> Message: ...
    def messages(self, phone: str) -> tuple[Message, ...]: ...
    def conversations(self) -> tuple[Message, ...]: ...
    def mark_reminder(self, appointment: int, kind: str, at: datetime) -> bool: ...
    def client_by_phone(self, phone: str) -> Client | None: ...
    def add_client(self, *, name: str, phone: str) -> Client: ...
    def update_client(self, client: int, *, name: str, phone: str) -> None: ...
    def client(self, client: int) -> Client: ...
    def add(
        self, *, client: int, service: str, resource: str, start: datetime, minutes: int, by: str
    ) -> Appointment: ...
    def update(
        self,
        appointment: int,
        *,
        by: str,
        service: str | None = None,
        resource: str | None = None,
        start: datetime | None = None,
        minutes: int | None = None,
        state: AppointmentState | None = None,
    ) -> Appointment: ...
    def get(self, appointment: int) -> Appointment | None: ...
    def between(self, start: datetime, end: datetime) -> tuple[Appointment, ...]:
        """Las que empiezan en `[start, end)`, ordenadas por `start` y, a igualdad, por `id`
        (determinismo: nunca desempatar por un float)."""
        ...


_ESQUEMA = """
CREATE TABLE IF NOT EXISTS clients (
  id INTEGER PRIMARY KEY,
  name BLOB NOT NULL,
  phone BLOB NOT NULL,
  phone_index TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS appointments (
  id INTEGER PRIMARY KEY,
  client INTEGER NOT NULL REFERENCES clients(id),
  service TEXT NOT NULL,
  resource TEXT NOT NULL,
  start TEXT NOT NULL,
  minutes INTEGER NOT NULL,
  state TEXT NOT NULL,
  created_by TEXT NOT NULL,
  updated_by TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS appointments_start ON appointments(start);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY,
  phone_index TEXT NOT NULL,
  phone BLOB NOT NULL,         -- Fernet(teléfono normalizado)
  direction TEXT NOT NULL,     -- 'in' (la persona escribe) o 'out' (el agente responde)
  text BLOB NOT NULL,          -- Fernet(texto)
  at TEXT NOT NULL,
  kind TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_phone ON messages(phone_index, id);
CREATE TABLE IF NOT EXISTS reminders (
  appointment INTEGER NOT NULL,
  kind TEXT NOT NULL,
  at TEXT NOT NULL,
  PRIMARY KEY (appointment, kind)
);
"""


def normalizar_telefono(telefono: str) -> str:
    """Solo dígitos: «300 555 0142» y «3005550142» son el mismo contacto."""
    return re.sub(r"\D", "", telefono)


class SqliteAgendaStore:
    def __init__(self, path: Path, key: bytes) -> None:
        self._fernet = Fernet(key)
        clave_indice = hashlib.blake2b(key, person=_PERSONA_INDICE, digest_size=32).digest()
        self._clave_indice = clave_indice
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not path.exists()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_ESQUEMA)
        self._db.commit()
        if nuevo:
            with contextlib.suppress(OSError):
                os.chmod(path, 0o600)

    def _indice(self, telefono: str) -> str:
        return hmac.new(
            self._clave_indice, normalizar_telefono(telefono).encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _contacto(self, fila: sqlite3.Row) -> Client:
        return Client(
            id=int(fila["id"]),
            name=self._fernet.decrypt(fila["name"]).decode("utf-8"),
            phone=self._fernet.decrypt(fila["phone"]).decode("utf-8"),
        )

    def _cita(self, fila: sqlite3.Row) -> Appointment:
        return Appointment(
            id=int(fila["id"]),
            client=int(fila["client"]),
            service=str(fila["service"]),
            resource=str(fila["resource"]),
            start=datetime.fromisoformat(str(fila["start"])),
            minutes=int(fila["minutes"]),
            state=AppointmentState(str(fila["state"])),
            created_by=str(fila["created_by"]),
            updated_by=str(fila["updated_by"]),
        )

    def client_by_phone(self, phone: str) -> Client | None:
        with self._lock:
            fila = self._db.execute(
                "SELECT * FROM clients WHERE phone_index = ?", (self._indice(phone),)
            ).fetchone()
            return self._contacto(fila) if fila is not None else None

    def add_client(self, *, name: str, phone: str) -> Client:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO clients (name, phone, phone_index) VALUES (?, ?, ?)",
                (
                    self._fernet.encrypt(name.encode("utf-8")),
                    self._fernet.encrypt(normalizar_telefono(phone).encode("utf-8")),
                    self._indice(phone),
                ),
            )
            self._db.commit()
            return self._contacto_por_id(int(cursor.lastrowid or 0))

    def update_client(self, client: int, *, name: str, phone: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE clients SET name = ?, phone = ?, phone_index = ? WHERE id = ?",
                (
                    self._fernet.encrypt(name.encode("utf-8")),
                    self._fernet.encrypt(normalizar_telefono(phone).encode("utf-8")),
                    self._indice(phone),
                    client,
                ),
            )
            self._db.commit()

    def client(self, client: int) -> Client:
        with self._lock:
            return self._contacto_por_id(client)

    def _contacto_por_id(self, client: int) -> Client:
        fila = self._db.execute("SELECT * FROM clients WHERE id = ?", (client,)).fetchone()
        if fila is None:
            raise LookupError(f"no hay contacto {client}")
        return self._contacto(fila)

    def add(
        self, *, client: int, service: str, resource: str, start: datetime, minutes: int, by: str
    ) -> Appointment:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO appointments (client, service, resource, start, minutes, state,"
                " created_by, updated_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    client,
                    service,
                    resource,
                    start.isoformat(),
                    minutes,
                    AppointmentState.BOOKED.value,
                    by,
                    by,
                ),
            )
            self._db.commit()
            fila = self._db.execute(
                "SELECT * FROM appointments WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return self._cita(fila)

    def update(
        self,
        appointment: int,
        *,
        by: str,
        service: str | None = None,
        resource: str | None = None,
        start: datetime | None = None,
        minutes: int | None = None,
        state: AppointmentState | None = None,
    ) -> Appointment:
        cambios: dict[str, object] = {"updated_by": by}
        if service is not None:
            cambios["service"] = service
        if resource is not None:
            cambios["resource"] = resource
        if start is not None:
            cambios["start"] = start.isoformat()
        if minutes is not None:
            cambios["minutes"] = minutes
        if state is not None:
            cambios["state"] = state.value
        columnas = ", ".join(f"{columna} = ?" for columna in cambios)
        with self._lock:
            cursor = self._db.execute(
                f"UPDATE appointments SET {columnas} WHERE id = ?",
                (*cambios.values(), appointment),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"no hay cita {appointment}")
            self._db.commit()
            fila = self._db.execute(
                "SELECT * FROM appointments WHERE id = ?", (appointment,)
            ).fetchone()
            return self._cita(fila)

    def get(self, appointment: int) -> Appointment | None:
        with self._lock:
            fila = self._db.execute(
                "SELECT * FROM appointments WHERE id = ?", (appointment,)
            ).fetchone()
            return self._cita(fila) if fila is not None else None

    def appointments_for_client(self, client: int) -> tuple[Appointment, ...]:
        with self._lock:
            filas = self._db.execute(
                "SELECT * FROM appointments WHERE client = ?", (client,)
            ).fetchall()
        return tuple(sorted((self._cita(f) for f in filas), key=lambda c: (c.start, c.id)))

    def _registro_texto(self, fila: sqlite3.Row) -> Message:
        return Message(
            id=int(fila["id"]),
            phone=self._fernet.decrypt(fila["phone"]).decode("utf-8"),
            direction=str(fila["direction"]),
            text=self._fernet.decrypt(fila["text"]).decode("utf-8"),
            at=datetime.fromisoformat(str(fila["at"])),
            kind=str(fila["kind"]),
        )

    def add_message(
        self, *, phone: str, direction: str, text: str, at: datetime, kind: str
    ) -> Message:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO messages (phone_index, phone, direction, text, at, kind)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self._indice(phone),
                    self._fernet.encrypt(normalizar_telefono(phone).encode("utf-8")),
                    direction,
                    self._fernet.encrypt(text.encode("utf-8")),
                    at.isoformat(),
                    kind,
                ),
            )
            self._db.commit()
            fila = self._db.execute(
                "SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return self._registro_texto(fila)

    def messages(self, phone: str) -> tuple[Message, ...]:
        with self._lock:
            filas = self._db.execute(
                "SELECT * FROM messages WHERE phone_index = ? ORDER BY id",
                (self._indice(phone),),
            ).fetchall()
        return tuple(self._registro_texto(f) for f in filas)

    def conversations(self) -> tuple[Message, ...]:
        """El último mensaje de cada conversación, la más reciente primero."""
        with self._lock:
            filas = self._db.execute(
                "SELECT * FROM messages WHERE id IN"
                " (SELECT MAX(id) FROM messages GROUP BY phone_index)"
            ).fetchall()
        return tuple(
            sorted((self._registro_texto(f) for f in filas), key=lambda m: m.id, reverse=True)
        )

    def mark_reminder(self, appointment: int, kind: str, at: datetime) -> bool:
        """Anota que ya se envió. `False` si ya estaba: un recordatorio no se manda dos veces."""
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO reminders (appointment, kind, at) VALUES (?, ?, ?)",
                    (appointment, kind, at.isoformat()),
                )
            except sqlite3.IntegrityError:
                return False
            self._db.commit()
            return True

    def between(self, start: datetime, end: datetime) -> tuple[Appointment, ...]:
        # Se filtra en Python: las fechas ISO con distinta zona no se comparan como texto.
        with self._lock:
            filas = self._db.execute("SELECT * FROM appointments").fetchall()
        citas = [self._cita(f) for f in filas]
        dentro = [c for c in citas if start <= c.start < end]
        return tuple(sorted(dentro, key=lambda c: (c.start, c.id)))
