"""Lo que el relé guarda mientras el equipo está apagado, detrás de un `Protocol`: hoy SQLite
en el propio relé; al desplegar en la nube, un almacén durable con la misma forma."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Pending:
    id: int
    sealed: bytes


class RelayStore(Protocol):
    def put(self, *, tenant: str, sealed: bytes) -> int: ...
    def pending(self, *, tenant: str, limit: int) -> tuple[Pending, ...]:
        """Las de ese inquilino, por `id` ascendente: el orden en que llegaron."""
        ...
    def ack(self, *, tenant: str, ids: Sequence[int]) -> int:
        """Borra las de **ese** inquilino; un `id` de otro no se toca. Devuelve cuántas."""
        ...


class SqliteRelayStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS items ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " tenant TEXT NOT NULL,"
            " sealed BLOB NOT NULL);"
            "CREATE INDEX IF NOT EXISTS items_tenant ON items(tenant, id);"
        )
        self._db.commit()

    def put(self, *, tenant: str, sealed: bytes) -> int:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO items (tenant, sealed) VALUES (?, ?)", (tenant, sealed)
            )
            self._db.commit()
            return int(cursor.lastrowid or 0)

    def pending(self, *, tenant: str, limit: int) -> tuple[Pending, ...]:
        with self._lock:
            filas = self._db.execute(
                "SELECT id, sealed FROM items WHERE tenant = ? ORDER BY id LIMIT ?",
                (tenant, limit),
            ).fetchall()
        return tuple(Pending(id=int(f[0]), sealed=bytes(f[1])) for f in filas)

    def ack(self, *, tenant: str, ids: Sequence[int]) -> int:
        if not ids:
            return 0
        marcas = ", ".join("?" for _ in ids)
        with self._lock:
            cursor = self._db.execute(
                f"DELETE FROM items WHERE tenant = ? AND id IN ({marcas})",
                (tenant, *ids),
            )
            self._db.commit()
            return cursor.rowcount
