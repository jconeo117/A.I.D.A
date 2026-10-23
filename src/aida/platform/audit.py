"""Qué pasó, quién lo hizo y sobre quién (plan 83).

Sobre la cadena común (`aida.ledger`): cada línea dice en claro cuándo, qué
profesional, desde qué equipo, sobre qué persona atendida, qué clase de hecho y sus
datos escalares (qué campos se editaron, qué códigos se confirmaron o quitaron). Nada
de lo que se dijo.

Lo que se dijo —el borrador que propuso AIDA y la versión que dejó el profesional— va
aparte, **cifrado** con la clave de la caché, en un archivo cuyo nombre es la huella de
su contenido; la línea solo lleva esa huella. Así, ante una reclamación, se puede
mostrar qué escribió cada uno; y si se pierde la clave se pierden los textos, pero no
el rastro, que es lo que una auditoría no puede perder.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from aida.platform.identity import IdentityError, Professional
from aida.platform.identity import workstation as _puesto
from aida.platform.ledger import ChainedLog

KINDS = (
    "session_opened",
    "session_closed",
    "locked",
    "unlocked",
    "profile_created",
    "profile_deleted",
    "consent_recorded",
    "history_viewed",
    "draft_generated",
    "review_saved",
    "exported",
    "license_loaded",
    "texts_purged",
    "recording_recovered",
    "recording_discarded",
    "recording_purged",
    "agenda_changed",
    "message_sent",
    "pin_recovered",
    "pin_reset",
    "recovery_code_created",
    "snapshot_created",
    "snapshot_restored",
)


class AuditLog:
    def __init__(
        self,
        path: Path | None = None,
        *,
        blobs: Path | None = None,
        key: bytes | None = None,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        workstation: Callable[[], str] | None = None,
    ) -> None:
        self._chain = ChainedLog(path)
        self._blobs = blobs
        self._key = key
        self._now = now
        self._workstation = workstation if workstation is not None else _puesto

    def record(
        self,
        kind: str,
        *,
        actor: Professional,
        owner: str = "",
        data: Mapping[str, str] | None = None,
        texts: Mapping[str, object] | None = None,
    ) -> dict[str, str]:
        """Añade un hecho. `texts` se cifra aparte y la línea lleva su huella; sin
        clave o sin carpeta de textos, no se guardan (y la línea lo dice con `blob`
        vacío).

        Todo hecho tiene autor (`I-156`, regla del usuario: «no pueden haber registros
        sin dueño»). El tipo ya lo pide, y se comprueba igual en ejecución: quien llama
        sin tipos, o con un `Optional` mal estrechado, no pasa por mypy."""
        if actor is None:
            raise IdentityError("no_actor", "todo hecho de auditoría tiene autor")
        if kind not in KINDS:
            raise ValueError(f"clase de hecho de auditoría desconocida: {kind!r}")
        blob = self._store(texts) if texts else ""
        return self._chain.append(
            {
                "at": self._now().isoformat(timespec="seconds"),
                "kind": kind,
                "actor": actor.id,
                "actor_name": actor.label(),
                "workstation": self._workstation(),
                "owner": owner,
                "data": json.dumps(dict(data or {}), ensure_ascii=False, sort_keys=True),
                "blob": blob,
            }
        )

    def _store(self, texts: Mapping[str, object]) -> str:
        if self._blobs is None or self._key is None:
            return ""
        from cryptography.fernet import Fernet

        claro = json.dumps(dict(texts), ensure_ascii=False, sort_keys=True).encode("utf-8")
        huella = hashlib.sha256(claro).hexdigest()
        self._blobs.mkdir(parents=True, exist_ok=True)
        destino = self._blobs / f"{huella}.bin"
        destino.write_bytes(Fernet(self._key).encrypt(claro))
        with contextlib.suppress(OSError):
            destino.chmod(0o600)
        return huella

    def show(self, blob: str) -> dict[str, object]:
        """Los textos de una huella, descifrados y comprobados: si el contenido no da
        esa huella, alguien lo cambió."""
        if self._blobs is None or self._key is None:
            raise ValueError("sin carpeta de textos o sin clave no se puede mostrar")
        from cryptography.fernet import Fernet

        archivo = self._blobs / f"{blob}.bin"
        if not archivo.is_file():
            # Borrado por retención (plan 85a): la línea queda, el texto no.
            raise ValueError("ese texto se borró al cumplir su plazo de retención")
        claro = Fernet(self._key).decrypt(archivo.read_bytes())
        if hashlib.sha256(claro).hexdigest() != blob:
            raise ValueError("el texto guardado no corresponde a su huella")
        datos = json.loads(claro)
        return dict(datos)

    def purge_texts(self, cutoff: datetime, *, actor: Professional) -> int:
        """Borra los textos cifrados de las líneas anteriores a `cutoff` (plan 85a: la
        retención declarada en el pack). Las líneas quedan: la auditoría no pierde el
        rastro, solo el contenido. Si borra algo, lo dice con una línea propia, firmada
        por quien la provoca; si no hay nada que borrar, no escribe nada."""
        if self._blobs is None:
            return 0
        borrados = 0
        for linea in self._chain.lines():
            huella = linea.get("blob", "")
            if not huella or datetime.fromisoformat(linea["at"]) >= cutoff:
                continue
            archivo = self._blobs / f"{huella}.bin"
            if archivo.is_file():
                archivo.unlink()
                borrados += 1
        if borrados:
            self.record(
                "texts_purged",
                actor=actor,
                data={"count": str(borrados), "before": cutoff.isoformat(timespec="seconds")},
            )
        return borrados

    def lines(self) -> tuple[dict[str, str], ...]:
        return self._chain.lines()

    def verify(self) -> tuple[str, ...]:
        return self._chain.verify()
