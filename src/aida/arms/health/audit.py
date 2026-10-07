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

from aida.arms.health.identity import Professional
from aida.ledger import ChainedLog

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
        from aida.arms.health.consent import workstation as puesto

        self._chain = ChainedLog(path)
        self._blobs = blobs
        self._key = key
        self._now = now
        self._workstation = workstation if workstation is not None else puesto

    def record(
        self,
        kind: str,
        *,
        actor: Professional | None,
        owner: str = "",
        data: Mapping[str, str] | None = None,
        texts: Mapping[str, object] | None = None,
    ) -> dict[str, str]:
        """Añade un hecho. `texts` se cifra aparte y la línea lleva su huella; sin
        clave o sin carpeta de textos, no se guardan (y la línea lo dice con `blob`
        vacío)."""
        if kind not in KINDS:
            raise ValueError(f"clase de hecho de auditoría desconocida: {kind!r}")
        blob = self._store(texts) if texts else ""
        return self._chain.append(
            {
                "at": self._now().isoformat(timespec="seconds"),
                "kind": kind,
                "actor": actor.id if actor else "",
                "actor_name": actor.label() if actor else "",
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

        claro = Fernet(self._key).decrypt((self._blobs / f"{blob}.bin").read_bytes())
        if hashlib.sha256(claro).hexdigest() != blob:
            raise ValueError("el texto guardado no corresponde a su huella")
        datos = json.loads(claro)
        return dict(datos)

    def lines(self) -> tuple[dict[str, str], ...]:
        return self._chain.lines()

    def verify(self) -> tuple[str, ...]:
        return self._chain.verify()
