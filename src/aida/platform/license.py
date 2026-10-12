"""Qué brazos tiene activos este cliente, y hasta cuándo (plan 100).

Una licencia es un JSON con su contenido (`payload`) y una firma Ed25519 sobre ese
contenido serializado con claves ordenadas. AIDA la verifica sin conexión con la clave
pública de `_public_key.py`, y la clave privada no viaja nunca con AIDA: la tiene quien
vende, en `tools/licencias.py`.

**Límite que queda escrito**: en una aplicación Python local, la licencia disuade, no
blinda. Quien edite el ejecutable puede quitar la comprobación. Lo que sí garantiza es
que no se puede fabricar ni alargar una licencia sin la clave privada.

Caducada, **solo lectura**: el brazo no empieza trabajo nuevo, pero nada de lo
registrado se borra ni se bloquea; la auditoría y las autorizaciones son evidencia.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aida.platform import _public_key

WARN_DAYS = 15
"""Desde cuántos días antes de caducar se avisa. Decisión del producto (debate de la
100), no de cada despliegue."""


@dataclass(frozen=True, slots=True)
class License:
    id: str
    client: str
    arms: tuple[str, ...]
    issued: date
    expires: date


@dataclass(frozen=True, slots=True)
class LicenseState:
    """Lo que la plataforma sabe de la licencia de este equipo."""

    status: str
    """`valid`, `expiring` (vigente y en los últimos `WARN_DAYS`), `expired`, `missing`
    o `invalid` (no se puede leer, o la firma no es de nuestra clave)."""
    license: License | None = None
    days_left: int = 0

    def allows(self, arm_id: str) -> bool:
        """Si el brazo puede **empezar trabajo nuevo**. Caducada, no; leer lo ya
        registrado no pasa por aquí."""
        return (
            self.status in ("valid", "expiring")
            and self.license is not None
            and arm_id in self.license.arms
        )

    def licensed(self, arm_id: str) -> bool:
        """Si el brazo está en la licencia, vigente o caducada: un brazo caducado se
        sigue enseñando, en solo lectura."""
        return self.license is not None and arm_id in self.license.arms and self.status in (
            "valid", "expiring", "expired",
        )


def canonical(payload: dict[str, object]) -> bytes:
    """Lo que se firma: el contenido con claves ordenadas y sin espacios. Una sola forma
    de escribirlo, o la firma dependería de quién lo serializó."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _verifier(public_key: str | None) -> Callable[[bytes, bytes], bool] | None:
    if not public_key:
        return None
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    clave = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key))

    def verifica(firma: bytes, datos: bytes) -> bool:
        try:
            clave.verify(firma, datos)
        except InvalidSignature:
            return False
        return True

    return verifica


def read_license(
    text: str, *, today: date, public_key: str | None = None
) -> LicenseState:
    """El estado de una licencia, a partir de su texto. `public_key` se lee de
    `_public_key` en cada llamada si no se pasa: así la batería la sustituye sin tocar
    el código."""
    clave = public_key if public_key is not None else _public_key.PUBLIC_KEY
    verifica = _verifier(clave)
    try:
        crudo = json.loads(text)
        payload = crudo["payload"]
        firma = base64.b64decode(crudo["signature"])
        if not isinstance(payload, dict):
            raise ValueError("payload")
        licencia = License(
            id=str(payload["id"]),
            client=str(payload["client"]),
            arms=tuple(sorted(str(a) for a in payload["arms"])),
            issued=date.fromisoformat(str(payload["issued"])),
            expires=date.fromisoformat(str(payload["expires"])),
        )
    except (ValueError, KeyError, TypeError):
        return LicenseState(status="invalid")
    if verifica is None or not verifica(firma, canonical(payload)):
        return LicenseState(status="invalid")
    dias = (licencia.expires - today).days
    if dias < 0:
        return LicenseState(status="expired", license=licencia, days_left=dias)
    estado = "expiring" if dias <= WARN_DAYS else "valid"
    return LicenseState(status=estado, license=licencia, days_left=dias)


def load_license(
    path: Path, *, today: date | None = None, public_key: str | None = None
) -> LicenseState:
    if not path.is_file():
        return LicenseState(status="missing")
    return read_license(
        path.read_text(encoding="utf-8"), today=today or date.today(), public_key=public_key
    )
