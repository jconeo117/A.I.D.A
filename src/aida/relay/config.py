"""La configuración del relé. Los secretos de Meta llegan por entorno (en la nube, desde un
gestor de secretos); el archivo de inquilinos no lleva ninguno: claves públicas y la huella
SHA-256 del token con que cada equipo recoge. Quien lea ese archivo no puede ni abrir lo
guardado ni recogerlo."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Tenant:
    id: str
    phone_number_id: str
    public_key: bytes
    token_sha256: str


@dataclass(frozen=True, slots=True)
class RelayConfig:
    app_secret: str
    verify_token: str
    tenants: tuple[Tenant, ...]

    def by_number(self, phone_number_id: str) -> Tenant | None:
        return next((t for t in self.tenants if t.phone_number_id == phone_number_id), None)

    def by_token(self, token: str) -> Tenant | None:
        """Compara con todos, sin cortar en el primero: el tiempo no dice cuál casi acierta."""
        huella = hashlib.sha256(token.encode("utf-8")).hexdigest()
        hallado: Tenant | None = None
        for t in self.tenants:
            if hmac.compare_digest(huella, t.token_sha256):
                hallado = t
        return hallado


def load_relay_config(path: Path, env: Mapping[str, str]) -> RelayConfig:
    secreto = env.get("AIDA_RELAY_APP_SECRET", "")
    verificacion = env.get("AIDA_RELAY_VERIFY_TOKEN", "")
    if not secreto or not verificacion:
        raise ValueError("faltan AIDA_RELAY_APP_SECRET o AIDA_RELAY_VERIFY_TOKEN")
    raw = json.loads(path.read_text(encoding="utf-8"))
    inquilinos: list[Tenant] = []
    for t in raw.get("tenants", []):
        clave = base64.b64decode(str(t["public_key"]), validate=True)
        if len(clave) != 32:
            raise ValueError(f"la clave pública de {t['id']!r} no tiene 32 bytes")
        inquilinos.append(
            Tenant(
                id=str(t["id"]),
                phone_number_id=str(t["phone_number_id"]),
                public_key=clave,
                token_sha256=str(t["token_sha256"]).lower(),
            )
        )
    for campo in ("id", "phone_number_id", "token_sha256"):
        valores = [getattr(t, campo) for t in inquilinos]
        if len(valores) != len(set(valores)):
            raise ValueError(f"'{campo}' repetido en {path}")
    if not inquilinos:
        raise ValueError(f"{path} no declara ningún inquilino")
    return RelayConfig(app_secret=secreto, verify_token=verificacion, tenants=tuple(inquilinos))
