"""Lo que el relé entiende de Meta: la firma del webhook y las entradas de texto.

La firma se comprueba sobre el cuerpo **en bytes, tal como llegó**: volver a serializar el JSON
cambiaría espacios u orden y ninguna firma cuadraría. El prototipo no la comprobaba; sin ella,
cualquiera que conozca la dirección del relé puede inyectar entradas.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

_PREFIX = "sha256="


def valid_signature(app_secret: str, body: bytes, header: str | None) -> bool:
    if not header or not header.startswith(_PREFIX):
        return False
    esperada = hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(esperada, header[len(_PREFIX) :])


@dataclass(frozen=True, slots=True)
class Incoming:
    phone_number_id: str
    payload: bytes
    """JSON canónico (claves ordenadas) de lo que viaja sellado: `wamid`, `from`, `name`,
    `type`, `text`, `timestamp`."""


def _entries(raw: Any) -> list[Incoming]:
    hallados: list[Incoming] = []
    for entrada in raw.get("entry", []):
        for cambio in entrada.get("changes", []):
            if cambio.get("field") != "messages":
                continue
            valor = cambio.get("value", {})
            numero = str(valor.get("metadata", {}).get("phone_number_id", ""))
            nombres = {
                str(c.get("wa_id", "")): str(c.get("profile", {}).get("name", ""))
                for c in valor.get("contacts", [])
            }
            for m in valor.get("messages", []):
                tipo = str(m.get("type", ""))
                remitente = str(m.get("from", ""))
                datos = {
                    "wamid": str(m["id"]),
                    "from": remitente,
                    "name": nombres.get(remitente, ""),
                    "type": tipo,
                    "text": str(m.get("text", {}).get("body", "")) if tipo == "text" else "",
                    "timestamp": str(m["timestamp"]),
                }
                hallados.append(
                    Incoming(
                        phone_number_id=numero,
                        payload=json.dumps(datos, ensure_ascii=False, sort_keys=True).encode(),
                    )
                )
    return hallados


def incoming(body: bytes) -> tuple[Incoming, ...]:
    """Las entradas de un webhook, en el orden en que vienen. Los estados (enviado, entregado,
    leído) no se recogen: nadie los lee hasta la 104. Un cuerpo con otra forma da `()`: Meta
    reintenta lo que no recibe un 200, y un cuerpo raro no debe atascar nada."""
    try:
        return tuple(_entries(json.loads(body)))
    except (ValueError, AttributeError, TypeError, KeyError):
        return ()
