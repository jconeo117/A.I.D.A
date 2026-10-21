"""Dónde está el relé y qué número de Meta es el del cliente. No lleva secretos: los tokens van
en `keys.py`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from aida.resources import user_dir


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    relay_url: str
    phone_number_id: str
    graph_url: str
    """Con la versión de la API, sin barra final: `https://graph.facebook.com/v21.0`."""
    poll_seconds: int
    country_code: str = ""
    """El prefijo del país sin `+` (`"57"`). Meta entrega los números con él; el mostrador los
    teclea sin él (102b, decisión 1)."""


def load_channel_config(path: Path | None = None) -> ChannelConfig | None:
    """`None` si no hay archivo: sin canal configurado AIDA trabaja igual. Si el archivo existe
    y no cuadra, revienta: un canal medio configurado se nota cuando alguien ya escribió."""
    ruta = path if path is not None else user_dir("config") / "channel.yaml"
    if not ruta.is_file():
        return None
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ruta} debe ser un mapeo YAML")
    for campo in ("relay_url", "phone_number_id", "graph_url"):
        if not isinstance(raw.get(campo), str) or not raw[campo].strip():
            raise ValueError(f"falta '{campo}' en {ruta}")
    segundos = raw.get("poll_seconds", 20)
    if not isinstance(segundos, int) or isinstance(segundos, bool) or segundos <= 0:
        raise ValueError(f"'poll_seconds' debe ser un entero positivo en {ruta}")
    pais = str(raw.get("country_code", "")).strip()
    if pais and not pais.isdigit():
        raise ValueError(f"'country_code' deben ser solo dígitos en {ruta}")
    return ChannelConfig(
        relay_url=raw["relay_url"].strip().rstrip("/"),
        phone_number_id=raw["phone_number_id"].strip(),
        graph_url=raw["graph_url"].strip().rstrip("/"),
        poll_seconds=segundos,
        country_code=pais,
    )
