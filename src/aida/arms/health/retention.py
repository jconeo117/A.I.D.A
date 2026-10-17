"""Los plazos de retención del brazo, leídos del pack (plan 85a).

Antes estaban escritos en el código (24 h, 1 h) o no existían (los textos de la
auditoría). Una política de borrado tiene que estar declarada y poder cambiarse por
despliegue sin tocar el código.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from aida.resources import data_root

_POR_DEFECTO = data_root() / "packs" / "health" / "consultorio" / "retention.yaml"


@dataclass(frozen=True, slots=True)
class Retention:
    audit_texts_days: int
    transcripts_hours: int
    summaries_hours: int


def load_retention(path: Path | None = None) -> Retention:
    """Revienta si falta un plazo o no es un entero positivo: un plazo a medias se
    descubre cuando ya se ha guardado de más."""
    ruta = path if path is not None else _POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    claves = ("audit_texts_days", "transcripts_hours", "summaries_hours")
    if not isinstance(raw, dict) or not all(
        isinstance(raw.get(c), int) and not isinstance(raw.get(c), bool) and raw[c] > 0
        for c in claves
    ):
        raise ValueError(f"{ruta} necesita {list(claves)} como enteros positivos")
    return Retention(**{c: int(raw[c]) for c in claves})
