"""Catálogo de lateralidad: vocabulario cerrado como dato, mismo criterio
que `sections.py` para el guardián `I-073`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from aida.resources import data_root

_RAIZ = data_root()
_LATERALITY_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "extraction.yaml"


@dataclass(frozen=True, slots=True)
class LateralityCatalog:
    """`categories` conserva el orden de declaración: en una frase ambigua,
    gana la primera categoría cuyo término aparece, no la última."""

    categories: tuple[tuple[str, tuple[str, ...]], ...]

    def find(self, text: str) -> str | None:
        lowered = text.lower()
        for category_id, terms in self.categories:
            if any(term in lowered for term in terms):
                return category_id
        return None


def load_laterality(path: Path | None = None) -> LateralityCatalog:
    ruta = path if path is not None else _LATERALITY_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("laterality"), dict):
        raise ValueError(f"falta 'laterality' (mapeo) en {ruta}")
    categories: list[tuple[str, tuple[str, ...]]] = []
    for category_id, terms in raw["laterality"].items():
        if not isinstance(terms, list) or not terms or not all(isinstance(t, str) for t in terms):
            raise ValueError(f"'{category_id}' debe traer una lista no vacía de cadenas, en {ruta}")
        categories.append((category_id, tuple(t.lower() for t in terms)))
    if not categories:
        raise ValueError(f"'laterality' no puede estar vacío, en {ruta}")
    return LateralityCatalog(categories=tuple(categories))
