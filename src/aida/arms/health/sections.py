"""Catálogo de secciones: carga y valida desde datos. Vocabulario
de dominio vive aquí como dato de archivo, nunca como literal de código —
mismo criterio que `wording.py` para el guardián `I-073`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_RAIZ = Path(__file__).resolve().parents[4]
_SECTIONS_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "sections.yaml"


@dataclass(frozen=True, slots=True)
class SectionSpec:
    section_id: str
    label: str
    trigger_phrases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SectionCatalog:
    sections: tuple[SectionSpec, ...]
    threshold: float
    fallback: SectionSpec


def load_sections(path: Path | None = None) -> SectionCatalog:
    """Carga y valida el catálogo. `fallback` debe apuntar a un `section_id`
    declarado; los ids deben ser únicos; el orden de declaración se
    conserva — es el orden en que las secciones aparecen en el resumen."""
    ruta = path if path is not None else _SECTIONS_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ruta} debe ser un mapeo YAML")

    threshold = raw.get("threshold")
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise ValueError(f"falta 'threshold' numérico en {ruta}")

    fallback_id = raw.get("fallback")
    if not isinstance(fallback_id, str):
        raise ValueError(f"falta 'fallback' (id de sección) en {ruta}")

    raw_sections = raw.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError(f"falta 'sections' (lista no vacía) en {ruta}")

    sections: list[SectionSpec] = []
    seen: set[str] = set()
    for entry in raw_sections:
        if not isinstance(entry, dict):
            raise ValueError(f"cada entrada de 'sections' debe ser un mapeo, en {ruta}")
        section_id = entry.get("id")
        label = entry.get("label")
        triggers = entry.get("trigger_phrases", [])
        if not isinstance(section_id, str) or not isinstance(label, str):
            raise ValueError(f"'id' y 'label' deben ser cadenas, en {ruta}")
        if not isinstance(triggers, list) or not all(isinstance(t, str) for t in triggers):
            raise ValueError(f"'trigger_phrases' debe ser una lista de cadenas, en {ruta}")
        if section_id in seen:
            raise ValueError(f"id de sección repetido: {section_id!r}, en {ruta}")
        seen.add(section_id)
        sections.append(
            SectionSpec(
                section_id=section_id,
                label=label,
                trigger_phrases=tuple(triggers),
            )
        )

    fallback = next((s for s in sections if s.section_id == fallback_id), None)
    if fallback is None:
        raise ValueError(
            f"'fallback' {fallback_id!r} no está entre las secciones declaradas, en {ruta}"
        )

    return SectionCatalog(
        sections=tuple(sections),
        threshold=float(threshold),
        fallback=fallback,
    )
