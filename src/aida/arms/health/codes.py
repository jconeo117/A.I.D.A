"""Catálogo de códigos diagnósticos, cargado de DOS archivos y validado
cruzado entre ellos — nunca se elige uno en silencio si discrepan.

`codes.json` (icdcode.info, un agregador) y `cie-10.csv` (icdcode.info +
deis.cl, el Departamento de Estadísticas e Información de Salud de Chile).
Ninguna de las dos es la publicación original de la OMS. Medido sobre los
archivos de esta unidad: cero conflictos donde se solapan (11268 códigos
compartidos, idénticos en código y descripción) y 3230 códigos que solo
declara `deis.cl` — la fuente más autorizada de las dos. Si algún día uno
de los dos archivos cambia y empiezan a discrepar, `load_catalog` revienta
con la lista completa de discrepancias: un catálogo médico no puede elegir
en silencio cuál de dos versiones contradictorias es la buena.

Ese es uno de dos caminos. El otro, `load_reference_catalog`, carga una **sola**
fuente ya autorizada —el catálogo que publica la autoridad sanitaria del país
donde se despliega— y no valida cruzado porque no hay con qué cruzar. Cuál de
los dos se usa, y con qué archivos, lo declara el pack (`matching.yaml`), no el
código: el mismo diagnóstico se reporta con un código distinto según el país.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


class CatalogConflictError(ValueError):
    """Un mismo código tiene descripciones distintas en las dos fuentes."""


@dataclass(frozen=True, slots=True)
class DiagnosticCode:
    code: str
    description: str
    level: int
    source: str
    """Fuente de origen: 'icdcode.info', 'deis.cl', o 'ambas' si las dos lo declaran."""


def _load_json(path: Path) -> dict[str, tuple[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {item["code"]: (item["description"].strip(), item["level"]) for item in raw}


def _load_csv(path: Path) -> dict[str, tuple[str, int, str]]:
    result: dict[str, tuple[str, int, str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            result[row["code"]] = (row["description"].strip(), int(row["level"]), row["source"])
    return result


def load_catalog(json_path: Path, csv_path: Path) -> tuple[DiagnosticCode, ...]:
    """Une las dos fuentes. Revienta con `CatalogConflictError` si algún
    código aparece en las dos con descripciones distintas — lista TODAS
    las discrepancias encontradas, no solo la primera (mismo criterio que
    `pack.py` al validar un pack)."""
    from_json = _load_json(json_path)
    from_csv = _load_csv(csv_path)

    conflicts: list[str] = []
    for code in sorted(set(from_json) & set(from_csv)):
        desc_json, _ = from_json[code]
        desc_csv, _, _ = from_csv[code]
        if desc_json.lower() != desc_csv.lower():
            conflicts.append(
                f"{code!r}: json dice {desc_json!r}, csv dice {desc_csv!r}"
            )
    if conflicts:
        raise CatalogConflictError(
            "el catálogo CIE-10 discrepa entre fuentes:\n"
            + "\n".join(f"  - {c}" for c in conflicts)
        )

    merged: dict[str, DiagnosticCode] = {}
    for code in sorted(from_csv):
        desc, level, source = from_csv[code]
        merged[code] = DiagnosticCode(code=code, description=desc, level=level, source=source)
    for code in sorted(from_json):
        desc, level = from_json[code]
        if code in merged:
            existing = merged[code]
            merged[code] = DiagnosticCode(
                code=code, description=existing.description, level=existing.level,
                source="ambas",
            )
        else:
            merged[code] = DiagnosticCode(
                code=code, description=desc, level=level, source="icdcode.info"
            )

    return tuple(merged[code] for code in sorted(merged))


def load_reference_catalog(path: Path) -> tuple[DiagnosticCode, ...]:
    """Una sola fuente, ya autorizada: el catálogo que publica la autoridad
    sanitaria del país donde se despliega. No hay validación cruzada porque no
    hay con qué cruzar, y tampoco hace falta: `load_catalog` la necesita porque
    une dos agregadores que pueden discrepar, y aquí no se une nada.

    Devuelto ordenado por código, igual que `load_catalog`: es lo que hace
    determinista el desempate del cotejo.
    """
    rows = _load_csv(path)
    return tuple(
        DiagnosticCode(
            code=code, description=rows[code][0], level=rows[code][1], source=rows[code][2]
        )
        for code in sorted(rows)
    )
