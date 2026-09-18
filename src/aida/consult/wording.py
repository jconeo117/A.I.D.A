"""Carga de la redacción por defecto para consultas desde archivo de datos.

Los textos específicos del dominio se inyectan como datos desde un archivo YAML
para preservar el agnosticismo y cumplir las reglas de vocabulario de `consult/`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from aida.consult.consultation import SummaryWording

_RAIZ = Path(__file__).resolve().parents[3]
_WORDING_POR_DEFECTO = _RAIZ / "packs" / "consultorio" / "wording.yaml"


def default_summary_wording(path: Path | None = None) -> SummaryWording:
    """Carga y valida la redacción por defecto desde un archivo YAML."""
    ruta = path if path is not None else _WORDING_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"El archivo de redacción {ruta} debe ser un mapeo YAML")

    required_keys = (
        "system_prompt",
        "said_template",
        "history_template",
        "matched_template",
        "no_match_text",
    )
    missing = [key for key in required_keys if key not in raw or not isinstance(raw[key], str)]
    if missing:
        raise ValueError(
            f"Faltan claves requeridas o no son cadenas en {ruta}: {missing}"
        )

    return SummaryWording(
        system_prompt=raw["system_prompt"],
        said_template=raw["said_template"],
        history_template=raw["history_template"],
        matched_template=raw["matched_template"],
        no_match_text=raw["no_match_text"],
    )

