"""Carga de la redacción por defecto para consultas desde archivo de datos.

Los textos específicos del dominio se inyectan como datos desde un archivo YAML
para preservar el agnosticismo y cumplir las reglas de vocabulario de `arms/health/`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from aida.arms.health.consultation import SummaryWording

_RAIZ = Path(__file__).resolve().parents[4]
_WORDING_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "wording.yaml"


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
        "professional_label",
        "patient_label",
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
        professional_label=raw["professional_label"],
        patient_label=raw["patient_label"],
    )


def switch_to_consult_phrases(path: Path | None = None) -> tuple[str, ...]:
    """Frases del operador para entrar en modo consulta. Viven aquí y no
    en `voice/` ni en `run.py`: es vocabulario del dominio, y `arms/health/` es
    el único sitio exento del guardián (`I-073`)."""
    ruta = path if path is not None else _WORDING_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "subject_noun" not in raw:
        raise KeyError("falta 'subject_noun' en wording")
    subject = raw["subject_noun"]
    if not isinstance(subject, str):
        raise ValueError("falta 'subject_noun' en wording o no es una cadena válida")
    return (
        "modo consulta",
        "entra en modo consulta",
        "vamos a empezar la consulta",
        f"escucha al {subject}",
    )


