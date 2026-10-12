"""Qué brazos hay (plan 100).

Cada brazo se describe en un archivo de datos (`packs/<brazo>/arm.yaml`): su id, cómo
se llama, qué hace y dónde está su punto de entrada. La plataforma los descubre leyendo
esos archivos y los abre importando ese punto de entrada **cuando hace falta**: ningún
módulo del núcleo importa un brazo. Así la licencia decide qué hay, y un brazo que no
está instalado simplemente no aparece.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aida.resources import data_root


@dataclass(frozen=True, slots=True)
class ArmManifest:
    id: str
    name: str
    description: str
    version: str
    app: str
    """`modulo:funcion` que abre la aplicación del brazo."""

    def entry(self) -> Callable[..., Any]:
        modulo, _sep, funcion = self.app.partition(":")
        objetivo: Callable[..., Any] = getattr(importlib.import_module(modulo), funcion)
        return objetivo


def load_manifest(path: Path) -> ArmManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    claves = ("id", "name", "description", "version", "app")
    if not isinstance(raw, dict) or not all(
        isinstance(raw.get(c), str) and raw[c].strip() for c in claves
    ):
        raise ValueError(f"{path} necesita {list(claves)} no vacías")
    if ":" not in raw["app"]:
        raise ValueError(f"'app' debe ser 'modulo:funcion' en {path}")
    return ArmManifest(**{c: str(raw[c]).strip() for c in claves})


def discover(root: Path | None = None) -> tuple[ArmManifest, ...]:
    """Los brazos instalados, ordenados por id: el orden no puede depender del sistema de
    archivos."""
    base = (root if root is not None else data_root()) / "packs"
    return tuple(
        sorted(
            (load_manifest(p) for p in base.glob("*/arm.yaml")),
            key=lambda m: m.id,
        )
    )
