"""Qué parte de una historia clínica impresa puede llevar un código.

Una historia viene por secciones, y no todas hablan del paciente: los
antecedentes familiares hablan de su madre, y la cabecera con la fecha y el sexo
no habla de nada. Cotejarlas propone diagnósticos que nadie tiene — medido,
«Sexo: Femenino» llevaba a «uretrocele femenino» y «Madre con hipertensión
esencial» atribuía al paciente la hipertensión de su madre.

Qué encabezado es codificable vive en datos: cada software de historia clínica
imprime los suyos, y eso cambia con el despliegue, no con el código.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

from aida.resources import data_root

_RAIZ = data_root()
_HISTORIA_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "history.yaml"


@dataclass(frozen=True, slots=True)
class HistorySection:
    heading: str
    codable: bool


@dataclass(frozen=True, slots=True)
class HistoryLayout:
    sections: tuple[HistorySection, ...]
    ignore: tuple[str, ...]
    """Prefijos de líneas que no describen ningún hecho clínico —cabeceras, pies
    de página, fechas sueltas—, aunque caigan bajo un encabezado codificable."""


def _plano(texto: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", texto.upper())
        if unicodedata.category(c) != "Mn"
    ).strip()


def load_history_layout(path: Path | None = None) -> HistoryLayout:
    ruta = path if path is not None else _HISTORIA_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("sections"), list):
        raise ValueError(f"falta 'sections' (lista) en {ruta}")
    secciones: list[HistorySection] = []
    for entrada in raw["sections"]:
        if not isinstance(entrada, dict):
            raise ValueError(f"cada sección debe ser un mapeo, en {ruta}")
        heading, codable = entrada.get("heading"), entrada.get("codable")
        if not isinstance(heading, str) or not isinstance(codable, bool):
            raise ValueError(f"'heading' (texto) y 'codable' (sí o no) son obligatorios, en {ruta}")
        secciones.append(HistorySection(heading=_plano(heading), codable=codable))
    if not secciones:
        raise ValueError(f"'sections' no puede estar vacío, en {ruta}")
    ignorar = raw.get("ignore", [])
    if not isinstance(ignorar, list) or not all(isinstance(i, str) for i in ignorar):
        raise ValueError(f"'ignore' debe ser una lista de cadenas, en {ruta}")
    return HistoryLayout(
        sections=tuple(secciones), ignore=tuple(_plano(i) for i in ignorar)
    )


def codable_history_lines(text: str, layout: HistoryLayout) -> tuple[str, ...]:
    """Las líneas de la historia que pueden llevar código, según el encabezado
    bajo el que estén.

    Lo que va **antes** del primer encabezado —cabecera, identificador, fecha—
    no se coteja: no es clínico. Y si el documento no trae ningún encabezado
    conocido, se devuelven **todas** las líneas: mejor cotejar de más que
    quedarse ciego ante un formato que no reconocemos.
    """
    lineas = [linea.strip() for linea in text.splitlines() if linea.strip()]
    codificables: list[str] = []
    reconocido = False
    activa: HistorySection | None = None
    for linea in lineas:
        plano = _plano(linea)
        if any(plano.startswith(prefijo) for prefijo in layout.ignore):
            continue
        encabezado = next((s for s in layout.sections if plano.startswith(s.heading)), None)
        if encabezado is not None:
            activa = encabezado
            reconocido = True
            continue
        if activa is not None and activa.codable:
            codificables.append(linea)
    if not reconocido:
        return tuple(lineas)
    return tuple(codificables)
