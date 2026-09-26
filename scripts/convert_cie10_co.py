"""Convierte la Tabla de Referencia CIE-10 de MinSalud/SISPRO a un CSV plano.

Se ejecuta a mano cuando la autoridad sanitaria publica una versión nueva; el
CSV resultante es lo que entra al repositorio. El catálogo no cambia entre
ejecuciones de AIDA, así que leer la hoja de cálculo en cada arranque sería
pagar una dependencia (y un tiempo de carga) por un dato estático.

Sin dependencias: un `.xlsx` es un ZIP con XML dentro, y eso lo abre la
biblioteca estándar.

Uso:
    python scripts/convert_cie10_co.py <TablaReferencia_CIE10.xlsx> <salida.csv>
"""

from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

COLUMNS = {"table": "A", "code": "B", "name": "C", "enabled": "E"}
"""Columnas por **referencia de celda**, nunca por posición. Medido sobre el
archivo del 2026-09-15: la cabecera declara 22 columnas y las filas de datos
traen entre 14 y 17 celdas, porque el formato omite las vacías. Leer por
posición desplaza las columnas en silencio."""


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{NS}t"))
        for item in root.findall(f"{NS}si")
    ]


def _cells(row: ET.Element, shared: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for cell in row.findall(f"{NS}c"):
        column = "".join(ch for ch in (cell.get("r") or "") if ch.isalpha())
        value = cell.find(f"{NS}v")
        if value is None or value.text is None:
            continue
        values[column] = shared[int(value.text)] if cell.get("t") == "s" else value.text
    return values


def convert(source: Path, destination: Path) -> int:
    with zipfile.ZipFile(source) as archive:
        shared = _shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = sheet.findall(f".//{NS}row")
    entries: list[tuple[str, str]] = []
    for row in rows[1:]:
        values = _cells(row, shared)
        if values.get(COLUMNS["table"]) != "CIE10":
            continue
        # Un código deshabilitado ya no se puede reportar: hoy no hay ninguno,
        # y el filtro existe para el día que lo haya.
        if values.get(COLUMNS["enabled"]) != "SI":
            continue
        code = (values.get(COLUMNS["code"]) or "").strip()
        name = (values.get(COLUMNS["name"]) or "").strip()
        if not code or not name:
            raise ValueError(f"fila sin código o sin nombre: {values!r}")
        entries.append((code, name))
    entries.sort()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("code", "description", "level", "source"))
        for code, name in entries:
            writer.writerow((code, name, "3", "minsalud"))
    return len(entries)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    total = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{total} códigos escritos en {sys.argv[2]}")
