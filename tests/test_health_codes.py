"""Pruebas unitarias del catálogo de códigos diagnósticos CIE-10 (`aida.arms.health.codes`).

Cubre la unidad U9 según la especificación de `.claude/workflow/26-plan-u9.md` §6.2.
Verifica de forma aislada y rigurosa sobre los datos REALES y fuentes sintéticas:
1. Carga exhaustiva sobre `data/health/cie10/codes.json` y `data/health/cie10/cie-10.csv`:
   - Exactamente 14498 registros totales.
   - Exactamente 11268 registros con `source == "ambas"`.
   - Exactamente 3230 registros con `source == "deis.cl"`.
   - Ordenamiento estricto ascendente por `code`.
2. Detección obligatoria de conflictos cruzados (`CatalogConflictError`) cuando un mismo
   código presenta descripciones distintas entre fuentes, listando el código infractor.
3. Tolerancia a variaciones exclusivas de caja (mayúsculas vs minúsculas): descripciones
   que difieren solo en mayúsculas no se consideran discrepancia médica sustantiva.
   (Protege la mutación M2).
4. Unión sin conflicto entre fuentes disyuntas (sin solapamiento de códigos).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aida.arms.health.codes import CatalogConflictError, DiagnosticCode, load_catalog

CIE10_JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "health" / "cie10" / "codes.json"
CIE10_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "health" / "cie10" / "cie-10.csv"


# ---------------------------------------------------------------------------
# Verificación sobre archivos reales (§6.2.1, §6.2.2, §6.2.3)
# ---------------------------------------------------------------------------


def test_load_catalog_real_files_counts_and_sources() -> None:
    """Valida los recuentos exactos del catálogo real cargado desde disco.

    Garantiza que el catálogo unificado contiene 14498 registros:
    - 11268 códigos comunes presentes en ambas fuentes ('ambas').
    - 3230 códigos exclusivos aportados por el DEIS de Chile ('deis.cl').
    - 0 códigos exclusivos de 'icdcode.info' en el cruce final.
    """
    assert CIE10_JSON_PATH.exists(), f"Falta el archivo real {CIE10_JSON_PATH}"
    assert CIE10_CSV_PATH.exists(), f"Falta el archivo real {CIE10_CSV_PATH}"

    catalog = load_catalog(CIE10_JSON_PATH, CIE10_CSV_PATH)

    assert len(catalog) == 14498

    ambas_count = sum(1 for item in catalog if item.source == "ambas")
    deis_count = sum(1 for item in catalog if item.source == "deis.cl")
    icd_count = sum(1 for item in catalog if item.source == "icdcode.info")

    assert ambas_count == 11268
    assert deis_count == 3230
    assert icd_count == 0


def test_load_catalog_real_files_sorted_ascending_by_code() -> None:
    """Asegura que el catálogo devuelto está estrictamente ordenado por código ascendente,
    invariante necesaria para el desempate determinista en búsquedas posteriores.
    """
    catalog = load_catalog(CIE10_JSON_PATH, CIE10_CSV_PATH)
    codes = [item.code for item in catalog]
    assert codes == sorted(codes)


def test_load_catalog_entries_are_diagnostic_code_instances() -> None:
    """Comprueba que cada elemento devuelto es una instancia inmutable DiagnosticCode
    con tipos válidos en todos sus atributos.
    """
    catalog = load_catalog(CIE10_JSON_PATH, CIE10_CSV_PATH)
    sample = catalog[0]
    assert isinstance(sample, DiagnosticCode)
    assert isinstance(sample.code, str)
    assert isinstance(sample.description, str)
    assert isinstance(sample.level, int)
    assert isinstance(sample.source, str)


# ---------------------------------------------------------------------------
# Detección de conflictos cruzados (§6.2.4)
# ---------------------------------------------------------------------------


def test_load_catalog_conflict_raises_catalog_conflict_error(tmp_path: Path) -> None:
    """Dos fuentes con el mismo código pero descripciones divergentes deben abortar
    la carga levantando `CatalogConflictError`, y el mensaje de error debe explicitar
    el código discordante y las versiones discrepantes.
    """
    json_file = tmp_path / "codes.json"
    csv_file = tmp_path / "cie-10.csv"

    json_file.write_text(
        json.dumps([{"code": "A00", "description": "Cólera debido a Vibrio cholerae", "level": 3}]),
        encoding="utf-8",
    )
    csv_file.write_text(
        "code,description,level,source\n"
        "A00,Infección intestinal por otra bacteria,3,icdcode.info\n",
        encoding="utf-8",
    )

    with pytest.raises(CatalogConflictError) as exc_info:
        load_catalog(json_file, csv_file)

    msg = str(exc_info.value)
    assert "A00" in msg
    assert "Cólera debido a Vibrio cholerae" in msg
    assert "Infección intestinal por otra bacteria" in msg


def test_load_catalog_multiple_conflicts_lists_all_offenders(tmp_path: Path) -> None:
    """Si existen múltiples discrepancias, deben listarse todas en el mensaje
    de error para evitar rondas iterativas ciegas de corrección.
    """
    json_file = tmp_path / "codes.json"
    csv_file = tmp_path / "cie-10.csv"

    json_file.write_text(
        json.dumps(
            [
                {"code": "A01", "description": "Fiebre tifoidea en json", "level": 3},
                {"code": "B01", "description": "Varicela en json", "level": 3},
            ]
        ),
        encoding="utf-8",
    )
    csv_file.write_text(
        "code,description,level,source\n"
        "A01,Fiebre tifoidea en csv,3,icdcode.info\n"
        "B01,Varicela en csv,3,icdcode.info\n",
        encoding="utf-8",
    )

    with pytest.raises(CatalogConflictError) as exc_info:
        load_catalog(json_file, csv_file)

    msg = str(exc_info.value)
    assert "A01" in msg
    assert "B01" in msg


# ---------------------------------------------------------------------------
# Invariancia de caja y protección de mutación M2 (§6.2.4 variante)
# ---------------------------------------------------------------------------


def test_load_catalog_case_difference_only_does_not_raise_conflict(tmp_path: Path) -> None:
    """Variaciones que difieren únicamente en mayúsculas/minúsculas no constituyen
    un conflicto diagnóstico semántico y deben aceptarse sin lanzar excepción.

    Protege la mutación M2 (§6.2): si en `load_catalog` se reemplaza
    `if desc_json.lower() != desc_csv.lower():` por `if desc_json != desc_csv:`,
    este caso pasa a considerarse erróneamente un conflicto y el test falla (RED).
    """
    json_file = tmp_path / "codes.json"
    csv_file = tmp_path / "cie-10.csv"

    # Misma descripción excepto por mayúsculas/minúsculas
    json_file.write_text(
        json.dumps(
            [{"code": "C01", "description": "TUMOR MALIGNO DE LA BASE DE LA LENGUA", "level": 3}]
        ),
        encoding="utf-8",
    )
    csv_file.write_text(
        "code,description,level,source\n"
        "C01,tumor maligno de la base de la lengua,3,icdcode.info\n",
        encoding="utf-8",
    )

    catalog = load_catalog(json_file, csv_file)
    assert len(catalog) == 1
    assert catalog[0].code == "C01"
    assert catalog[0].source == "ambas"


# ---------------------------------------------------------------------------
# Unión de fuentes sin solapamiento (§6.2.5)
# ---------------------------------------------------------------------------


def test_load_catalog_disjoint_sources_unioned_successfully(tmp_path: Path) -> None:
    """Dos fuentes sin solapamiento de códigos forman una unión limpia sin conflicto,
    asignando las fuentes correspondientes a cada código.
    """
    json_file = tmp_path / "codes.json"
    csv_file = tmp_path / "cie-10.csv"

    json_file.write_text(
        json.dumps([{"code": "A00", "description": "Cólera", "level": 3}]),
        encoding="utf-8",
    )
    csv_file.write_text(
        "code,description,level,source\n"
        "B00,Infecciones por virus del herpes,3,deis.cl\n",
        encoding="utf-8",
    )

    catalog = load_catalog(json_file, csv_file)
    assert len(catalog) == 2
    by_code = {item.code: item for item in catalog}

    assert "A00" in by_code
    assert by_code["A00"].source == "icdcode.info"
    assert by_code["A00"].description == "Cólera"

    assert "B00" in by_code
    assert by_code["B00"].source == "deis.cl"
    assert by_code["B00"].description == "Infecciones por virus del herpes"
