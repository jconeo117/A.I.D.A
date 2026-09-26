"""Pruebas unitarias de segmentación de historia clínica por secciones (Plan 64).

Cubre §B.1 de `.claude/workflow/64-plan-historia-por-secciones.md`:
1. Lo que va antes del primer encabezado no sale: ni el centro, ni el sexo, ni el identificador.
2. Una sección codificable aporta sus líneas; una no codificable, ninguna.
3. Los prefijos de ignore se descartan aunque estén bajo una sección codificable.
4. Sin ningún encabezado reconocido, se devuelven todas las líneas (regla de seguridad).
5. Los encabezados se reconocen sin tildes y en cualquier caja.
6. Los encabezados se reconocen por prefijo (ej. con dos puntos al final).
7. load_history_layout valida formato/claves requeridas y carga el pack real con 10 secciones.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aida.arms.health.history import (
    HistoryLayout,
    HistorySection,
    codable_history_lines,
    load_history_layout,
)


@pytest.fixture
def sample_layout() -> HistoryLayout:
    return HistoryLayout(
        sections=(
            HistorySection(heading="ANTECEDENTES PERSONALES", codable=True),
            HistorySection(heading="DIAGNOSTICOS", codable=True),
            HistorySection(heading="ANTECEDENTES FAMILIARES", codable=False),
            HistorySection(heading="MEDICACION ACTUAL", codable=False),
        ),
        ignore=("PAGINA", "DOCUMENTO GENERADO", "FECHA:"),
    )


def test_history_lineas_antes_del_primer_encabezado_no_salen(
    sample_layout: HistoryLayout,
) -> None:
    """Caso 1 (§B.1): Lo que va antes del primer encabezado no se coteja
    (centro, sexo, edad, fecha, identificador)."""
    texto = """Centro Médico Demo
Identificador: 12345
Edad: 58 años
Sexo: Femenino
Fecha: 24/09/2026
ANTECEDENTES PERSONALES
Diabetes mellitus tipo 2
"""
    lineas = codable_history_lines(texto, sample_layout)
    assert lineas == ("Diabetes mellitus tipo 2",)
    assert "Centro Médico Demo" not in lineas
    assert "Sexo: Femenino" not in lineas


def test_history_seccion_codificable_aporta_no_codificable_no(
    sample_layout: HistoryLayout,
) -> None:
    """Caso 2 (§B.1): Una sección codificable aporta sus líneas; una no
    codificable (ej. antecedentes familiares o medicación), ninguna."""
    texto = """ANTECEDENTES PERSONALES
Hipertensión arterial
DIAGNOSTICOS
Reflujo gastroesofágico
ANTECEDENTES FAMILIARES
Madre con hipertensión esencial
Padre con infarto
MEDICACION ACTUAL
Losartán 50mg cada 12 horas
"""
    lineas = codable_history_lines(texto, sample_layout)
    assert lineas == (
        "Hipertensión arterial",
        "Reflujo gastroesofágico",
    )
    assert "Madre con hipertensión esencial" not in lineas
    assert "Losartán 50mg cada 12 horas" not in lineas


def test_history_prefijos_ignore_se_descartan_incluso_en_seccion_codificable(
    sample_layout: HistoryLayout,
) -> None:
    """Caso 3 (§B.1): Los prefijos de ignore se descartan aunque caigan bajo
    una sección codificable (ej. Página 2 de 2, fechas sueltas)."""
    texto = """ANTECEDENTES PERSONALES
Diabetes mellitus tipo 2
Página 2 de 2
FECHA: 2026-09-24
Obesidad grado I
DOCUMENTO GENERADO POR SISTEMA
"""
    lineas = codable_history_lines(texto, sample_layout)
    assert lineas == (
        "Diabetes mellitus tipo 2",
        "Obesidad grado I",
    )


def test_history_sin_encabezados_reconocidos_devuelve_todas_las_lineas(
    sample_layout: HistoryLayout,
) -> None:
    """Caso 4 (§B.1): Si no se reconoce ningún encabezado, se devuelven todas
    las líneas para no quedarse ciego ante un formato no previsto."""
    texto = """Paciente acude a control clínico general.
Refiere dolor precordial opresivo ocasional.
Se solicita electrocardiograma y enzimas cardíacas.
"""
    lineas = codable_history_lines(texto, sample_layout)
    assert lineas == (
        "Paciente acude a control clínico general.",
        "Refiere dolor precordial opresivo ocasional.",
        "Se solicita electrocardiograma y enzimas cardíacas.",
    )


def test_history_encabezados_insensibles_a_caja_y_tildes() -> None:
    """Caso 5 (§B.1): Los encabezados se reconocen sin tildes y en cualquier caja
    (ej. 'Medicación Actual', 'MEDICACIÓN ACTUAL', 'medicacion actual')."""
    layout = HistoryLayout(
        sections=(
            HistorySection(heading="MEDICACION ACTUAL", codable=False),
            HistorySection(heading="DIAGNOSTICOS", codable=True),
        ),
        ignore=(),
    )
    texto = """Medicación Actual
Paracetamol 1g
Diagnósticos
Cefalea tensional
"""
    lineas = codable_history_lines(texto, layout)
    assert lineas == ("Cefalea tensional",)


def test_history_encabezados_se_reconocen_por_prefijo(
    sample_layout: HistoryLayout,
) -> None:
    """Caso 6 (§B.1): Los encabezados se reconocen por prefijo, admitiendo dos puntos,
    guiones u otros añadidos al final de la línea."""
    texto = """ANTECEDENTES PERSONALES:
Hipertensión esencial
ANTECEDENTES FAMILIARES - NO CONTRIBUTORIOS
Padre sano
"""
    lineas = codable_history_lines(texto, sample_layout)
    assert lineas == ("Hipertensión esencial",)


def test_load_history_layout_validaciones_y_pack_real(tmp_path: Path) -> None:
    """Caso 7 (§B.1): load_history_layout valida formato y claves obligatorias
    ('sections', 'heading', 'codable', 'ignore'), y el pack real carga con 10 secciones."""
    # 1. Carga del pack real
    real = load_history_layout()
    assert len(real.sections) == 10
    codables = [s.heading for s in real.sections if s.codable]
    no_codables = [s.heading for s in real.sections if not s.codable]
    assert "ANTECEDENTES PERSONALES" in codables
    assert "DIAGNOSTICOS" in codables
    assert "PROBLEMAS ACTIVOS" in codables
    assert "ANTECEDENTES QUIRURGICOS" in codables
    assert "ANTECEDENTES FAMILIARES" in no_codables
    assert "ALERGIAS" in no_codables
    assert "MEDICACION ACTUAL" in no_codables
    assert "TRATAMIENTO" in no_codables
    assert "CONSULTAS PREVIAS" in no_codables
    assert "HISTORIA CLINICA" in no_codables
    assert len(real.ignore) == 7

    # 2. Archivo sin 'sections'
    f_sin_sections = tmp_path / "sin_sections.yaml"
    f_sin_sections.write_text("ignore: ['PAGINA']\n", encoding="utf-8")
    with pytest.raises(ValueError, match="falta 'sections'"):
        load_history_layout(f_sin_sections)

    # 3. 'sections' vacío
    f_vacio = tmp_path / "vacio.yaml"
    f_vacio.write_text("sections: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'sections' no puede estar vacío"):
        load_history_layout(f_vacio)

    # 4. Sección sin 'heading' o sin 'codable'
    f_sin_heading = tmp_path / "sin_heading.yaml"
    f_sin_heading.write_text(
        yaml.safe_dump({"sections": [{"codable": True}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'heading'.*obligatorios"):
        load_history_layout(f_sin_heading)

    f_sin_codable = tmp_path / "sin_codable.yaml"
    f_sin_codable.write_text(
        yaml.safe_dump({"sections": [{"heading": "ANTECEDENTES"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'heading'.*obligatorios"):
        load_history_layout(f_sin_codable)

    # 5. Entrada que no es mapeo
    f_no_dict = tmp_path / "no_dict.yaml"
    f_no_dict.write_text("sections: ['ANTECEDENTES']\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cada sección debe ser un mapeo"):
        load_history_layout(f_no_dict)

    # 6. 'ignore' no es lista de cadenas
    f_bad_ignore = tmp_path / "bad_ignore.yaml"
    f_bad_ignore.write_text(
        yaml.safe_dump({"sections": [{"heading": "A", "codable": True}], "ignore": [123]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'ignore' debe ser una lista de cadenas"):
        load_history_layout(f_bad_ignore)
