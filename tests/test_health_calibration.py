"""Pruebas para el volcado y reporte de calibración clínica (Plan 73 — §B.1).

Casos 1 al 9:
1. El volcado sale con las tres columnas (text, assigned, expected) y una fila
   por frase en el orden del modelo exportado.
2. Una frase con comas y comillas sobrevive al viaje (módulo csv, no texto crudo).
3. `expected` sale vacío en el volcado.
4. La referencia se lee y una fila sin `expected` se cuenta como «sin etiquetar».
5. Acierto, contaminación y pérdida se cuentan por separado (§3.1).
6. Una frase que debía ir a la reserva y fue a un apartado cuenta como contaminación.
7. El informe trae la puntuación de cada fallo.
8. El catálogo se recibe, no se carga dentro (dos catálogos dan informes distintos).
9. Una referencia con un `section_id` desconocido falla con ValueError nombrándolo.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from aida.arms.health.calibration import dump_phrases, section_report
from aida.arms.health.sections import SectionCatalog, SectionSpec


def _build_dummy_catalog(
    threshold: float = 0.20, skip_interrogative: bool = True
) -> SectionCatalog:
    sec_motivo = SectionSpec(
        section_id="motivo_consulta",
        label="Motivo de consulta",
        trigger_phrases=("vengo porque me duele la cabeza", "motivo de consulta"),
        codable=True,
        in_note=True,
    )
    sec_sintomas = SectionSpec(
        section_id="sintomas",
        label="Síntomas",
        trigger_phrases=("tengo fiebre y dolor de garganta", "mareos intensos"),
        codable=True,
        in_note=True,
    )
    sec_plan = SectionSpec(
        section_id="plan",
        label="Plan",
        trigger_phrases=("vamos a recetar paracetamol", "el plan de tratamiento"),
        codable=False,
        in_note=True,
    )
    sec_fallback = SectionSpec(
        section_id="otros_comentarios",
        label="Otros comentarios",
        trigger_phrases=(),
        codable=False,
        in_note=False,
    )
    return SectionCatalog(
        sections=(sec_motivo, sec_sintomas, sec_plan, sec_fallback),
        threshold=threshold,
        fallback=sec_fallback,
        skip_interrogative=skip_interrogative,
    )


def test_dump_phrases_columnas_y_orden(tmp_path: Path) -> None:
    """Caso 1 (§B.1): El volcado sale con las tres columnas en orden
    ('text', 'assigned', 'expected') y una fila por frase en el orden del modelo."""
    model_json = tmp_path / "modelo.json"
    model_data = {
        "motivo_consulta": [
            {"text": "Frase motivo 1"},
            {"text": "Frase motivo 2"},
        ],
        "sintomas": [
            {"text": "Frase sintomas 1"},
        ],
    }
    model_json.write_text(json.dumps(model_data), encoding="utf-8")
    out_csv = tmp_path / "salida.csv"

    volcadas = dump_phrases(model_json, out_csv)
    assert volcadas == 3

    with out_csv.open(encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))

    assert reader[0] == ["text", "assigned", "expected"]
    assert reader[1] == ["Frase motivo 1", "motivo_consulta", ""]
    assert reader[2] == ["Frase motivo 2", "motivo_consulta", ""]
    assert reader[3] == ["Frase sintomas 1", "sintomas", ""]
    assert len(reader) == 4


def test_dump_phrases_comas_y_comillas_sobreviven(tmp_path: Path) -> None:
    """Caso 2 (§B.1, M6): Una frase con comas y comillas sobrevive al viaje:
    se vuelca con csv.writer y se vuelve a leer idéntica."""
    frase_compleja = 'Dijo: "tengo dolor, mareos y fatiga", doctor; ¿entiende?'
    model_json = tmp_path / "complejo.json"
    model_json.write_text(
        json.dumps({"motivo_consulta": [{"text": frase_compleja}]}), encoding="utf-8"
    )
    out_csv = tmp_path / "complejo.csv"

    dump_phrases(model_json, out_csv)

    with out_csv.open(encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 1
    assert reader[0]["text"] == frase_compleja
    assert reader[0]["assigned"] == "motivo_consulta"


def test_dump_phrases_expected_sale_vacio(tmp_path: Path) -> None:
    """Caso 3 (§B.1, M8): 'expected' sale vacío en el volcado: el worker no
    rellena etiquetas inventadas ni preasignadas."""
    model_json = tmp_path / "modelo.json"
    model_json.write_text(
        json.dumps({"plan": [{"text": "Reposo absoluto"}], "sintomas": [{"text": "Dolor"}]}),
        encoding="utf-8",
    )
    out_csv = tmp_path / "salida.csv"

    dump_phrases(model_json, out_csv)

    with out_csv.open(encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))

    assert all(row["expected"] == "" for row in reader)


def test_section_report_fila_sin_expected_cuenta_sin_etiquetar(tmp_path: Path) -> None:
    """Caso 4 (§B.1, M2): Una fila sin 'expected' se cuenta como 'sin etiquetar',
    no como acierto ni como fallo."""
    catalog = _build_dummy_catalog()
    ref_csv = tmp_path / "ref.csv"
    with ref_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "assigned", "expected"])
        w.writerow(["vengo porque me duele la cabeza", "motivo_consulta", ""])
        w.writerow(["mareos intensos", "sintomas", "   "])

    report = section_report(ref_csv, catalog)
    assert "2 frases, 0 aciertos, 0 contaminadas, 0 perdidas, 2 sin etiquetar" in report


def test_section_report_acierto_contaminacion_perdida_separados(tmp_path: Path) -> None:
    """Caso 5 (§B.1, M1): Acierto, contaminación y pérdida se cuentan por separado.
    Tres frases preparadas:
    - Una que va a su apartado (acierto)
    - Una que va a otro apartado (contaminación)
    - Una que cae en la reserva debiendo ir a un apartado (pérdida)
    """
    catalog = _build_dummy_catalog(threshold=0.30)
    ref_csv = tmp_path / "ref.csv"
    with ref_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "assigned", "expected"])
        # 1. Acierto: texto coincide con gatillos de motivo_consulta
        w.writerow(["vengo porque me duele la cabeza", "motivo_consulta", "motivo_consulta"])
        # 2. Contaminación: texto coincide con gatillos de motivo_consulta pero se esperaba sintomas
        w.writerow(["vengo porque me duele la cabeza", "motivo_consulta", "sintomas"])
        # 3. Pérdida: texto no supera el umbral, cae a otros_comentarios
        # (reserva), pero esperaba plan
        w.writerow(["palabra aislada sin similitud suficiente", "otros_comentarios", "plan"])

    report = section_report(ref_csv, catalog)
    assert "3 frases, 1 aciertos, 1 contaminadas, 1 perdidas, 0 sin etiquetar" in report

    # Verificar que el desglose por apartado registra las cifras
    assert "motivo_consulta: 1 / 0 / 0" in report
    assert "sintomas: 0 / 1 / 0" in report
    assert "plan: 0 / 0 / 1" in report


def test_section_report_debia_ir_a_reserva_y_fue_a_apartado_cuenta_como_contaminacion(
    tmp_path: Path,
) -> None:
    """Caso 6 (§B.1, M3): Una frase que debía ir a la reserva (otros_comentarios)
    y fue a un apartado clínico cuenta como contaminación, no como pérdida."""
    catalog = _build_dummy_catalog(threshold=0.20)
    ref_csv = tmp_path / "ref.csv"
    with ref_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "assigned", "expected"])
        # Coincide con motivo_consulta, pero expected es la reserva
        w.writerow(["vengo porque me duele la cabeza", "motivo_consulta", "otros_comentarios"])

    report = section_report(ref_csv, catalog)
    assert "1 frases, 0 aciertos, 1 contaminadas, 0 perdidas, 0 sin etiquetar" in report
    assert "otros_comentarios: 0 / 1 / 0" in report


def test_section_report_trae_puntuacion_de_cada_fallo(tmp_path: Path) -> None:
    """Caso 7 (§B.1, M4): El informe trae la puntuación numérica de cada fallo
    en formato '[x.xxxx]'."""
    catalog = _build_dummy_catalog(threshold=0.20)
    ref_csv = tmp_path / "ref.csv"
    with ref_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "assigned", "expected"])
        w.writerow(["vengo porque me duele la cabeza", "motivo_consulta", "sintomas"])

    report = section_report(ref_csv, catalog)
    assert "Fallos (puntuación, asignada, esperada, frase):" in report
    assert "asignada=motivo_consulta esperada=sintomas:" in report
    # Verifica que aparece una puntuación entre corchetes con decimales
    linea_fallo = next(
        linea for linea in report.splitlines() if "asignada=motivo_consulta" in linea
    )
    assert "[" in linea_fallo and "]" in linea_fallo
    score_str = linea_fallo.split("[")[1].split("]")[0]
    score = float(score_str)
    assert 0.0 <= score <= 1.0


def test_section_report_catalogo_se_recibe_no_se_carga(tmp_path: Path) -> None:
    """Caso 8 (§B.1, M5): El catálogo se recibe como argumento en vez de cargarse
    dentro: dos catálogos distintos dan informes distintos sobre la misma referencia."""
    ref_csv = tmp_path / "ref.csv"
    with ref_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "assigned", "expected"])
        w.writerow(["vengo porque me duele la cabeza", "motivo_consulta", "motivo_consulta"])

    catalog_bajo = _build_dummy_catalog(threshold=0.10)
    # Catálogo con umbral imposible: clasifica como reserva
    catalog_alto = _build_dummy_catalog(threshold=0.99)

    report_bajo = section_report(ref_csv, catalog_bajo)
    report_alto = section_report(ref_csv, catalog_alto)

    assert "1 aciertos, 0 contaminadas, 0 perdidas" in report_bajo
    assert "0 aciertos, 0 contaminadas, 1 perdidas" in report_alto
    assert report_bajo != report_alto


def test_section_report_section_id_desconocido_falla_nombrandolo(tmp_path: Path) -> None:
    """Caso 9 (§B.1, M10): Una referencia con un section_id que el catálogo no
    declara falla con ValueError nombrando el identificador desconocido."""
    catalog = _build_dummy_catalog()
    ref_csv = tmp_path / "ref.csv"
    with ref_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "assigned", "expected"])
        w.writerow(["algo de texto", "motivo_consulta", "seccion_inexistente_xyz"])

    with pytest.raises(ValueError) as excinfo:
        section_report(ref_csv, catalog)

    assert "seccion_inexistente_xyz" in str(excinfo.value)
    assert "no es un section_id declarado en el catálogo" in str(excinfo.value)
