"""Pruebas unitarias de exportación de consultas (`aida.arms.health.export`).

Cubre el contrato de `60-plan-exportar.md` §B.1:
1. Destino directorio: el archivo se nombra AAAA-MM-DD-HHMM-<huella>.md y el ID
   del paciente no aparece en el nombre.
2. Dos sujetos distintos dan huellas distintas en el mismo minuto; el mismo sujeto
   da la misma huella determinista.
3. Destino archivo (.md): se respeta tal cual y se crea el directorio padre si no existe.
4. Con include_model=True y Encounter, se escribe el .json al lado con el mismo nombre;
   con include_model=False, model es None y no hay .json.
5. Con include_model=True pero encounter=None, model es None y no revienta.
6. La transcripción no se escribe en ninguno de los archivos exportados.
7. El resumen se escribe tal cual con salto de línea final.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from aida.arms.health.consultation import ConsultationResult
from aida.arms.health.export import (
    default_export_dir,
    export_consultation,
    export_name,
)
from aida.arms.health.model import ClinicalFinding, Encounter
from aida.engine.ontology import AgentId


def _crear_resultado_prueba(
    *,
    owner: str = "paciente_juan_perez_12345",
    transcript: str = "TRANSCRIPCION_CONFIDENCIAL_RECONOCIBLE_999",
    summary: str = "## Resumen Clínico\n- Motivo: Cefalea intensa",
    con_encounter: bool = False,
) -> ConsultationResult:
    encounter = None
    if con_encounter:
        encounter = Encounter(
            owner=AgentId(owner),
            findings={
                "motivo_consulta": (
                    ClinicalFinding(
                        text="cefalea intensa",
                        section_id="motivo_consulta",
                        duration="hace 3 dias",
                    ),
                )
            },
        )
    return ConsultationResult(
        owner=AgentId(owner),
        transcript=transcript,
        history_excerpt="Sin antecedentes relevantes.",
        matched_codes=("R51",),
        summary_markdown=summary,
        encounter=encounter,
    )


def test_default_export_dir_bajo_documentos_aida_consultas() -> None:
    """Verifica que la carpeta por defecto es visible y fuera de directorios ocultos."""
    assert default_export_dir() == Path.home() / "Documentos" / "aida" / "consultas"


def test_export_consultation_destino_directorio_formato_y_sin_id_paciente(
    tmp_path: Path,
) -> None:
    """Caso 1 (§B.1): Destino directorio: el archivo se llama AAAA-MM-DD-HHMM-<huella>.md,
    y el identificador del paciente no aparece en el nombre."""
    owner_str = "paciente_juan_perez_12345"
    res = _crear_resultado_prueba(owner=owner_str)
    momento = datetime(2026, 9, 23, 14, 30)

    archivos = export_consultation(
        res,
        destination=tmp_path,
        moment=momento,
        include_model=False,
    )

    assert archivos.summary.parent == tmp_path
    assert archivos.summary.suffix == ".md"
    assert archivos.summary.name.startswith("2026-09-23-1430-")
    # El identificador del paciente no debe aparecer en el nombre del archivo
    assert owner_str not in archivos.summary.name
    assert "juan" not in archivos.summary.name
    assert archivos.summary.exists()


def test_export_name_determinismo_y_diferenciacion_entre_sujetos() -> None:
    """Caso 2 (§B.1): Dos sujetos distintos en el mismo minuto dan huellas distintas;
    el mismo sujeto da siempre la misma huella determinista
    (incluso con distinto PYTHONHASHSEED)."""
    import os
    import subprocess
    import sys

    momento = datetime(2026, 9, 23, 14, 30)
    sujeto_a = AgentId("paciente_alfa")
    sujeto_b = AgentId("paciente_beta")

    nombre_a1 = export_name(sujeto_a, momento)
    nombre_a2 = export_name(sujeto_a, momento)
    nombre_b = export_name(sujeto_b, momento)

    # Mismo sujeto, misma fecha/hora -> idéntico nombre
    assert nombre_a1 == nombre_a2
    # Distintos sujetos -> distinta huella
    assert nombre_a1 != nombre_b
    assert nombre_a1.split("-")[:4] == nombre_b.split("-")[:4]  # Mismo minuto

    # Determinismo entre procesos con PYTHONHASHSEED distinto (protege M8)
    cmd = (
        "from datetime import datetime; "
        "from aida.engine.ontology import AgentId; "
        "from aida.arms.health.export import export_name; "
        "print(export_name(AgentId('paciente_alfa'), datetime(2026, 9, 23, 14, 30)))"
    )
    env0 = {**os.environ, "PYTHONHASHSEED": "0"}
    env1 = {**os.environ, "PYTHONHASHSEED": "1"}
    res0 = subprocess.check_output([sys.executable, "-c", cmd], env=env0, text=True).strip()
    res1 = subprocess.check_output([sys.executable, "-c", cmd], env=env1, text=True).strip()
    assert res0 == res1 == nombre_a1


def test_export_consultation_destino_archivo_crea_directorio_padre(
    tmp_path: Path,
) -> None:
    """Caso 3 (§B.1): Destino archivo (.md): se respeta tal cual, y el directorio
    padre se crea si no existía."""
    res = _crear_resultado_prueba()
    momento = datetime(2026, 9, 23, 14, 30)
    destino_archivo = tmp_path / "nueva_carpeta" / "subcarpeta" / "consulta_personalizada.md"
    assert not destino_archivo.parent.exists()

    archivos = export_consultation(
        res,
        destination=destino_archivo,
        moment=momento,
        include_model=False,
    )

    assert archivos.summary == destino_archivo
    assert destino_archivo.exists()
    assert destino_archivo.is_file()


def test_export_consultation_include_model_true_y_false(tmp_path: Path) -> None:
    """Caso 4 (§B.1): Con include_model=True y Encounter, escribe el .json al lado con el
    mismo nombre; con include_model=False, model es None y no hay .json."""
    res = _crear_resultado_prueba(con_encounter=True)
    momento = datetime(2026, 9, 23, 14, 30)

    # Con include_model=True
    archivos_con = export_consultation(
        res,
        destination=tmp_path / "con_modelo",
        moment=momento,
        include_model=True,
    )
    assert archivos_con.model is not None
    assert archivos_con.model == archivos_con.summary.with_suffix(".json")
    assert archivos_con.model.exists()

    # Contenido válido del Clinical Model en JSON
    datos_json = json.loads(archivos_con.model.read_text(encoding="utf-8"))
    assert "motivo_consulta" in datos_json
    assert datos_json["motivo_consulta"][0]["text"] == "cefalea intensa"

    # Con include_model=False
    archivos_sin = export_consultation(
        res,
        destination=tmp_path / "sin_modelo",
        moment=momento,
        include_model=False,
    )
    assert archivos_sin.model is None
    assert not (archivos_sin.summary.with_suffix(".json")).exists()


def test_export_consultation_include_model_true_pero_sin_encounter_no_revienta(
    tmp_path: Path,
) -> None:
    """Caso 5 (§B.1): Con include_model=True pero encounter is None, model es None
    y no revienta."""
    res = _crear_resultado_prueba(con_encounter=False)
    momento = datetime(2026, 9, 23, 14, 30)

    archivos = export_consultation(
        res,
        destination=tmp_path,
        moment=momento,
        include_model=True,
    )
    assert archivos.model is None
    assert archivos.summary.exists()
    assert not archivos.summary.with_suffix(".json").exists()


def test_export_consultation_transcripcion_no_se_escribe(tmp_path: Path) -> None:
    """Caso 6 (§B.1): La transcripción no se escribe: con un transcript reconocible,
    esa cadena no aparece en ninguno de los archivos exportados."""
    cadena_secreta = "CADENA_ULTRA_CONFIDENCIAL_NO_EXPORTAR_XYZ789"
    res = _crear_resultado_prueba(
        transcript=cadena_secreta,
        summary="## Resumen\nTexto público y seguro.",
        con_encounter=True,
    )
    momento = datetime(2026, 9, 23, 14, 30)

    archivos = export_consultation(
        res,
        destination=tmp_path,
        moment=momento,
        include_model=True,
    )

    contenido_md = archivos.summary.read_text(encoding="utf-8")
    assert cadena_secreta not in contenido_md

    assert archivos.model is not None
    contenido_json = archivos.model.read_text(encoding="utf-8")
    assert cadena_secreta not in contenido_json


def test_export_consultation_resumen_fiel_con_salto_de_linea_final(
    tmp_path: Path,
) -> None:
    """Caso 7 (§B.1): El resumen se escribe tal cual, con salto de línea final."""
    texto_resumen = "## Consulta\n- Diagnóstico: Migraña sin aura\n- Plan: Reposo"
    res = _crear_resultado_prueba(summary=texto_resumen)
    momento = datetime(2026, 9, 23, 14, 30)

    archivos = export_consultation(
        res,
        destination=tmp_path / "consulta.md",
        moment=momento,
        include_model=False,
    )

    contenido_leido = archivos.summary.read_text(encoding="utf-8")
    assert contenido_leido == texto_resumen + "\n"
