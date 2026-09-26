"""Pruebas para arms/health/extract.py (§7.3 de 46-plan-extraccion-determinista.md).

Cubre:
- `extract_reported_by`: prefijo exacto de cada etiqueta reconocido; una línea del paciente
  que menciona la palabra usada como etiqueta del profesional en medio del texto (no como prefijo)
  -> devuelve el rol correcto por prefijo (paciente), no por contenido.
- Con etiquetas pasadas y texto sin prefijo conocido -> `reported_by=None`, no una excepción.
- `ClinicalExtractor.extract` sobre una línea con duración y lateralidad a la vez -> los dos
  campos llenos, `reported_by=None` si no se pasan etiquetas.
- `ClinicalExtractor.extract` con etiquetas y prefijo -> los tres campos llenos.
- `build_health_extractor`: construye extractor con catálogo cargado.
"""

from __future__ import annotations

from pathlib import Path

from aida.arms.health.extract import (
    ClinicalExtractor,
    build_health_extractor,
    extract_reported_by,
)
from aida.arms.health.laterality import LateralityCatalog
from aida.arms.health.model import ClinicalFinding


def test_extract_reported_by_prefijo_exacto_reconocido() -> None:
    prof = "Médico"
    pat = "Paciente"
    assert (
        extract_reported_by(
            "Médico: ¿Cómo se siente hoy?",
            professional_label=prof,
            patient_label=pat,
        )
        == prof
    )
    assert (
        extract_reported_by(
            "Paciente: Me duele mucho la cabeza",
            professional_label=prof,
            patient_label=pat,
        )
        == pat
    )


def test_build_health_extractor_instancia() -> None:
    extractor = build_health_extractor()
    assert isinstance(extractor, ClinicalExtractor)


def test_extract_reported_by_mencion_en_medio_del_texto_no_confunde_rol() -> None:
    """Una línea del paciente que menciona la palabra usada como etiqueta del profesional
    en medio del texto (no como prefijo) -> devuelve el rol correcto por prefijo (paciente),
    no por contenido.
    """
    prof = "Médico"
    pat = "Paciente"
    # El paciente menciona la etiqueta del médico seguida de dos puntos dentro de su intervención:
    linea = "Paciente: el informe decia Médico: reposo tres dias"
    assert extract_reported_by(linea, professional_label=prof, patient_label=pat) == pat


def test_extract_reported_by_sin_prefijo_conocido_devuelve_none() -> None:
    prof = "Médico"
    pat = "Paciente"
    linea = "El paciente refiere que fue al médico"
    assert extract_reported_by(linea, professional_label=prof, patient_label=pat) is None


def test_clinical_extractor_duracion_y_lateralidad_sin_etiquetas() -> None:
    """Línea con duración y lateralidad a la vez -> los dos campos llenos,
    `reported_by=None` si no se pasan etiquetas.
    """
    extractor = build_health_extractor()
    finding = ClinicalFinding(
        text="dolor agudo en ojo derecho desde hace dos semanas",
        section_id="motivo_consulta",
    )
    enriched = extractor.extract(
        finding,
        professional_label=None,
        patient_label=None,
    )
    assert enriched.duration == "2 semanas"
    assert enriched.laterality == "derecho"
    assert enriched.reported_by is None
    assert enriched.text == finding.text
    assert enriched.section_id == finding.section_id


def test_clinical_extractor_con_etiquetas_y_prefijo_completa_los_tres_campos() -> None:
    extractor = build_health_extractor()
    finding = ClinicalFinding(
        text="Paciente: presenta inflamación en ambos ojos hace 10 días",
        section_id="enfermedad_actual",
    )
    enriched = extractor.extract(
        finding,
        professional_label="Médico",
        patient_label="Paciente",
    )
    assert enriched.duration == "10 días"
    assert enriched.laterality == "bilateral"
    assert enriched.reported_by == "Paciente"


def test_clinical_extractor_con_etiquetas_pero_sin_prefijo_deja_reported_by_none() -> None:
    """Con etiquetas pasadas y texto sin prefijo conocido -> `reported_by=None`,
    no una excepción.
    """
    extractor = build_health_extractor()
    finding = ClinicalFinding(
        text="inició con dolor hace 3 días",
        section_id="motivo_consulta",
    )
    enriched = extractor.extract(
        finding,
        professional_label="Médico",
        patient_label="Paciente",
    )
    assert enriched.duration == "3 días"
    assert enriched.laterality is None
    assert enriched.reported_by is None


def test_clinical_extractor_sin_coincidencias_conserva_nones() -> None:
    extractor = build_health_extractor()
    finding = ClinicalFinding(
        text="consulta de rutina",
        section_id="motivo_consulta",
    )
    enriched = extractor.extract(
        finding,
        professional_label="Médico",
        patient_label="Paciente",
    )
    assert enriched.duration is None
    assert enriched.laterality is None
    assert enriched.reported_by is None


# ---------------------------------------------------------------------------
# Plan 68: Extracción de negated (§B.3)
# ---------------------------------------------------------------------------


def test_clinical_extractor_rellena_negated() -> None:
    """Caso 15 (§B.3): El extractor rellena negated en hallazgo con negación."""
    extractor = build_health_extractor()
    finding = ClinicalFinding(text="No refiere fiebre", section_id="motivo")
    enriched = extractor.extract(finding, professional_label=None, patient_label=None)
    assert enriched.negated == "No refiere fiebre"


def test_clinical_extractor_sin_negacion_negated_is_none() -> None:
    """Caso 16 (§B.3): Sin negación, negated es None (no cadena vacía)."""
    extractor = build_health_extractor()
    finding = ClinicalFinding(text="Refiere dolor de cabeza", section_id="motivo")
    enriched = extractor.extract(finding, professional_label=None, patient_label=None)
    assert enriched.negated is None


def test_clinical_extractor_catalogo_por_defecto_vacio() -> None:
    """Caso 17 (§B.3): Catálogo por defecto vacío deja negated=None incluso si hay negación."""
    extractor = ClinicalExtractor(laterality=LateralityCatalog(categories=()))
    finding = ClinicalFinding(text="No refiere fiebre", section_id="motivo")
    enriched = extractor.extract(finding, professional_label=None, patient_label=None)
    assert enriched.negated is None


def test_build_health_extractor_carga_ambas_cosas(tmp_path: Path) -> None:
    """Caso 18 (§B.3): build_health_extractor carga lateralidad y negación del mismo archivo."""
    yaml_path = tmp_path / "custom_extraction.yaml"
    yaml_path.write_text(
        "laterality:\n"
        "  derecho: ['derecho']\n"
        "negation:\n"
        "  cues:\n"
        "    - 'niega'\n"
        "  stoppers:\n"
        "    - 'pero'\n",
        encoding="utf-8",
    )
    extractor = build_health_extractor(yaml_path)
    assert extractor.laterality.find("ojo derecho") == "derecho"
    assert extractor.negation.cues == (("niega",),)


def test_clinical_extractor_duration_y_laterality_sobre_texto_entero() -> None:
    """Caso 19 (§B.3): duration y laterality siguen saliendo del texto entero
    aunque haya negación."""
    extractor = build_health_extractor()
    finding = ClinicalFinding(
        text="No le duele el ojo derecho desde hace dos semanas", section_id="motivo"
    )
    enriched = extractor.extract(finding, professional_label=None, patient_label=None)
    assert enriched.laterality == "derecho"
    assert enriched.duration == "2 semanas"
    assert enriched.negated is not None

