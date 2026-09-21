from __future__ import annotations

from aida.arms.health.model import ClinicalFinding, Encounter, build_encounter
from aida.arms.health.sections import SectionSpec
from aida.engine.ontology import AgentId


def test_build_encounter_dos_secciones_varias_lineas() -> None:
    """5.1: `build_encounter` con dos secciones y varias líneas cada una ->
    `Encounter` con `findings` indexado por `section_id`, cada entrada con `text`
    exacto y `section_id` correcto.
    """
    spec_motivo = SectionSpec(
        section_id="motivo_consulta",
        label="Motivo de Consulta",
        trigger_phrases=("consulta por",),
    )
    spec_plan = SectionSpec(
        section_id="plan",
        label="Plan y Tratamiento",
        trigger_phrases=("se indica",),
    )
    grouped_sections = (
        (spec_motivo, ("consulta por cefalea", "refiere mareos")),
        (spec_plan, ("reposo por 3 días", "paracetamol cada 8 horas")),
    )
    owner = AgentId("dra_garcia")

    encounter = build_encounter(owner=owner, grouped_sections=grouped_sections)

    assert isinstance(encounter, Encounter)
    assert encounter.owner == owner
    assert set(encounter.findings.keys()) == {"motivo_consulta", "plan"}

    # Se indexa por spec.section_id, no por spec.label (M4)
    assert "Motivo de Consulta" not in encounter.findings
    assert "Plan y Tratamiento" not in encounter.findings

    findings_motivo = encounter.findings["motivo_consulta"]
    assert len(findings_motivo) == 2
    assert findings_motivo[0] == ClinicalFinding(
        text="consulta por cefalea",
        section_id="motivo_consulta",
    )
    assert findings_motivo[1] == ClinicalFinding(
        text="refiere mareos",
        section_id="motivo_consulta",
    )

    findings_plan = encounter.findings["plan"]
    assert len(findings_plan) == 2
    assert findings_plan[0] == ClinicalFinding(
        text="reposo por 3 días",
        section_id="plan",
    )
    assert findings_plan[1] == ClinicalFinding(
        text="paracetamol cada 8 horas",
        section_id="plan",
    )


def test_build_encounter_vacio() -> None:
    """5.1: `grouped_sections` vacío -> `Encounter` con `findings == {}`."""
    owner = AgentId("dra_garcia")
    encounter = build_encounter(owner=owner, grouped_sections=())

    assert encounter.owner == owner
    assert encounter.findings == {}
    assert encounter.to_dict() == {}


def test_to_dict_conserva_orden_no_alfabetico() -> None:
    """5.1: `to_dict()` conserva el orden de inserción de `findings` (usando orden
    deliberadamente no alfabético, p.ej. 'motivo_consulta' antes que 'enfermedad_actual',
    asegurando que `list(enc.to_dict().keys())` mantiene ese orden). (M2)
    """
    spec_motivo = SectionSpec(
        section_id="motivo_consulta",
        label="Motivo de consulta",
        trigger_phrases=(),
    )
    spec_enfermedad = SectionSpec(
        section_id="enfermedad_actual",
        label="Enfermedad actual",
        trigger_phrases=(),
    )
    # Alfabéticamente 'enfermedad_actual' va antes que 'motivo_consulta',
    # pero grouped_sections inserta 'motivo_consulta' primero.
    grouped_sections = (
        (spec_motivo, ("dolor agudo",)),
        (spec_enfermedad, ("hace 3 días",)),
    )
    encounter = build_encounter(owner=AgentId("doc"), grouped_sections=grouped_sections)

    keys = list(encounter.to_dict().keys())
    assert keys == ["motivo_consulta", "enfermedad_actual"]
    assert keys != sorted(keys)


def test_to_dict_solo_devuelve_text_sin_section_id() -> None:
    """7.4: `to_dict()` devuelve una lista de diccionarios con las cuatro claves:
    'text', 'duration', 'laterality', 'reported_by', los tres últimos en None cuando
    no hubo extracción, sin incluir el `section_id` dentro de las listas.
    """
    owner = AgentId("doc")
    spec = SectionSpec(section_id="motivo_consulta", label="Motivo", trigger_phrases=())
    encounter = build_encounter(
        owner=owner,
        grouped_sections=((spec, ("consulta por dolor", "inició ayer")),),
    )

    result_dict = encounter.to_dict()
    assert result_dict == {
        "motivo_consulta": [
            {
                "text": "consulta por dolor",
                "duration": None,
                "laterality": None,
                "reported_by": None,
            },
            {
                "text": "inició ayer",
                "duration": None,
                "laterality": None,
                "reported_by": None,
            },
        ]
    }
    for items in result_dict.values():
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, dict)
            assert set(item.keys()) == {"text", "duration", "laterality", "reported_by"}
            assert item["duration"] is None
            assert item["laterality"] is None
            assert item["reported_by"] is None


def test_to_dict_con_campos_extraidos() -> None:
    """7.4: `to_dict()` preserva duration, laterality y reported_by cuando están presentes."""
    finding = ClinicalFinding(
        text="dolor en ojo derecho hace 2 semanas",
        section_id="motivo_consulta",
        duration="2 semanas",
        laterality="derecho",
        reported_by="Paciente",
    )
    encounter = Encounter(
        owner=AgentId("doc"),
        findings={"motivo_consulta": (finding,)},
    )
    assert encounter.to_dict() == {
        "motivo_consulta": [
            {
                "text": "dolor en ojo derecho hace 2 semanas",
                "duration": "2 semanas",
                "laterality": "derecho",
                "reported_by": "Paciente",
            }
        ]
    }
