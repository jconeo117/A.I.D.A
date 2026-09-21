"""El Clinical Model: una representación interna de la consulta,
independiente de cualquier software de Historia Clínica externo. Es una
repaquetización sin pérdida de lo que ya produjo la segmentación
(`segment.py`) — no clasifica nada nuevo, no añade certeza que no exista.
Extraer campos finos dentro de cada hallazgo (duración, lateralidad, quién
lo dijo) es trabajo de una unidad futura, no de este módulo.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from aida.arms.health.sections import SectionSpec
from aida.engine.ontology import AgentId


@dataclass(frozen=True, slots=True)
class ClinicalFinding:
    """Una línea, tal como se dijo — nunca solo un fragmento suelto: la
    procedencia completa importa tanto como la categoría (mismo criterio
    que pide `AIDA-health.md` en su Fase 3: nunca solo 'visión borrosa')."""

    text: str
    section_id: str
    duration: str | None = None
    laterality: str | None = None
    reported_by: str | None = None


@dataclass(frozen=True, slots=True)
class Encounter:
    """`findings` está indexado por `section_id` del catálogo de
    secciones, no por nombres fijos de campo — ver §0."""

    owner: AgentId
    findings: Mapping[str, tuple[ClinicalFinding, ...]]

    def to_dict(self) -> dict[str, list[dict[str, str | None]]]:
        """El orden de `findings` ya es el de declaración del catálogo
        (`group_by_section` lo garantiza en el llamante) — no hay
        reordenamiento que hacer aquí."""
        return {
            section_id: [
                {
                    "text": f.text,
                    "duration": f.duration,
                    "laterality": f.laterality,
                    "reported_by": f.reported_by,
                }
                for f in items
            ]
            for section_id, items in self.findings.items()
        }


def build_encounter(
    *,
    owner: AgentId,
    grouped_sections: tuple[tuple[SectionSpec, tuple[str, ...]], ...],
) -> Encounter:
    findings = {
        spec.section_id: tuple(
            ClinicalFinding(text=text, section_id=spec.section_id) for text in texts
        )
        for spec, texts in grouped_sections
    }
    return Encounter(owner=owner, findings=findings)
