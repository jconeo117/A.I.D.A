"""Extracción determinista de campos ya presentes en la línea: duración
(genérica, `timeparse.py`), lateralidad (vocabulario cerrado de este
brazo) y quién lo dijo (leído de la etiqueta que ya puso la diarización,
nunca adivinado). Nada de esto es comprensión de lenguaje libre — aislar
la frase de síntoma/motivo queda fuera, ver `46-plan-extraccion-determinista.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from aida.arms.health.laterality import LateralityCatalog, load_laterality
from aida.arms.health.model import ClinicalFinding
from aida.arms.health.negation import NO_NEGATION, NegationCatalog, load_negation, split_polarity
from aida.timeparse import find_duration


def extract_reported_by(text: str, *, professional_label: str, patient_label: str) -> str | None:
    """Confirma qué prefijo lleva la línea — no clasifica nada nuevo, solo
    lee de vuelta lo que `_diarize_and_label` ya escribió. Compara el
    prefijo exacto (`startswith`), no una búsqueda dentro del texto: que
    una línea del paciente MENCIONE la palabra "médico" no la convierte en
    una línea del profesional."""
    if text.startswith(f"{professional_label}: "):
        return professional_label
    if text.startswith(f"{patient_label}: "):
        return patient_label
    return None


@dataclass(frozen=True, slots=True)
class ClinicalExtractor:
    laterality: LateralityCatalog
    negation: NegationCatalog = NO_NEGATION

    def extract(
        self,
        finding: ClinicalFinding,
        *,
        professional_label: str | None,
        patient_label: str | None,
    ) -> ClinicalFinding:
        reported_by = None
        if professional_label is not None and patient_label is not None:
            reported_by = extract_reported_by(
                finding.text, professional_label=professional_label, patient_label=patient_label
            )
        partes = split_polarity(finding.text, self.negation)
        return replace(
            finding,
            duration=find_duration(finding.text),
            laterality=self.laterality.find(finding.text),
            reported_by=reported_by,
            negated=partes.negated or None,
        )


def build_health_extractor(path: Path | None = None) -> ClinicalExtractor:
    return ClinicalExtractor(laterality=load_laterality(path), negation=load_negation(path))
