"""La revisión del profesional: cada hallazgo y cada código propuesto pasa
por un sí, un cambio o un no explícitos antes de contar. Es la frontera entre
proponer y registrar — sin ella, lo que la consulta sugiere se guardaría como
si alguien lo hubiera aprobado.

Quién contesta se inyecta (`Reviewer`): la batería pone un doble y la CLI el
teclado. Qué veredictos admite cada ítem lo fija este módulo, no quien
contesta: la CLI los lee de `ReviewItem.allowed` para preguntar, y
`review_encounter` rechaza cualquier otro. Una sola puerta.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from aida.arms.health.match import MatchedCode
from aida.arms.health.model import ClinicalFinding, Encounter
from aida.arms.health.sections import SectionSpec
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId


class Verdict(StrEnum):
    CONFIRMED = "confirmed"
    EDITED = "edited"
    REJECTED = "rejected"


class ItemKind(StrEnum):
    FINDING = "finding"
    CODE = "code"


FINDING_VERDICTS = (Verdict.CONFIRMED, Verdict.EDITED, Verdict.REJECTED)
CODE_VERDICTS = (Verdict.CONFIRMED, Verdict.REJECTED)
"""Un código no se edita: un texto libre tecleado como código se saltaría el
catálogo, que es justo lo que el cotejo existe para impedir."""


@dataclass(frozen=True, slots=True)
class ReviewItem:
    kind: ItemKind
    ref: str
    """`section_id:posición` para un hallazgo; el código, para un código. Es lo
    que queda en el evento — el texto no."""
    label: str
    text: str
    allowed: tuple[Verdict, ...]


@dataclass(frozen=True, slots=True)
class Answer:
    verdict: Verdict
    text: str | None = None


Reviewer = Callable[[ReviewItem], Answer]


@dataclass(frozen=True, slots=True)
class ReviewedConsultation:
    encounter: Encounter
    grouped_sections: tuple[tuple[SectionSpec, tuple[str, ...]], ...]
    confirmed_codes: tuple[MatchedCode, ...]


def _check(item: ReviewItem, answer: Answer) -> None:
    if answer.verdict not in item.allowed:
        raise ValueError(f"veredicto {answer.verdict.value!r} no admitido para {item.ref!r}")
    if answer.verdict is Verdict.EDITED:
        if answer.text is None or not answer.text.strip():
            raise ValueError(f"edición sin texto para {item.ref!r}")
    elif answer.text is not None:
        raise ValueError(f"texto sin edición para {item.ref!r}")


def review_encounter(
    *,
    owner: AgentId,
    grouped_sections: tuple[tuple[SectionSpec, tuple[str, ...]], ...],
    encounter: Encounter,
    matches: tuple[MatchedCode, ...],
    reviewer: Reviewer,
    log: EventLog,
    refine: Callable[[ClinicalFinding, str], ClinicalFinding],
) -> ReviewedConsultation:
    """Recorre `grouped_sections` —una tupla, en el orden del catálogo— y no
    `encounter.findings`, que es un mapa: el orden de las preguntas no puede
    depender de cómo se construyó un `dict`. Después, los códigos en el orden
    en que los devolvió el cotejo.

    No captura nada. Si `reviewer` levanta a mitad, no hay resultado parcial:
    lo contestado queda en el registro porque ocurrió, pero nada cuenta como
    revisado."""
    kept_sections: list[tuple[SectionSpec, tuple[str, ...]]] = []
    kept_findings: dict[str, tuple[ClinicalFinding, ...]] = {}
    for spec, _texts in grouped_sections:
        kept: list[ClinicalFinding] = []
        for position, finding in enumerate(encounter.findings[spec.section_id]):
            item = ReviewItem(
                kind=ItemKind.FINDING,
                ref=f"{spec.section_id}:{position}",
                label=spec.label,
                text=finding.text,
                allowed=FINDING_VERDICTS,
            )
            answer = reviewer(item)
            _check(item, answer)
            if answer.verdict is Verdict.EDITED and answer.text is not None:
                log.emit(tick=0, kind=EventKind.INPUT_OBSERVED, owner=owner, content=answer.text)
                kept.append(refine(finding, answer.text))
            elif answer.verdict is Verdict.CONFIRMED:
                kept.append(finding)
            log.emit(
                tick=0,
                kind=EventKind.ITEM_REVIEWED,
                owner=owner,
                data={"item": item.kind.value, "ref": item.ref, "verdict": answer.verdict.value},
            )
        if kept:
            kept_findings[spec.section_id] = tuple(kept)
            kept_sections.append((spec, tuple(f.text for f in kept)))

    confirmed: list[MatchedCode] = []
    for match in matches:
        item = ReviewItem(
            kind=ItemKind.CODE,
            ref=match.diagnostic.code,
            label=match.diagnostic.code,
            text=match.diagnostic.description,
            allowed=CODE_VERDICTS,
        )
        answer = reviewer(item)
        _check(item, answer)
        if answer.verdict is Verdict.CONFIRMED:
            confirmed.append(match)
        log.emit(
            tick=0,
            kind=EventKind.ITEM_REVIEWED,
            owner=owner,
            data={"item": item.kind.value, "ref": item.ref, "verdict": answer.verdict.value},
        )

    return ReviewedConsultation(
        encounter=Encounter(owner=encounter.owner, findings=kept_findings),
        grouped_sections=tuple(kept_sections),
        confirmed_codes=tuple(confirmed),
    )
