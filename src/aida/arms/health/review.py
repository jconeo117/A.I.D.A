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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from aida.arms.health.match import CodeOrigin, MatchedCode
from aida.arms.health.model import ClinicalFinding, Encounter
from aida.arms.health.note import RenderedNote
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
    FIELD = "field"
    """Un campo de la nota. No es un `ReviewItem`: no se pregunta, se compara lo
    entregado con lo redactado (`review_note`). Existe para el evento y la métrica."""


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


def finding_item(
    spec: SectionSpec,
    position: int,
    finding: ClinicalFinding,
    *,
    negated_suffix: str | None = None,
) -> ReviewItem:
    """La forma de un ítem de hallazgo, en un solo sitio: la usan el bucle que
    pregunta y la pantalla que lo muestra. Con dos construcciones distintas, el
    profesional vería un texto y el registro guardaría otro."""
    return ReviewItem(
        kind=ItemKind.FINDING,
        ref=f"{spec.section_id}:{position}",
        label=spec.label,
        text=(
            f"{finding.text} · {negated_suffix}"
            if negated_suffix and finding.negated == finding.text.strip()
            else finding.text
        ),
        allowed=FINDING_VERDICTS,
    )


def code_item(
    match: MatchedCode,
    *,
    known_code_suffix: str | None = None,
    evidence_template: str | None = None,
) -> ReviewItem:
    descripcion = match.diagnostic.description
    if known_code_suffix and match.origin is not CodeOrigin.TODAY:
        descripcion = f"{descripcion} · {known_code_suffix}"
    if evidence_template and match.evidence:
        descripcion = evidence_template.format(descripcion, match.evidence)
    return ReviewItem(
        kind=ItemKind.CODE,
        ref=match.diagnostic.code,
        label=match.diagnostic.code,
        text=descripcion,
        allowed=CODE_VERDICTS,
    )


def review_items(
    *,
    grouped_sections: tuple[tuple[SectionSpec, tuple[str, ...]], ...],
    encounter: Encounter,
    matches: tuple[MatchedCode, ...],
    known_code_suffix: str | None = None,
    evidence_template: str | None = None,
    negated_suffix: str | None = None,
    review_findings: bool = True,
) -> tuple[ReviewItem, ...]:
    """Todos los ítems que `review_encounter` va a preguntar, **en el mismo
    orden**: primero los hallazgos recorriendo el catálogo de secciones, después
    los códigos como los devolvió el cotejo.

    Existe para que la pantalla se pueda pintar antes de revisar. Que el orden y
    el contenido coincidan con lo que pregunta el bucle no es cortesía: si
    divergen, la pantalla recoge respuestas para ítems que no existen y el bucle
    pregunta por otros que nadie contestó."""
    items = (
        [
            finding_item(spec, position, finding, negated_suffix=negated_suffix)
            for spec, _texts in grouped_sections
            for position, finding in enumerate(encounter.findings[spec.section_id])
        ]
        if review_findings
        else []
    )
    items.extend(
        code_item(
            match,
            known_code_suffix=known_code_suffix,
            evidence_template=evidence_template,
        )
        for match in matches
    )
    return tuple(items)


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
    known_code_suffix: str | None = None,
    evidence_template: str | None = None,
    negated_suffix: str | None = None,
    review_findings: bool = True,
) -> ReviewedConsultation:
    """Recorre `grouped_sections` —una tupla, en el orden del catálogo— y no
    `encounter.findings`, que es un mapa: el orden de las preguntas no puede
    depender de cómo se construyó un `dict`. Después, los códigos en el orden
    en que los devolvió el cotejo.

    No captura nada. Si `reviewer` levanta a mitad, no hay resultado parcial:
    lo contestado queda en el registro porque ocurrió, pero nada cuenta como
    revisado.

    Con `review_findings=False` los hallazgos se conservan todos sin preguntar ni
    dejar evento: es la pantalla, donde el profesional lee la nota y no frase a
    frase. Un `ITEM_REVIEWED` por hallazgo diría que alguien los miró uno por uno.
    """
    kept_sections: list[tuple[SectionSpec, tuple[str, ...]]] = []
    kept_findings: dict[str, tuple[ClinicalFinding, ...]] = {}
    if review_findings:
        for spec, _texts in grouped_sections:
            kept: list[ClinicalFinding] = []
            for position, finding in enumerate(encounter.findings[spec.section_id]):
                item = finding_item(spec, position, finding, negated_suffix=negated_suffix)
                answer = reviewer(item)
                _check(item, answer)
                if answer.verdict is Verdict.EDITED and answer.text is not None:
                    log.emit(
                        tick=0, kind=EventKind.INPUT_OBSERVED, owner=owner, content=answer.text
                    )
                    kept.append(refine(finding, answer.text))
                elif answer.verdict is Verdict.CONFIRMED:
                    kept.append(finding)
                log.emit(
                    tick=0,
                    kind=EventKind.ITEM_REVIEWED,
                    owner=owner,
                    data={
                        "item": item.kind.value,
                        "ref": item.ref,
                        "verdict": answer.verdict.value,
                    },
                )
            if kept:
                kept_findings[spec.section_id] = tuple(kept)
                kept_sections.append((spec, tuple(f.text for f in kept)))
    else:
        kept_sections = list(grouped_sections)
        kept_findings = {
            spec.section_id: encounter.findings[spec.section_id]
            for spec, _texts in grouped_sections
        }

    confirmed: list[MatchedCode] = []
    for match in matches:
        item = code_item(
            match,
            known_code_suffix=known_code_suffix,
            evidence_template=evidence_template,
        )
        answer = reviewer(item)
        _check(item, answer)
        if answer.verdict is Verdict.CONFIRMED:
            confirmed.append(match)
        log.emit(
            tick=0,
            kind=EventKind.ITEM_REVIEWED,
            owner=owner,
            data={
                "item": item.kind.value,
                "ref": item.ref,
                "verdict": answer.verdict.value,
                "origin": match.origin.value,
            },
        )

    return ReviewedConsultation(
        encounter=Encounter(owner=encounter.owner, findings=kept_findings),
        grouped_sections=tuple(kept_sections),
        confirmed_codes=tuple(confirmed),
    )


def review_note(
    *,
    owner: AgentId,
    note: RenderedNote,
    edits: Mapping[str, str],
    log: EventLog,
) -> RenderedNote:
    """Lo que el profesional entregó de cada campo, contra lo que se redactó.

    El veredicto se decide aquí y no en la pantalla: comparar en el navegador sería
    una segunda puerta, y la que manda es la que deja el evento. Se compara sin los
    espacios de los bordes: un salto de línea añadido al final no es una edición.

    Las ediciones tienen que cubrir exactamente los campos de la nota. Un campo sin
    respuesta no se da por bueno: sería un `confirmed` que nadie entregó."""
    esperados = [spec.section_id for spec, _texto in note.fields]
    if sorted(edits) != sorted(esperados):
        raise ValueError("las ediciones no cubren exactamente los campos de la nota")
    campos: list[tuple[SectionSpec, str]] = []
    editados: list[str] = []
    for spec, texto in note.fields:
        entregado = edits[spec.section_id]
        if entregado.strip() != texto.strip():
            log.emit(tick=0, kind=EventKind.INPUT_OBSERVED, owner=owner, content=entregado)
            campos.append((spec, entregado))
            editados.append(spec.section_id)
            veredicto = Verdict.EDITED
        else:
            campos.append((spec, texto))
            veredicto = Verdict.CONFIRMED
        log.emit(
            tick=0,
            kind=EventKind.ITEM_REVIEWED,
            owner=owner,
            data={"item": ItemKind.FIELD.value, "ref": spec.section_id, "verdict": veredicto.value},
        )
    return replace(note, fields=tuple(campos), edited=tuple(editados))
