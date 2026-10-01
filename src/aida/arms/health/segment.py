"""Segmentación: qué línea de la transcripción pertenece a qué
sección. Mismo mecanismo que `intent/classify.py` (bolsa de palabras y
coseno, U6): una línea, la sección de mayor coseno sobre un umbral — nunca
varias, nunca ninguna: por debajo del umbral cae en la sección de reserva
que declara el catálogo (`sections.py`), no en una constante de este
módulo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aida.arms.health.sections import SectionCatalog, SectionSpec, load_sections
from aida.textmatch import cosine, vectorize

SectionIndex = tuple[tuple[SectionSpec, dict[str, int]], ...]


def build_section_index(sections: tuple[SectionSpec, ...]) -> SectionIndex:
    return tuple((spec, vectorize(" ".join(spec.trigger_phrases))) for spec in sections)


@dataclass(frozen=True, slots=True)
class SegmentedLine:
    section: SectionSpec
    text: str


@dataclass(frozen=True, slots=True)
class ScoredSection:
    """La sección ganadora y con qué puntuación — la puntuación no la usa
    `segment_lines`, pero sí el informe de calibración (`calibration.py`) para
    decir si un fallo se arregla con el umbral o con los gatillos."""

    section: SectionSpec
    score: float


def classify_line(
    line: str, index: SectionIndex, *, threshold: float, fallback: SectionSpec
) -> ScoredSection:
    """La sección de mayor coseno sobre el umbral, o la reserva si ninguna lo
    alcanza. Puerta única de la clasificación: la usan `segment_lines` y el
    informe de calibración, para que los dos vean siempre el mismo criterio y
    no haya dos clasificadores que puedan divergir.

    Con reserva, la puntuación que se devuelve es la mejor que hubo —aunque no
    alcanzara el umbral—, no un `0.0`: es lo que le dice a quien calibra si la
    frase casi entra o si no se parecía a nada."""
    query = vectorize(line)
    if not query:
        return ScoredSection(section=fallback, score=0.0)
    scored = [(round(cosine(query, vector), 4), spec) for spec, vector in index]
    scored.sort(key=lambda pair: (-pair[0], pair[1].section_id))
    best_score, best_spec = scored[0]
    if best_score >= threshold:
        return ScoredSection(section=best_spec, score=best_score)
    return ScoredSection(section=fallback, score=best_score)


def segment_lines(
    lines: tuple[str, ...], index: SectionIndex, *, threshold: float, fallback: SectionSpec
) -> tuple[SegmentedLine, ...]:
    result: list[SegmentedLine] = []
    for line in lines:
        if not line.strip():
            continue
        winner = classify_line(line, index, threshold=threshold, fallback=fallback)
        result.append(SegmentedLine(section=winner.section, text=line))
    return tuple(result)


def group_by_section(
    segmented: tuple[SegmentedLine, ...], sections: tuple[SectionSpec, ...]
) -> tuple[tuple[SectionSpec, tuple[str, ...]], ...]:
    """Conserva el orden de declaración del catálogo. Una sección sin
    ninguna línea se omite — no hay nada que mostrar bajo un título vacío."""
    grouped = []
    for spec in sections:
        texts = tuple(item.text for item in segmented if item.section.section_id == spec.section_id)
        if texts:
            grouped.append((spec, texts))
    return tuple(grouped)


class ClinicalSegmenter:
    def __init__(self, catalog: SectionCatalog) -> None:
        self._catalog = catalog
        self._index = build_section_index(catalog.sections)

    def segment(self, transcript: str) -> tuple[tuple[SectionSpec, tuple[str, ...]], ...]:
        lines = tuple(transcript.split("\n"))
        segmented = segment_lines(
            lines, self._index, threshold=self._catalog.threshold, fallback=self._catalog.fallback
        )
        return group_by_section(segmented, self._catalog.sections)


def build_health_segmenter(path: Path | None = None) -> ClinicalSegmenter:
    return ClinicalSegmenter(load_sections(path))
