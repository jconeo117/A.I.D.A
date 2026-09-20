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


def segment_lines(
    lines: tuple[str, ...], index: SectionIndex, *, threshold: float, fallback: SectionSpec
) -> tuple[SegmentedLine, ...]:
    result: list[SegmentedLine] = []
    for line in lines:
        if not line.strip():
            continue
        query = vectorize(line)
        scored = (
            [(round(cosine(query, vector), 4), spec) for spec, vector in index] if query else []
        )
        scored = [(score, spec) for score, spec in scored if score >= threshold]
        if not scored:
            result.append(SegmentedLine(section=fallback, text=line))
            continue
        scored.sort(key=lambda pair: (-pair[0], pair[1].section_id))
        result.append(SegmentedLine(section=scored[0][1], text=line))
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
