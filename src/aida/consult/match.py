"""Cotejo determinista contra el catálogo CIE-10: los N mejores resultados
por encima de un umbral, no uno solo — a diferencia de `IntentClassifier`
(U6), una consulta puede tocar varios códigos relevantes a la vez.

El catálogo se pre-vectoriza una vez (`build_index`); son más de 14000
entradas y volver a tokenizar cada descripción en cada llamada sería
trabajo repetido sin ninguna razón.
"""

from __future__ import annotations

from dataclasses import dataclass

from aida.consult.codes import DiagnosticCode
from aida.textmatch import cosine, vectorize

CatalogIndex = tuple[tuple[DiagnosticCode, dict[str, int]], ...]


def build_index(catalog: tuple[DiagnosticCode, ...]) -> CatalogIndex:
    """Catálogo ya recibido ordenado por código (`load_catalog` lo
    garantiza) — el índice conserva ese orden, es lo que hace determinista
    el desempate en `match_codes`."""
    return tuple((entry, vectorize(entry.description)) for entry in catalog)


@dataclass(frozen=True, slots=True)
class MatchedCode:
    diagnostic: DiagnosticCode
    score: float


def match_codes(
    text: str, index: CatalogIndex, *, threshold: float, top_n: int
) -> tuple[MatchedCode, ...]:
    """Los `top_n` códigos con mayor coseno por encima de `threshold`,
    ordenados de mayor a menor puntuación y, en empate, por código
    ascendente (desempate por identificador, nunca solo por el float de la
    puntuación)."""
    query = vectorize(text)
    if not query:
        return ()
    scored = [
        (round(cosine(query, vector), 4), entry)
        for entry, vector in index
    ]
    scored = [(score, entry) for score, entry in scored if score >= threshold]
    scored.sort(key=lambda pair: (-pair[0], pair[1].code))
    return tuple(
        MatchedCode(diagnostic=entry, score=score)
        for score, entry in scored[:top_n]
    )
