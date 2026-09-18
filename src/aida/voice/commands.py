"""Comandos del operador, reconocidos por bolsa de palabras contra frases
de ejemplo — mismo mecanismo que `intent/classify.py` (U6), catálogo
distinto. Las frases las declara quien construye el catálogo: este módulo
no sabe qué significa "cambiar de modo" en ningún dominio.

Umbral alto a propósito (0.6, frente a 0.35 de las intenciones): un
comando reconocido por error termina la sesión o cambia de modo — un
falso negativo solo obliga a repetirlo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from aida.textmatch import cosine, vectorize


class OperatorCommand(StrEnum):
    SWITCH_MODE = "switch_mode"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class CommandCatalog:
    references: tuple[tuple[OperatorCommand, dict[str, int]], ...]
    threshold: float

    @classmethod
    def from_phrases(
        cls, phrases: Mapping[OperatorCommand, Sequence[str]], *, threshold: float = 0.6
    ) -> CommandCatalog:
        references = tuple(
            (command, vectorize(phrase))
            for command in sorted(phrases, key=lambda c: c.value)
            for phrase in phrases[command]
        )
        return cls(references=references, threshold=threshold)

    def recognize(self, utterance: str) -> OperatorCommand | None:
        """El comando cuya frase más se parece, si supera el umbral.
        Empate: la primera vista (orden fijo por valor del enum), con `>`
        estricto — nunca decide un float solo."""
        query = vectorize(utterance)
        if not query:
            return None
        best_score = 0.0
        best: OperatorCommand | None = None
        for command, vector in self.references:
            score = cosine(query, vector)
            if score > best_score:
                best_score = score
                best = command
        if best is None or best_score < self.threshold:
            return None
        return best
