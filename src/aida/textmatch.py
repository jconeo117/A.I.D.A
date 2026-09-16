"""Bolsa de palabras y coseno: la única aritmética de "esto se parece a
esto" del repositorio. Compartida por `intent/classify.py` (U6) y
`consult/match.py` (U9) para que las dos no puedan divergir en el mismo
cálculo con criterios distintos — una sola puerta por mecanismo.

Sin modelo, sin dependencia nueva: un recuento de palabras y coseno entre
recuentos. Determinismo: toda suma sobre un `dict` recorre sus claves
ordenadas, nunca el orden de inserción crudo ni el de un `set`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ]+")


def tokenize(text: str) -> tuple[str, ...]:
    """Palabras en minúscula, sin puntuación. Nada de raíces ni sinónimos."""
    return tuple(match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text))


def vectorize(text: str) -> dict[str, int]:
    """Recuento de palabras. Vacío si el texto no tiene ningún token."""
    vector: dict[str, int] = {}
    for token in tokenize(text):
        vector[token] = vector.get(token, 0) + 1
    return vector


def _norm(vector: Mapping[str, int]) -> float:
    return math.sqrt(sum(vector[key] ** 2 for key in sorted(vector)))


def cosine(a: Mapping[str, int], b: Mapping[str, int]) -> float:
    """Similitud coseno entre dos bolsas de palabras. 0.0 si no comparten
    ningún token o si alguna de las dos está vacía."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return 0.0
    dot = sum(a[key] * b[key] for key in shared)
    norm_a = _norm(a)
    norm_b = _norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
