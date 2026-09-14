"""Clasificación de intención por bolsa de palabras, sin modelo de lenguaje.

No hay ninguna librería de embeddings en `pyproject.toml`
(`sentence-transformers` y compañía no están instaladas, y el README exige
runtime sin PyTorch). Esto es lo único que este repositorio puede ejecutar
hoy sin añadir una dependencia pesada que nadie ha evaluado: cada frase se
representa como un recuento de palabras, y "se parece a" se mide con coseno
entre esos recuentos. No es tan fino como un modelo entrenado, pero es
determinista, cero tokens, y compara exactamente contra el catálogo cerrado
que declara el pack — nunca inventa una categoría que no esté ahí.

Determinismo: toda suma sobre un `dict` recorre sus claves ordenadas
(`sorted`), nunca el orden de inserción crudo ni el de un `set` — la misma
disciplina que el resto del motor, aplicada aquí aunque en la práctica el
orden de inserción de estos diccionarios ya sea estable (viene de una lista,
no de un `set`): más vale no depender de ese detalle.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from aida.engine.ontology import IntentId
from aida.engine.pack import ContentPack

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ]+")


def _tokenize(text: str) -> tuple[str, ...]:
    """Palabras en minúscula, sin puntuación. Nada de raíces ni sinónimos:
    ver §1 del plan sobre por qué eso queda fuera de esta unidad."""
    return tuple(match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text))


def _vectorize(text: str) -> dict[str, int]:
    """Recuento de palabras. Vacío si el texto no tiene ningún token."""
    vector: dict[str, int] = {}
    for token in _tokenize(text):
        vector[token] = vector.get(token, 0) + 1
    return vector


def _norm(vector: Mapping[str, int]) -> float:
    return math.sqrt(sum(vector[key] ** 2 for key in sorted(vector)))


def _cosine(a: Mapping[str, int], b: Mapping[str, int]) -> float:
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


@dataclass(frozen=True, slots=True)
class BagOfWordsClassifier:
    """Satisface `aida.engine.ports.IntentClassifier` por estructura, sin
    heredar de él — es un `Protocol`.

    `references` es una lista plana de (id de intención, vector de UNA de
    sus frases de ejemplo), no una por intención promediada: comparar contra
    cada ejemplo por separado y quedarse con el mejor es más simple de
    razonar y de anclar en un test que promediar varias frases en un solo
    vector borroso.
    """

    references: tuple[tuple[IntentId, dict[str, int]], ...]
    threshold: float

    def classify(self, utterance: str) -> str | None:
        """`None` si nada del catálogo alcanza `threshold`. El desempate
        entre dos frases con la misma puntuación lo decide el orden de
        `references` (`build_classifier` lo construye ordenado por id de
        intención): con `>` estricto, la primera vista —la de id menor—
        conserva el mejor resultado, nunca un `float` compara para decidir
        el empate.
        """
        query = _vectorize(utterance)
        if not query:
            return None
        best_score = 0.0
        best_id: IntentId | None = None
        for intent_id, example_vector in self.references:
            score = _cosine(query, example_vector)
            if score > best_score:
                best_score = score
                best_id = intent_id
        if best_id is None or best_score < self.threshold:
            return None
        return str(best_id)


def build_classifier(pack: ContentPack) -> BagOfWordsClassifier | None:
    """`None` si el pack no declara ninguna intención: no hay contra qué
    clasificar, y `Loop` trata `None` como "sin clasificador" — ver 5.6."""
    if not pack.intents:
        return None
    references = tuple(
        (intent.id, _vectorize(example))
        for intent in sorted(pack.intents, key=lambda i: str(i.id))
        for example in intent.examples
    )
    return BagOfWordsClassifier(
        references=references, threshold=pack.tuning.intent_confidence_threshold
    )
