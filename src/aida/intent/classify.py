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

from dataclasses import dataclass

from aida.engine.ontology import IntentId
from aida.engine.pack import ContentPack
from aida.textmatch import cosine, tokenize, vectorize

# Alias para pruebas existentes de U6
_tokenize = tokenize
_vectorize = vectorize
_cosine = cosine


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
        query = vectorize(utterance)
        if not query:
            return None
        best_score = 0.0
        best_id: IntentId | None = None
        for intent_id, example_vector in self.references:
            score = cosine(query, example_vector)
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
        (intent.id, vectorize(example))
        for intent in sorted(pack.intents, key=lambda i: str(i.id))
        for example in intent.examples
    )
    return BagOfWordsClassifier(
        references=references, threshold=pack.tuning.intent_confidence_threshold
    )
