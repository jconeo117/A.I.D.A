"""Pruebas unitarias del módulo de similitud léxica (`aida.textmatch`).

Cubre la unidad U9 según la especificación de `.claude/workflow/26-plan-u9.md` §6.1.
Verifica de forma aislada la aritmética y normalización de bolsa de palabras y coseno:
1. Tokenización en minúsculas y eliminación de puntuación.
2. Preservación de diacríticos y caracteres específicos del español (tildes, diéresis, eñe).
3. Recuento de frecuencias léxicas y tratamiento de cadenas sin tokens.
4. Cálculo euclídeo de norma con ordenamiento determinista de claves.
5. Similitud coseno: identidad exacta (1.0), solapamiento parcial, vectores disyuntos (0.0),
   y vectores vacíos (0.0).
6. Guarda de cortocircuito temprano ante solapamiento nulo para evitar cómputos innecesarios
   y proteger la invariante de mutación M1.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping

import pytest

from aida.textmatch import _norm, cosine, tokenize, vectorize

# ---------------------------------------------------------------------------
# Tokenización léxica
# ---------------------------------------------------------------------------


def test_tokenize_converts_to_lowercase() -> None:
    """Verifica que la tokenización normaliza todo el texto a minúsculas
    para asegurar invariancia de mayúsculas en comparaciones posteriores.
    """
    raw_text = "HOLA Mundo CONSULTA Médica"
    tokens = tokenize(raw_text)
    assert tokens == ("hola", "mundo", "consulta", "médica")


def test_tokenize_preserves_spanish_accents_and_special_chars() -> None:
    """Protege la integridad de diacríticos españoles (á, é, í, ó, ú, ü, ñ)
    impidiendo que el analizador léxico los mutile o los descarte como símbolos.
    """
    sample = "Árboles en el camión del pingüino y la niña de Ñuñoa"
    tokens = tokenize(sample)
    expected = (
        "árboles",
        "en",
        "el",
        "camión",
        "del",
        "pingüino",
        "y",
        "la",
        "niña",
        "de",
        "ñuñoa",
    )
    assert tokens == expected


def test_tokenize_strips_punctuation_and_symbols() -> None:
    """Comprueba que signos de puntuación, delimitadores y símbolos matemáticos
    son descartados y no contaminan los tokens generados.
    """
    sample = "¡Atención! Consulta: ¿fiebre (38.5°C), tos, o cefalea? [urgente] -> código #123."
    tokens = tokenize(sample)
    assert "¡" not in tokens
    assert "!" not in tokens
    assert "¿" not in tokens
    assert "?" not in tokens
    assert "atención" in tokens
    assert "consulta" in tokens
    assert "fiebre" in tokens
    assert "tos" in tokens
    assert "cefalea" in tokens
    assert "urgente" in tokens
    assert "código" in tokens
    assert "123" in tokens


def test_tokenize_empty_or_whitespace_returns_empty_tuple() -> None:
    """Asegura que entradas vacías, con solo espacios en blanco o únicamente
    puntuación devuelven una tupla vacía sin lanzar excepciones.
    """
    assert tokenize("") == ()
    assert tokenize("   \t\n  ") == ()
    assert tokenize("... --- !!! ??? ,,, ;;; :::") == ()


# ---------------------------------------------------------------------------
# Vectorización (recuento de frecuencias)
# ---------------------------------------------------------------------------


def test_vectorize_counts_word_occurrences() -> None:
    """Verifica el recuento exacto de repeticiones de palabras para formar
    el vector de frecuencias de bolsa de palabras.
    """
    text = "dolor de cabeza y dolor de garganta"
    vec = vectorize(text)
    assert vec == {
        "dolor": 2,
        "de": 2,
        "cabeza": 1,
        "y": 1,
        "garganta": 1,
    }


def test_vectorize_empty_text_returns_empty_dict() -> None:
    """Verifica que un texto sin tokens genera un diccionario de frecuencias vacío."""
    assert vectorize("") == {}
    assert vectorize("   !!!   ") == {}


# ---------------------------------------------------------------------------
# Norma vectorial
# ---------------------------------------------------------------------------


def test_norm_calculates_euclidean_length() -> None:
    """Verifica el cálculo de la norma euclídea euclidiana sobre las frecuencias,
    probando la terna pitagórica clásica 3-4-5 y el caso de vector vacío.
    """
    assert _norm({}) == 0.0
    assert math.isclose(_norm({"a": 3, "b": 4}), 5.0)
    assert math.isclose(_norm({"x": 1}), 1.0)


def test_norm_iterates_keys_in_sorted_order() -> None:
    """Asegura que el cálculo de la norma itera sobre claves ordenadas (`sorted`),
    garantizando reproducibilidad y determinismo en cualquier arquitectura.
    """
    seen_keys: list[str] = []

    class TrackedMapping(Mapping[str, int]):
        def __init__(self, data: dict[str, int]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> int:
            seen_keys.append(key)
            return self._data[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    mapping = TrackedMapping({"zebra": 1, "apple": 2, "mango": 3})
    val = _norm(mapping)
    assert math.isclose(val, math.sqrt(1 + 4 + 9))
    assert seen_keys == sorted(seen_keys)


# ---------------------------------------------------------------------------
# Similitud Coseno
# ---------------------------------------------------------------------------


def test_cosine_exact_match_returns_one() -> None:
    """Dos vectores idénticos producen similitud coseno exactamente 1.0."""
    v = vectorize("tumor maligno de lengua")
    assert math.isclose(cosine(v, v), 1.0)


def test_cosine_partial_overlap_returns_fractional_score() -> None:
    """Dos vectores con solapamiento parcial generan un valor coseno entre 0 y 1,
    idéntico al valor teórico calculado manualmente.
    """
    v1 = vectorize("cefalea intensa")  # {"cefalea": 1, "intensa": 1}, norm = sqrt(2)
    v2 = vectorize("cefalea moderada")  # {"cefalea": 1, "moderada": 1}, norm = sqrt(2)
    # shared: {"cefalea"}, dot = 1 * 1 = 1, norm_a * norm_b = 2.0 -> score = 0.5
    score = cosine(v1, v2)
    assert math.isclose(score, 0.5)


def test_cosine_zero_overlap_returns_zero() -> None:
    """Dos vectores sin ningún token en común producen una similitud coseno de 0.0."""
    v1 = vectorize("fiebre alta")
    v2 = vectorize("fractura de fémur")
    assert cosine(v1, v2) == 0.0


def test_cosine_empty_vector_returns_zero() -> None:
    """Cualquier vector comparado contra un vector vacío devuelve similitud 0.0."""
    v_full = vectorize("gripe estacional")
    v_empty = vectorize("")
    assert cosine(v_full, v_empty) == 0.0
    assert cosine(v_empty, v_full) == 0.0
    assert cosine(v_empty, v_empty) == 0.0


class _ExplosiveDisjointMapping(dict[str, int]):
    """Diccionario que permite iteración de claves pero explota con ZeroDivisionError
    si se intenta acceder a un valor para calcular normas sin haber tokens compartidos.
    """

    def __getitem__(self, key: str) -> int:
        raise ZeroDivisionError("Cálculo innecesario sobre vector disyunto")


def test_cosine_zero_overlap_short_circuits_explosive_mapping() -> None:
    """Verifica que cuando no hay tokens compartidos, la función aborta de inmediato
    sin invocar cálculo de normas o acceso a valores.

    Protege la mutación M1 (§6.1): si se retira `if not shared: return 0.0`,
    el cálculo continúa innecesariamente evaluando normas y genera una excepción.
    """
    vec_a = _ExplosiveDisjointMapping({"token_alfa": 1})
    vec_b = {"token_beta": 1}
    assert cosine(vec_a, vec_b) == 0.0


def test_cosine_zero_overlap_short_circuits_without_computing_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica mediante espía que `_norm` no es invocado si el conjunto de
    tokens compartidos está vacío.

    Protege la mutación M1 (§6.1): retirar `if not shared: return 0.0`
    provoca la invocación de `_norm` sobre vectores disyuntos.
    """

    def _should_not_be_called(vector: Mapping[str, int]) -> float:
        pytest.fail("No debe calcularse la norma cuando no hay tokens compartidos")

    monkeypatch.setattr("aida.textmatch._norm", _should_not_be_called)
    res = cosine({"palabra_a": 1}, {"palabra_b": 1})
    assert res == 0.0
