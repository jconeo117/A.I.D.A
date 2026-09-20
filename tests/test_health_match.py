"""Pruebas unitarias del cotejo determinista contra el catálogo CIE-10 (`aida.arms.health.match`).

Cubre la unidad U9 según la especificación de `.claude/workflow/26-plan-u9.md` §6.3.
Verifica de forma aislada la selección, filtrado y ordenamiento de códigos diagnósticos:
1. `match_codes` sobre índice sintético pequeño devuelve los `top_n` ordenados
   de mayor a menor puntuación.
2. Desempate determinista ante puntuaciones idénticas: orden lexicográfico por `code` ascendente.
3. Texto sin tokens (vacío o solo puntuación) devuelve una tupla vacía sin lanzar excepciones.
4. Umbral inalcanzable (`threshold > max_score`) devuelve una tupla vacía.
5. Cotejo real contra el catálogo CIE-10 cargado desde disco:
   - "tumor maligno de la lengua" con umbral 0.3 incluye códigos de la familia `C02*`.
6. Condición de frontera exacta `score == threshold`: los resultados cuya puntuación iguala
   exactamente el umbral deben incluirse en los resultados. (Protege la mutación M3).
"""

from __future__ import annotations

import math
from pathlib import Path

from aida.arms.health.codes import DiagnosticCode, load_catalog
from aida.arms.health.match import MatchedCode, build_index, match_codes

CIE10_JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "health" / "cie10" / "codes.json"
CIE10_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "health" / "cie10" / "cie-10.csv"


# ---------------------------------------------------------------------------
# Casos sintéticos con índice reducido (§6.3.1, §6.3.2, §6.3.3, §6.3.4)
# ---------------------------------------------------------------------------


def test_match_codes_top_n_sorted_descending_by_score() -> None:
    """Verifica que match_codes selecciona los top_n códigos con mayor afinidad
    ordenados estrictamente de mayor a menor puntuación.
    """
    catalog = (
        DiagnosticCode(
            code="A01", description="fiebre alta con dolor de cabeza", level=3, source="test"
        ),
        DiagnosticCode(code="A02", description="fiebre persistente", level=3, source="test"),
        DiagnosticCode(
            code="A03", description="fractura expuesta de tibia", level=3, source="test"
        ),
        DiagnosticCode(
            code="A04", description="fiebre alta y malestar general", level=3, source="test"
        ),
    )
    index = build_index(catalog)

    results = match_codes("fiebre alta", index, threshold=0.2, top_n=2)
    assert len(results) == 2

    # A02 tiene 2 palabras ("fiebre", "persistente"), comparte 1 de 2 ("fiebre")
    # A01 tiene 6 palabras ("fiebre", "alta", "con", "dolor", "de", "cabeza"), comparte 2
    # A04 tiene 5 palabras ("fiebre", "alta", "y", "malestar", "general"), comparte 2
    # Ambos comparten "fiebre" y "alta" de la query de 2 palabras
    assert all(isinstance(m, MatchedCode) for m in results)
    assert results[0].score >= results[1].score
    assert results[0].diagnostic.code in {"A01", "A04"}


def test_match_codes_exact_score_tie_breaking_by_code_ascending() -> None:
    """Ante dos diagnósticos que producen exactamente la misma similitud coseno,
    el desempate debe resolverse estrictamente por código en orden alfabético ascendente.
    """
    # Dos códigos con idéntica descripción generan idéntico vector y coseno
    code_z = DiagnosticCode(code="Z99", description="cefalea aguda intensa", level=3, source="test")
    code_a = DiagnosticCode(code="A01", description="cefalea aguda intensa", level=3, source="test")
    code_m = DiagnosticCode(code="M50", description="cefalea aguda intensa", level=3, source="test")

    catalog = (code_z, code_a, code_m)
    index = build_index(catalog)

    results = match_codes("cefalea aguda intensa", index, threshold=0.5, top_n=3)

    assert len(results) == 3
    # Todos tienen score 1.0
    assert all(math.isclose(m.score, 1.0) for m in results)

    # Orden alfabético por código: A01, M50, Z99
    ordered_codes = [m.diagnostic.code for m in results]
    assert ordered_codes == ["A01", "M50", "Z99"]


def test_match_codes_text_without_tokens_returns_empty_tuple() -> None:
    """Una consulta sin tokens léxicos válidos (vacía o signos de puntuación)
    debe retornar una tupla vacía sin evaluar el índice.
    """
    catalog = (DiagnosticCode(code="A01", description="gripe", level=3, source="test"),)
    index = build_index(catalog)

    assert match_codes("", index, threshold=0.1, top_n=5) == ()
    assert match_codes("   \t\n  ", index, threshold=0.1, top_n=5) == ()
    assert match_codes("... ??? !!! ---", index, threshold=0.1, top_n=5) == ()


def test_match_codes_threshold_higher_than_any_score_returns_empty_tuple() -> None:
    """Si el umbral exigido supera la afinidad máxima alcanzada por cualquier código,
    se devuelve una tupla vacía.
    """
    catalog = (
        DiagnosticCode(code="A01", description="tos seca", level=3, source="test"),
        DiagnosticCode(code="A02", description="tos con flema", level=3, source="test"),
    )
    index = build_index(catalog)

    # Query parcialmente coincidente pero con umbral exigente de 0.99
    results = match_codes("tos continua persistente nocturna", index, threshold=0.99, top_n=5)
    assert results == ()


# ---------------------------------------------------------------------------
# Frontera exacta y protección de mutación M3 (§6.3.6)
# ---------------------------------------------------------------------------


def test_match_codes_boundary_score_exactly_equal_to_threshold() -> None:
    """Un resultado cuya puntuación redondeada coincide con el umbral debe incluirse
    en los resultados devueltos (invariante `score >= threshold`).

    Protege la mutación M3 (§6.3): si en `match_codes` se sustituye
    `if score >= threshold` por `if score > threshold`, el resultado en la frontera
    exacta es descartado erróneamente y el test falla (RED).
    """
    catalog = (
        DiagnosticCode(code="B10", description="cefalea tensional", level=3, source="test"),
    )
    index = build_index(catalog)

    # Consulta idéntica a la descripción: score = 1.0000
    # Fijando threshold = 1.0, la condición score >= threshold (1.0 >= 1.0) se cumple
    results = match_codes("cefalea tensional", index, threshold=1.0, top_n=1)

    assert len(results) == 1
    assert results[0].diagnostic.code == "B10"
    assert math.isclose(results[0].score, 1.0)


# ---------------------------------------------------------------------------
# Cotejo sobre catálogo real CIE-10 (§6.3.5)
# ---------------------------------------------------------------------------


def test_match_codes_real_catalog_tongue_neoplasm_matches_c02() -> None:
    """Sobre el catálogo real de 14498 registros, la consulta médica
    'tumor maligno de la lengua' con umbral 0.3 debe devolver al menos un código
    de la categoría C02 (Tumor maligno de otras partes y de las no especificadas de la lengua).
    """
    assert CIE10_JSON_PATH.exists()
    assert CIE10_CSV_PATH.exists()

    catalog = load_catalog(CIE10_JSON_PATH, CIE10_CSV_PATH)
    index = build_index(catalog)

    matches = match_codes("tumor maligno de la lengua", index, threshold=0.3, top_n=10)

    assert len(matches) > 0

    matched_codes = [m.diagnostic.code for m in matches]
    c02_codes = [c for c in matched_codes if c.startswith("C02")]
    assert len(c02_codes) > 0, f"Ningún código C02* encontrado entre resultados: {matched_codes}"
