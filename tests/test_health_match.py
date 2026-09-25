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

import pytest

from aida.arms.health.codes import DiagnosticCode, load_catalog
from aida.arms.health.match import (
    CodeOrigin,
    MatchedCode,
    build_index,
    load_matching_settings,
    match_by_origin,
    match_codes,
    match_codes_each,
)

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


# ---------------------------------------------------------------------------
# Plan 59: Índice con stopwords, match_codes_each y ajustes (§B.1, §B.2, §B.3)
# ---------------------------------------------------------------------------


def test_build_index_sin_stopwords_deja_vacio_y_vectores_completos() -> None:
    """Caso 3 (§B.1): build_index sin palabras vacías deja index.stopwords vacío
    y los vectores completos con todas las palabras."""
    catalog = (
        DiagnosticCode(
            code="A01", description="el dolor de cabeza", level=3, source="test"
        ),
    )
    index = build_index(catalog)
    assert index.stopwords == frozenset()
    assert index.entries[0][1] == {"el": 1, "dolor": 1, "de": 1, "cabeza": 1}


def test_build_index_con_stopwords_y_match_codes_filtra_consulta() -> None:
    """Caso 4 (§B.1): build_index guarda stopwords en el índice, match_codes
    las aplica a la consulta: un texto de solo stopwords da (); y una consulta
    con stopwords las descarta para alcanzar score perfecto."""
    catalog = (
        DiagnosticCode(code="A01", description="dolor agudo", level=3, source="test"),
    )
    index = build_index(catalog, stopwords=frozenset({"de"}))
    assert index.stopwords == frozenset({"de"})
    assert "de" not in index.entries[0][1]

    # Texto con solo stopwords no genera query
    assert match_codes("de", index, threshold=0.1, top_n=5) == ()

    # Consulta con stopwords: al descartar 'de', coincide exactamente con 'dolor agudo'
    matches = match_codes("dolor de agudo", index, threshold=0.9, top_n=1)
    assert len(matches) == 1
    assert math.isclose(matches[0].score, 1.0)
    assert matches[0].diagnostic.code == "A01"


def test_match_codes_each_dos_textos_codigos_distintos() -> None:
    """Caso 5 (§B.2): dos textos con códigos distintos: ambos aparecen."""
    catalog = (
        DiagnosticCode(code="A01", description="dolor lumbar", level=3, source="test"),
        DiagnosticCode(
            code="B02", description="infección respiratoria", level=3, source="test"
        ),
    )
    index = build_index(catalog)
    textos = ("dolor lumbar agudo", "infección respiratoria alta")
    matches = match_codes_each(textos, index, threshold=0.3, top_n=5)
    codes = {m.diagnostic.code for m in matches}
    assert "A01" in codes
    assert "B02" in codes


def test_match_codes_each_codigo_repetido_conserva_mejor_puntuacion() -> None:
    """Caso 6 (§B.2): un código que sale en dos textos aparece una sola vez
    con su mejor puntuación."""
    catalog = (
        DiagnosticCode(code="A01", description="dolor lumbar agudo", level=3, source="test"),
    )
    index = build_index(catalog)
    textos = ("dolor lumbar agudo", "dolor lumbar")
    matches = match_codes_each(textos, index, threshold=0.5, top_n=5)
    assert len(matches) == 1
    assert matches[0].diagnostic.code == "A01"
    assert math.isclose(matches[0].score, 1.0)


def test_match_codes_each_garantiza_mejor_codigo_de_cada_texto() -> None:
    """Caso 7 (§B.2): el mejor de cada texto está garantizado ante dos textos,
    uno con 4 códigos con puntuación mayor y otro con 1 de puntuación menor.
    Con top_n=3, el del segundo texto no desaparece."""
    catalog = (
        DiagnosticCode(
            code="D01", description="diabetes avanzada tipo uno", level=3, source="test"
        ),
        DiagnosticCode(
            code="D02", description="diabetes avanzada tipo dos", level=3, source="test"
        ),
        DiagnosticCode(
            code="D03", description="diabetes avanzada tipo tres", level=3, source="test"
        ),
        DiagnosticCode(
            code="D04", description="diabetes avanzada tipo cuatro", level=3, source="test"
        ),
        DiagnosticCode(
            code="G01", description="gonartrosis leve moderada severa", level=3, source="test"
        ),
    )
    index = build_index(catalog)
    # Texto 1 produce score ~0.7071 para D01..D04.
    # Texto 2 produce score 0.50 para G01.
    matches = match_codes_each(("diabetes avanzada", "gonartrosis"), index, threshold=0.4, top_n=3)
    assert len(matches) == 3
    codes = [m.diagnostic.code for m in matches]
    assert "G01" in codes, (
        f"G01 debe estar garantizado por ser el mejor de texto 2; obtenidos: {codes}"
    )


def test_match_codes_each_orden_descendente_score_y_empate_codigo_ascendente() -> None:
    """Caso 8 (§B.2): el orden final es puntuación descendente y, en empate, código ascendente."""
    catalog = (
        DiagnosticCode(code="Z01", description="cefalea aguda", level=3, source="test"),
        DiagnosticCode(code="A01", description="cefalea aguda", level=3, source="test"),
        DiagnosticCode(code="M01", description="cefalea intensa", level=3, source="test"),
    )
    index = build_index(catalog)
    matches = match_codes_each(("cefalea aguda",), index, threshold=0.1, top_n=5)
    codes = [m.diagnostic.code for m in matches]
    # Z01 y A01 empatan en 1.0 -> A01 antes que Z01. M01 tiene menor score -> al final.
    assert codes == ["A01", "Z01", "M01"]


def test_match_codes_each_sin_textos_o_sin_aciertos_devuelve_tupla_vacia() -> None:
    """Caso 9 (§B.2): sin textos, o ninguno por encima del umbral, devuelve ()."""
    catalog = (DiagnosticCode(code="A01", description="cefalea", level=3, source="test"),)
    index = build_index(catalog)
    assert match_codes_each((), index, threshold=0.5, top_n=5) == ()
    assert match_codes_each(("fractura de pie",), index, threshold=0.5, top_n=5) == ()


def test_load_matching_settings_pack_real_y_validaciones(tmp_path: Path) -> None:
    """Caso 10 (§B.3): load_matching_settings sobre el pack real da threshold=0.5,
    top_n=6 y palabras en minúsculas; si falta una clave o el tipo no cuadra,
    ValueError que la nombra."""
    real = load_matching_settings()
    assert real.threshold == 0.5
    assert real.top_n == 6
    assert len(real.stopwords) > 0
    assert all(s == s.lower() for s in real.stopwords)
    assert "de" in real.stopwords
    assert "en" in real.stopwords

    # YAML no mapeo
    no_map = tmp_path / "no_map.yaml"
    no_map.write_text("- item1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"debe ser un mapeo YAML"):
        load_matching_settings(no_map)

    # Clave faltante
    sin_thresh = tmp_path / "sin_thresh.yaml"
    sin_thresh.write_text("top_n: 5\nstopwords: ['de']\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"faltan claves.*threshold"):
        load_matching_settings(sin_thresh)

    # threshold no numérico o bool
    thresh_str = tmp_path / "thresh_str.yaml"
    thresh_str.write_text(
        "threshold: medio\ntop_n: 5\nstopwords: ['de']\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'threshold' debe ser numérico"):
        load_matching_settings(thresh_str)

    thresh_bool = tmp_path / "thresh_bool.yaml"
    thresh_bool.write_text(
        "threshold: true\ntop_n: 5\nstopwords: ['de']\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'threshold' debe ser numérico"):
        load_matching_settings(thresh_bool)

    # top_n no int, bool o <= 0
    top_cero = tmp_path / "top_cero.yaml"
    top_cero.write_text(
        "threshold: 0.5\ntop_n: 0\nstopwords: ['de']\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'top_n' debe ser un entero positivo"):
        load_matching_settings(top_cero)

    top_bool = tmp_path / "top_bool.yaml"
    top_bool.write_text(
        "threshold: 0.5\ntop_n: true\nstopwords: ['de']\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'top_n' debe ser un entero positivo"):
        load_matching_settings(top_bool)

    top_str = tmp_path / "top_str.yaml"
    top_str.write_text(
        "threshold: 0.5\ntop_n: '5'\nstopwords: ['de']\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'top_n' debe ser un entero positivo"):
        load_matching_settings(top_str)

    # stopwords no lista o elementos no str
    stop_str = tmp_path / "stop_str.yaml"
    stop_str.write_text(
        "threshold: 0.5\ntop_n: 5\nstopwords: de\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'stopwords' debe ser una lista de cadenas"):
        load_matching_settings(stop_str)

    stop_num = tmp_path / "stop_num.yaml"
    stop_num.write_text(
        "threshold: 0.5\ntop_n: 5\nstopwords: [123]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'stopwords' debe ser una lista de cadenas"):
        load_matching_settings(stop_num)


# ---------------------------------------------------------------------------
# Plan 62: match_by_origin (§B.1)
# ---------------------------------------------------------------------------


def test_match_by_origin_today_history_y_both_con_mejor_puntuacion() -> None:
    """Caso 1 (§B.1): Un código que solo sale hoy es TODAY; uno que solo sale en
    la historia es HISTORY; uno en los dos es BOTH con la mejor de las dos puntuaciones."""
    catalog = (
        DiagnosticCode(code="A01", description="cefalea aguda", level=3, source="test"),
        DiagnosticCode(code="B02", description="diabetes mellitus", level=3, source="test"),
        DiagnosticCode(code="C03", description="hipertensión arterial", level=3, source="test"),
    )
    index = build_index(catalog)

    # A01 solo en today (score 1.0)
    # B02 en today ("diabetes mellitus descompensada" -> score ~0.8165)
    # B02 en history ("diabetes mellitus" -> score 1.0)
    # C03 solo en history (score 1.0)
    today = ("cefalea aguda", "diabetes mellitus descompensada")
    history = ("diabetes mellitus", "hipertensión arterial")

    matches = match_by_origin(
        today=today,
        history=history,
        index=index,
        threshold=0.5,
        top_n=5,
    )
    por_codigo = {m.diagnostic.code: m for m in matches}

    assert por_codigo["A01"].origin == CodeOrigin.TODAY
    assert math.isclose(por_codigo["A01"].score, 1.0)

    assert por_codigo["C03"].origin == CodeOrigin.HISTORY
    assert math.isclose(por_codigo["C03"].score, 1.0)

    assert por_codigo["B02"].origin == CodeOrigin.BOTH
    # B02 debe quedarse con la mejor de las dos puntuaciones (1.0 vs 0.8165)
    assert math.isclose(por_codigo["B02"].score, 1.0)


def test_match_by_origin_tope_es_por_origen() -> None:
    """Caso 2 (§B.1): El tope es por origen: con top_n=2, dos de hoy y dos de la
    historia pueden convivir; ninguno tapa al otro."""
    catalog = (
        DiagnosticCode(code="T01", description="dolor brazo", level=3, source="test"),
        DiagnosticCode(code="T02", description="dolor pierna", level=3, source="test"),
        DiagnosticCode(code="H01", description="asma cronica", level=3, source="test"),
        DiagnosticCode(code="H02", description="rinitis cronica", level=3, source="test"),
    )
    index = build_index(catalog)

    today = ("dolor brazo", "dolor pierna")
    history = ("asma cronica", "rinitis cronica")

    matches = match_by_origin(
        today=today,
        history=history,
        index=index,
        threshold=0.5,
        top_n=2,
    )
    assert len(matches) == 4
    codigos = {m.diagnostic.code for m in matches}
    assert codigos == {"T01", "T02", "H01", "H02"}
    assert {m.diagnostic.code for m in matches if m.origin == CodeOrigin.TODAY} == {"T01", "T02"}
    assert {m.diagnostic.code for m in matches if m.origin == CodeOrigin.HISTORY} == {"H01", "H02"}


def test_match_by_origin_orden_final_puntuacion_descendente_codigo_ascendente() -> None:
    """Caso 3 (§B.1): Orden final: puntuación descendente, código ascendente en empate."""
    catalog = (
        DiagnosticCode(code="Z01", description="tos severa", level=3, source="test"),
        DiagnosticCode(code="A01", description="tos severa", level=3, source="test"),
        DiagnosticCode(code="M01", description="fiebre leve moderada", level=3, source="test"),
    )
    index = build_index(catalog)
    # Z01 y A01 tienen score 1.0; M01 tiene score 0.5774
    today = ("tos severa", "fiebre")
    history = ()

    matches = match_by_origin(
        today=today,
        history=history,
        index=index,
        threshold=0.4,
        top_n=5,
    )
    assert len(matches) == 3
    # A01 empata con Z01 a 1.0, pero 'A01' < 'Z01' en desempate léxico
    assert [m.diagnostic.code for m in matches] == ["A01", "Z01", "M01"]


def test_match_by_origin_history_vacio_todos_today_y_mismo_resultado() -> None:
    """Caso 4 (§B.1): Con history=(), todos son TODAY y el resultado coincide
    con el de match_codes_each."""
    catalog = (
        DiagnosticCode(code="A01", description="cefalea aguda", level=3, source="test"),
        DiagnosticCode(code="B02", description="lumbalgia", level=3, source="test"),
    )
    index = build_index(catalog)
    today = ("cefalea aguda", "lumbalgia")

    matches_origin = match_by_origin(
        today=today,
        history=(),
        index=index,
        threshold=0.5,
        top_n=5,
    )
    matches_each = match_codes_each(
        today,
        index,
        threshold=0.5,
        top_n=5,
    )

    assert len(matches_origin) == len(matches_each)
    assert all(m.origin == CodeOrigin.TODAY for m in matches_origin)
    for mo, me in zip(matches_origin, matches_each, strict=True):
        assert mo.diagnostic.code == me.diagnostic.code
        assert math.isclose(mo.score, me.score)


