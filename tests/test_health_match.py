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
    CodableText,
    CodeOrigin,
    MatchedCode,
    build_index,
    load_matching_settings,
    match_by_origin,
    match_codes,
    match_codes_each,
)


def _codables(*textos: str) -> tuple[CodableText, ...]:
    return tuple(CodableText(query=t, source=t) for t in textos)


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
    matches = match_codes_each(_codables(*textos), index, threshold=0.3, top_n=5)
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
    matches = match_codes_each(_codables(*textos), index, threshold=0.5, top_n=5)
    assert len(matches) == 1
    assert matches[0].diagnostic.code == "A01"
    assert math.isclose(matches[0].score, 1.0)


def test_match_codes_each_garantiza_mejor_codigo_de_cada_texto() -> None:
    """Caso 7 (§B.2, adaptado Plan 80 §B.4): el mejor de cada texto está
    garantizado ante dos textos. Con 'diabetes avanzada tipo uno dos tres',
    D01..D03 compiten por explicar palabras nuevas (0.8165), D04 no añade
    palabra nueva, y G01 (0.50) es el mejor de texto 2. Con top_n=3, G01 no
    desaparece."""
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
    matches = match_codes_each(
        _codables("diabetes avanzada tipo uno dos tres", "gonartrosis"),
        index,
        threshold=0.4,
        top_n=3,
    )
    assert len(matches) == 3
    codes = [m.diagnostic.code for m in matches]
    assert "G01" in codes, (
        f"G01 debe estar garantizado por ser el mejor de texto 2; obtenidos: {codes}"
    )
    assert set(codes) == {"D01", "D02", "G01"}


def test_match_codes_each_orden_descendente_score_y_empate_codigo_ascendente() -> None:
    """Caso 8 (§B.2, adaptado Plan 80 §B.4): el orden final es puntuación descendente
    y, en empate, código ascendente."""
    catalog = (
        DiagnosticCode(code="Z01", description="cefalea aguda", level=3, source="test"),
        DiagnosticCode(code="A01", description="migraña aguda", level=3, source="test"),
        DiagnosticCode(code="M01", description="cefalea intensa leve", level=3, source="test"),
    )
    index = build_index(catalog)
    matches = match_codes_each(
        _codables("cefalea aguda", "migraña aguda", "cefalea intensa"),
        index,
        threshold=0.1,
        top_n=5,
    )
    codes = [m.diagnostic.code for m in matches]
    # Z01 y A01 empatan en 1.0 -> A01 antes que Z01. M01 tiene menor score -> al final.
    assert codes == ["A01", "Z01", "M01"]


def test_match_codes_each_sin_textos_o_sin_aciertos_devuelve_tupla_vacia() -> None:
    """Caso 9 (§B.2): sin textos, o ninguno por encima del umbral, devuelve ()."""
    catalog = (DiagnosticCode(code="A01", description="cefalea", level=3, source="test"),)
    index = build_index(catalog)
    assert match_codes_each((), index, threshold=0.5, top_n=5) == ()
    assert match_codes_each(_codables("fractura de pie"), index, threshold=0.5, top_n=5) == ()


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

    cat_yaml = (
        "absence_markers: ['sin']\n"
        "catalog:\n  kind: reference\n  files: ['data/health/cie10/co/cie10-co.csv']\n"
    )

    # Clave faltante
    sin_thresh = tmp_path / "sin_thresh.yaml"
    sin_thresh.write_text(f"top_n: 5\nstopwords: ['de']\n{cat_yaml}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"faltan claves.*threshold"):
        load_matching_settings(sin_thresh)

    # threshold no numérico o bool
    thresh_str = tmp_path / "thresh_str.yaml"
    thresh_str.write_text(
        f"threshold: medio\ntop_n: 5\nstopwords: ['de']\n{cat_yaml}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'threshold' debe ser numérico"):
        load_matching_settings(thresh_str)

    thresh_bool = tmp_path / "thresh_bool.yaml"
    thresh_bool.write_text(
        f"threshold: true\ntop_n: 5\nstopwords: ['de']\n{cat_yaml}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'threshold' debe ser numérico"):
        load_matching_settings(thresh_bool)

    # top_n no int, bool o <= 0
    top_cero = tmp_path / "top_cero.yaml"
    top_cero.write_text(
        f"threshold: 0.5\ntop_n: 0\nstopwords: ['de']\n{cat_yaml}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'top_n' debe ser un entero positivo"):
        load_matching_settings(top_cero)

    top_bool = tmp_path / "top_bool.yaml"
    top_bool.write_text(
        f"threshold: 0.5\ntop_n: true\nstopwords: ['de']\n{cat_yaml}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'top_n' debe ser un entero positivo"):
        load_matching_settings(top_bool)

    top_str = tmp_path / "top_str.yaml"
    top_str.write_text(
        f"threshold: 0.5\ntop_n: '5'\nstopwords: ['de']\n{cat_yaml}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'top_n' debe ser un entero positivo"):
        load_matching_settings(top_str)

    # stopwords no lista o elementos no str
    stop_str = tmp_path / "stop_str.yaml"
    stop_str.write_text(
        f"threshold: 0.5\ntop_n: 5\nstopwords: de\n{cat_yaml}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"'stopwords' debe ser una lista de cadenas"):
        load_matching_settings(stop_str)

    stop_num = tmp_path / "stop_num.yaml"
    stop_num.write_text(
        f"threshold: 0.5\ntop_n: 5\nstopwords: [123]\n{cat_yaml}", encoding="utf-8"
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
        today=_codables(*today),
        history=_codables(*history),
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
        today=_codables(*today),
        history=_codables(*history),
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
    """Caso 3 (§B.1, adaptado Plan 80 §B.4): Orden final: puntuación descendente,
    código ascendente en empate."""
    catalog = (
        DiagnosticCode(code="Z01", description="tos severa", level=3, source="test"),
        DiagnosticCode(code="A01", description="disnea severa", level=3, source="test"),
        DiagnosticCode(code="M01", description="fiebre leve moderada", level=3, source="test"),
    )
    index = build_index(catalog)
    # Z01 y A01 tienen score 1.0; M01 tiene score 0.5774
    today = ("tos severa", "disnea severa", "fiebre")
    history = ()

    matches = match_by_origin(
        today=_codables(*today),
        history=_codables(*history),
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
        today=_codables(*today),
        history=(),
        index=index,
        threshold=0.5,
        top_n=5,
    )
    matches_each = match_codes_each(
        _codables(*today),
        index,
        threshold=0.5,
        top_n=5,
    )

    assert len(matches_origin) == len(matches_each)
    assert all(m.origin == CodeOrigin.TODAY for m in matches_origin)
    for mo, me in zip(matches_origin, matches_each, strict=True):
        assert mo.diagnostic.code == me.diagnostic.code
        assert math.isclose(mo.score, me.score)


# ---------------------------------------------------------------------------
# Pruebas B.4: Evidencia en el cotejo (68-plan-negacion-evidencia.md)
# ---------------------------------------------------------------------------


def test_match_codes_each_evidencia_es_la_frase_fuente() -> None:
    """Caso 20 (§B.4): La evidencia es la frase fuente, no el término sintético."""
    catalog = (
        DiagnosticCode(code="E11", description="diabetes mellitus tipo 2", level=3, source="test"),
    )
    index = build_index(catalog)
    codable = CodableText(query="diabetes mellitus", source="sufro del azúcar")
    matches = match_codes_each((codable,), index, threshold=0.4, top_n=5)
    assert len(matches) == 1
    assert matches[0].diagnostic.code == "E11"
    assert matches[0].evidence == "sufro del azúcar"


def test_match_codes_each_gana_frase_de_mejor_puntuacion() -> None:
    """Caso 21 (§B.4): Gana la frase que produjo la mejor puntuación del código."""
    catalog = (
        DiagnosticCode(code="E11", description="diabetes mellitus", level=3, source="test"),
    )
    index = build_index(catalog)
    c1 = CodableText(query="diabetes", source="frase con diabetes leve")
    c2 = CodableText(query="diabetes mellitus", source="frase exacta diabetes mellitus")
    matches = match_codes_each((c1, c2), index, threshold=0.4, top_n=5)
    assert len(matches) == 1
    assert matches[0].evidence == "frase exacta diabetes mellitus"


def test_match_codes_each_empate_gana_la_primera() -> None:
    """Caso 22 (§B.4): En empate de puntuación, gana la primera frase que apareció."""
    catalog = (
        DiagnosticCode(code="R51", description="cefalea aguda", level=3, source="test"),
    )
    index = build_index(catalog)
    c1 = CodableText(query="cefalea aguda", source="primera mención de dolor")
    c2 = CodableText(query="cefalea aguda", source="segunda mención de dolor")
    matches = match_codes_each((c1, c2), index, threshold=0.4, top_n=5)
    assert len(matches) == 1
    assert matches[0].evidence == "primera mención de dolor"


def test_match_codes_no_inventa_evidencia() -> None:
    """Caso 23 (§B.4): match_codes de bajo nivel sigue devolviendo evidence=""."""
    catalog = (
        DiagnosticCode(code="R51", description="cefalea aguda", level=3, source="test"),
    )
    index = build_index(catalog)
    matches = match_codes("cefalea aguda", index, threshold=0.4, top_n=5)
    assert len(matches) == 1
    assert matches[0].evidence == ""


def test_match_by_origin_propaga_evidencia_en_todos_los_origenes() -> None:
    """Caso 24 (§B.4): match_by_origin propaga la evidencia en TODAY, HISTORY y BOTH."""
    catalog = (
        DiagnosticCode(code="A01", description="cefalea intensa", level=3, source="test"),
        DiagnosticCode(code="B02", description="hipertensión arterial", level=3, source="test"),
        DiagnosticCode(code="C03", description="asma bronquial", level=3, source="test"),
    )
    index = build_index(catalog)
    today = (
        CodableText(query="cefalea intensa", source="dijo cefalea hoy"),
        CodableText(query="hipertensión", source="dijo hipertensión hoy"),
    )
    history = (
        CodableText(query="hipertensión arterial", source="antecedente de hipertensión"),
        CodableText(query="asma bronquial", source="antecedente de asma"),
    )
    matches = match_by_origin(today=today, history=history, index=index, threshold=0.4, top_n=5)
    por_codigo = {m.diagnostic.code: m for m in matches}
    assert por_codigo["A01"].origin == CodeOrigin.TODAY
    assert por_codigo["A01"].evidence == "dijo cefalea hoy"
    assert por_codigo["C03"].origin == CodeOrigin.HISTORY
    assert por_codigo["C03"].evidence == "antecedente de asma"
    assert por_codigo["B02"].origin == CodeOrigin.BOTH
    assert por_codigo["B02"].evidence == "antecedente de hipertensión"


# ---------------------------------------------------------------------------
# Plan 69: Catálogo CIE-10 de Colombia y Plegado (§B.2)
# ---------------------------------------------------------------------------


def test_match_codes_plegado_resuelve_mayusculas_sin_tildes() -> None:
    """Caso 5 (§B.2): Con descripciones en mayúsculas y sin tildes (MIGRANA, COLERA),
    cotejar 'migraña' y 'cólera' devuelve el código con puntuación 1.00.
    Y con mayúsculas con tildes (HIPERTENSIÓN), cotejar sin tildes devuelve 1.00."""
    catalog = (
        DiagnosticCode(code="G439", description="MIGRANA", level=3, source="test"),
        DiagnosticCode(code="A009", description="COLERA", level=3, source="test"),
        DiagnosticCode(code="I10X", description="HIPERTENSIÓN", level=3, source="test"),
    )
    index = build_index(catalog)
    r_mig = match_codes("migraña", index, threshold=0.5, top_n=5)
    assert len(r_mig) >= 1
    assert r_mig[0].diagnostic.code == "G439"
    assert math.isclose(r_mig[0].score, 1.0, rel_tol=1e-3)

    r_col = match_codes("cólera", index, threshold=0.5, top_n=5)
    assert len(r_col) >= 1
    assert r_col[0].diagnostic.code == "A009"
    assert math.isclose(r_col[0].score, 1.0, rel_tol=1e-3)

    r_hip = match_codes("hipertension", index, threshold=0.5, top_n=5)
    assert len(r_hip) >= 1
    assert r_hip[0].diagnostic.code == "I10X"
    assert math.isclose(r_hip[0].score, 1.0, rel_tol=1e-3)


def test_match_codes_plegado_es_simetrico() -> None:
    """Caso 6 (§B.2): Plegar es simétrico: catálogo con tildes y consulta sin ellas
    emparejan igual que al revés."""
    catalog_tildes = (
        DiagnosticCode(code="G439", description="migraña severa", level=3, source="test"),
    )
    idx_tildes = build_index(catalog_tildes)
    r1 = match_codes("migrana severa", idx_tildes, threshold=0.5, top_n=5)
    assert len(r1) == 1
    assert math.isclose(r1[0].score, 1.0, rel_tol=1e-3)

    catalog_sin = (
        DiagnosticCode(code="G439", description="migrana severa", level=3, source="test"),
    )
    idx_sin = build_index(catalog_sin)
    r2 = match_codes("migraña severa", idx_sin, threshold=0.5, top_n=5)
    assert len(r2) == 1
    assert math.isclose(r2[0].score, 1.0, rel_tol=1e-3)


def test_load_matching_settings_stopwords_plegadas(tmp_path: Path) -> None:
    """Caso 7 (§B.2): Las palabras vacías se pliegan al cargarse: si se declara 'más',
    descarta el token 'mas'."""
    yaml_file = tmp_path / "matching_stop.yaml"
    yaml_file.write_text(
        "threshold: 0.5\ntop_n: 5\nstopwords: ['más']\nabsence_markers: []\n"
        "catalog:\n  kind: reference\n  files: ['data/health/cie10/co/cie10-co.csv']\n",
        encoding="utf-8",
    )
    settings = load_matching_settings(yaml_file)
    assert "mas" in settings.stopwords


def test_load_matching_settings_catalog_obligatorio(tmp_path: Path) -> None:
    """Caso 8 (§B.2): catalog es obligatorio en matching.yaml; si falta, levanta ValueError."""
    yaml_file = tmp_path / "sin_catalog.yaml"
    yaml_file.write_text(
        "threshold: 0.5\ntop_n: 5\nstopwords: ['de']\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"catalog"):
        load_matching_settings(yaml_file)


def test_load_matching_settings_kind_y_archivos_concordancia(tmp_path: Path) -> None:
    """Caso 9 (§B.2): kind y número de archivos deben concordar.
    reference con 2 archivos o dual con 1 levantan ValueError indicando cuántos esperaba."""
    # reference con 2
    ref_dos = tmp_path / "ref_dos.yaml"
    ref_dos.write_text(
        "threshold: 0.5\ntop_n: 5\nstopwords: []\nabsence_markers: []\n"
        "catalog:\n  kind: reference\n  files: ['f1.csv', 'f2.csv']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"reference.*exige 1.*trae 2"):
        load_matching_settings(ref_dos)

    # dual con 1
    dual_uno = tmp_path / "dual_uno.yaml"
    dual_uno.write_text(
        "threshold: 0.5\ntop_n: 5\nstopwords: []\nabsence_markers: []\n"
        "catalog:\n  kind: dual\n  files: ['f1.csv']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"dual.*exige 2.*trae 1"):
        load_matching_settings(dual_uno)

    # kind desconocido
    desconocido = tmp_path / "desconocido.yaml"
    desconocido.write_text(
        "threshold: 0.5\ntop_n: 5\nstopwords: []\nabsence_markers: []\n"
        "catalog:\n  kind: desconocido\n  files: ['f1.csv']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"debe ser 'reference' o 'dual'"):
        load_matching_settings(desconocido)


def test_load_matching_settings_catalogo_ausente_falla_con_ruta(tmp_path: Path) -> None:
    """Caso 10 (§B.2): Un catálogo declarado y ausente falla al cargar los ajustes,
    con la ruta en el mensaje de error."""
    yaml_file = tmp_path / "ausente.yaml"
    yaml_file.write_text(
        "threshold: 0.5\ntop_n: 5\nstopwords: []\nabsence_markers: []\n"
        "catalog:\n  kind: reference\n  files: ['archivo_inexistente_123.csv']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"archivo_inexistente_123.csv"):
        load_matching_settings(yaml_file)


def test_load_matching_settings_rutas_resueltas_contra_raiz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caso 11 (§B.2): Las rutas se resuelven contra la raíz del repositorio,
    no contra el cwd actual (I-094)."""
    monkeypatch.chdir(tmp_path)
    settings = load_matching_settings()
    assert settings.catalog.paths[0].is_file()
    assert settings.catalog.paths[0].name == "cie10-co.csv"


def test_pack_consultorio_declara_reference_con_archivo_existente() -> None:
    """Caso 12 (§B.2): El pack real de consultorio declara reference con un archivo
    y ese archivo existe en el sistema de archivos."""
    settings = load_matching_settings()
    assert settings.catalog.kind == "reference"
    assert len(settings.catalog.paths) == 1
    assert settings.catalog.paths[0].is_file()
    assert settings.catalog.paths[0].name == "cie10-co.csv"


# ---------------------------------------------------------------------------
# Plan 79 (§B.1, §B.2, §B.3): Marcadores de ausencia en catálogo (I-144)
# ---------------------------------------------------------------------------

_CATALOGO_P79 = (
    DiagnosticCode("E140", "DIABETES MELLITUS NO ESPECIFICADA, CON COMA", 3, "test"),
    DiagnosticCode(
        "E149",
        "DIABETES MELLITUS NO ESPECIFICADA, SIN MENCION DE COMPLICACION",
        3,
        "test",
    ),
    DiagnosticCode(
        "B162",
        "HEPATITIS AGUDA TIPO B, SIN AGENTE DELTA, CON COMA HEPATICO",
        3,
        "test",
    ),
)
_VACIAS_P79 = frozenset({"no", "especificada", "con", "sin", "de"})


def test_build_index_con_marcador_variante_sencilla_gana() -> None:
    """Caso 1 (§B.1, M1, M2): Con el marcador, la variante sencilla gana.
    build_index con absence_markers={'sin'}, consulta 'diabetes mellitus', umbral 0.5:
    primer código es E149 con puntuación 1.0."""
    index = build_index(
        _CATALOGO_P79,
        stopwords=_VACIAS_P79,
        absence_markers=frozenset({"sin"}),
    )
    matches = match_codes("diabetes mellitus", index, threshold=0.5, top_n=3)
    assert len(matches) > 0
    assert matches[0].diagnostic.code == "E149"
    assert math.isclose(matches[0].score, 1.0, rel_tol=1e-3)


def test_build_index_sin_marcador_gana_complicada() -> None:
    """Caso 2 (§B.1): Sin el marcador, gana la complicada.
    El mismo catálogo con absence_markers=frozenset(): el primer código es E140."""
    index = build_index(
        _CATALOGO_P79,
        stopwords=_VACIAS_P79,
        absence_markers=frozenset(),
    )
    matches = match_codes("diabetes mellitus", index, threshold=0.5, top_n=3)
    assert len(matches) > 0
    assert matches[0].diagnostic.code == "E140"


def test_build_index_tramo_acaba_en_coma() -> None:
    """Caso 3 (§B.1, M3): El tramo acaba en la coma.
    Consulta 'coma hepatico' con marcador: B162 sale, con puntuación >= 0.5."""
    index = build_index(
        _CATALOGO_P79,
        stopwords=_VACIAS_P79,
        absence_markers=frozenset({"sin"}),
    )
    matches = match_codes("coma hepatico", index, threshold=0.5, top_n=3)
    assert any(m.diagnostic.code == "B162" for m in matches)
    matched_b162 = next(m for m in matches if m.diagnostic.code == "B162")
    assert matched_b162.score >= 0.5


def test_build_index_descripcion_devuelta_intacta() -> None:
    """Caso 4 (§B.1): La descripción que se devuelve está intacta.
    En Caso 1, matches[0].diagnostic.description es la cadena original completa."""
    index = build_index(
        _CATALOGO_P79,
        stopwords=_VACIAS_P79,
        absence_markers=frozenset({"sin"}),
    )
    matches = match_codes("diabetes mellitus", index, threshold=0.5, top_n=3)
    assert len(matches) > 0
    assert (
        matches[0].diagnostic.description
        == "DIABETES MELLITUS NO ESPECIFICADA, SIN MENCION DE COMPLICACION"
    )


def test_build_index_consulta_no_se_recorta() -> None:
    """Caso 5 (§B.1, M4): La consulta no se recorta.
    Catálogo con R51X CEFALEA CON NAUSEAS, stopwords=VACIAS, marcador {'sin'};
    consulta 'cefalea sin nauseas': puntuación es 1.0."""
    cat = (DiagnosticCode("R51X", "CEFALEA CON NAUSEAS", 3, "test"),)
    index = build_index(
        cat,
        stopwords=_VACIAS_P79,
        absence_markers=frozenset({"sin"}),
    )
    matches = match_codes("cefalea sin nauseas", index, threshold=0.5, top_n=1)
    assert len(matches) == 1
    assert matches[0].diagnostic.code == "R51X"
    assert math.isclose(matches[0].score, 1.0, rel_tol=1e-3)


def test_build_index_por_defecto_no_descarta_nada() -> None:
    """Caso 6 (§B.1, M10): Por defecto no se descarta nada.
    build_index(CATALOGO, stopwords=VACIAS) produce exactamente los mismos vectores
    que vectorize(fold(d), stopwords=VACIAS) para cada descripción."""
    from aida.textmatch import fold, vectorize

    index = build_index(_CATALOGO_P79, stopwords=_VACIAS_P79)
    for entry, vec in index.entries:
        expected_vec = vectorize(fold(entry.description), stopwords=_VACIAS_P79)
        assert vec == expected_vec


def test_load_matching_settings_pack_real_declara_absence_markers() -> None:
    """Caso 7 (§B.2): El pack real declara el marcador:
    load_matching_settings().absence_markers == frozenset({'sin'})."""
    settings = load_matching_settings()
    assert settings.absence_markers == frozenset({"sin"})


def test_load_matching_settings_absence_markers_se_pliega(tmp_path: Path) -> None:
    """Caso 8 (§B.2, M5): Se pliega al cargar:
    un YAML con absence_markers: ['SÍN'] da frozenset({'sin'})."""
    yaml_file = tmp_path / "matching_sin.yaml"
    yaml_file.write_text(
        "threshold: 0.5\ntop_n: 5\nstopwords: []\nabsence_markers: ['SÍN']\n"
        "catalog:\n  kind: reference\n  files: ['data/health/cie10/co/cie10-co.csv']\n",
        encoding="utf-8",
    )
    settings = load_matching_settings(yaml_file)
    assert settings.absence_markers == frozenset({"sin"})


def test_load_matching_settings_absence_markers_obligatorio_y_validado(tmp_path: Path) -> None:
    """Caso 9 (§B.2, M6): absence_markers es obligatorio y se valida.
    Sin la clave, ValueError con 'absence_markers' en el mensaje;
    con cadena o enteros, ValueError con 'debe ser una lista de cadenas'."""
    cat_yaml = "catalog:\n  kind: reference\n  files: ['data/health/cie10/co/cie10-co.csv']\n"

    # Falta clave
    sin_markers = tmp_path / "sin_markers.yaml"
    sin_markers.write_text(
        f"threshold: 0.5\ntop_n: 5\nstopwords: []\n{cat_yaml}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"absence_markers"):
        load_matching_settings(sin_markers)

    # Cadena en vez de lista
    str_markers = tmp_path / "str_markers.yaml"
    str_markers.write_text(
        f"threshold: 0.5\ntop_n: 5\nstopwords: []\nabsence_markers: sin\n{cat_yaml}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'absence_markers' debe ser una lista de cadenas"):
        load_matching_settings(str_markers)

    # Lista con entero
    int_markers = tmp_path / "int_markers.yaml"
    int_markers.write_text(
        f"threshold: 0.5\ntop_n: 5\nstopwords: []\nabsence_markers: [1]\n{cat_yaml}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'absence_markers' debe ser una lista de cadenas"):
        load_matching_settings(int_markers)


def test_match_catalogo_real_con_absence_markers() -> None:
    """Caso 10 (§B.3, M1): Con load_matching_settings() y load_reference_catalog(...) del pack,
    y build_index(..., stopwords=s.stopwords, absence_markers=s.absence_markers):
    - diabetes mellitus -> E149
    - reflujo gastroesofágico -> K219
    - infarto agudo de miocardio -> I219"""
    from aida.arms.health.codes import load_reference_catalog

    s = load_matching_settings()
    catalog = load_reference_catalog(s.catalog.paths[0])
    index = build_index(
        catalog,
        stopwords=s.stopwords,
        absence_markers=s.absence_markers,
    )
    m_dm = match_codes("diabetes mellitus", index, threshold=s.threshold, top_n=1)
    assert len(m_dm) == 1
    assert m_dm[0].diagnostic.code == "E149"

    m_ref = match_codes("reflujo gastroesofágico", index, threshold=s.threshold, top_n=1)
    assert len(m_ref) == 1
    assert m_ref[0].diagnostic.code == "K219"

    m_iam = match_codes("infarto agudo de miocardio", index, threshold=s.threshold, top_n=1)
    assert len(m_iam) == 1
    assert m_iam[0].diagnostic.code == "I219"


# ===========================================================================
# Casos del Plan 80 (§B.1): Regla de «palabra nueva» en el cotejo
# ===========================================================================


def test_match_codes_each_alternativa_del_mismo_texto_no_entra() -> None:
    """Caso 1 (§B.1): Una alternativa no entra.
    Catálogo (E039 HIPOTIROIDISMO, E031 HIPOTIROIDISMO CONGENITO), un texto 'hipotiroidismo':
    match_codes_each devuelve solo E039."""
    catalog = (
        DiagnosticCode("E039", "HIPOTIROIDISMO", 3, "test"),
        DiagnosticCode("E031", "HIPOTIROIDISMO CONGENITO", 3, "test"),
    )
    index = build_index(catalog)
    matches = match_codes_each(_codables("hipotiroidismo"), index, threshold=0.5, top_n=6)
    assert tuple(m.diagnostic.code for m in matches) == ("E039",)


def test_match_codes_each_segundo_diagnostico_entra_y_ya_cubierto_no() -> None:
    """Caso 2 (§B.1): Un segundo diagnóstico de la misma frase sí entra, y uno ya cubierto no.
    Catálogo (A01 DIABETES MELLITUS, A02 HIPERTENSION DIABETES, A03 HIPERTENSION),
    un texto 'diabetes mellitus hipertension': devuelve A01 y A02, no A03.
    (A01 cubre diabetes y mellitus; A02 añade hipertension; A03 ya no explica
    palabra nueva.)"""
    catalog = (
        DiagnosticCode("A01", "DIABETES MELLITUS", 3, "test"),
        DiagnosticCode("A02", "HIPERTENSION DIABETES", 3, "test"),
        DiagnosticCode("A03", "HIPERTENSION", 3, "test"),
    )
    index = build_index(catalog)
    matches = match_codes_each(
        _codables("diabetes mellitus hipertension"), index, threshold=0.5, top_n=6
    )
    assert tuple(m.diagnostic.code for m in matches) == ("A01", "A02")


def test_match_codes_each_regla_es_por_texto_no_global() -> None:
    """Caso 3 (§B.1): La regla es por texto, no global.
    Dos textos en orden 'hipotiroidismo congenito' y 'hipotiroidismo', con el catálogo de Caso 1:
    salen E031 y E039. En el primer texto E031 es el primero y E039 no añade palabra;
    en el segundo, E039 es el primero y entra siempre."""
    catalog = (
        DiagnosticCode("E039", "HIPOTIROIDISMO", 3, "test"),
        DiagnosticCode("E031", "HIPOTIROIDISMO CONGENITO", 3, "test"),
    )
    index = build_index(catalog)
    matches = match_codes_each(
        _codables("hipotiroidismo congenito", "hipotiroidismo"), index, threshold=0.5, top_n=6
    )
    assert tuple(m.diagnostic.code for m in matches) == ("E031", "E039")


def test_match_codes_matched_words_es_interseccion_no_descripcion() -> None:
    """Caso 4 (§B.1): matched_words es la intersección, no la descripción.
    match_codes('hipotiroidismo', ...) con catálogo (E039, E031):
    E031 trae matched_words == frozenset({'hipotiroidismo'}), sin 'congenito'."""
    catalog = (
        DiagnosticCode("E039", "HIPOTIROIDISMO", 3, "test"),
        DiagnosticCode("E031", "HIPOTIROIDISMO CONGENITO", 3, "test"),
    )
    index = build_index(catalog)
    matches = match_codes("hipotiroidismo", index, threshold=0.5, top_n=6)
    m_e031 = next(m for m in matches if m.diagnostic.code == "E031")
    assert m_e031.matched_words == frozenset({"hipotiroidismo"})
    assert "congenito" not in m_e031.matched_words


def test_match_codes_each_catalogo_real_hipotiroidismo_solo_e039() -> None:
    """Caso 5 (§B.1): El catálogo real: match_codes_each sobre 'hipotiroidismo'
    con el pack da ('E039',)."""
    from aida.arms.health.codes import load_reference_catalog

    s = load_matching_settings()
    catalog = load_reference_catalog(s.catalog.paths[0])
    index = build_index(catalog, stopwords=s.stopwords, absence_markers=s.absence_markers)
    matches = match_codes_each(
        _codables("hipotiroidismo"), index, threshold=s.threshold, top_n=s.top_n
    )
    assert tuple(m.diagnostic.code for m in matches) == ("E039",)





