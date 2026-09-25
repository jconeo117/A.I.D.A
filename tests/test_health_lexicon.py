"""Pruebas unitarias del léxico coloquial-clínico (Plan 63: Léxico en datos).

Cubre §B.1 de `.claude/workflow/63-plan-lexico.md`:
1. load_lexicon sobre el pack real: 56 entradas, en minúsculas y sin tildes en la clave.
2. Formato YAML inválido, sin 'terms', 'terms' vacío o no cadena levanta ValueError.
3. clinical_terms reconoce con y sin tildes ("azucar" y "azúcar" dan lo mismo).
4. Límites de palabra: "gonartrosis" no dispara "artrosis"; "del azúcar" sí dispara "azúcar".
5. Dos coloquiales que apuntan al mismo término clínico lo devuelven una sola vez.
6. Lenguaje clínico ("hipertensión esencial") no dispara nada (traduce, no duplica).
7. Orden de salida es el del léxico, no el de aparición en la frase.
8. Cada una de las 56 entradas lleva al menos a un código CIE-10 en el catálogo real.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from aida.arms.health.codes import load_catalog
from aida.arms.health.lexicon import Lexicon, clinical_terms, load_lexicon
from aida.arms.health.match import build_index, load_matching_settings, match_codes

CIE10_JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "health" / "cie10" / "codes.json"
CIE10_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "health" / "cie10" / "cie-10.csv"


def test_load_lexicon_pack_real_56_entradas_minusculas_sin_tildes() -> None:
    """Caso 1 (§B.1): load_lexicon sobre el pack real tiene 56 entradas,
    todas con clave en minúsculas y sin tildes."""
    lex = load_lexicon()
    assert len(lex) == 56
    for coloquial, clinico in lex:
        assert isinstance(coloquial, str) and coloquial
        assert isinstance(clinico, str) and clinico
        # Minúsculas
        assert coloquial == coloquial.lower()
        # Sin tildes / marcas diacríticas
        assert not any(unicodedata.category(c) == "Mn" for c in coloquial)


def test_load_lexicon_validaciones_de_formato(tmp_path: Path) -> None:
    """Caso 2 (§B.1): Sin 'terms', con 'terms' vacío, o con algo que no sea
    cadena levanta ValueError que lo nombra."""
    # 1. No es mapeo YAML
    no_map = tmp_path / "no_map.yaml"
    no_map.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="falta 'terms' en"):
        load_lexicon(no_map)

    # 2. Falta 'terms'
    no_terms = tmp_path / "no_terms.yaml"
    no_terms.write_text("otro: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="falta 'terms' en"):
        load_lexicon(no_terms)

    # 3. 'terms' vacío
    terms_vacio = tmp_path / "empty_terms.yaml"
    terms_vacio.write_text("terms: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'terms' debe ser un mapeo no vacío"):
        load_lexicon(terms_vacio)

    # 4. 'terms' no es mapeo
    terms_lista = tmp_path / "list_terms.yaml"
    terms_lista.write_text("terms:\n  - a\n  - b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'terms' debe ser un mapeo no vacío"):
        load_lexicon(terms_lista)

    # 5. Valor no cadena
    val_int = tmp_path / "val_int.yaml"
    val_int.write_text("terms:\n  azucar: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'terms' solo admite cadenas"):
        load_lexicon(val_int)


def test_clinical_terms_reconoce_con_y_sin_tildes() -> None:
    """Caso 3 (§B.1): clinical_terms reconoce con y sin tildes:
    'azucar' y 'azúcar' dan lo mismo."""
    lex: Lexicon = (("azucar", "diabetes mellitus"),)
    res_sin_tilde = clinical_terms("el paciente sufre de azucar alta", lex)
    res_con_tilde = clinical_terms("el paciente sufre de azúcar alta", lex)

    assert res_sin_tilde == ("diabetes mellitus",)
    assert res_con_tilde == ("diabetes mellitus",)
    assert res_sin_tilde == res_con_tilde


def test_clinical_terms_limites_de_palabra() -> None:
    """Caso 4 (§B.1): Límites de palabra: 'gonartrosis' no dispara 'artrosis';
    'del azúcar' sí dispara 'azúcar'."""
    lex: Lexicon = (
        ("artrosis", "artrosis"),
        ("azucar", "diabetes mellitus"),
    )
    # Subcadena dentro de palabra larga: NO debe disparar
    assert clinical_terms("consulta por gonartrosis derecha", lex) == ()

    # Palabra exacta con límites de palabra: SÍ dispara
    assert clinical_terms("tiene artrosis en la rodilla", lex) == ("artrosis",)
    assert clinical_terms("enfermedad del azúcar desde hace años", lex) == ("diabetes mellitus",)


def test_clinical_terms_dos_coloquiales_mismo_termino_una_sola_vez() -> None:
    """Caso 5 (§B.1): Dos expresiones coloquiales que apuntan al mismo término
    clínico lo devuelven una sola vez."""
    lex: Lexicon = (
        ("azucar", "diabetes mellitus"),
        ("problemas del azucar", "diabetes mellitus"),
    )
    resultado = clinical_terms("tengo azucar y problemas del azucar", lex)
    assert resultado == ("diabetes mellitus",)


def test_clinical_terms_frase_en_lenguaje_clinico_no_dispara_nada() -> None:
    """Caso 6 (§B.1): Una frase en lenguaje clínico —«hipertensión esencial»— no
    dispara nada: el léxico traduce, no duplica."""
    lex = load_lexicon()
    assert clinical_terms("paciente diagnosticado de hipertensión esencial", lex) == ()
    assert clinical_terms("presenta diabetes mellitus tipo 2 sin complicaciones", lex) == ()


def test_clinical_terms_orden_es_el_del_lexico_no_aparicion() -> None:
    """Caso 7 (§B.1): El orden de salida es el del léxico (ordenado determinista),
    no el de aparición en la frase: dos corridas dan la misma tupla."""
    lex = load_lexicon()
    assert lex == tuple(sorted(lex))

    # En el yaml 'azúcar' está antes que 'acidez', pero al ordenar 'acidez' queda primero.
    # En la frase ponemos primero azúcar y luego acidez:
    frase = "tengo problemas con el azucar y ademas mucha acidez"
    res1 = clinical_terms(frase, lex)
    res2 = clinical_terms(frase, lex)

    assert res1 == ("reflujo gastroesofágico", "diabetes mellitus")
    assert res1 == res2


def test_lexicon_cada_entrada_lleva_a_un_codigo_cie10() -> None:
    """Caso 8 (§B.1): Para cada una de las 56 entradas del léxico, match_codes
    con el catálogo real y ajustes del pack devuelve al menos un código."""
    catalog = load_catalog(CIE10_JSON_PATH, CIE10_CSV_PATH)
    settings = load_matching_settings()
    index = build_index(catalog, stopwords=settings.stopwords)
    lex = load_lexicon()

    assert len(lex) == 56
    for coloquial, clinico in lex:
        matches = match_codes(
            clinico,
            index,
            threshold=settings.threshold,
            top_n=settings.top_n,
        )
        assert len(matches) >= 1, (
            f"El término clínico '{clinico}' (de '{coloquial}') no devuelve ningún "
            f"código en el catálogo CIE-10 real con threshold={settings.threshold}"
        )
