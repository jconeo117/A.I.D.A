"""Pruebas unitarias de detección y separación de polaridad clínica (`aida.arms.health.negation`).

Cubre §B.1 de `.claude/workflow/68-plan-negacion-evidencia.md` (Casos 1 a 11).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aida.arms.health.negation import (
    NegationCatalog,
    PolarityParts,
    RetractionRule,
    ThirdPartyCatalog,
    is_third_party,
    load_negation,
    load_retraction,
    load_third_party,
    retracted_replies,
    split_polarity,
)


def test_split_polarity_catalogo_vacio_no_niega_nada() -> None:
    """Caso 1 (§B.1): Catálogo vacío no niega nada."""
    texto = "Refiere dolor torácico"
    assert split_polarity(texto, NegationCatalog()) == PolarityParts(affirmed=texto, negated="")


def test_split_polarity_frase_mixta_con_pack_real() -> None:
    """Caso 2 (§B.1): Frase mixta con catálogo del pack real conserva dolor y aísla negación."""
    catalogo = load_negation()
    resultado = split_polarity("Tiene dolor de cabeza pero no fiebre", catalogo)
    assert resultado.affirmed == "Tiene dolor de cabeza"
    assert resultado.negated == "no fiebre"


def test_split_polarity_frase_enteramente_negada() -> None:
    """Caso 3 (§B.1): Frase enteramente negada deja affirmed vacío."""
    catalogo = load_negation()
    resultado = split_polarity("No refiere fiebre", catalogo)
    assert resultado.affirmed == ""
    assert resultado.negated == "No refiere fiebre"


def test_split_polarity_coma_no_corta_ambito() -> None:
    """Caso 4 (§B.1): La coma no corta ámbito: todo queda en negated."""
    catalogo = load_negation()
    texto = "No refiere fiebre, tos ni dolor"
    resultado = split_polarity(texto, catalogo)
    assert resultado.affirmed == ""
    assert resultado.negated == texto
    assert "fiebre" in resultado.negated
    assert "tos" in resultado.negated
    assert "dolor" in resultado.negated


def test_split_polarity_precedencia_cortador_sobre_senal() -> None:
    """Caso 5 (§B.1): Precedencia del cortador 'sin embargo' sobre la señal 'sin'."""
    catalogo = load_negation()
    resultado = split_polarity("Sin embargo tiene fiebre", catalogo)
    assert resultado.affirmed == "tiene fiebre"
    assert resultado.negated == ""


def test_split_polarity_tildes_y_mayusculas(tmp_path: Path) -> None:
    """Caso 6 (§B.1): Tildes y mayúsculas no impiden la detección; plegado bidireccional."""
    catalogo = load_negation()
    res1 = split_polarity("NO REFIERE FIEBRE", catalogo)
    assert res1.affirmed == ""
    assert res1.negated == "NO REFIERE FIEBRE"

    res2 = split_polarity("Niega fiebre", catalogo)
    assert res2.affirmed == ""
    assert res2.negated == "Niega fiebre"

    # Cue declarado con tilde en pack de prueba
    yaml_tilde = tmp_path / "extraction_tilde.yaml"
    yaml_tilde.write_text(
        "negation:\n  cues:\n    - 'descartó'\n  stoppers:\n    - 'pero'\n",
        encoding="utf-8",
    )
    cat_tilde = load_negation(yaml_tilde)
    res3 = split_polarity("descarto neumonia", cat_tilde)
    assert res3.affirmed == ""
    assert res3.negated == "descarto neumonia"


def test_split_polarity_sin_tokens_no_levanta() -> None:
    """Caso 7 (§B.1): Texto sin tokens utilizables devuelve tramos coherentes sin excepción."""
    catalogo = load_negation()
    resultado = split_polarity("... ???", catalogo)
    assert resultado.affirmed == "... ???"
    assert resultado.negated == ""


def test_split_polarity_doble_negacion_no_reabre() -> None:
    """Caso 8 (§B.1): Doble negación no reabre ámbito; nada vuelve a affirmed."""
    catalogo = load_negation()
    resultado = split_polarity("No está sin fiebre", catalogo)
    assert resultado.affirmed == ""
    assert resultado.negated == "No está sin fiebre"


def test_load_negation_orden_del_catalogo() -> None:
    """Caso 9 (§B.1): Cues y stoppers ordenados por longitud descendente y luego alfabéticamente."""
    catalogo = load_negation()
    assert ("ausencia", "de") in catalogo.cues
    assert ("no",) in catalogo.cues
    idx_ausencia_de = catalogo.cues.index(("ausencia", "de"))
    idx_no = catalogo.cues.index(("no",))
    assert idx_ausencia_de < idx_no
    assert catalogo.cues == tuple(sorted(set(catalogo.cues), key=lambda s: (-len(s), s)))
    assert catalogo.stoppers == tuple(sorted(set(catalogo.stoppers), key=lambda s: (-len(s), s)))


def test_load_negation_carga_estricta(tmp_path: Path) -> None:
    """Caso 10 (§B.1): Carga estricta: formato inválido levanta ValueError nombrando la ruta."""
    def _crear_y_cargar(contenido: str) -> None:
        p = tmp_path / "neg.yaml"
        p.write_text(contenido, encoding="utf-8")
        load_negation(p)

    with pytest.raises(ValueError, match="falta 'negation'") as exc_info:
        _crear_y_cargar("laterality: {}")
    assert str(tmp_path) in str(exc_info.value)

    with pytest.raises(ValueError, match="'cues' debe ser una lista no vacía") as exc_info:
        _crear_y_cargar("negation:\n  stoppers:\n    - pero")
    assert str(tmp_path) in str(exc_info.value)

    with pytest.raises(ValueError, match="'cues' debe ser una lista no vacía") as exc_info:
        _crear_y_cargar("negation:\n  cues: []\n  stoppers:\n    - pero")
    assert str(tmp_path) in str(exc_info.value)

    with pytest.raises(ValueError, match="'cues' debe ser una lista no vacía") as exc_info:
        _crear_y_cargar("negation:\n  cues: 'no'\n  stoppers:\n    - pero")
    assert str(tmp_path) in str(exc_info.value)

    with pytest.raises(ValueError, match="'cues' no admite entradas vacías") as exc_info:
        _crear_y_cargar("negation:\n  cues:\n    - ''\n  stoppers:\n    - pero")
    assert str(tmp_path) in str(exc_info.value)


def test_split_polarity_determinismo() -> None:
    """Caso 11 (§B.1): Determinismo estricto en la separación y carga."""
    cat1 = load_negation()
    cat2 = load_negation()
    assert cat1 == cat2
    texto = "Tiene cefalea pero no náuseas ni vómitos"
    assert split_polarity(texto, cat1) == split_polarity(texto, cat2)


# ---------------------------------------------------------------------------
# Plan 70: Retractación de negaciones cortas (§B.3)
# ---------------------------------------------------------------------------


def test_retracted_replies_caso_real_anamnesis() -> None:
    """Caso 9 (§B.3): Caso real de la consulta retracta la frase anterior."""
    lineas = ("Drolar en el pecho, falta de aire, palpitaciones.", "No, eso no.")
    retractadas = retracted_replies(lineas, load_negation(), RetractionRule(max_chars=24))
    assert "Drolar en el pecho, falta de aire, palpitaciones." in retractadas


def test_retracted_replies_negacion_larga_no_retracta() -> None:
    """Caso 10 (§B.3): Una negación larga no retracta (descripción propia)."""
    lineas = (
        "Y el peso, ha cambiado.",
        "No, no me da vuelta, es más como que me voy a caer como flojera.",
    )
    retractadas = retracted_replies(lineas, load_negation(), RetractionRule(max_chars=24))
    assert len(retractadas) == 0


def test_retracted_replies_frase_mixta_no_retracta() -> None:
    """Caso 11 (§B.3): Una frase mixta con tramo afirmado no retracta."""
    lineas = ("Antecedentes de tabaquismo.", "Fumar, no, lo dejé hace 10 años.")
    retractadas = retracted_replies(lineas, load_negation(), RetractionRule(max_chars=24))
    assert len(retractadas) == 0

    # Frase mixta corta (<= 24 caracteres): sin `not partes.affirmed`, entraría por su longitud.
    lineas_cortas = ("Antecedentes de tabaquismo.", "Fumar, no, lo dejé.")
    assert len("Fumar, no, lo dejé.") <= 24
    retractadas_cortas = retracted_replies(
        lineas_cortas, load_negation(), RetractionRule(max_chars=24)
    )
    assert len(retractadas_cortas) == 0


def test_retracted_replies_primera_frase_nunca_retracta() -> None:
    """Caso 12 (§B.3): La primera frase nunca retracta nada (no hay anterior)."""
    lineas = ("No, eso no.", "Dolor precordial.")
    retractadas = retracted_replies(lineas, load_negation(), RetractionRule(max_chars=24))
    assert len(retractadas) == 0


def test_retracted_replies_previa_vacia_no_entra() -> None:
    """Caso 13 (§B.3): Una previa vacía o en blanco no entra al conjunto."""
    lineas = ("   ", "No, eso no.")
    retractadas = retracted_replies(lineas, load_negation(), RetractionRule(max_chars=24))
    assert len(retractadas) == 0


def test_retracted_replies_frase_identica_repetida_anclada() -> None:
    """Caso 14 (§B.3): Frases idénticas repetidas se retractan ambas (anclado)."""
    lineas = ("Falta de aire.", "No, eso no.", "Falta de aire.", "Sí, eso sí.")
    retractadas = retracted_replies(lineas, load_negation(), RetractionRule(max_chars=24))
    assert "Falta de aire." in retractadas
    assert len(retractadas) == 1


def test_load_retraction_valida_campos_y_valores(tmp_path: Path) -> None:
    """Caso 15 (§B.3): load_retraction valida presencia de 'retraction' y max_chars."""
    def _probar(contenido: str, match: str) -> None:
        p = tmp_path / "ext.yaml"
        p.write_text(contenido, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_retraction(p)

    _probar("negation: {}", "falta 'retraction'")
    _probar("retraction: {}", "retraction.max_chars.*entero positivo")
    _probar("retraction:\n  max_chars: 0", "retraction.max_chars.*entero positivo")
    _probar("retraction:\n  max_chars: -5", "retraction.max_chars.*entero positivo")
    _probar("retraction:\n  max_chars: true", "retraction.max_chars.*entero positivo")
    _probar("retraction:\n  max_chars: veinte", "retraction.max_chars.*entero positivo")


def test_load_retraction_pack_real_declara_24() -> None:
    """Caso 16 (§B.3): El pack real declara max_chars: 24."""
    regla = load_retraction()
    assert regla.max_chars == 24


def test_is_third_party_detecta_familiar_y_no_paciente() -> None:
    """Caso 1 (§B.1): «Mi padre murió de un infarto» es de tercero;
    «Tengo la presión alta» no."""
    catalog = load_third_party()
    assert is_third_party("Mi padre murió de un infarto a los 60.", catalog) is True
    assert is_third_party("Tengo la presión alta desde hace años.", catalog) is False


def test_is_third_party_compara_plegado() -> None:
    """Caso 2 (§B.1, M4): Se compara con texto plegado: «Mi Padre», «MI PADRE»,
    «mi padre» y tildes como «Mí Padre» dan lo mismo."""
    catalog = ThirdPartyCatalog(cues=(("mi", "padre"),))
    assert is_third_party("Mi Padre tuvo un infarto.", catalog) is True
    assert is_third_party("MI PADRE tuvo un infarto.", catalog) is True
    assert is_third_party("mi padre tuvo un infarto.", catalog) is True
    assert is_third_party("Mí Padre tuvo un infarto.", catalog) is True


def test_is_third_party_senal_multipalabra_entera() -> None:
    """Caso 3 (§B.1, M5): Una señal multipalabra se detecta entera:
    «antecedentes familiares» sí, «antecedentes» a secas no."""
    catalog = ThirdPartyCatalog(cues=(("antecedentes", "familiares"),))
    assert is_third_party("Tiene antecedentes familiares de diabetes.", catalog) is True
    assert is_third_party("Tiene antecedentes personales de diabetes.", catalog) is False
    assert is_third_party("Antecedentes quirúrgicos previos.", catalog) is False


def test_load_third_party_valida_claves_y_tipos(tmp_path: Path) -> None:
    """Caso 4 (§B.1): El cargador valida: sin third_party, sin cues, con lista
    vacía o con elemento no string -> ValueError que nombra la ruta."""
    def _probar(contenido: str, match: str) -> None:
        p = tmp_path / "tp.yaml"
        p.write_text(contenido, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_third_party(p)

    _probar("negation: {}", "falta 'third_party'")
    _probar("third_party: {}", r"'cues' debe ser una lista no vacía de cadenas")
    _probar("third_party:\n  cues: []", r"'cues' debe ser una lista no vacía de cadenas")
    _probar("third_party:\n  cues: [123]", r"'cues' debe ser una lista no vacía de cadenas")
    _probar("third_party:\n  cues: 'no_es_lista'", r"'cues' debe ser una lista no vacía de cadenas")


def test_load_third_party_pack_real_trae_senales() -> None:
    """Caso 5 (§B.1): El pack real trae las señales requeridas para terceros."""
    catalog = load_third_party()
    assert len(catalog.cues) >= 10
    cues_strings = {" ".join(cue) for cue in catalog.cues}
    assert "mi padre" in cues_strings
    assert "mi madre" in cues_strings
    assert "antecedentes familiares" in cues_strings
    assert "un vecino" in cues_strings

