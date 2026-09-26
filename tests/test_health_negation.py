"""Pruebas unitarias de detección y separación de polaridad clínica (`aida.arms.health.negation`).

Cubre §B.1 de `.claude/workflow/68-plan-negacion-evidencia.md` (Casos 1 a 11).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aida.arms.health.negation import (
    NegationCatalog,
    PolarityParts,
    load_negation,
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
