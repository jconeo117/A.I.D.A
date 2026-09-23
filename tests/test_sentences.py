"""Pruebas unitarias de segmentación por frases (I-130, plan 58).

Cubre §B.1 de `.claude/workflow/58-plan-frases.md`:
1. Tres frases con punto conservan su puntuación.
2. Decimales (p.ej. 8.5) no parten la frase, el punto final sí.
3. Abreviaturas declaradas se usan y no parten; sin ellas la misma frase da cuatro.
4. Sin puntuación fuerte el texto entero es una sola frase intacta.
5. Signos de interrogación y exclamación trocean en frases.
6. Puntos suspensivos se conservan en la frase previa.
7. Cadena vacía o de solo espacios devuelve tupla vacía.
"""

from __future__ import annotations

from aida.sentences import split_sentences


def test_split_sentences_tres_frases_con_punto() -> None:
    """Caso 1: tres frases con punto -> tres, con su puntuación conservada."""
    texto = "Primera frase. Segunda frase. Tercera frase."
    assert split_sentences(texto) == ("Primera frase.", "Segunda frase.", "Tercera frase.")


def test_split_sentences_decimal_no_parte() -> None:
    """Caso 2: decimal no parte, punto final sí."""
    texto = "Dolor de 8.5 sobre 10. Nada más."
    assert split_sentences(texto) == ("Dolor de 8.5 sobre 10.", "Nada más.")


def test_split_sentences_abreviaturas_se_usan_y_sin_ellas_cuatro() -> None:
    """Caso 3: abreviaturas declaradas previenen partición; sin abreviaturas parte en cuatro."""
    texto = "El Dr. Pérez lo vio. Tomar 400 mg. cada ocho horas."
    abbrevs = frozenset({"dr.", "mg."})
    con_abbrevs = split_sentences(texto, abbreviations=abbrevs)
    assert con_abbrevs == ("El Dr. Pérez lo vio.", "Tomar 400 mg. cada ocho horas.")

    sin_abbrevs = split_sentences(texto)
    assert sin_abbrevs == ("El Dr.", "Pérez lo vio.", "Tomar 400 mg.", "cada ocho horas.")
    assert len(sin_abbrevs) == 4


def test_split_sentences_sin_puntuacion_fuerte_una_sola_frase() -> None:
    """Caso 4: sin puntuación fuerte, una sola frase con texto intacto."""
    texto = "esta es una frase sin puntuacion fuerte pero con coma, y sigue"
    assert split_sentences(texto) == (texto,)


def test_split_sentences_signos_interrogacion_y_exclamacion() -> None:
    """Caso 5: interrogación y exclamación trocean adecuadamente."""
    texto = "¿Desde cuándo? Desde el martes. ¡Ya!"
    assert split_sentences(texto) == ("¿Desde cuándo?", "Desde el martes.", "¡Ya!")


def test_split_sentences_puntos_suspensivos() -> None:
    """Caso 6: puntos suspensivos trocean y el primer segmento conserva los puntos."""
    texto = "Dos puntos seguidos... y sigue."
    assert split_sentences(texto) == ("Dos puntos seguidos...", "y sigue.")


def test_split_sentences_vacio_o_solo_espacios() -> None:
    """Caso 7: texto vacío o de solo espacios devuelve tupla vacía."""
    assert split_sentences("") == ()
    assert split_sentences("   \n\t  ") == ()
