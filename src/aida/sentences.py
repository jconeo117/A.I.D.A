"""Partir un texto en frases, sin modelo y sin dependencias: puntuación fuerte,
salvo cuando el punto cierra una abreviatura o va pegado a algo.

Vive aquí y no en un brazo porque es aritmética de texto, como `textmatch.py` y
`timeparse.py`. Las abreviaturas llegan como dato de quien llama: cuáles son
depende del dominio, y el motor no sabe en qué dominio corre.
"""

from __future__ import annotations

import re

_FIN_DE_FRASE = re.compile(r"[.!?]+")


def split_sentences(text: str, *, abbreviations: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """Frases en el orden en que aparecen. Sin puntuación fuerte, el texto
    entero es una sola frase: partir por comas trocearía enumeraciones que son
    un solo hecho."""
    frases: list[str] = []
    inicio = 0
    for marca in _FIN_DE_FRASE.finditer(text):
        fin = marca.end()
        siguiente = text[fin : fin + 1]
        if siguiente and not siguiente.isspace():
            # Un punto con algo pegado detrás no cierra frase: `8.5`, `a.m.`
            continue
        anterior = text[inicio : marca.start()].rstrip()
        ultima = anterior.split()[-1].lower() if anterior.split() else ""
        if f"{ultima}{marca.group(0)}" in abbreviations:
            continue
        frase = text[inicio:fin].strip()
        if frase:
            frases.append(frase)
        inicio = fin
    resto = text[inicio:].strip()
    if resto:
        frases.append(resto)
    return tuple(frases) if frases else ()
