"""Expresiones temporales relativas en español: 'hace dos semanas', 'desde
hace aproximadamente un mes'. Genérico — ningún brazo es dueño del tiempo;
útil para cualquiera que necesite leer una duración de una frase libre, no
solo salud.
"""

from __future__ import annotations

import re

_UNIT_SINGULAR = {
    "día": "día",
    "días": "día",
    "dia": "día",
    "dias": "día",
    "semana": "semana",
    "semanas": "semana",
    "mes": "mes",
    "meses": "mes",
    "año": "año",
    "años": "año",
    "ano": "año",
    "anos": "año",
}
_UNIT_PLURAL = {"día": "días", "semana": "semanas", "mes": "meses", "año": "años"}

_NUMBER_WORDS = {
    "un": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
}

_PATTERN = re.compile(
    r"(?:desde\s+)?hace\s+(?:aproximadamente\s+)?"
    r"(?P<number>\d+|" + "|".join(_NUMBER_WORDS) + r")\s+"
    r"(?P<unit>" + "|".join(_UNIT_SINGULAR) + r")\b",
    re.IGNORECASE,
)


def find_duration(text: str) -> str | None:
    """La primera expresión de duración relativa que aparece, normalizada
    a 'N unidad' con singular/plural correcto (`'1 semana'`, `'2 semanas'`).
    `None` si no hay ninguna — nunca se adivina ni se aproxima."""
    match = _PATTERN.search(text.lower())
    if match is None:
        return None
    number_raw = match.group("number")
    number = int(number_raw) if number_raw.isdigit() else _NUMBER_WORDS[number_raw]
    singular = _UNIT_SINGULAR[match.group("unit")]
    unit = singular if number == 1 else _UNIT_PLURAL[singular]
    return f"{number} {unit}"
