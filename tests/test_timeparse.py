"""Pruebas para timeparse.py (§7.1 de 46-plan-extraccion-determinista.md).

Cubre:
- "hace dos semanas" -> "2 semanas"
- "desde hace aproximadamente un mes" -> "1 mes" (singular correcto)
- "hace 10 días" -> "10 días"
- Texto sin ninguna expresión temporal -> None
- "HACE DOS SEMANAS" (mayúsculas) -> "2 semanas"
- Variantes de números en palabras y dígitos, singular y plural.
"""

from __future__ import annotations

import pytest

from aida.timeparse import find_duration


def test_timeparse_hace_dos_semanas() -> None:
    assert find_duration("el paciente refiere dolor hace dos semanas") == "2 semanas"


def test_timeparse_desde_hace_aproximadamente_un_mes_singular() -> None:
    assert (
        find_duration("presenta tos desde hace aproximadamente un mes de evolución")
        == "1 mes"
    )


def test_timeparse_hace_10_dias() -> None:
    assert find_duration("inició síntomas hace 10 días") == "10 días"


def test_timeparse_sin_expresion_temporal_devuelve_none() -> None:
    assert find_duration("paciente refiere cefalea intensa y náuseas") is None
    assert find_duration("") is None
    assert find_duration("hace frío en la sala") is None


def test_timeparse_mayusculas() -> None:
    assert find_duration("HACE DOS SEMANAS COMENZÓ EL DOLOR") == "2 semanas"
    assert find_duration("DESDE HACE APROXIMADAMENTE UN MES") == "1 mes"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hace una semana", "1 semana"),
        ("hace 1 día", "1 día"),
        ("hace un año", "1 año"),
        ("hace 3 años", "3 años"),
        ("hace cinco meses", "5 meses"),
        ("desde hace 2 días", "2 días"),
        ("hace aproximadamente tres semanas", "3 semanas"),
    ],
)
def test_timeparse_variantes_numeros_y_unidades(text: str, expected: str) -> None:
    assert find_duration(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hace 10 dias", "10 días"),
        ("hace dos dias", "2 días"),
        ("hace 3 anos", "3 años"),
        ("hace 1 ano", "1 año"),
        ("hace 1 dia", "1 día"),
        ("HACE DOS DIAS", "2 días"),
    ],
)
def test_timeparse_sin_tildes(text: str, expected: str) -> None:
    assert find_duration(text) == expected

