"""Pruebas unitarias del catálogo de comandos de voz del operador.

Cubren `aida.voice.commands.CommandCatalog` y `OperatorCommand` (§6.2 de
33-plan-consultorio-voz.md).
"""

from __future__ import annotations

from aida.voice.commands import CommandCatalog, OperatorCommand


def test_command_catalog_recognizes_phrases() -> None:
    """Caso 1: from_phrases con EXIT y SWITCH_MODE reconoce frases esperadas."""
    catalog = CommandCatalog.from_phrases(
        {
            OperatorCommand.EXIT: ("salir",),
            OperatorCommand.SWITCH_MODE: ("cambiar de modo",),
        }
    )
    assert catalog.recognize("salir") is OperatorCommand.EXIT
    assert catalog.recognize("cambiar de modo ya") is OperatorCommand.SWITCH_MODE


def test_command_catalog_unrelated_phrase_returns_none() -> None:
    """Caso 2: Frase sin ningún token en común devuelve None."""
    catalog = CommandCatalog.from_phrases(
        {
            OperatorCommand.EXIT: ("salir", "terminar"),
            OperatorCommand.SWITCH_MODE: ("modo consulta",),
        }
    )
    assert catalog.recognize("palabras completamente diferentes y ajenas") is None


def test_command_catalog_threshold_boundary_strict_less_than() -> None:
    """Caso 3: Puntuación exactamente igual al umbral -> reconocido (< estricto);
    justo por debajo -> None. Umbral 0.5 y frase que comparte la mitad de tokens.
    Mata M7.
    """
    catalog = CommandCatalog.from_phrases(
        {OperatorCommand.EXIT: ("alfa bravo charlie delta",)},
        threshold=0.5,
    )
    # "alfa bravo eco foxtrot" comparte 2 de 4 tokens con norma entera sqrt(4)=2.
    # Cosine = 2 / (2.0 * 2.0) = 0.5 exacto. Igual al umbral -> reconocido (< estricto).
    assert catalog.recognize("alfa bravo eco foxtrot") is OperatorCommand.EXIT

    # "alfa bravo eco foxtrot golf": dot=2, norm_a=2.0, norm_b=sqrt(5) ≈ 2.236.
    # Cosine = 2 / (2 * sqrt(5)) ≈ 0.4472 < 0.5 -> rechaza.
    assert catalog.recognize("alfa bravo eco foxtrot golf") is None


def test_command_catalog_tie_prefers_exit_over_switch_mode() -> None:
    """Caso 4: Empate exacto entre dos comandos -> gana EXIT por orden de valor
    ('exit' < 'switch_mode') y desempate estricto (> best_score).
    Mata M6.
    """
    # Ambos comandos tienen la misma frase de referencia, dando idéntico score
    catalog = CommandCatalog.from_phrases(
        {
            OperatorCommand.SWITCH_MODE: ("comando",),
            OperatorCommand.EXIT: ("comando",),
        }
    )
    assert catalog.recognize("comando") is OperatorCommand.EXIT


def test_command_catalog_empty_string_returns_none() -> None:
    """Caso 5: Cadena vacía o de solo espacios en blanco devuelve None."""
    catalog = CommandCatalog.from_phrases(
        {
            OperatorCommand.EXIT: ("salir",),
            OperatorCommand.SWITCH_MODE: ("consulta",),
        }
    )
    assert catalog.recognize("") is None
    assert catalog.recognize("   ") is None


def test_command_catalog_default_threshold_is_point_six() -> None:
    """I-107: ancla que el umbral por defecto de CommandCatalog es 0.6.
    Una frase con similitud 0.5 no se reconoce con el umbral por defecto (0.5 < 0.6),
    pero sí con un umbral de 0.01.
    """
    catalog = CommandCatalog.from_phrases(
        {OperatorCommand.EXIT: ("alfa bravo charlie delta",)}
    )
    assert catalog.threshold == 0.6
    # "alfa bravo eco foxtrot" comparte 2 de 4 tokens con norma entera sqrt(4)=2.
    # Cosine = 2 / (2.0 * 2.0) = 0.5 exacto.
    assert catalog.recognize("alfa bravo eco foxtrot") is None

    catalog_low = CommandCatalog.from_phrases(
        {OperatorCommand.EXIT: ("alfa bravo charlie delta",)},
        threshold=0.01,
    )
    assert catalog_low.recognize("alfa bravo eco foxtrot") is OperatorCommand.EXIT

