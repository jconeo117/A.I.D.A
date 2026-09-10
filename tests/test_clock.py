"""Pruebas del reloj híbrido (`aida.engine.clock`).

Cubre la unidad U5a según la especificación de `.claude/workflow/12-plan-u5a.md` §6.4.
Verifica la función pura `next_wake(current_tick, pending_tick)` y protege el
ancla de mutación 7 contra la omisión de la protección anti-retroceso temporal.
"""

from __future__ import annotations

import pytest

import aida.engine.clock as clock


def test_next_wake_solo_latido_sin_evento_pendiente() -> None:
    """Caso 1: sin evento pendiente en cola, el próximo despertar es el siguiente latido."""
    assert clock.next_wake(current_tick=10, pending_tick=None) == 11
    assert clock.next_wake(current_tick=0, pending_tick=None) == 1


def test_next_wake_latido_gana_cuando_evento_es_posterior() -> None:
    """Caso 2: si el evento pendiente es posterior al latido, el latido llega antes."""
    assert clock.next_wake(current_tick=10, pending_tick=15) == 11
    assert clock.next_wake(current_tick=10, pending_tick=12) == 11


def test_next_wake_empate_latido_y_evento() -> None:
    """Caso 3: si el evento coincide con el siguiente latido, despierta en ese instante."""
    assert clock.next_wake(current_tick=10, pending_tick=11) == 11
    assert clock.next_wake(current_tick=0, pending_tick=1) == 1


def test_next_wake_evento_atrasado_despierta_ahora_mismo_no_en_el_pasado() -> None:
    """Caso 4: un evento pendiente vencido (atrasado) se procesa ahora mismo (current_tick).

    Garantiza que el reloj no retroceda al pasado (`pending_tick < current_tick`),
    despertando en `current_tick` para atenderlo sin esperar al siguiente latido.
    Ancla de mutación 7: si se sustituye `max(pending_tick, current_tick)` por solo
    `pending_tick`, este test cae porque devolvería 3 en lugar de 10.
    """
    assert clock.next_wake(current_tick=10, pending_tick=3) == 10
    assert clock.next_wake(current_tick=10, pending_tick=0) == 10
    assert clock.next_wake(current_tick=10, pending_tick=10) == 10


@pytest.mark.parametrize(
    ("current", "pending", "expected"),
    [
        (0, None, 1),
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 1),
        (5, None, 6),
        (5, 4, 5),
        (5, 5, 5),
        (5, 6, 6),
        (5, 7, 6),
        (100, 50, 100),
        (100, 100, 100),
        (100, 101, 101),
        (100, 102, 101),
    ],
)
def test_next_wake_parametrizado_fronteras(
    current: int, pending: int | None, expected: int
) -> None:
    """Comprueba de forma exhaustiva las fronteras del reloj híbrido."""
    assert clock.next_wake(current_tick=current, pending_tick=pending) == expected
