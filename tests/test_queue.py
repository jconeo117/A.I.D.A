"""Pruebas del buzón de entrada (`aida.engine.queue`).

Cubre la unidad U5a según la especificación de `.claude/workflow/12-plan-u5a.md` §6.5.
Verifica:
1. Capacidad dura y excepción propia `QueueFullError`.
2. Disciplina de servicio FIFO con desempate mediante `seq`.
3. Gating temporal de `pop_ready` y `peek_tick`.
4. Invariantes de inmutabilidad y `__len__`.
Protege las anclas de mutación 8 (límite `>=` vs `>`) y 9 (`popleft` vs `pop`).
"""

from __future__ import annotations

import dataclasses

import pytest

import aida.engine.ontology as ontology
import aida.engine.queue as queue

_AGENT_A = ontology.AgentId("alicia")
_AGENT_B = ontology.AgentId("bob")


def test_queue_capacity_limit_raises_queue_full_error() -> None:
    """Verifica que la cola no admita más elementos al alcanzar su capacidad.

    Ancla de mutación 8: si se cambia `>=` por `>` en la comprobación de capacidad
    de `push()`, este test cae porque permitiría insertar capacity + 1 elementos
    sin levantar `QueueFullError`.
    """
    q = queue.EventQueue(capacity=3)
    assert len(q) == 0

    ev1 = q.push(tick=1, owner=_AGENT_A, content="alerta_1")
    assert ev1.seq == 0
    assert len(q) == 1

    ev2 = q.push(tick=2, owner=_AGENT_A, content="alerta_2")
    assert ev2.seq == 1
    assert len(q) == 2

    ev3 = q.push(tick=3, owner=_AGENT_B, content="alerta_3")
    assert ev3.seq == 2
    assert len(q) == 3

    # El siguiente push debe reventar de inmediato por capacidad estricta
    with pytest.raises(queue.QueueFullError) as exc_info:
        q.push(tick=4, owner=_AGENT_A, content="alerta_desborde")

    # El mensaje de error debe incluir explícitamente la capacidad
    assert "3" in str(exc_info.value)
    assert "capacidad" in str(exc_info.value).lower() or "llena" in str(exc_info.value).lower()
    # La cola permanece con su tamaño máximo sin corromperse
    assert len(q) == 3


def test_queue_fifo_order_with_identical_tick() -> None:
    """Verifica el orden FIFO estricto para eventos recibidos con el mismo tick.

    Ancla de mutación 9: si se cambia `popleft()` por `pop()` en `pop_ready()`,
    este test cae porque los eventos saldrían en orden inverso (LIFO).
    """
    q = queue.EventQueue(capacity=5)
    ev_a = q.push(tick=10, owner=_AGENT_A, content="primero")
    ev_b = q.push(tick=10, owner=_AGENT_B, content="segundo")
    ev_c = q.push(tick=10, owner=_AGENT_A, content="tercero")

    assert ev_a.seq == 0
    assert ev_b.seq == 1
    assert ev_c.seq == 2

    ready = q.pop_ready(current_tick=10)
    assert len(ready) == 3
    assert [item.seq for item in ready] == [0, 1, 2]
    assert [item.content for item in ready] == ["primero", "segundo", "tercero"]
    assert len(q) == 0


def test_queue_pop_ready_temporal_gating() -> None:
    """Verifica que pop_ready retenga eventos futuros y los libere al vencer su tick."""
    q = queue.EventQueue(capacity=5)
    q.push(tick=20, owner=_AGENT_A, content="evento_futuro")

    # A tick 5, no está listo
    assert q.pop_ready(current_tick=5) == []
    assert len(q) == 1
    assert q.peek_tick() == 20

    # A tick 19, aún no está listo
    assert q.pop_ready(current_tick=19) == []
    assert len(q) == 1

    # A tick 20, ya está listo y se extrae
    ready = q.pop_ready(current_tick=20)
    assert len(ready) == 1
    assert ready[0].content == "evento_futuro"
    assert len(q) == 0
    assert q.peek_tick() is None


def test_queue_peek_tick_and_partial_pop() -> None:
    """Verifica peek_tick con cola vacía y con elementos, y extracción parcial."""
    q = queue.EventQueue(capacity=10)
    assert q.peek_tick() is None

    q.push(tick=2, owner=_AGENT_A, content="dos")
    assert q.peek_tick() == 2

    q.push(tick=4, owner=_AGENT_A, content="cuatro")
    q.push(tick=6, owner=_AGENT_A, content="seis")

    # Al tick 3, solo el de tick 2 debe salir
    popped = q.pop_ready(current_tick=3)
    assert len(popped) == 1
    assert popped[0].content == "dos"

    # Ahora el primero de la cola es el de tick 4
    assert q.peek_tick() == 4
    assert len(q) == 2


def test_incoming_event_immutability() -> None:
    """Verifica que IncomingEvent sea frozen dataclass inmutable."""
    ev = queue.IncomingEvent(tick=1, owner=_AGENT_A, content="dato", seq=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.content = "modificado"  # type: ignore[misc]


def test_queue_pending_returns_all_events_fifo_without_consuming() -> None:
    """Verifica que pending() devuelva todos los eventos encolados en orden FIFO sin consumirlos.

    Ancla de mutación 1 (U5b): si se cambia `tuple(self._items)` por `tuple(self._items)[:0]`,
    `pending()` retorna una tupla vacía y este test cae en rojo al comprobar que se
    recuperan los 3 eventos encolados en orden.
    """
    q = queue.EventQueue(capacity=5)
    ev1 = q.push(tick=1, owner=_AGENT_A, content="evento_listo")
    ev2 = q.push(tick=10, owner=_AGENT_B, content="evento_futuro_1")
    ev3 = q.push(tick=20, owner=_AGENT_A, content="evento_futuro_2")

    # pending() debe retornar una tupla con los 3 eventos en orden de llegada sin consumirlos
    snapshot = q.pending()
    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 3
    assert snapshot == (ev1, ev2, ev3)
    assert [e.seq for e in snapshot] == [0, 1, 2]
    assert [e.content for e in snapshot] == [
        "evento_listo",
        "evento_futuro_1",
        "evento_futuro_2",
    ]
    # La cola permanece intacta (no se consumió nada)
    assert len(q) == 3

    # En contraste, pop_ready(current_tick=5) solo consume lo vencido (tick <= 5)
    ready = q.pop_ready(current_tick=5)
    assert len(ready) == 1
    assert ready[0] == ev1
    assert len(q) == 2

    # pending() posterior refleja exactamente los 2 eventos restantes
    remaining = q.pending()
    assert len(remaining) == 2
    assert remaining == (ev2, ev3)


def test_queue_restore_preserves_events_order_and_advances_next_seq() -> None:
    """Verifica que restore() reconstruya el buzón preservando capacidad, eventos y next_seq.

    Ancla de mutación 2 (U5b): si en `restore()` se omite `queue._next_seq = next_seq`,
    `_next_seq` queda en 0 y un push posterior reutiliza un seq ya existente (seq=0),
    haciendo caer este test.
    """
    q = queue.EventQueue(capacity=4)
    ev1 = q.push(tick=5, owner=_AGENT_A, content="alfa")
    ev2 = q.push(tick=6, owner=_AGENT_B, content="beta")
    assert ev1.seq == 0
    assert ev2.seq == 1
    assert q.next_seq == 2

    cap = q.capacity
    n_seq = q.next_seq
    items = q.pending()

    restored = queue.EventQueue.restore(capacity=cap, items=items, next_seq=n_seq)
    assert restored.capacity == cap
    assert restored.next_seq == n_seq
    assert restored.pending() == (ev1, ev2)
    assert len(restored) == 2

    # Un push nuevo en la cola restaurada debe obtener seq=2 y no colisionar con 0 ni 1
    ev3 = restored.push(tick=7, owner=_AGENT_A, content="gamma")
    assert ev3.seq == 2
    assert ev3.seq not in {ev1.seq, ev2.seq}
    assert restored.next_seq == 3
    assert [e.seq for e in restored.pending()] == [0, 1, 2]

