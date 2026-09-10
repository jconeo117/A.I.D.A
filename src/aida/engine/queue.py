"""El buzón de entrada: eventos externos a la espera de procesarse.

Capacidad declarada y disciplina de servicio FIFO (primero el que llegó
primero) — ver README §5.4. Prioridad y triaje se dejan fuera a propósito:
hoy solo hay un remitente posible, y una disciplina que nadie necesita
todavía es superficie muerta esperando a que alguien la escriba mal.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from aida.engine.ontology import AgentId


@dataclass(frozen=True, slots=True)
class IncomingEvent:
    """Algo que llegó de fuera y todavía no se ha procesado."""

    tick: int
    """El instante en que se recibió, no en el que se procesa — pueden
    diferir si la cola lleva trabajo atrasado."""

    owner: AgentId

    content: str
    """El contenido exacto, sin resumir ni normalizar. Es dato, nunca
    instrucción — la regla no negociable de `CLAUDE.md`."""

    seq: int
    """Orden de llegada. Desempate estable cuando dos eventos comparten
    `tick`: nunca un `float`, nunca el orden accidental de un `dict`."""


class QueueFullError(RuntimeError):
    """La cola llegó a su capacidad declarada y no admite más hasta vaciarse."""


@dataclass(slots=True)
class EventQueue:
    """Buzón con capacidad y disciplina FIFO.

    `capacity` es dura: llegar al límite levanta `QueueFullError` en vez de
    descartar en silencio o crecer sin tope. Un buzón que crece sin límite
    es la forma más barata de tumbar un proceso de fondo que nadie vigila.
    """

    capacity: int
    _items: deque[IncomingEvent] = field(default_factory=deque, repr=False)
    _next_seq: int = field(default=0, repr=False)

    def push(self, *, tick: int, owner: AgentId, content: str) -> IncomingEvent:
        """Encola un evento nuevo. Revienta si la cola está llena."""
        if len(self._items) >= self.capacity:
            raise QueueFullError(
                f"cola llena: capacidad={self.capacity}, no admite un evento más"
            )
        item = IncomingEvent(tick=tick, owner=owner, content=content, seq=self._next_seq)
        self._next_seq += 1
        self._items.append(item)
        return item

    def pop_ready(self, current_tick: int) -> list[IncomingEvent]:
        """Todo lo que ya venció (`tick <= current_tick`), en orden de llegada.

        Se detiene en el primero que todavía no vence: como los eventos
        entran siempre con `tick` no decreciente respecto al reloj real, la
        cola ya está ordenada por instante y no hace falta recorrerla entera.
        """
        ready: list[IncomingEvent] = []
        while self._items and self._items[0].tick <= current_tick:
            ready.append(self._items.popleft())
        return ready

    def peek_tick(self) -> int | None:
        """El instante del próximo evento sin procesar, o `None` si no hay
        ninguno. Es lo que `clock.next_wake` necesita para decidir si vale
        la pena despertar antes del próximo latido."""
        return self._items[0].tick if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    @property
    def next_seq(self) -> int:
        """El próximo número de orden que se asignaría. Parte del estado que
        hace falta guardar: sin él, un evento nuevo tras cargar podría repetir
        el número de uno que ya se sacó de la cola antes de guardar."""
        return self._next_seq

    def pending(self) -> tuple[IncomingEvent, ...]:
        """Instantánea de TODO lo que espera, venza o no, en orden de
        llegada. No consume nada — a diferencia de `pop_ready`, es solo para
        mirar. Es lo que hace falta para guardar el buzón entero, no solo lo
        que ya está listo para procesarse."""
        return tuple(self._items)

    @classmethod
    def restore(
        cls, *, capacity: int, items: Iterable[IncomingEvent], next_seq: int
    ) -> EventQueue:
        """Reconstruye un buzón desde una instantánea guardada.

        `next_seq` se recibe explícito y no se recalcula a partir de
        `items`: un buzón vacío en el momento de guardar (`items=()`) no
        puede perder la cuenta de cuántos eventos ya entraron y salieron, o
        un evento nuevo tras cargar reutilizaría el número de uno que ya se
        procesó antes de guardar.
        """
        queue = cls(capacity=capacity)
        queue._items = deque(items)
        queue._next_seq = next_seq
        return queue
