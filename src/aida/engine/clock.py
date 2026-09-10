"""El reloj híbrido: cuándo toca procesar el siguiente instante.

Dos fuerzas compiten. El latido de fondo quiere avanzar un instante cada vez
que no hay nada más urgente — es lo que detecta que una necesidad se acerca
a su límite sin que llegue ningún aviso de fuera. Un evento entrante, si lo
hay, puede pedir despertar antes. Esta pieza solo sabe comparar esos dos
números y decir cuál gana.

No guarda el instante actual. `ExecutionState.tick` ya es la única fuente de
verdad de en qué instante va la ejecución; duplicarlo aquí sería la misma
regla contada en dos sitios que ya ha costado caro en este proyecto — ver
`I-001`, `I-014`, `I-016` e `I-021` en `.claude/workflow/40-issues.md`.
"""

from __future__ import annotations


def next_wake(current_tick: int, pending_tick: int | None) -> int:
    """El instante en el que debe procesarse el siguiente paso.

    Sin nada pendiente en la cola, es el latido: el instante inmediatamente
    siguiente al actual. Con algo pendiente, gana quien llegue antes — y si
    lo pendiente ya venció (`pending_tick <= current_tick`, por ejemplo
    porque el bucle llevaba un rato sin correr), se despierta ahora mismo,
    no se espera al siguiente latido.
    """
    heartbeat = current_tick + 1
    if pending_tick is None:
        return heartbeat
    return min(heartbeat, max(pending_tick, current_tick))
