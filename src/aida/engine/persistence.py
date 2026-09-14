"""Guardar y reanudar la ejecución de un agente.

Se guarda lo que NO se puede recalcular: necesidades, deuda, racha en el
suelo, rasgos, metas, hastío por acción, lo que está haciendo, lo que ha
aprendido, y lo que espera en el buzón de entrada. Lo derivado —pesos de
necesidad, aptitud, temple, techos, temperamento, y los gustos— se recompone
al cargar, llamando a las mismas puertas que ya usa cualquier cambio de
rasgos (`AgentState.recompose`, `AgentState.refresh_state`,
`disposition.compose_tastes`). Guardar el número ya calculado lo dejaría
desincronizado en cuanto el pack se retoque entre una corrida y la
siguiente, que es el caso normal mientras se calibra.

El registro de eventos NO va aquí: es historia, no estado. Ya tiene su
propio mecanismo de persistencia, el volcado a JSONL de `EventLog` (U1b).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aida.engine.authority import Authority
from aida.engine.disposition import compose_tastes
from aida.engine.events import EventLog
from aida.engine.loop import Loop
from aida.engine.ontology import ActionId, AgentId, GoalId, GoalStatus, NeedId, SituationId, TraitId
from aida.engine.pack import ContentPack
from aida.engine.queue import EventQueue, IncomingEvent
from aida.engine.rng import RngSource
from aida.engine.state import AgentState, ExecutionState, PerformingAction

FORMATO = 1
"""Versión del formato en disco. Un guardado de otra versión se rechaza en
vez de cargarse a medias: un estado mal leído es peor que no cargarlo."""


class SaveError(ValueError):
    """El guardado no se puede leer con este pack, o no es de este formato."""


def save(loop: Loop, path: Path) -> None:
    """Vuelca a `path` lo que hace falta para reconstruir `loop` más tarde."""
    agent = loop.execution.agent
    datos: dict[str, Any] = {
        "formato": FORMATO,
        "pack": loop.pack.id,
        "agent_id": str(agent.agent_id),
        "tick": loop.execution.tick,
        "seed": loop.execution.rng.seed,
        "situations": sorted(
            [str(sid), -1 if until is None else until]
            for sid, until in loop.execution.situations.items()
        ),
        "inbox": {
            "capacity": loop.inbox.capacity,
            "next_seq": loop.inbox.next_seq,
            "items": [
                [item.tick, str(item.owner), item.content, item.seq]
                for item in loop.inbox.pending()
            ],
        },
        "agent": {
            "needs": {str(k): v for k, v in sorted(agent.needs.items())},
            "debt": {str(k): v for k, v in sorted(agent.debt.items())},
            "floor_streak": {
                str(k): v for k, v in sorted(agent.floor_streak.items())
            },
            "traits": {str(k): v for k, v in sorted(agent.traits.items())},
            "goals": {str(k): v.value for k, v in sorted(agent.goals.items())},
            "saturation": {
                str(k): v for k, v in sorted(agent.saturation.items())
            },
            "gain_beliefs": {
                str(a): {str(n): v for n, v in sorted(c.items())}
                for a, c in sorted(agent.gain_beliefs.items())
            },
            "cost_beliefs": {
                str(a): {str(n): v for n, v in sorted(c.items())}
                for a, c in sorted(agent.cost_beliefs.items())
            },
            "performing": (
                None
                if agent.performing is None
                else {
                    "action": str(agent.performing.action),
                    "ticks_remaining": agent.performing.ticks_remaining,
                    "total_ticks": agent.performing.total_ticks,
                }
            ),
        },
    }
    path.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")


def load(path: Path, pack: ContentPack, *, log_window: int | None = None) -> Loop:
    """Reconstruye un `Loop` desde `path`, sobre el pack dado.

    El pack puede no ser exactamente el que se guardó — es el caso normal
    mientras se calibra. Lo que no puede ser es OTRO asistente: se comprueba
    el identificador del pack y el del agente antes de tocar nada más.
    """
    datos = json.loads(path.read_text(encoding="utf-8"))

    if datos.get("formato") != FORMATO:
        raise SaveError(
            f"{path}: formato {datos.get('formato')!r}, se esperaba "
            f"{FORMATO}. Un guardado a medias es peor que no cargarlo."
        )
    if datos["pack"] != pack.id:
        raise SaveError(
            f"{path}: guardado del pack {datos['pack']!r} y se está "
            f"cargando {pack.id!r}. Son asistentes distintos."
        )

    agent_id = AgentId(datos["agent_id"])
    spec = pack.agent_by_id.get(agent_id)
    if spec is None:
        raise SaveError(
            f"{path}: el agente {agent_id!r} ya no está declarado en "
            "este pack."
        )

    crudo = datos["agent"]
    known_traits = frozenset(pack.trait_by_id)

    # Reconciliación obligatoria: sin ella, `refresh_state` y `recompose`
    # revientan con un error de clave en cuanto el pack cambió un eje o un
    # rasgo desde que se guardó.
    needs: dict[NeedId, float] = {
        need_id: crudo["needs"].get(str(need_id), need.initial)
        for need_id, need in pack.need_by_id.items()
    }
    traits = {
        TraitId(k): v for k, v in crudo["traits"].items() if TraitId(k) in known_traits
    }

    performing: PerformingAction | None = None
    if crudo["performing"] is not None:
        candidate = PerformingAction(
            action=ActionId(crudo["performing"]["action"]),
            ticks_remaining=crudo["performing"]["ticks_remaining"],
            total_ticks=crudo["performing"]["total_ticks"],
        )
        # Si la acción ya no está en el pack, el próximo `step()` reventaría
        # al buscarla. Se llega como si el agente estuviera libre, no como
        # un error de carga: no es un guardado corrupto, es un pack que
        # cambió.
        if candidate.action in pack.action_by_id:
            performing = candidate

    agent = AgentState(
        agent_id=agent_id,
        needs=needs,
        debt={NeedId(k): v for k, v in crudo["debt"].items()},
        floor_streak={NeedId(k): v for k, v in crudo["floor_streak"].items()},
        traits=traits,
        goals={GoalId(k): GoalStatus(v) for k, v in crudo["goals"].items()},
        saturation={ActionId(k): v for k, v in crudo["saturation"].items()},
        performing=performing,
        gain_beliefs={
            ActionId(a): {NeedId(n): v for n, v in c.items()}
            for a, c in crudo["gain_beliefs"].items()
        },
        cost_beliefs={
            ActionId(a): {NeedId(n): v for n, v in c.items()}
            for a, c in crudo["cost_beliefs"].items()
        },
        # Nunca se guarda: se recalcula siempre desde lo que el pack de HOY
        # declara para este agente. Ver §3.2 del plan de esta unidad.
        tastes=compose_tastes(spec.tastes),
    )

    situations = {
        SituationId(sid): (None if until < 0 else until)
        for sid, until in datos["situations"]
    }
    execution = ExecutionState(
        agent=agent,
        tick=datos["tick"],
        rng=RngSource(datos["seed"]),
        situations=situations,
    )

    # Lo derivado se recompone, nunca se guarda. Primero el desgaste y
    # después la recomposición, porque `compose_temperament` lee `strain`:
    # al revés, el agente cargado tendría el temperamento de un desgaste que
    # ya no es el suyo. `from_spec` no necesita este paso porque un agente
    # recién nacido arranca con `strain = 0.0`.
    agent.refresh_state(pack.need_by_id)
    agent.recompose(pack, active_situations=execution.active_situations(pack.situations))

    inbox_data = datos["inbox"]
    items = tuple(
        IncomingEvent(tick=t, owner=AgentId(owner), content=content, seq=seq)
        for t, owner, content, seq in inbox_data["items"]
    )
    inbox = EventQueue.restore(
        capacity=inbox_data["capacity"], items=items, next_seq=inbox_data["next_seq"]
    )

    return Loop(
        pack=pack,
        execution=execution,
        log=EventLog(log_window),
        inbox=inbox,
        # La autoridad NO se guarda, y es a propósito: un permiso concedido
        # antes de apagar no puede sobrevivir al reinicio, o sería una
        # autorización que nadie dio en esta ejecución. Las propuestas vivas
        # tampoco: al reanudar, el asistente vuelve a proponer y vuelve a
        # preguntar. Volver a preguntar es seguro; recordar un sí, no.
        authority=Authority(spec=pack.authority),
    )
