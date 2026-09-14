"""El bucle de deliberación: hace correr a un agente, instante a instante.

Ata lo que hasta esta unidad eran piezas sueltas y probadas una por una:
`AgentState.from_spec` para nacer, `clock.next_wake` y `EventQueue` para
saber cuándo tocar el siguiente paso, `decide` para elegir, y los métodos ya
existentes de `AgentState` (`decay`, `advance_floor_streak`,
`recover_novelty`, `refresh_state`, `recompose`, `advance`, `apply`,
`saturate`) para hacer avanzar el resto.

El orden de las operaciones dentro de un paso no es arbitrario, y es el
mismo orden —adaptado a un solo agente— que ya está probado en el bucle de
Agora (`engine/world.py:174-207` de esa referencia): decaer antes de
decidir, para que todo se decida sobre el mismo instante; recomponer la
personalidad con el desgaste ya fresco, para que el temperamento con el que
se delibera sea el de ahora.
"""

from __future__ import annotations

from dataclasses import dataclass

from aida.agents.decide import critical_need, decide
from aida.engine.authority import Authority
from aida.engine.clock import next_wake
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import ActionId, AgentSpec, NeedId
from aida.engine.pack import ContentPack
from aida.engine.queue import EventQueue
from aida.engine.rng import RngSource
from aida.engine.state import AgentState, ExecutionState, PerformingAction


@dataclass(slots=True)
class Loop:
    """Un agente corriendo, con su registro y su buzón de entrada."""

    pack: ContentPack
    execution: ExecutionState
    log: EventLog
    inbox: EventQueue
    authority: Authority

    @classmethod
    def start(
        cls,
        pack: ContentPack,
        spec: AgentSpec,
        *,
        seed: int,
        inbox_capacity: int,
        log_window: int | None = None,
    ) -> Loop:
        """Arranca un agente nuevo a partir de lo que el pack declaró para él."""
        agent = AgentState.from_spec(pack, spec)
        execution = ExecutionState(agent=agent, tick=0, rng=RngSource(seed))
        return cls(
            pack=pack,
            execution=execution,
            log=EventLog(log_window),
            inbox=EventQueue(capacity=inbox_capacity),
            authority=Authority(spec=pack.authority),
        )

    def next_wake(self) -> int:
        """El próximo instante en el que hace falta volver a llamar a `step`."""
        return next_wake(self.execution.tick, self.inbox.peek_tick())

    def step(self) -> None:
        """Un paso del bucle. Ver el orden documentado en la cabecera del módulo."""
        self.execution.tick += 1
        tick = self.execution.tick
        agent = self.execution.agent
        pack = self.pack

        for item in self.inbox.pop_ready(tick):
            self.log.emit(
                tick=tick,
                kind=EventKind.INPUT_OBSERVED,
                owner=agent.agent_id,
                content=item.content,
            )

        for caducada in self.authority.expired(tick):
            self.log.emit(
                tick=tick,
                kind=EventKind.ACTION_EXPIRED,
                owner=agent.agent_id,
                action=caducada,
            )

        agent.decay(pack.need_by_id)
        agent.advance_floor_streak(
            pack.need_by_id, event_log=self.log, owner=agent.agent_id, tick=tick
        )
        agent.recover_novelty(pack.tuning.saturation_recovery)
        agent.refresh_state(pack.need_by_id)
        agent.recompose(pack)

        forced: NeedId | None = None
        performing = agent.performing
        if performing is not None:
            agent.advance()
            performing = agent.performing
            if performing is not None and performing.ticks_remaining == 0:
                self._finish(performing)
            elif (
                performing is not None
                and pack.action_by_id[performing.action].interruptible
            ):
                urgent = critical_need(self.execution, pack)
                if urgent is not None:
                    self._abort(performing, urgent)
                    forced = urgent

        if self.execution.agent.performing is None:
            self._begin(forced)

    def approve(self, action_id: ActionId) -> bool:
        """Concede UNA ejecución de lo propuesto. `False` si no había propuesta."""
        if not self.authority.approve(action_id):
            return False
        self.log.emit(
            tick=self.execution.tick,
            kind=EventKind.ACTION_AUTHORIZED,
            owner=self.execution.agent.agent_id,
            action=action_id,
        )
        return True

    def deny(self, action_id: ActionId) -> bool:
        """Retira lo propuesto sin conceder nada. `False` si no había propuesta."""
        if not self.authority.deny(action_id):
            return False
        self.log.emit(
            tick=self.execution.tick,
            kind=EventKind.ACTION_DENIED,
            owner=self.execution.agent.agent_id,
            action=action_id,
        )
        return True

    def _begin(self, must_address: NeedId | None) -> None:
        agent = self.execution.agent
        vetadas: set[ActionId] = set()
        while True:
            decision = decide(
                self.execution,
                self.pack,
                must_address=must_address,
                forbidden=frozenset(vetadas),
            )
            if decision is None:
                return
            action = self.pack.action_by_id[decision.action]
            if self.authority.may_start(action):
                break
            # Propone y sigue: el techo de autoridad no deja al asistente
            # parado esperando: descarta esta opción y vuelve a deliberar
            # con lo que sí puede hacer solo.
            if self.authority.propose(action, self.execution.tick):
                self.log.emit(
                    tick=self.execution.tick,
                    kind=EventKind.ACTION_PROPOSED,
                    owner=agent.agent_id,
                    action=action.id,
                    data={"expires_at": self.authority.pending[action.id]},
                )
            vetadas.add(action.id)
        agent.performing = PerformingAction(
            action=action.id,
            ticks_remaining=action.duration_ticks,
            total_ticks=action.duration_ticks,
        )
        self.log.emit(
            tick=self.execution.tick,
            kind=EventKind.ACTION_STARTED,
            owner=agent.agent_id,
            action=action.id,
        )

    def _finish(self, performing: PerformingAction) -> None:
        agent = self.execution.agent
        action = self.pack.action_by_id[performing.action]
        agent.apply(action.effects, self.pack.need_by_id)
        agent.saturate(action.id, action.saturation_gain)
        agent.performing = None
        self.log.emit(
            tick=self.execution.tick,
            kind=EventKind.ACTION_COMPLETED,
            owner=agent.agent_id,
            action=action.id,
        )

    def _abort(self, performing: PerformingAction, urgent: NeedId) -> None:
        agent = self.execution.agent
        self.log.emit(
            tick=self.execution.tick,
            kind=EventKind.ACTION_ABORTED,
            owner=agent.agent_id,
            action=performing.action,
            data={"need": urgent},
        )
        agent.performing = None
