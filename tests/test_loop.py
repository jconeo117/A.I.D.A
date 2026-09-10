"""Pruebas del bucle de deliberación (`aida.engine.loop`).

Cubre la unidad U5a según la especificación de `.claude/workflow/12-plan-u5a.md` §6.6.
Verifica:
1. Agente ocioso delibera e inicia acción (`ACTION_STARTED`).
2. Acción completa de 1 tick consume, finaliza, aplica efecto y emite `ACTION_COMPLETED`.
3. Emergencia interrumpe acción interrumpible, emite `ACTION_ABORTED`
   y redelibera con `must_address`.
4. Acción no interrumpible no se abandona ante emergencia.
5. Los eventos de entrada (`inbox`) se vacían y emiten `INPUT_OBSERVED` antes de deliberar.
6. Delegación de `next_wake` en el reloj híbrido.

Protege las 5 mutaciones del bucle:
- Omisión de `agent.advance()`.
- Omisión de `agent.apply(...)` en `_finish`.
- Forzar `interruptible=True` fijo.
- Omisión del bucle sobre `inbox.pop_ready(tick)`.
- Omisión del bloque de avance (`if performing is not None and False`).
"""

from __future__ import annotations

import pytest

import aida.engine.disposition as disposition
import aida.engine.events as events
import aida.engine.loop as loop
import aida.engine.ontology as ontology
import aida.engine.pack as pack


def _make_need(
    id_: str,
    *,
    decay_per_tick: float = 0.01,
    initial: float = 0.5,
    cap: float = 1.0,
    exponent: float = 1.0,
    urgent_when_low: bool = True,
    escalates_after: int = 0,
    strain_weight: float = 0.0,
) -> ontology.NeedSpec:
    return ontology.NeedSpec(
        id=ontology.NeedId(id_),
        label=f"Need {id_}",
        decay_per_tick=decay_per_tick,
        initial=initial,
        cap=cap,
        urgency_exponent=exponent,
        urgent_when_low=urgent_when_low,
        escalates_after=escalates_after,
        strain_weight=strain_weight,
    )


def _make_action(
    id_: str,
    *,
    duration_ticks: int = 1,
    interruptible: bool = True,
    satisfies: dict[str, float] | None = None,
    costs: dict[str, float] | None = None,
    effects: dict[str, float] | None = None,
    saturation_gain: float = 0.2,
) -> ontology.ActionSpec:
    sat = {ontology.NeedId(k): v for k, v in (satisfies or {}).items()}
    cst = {ontology.NeedId(k): v for k, v in (costs or {}).items()}
    eff = {ontology.NeedId(k): v for k, v in (effects or {}).items()}
    if not eff:
        eff = {**sat, **{k: -v for k, v in cst.items()}}
    return ontology.ActionSpec(
        id=ontology.ActionId(id_),
        label=f"Action {id_}",
        duration_ticks=duration_ticks,
        reversibility=ontology.Reversibility.REVERSIBLE,
        interruptible=interruptible,
        satisfies=sat,
        costs=cst,
        effects=eff,
        saturation_gain=saturation_gain,
    )


def _make_pack(
    needs: list[ontology.NeedSpec],
    actions: list[ontology.ActionSpec],
    *,
    critical_need_threshold: float = 0.4,
    switch_cost: float = 0.0,
    saturation_recovery: float = 0.05,
    loss_aversion: float = 1.0,
    strain_multipliers: dict[ontology.TemperamentKnob, float] | None = None,
    agent_spec: ontology.AgentSpec | None = None,
) -> pack.ContentPack:
    tuning = ontology.TuningSpec(
        critical_need_threshold=critical_need_threshold,
        switch_cost=switch_cost,
        saturation_recovery=saturation_recovery,
        loss_aversion=loss_aversion,
        strain_multipliers=strain_multipliers or {},
        candidate_pool=len(actions) or 1,
        temperature=0.01,
    )
    if agent_spec is None:
        agent_spec = ontology.AgentSpec(
            id=ontology.AgentId("agent_01"),
            label="Agent 01",
            initial_needs={n.id: n.initial for n in needs},
        )
    return pack.ContentPack(
        id="pack_loop_test",
        label="Pack Loop Test",
        needs=needs,
        actions=actions,
        agents=[agent_spec],
        tuning=tuning,
    )


def test_loop_idle_agent_deliberates_and_starts_action() -> None:
    """Caso 1: un agente ocioso delibera en step() y arranca la única acción disponible.

    Verifica que:
    - execution.agent.performing no es None.
    - performing.action es la acción elegida.
    - En event log aparece exactamente un evento ACTION_STARTED en tick=1.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", duration_ticks=3, satisfies={"n1": 0.4})
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack([n1], [act1], agent_spec=spec)

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)
    assert assist.execution.tick == 0

    assist.step()

    assert assist.execution.tick == 1
    performing = assist.execution.agent.performing
    assert performing is not None
    assert performing.action == ontology.ActionId("act1")
    assert performing.ticks_remaining == 3
    assert performing.total_ticks == 3

    started_events = assist.log.of_kind(events.EventKind.ACTION_STARTED)
    assert len(started_events) == 1
    assert started_events[0].tick == 1
    assert started_events[0].action == ontology.ActionId("act1")
    assert started_events[0].owner == ontology.AgentId("agent_01")


def test_loop_action_finishes_and_applies_effect() -> None:
    """Caso 2: una acción con duration_ticks=1 consume su tick, finaliza y aplica su efecto.

    En step 1: arranca la acción (duration=1).
    En step 2: avanza, alcanza ticks_remaining=0, llama a _finish(), aplica efectos,
    satura y emite ACTION_COMPLETED.

    Anclas de mutación (§6.6):
    - Quitar 'agent.advance()' antes de ticks_remaining==0 impide que la acción finalice.
    - Quitar 'agent.apply(...)' en '_finish' impide que el eje reponedor suba.
    - Cambiar 'if performing is not None:' por '... and False' nunca avanza ni completa nada.
    """
    n1 = _make_need("n1", initial=0.4, decay_per_tick=0.01)
    act1 = _make_action(
        "act1",
        duration_ticks=1,
        satisfies={"n1": 0.4},
        effects={"n1": 0.4},
        saturation_gain=0.3,
    )
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack([n1], [act1], agent_spec=spec)

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)

    # Step 1: arranca act1
    assist.step()
    assert assist.execution.tick == 1
    assert assist.execution.agent.performing is not None
    assert assist.execution.agent.performing.action == ontology.ActionId("act1")

    # Guardamos el nivel de n1 antes de que se complete
    level_step1 = assist.execution.agent.needs[ontology.NeedId("n1")]

    # Step 2: avanza a 0, completa, aplica efecto y satura
    assist.step()
    assert assist.execution.tick == 2

    # Evento ACTION_COMPLETED emitido en tick 2
    completed_events = assist.log.of_kind(events.EventKind.ACTION_COMPLETED)
    assert len(completed_events) == 1
    assert completed_events[0].tick == 2
    assert completed_events[0].action == ontology.ActionId("act1")

    # Efecto aplicado sobre la necesidad: subió a pesar del decaimiento
    level_step2 = assist.execution.agent.needs[ontology.NeedId("n1")]
    # level_step2 = level_step1 - decay(0.01) + effect(0.4) = level_step1 + 0.39
    assert level_step2 > level_step1
    assert level_step2 == pytest.approx(level_step1 - 0.01 + 0.4)

    # Saturación de la acción se incrementó
    sat = assist.execution.agent.saturation.get(ontology.ActionId("act1"), 0.0)
    assert sat > 0.0


def test_loop_emergency_aborts_interruptible_action() -> None:
    """Caso 3: una necesidad crítica aborta una acción interrumpible en curso.

    El agente inicia una tarea larga 'act_work' (duration=5) en tick 1.
    En tick 2, 'n_crit' cae por debajo de critical_need_threshold (0.4).
    Al ser 'act_work' interrumpible, se emite ACTION_ABORTED con data={"need": n_crit},
    y el agente queda libre para deliberar de inmediato con forced=n_crit,
    seleccionando 'act_relief'.

    Ancla de mutación: omitir _abort o forzar interruptible=False mantendría act_work.
    """
    n_work = _make_need("n_work", initial=0.5, decay_per_tick=0.01)
    n_crit = _make_need(
        "n_crit", initial=0.8, decay_per_tick=0.25, urgent_when_low=True
    )
    act_work = _make_action(
        "act_work",
        duration_ticks=5,
        interruptible=True,
        satisfies={"n_work": 0.8},
    )
    act_relief = _make_action(
        "act_relief",
        duration_ticks=1,
        interruptible=True,
        satisfies={"n_crit": 0.2},
    )
    spec = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="A1",
        tastes=(ontology.TasteSpec(target="act_work", multiplier=3.0),),
    )
    p = _make_pack(
        [n_work, n_crit],
        [act_work, act_relief],
        critical_need_threshold=0.4,
        switch_cost=0.0,
        agent_spec=spec,
    )

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)

    # Step 1: n_crit decae de 0.8 a 0.55 (> 0.4, no crítico).
    # act_work arranca con duración 5 ticks.
    assist.step()
    assert assist.execution.tick == 1
    assert assist.execution.agent.performing is not None
    assert assist.execution.agent.performing.action == ontology.ActionId("act_work")
    assert assist.execution.agent.performing.ticks_remaining == 5
    assert (
        assist.execution.agent.needs[ontology.NeedId("n_crit")] == pytest.approx(0.55)
    )

    # Step 2: n_crit decae de 0.55 a 0.30 (< 0.4 crítico), interrumpe act_work y arranca act_relief
    assist.step()
    assert assist.execution.tick == 2

    # Verifica emisión de ACTION_ABORTED con el need exacto
    aborted_events = assist.log.of_kind(events.EventKind.ACTION_ABORTED)
    assert len(aborted_events) == 1
    assert aborted_events[0].tick == 2
    assert aborted_events[0].action == ontology.ActionId("act_work")
    assert aborted_events[0].data == {"need": ontology.NeedId("n_crit")}

    # Redeliberación forzada eligió act_relief
    performing = assist.execution.agent.performing
    assert performing is not None
    assert performing.action == ontology.ActionId("act_relief")


def test_loop_non_interruptible_action_is_not_aborted() -> None:
    """Caso 4: una acción no interrumpible (interruptible=False) NO se abandona ante emergencia.

    Mismo escenario que el caso 3 pero act_work declara interruptible=False.
    Al llegar la emergencia, NO se emite ACTION_ABORTED y la tarea continúa.

    Ancla de mutación: cambiar 'action.interruptible' por 'True' fijo hace que
    este test caiga porque abortaría una acción no interrumpible.
    """
    n_work = _make_need("n_work", initial=0.5, decay_per_tick=0.01)
    n_crit = _make_need(
        "n_crit", initial=0.8, decay_per_tick=0.25, urgent_when_low=True
    )
    act_work = _make_action(
        "act_work",
        duration_ticks=5,
        interruptible=False,  # NO INTERRUMPIBLE
        satisfies={"n_work": 0.8},
    )
    act_relief = _make_action(
        "act_relief",
        duration_ticks=1,
        interruptible=True,
        satisfies={"n_crit": 0.2},
    )
    spec = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="A1",
        tastes=(ontology.TasteSpec(target="act_work", multiplier=3.0),),
    )
    p = _make_pack(
        [n_work, n_crit],
        [act_work, act_relief],
        critical_need_threshold=0.4,
        switch_cost=0.0,
        agent_spec=spec,
    )

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)

    # Step 1: arranca act_work (ticks_remaining=5)
    assist.step()
    assert assist.execution.agent.performing is not None
    assert assist.execution.agent.performing.action == ontology.ActionId("act_work")
    assert assist.execution.agent.performing.ticks_remaining == 5

    # Step 2: n_crit entra en zona crítica (< 0.4), pero act_work es NO interrumpible
    assist.step()

    # CERO eventos de aborto
    aborted_events = assist.log.of_kind(events.EventKind.ACTION_ABORTED)
    assert len(aborted_events) == 0

    # Sigue ejecutando act_work habiendo consumido 1 tick (5 -> 4)
    performing = assist.execution.agent.performing
    assert performing is not None
    assert performing.action == ontology.ActionId("act_work")
    assert performing.ticks_remaining == 4


def test_loop_inbox_events_emitted_before_deliberating() -> None:
    """Caso 5: eventos del buzón se vacían y emiten INPUT_OBSERVED antes de deliberar.

    Ancla de mutación: omitir 'for item in self.inbox.pop_ready(tick)' provoca que
    los eventos nunca se procesen ni se emitan en el registro.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", duration_ticks=2, satisfies={"n1": 0.4})
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack([n1], [act1], agent_spec=spec)

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)

    # Encolamos evento para tick 2
    assist.inbox.push(
        tick=2, owner=ontology.AgentId("agent_01"), content="alerta_seguridad"
    )

    # Step 1: tick 1, el evento no debe emitirse todavía
    assist.step()
    assert assist.execution.tick == 1
    input_events_t1 = assist.log.of_kind(events.EventKind.INPUT_OBSERVED)
    assert len(input_events_t1) == 0

    # Step 2: tick 2, el evento vence y se emite
    assist.step()
    assert assist.execution.tick == 2
    input_events_t2 = assist.log.of_kind(events.EventKind.INPUT_OBSERVED)
    assert len(input_events_t2) == 1
    assert input_events_t2[0].tick == 2
    assert input_events_t2[0].content == "alerta_seguridad"
    assert input_events_t2[0].owner == ontology.AgentId("agent_01")
    assert len(assist.inbox) == 0


def test_loop_next_wake_delegation() -> None:
    """Verifica que Loop.next_wake delegue correctamente en clock.next_wake."""
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.4})
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack([n1], [act1], agent_spec=spec)

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)
    # Al inicio tick=0, sin inbox -> próximo wake es 1 (latido)
    assert assist.next_wake() == 1

    # Encolar un evento para tick 10 -> latido (1) gana
    assist.inbox.push(tick=10, owner=ontology.AgentId("agent_01"), content="futuro")
    assert assist.next_wake() == 1


def test_loop_next_wake_delegation_inbox_advances_wakeup() -> None:
    """Verifica que Loop.next_wake delegue en clock.next_wake y que el buzón adelante el despertar.

    Ancla I-031: Si en loop.py:63 se reemplaza 'self.inbox.peek_tick()' por 'None',
    el bucle siempre reporta el latido regular (current_tick + 1) e ignora eventos
    urgentes o pendientes en el buzón.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.4})
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack([n1], [act1], agent_spec=spec)

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)

    # 1. Al inicio (tick=0), sin buzón -> latido 1
    assert assist.next_wake() == 1

    # 2. Avanzar a tick=1; buzón vacío -> latido es current_tick + 1 = 2
    assist.step()
    assert assist.execution.tick == 1
    assert assist.next_wake() == 2

    # 3. Encolar evento para tick=1 (justo al día):
    # Con el buzón activo (Opción A), next_wake devuelve 1 ("despertar ya").
    # Si se muta a None, devolvería 2 (el latido normal).
    assist.inbox.push(
        tick=1, owner=ontology.AgentId("agent_01"), content="evento_inmediato"
    )
    assert assist.next_wake() == 1

    # 4. También con un evento atrasado (tick=0), bajo Opción A debe devolver 1 (current_tick)
    assist2 = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)
    assist2.step()
    assert assist2.execution.tick == 1
    assist2.inbox.push(
        tick=0, owner=ontology.AgentId("agent_01"), content="evento_atrasado"
    )
    assert assist2.next_wake() == 1


def test_loop_maintenance_recover_novelty() -> None:
    """Ancla I-032: verifica que Loop.step ejecute agent.recover_novelty.

    Protege la línea loop.py:84 ('agent.recover_novelty(...)').
    Si se elimina esa línea, la saturación acumulada por acciones completadas
    nunca decae con el paso de los ticks, violando la recuperación de novedad.
    """
    n1 = _make_need("n1", initial=0.5, decay_per_tick=0.01)
    act1 = _make_action(
        "act1",
        duration_ticks=2,
        satisfies={"n1": 0.4},
        effects={"n1": 0.4},
        saturation_gain=0.30,
    )
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack([n1], [act1], saturation_recovery=0.05, agent_spec=spec)

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)

    # Step 1: arranca act1 (duration=2)
    assist.step()
    # Step 2: avanza act1 (ticks_remaining: 2 -> 1)
    assist.step()
    # Step 3: avanza a 0, completa, satura a 0.30 y arranca act1 de nuevo (duration=2)
    assist.step()

    act_id = ontology.ActionId("act1")
    assert assist.execution.agent.saturation[act_id] == pytest.approx(0.30)

    # Step 4: durante el mantenimiento, recover_novelty debe restar 0.05 -> 0.25.
    # Como act1 sigue en curso (ticks_remaining pasa de 2 a 1), _finish no corre.
    assist.step()

    assert assist.execution.agent.saturation[act_id] == pytest.approx(0.25)


def test_loop_maintenance_advance_floor_streak() -> None:
    """Ancla I-032: verifica que Loop.step ejecute agent.advance_floor_streak.

    Protege las líneas loop.py:81-83 ('agent.advance_floor_streak(...)').
    Si se eliminan esas líneas, los ejes en 0.0 nunca incrementan su
    floor_streak ni emiten el evento NEED_ESCALATED al alcanzar escalates_after.
    """
    n_floor = _make_need(
        "n_floor",
        initial=0.0,
        decay_per_tick=0.01,
        escalates_after=1,
    )
    act1 = _make_action("act1", duration_ticks=2)
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack([n_floor], [act1], agent_spec=spec)

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)
    need_id = ontology.NeedId("n_floor")

    assert assist.execution.agent.needs[need_id] == 0.0
    assert assist.execution.agent.floor_streak == {}
    assert len(assist.log.of_kind(events.EventKind.NEED_ESCALATED)) == 0

    # Step 1: n_floor decae manteniéndose en 0.0; advance_floor_streak incrementa
    # el streak a 1 y emite NEED_ESCALATED (escalates_after=1).
    assist.step()

    assert assist.execution.agent.floor_streak[need_id] == 1
    escalated = assist.log.of_kind(events.EventKind.NEED_ESCALATED)
    assert len(escalated) == 1
    assert escalated[0].tick == 1
    assert escalated[0].owner == ontology.AgentId("agent_01")
    assert escalated[0].data == {"need": need_id}


def test_loop_maintenance_refresh_state() -> None:
    """Ancla I-032: verifica que Loop.step ejecute agent.refresh_state.

    Protege la línea loop.py:85 ('agent.refresh_state(pack.need_by_id)').
    Si se elimina esa línea, agent.strain no se recalcula tras el decaimiento
    de las necesidades y se desincroniza del estado real del agente.
    """
    n1 = _make_need(
        "n1",
        initial=0.5,
        decay_per_tick=0.1,
        cap=1.0,
        exponent=1.0,
        strain_weight=1.0,
    )
    act1 = _make_action("act1", duration_ticks=3, satisfies={"n1": 0.1})
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack([n1], [act1], agent_spec=spec)

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)
    assert assist.execution.agent.strain == 0.0

    # Step 1: n1 decae de 0.5 a 0.4. Con cap=1.0, exponent=1.0, strain_weight=1.0:
    # compute_strain = urgency(0.4, 1.0, 1.0) = (1 - 0.4) = 0.6.
    assist.step()

    expected_strain = disposition.compute_strain(
        assist.pack.need_by_id,
        assist.execution.agent.needs,
        assist.execution.agent.caps,
    )
    assert expected_strain == pytest.approx(0.6)
    assert assist.execution.agent.strain == pytest.approx(0.6)
    assert assist.execution.agent.strain == pytest.approx(expected_strain)


def test_loop_maintenance_recompose() -> None:
    """Ancla I-032: verifica que Loop.step ejecute agent.recompose.

    Protege la línea loop.py:86 ('agent.recompose(pack)').
    Si se elimina esa línea, agent.temperament conserva los parámetros calculados
    al inicio con strain=0.0 y no refleja el desgaste operativo tras el decaimiento.
    """
    n1 = _make_need(
        "n1",
        initial=0.6,
        decay_per_tick=0.2,
        cap=1.0,
        exponent=1.0,
        strain_weight=1.0,
    )
    act1 = _make_action("act1", duration_ticks=3, satisfies={"n1": 0.1})
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack(
        [n1],
        [act1],
        agent_spec=spec,
        loss_aversion=1.5,
        strain_multipliers={ontology.TemperamentKnob.LOSS_AVERSION: 4.0},
    )

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)

    # Al inicio (tick 0), strain=0.0 -> loss_aversion = 1.5 * (4.0 ** 0.0) = 1.5
    assert assist.execution.agent.temperament is not None
    assert assist.execution.agent.temperament.loss_aversion == pytest.approx(1.5)

    # Step 1: n1 decae de 0.6 a 0.4 -> strain se actualiza a 0.6.
    # recompose debe actualizar temperament con loss_aversion = 1.5 * (4.0 ** 0.6) ≈ 3.4461.
    assist.step()

    expected_loss_aversion = 1.5 * (4.0**0.6)
    assert assist.execution.agent.temperament.loss_aversion == pytest.approx(
        expected_loss_aversion
    )


def test_loop_maintenance_recompose_order_after_decay() -> None:
    """Ancla I-032: verifica que agent.recompose se ejecute DESPUÉS de agent.decay.

    Protege el orden relativo de loop.py:80 ('decay') y loop.py:86 ('recompose').
    Si se mueve recompose antes de decay, el temperamento se calcula con el desgaste
    del tick anterior (pre-decay), en lugar del desgaste fresco del tick actual (post-decay).
    Usa valores numéricos concretos que distinguen ambos estados.
    """
    n1 = _make_need(
        "n1",
        initial=1.0,
        decay_per_tick=0.25,
        cap=1.0,
        exponent=1.0,
        strain_weight=1.0,
    )
    act1 = _make_action("act1", duration_ticks=5, satisfies={"n1": 0.1})
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack(
        [n1],
        [act1],
        agent_spec=spec,
        loss_aversion=1.0,
        strain_multipliers={ontology.TemperamentKnob.LOSS_AVERSION: 4.0},
    )

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)

    # Step 1: n1 decae de 1.0 a 0.75 -> strain post-decay es 0.25.
    # Temperamento post-decay: 1.0 * (4.0 ** 0.25) = 2 ** 0.5 ≈ 1.41421356.
    assist.step()
    assert assist.execution.agent.strain == pytest.approx(0.25)
    assert assist.execution.agent.temperament is not None
    assert assist.execution.agent.temperament.loss_aversion == pytest.approx(
        4.0**0.25
    )

    # Step 2:
    # Estado antes de Step 2: strain=0.25 (pre-decay de Step 2).
    # Durante Step 2: n1 decae de 0.75 a 0.50 -> strain post-decay es 0.50.
    # - Si recompose corre DESPUÉS de decay y refresh_state:
    #     temperament.loss_aversion = 1.0 * (4.0 ** 0.50) = 2.0 (correcto).
    # - Si recompose corre ANTES de decay (mutación):
    #     temperament.loss_aversion = 1.0 * (4.0 ** 0.25) ≈ 1.41421356 (mutante).
    assist.step()

    strain_pre_decay = 0.25
    strain_post_decay = 0.50
    loss_aversion_if_pre_decay = 4.0**strain_pre_decay  # ≈ 1.41421356
    loss_aversion_expected = 4.0**strain_post_decay  # exactamente 2.0

    assert assist.execution.agent.strain == pytest.approx(strain_post_decay)
    assert assist.execution.agent.temperament is not None
    assert assist.execution.agent.temperament.loss_aversion == pytest.approx(
        loss_aversion_expected
    )
    assert assist.execution.agent.temperament.loss_aversion != pytest.approx(
        loss_aversion_if_pre_decay, rel=1e-3
    )


def test_loop_inbox_observed_seq_precedes_action_started() -> None:
    """Ancla I-033: INPUT_OBSERVED se emite con seq menor que ACTION_STARTED en el mismo tick.

    Verifica el orden estricto de ejecución en Loop.step():
    el buzón de entrada se procesa antes de la deliberación (_begin), de modo
    que el seq en EventLog de INPUT_OBSERVED es estrictamente menor que el seq
    de ACTION_STARTED generado en el mismo paso.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.4})
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="A1")
    p = _make_pack([n1], [act1], agent_spec=spec)

    assist = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)
    assist.inbox.push(
        tick=1, owner=ontology.AgentId("agent_01"), content="alerta_entrante"
    )

    # En tick=1 el agente ocioso procesa el buzón y luego delibera arrancando act1
    assist.step()
    assert assist.execution.tick == 1

    input_events = assist.log.of_kind(events.EventKind.INPUT_OBSERVED)
    action_events = assist.log.of_kind(events.EventKind.ACTION_STARTED)

    assert len(input_events) == 1
    assert len(action_events) == 1
    assert input_events[0].seq < action_events[0].seq

