"""Pruebas del módulo de persistencia (`aida.engine.persistence`).

Cubre la unidad U5b según la especificación de `.claude/workflow/14-plan-u5b.md` §6.2-§6.8.
Verifica:
1. Viaje de ida y vuelta (roundtrip) preservando campos persistidos y recomponiendo
   el estado derivado (temperament, need_weights).
2. Reconciliación de necesidades:
   - Nuevos ejes en el pack adoptan su valor inicial (`need.initial != 0.0`).
   - Ejes eliminados del pack se descartan limpiamente sin error.
3. Reconciliación de rasgos:
   - Rasgos eliminados del catálogo del pack se filtran evitando `KeyError` en `recompose`.
4. Reconciliación de acción en curso huérfana:
   - Si la acción fue eliminada del pack, `performing` se normaliza a `None` y `.step()`
     se ejecuta sin error.
5. Gustos del agente:
   - Se recalculan siempre a partir del pack actual y nunca se leen de disco.
6. Validación de formato, identificador de pack y agente:
   - `SaveError` ante formato != 1, pack id discordante o agente no declarado.
7. Determinismo interproceso (I-039):
   - Salvado idéntico byte a byte con PYTHONHASHSEED en "0", "1" y "42".

Protege las anclas de mutación 3 a 9 de U5b.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import aida.engine.events as events
import aida.engine.loop as loop
import aida.engine.ontology as ontology
import aida.engine.pack as pack
import aida.engine.persistence as persistence
import aida.engine.state as state

_TEMPLATE_PACK = Path(__file__).resolve().parents[1] / "packs" / "_plantilla" / "pack.yaml"


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
    max_debt: float = 0.0,
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
        max_debt=max_debt,
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
    *,
    pack_id: str = "pack_persist",
    needs: list[ontology.NeedSpec] | None = None,
    traits: list[ontology.TraitSpec] | None = None,
    actions: list[ontology.ActionSpec] | None = None,
    agents: list[ontology.AgentSpec] | None = None,
    situations: list[ontology.SituationSpec] | None = None,
    tuning: ontology.TuningSpec | None = None,
) -> pack.ContentPack:
    actual_needs = needs if needs is not None else [_make_need("n1", initial=0.5)]
    actual_actions = (
        actions
        if actions is not None
        else [_make_action("act1", duration_ticks=1, satisfies={"n1": 0.3})]
    )
    actual_agents = agents
    if actual_agents is None:
        actual_agents = [
            ontology.AgentSpec(
                id=ontology.AgentId("agent_01"),
                label="Agent 01",
                initial_needs={n.id: n.initial for n in actual_needs},
            )
        ]
    actual_tuning = tuning or ontology.TuningSpec(
        candidate_pool=len(actual_actions) or 1,
        temperature=0.01,
    )
    return pack.ContentPack(
        id=pack_id,
        label=f"Pack {pack_id}",
        needs=actual_needs,
        traits=traits or [],
        actions=actual_actions,
        agents=actual_agents,
        situations=situations or [],
        tuning=actual_tuning,
    )


def test_save_load_roundtrip_preserves_fields_and_recomposes_derived_state(
    tmp_path: Path,
) -> None:
    """§6.2: Viaje completo de ida y vuelta preservando estado y recomponiendo derivados.

    Ancla de mutación 3: si se elimina `agent.recompose` al final de `load`,
    el temperamento no se recompone (`temperament is None`) y este test cae en rojo.
    """
    n1 = _make_need("n1", initial=0.5, decay_per_tick=0.02)
    t1 = ontology.TraitSpec(
        id=ontology.TraitId("t1"),
        label="Trait 1",
        need_weight_multipliers={ontology.NeedId("n1"): 1.4},
    )
    act1 = _make_action("act1", duration_ticks=2, satisfies={"n1": 0.3})
    agent_spec = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        initial_needs={ontology.NeedId("n1"): 0.5},
        traits={ontology.TraitId("t1"): 0.6},
    )
    sit1 = ontology.SituationSpec(id=ontology.SituationId("sit1"), label="Sit 1")
    p = _make_pack(
        pack_id="pack_roundtrip",
        needs=[n1],
        traits=[t1],
        actions=[act1],
        agents=[agent_spec],
        situations=[sit1],
    )

    orig_loop = loop.Loop.start(p, agent_spec, seed=42, inbox_capacity=5)
    # Encolar un evento en el buzón y fijar una situación activa
    orig_loop.inbox.push(tick=10, owner=agent_spec.id, content="alerta_inbox")
    orig_loop.execution.situations[ontology.SituationId("sit1")] = 25

    # Asignar creencias para verificar su serialización y restauración
    orig_loop.execution.agent.gain_beliefs[ontology.ActionId("act1")] = {
        ontology.NeedId("n1"): 1.1
    }
    orig_loop.execution.agent.cost_beliefs[ontology.ActionId("act1")] = {
        ontology.NeedId("n1"): 0.95
    }

    # Avanzar el bucle dos pasos para acumular historia de ejecución
    orig_loop.step()
    orig_loop.step()

    save_path = tmp_path / "partida.json"
    persistence.save(orig_loop, save_path)
    assert save_path.exists()

    # Comprobar la estructura serializada en disco:
    # No debe guardar derivados (ni temperament, ni need_weights, ni tastes)
    datos_crudos = json.loads(save_path.read_text(encoding="utf-8"))
    assert datos_crudos["formato"] == 1
    assert datos_crudos["pack"] == "pack_roundtrip"
    assert datos_crudos["agent_id"] == "agent_01"
    assert "temperament" not in datos_crudos["agent"]
    assert "need_weights" not in datos_crudos["agent"]
    assert "tastes" not in datos_crudos["agent"]

    loaded_loop = persistence.load(save_path, p)

    # 1. Comprobar igualdad estricta de todos los campos de ejecución
    assert loaded_loop.execution.tick == orig_loop.execution.tick
    assert loaded_loop.execution.rng.seed == orig_loop.execution.rng.seed
    assert loaded_loop.execution.situations == orig_loop.execution.situations

    # 2. Comprobar buzón de entrada
    assert loaded_loop.inbox.capacity == orig_loop.inbox.capacity
    assert loaded_loop.inbox.next_seq == orig_loop.inbox.next_seq
    assert loaded_loop.inbox.pending() == orig_loop.inbox.pending()

    # 3. Comprobar todos los campos persistidos del agente
    loaded_agent = loaded_loop.execution.agent
    orig_agent = orig_loop.execution.agent
    assert loaded_agent.agent_id == orig_agent.agent_id
    assert loaded_agent.needs == orig_agent.needs
    assert loaded_agent.debt == orig_agent.debt
    assert loaded_agent.floor_streak == orig_agent.floor_streak
    assert loaded_agent.traits == orig_agent.traits
    assert loaded_agent.goals == orig_agent.goals
    assert loaded_agent.saturation == orig_agent.saturation
    assert loaded_agent.gain_beliefs == orig_agent.gain_beliefs
    assert loaded_agent.cost_beliefs == orig_agent.cost_beliefs
    assert loaded_agent.performing == orig_agent.performing

    # 4. Comprobar que el estado derivado se recompuso correctamente (Ancla Mutante 3)
    assert loaded_agent.temperament is not None
    assert len(loaded_agent.need_weights) > 0
    assert loaded_agent.temperament == orig_agent.temperament
    assert loaded_agent.need_weights == orig_agent.need_weights


def test_load_reconciliation_new_need_in_pack_gets_initial_value(tmp_path: Path) -> None:
    """§6.3: Un eje nuevo añadido en el pack de carga adopta su valor inicial no nulo.

    Ancla de mutación 4: si se sustituye `need.initial` por `0.0` fijo en `load`,
    el eje nuevo queda en 0.0 y este test cae en rojo al esperar 0.85.
    Ancla de mutación 5: si se indexa directamente `crudo["needs"][str(need_id)]`
    sin fallback, cae con `KeyError` al no encontrar el eje nuevo en el guardado.
    """
    n1 = _make_need("n1", initial=0.4)
    spec1 = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        initial_needs={ontology.NeedId("n1"): 0.4},
    )
    p1 = _make_pack(pack_id="pack_needs", needs=[n1], agents=[spec1])
    loop1 = loop.Loop.start(p1, spec1, seed=42, inbox_capacity=5)

    save_path = tmp_path / "save_needs.json"
    persistence.save(loop1, save_path)

    # Segundo pack: añade la necesidad "n2" con un initial claramente distinto de 0.0 (0.85)
    n2 = _make_need("n2", initial=0.85)
    spec2 = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        initial_needs={ontology.NeedId("n1"): 0.4, ontology.NeedId("n2"): 0.85},
    )
    p2 = _make_pack(pack_id="pack_needs", needs=[n1, n2], agents=[spec2])

    loaded_loop = persistence.load(save_path, p2)
    loaded_agent = loaded_loop.execution.agent

    assert loaded_agent.needs[ontology.NeedId("n2")] == 0.85
    expected_n1 = loop1.execution.agent.needs[ontology.NeedId("n1")]
    assert loaded_agent.needs[ontology.NeedId("n1")] == expected_n1
    assert set(loaded_agent.needs.keys()) == {ontology.NeedId("n1"), ontology.NeedId("n2")}


def test_load_reconciliation_removed_need_in_pack_is_pruned(tmp_path: Path) -> None:
    """§6.3: Un eje eliminado en el pack de carga se descarta limpiamente.

    Verifica que `agent.needs` contiene exactamente las claves del nuevo pack
    sin corromper el estado ni reventar en la carga.
    """
    n1 = _make_need("n1", initial=0.4)
    n2 = _make_need("n2", initial=0.7)
    spec1 = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        initial_needs={ontology.NeedId("n1"): 0.4, ontology.NeedId("n2"): 0.7},
    )
    p1 = _make_pack(pack_id="pack_prune_need", needs=[n1, n2], agents=[spec1])
    loop1 = loop.Loop.start(p1, spec1, seed=42, inbox_capacity=5)

    save_path = tmp_path / "save_prune.json"
    persistence.save(loop1, save_path)

    # Segundo pack: elimina "n2" del catálogo
    spec2 = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        initial_needs={ontology.NeedId("n1"): 0.4},
    )
    p2 = _make_pack(pack_id="pack_prune_need", needs=[n1], agents=[spec2])

    loaded_loop = persistence.load(save_path, p2)
    loaded_agent = loaded_loop.execution.agent

    assert set(loaded_agent.needs.keys()) == {ontology.NeedId("n1")}
    assert ontology.NeedId("n2") not in loaded_agent.needs
    assert loaded_agent.temperament is not None


def test_load_reconciliation_removed_trait_from_catalog_is_pruned(tmp_path: Path) -> None:
    """§6.4: Un rasgo eliminado del catálogo del pack de carga se descarta.

    Ancla de mutación 6: si se quita el filtro `if TraitId(k) in known_traits` en `load`,
    el rasgo no declarado se conserva en `agent.traits`, provocando un `KeyError` dentro
    de `agent.recompose` al intentar acceder a `pack.trait_by_id[trait_id]`.
    """
    n1 = _make_need("n1", initial=0.5)
    t1 = ontology.TraitSpec(
        id=ontology.TraitId("t1"),
        label="Trait 1",
        need_weight_multipliers={ontology.NeedId("n1"): 1.5},
    )
    spec1 = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        traits={ontology.TraitId("t1"): 0.7},
    )
    p1 = _make_pack(pack_id="pack_traits", needs=[n1], traits=[t1], agents=[spec1])
    loop1 = loop.Loop.start(p1, spec1, seed=42, inbox_capacity=5)

    save_path = tmp_path / "save_trait.json"
    persistence.save(loop1, save_path)

    # Segundo pack: el rasgo "t1" es eliminado del catálogo de rasgos por completo
    spec2 = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="Agent 01", traits={})
    p2 = _make_pack(pack_id="pack_traits", needs=[n1], traits=[], agents=[spec2])

    loaded_loop = persistence.load(save_path, p2)
    loaded_agent = loaded_loop.execution.agent

    assert ontology.TraitId("t1") not in loaded_agent.traits
    assert loaded_agent.traits == {}
    assert loaded_agent.temperament is not None


def test_load_reconciliation_orphan_performing_action_cleared_and_step_survives(
    tmp_path: Path,
) -> None:
    """§6.5: Acción en curso borrada del pack: performing se vuelve None y step() sobrevive.

    Ancla de mutación 7: si se omite el chequeo `if candidate.action in pack.action_by_id:`
    dejando `performing = candidate`, la llamada a `.step()` sobre el bucle cargado revienta
    con `KeyError` al buscar la acción huérfana en el catálogo del pack.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.3})
    act2 = _make_action("act2", duration_ticks=5, satisfies={"n1": 0.5})
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="Agent 01")
    p1 = _make_pack(pack_id="pack_orphan", needs=[n1], actions=[act1, act2], agents=[spec])

    loop1 = loop.Loop.start(p1, spec, seed=42, inbox_capacity=5)
    # Forzar que el agente esté ejecutando la acción "act2"
    loop1.execution.agent.performing = state.PerformingAction(
        action=ontology.ActionId("act2"),
        ticks_remaining=3,
        total_ticks=5,
    )

    save_path = tmp_path / "save_orphan.json"
    persistence.save(loop1, save_path)

    # Segundo pack: la acción "act2" ha sido eliminada del catálogo
    p2 = _make_pack(pack_id="pack_orphan", needs=[n1], actions=[act1], agents=[spec])

    loaded_loop = persistence.load(save_path, p2)
    assert loaded_loop.execution.agent.performing is None

    # Verificar que el bucle puede seguir deliberando y ejecutando sin reventar con KeyError
    loaded_loop.step()
    assert loaded_loop.execution.tick == loop1.execution.tick + 1


def test_load_recalculates_tastes_from_current_pack_not_persisted(tmp_path: Path) -> None:
    """§6.6: Los gustos nunca se guardan en disco y se recalculan del pack de carga.

    Ancla de mutación 8: si en vez de recalcular desde el pack con `compose_tastes(spec.tastes)`
    se fija `tastes={}`, este test cae en rojo porque el agente cargado no refleja el valor 0.6.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.3})
    taste1 = ontology.TasteSpec(target=ontology.ActionId("act1"), multiplier=1.5)
    spec1 = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        tastes=[taste1],
    )
    p1 = _make_pack(pack_id="pack_tastes", needs=[n1], actions=[act1], agents=[spec1])

    loop1 = loop.Loop.start(p1, spec1, seed=42, inbox_capacity=5)
    assert loop1.execution.agent.tastes["act1"] == 1.5

    save_path = tmp_path / "save_tastes.json"
    persistence.save(loop1, save_path)

    # Verificar que no hay campo 'tastes' dentro del bloque 'agent' en el JSON
    datos_disco = json.loads(save_path.read_text(encoding="utf-8"))
    assert "tastes" not in datos_disco["agent"]

    # Segundo pack: el gusto para act1 se recalibra a 0.6
    taste2 = ontology.TasteSpec(target=ontology.ActionId("act1"), multiplier=0.6)
    spec2 = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        tastes=[taste2],
    )
    p2 = _make_pack(pack_id="pack_tastes", needs=[n1], actions=[act1], agents=[spec2])

    loaded_loop = persistence.load(save_path, p2)
    assert loaded_loop.execution.agent.tastes["act1"] == 0.6
    assert loaded_loop.execution.agent.tastes["act1"] != 1.5


def test_load_wrong_format_raises_save_error(tmp_path: Path) -> None:
    """§6.7: Cargar un archivo con formato != FORMATO (1) levanta SaveError.

    Ancla de mutación 9a: omitir el chequeo de formato hace que este test caiga.
    """
    n1 = _make_need("n1", initial=0.5)
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="Agent 01")
    p = _make_pack(pack_id="pack_fmt", needs=[n1], agents=[spec])
    loop1 = loop.Loop.start(p, spec, seed=42, inbox_capacity=5)

    save_path = tmp_path / "save_bad_format.json"
    persistence.save(loop1, save_path)

    # Alterar el número de formato a 2 en el JSON guardado
    raw = json.loads(save_path.read_text(encoding="utf-8"))
    raw["formato"] = 2
    save_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(persistence.SaveError) as exc_info:
        persistence.load(save_path, p)
    assert "formato" in str(exc_info.value).lower()


def test_load_wrong_pack_id_raises_save_error(tmp_path: Path) -> None:
    """§6.7: Cargar un archivo guardado con otro pack.id levanta SaveError.

    Ancla de mutación 9b: omitir el chequeo de pack.id hace que este test caiga.
    """
    n1 = _make_need("n1", initial=0.5)
    spec = ontology.AgentSpec(id=ontology.AgentId("agent_01"), label="Agent 01")
    p_a = _make_pack(pack_id="pack_a", needs=[n1], agents=[spec])
    p_b = _make_pack(pack_id="pack_b", needs=[n1], agents=[spec])
    loop1 = loop.Loop.start(p_a, spec, seed=42, inbox_capacity=5)

    save_path = tmp_path / "save_pack_mismatch.json"
    persistence.save(loop1, save_path)

    with pytest.raises(persistence.SaveError) as exc_info:
        persistence.load(save_path, p_b)
    assert "pack_a" in str(exc_info.value) or "pack_b" in str(exc_info.value)


def test_load_agent_not_declared_in_pack_raises_save_error(tmp_path: Path) -> None:
    """§6.7: Cargar un archivo cuyo agent_id no está en pack.agent_by_id levanta SaveError.

    Ancla de mutación 9c: omitir el chequeo de agente no declarado hace que este test caiga.
    """
    n1 = _make_need("n1", initial=0.5)
    spec_a = ontology.AgentSpec(id=ontology.AgentId("agent_a"), label="Agent A")
    spec_b = ontology.AgentSpec(id=ontology.AgentId("agent_b"), label="Agent B")
    p_with_a = _make_pack(pack_id="pack_ag", needs=[n1], agents=[spec_a])
    p_with_b = _make_pack(pack_id="pack_ag", needs=[n1], agents=[spec_b])
    loop1 = loop.Loop.start(p_with_a, spec_a, seed=42, inbox_capacity=5)

    save_path = tmp_path / "save_agent_mismatch.json"
    persistence.save(loop1, save_path)

    with pytest.raises(persistence.SaveError) as exc_info:
        persistence.load(save_path, p_with_b)
    assert "agent_a" in str(exc_info.value)


def test_save_load_roundtrip_preserves_debt_floor_streak_saturation_goals(
    tmp_path: Path,
) -> None:
    """§6.2 (I-040): debt, floor_streak, saturation y goals con valores de verdad.

    El roundtrip original comparaba `{} == {}` para estos cuatro campos, así
    que vaciarlos en `save` no lo hacía caer. Aquí cada campo se puebla con un
    valor distinto de los demás y el aserto es contra el valor concreto
    esperado, nunca contra `orig_agent.<campo>`.

    Ancla de mutación: sustituir cualquiera de los cuatro por `{}` en
    `persistence.save` tiene que hacer caer el aserto correspondiente.
    """
    n1 = _make_need(
        "n1", initial=0.05, decay_per_tick=0.2, max_debt=1.0, escalates_after=1
    )
    n2 = _make_need("n2", initial=0.5, decay_per_tick=0.01)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n2": 0.3})
    goal_base = ontology.GoalSpec(
        id=ontology.GoalId("g_base"), label="Base", need=ontology.NeedId("n2"), at_least=0.9
    )
    goal_dependent = ontology.GoalSpec(
        id=ontology.GoalId("g_dep"),
        label="Dependiente",
        need=ontology.NeedId("n2"),
        at_least=0.95,
        requires_goal=ontology.GoalId("g_base"),
    )
    agent_spec = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        initial_needs={ontology.NeedId("n1"): 0.05, ontology.NeedId("n2"): 0.5},
        goals=(goal_base, goal_dependent),
    )
    p = _make_pack(
        pack_id="pack_valores", needs=[n1, n2], actions=[act1], agents=[agent_spec]
    )

    orig_loop = loop.Loop.start(p, agent_spec, seed=42, inbox_capacity=5)
    orig_agent = orig_loop.execution.agent

    # debt: un gasto que no cabe en n1 (decay 0.2 sobre un eje casi vacío).
    orig_loop.step()
    assert orig_agent.debt != {}
    # floor_streak: n1 llega a 0.0 y se queda, con escalates_after=1.
    assert orig_agent.floor_streak != {}
    # saturation: act1 satisface n2 y con duration_ticks=1 completa en el segundo step.
    orig_loop.step()
    assert orig_agent.saturation != {}
    # goals: dos metas con estados distintos (LOCKED vs ACTIVE).
    assert orig_agent.goals == {
        ontology.GoalId("g_base"): ontology.GoalStatus.ACTIVE,
        ontology.GoalId("g_dep"): ontology.GoalStatus.LOCKED,
    }

    expected_debt = dict(orig_agent.debt)
    expected_floor_streak = dict(orig_agent.floor_streak)
    expected_saturation = dict(orig_agent.saturation)
    expected_goals = dict(orig_agent.goals)
    assert expected_debt and expected_floor_streak and expected_saturation and expected_goals

    save_path = tmp_path / "valores.json"
    persistence.save(orig_loop, save_path)
    loaded_agent = persistence.load(save_path, p).execution.agent

    assert loaded_agent.debt == expected_debt
    assert loaded_agent.floor_streak == expected_floor_streak
    assert loaded_agent.saturation == expected_saturation
    assert loaded_agent.goals == expected_goals


def test_save_load_roundtrip_recomputes_nonzero_strain_and_temperament(
    tmp_path: Path,
) -> None:
    """§6.2 (I-041): el desgaste recompuesto al cargar no es cero.

    El roundtrip original usaba `strain_weight=0.0`, así que `compute_strain`
    daba 0.0 con o sin `refresh_state`. Aquí una `NeedSpec` con
    `strain_weight > 0` y un `TuningSpec` con `strain_multipliers` hacen que
    el temperamento cargado dependa realmente del desgaste.

    Ancla de mutación: quitar `agent.refresh_state(pack.need_by_id)` de
    `load` deja `strain` en 0.0 en vez del valor real, y el aserto de
    `loss_aversion` cae.
    """
    n1 = _make_need(
        "n1", initial=1.0, decay_per_tick=0.25, cap=1.0, exponent=1.0, strain_weight=1.0
    )
    act1 = _make_action("act1", duration_ticks=5, satisfies={"n1": 0.1})
    agent_spec = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        initial_needs={ontology.NeedId("n1"): 1.0},
    )
    tuning = ontology.TuningSpec(
        candidate_pool=1,
        temperature=0.01,
        loss_aversion=1.0,
        strain_multipliers={ontology.TemperamentKnob.LOSS_AVERSION: 4.0},
    )
    p = _make_pack(
        pack_id="pack_strain",
        needs=[n1],
        actions=[act1],
        agents=[agent_spec],
        tuning=tuning,
    )

    orig_loop = loop.Loop.start(p, agent_spec, seed=42, inbox_capacity=5)
    orig_loop.step()

    orig_agent = orig_loop.execution.agent
    assert orig_agent.strain == pytest.approx(0.25)
    assert orig_agent.temperament is not None
    expected_loss_aversion = 1.0 * (4.0**0.25)
    assert orig_agent.temperament.loss_aversion == pytest.approx(expected_loss_aversion)
    assert expected_loss_aversion != pytest.approx(1.0)

    save_path = tmp_path / "strain.json"
    persistence.save(orig_loop, save_path)
    loaded_agent = persistence.load(save_path, p).execution.agent

    assert loaded_agent.strain == pytest.approx(0.25)
    assert loaded_agent.temperament is not None
    assert loaded_agent.temperament.loss_aversion == pytest.approx(expected_loss_aversion)
    assert loaded_agent.temperament.loss_aversion != pytest.approx(1.0)


def test_save_orden_estable_independiente_de_insercion(tmp_path: Path) -> None:
    """E-001 (I-042): `save` produce texto idéntico con inserción en orden distinto.

    Los diccionarios de `AgentState` se pueblan en el orden en que los ejes
    y acciones se disparan durante la corrida; dos agentes con las mismas
    claves insertadas al revés deben serializar byte a byte igual, porque
    `save` ordena antes de volcar. La prueba entre procesos (I-039) no cubre
    esto: ambos procesos insertan en el mismo orden.
    """
    n_a = _make_need("n_a", initial=0.5)
    n_b = _make_need("n_b", initial=0.6)
    agent_spec = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agent 01",
        initial_needs={ontology.NeedId("n_a"): 0.5, ontology.NeedId("n_b"): 0.6},
    )
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n_a": 0.1})
    act2 = _make_action("act2", duration_ticks=1, satisfies={"n_b": 0.1})
    p = _make_pack(
        pack_id="pack_orden", needs=[n_a, n_b], actions=[act1, act2], agents=[agent_spec]
    )

    loop_directo = loop.Loop.start(p, agent_spec, seed=1, inbox_capacity=5)
    loop_directo.execution.agent.needs = {
        ontology.NeedId("n_a"): 0.4,
        ontology.NeedId("n_b"): 0.7,
    }
    loop_directo.execution.agent.debt = {
        ontology.NeedId("n_a"): 0.1,
        ontology.NeedId("n_b"): 0.2,
    }
    loop_directo.execution.agent.floor_streak = {
        ontology.NeedId("n_a"): 1,
        ontology.NeedId("n_b"): 2,
    }
    loop_directo.execution.agent.saturation = {
        ontology.ActionId("act1"): 1,
        ontology.ActionId("act2"): 2,
    }
    loop_directo.execution.agent.traits = {
        ontology.TraitId("t_a"): 0.3,
        ontology.TraitId("t_b"): 0.7,
    }
    loop_directo.execution.agent.goals = {
        ontology.GoalId("g_a"): ontology.GoalStatus.ACTIVE,
        ontology.GoalId("g_b"): ontology.GoalStatus.ACHIEVED,
    }
    loop_directo.execution.agent.gain_beliefs = {
        ontology.ActionId("act1"): {
            ontology.NeedId("n_a"): 0.1,
            ontology.NeedId("n_b"): 0.2,
        },
        ontology.ActionId("act2"): {
            ontology.NeedId("n_a"): 0.3,
            ontology.NeedId("n_b"): 0.4,
        },
    }
    loop_directo.execution.agent.cost_beliefs = {
        ontology.ActionId("act1"): {
            ontology.NeedId("n_a"): 0.05,
            ontology.NeedId("n_b"): 0.06,
        },
        ontology.ActionId("act2"): {
            ontology.NeedId("n_a"): 0.07,
            ontology.NeedId("n_b"): 0.08,
        },
    }
    loop_directo.execution.situations = {
        ontology.SituationId("sit_a"): 10,
        ontology.SituationId("sit_b"): 20,
    }

    loop_invertido = loop.Loop.start(p, agent_spec, seed=1, inbox_capacity=5)
    loop_invertido.execution.agent.needs = {
        ontology.NeedId("n_b"): 0.7,
        ontology.NeedId("n_a"): 0.4,
    }
    loop_invertido.execution.agent.debt = {
        ontology.NeedId("n_b"): 0.2,
        ontology.NeedId("n_a"): 0.1,
    }
    loop_invertido.execution.agent.floor_streak = {
        ontology.NeedId("n_b"): 2,
        ontology.NeedId("n_a"): 1,
    }
    loop_invertido.execution.agent.saturation = {
        ontology.ActionId("act2"): 2,
        ontology.ActionId("act1"): 1,
    }
    loop_invertido.execution.agent.traits = {
        ontology.TraitId("t_b"): 0.7,
        ontology.TraitId("t_a"): 0.3,
    }
    loop_invertido.execution.agent.goals = {
        ontology.GoalId("g_b"): ontology.GoalStatus.ACHIEVED,
        ontology.GoalId("g_a"): ontology.GoalStatus.ACTIVE,
    }
    loop_invertido.execution.agent.gain_beliefs = {
        ontology.ActionId("act2"): {
            ontology.NeedId("n_b"): 0.4,
            ontology.NeedId("n_a"): 0.3,
        },
        ontology.ActionId("act1"): {
            ontology.NeedId("n_b"): 0.2,
            ontology.NeedId("n_a"): 0.1,
        },
    }
    loop_invertido.execution.agent.cost_beliefs = {
        ontology.ActionId("act2"): {
            ontology.NeedId("n_b"): 0.08,
            ontology.NeedId("n_a"): 0.07,
        },
        ontology.ActionId("act1"): {
            ontology.NeedId("n_b"): 0.06,
            ontology.NeedId("n_a"): 0.05,
        },
    }
    loop_invertido.execution.situations = {
        ontology.SituationId("sit_b"): 20,
        ontology.SituationId("sit_a"): 10,
    }

    path_directo = tmp_path / "directo.json"
    path_invertido = tmp_path / "invertido.json"
    persistence.save(loop_directo, path_directo)
    persistence.save(loop_invertido, path_invertido)

    assert path_directo.read_text(encoding="utf-8") == path_invertido.read_text(
        encoding="utf-8"
    )



def test_persistence_cross_process_determinism_across_hashseeds(tmp_path: Path) -> None:
    """§6.8: I-039 determinismo interproceso verificado entre diferentes PYTHONHASHSEED.

    Ejecuta un script independiente en subprocesos con PYTHONHASHSEED en '0', '1' y '42',
    corriendo 20 pasos de bucle y guardando a disco. Comprueba que el archivo producido
    es exactamente idéntico byte a byte en todas las ejecuciones.
    """
    script = """
import sys
from pathlib import Path
import aida.engine.loop as loop
import aida.engine.ontology as ontology
import aida.engine.pack as pack
import aida.engine.persistence as persistence

need1 = ontology.NeedSpec(
    id=ontology.NeedId("energia"),
    label="Energia",
    decay_per_tick=0.015,
    initial=0.8,
    cap=1.0,
)
need2 = ontology.NeedSpec(
    id=ontology.NeedId("animo"),
    label="Animo",
    decay_per_tick=0.02,
    initial=0.6,
    cap=1.0,
)
act1 = ontology.ActionSpec(
    id=ontology.ActionId("descansar"),
    label="Descansar",
    duration_ticks=2,
    reversibility=ontology.Reversibility.REVERSIBLE,
    satisfies={ontology.NeedId("energia"): 0.3},
    effects={ontology.NeedId("energia"): 0.3},
)
act2 = ontology.ActionSpec(
    id=ontology.ActionId("divertirse"),
    label="Divertirse",
    duration_ticks=1,
    reversibility=ontology.Reversibility.REVERSIBLE,
    satisfies={ontology.NeedId("animo"): 0.4},
    costs={ontology.NeedId("energia"): 0.1},
    effects={ontology.NeedId("animo"): 0.4, ontology.NeedId("energia"): -0.1},
)
trait = ontology.TraitSpec(
    id=ontology.TraitId("activo"),
    label="Activo",
    need_weight_multipliers={ontology.NeedId("energia"): 1.2},
)
spec = ontology.AgentSpec(
    id=ontology.AgentId("agente_det"),
    label="Agente Determinista",
    initial_needs={ontology.NeedId("energia"): 0.8, ontology.NeedId("animo"): 0.6},
    traits={ontology.TraitId("activo"): 0.5},
)
sit = ontology.SituationSpec(id=ontology.SituationId("noche"), label="Noche")
p = pack.ContentPack(
    id="pack_determ",
    label="Pack Determinista",
    needs=[need1, need2],
    traits=[trait],
    actions=[act1, act2],
    agents=[spec],
    situations=[sit],
    tuning=ontology.TuningSpec(candidate_pool=2, temperature=0.05),
)

assist = loop.Loop.start(p, spec, seed=12345, inbox_capacity=5)
assist.inbox.push(tick=5, owner=spec.id, content="alerta_temp")
assist.execution.situations[ontology.SituationId("noche")] = 10

for _ in range(20):
    assist.step()

out_path = Path(sys.argv[1])
persistence.save(assist, out_path)
sys.stdout.buffer.write(out_path.read_bytes())
"""

    outputs: list[bytes] = []
    seeds = ["0", "1", "42"]

    for seed in seeds:
        save_file = tmp_path / f"dump_{seed}.json"
        res = subprocess.run(
            [sys.executable, "-c", script, str(save_file)],
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.append(res.stdout)

    assert len(outputs) == 3
    diff_msg_01 = (
        f"Salida difiere entre PYTHONHASHSEED=0 y 1: {outputs[0][:100]!r} vs {outputs[1][:100]!r}"
    )
    diff_msg_12 = (
        f"Salida difiere entre PYTHONHASHSEED=1 y 42: {outputs[1][:100]!r} vs {outputs[2][:100]!r}"
    )
    assert outputs[0] == outputs[1], diff_msg_01
    assert outputs[1] == outputs[2], diff_msg_12


def test_persistence_authority_not_saved_and_requires_reproposal(tmp_path: Path) -> None:
    """La autoridad no se persiste: tras load, el asistente vuelve a proponer en vez de arrancar.

    Garantiza la propiedad de seguridad de §3.2: un permiso previo concedido en tiempo de
    ejecución anterior no sobrevive al reinicio de proceso. Al reanudar, authority.approved
    y authority.pending deben estar limpios, forzando una nueva propuesta y deliberación.
    """
    n_a = ontology.NeedSpec(
        id=ontology.NeedId("n_a"),
        label="Need A",
        initial=0.8,
        decay_per_tick=0.02,
        cap=1.0,
    )
    n_b = ontology.NeedSpec(
        id=ontology.NeedId("n_b"),
        label="Need B",
        initial=0.1,
        decay_per_tick=0.02,
        cap=1.0,
    )
    rutina = ontology.ActionSpec(
        id=ontology.ActionId("rutina"),
        label="Rutina",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={ontology.NeedId("n_a"): 0.2},
        costs={},
        effects={ontology.NeedId("n_a"): 0.2},
    )
    peligrosa = ontology.ActionSpec(
        id=ontology.ActionId("peligrosa"),
        label="Peligrosa",
        duration_ticks=1,
        reversibility=ontology.Reversibility.IRREVERSIBLE,
        satisfies={ontology.NeedId("n_b"): 0.9},
        costs={},
        effects={ontology.NeedId("n_b"): 0.9},
    )
    spec = ontology.AgentSpec(
        id=ontology.AgentId("agente_01"),
        label="Agente U8",
        initial_needs={ontology.NeedId("n_a"): 0.8, ontology.NeedId("n_b"): 0.1},
    )
    tuning = ontology.TuningSpec(
        candidate_pool=1,
        temperature=0.01,
    )
    authority = ontology.AuthoritySpec(
        autonomous_up_to=ontology.Reversibility.REVERSIBLE,
        confirmation_ttl_ticks=48,
    )
    p = pack.ContentPack(
        id="pack_persist_u8",
        label="Pack Persist U8",
        needs=[n_a, n_b],
        actions=[rutina, peligrosa],
        agents=[spec],
        tuning=tuning,
        authority=authority,
    )

    assist = loop.Loop.start(p, spec, seed=7, inbox_capacity=2)

    # Tick 1: propone 'peligrosa' y arranca 'rutina'
    assist.step()
    assert assist.execution.tick == 1
    assert ontology.ActionId("peligrosa") in assist.authority.pending

    # Autorizamos 'peligrosa' antes de guardar
    assert assist.approve(ontology.ActionId("peligrosa")) is True
    assert ontology.ActionId("peligrosa") in assist.authority.approved

    # Guardamos estado en archivo
    save_file = tmp_path / "saved_state.json"
    persistence.save(assist, save_file)

    # 1. El JSON guardado no menciona autoridad ni propuestas
    raw_content = save_file.read_text(encoding="utf-8")
    saved_data = json.loads(raw_content)
    assert "authority" not in saved_data
    assert "peligrosa" not in str(saved_data.get("authority", {}))
    assert "approved" not in raw_content

    # 2. Tras load, la autoridad está completamente vacía
    loaded = persistence.load(save_file, p)
    assert len(loaded.authority.pending) == 0
    assert len(loaded.authority.approved) == 0

    # 3. Un step() sobre el bucle cargado no arranca peligrosa sino que vuelve a proponerla
    loaded.step()
    assert loaded.execution.tick == 2
    assert loaded.execution.agent.performing is not None
    assert loaded.execution.agent.performing.action == ontology.ActionId("rutina")
    assert ontology.ActionId("peligrosa") in loaded.authority.pending

    reproposed = [
        ev
        for ev in loaded.log.of_kind(events.EventKind.ACTION_PROPOSED)
        if ev.tick == 2
    ]
    assert len(reproposed) == 1
    assert reproposed[0].action == ontology.ActionId("peligrosa")


def test_persistence_load_with_intent_preserves_classifier_and_classifies_events(
    tmp_path: Path,
) -> None:
    """I-057 / V1: la carga de bucle debe permitir inyectar el clasificador de intenciones
    para que la sesión reanudada conserve la capacidad de comprender lenguaje natural
    y clasificar peticiones del buzón de entrada tras una parada técnica.
    """
    from aida.intent.classify import build_classifier

    loaded_pack = pack.load_pack(_TEMPLATE_PACK)
    classifier = build_classifier(loaded_pack)
    assert classifier is not None

    agent = loaded_pack.agents[0]
    assist = loop.Loop.start(
        loaded_pack,
        agent,
        seed=7,
        inbox_capacity=10,
        intent=classifier,
    )
    assist.step()

    save_path = tmp_path / "persisted_with_intent.json"
    persistence.save(assist, save_path)

    restored_loop = persistence.load(save_path, loaded_pack, intent=classifier)
    assert restored_loop.intent is classifier

    next_tick = restored_loop.execution.tick + 1
    restored_loop.inbox.push(
        tick=next_tick,
        owner=agent.id,
        content="libera espacio en disco",
    )
    restored_loop.step()

    classified_events = [
        ev
        for ev in restored_loop.log.of_kind(events.EventKind.INTENT_CLASSIFIED)
        if ev.tick == restored_loop.execution.tick
    ]
    assert len(classified_events) == 1
    assert classified_events[0].data == {"intent": "pedir_liberar_espacio"}

