"""Pruebas exhaustivas de deliberación y selección de acciones (`aida.agents.decide`).

Cubre la unidad U4 de A.I.D.A. según la especificación de `.claude/workflow/07-plan-u4.md`:
1. Ponderación (`_weigh`, `benefit`, `cost`):
   - Uso canónico de `urgency` desde `disposition.py` y techos vía `agent.cap_of(need)`.
   - Creencias ausentes por defecto en 1.0 (`beliefs.get(need_id, 1.0)`).
   - Suma ordenada determinista con `sorted(terms)`.
2. Ancla de mutación 1: La urgencia pondera el coste.
   - Inversión de preferencia cuando un eje de coste desciende (gastar duele más).
   - Se vuelve rojo si el coste omite la ponderación por urgencia.
3. Ancla de mutación 2: Los moduladores multiplican solo el beneficio y nunca el neto.
   - Verificado con una opción de utilidad neta negativa.
   - Se vuelve rojo si los moduladores multiplican el neto.
4. Ancla de mutación 3: Suelo de novedad (`novelty`).
   - Con saturación 1.0, la novedad no es cero si peso < 1.0 (`1.0 - saturation * weight`).
   - Altera planes sin vetar acciones permanentemente.
5. Moduladores de franja (`phase_fit`) y situación (`situation_fit`):
   - Multiplicación determinista en orden `sorted(situations)`.
6. Ancla de mutación 6: Puertas duras (`is_available`).
   - Devuelve el nombre exacto de la puerta (`"requires"`, `"requires_traits"`) o `None`.
   - No filtra opciones de utilidad cero.
7. Ancla de mutación 4: Desempate determinista y estable (`_rank`).
   - El ordenamiento por `(-utility, str(action_id))` garantiza invariancia ante el orden del pack.
8. Puntuación y candidatos (`candidates`).
9. Ancla de mutación 10: Guarda de temperamento en `decide`.
   - Lanza error legible propio si `agent.temperament is None` (agente sin componer).
10. Ancla de mutación 7: Interrupción con `must_address`.
    - Solo considera acciones que atienden el eje crítico.
    - Si ninguna lo atiende, devuelve `None` sin degradar a elección libre.
11. Ancla de mutación 8: Ejes críticos (`critical_needs`).
    - Filtra los ejes con `urgent_when_low = False`.
    - Ordena por urgencia descendente con desempate por identificador.
12. Ancla de mutación 5: Softmax reproducible y derivación de RNG.
    - La clave del stream incluye `("decide", agent.agent_id, tick)`.
    - Determinismo idéntico entre procesos con distinto `PYTHONHASHSEED`.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import aida.agents.decide as decide
import aida.engine.ontology as ontology
import aida.engine.pack as pack
import aida.engine.rng as rng
import aida.engine.state as state


def _make_need(
    id_: str = "need_a",
    *,
    decay_per_tick: float = 0.01,
    initial: float = 0.5,
    cap: float = 1.0,
    exponent: float = 1.0,
    urgent_when_low: bool = True,
    label: str = "Test need",
) -> ontology.NeedSpec:
    """Construye un NeedSpec de prueba sin vocabulario de dominio."""
    return ontology.NeedSpec(
        id=ontology.NeedId(id_),
        label=label,
        decay_per_tick=decay_per_tick,
        initial=initial,
        cap=cap,
        urgency_exponent=exponent,
        urgent_when_low=urgent_when_low,
    )


def _make_action(
    id_: str = "action_1",
    *,
    label: str = "Test action",
    duration_ticks: int = 1,
    reversibility: str = "reversible",
    satisfies: dict[str, float] | None = None,
    costs: dict[str, float] | None = None,
    effects: dict[str, float] | None = None,
    requires: dict[str, float] | None = None,
    requires_traits: dict[str, float] | None = None,
    phase_multipliers: dict[str, float] | None = None,
    situation_multipliers: dict[str, float] | None = None,
) -> ontology.ActionSpec:
    """Construye un ActionSpec válido asegurando coherencia de signos."""
    sat = {ontology.NeedId(k): v for k, v in (satisfies or {}).items()}
    cst = {ontology.NeedId(k): v for k, v in (costs or {}).items()}
    eff = {ontology.NeedId(k): v for k, v in (effects or {}).items()}
    if not eff:
        eff = {**sat, **{k: -v for k, v in cst.items()}}
    req = {ontology.NeedId(k): v for k, v in (requires or {}).items()}
    req_tr = {ontology.TraitId(k): v for k, v in (requires_traits or {}).items()}
    pm = {ontology.PhaseId(k): v for k, v in (phase_multipliers or {}).items()}
    sm = {ontology.SituationId(k): v for k, v in (situation_multipliers or {}).items()}
    return ontology.ActionSpec(
        id=ontology.ActionId(id_),
        label=label,
        duration_ticks=duration_ticks,
        reversibility=ontology.Reversibility(reversibility),
        satisfies=sat,
        costs=cst,
        effects=eff,
        requires=req,
        requires_traits=req_tr,
        phase_multipliers=pm,
        situation_multipliers=sm,
    )


def _make_pack(
    *,
    pack_id: str = "test_pack",
    needs: list[ontology.NeedSpec] | None = None,
    actions: list[ontology.ActionSpec] | None = None,
    traits: list[ontology.TraitSpec] | None = None,
    phases: list[ontology.PhaseSpec] | None = None,
    situations: list[ontology.SituationSpec] | None = None,
    tuning: ontology.TuningSpec | None = None,
    candidate_pool: int = 3,
    temperature: float = 0.15,
) -> pack.ContentPack:
    """Construye un ContentPack válido para pruebas de deliberación."""
    ns = needs or [_make_need("need_a")]
    first_need_id = str(ns[0].id)
    acts = (
        actions
        if actions is not None
        else [_make_action("action_1", satisfies={first_need_id: 0.5})]
    )
    trs = traits or []

    # Fases referenciadas en acciones
    needed_phases = {p for a in acts for p in a.phase_multipliers}
    ph_list = list(phases) if phases is not None else [
        ontology.PhaseSpec(id=pid, label=str(pid), starts_at=idx * 10)
        for idx, pid in enumerate(sorted(needed_phases))
    ]

    # Situaciones referenciadas en acciones
    needed_situations = {s for a in acts for s in a.situation_multipliers}
    sit_list = list(situations) if situations is not None else [
        ontology.SituationSpec(id=sid, label=str(sid)) for sid in sorted(needed_situations)
    ]

    # Requisitos de acciones para que el pack sea válido (no catálogo muerto)
    initial_needs: dict[ontology.NeedId, float] = {}
    for n in ns:
        req_max = max([a.requires.get(n.id, 0.0) for a in acts] + [0.0])
        initial_needs[n.id] = min(n.cap, max(n.initial, req_max))

    agent_traits: dict[ontology.TraitId, float] = {}
    for t in trs:
        req_tr_max = max([a.requires_traits.get(t.id, 0.0) for a in acts] + [0.0])
        agent_traits[t.id] = min(1.0, max(0.5, req_tr_max))

    tun = tuning or ontology.TuningSpec(candidate_pool=candidate_pool, temperature=temperature)
    agent_spec = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Test Agent",
        initial_needs=initial_needs,
        traits=agent_traits,
    )
    return pack.ContentPack(
        id=pack_id,
        label="Pack de prueba deliberacion",
        needs=ns,
        traits=trs,
        phases=ph_list,
        situations=sit_list,
        actions=acts,
        agents=[agent_spec],
        tuning=tun,
    )


def _make_execution(
    pack_: pack.ContentPack,
    *,
    agent_id: str = "agent_01",
    needs: dict[str, float] | None = None,
    traits: dict[str, float] | None = None,
    tick: int = 0,
    seed: int = 42,
    compose: bool = True,
    gain_beliefs: dict[str, dict[str, float]] | None = None,
    cost_beliefs: dict[str, dict[str, float]] | None = None,
    saturation: dict[str, float] | None = None,
) -> state.ExecutionState:
    """Construye un ExecutionState con AgentState inicializado y compuesto."""
    ag_id = ontology.AgentId(agent_id)
    initial_needs = {
        n.id: (needs.get(str(n.id), n.initial) if needs else n.initial) for n in pack_.needs
    }
    ag_traits = {ontology.TraitId(k): v for k, v in (traits or {}).items()}
    sat = {ontology.ActionId(k): v for k, v in (saturation or {}).items()}
    gb = {
        ontology.ActionId(act): {ontology.NeedId(n): val for n, val in n_map.items()}
        for act, n_map in (gain_beliefs or {}).items()
    }
    cb = {
        ontology.ActionId(act): {ontology.NeedId(n): val for n, val in n_map.items()}
        for act, n_map in (cost_beliefs or {}).items()
    }
    agent_st = state.AgentState(
        agent_id=ag_id,
        needs=initial_needs,
        traits=ag_traits,
        saturation=sat,
        gain_beliefs=gb,
        cost_beliefs=cb,
    )
    if compose:
        agent_st.recompose(pack_)
    return state.ExecutionState(
        agent=agent_st,
        tick=tick,
        rng=rng.RngSource(seed),
    )


# =============================================================================
# 1. Ponderación canónica: _weigh, benefit, cost
# =============================================================================


def test_weigh_usa_urgency_canonica_y_cap_of() -> None:
    """`_weigh` debe calcular la presión con `urgency` canónica y `agent.cap_of(need)`.

    Si se usara `need.cap` en vez de `agent.cap_of(need)`, un techo modificado
    por rasgos/deuda no tendría efecto sobre la presión de urgencia.
    """
    need = _make_need("need_a", cap=1.0, exponent=2.0)
    p = _make_pack(needs=[need])
    exec_st = _make_execution(p, needs={"need_a": 0.5})

    # Forzar un cap diferente en el agente
    exec_st.agent.caps[need.id] = 0.8

    # Urgencia con cap 0.8: urgency(0.5, 2.0, cap=0.8) = ((0.8 - 0.5)/0.8)^2 = (0.375)^2 = 0.140625
    expected_urgency = ((0.8 - 0.5) / 0.8) ** 2
    weight = exec_st.agent.need_weights[need.id]

    terms = {need.id: 2.0}
    beliefs: dict[ontology.NeedId, float] = {}

    result = decide._weigh(p, exec_st.agent, terms, beliefs)
    expected = weight * expected_urgency * 2.0 * 1.0
    assert result == pytest.approx(expected, rel=1e-6)

    # Si se hubiera usado need.cap (1.0), daría (0.5)^2 = 0.25
    wrong = weight * (0.5**2) * 2.0 * 1.0
    assert result != pytest.approx(wrong, rel=1e-3)


def test_weigh_creencia_ausente_vale_uno() -> None:
    """Una necesidad no registrada en creencias aprendidas debe valorarse como 1.0."""
    need = _make_need("need_a", exponent=1.0, cap=1.0)
    p = _make_pack(needs=[need])
    exec_st = _make_execution(p, needs={"need_a": 0.2})

    terms = {need.id: 1.0}
    # Creencia vacía: debe usar 1.0
    val_default = decide._weigh(p, exec_st.agent, terms, {})

    # Creencia explícita en 1.5: debe escalar proporcionalmente
    val_with_belief = decide._weigh(p, exec_st.agent, terms, {need.id: 1.5})
    assert val_with_belief == pytest.approx(val_default * 1.5)


def test_weigh_suma_ordenada_por_identificador() -> None:
    """La suma en coma flotante no es asociativa; `_weigh` debe iterar `sorted(terms)`."""
    needs = [
        _make_need("need_z", initial=0.5),
        _make_need("need_a", initial=0.5),
        _make_need("need_m", initial=0.5),
    ]
    p = _make_pack(needs=needs)
    exec_st = _make_execution(p)

    terms = {
        ontology.NeedId("need_z"): 1e-16,
        ontology.NeedId("need_a"): 1.0,
        ontology.NeedId("need_m"): -1.0,
    }
    res1 = decide._weigh(p, exec_st.agent, terms, {})
    # Invertir el orden del diccionario no debe cambiar ni un bit del resultado float
    terms_inverted = {
        ontology.NeedId("need_a"): 1.0,
        ontology.NeedId("need_m"): -1.0,
        ontology.NeedId("need_z"): 1e-16,
    }
    res2 = decide._weigh(p, exec_st.agent, terms_inverted, {})
    assert res1.hex() == res2.hex()


def test_benefit_y_cost_leen_creencias_indexadas_por_accion() -> None:
    """`benefit` y `cost` deben pasar las creencias indexadas por `action.id`."""
    need_a = _make_need("need_a", initial=0.5, cap=1.0, exponent=1.0)
    need_b = _make_need("need_b", initial=0.5, cap=1.0, exponent=1.0)
    action_1 = _make_action("act_1", satisfies={"need_a": 0.4}, costs={"need_b": 0.1})
    action_2 = _make_action("act_2", satisfies={"need_a": 0.4}, costs={"need_b": 0.1})
    p = _make_pack(needs=[need_a, need_b], actions=[action_1, action_2])

    # Creencias solo para act_1
    exec_st = _make_execution(
        p,
        gain_beliefs={"act_1": {"need_a": 2.0}},
        cost_beliefs={"act_1": {"need_b": 0.5}},
    )

    b1 = decide.benefit(p, exec_st.agent, action_1)
    b2 = decide.benefit(p, exec_st.agent, action_2)
    # act_1 tiene creencia 2.0, act_2 tiene creencia por defecto 1.0
    assert b1 == pytest.approx(b2 * 2.0)

    c1 = decide.cost(p, exec_st.agent, action_1)
    c2 = decide.cost(p, exec_st.agent, action_2)
    # act_1 tiene creencia 0.5, act_2 tiene creencia por defecto 1.0
    assert c1 == pytest.approx(c2 * 0.5)


# =============================================================================
# 2. Ancla de mutación 1: La urgencia pondera también el coste
# =============================================================================


def test_urgencia_pondera_el_coste_y_el_orden_de_preferencia_se_invierte() -> None:
    """Ancla contra mutación: la urgencia debe ponderar el coste tanto como el beneficio.

    Escenario de prueba:
    - Eje de beneficio `need_m` (urgencia alta y constante = 1.0, value = 0.0).
    - Eje de recurso/coste `need_n` (cap = 1.0, exponent = 1.0).
    - Acción A: alto beneficio (1.0 en M), alto coste (0.8 en N).
    - Acción B: beneficio moderado (0.6 en M), coste bajo (0.2 en N).

    Cuando N es abundante (valor 0.8 -> urgencia = 0.2):
      Coste A = 0.2 * 0.8 = 0.16. Utilidad neta A = 1.0 - 0.16 = 0.84.
      Coste B = 0.2 * 0.2 = 0.04. Utilidad neta B = 0.6 - 0.04 = 0.56.
      Acción A GANA (0.84 > 0.56).

    Cuando N es escaso (valor 0.1 -> urgencia = 0.9):
      Coste A = 0.9 * 0.8 = 0.72. Utilidad neta A = 1.0 - 0.72 = 0.28.
      Coste B = 0.9 * 0.2 = 0.18. Utilidad neta B = 0.6 - 0.18 = 0.42.
      Acción B GANA (0.42 > 0.28). El orden de preferencia SE INVIERTE.

    Si el coste no estuviera ponderado por urgencia (mutante: urgencia fija a 1.0):
      Coste A sería 0.8 constante -> Utilidad A = 0.2.
      Coste B sería 0.2 constante -> Utilidad B = 0.4.
      Acción B ganaría SIEMPRE, incluso cuando N es abundante. El test se pone rojo.
    """
    need_m = _make_need("need_m", cap=1.0, exponent=1.0, initial=0.0)
    need_n = _make_need("need_n", cap=1.0, exponent=1.0, initial=0.8)
    act_a = _make_action("action_a", satisfies={"need_m": 1.0}, costs={"need_n": 0.8})
    act_b = _make_action("action_b", satisfies={"need_m": 0.6}, costs={"need_n": 0.2})
    p = _make_pack(needs=[need_m, need_n], actions=[act_a, act_b], candidate_pool=1)

    # Caso 1: N abundante (0.8). Acción A debe ganar
    exec_abundant = _make_execution(p, needs={"need_m": 0.0, "need_n": 0.8})
    choice_abundant = decide.decide(exec_abundant, p)
    assert choice_abundant is not None
    assert choice_abundant.action == ontology.ActionId("action_a")
    assert choice_abundant.utility == pytest.approx(0.84, abs=1e-4)

    # Caso 2: N escaso (0.1). Acción B debe ganar
    exec_scarce = _make_execution(p, needs={"need_m": 0.0, "need_n": 0.1})
    choice_scarce = decide.decide(exec_scarce, p)
    assert choice_scarce is not None
    assert choice_scarce.action == ontology.ActionId("action_b")
    assert choice_scarce.utility == pytest.approx(0.42, abs=1e-4)


# =============================================================================
# 3. Ancla de mutación 2: Los moduladores multiplican solo beneficio
# =============================================================================


def test_moduladores_multiplican_solo_beneficio_no_el_neto() -> None:
    """Ancla contra mutación: los moduladores (fase, situación, novedad) nunca multiplican el neto.

    Si una acción tiene utilidad neta negativa (beneficio 1.0, coste 2.0, aversión 1.0):
      Fórmula correcta: beneficio * modulador - coste = 1.0 * 0.5 - 2.0 = -1.5.
      Fórmula mutante: (beneficio - coste) * modulador = (1.0 - 2.0) * 0.5 = -0.5.
    Una opción mala no debe parecer menos mala (-0.5 en vez de -1.5) por estar
    en la franja equivocada.
    """
    need_a = _make_need("need_a", cap=1.0, exponent=1.0, initial=0.0)
    need_b = _make_need("need_b", cap=1.0, exponent=1.0, initial=0.0)
    act = _make_action(
        "act_neg",
        satisfies={"need_a": 1.0},
        costs={"need_b": 2.0},
        phase_multipliers={"phase_night": 0.5},
    )
    p = _make_pack(needs=[need_a, need_b], actions=[act])
    exec_st = _make_execution(p, needs={"need_a": 0.0, "need_b": 0.0})

    cands, _ = decide.candidates(exec_st, p, phase=ontology.PhaseId("phase_night"))
    assert len(cands) == 1
    d = cands[0]

    # Beneficio es 1.0, Coste es 2.0, Phase fit es 0.5
    assert d.benefit == pytest.approx(1.0)
    assert d.cost == pytest.approx(2.0)
    assert d.phase_fit == pytest.approx(0.5)

    # Utilidad esperada: 1.0 * 0.5 - 2.0 = -1.5
    assert d.utility == pytest.approx(-1.5)
    # Mutación que multiplicaría el neto: (1.0 - 2.0) * 0.5 = -0.5
    assert d.utility != pytest.approx(-0.5)


# =============================================================================
# 4. Ancla de mutación 3: Suelo de novedad (Novelty floor)
# =============================================================================


def test_novelty_floor_accion_saturada_al_tope_sigue_siendo_elegible() -> None:
    """Ancla contra mutación: acción saturada al máximo (1.0) mantiene novedad > 0 con weight < 1.0.

    El hastío debe alterar planes sin vetar acciones permanentemente.
    Con saturación 1.0 y peso 0.2: novelty = 1.0 - 1.0 * 0.2 = 0.8 > 0.
    """
    need = _make_need("need_a", initial=0.0)
    act = _make_action("act_sat", satisfies={"need_a": 1.0})
    p = _make_pack(needs=[need], actions=[act])

    # Agente con saturación máxima (1.0) en la única acción existente
    exec_st = _make_execution(p, saturation={"act_sat": 1.0})
    assert exec_st.agent.temperament is not None
    weight = exec_st.agent.temperament.novelty_weight
    assert 0.0 < weight < 1.0

    nov = decide.novelty(exec_st.agent, act.id, weight)
    assert nov == pytest.approx(1.0 - weight)
    assert nov > 0.0

    # No debe ser descartada por ninguna puerta dura
    assert decide.is_available(exec_st.agent, act) is None

    # decide debe devolverla porque es elegible y única
    chosen = decide.decide(exec_st, p)
    assert chosen is not None
    assert chosen.action == act.id
    assert chosen.novelty == pytest.approx(1.0 - weight)


def test_novelty_cambia_el_plan_sin_vetar_la_accion_permanentemente() -> None:
    """La saturación progresiva hace perder preferencia ante otra opción pero sigue en carrera."""
    need = _make_need("need_a", initial=0.0)
    act_a = _make_action("act_a", satisfies={"need_a": 1.0})
    act_b = _make_action("act_b", satisfies={"need_a": 0.85})
    p = _make_pack(needs=[need], actions=[act_a, act_b], candidate_pool=1)

    # Sin saturación, act_a (1.0) gana sobre act_b (0.85)
    exec_fresh = _make_execution(p)
    d_fresh = decide.decide(exec_fresh, p)
    assert d_fresh is not None
    assert d_fresh.action == act_a.id

    # Con saturación en act_a tal que 1.0 * (1 - sat * weight) < 0.85, act_b gana
    # Si weight = 0.2 y sat = 1.0: act_a vale 1.0 * 0.8 = 0.80 < 0.85 (act_b)
    exec_saturated = _make_execution(p, saturation={"act_a": 1.0})
    d_sat = decide.decide(exec_saturated, p)
    assert d_sat is not None
    assert d_sat.action == act_b.id

    # Sin embargo, act_a sigue en la lista de candidatos evaluados con utilidad positiva
    cands, _ = decide.candidates(exec_saturated, p)
    cands_by_id = {c.action: c for c in cands}
    assert act_a.id in cands_by_id
    assert cands_by_id[act_a.id].novelty > 0.0


# =============================================================================
# 5. Moduladores de franja y situación
# =============================================================================


def test_phase_fit_comportamiento_con_y_sin_fase() -> None:
    """`phase_fit` devuelve 1.0 si phase es None o no está en la acción, o el factor registrado."""
    act = _make_action("act_1", phase_multipliers={"phase_work": 1.5})

    assert decide.phase_fit(act, None) == 1.0
    assert decide.phase_fit(act, ontology.PhaseId("phase_other")) == 1.0
    assert decide.phase_fit(act, ontology.PhaseId("phase_work")) == 1.5


def test_situation_fit_multiplica_en_orden_determinista() -> None:
    """`situation_fit` multiplica los factores de situaciones ordenadas deterministamente."""
    act = _make_action(
        "act_1",
        situation_multipliers={
            "sit_b": 0.5,
            "sit_a": 1.2,
        },
    )
    # Sin situaciones
    assert decide.situation_fit(act, ()) == 1.0

    # Con situaciones
    situations = [ontology.SituationId("sit_b"), ontology.SituationId("sit_a")]
    fit = decide.situation_fit(act, situations)
    assert fit == pytest.approx(1.2 * 0.5)


# =============================================================================
# 6. Ancla de mutación 6: Puertas duras (is_available)
# =============================================================================


def test_is_available_devuelve_exactamente_requires_o_requires_traits_o_none() -> None:
    """`is_available` devuelve el nombre exacto de la puerta que descarta o None si es elegible.

    Nunca debe devolver un booleano (True/False): la traza de decisión requiere
    saber exactamente cuál puerta descartó la acción para diagnóstico.
    """
    need = _make_need("need_a")
    trait = ontology.TraitSpec(id=ontology.TraitId("trait_a"), label="Trait A")
    act = _make_action(
        "act_locked",
        requires={"need_a": 0.8},
        requires_traits={"trait_a": 0.6},
    )
    p = _make_pack(needs=[need], traits=[trait], actions=[act])

    # 1. Falla need: debe devolver 'requires'
    exec_fail_need = _make_execution(p, needs={"need_a": 0.5}, traits={"trait_a": 0.8})
    door_need = decide.is_available(exec_fail_need.agent, act)
    assert door_need == "requires"

    # 2. Cumple need pero falla trait: debe devolver 'requires_traits'
    exec_fail_trait = _make_execution(p, needs={"need_a": 0.9}, traits={"trait_a": 0.3})
    door_trait = decide.is_available(exec_fail_trait.agent, act)
    assert door_trait == "requires_traits"

    # 3. Cumple ambos: debe devolver None
    exec_ok = _make_execution(p, needs={"need_a": 0.9}, traits={"trait_a": 0.7})
    assert decide.is_available(exec_ok.agent, act) is None


def test_is_available_prioridad_y_orden_sorted() -> None:
    """Si fallan requires y requires_traits simultáneamente, 'requires' tiene prioridad."""
    need = _make_need("need_a")
    trait = ontology.TraitSpec(id=ontology.TraitId("trait_a"), label="Trait A")
    act = _make_action(
        "act_both_fail",
        requires={"need_a": 0.9},
        requires_traits={"trait_a": 0.9},
    )
    p = _make_pack(needs=[need], traits=[trait], actions=[act])
    exec_st = _make_execution(p, needs={"need_a": 0.1}, traits={"trait_a": 0.1})

    assert decide.is_available(exec_st.agent, act) == "requires"


def test_is_available_no_filtra_utilidad_cero() -> None:
    """Una acción que cumple requisitos pero tiene beneficio y coste cero no se descarta."""
    need = _make_need("need_a", initial=1.0)
    act_idle = _make_action("act_idle")
    p = _make_pack(needs=[need], actions=[act_idle])
    exec_st = _make_execution(p)

    assert decide.is_available(exec_st.agent, act_idle) is None
    cands, rejected = decide.candidates(exec_st, p)
    assert len(cands) == 1
    assert cands[0].utility == 0.0
    assert not rejected


# =============================================================================
# 7. Ancla de mutación 4: Desempate estable por identificador (_rank)
# =============================================================================


def test_desempate_estable_por_identificador_invariante_al_orden_del_pack() -> None:
    """Ancla contra mutación: ante opciones de igual utilidad, el desempate por id es estricto.

    Reordenar las acciones en el pack NO debe cambiar la elección bajo la misma semilla.
    Si se eliminara el desempate por identificador (mutante: ordenar solo por -utility),
    la estabilidad de sort() de Python preservaría el orden de inserción y daría ganador distinto.
    """
    need = _make_need("need_a", initial=0.0)
    # Dos acciones con parámetros idénticos: misma utilidad exacta
    act_alpha = _make_action("action_alpha", satisfies={"need_a": 0.5})
    act_omega = _make_action("action_omega", satisfies={"need_a": 0.5})

    # Pack 1 con orden [omega, alpha]
    pack_1 = _make_pack(
        needs=[need],
        actions=[act_omega, act_alpha],
        candidate_pool=1,
    )
    exec_1 = _make_execution(pack_1, seed=123)
    choice_1 = decide.decide(exec_1, pack_1)

    # Pack 2 con orden [alpha, omega]
    pack_2 = _make_pack(
        needs=[need],
        actions=[act_alpha, act_omega],
        candidate_pool=1,
    )
    exec_2 = _make_execution(pack_2, seed=123)
    choice_2 = decide.decide(exec_2, pack_2)

    assert choice_1 is not None
    assert choice_2 is not None
    # En ambos casos DEBE ganar 'action_alpha' (lexicográficamente menor)
    assert choice_1.action == ontology.ActionId("action_alpha")
    assert choice_2.action == ontology.ActionId("action_alpha")


def test_rank_desempata_lexicograficamente_por_accion_con_utilidad_empatada() -> None:
    """Ancla contra mutación: _rank desempata por ActionId cuando las utilidades empatan.

    Verifica que al ordenar una lista donde la acción lexicográficamente mayor
    aparece primero ([d_omega, d_alpha]), el resultado sea estrictamente
    [d_alpha, d_omega]. Si _rank omitiera el desempate por identificador (mutante:
    retornar solo -decision.utility), la estabilidad de sort() preservaría
    [d_omega, d_alpha] y esta prueba fallaría inmediatamente.
    """
    d_omega = decide.Decision(
        action=ontology.ActionId("action_omega"),
        utility=0.5,
        benefit=0.5,
        cost=0.0,
        novelty=1.0,
        phase_fit=1.0,
        situation_fit=1.0,
    )
    d_alpha = decide.Decision(
        action=ontology.ActionId("action_alpha"),
        utility=0.5,
        benefit=0.5,
        cost=0.0,
        novelty=1.0,
        phase_fit=1.0,
        situation_fit=1.0,
    )
    d_beta = decide.Decision(
        action=ontology.ActionId("action_beta"),
        utility=0.5,
        benefit=0.5,
        cost=0.0,
        novelty=1.0,
        phase_fit=1.0,
        situation_fit=1.0,
    )
    d_zeta = decide.Decision(
        action=ontology.ActionId("action_zeta"),
        utility=0.5,
        benefit=0.5,
        cost=0.0,
        novelty=1.0,
        phase_fit=1.0,
        situation_fit=1.0,
    )

    # Orden deliberadamente inverso: mayor lexicográfico primero
    ranked = sorted([d_omega, d_alpha], key=decide._rank)
    assert ranked[0].action == ontology.ActionId("action_alpha")
    assert ranked[1].action == ontology.ActionId("action_omega")

    # Caso múltiple desordenado
    ranked_multi = sorted([d_zeta, d_omega, d_beta, d_alpha], key=decide._rank)
    assert [d.action for d in ranked_multi] == [
        ontology.ActionId("action_alpha"),
        ontology.ActionId("action_beta"),
        ontology.ActionId("action_omega"),
        ontology.ActionId("action_zeta"),
    ]


def test_rank_retorna_tupla_con_utilidad_negativa_e_identificador_de_accion() -> None:
    """Verifica directamente la tupla devuelta por _rank.

    Debe retornar (-decision.utility, str(decision.action)).
    """
    dec = decide.Decision(
        action=ontology.ActionId("action_test"),
        utility=0.75,
        benefit=1.0,
        cost=0.25,
        novelty=1.0,
        phase_fit=1.0,
        situation_fit=1.0,
    )
    key = decide._rank(dec)
    assert key == (-0.75, "action_test")
    assert isinstance(key, tuple)
    assert len(key) == 2
    assert key[0] == -dec.utility
    assert key[1] == str(dec.action)


# =============================================================================
# 8. Ancla de mutación 10: Guarda de temperamento compuesto
# =============================================================================


def test_decide_sin_temperamento_compuesto_lanza_error_propio_y_legible() -> None:
    """Ancla contra mutación: invocar `decide` con `agent.temperament is None` falla limpiamente.

    No debe levantar un `AttributeError` críptico al intentar acceder a `.temperature`.
    Debe lanzar una excepción con un mensaje claro que indique que el agente no está recompuesto.
    """
    need = _make_need("need_a")
    act = _make_action("act_1", satisfies={"need_a": 0.5})
    p = _make_pack(needs=[need], actions=[act])

    # Construir sin recompose (temperament = None)
    exec_st = _make_execution(p, compose=False)
    assert exec_st.agent.temperament is None

    with pytest.raises((RuntimeError, ValueError)) as excinfo:
        decide.decide(exec_st, p)

    # Verificar que el mensaje menciona recomposición o temperamento no compuesto
    msg = str(excinfo.value).lower()
    assert (
        "temperament" in msg
        or "recompose" in msg
        or "componer" in msg
        or "compuesto" in msg
        or "composed" in msg
    )


# =============================================================================
# 9. Ancla de mutación 7: Interrupción con must_address
# =============================================================================


def test_must_address_filtra_solo_acciones_que_atienden_el_eje() -> None:
    """Si se pasa `must_address`, solo se consideran acciones con satisfies[must_address] > 0."""
    need_a = _make_need("need_a", initial=0.0)
    need_b = _make_need("need_b", initial=0.0)
    act_a = _make_action("act_a", satisfies={"need_a": 1.0})
    act_b = _make_action("act_b", satisfies={"need_b": 1.0})
    p = _make_pack(needs=[need_a, need_b], actions=[act_a, act_b], candidate_pool=1)
    exec_st = _make_execution(p)

    choice = decide.decide(exec_st, p, must_address=ontology.NeedId("need_b"))
    assert choice is not None
    assert choice.action == act_b.id


def test_must_address_sin_acciones_disponibles_devuelve_none_sin_caer_a_eleccion_libre() -> None:
    """Ancla contra mutación: si ninguna acción atiende el eje crítico, se devuelve None.

    NO debe degradar a elección libre. En Agora, caer a elección libre destruía planes
    haciendo que el agente abandonara tareas tick a tick para elegir lo mismo.
    """
    need_a = _make_need("need_a", initial=0.0)
    need_c = _make_need("need_c", initial=0.0)  # Eje sin acciones que lo atiendan
    act_a = _make_action("act_a", satisfies={"need_a": 1.0})
    p = _make_pack(needs=[need_a, need_c], actions=[act_a])
    exec_st = _make_execution(p)

    choice = decide.decide(exec_st, p, must_address=ontology.NeedId("need_c"))
    # Debe ser None estricto; si cayera a elección libre devolvería act_a
    assert choice is None


# =============================================================================
# 10. Ancla de mutación 8: critical_needs
# =============================================================================


def test_critical_needs_filtra_urgent_when_low_false() -> None:
    """`critical_needs` descarta necesidades con `urgent_when_low = False`.

    Un eje acumulativo o de progresión que está en cero no es una emergencia.
    """
    need_progression = _make_need(
        "need_progress",
        initial=0.05,
        urgent_when_low=False,
    )
    need_vulnerable = _make_need(
        "need_danger",
        initial=0.05,
        urgent_when_low=True,
    )
    p = _make_pack(needs=[need_progression, need_vulnerable])
    exec_st = _make_execution(p, needs={"need_progress": 0.05, "need_danger": 0.05})

    crit = decide.critical_needs(exec_st, p)
    assert ontology.NeedId("need_progress") not in crit
    assert ontology.NeedId("need_danger") in crit


def test_critical_needs_ordena_por_urgencia_descendente_y_desempata_por_id() -> None:
    """`critical_needs` ordena de mayor urgencia a menor, y desempata por ID lexicográfico."""
    need_1 = _make_need("need_z_severe", initial=0.05, cap=1.0, exponent=1.0)  # urgencia 0.95
    need_2 = _make_need("need_a_moderate", initial=0.15, cap=1.0, exponent=1.0)  # urgencia 0.85
    need_tie_b = _make_need("need_tie_b", initial=0.10, cap=1.0, exponent=1.0)  # urgencia 0.90
    need_tie_a = _make_need("need_tie_a", initial=0.10, cap=1.0, exponent=1.0)  # urgencia 0.90

    p = _make_pack(
        needs=[need_1, need_2, need_tie_b, need_tie_a],
        tuning=ontology.TuningSpec(critical_need_threshold=0.3),
    )
    exec_st = _make_execution(
        p,
        needs={
            "need_z_severe": 0.05,
            "need_a_moderate": 0.15,
            "need_tie_b": 0.10,
            "need_tie_a": 0.10,
        },
    )

    crit = decide.critical_needs(exec_st, p)
    # Orden esperado:
    # 1. need_z_severe (0.95)
    # 2. need_tie_a (0.90, desempate por id antes que need_tie_b)
    # 3. need_tie_b (0.90)
    # 4. need_a_moderate (0.85)
    assert crit == [
        ontology.NeedId("need_z_severe"),
        ontology.NeedId("need_tie_a"),
        ontology.NeedId("need_tie_b"),
        ontology.NeedId("need_a_moderate"),
    ]


def test_critical_needs_ignora_ejes_por_encima_del_umbral() -> None:
    """Necesidades por encima del critical_need_threshold no entran en critical_needs."""
    need = _make_need("need_safe", initial=0.8, cap=1.0, exponent=1.0)
    p = _make_pack(needs=[need])
    exec_st = _make_execution(p, needs={"need_safe": 0.8})

    assert decide.critical_needs(exec_st, p) == []


# =============================================================================
# 11. Ancla de mutación 5: Softmax reproducible y RNG
# =============================================================================


def test_softmax_reproducible_mismo_tick_y_semilla_da_misma_eleccion() -> None:
    """La elección de softmax con misma semilla, agente y tick es idéntica bit a bit."""
    need = _make_need("need_a", initial=0.0)
    actions = [
        _make_action(f"act_{i}", satisfies={"need_a": 0.5 + i * 0.01})
        for i in range(5)
    ]
    p = _make_pack(needs=[need], actions=actions, candidate_pool=5, temperature=0.5)

    exec_1 = _make_execution(p, seed=999, tick=10)
    exec_2 = _make_execution(p, seed=999, tick=10)

    res_1 = decide.decide(exec_1, p)
    res_2 = decide.decide(exec_2, p)

    assert res_1 is not None and res_2 is not None
    assert res_1.action == res_2.action
    assert res_1.utility == res_2.utility


def test_softmax_clave_incluye_tick_y_agente() -> None:
    """Diferente tick o agent_id deriva un stream RNG distinto."""
    need = _make_need("need_a", initial=0.0)
    actions = [
        _make_action(f"act_{i}", satisfies={"need_a": 0.5})
        for i in range(5)
    ]
    p = _make_pack(needs=[need], actions=actions, candidate_pool=5, temperature=2.0)

    # Con temperatura alta y candidatos idénticos, ticks sucesivos producirán
    # al menos alguna elección distinta en un recorrido de varios ticks
    choices = set()
    for t in range(20):
        exec_st = _make_execution(p, seed=42, tick=t)
        d = decide.decide(exec_st, p)
        assert d is not None
        choices.add(d.action)

    assert len(choices) > 1, "Ticks distintos deben diversificar la secuencia del stream RNG"


def test_softmax_determinismo_entre_procesos_con_pythonhashseed() -> None:
    """La decisión bajo misma semilla no varía entre procesos con diferente PYTHONHASHSEED."""
    code = """
import aida.agents.decide as decide
import aida.engine.ontology as ontology
import aida.engine.pack as pack
import aida.engine.state as state
import aida.engine.rng as rng

need = ontology.NeedSpec(id=ontology.NeedId("n1"), label="N1", decay_per_tick=0.01, initial=0.0)
acts = [
    ontology.ActionSpec(
        id=ontology.ActionId(f"act_{i}"),
        label=f"Act {i}",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={ontology.NeedId("n1"): 0.5 + i * 0.05},
        effects={ontology.NeedId("n1"): 0.5 + i * 0.05},
    )
    for i in range(4)
]
agent_spec = ontology.AgentSpec(
    id=ontology.AgentId("ag1"), label="Ag1", initial_needs={need.id: 0.0}
)
p = pack.ContentPack(
    id="p1",
    label="P1",
    needs=[need],
    actions=acts,
    agents=[agent_spec],
    tuning=ontology.TuningSpec(candidate_pool=3, temperature=0.3),
)
agent_st = state.AgentState(agent_id=ontology.AgentId("ag1"), needs={need.id: 0.0})
agent_st.recompose(p)
exec_st = state.ExecutionState(agent=agent_st, tick=5, rng=rng.RngSource(12345))

d = decide.decide(exec_st, p)
print(f"{d.action}:{d.utility:.6f}")
"""
    results = []
    for hseed in ["0", "1", "42", "999"]:
        p_res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hseed},
        )
        results.append(p_res.stdout.strip())

    assert len(set(results)) == 1, f"Decisiones difieren entre PYTHONHASHSEED: {results}"


# =============================================================================
# 13. Ancla de remediación I-024 (§5.1): Aversión al coste (loss_aversion)
# =============================================================================


def test_loss_aversion_altera_preferencia_entre_accion_costosa_y_accion_barata() -> None:
    """Demuestra que loss_aversion cambia la preferencia de acción (I-024).

    - Con aversión al coste baja (loss_aversion = 1.0), gana una acción cara
      pero muy rentable (alto beneficio neto).
    - Con aversión al coste alta (loss_aversion = 3.0), gana una acción barata
      aunque rinda menos beneficio bruto, porque el coste penaliza severamente.

    Ancla contra mutación: quitar 'agent.temperament.loss_aversion *' en la
    fórmula de utilidad en candidates(...) deja la deliberación inmune a la
    aversión al coste, provocando el fallo de este test.
    """
    need_b = _make_need("need_benefit", initial=0.0, cap=1.0, exponent=1.0)
    need_c = _make_need("need_cost", initial=0.0, cap=1.0, exponent=1.0)

    act_expensive = _make_action(
        "act_expensive",
        satisfies={"need_benefit": 1.0},
        costs={"need_cost": 0.4},
    )
    act_cheap = _make_action(
        "act_cheap",
        satisfies={"need_benefit": 0.5},
        costs={"need_cost": 0.1},
    )

    trait_prudent = ontology.TraitSpec(
        id=ontology.TraitId("t_prudent"),
        label="Prudent",
        temperament_multipliers={ontology.TemperamentKnob.LOSS_AVERSION: 3.0},
    )

    p = _make_pack(
        needs=[need_b, need_c],
        actions=[act_expensive, act_cheap],
        traits=[trait_prudent],
        tuning=ontology.TuningSpec(loss_aversion=1.0, candidate_pool=1),
    )

    # Caso 1: Agente neutro sin rasgo (loss_aversion = 1.0)
    # util(expensive) = 1.0 - 1.0 * 0.4 = 0.6
    # util(cheap)     = 0.5 - 1.0 * 0.1 = 0.4
    # -> Gana act_expensive
    exec_neutral = _make_execution(p, traits={"t_prudent": 0.0})
    assert exec_neutral.agent.temperament is not None
    assert exec_neutral.agent.temperament.loss_aversion == pytest.approx(1.0)

    act_exp_id = ontology.ActionId("act_expensive")
    act_chp_id = ontology.ActionId("act_cheap")

    pool_neutral, _ = decide.candidates(exec_neutral, p)
    utils_neutral = {d.action: d.utility for d in pool_neutral}
    assert utils_neutral[act_exp_id] == pytest.approx(0.6)
    assert utils_neutral[act_chp_id] == pytest.approx(0.4)
    assert utils_neutral[act_exp_id] > utils_neutral[act_chp_id]

    decision_neutral = decide.decide(exec_neutral, p)
    assert decision_neutral is not None
    assert decision_neutral.action == act_exp_id

    # Caso 2: Agente con rasgo prudente (loss_aversion = 3.0)
    # util(expensive) = 1.0 - 3.0 * 0.4 = 1.0 - 1.2 = -0.2
    # util(cheap)     = 0.5 - 3.0 * 0.1 = 0.5 - 0.3 = 0.2
    # -> Gana act_cheap (inversión estricta de orden de preferencia)
    exec_prudent = _make_execution(p, traits={"t_prudent": 1.0})
    assert exec_prudent.agent.temperament is not None
    assert exec_prudent.agent.temperament.loss_aversion == pytest.approx(3.0)

    pool_prudent, _ = decide.candidates(exec_prudent, p)
    utils_prudent = {d.action: d.utility for d in pool_prudent}
    assert utils_prudent[act_exp_id] == pytest.approx(-0.2)
    assert utils_prudent[act_chp_id] == pytest.approx(0.2)
    assert utils_prudent[act_chp_id] > utils_prudent[act_exp_id]

    decision_prudent = decide.decide(exec_prudent, p)
    assert decision_prudent is not None
    assert decision_prudent.action == act_chp_id


# =============================================================================
# 14. Ancla de remediación I-027 (§5.4): Modulación de critical_needs por tarea en curso
# =============================================================================


def test_critical_needs_modulado_por_avance_de_tarea_en_curso_y_switch_cost() -> None:
    """El avance de una tarea en curso reduce el umbral de necesidad crítica (I-027).

    Regla: Si el agente está ejecutando una acción (agent.performing), el umbral
    se modula según:
        threshold = critical_need_threshold * (1.0 - progress * switch_cost)
    donde progress = 1.0 - ticks_remaining / total_ticks.

    A mayor avance de la tarea en curso, más difícil resulta interrumpir al agente:
    hace falta que el eje esté peor (valor menor) para catalogarse como emergencia.
    Un valor de necesidad intermedio (ej. 0.3):
    - Con la acción recién empezada (progress=0.0, threshold=0.4): 0.3 < 0.4 -> ES crítico.
    - Con la acción casi terminada (progress=0.8, threshold=0.24): 0.3 > 0.24 -> NO es crítico.

    Ancla contra mutación: borrar el bloque de modulación por agent.performing
    deja el umbral estático en 0.4, provocando que la tarea casi terminada siga
    reportando la necesidad como crítica y haciendo fallar el test.
    """
    need_a = _make_need("need_crit", initial=0.3, cap=1.0, exponent=1.0, urgent_when_low=True)
    act_work = _make_action("act_work", duration_ticks=10)
    p = _make_pack(
        needs=[need_a],
        actions=[act_work],
        tuning=ontology.TuningSpec(
            critical_need_threshold=0.4,
            switch_cost=0.5,
        ),
    )

    # Escenario 1: Tarea recién empezada (ticks_remaining = 10, total_ticks = 10 -> progress = 0.0)
    # threshold = 0.4 * (1.0 - 0.0 * 0.5) = 0.4
    # need value = 0.3 < 0.4 -> es necesidad crítica
    exec_early = _make_execution(p, needs={"need_crit": 0.3})
    assert exec_early.agent.temperament is not None
    assert exec_early.agent.temperament.critical_need_threshold == pytest.approx(0.4)
    assert exec_early.agent.temperament.switch_cost == pytest.approx(0.5)

    exec_early.agent.performing = state.PerformingAction(
        action=ontology.ActionId("act_work"),
        ticks_remaining=10,
        total_ticks=10,
    )
    assert decide.critical_needs(exec_early, p) == [ontology.NeedId("need_crit")]

    # Escenario 2: Tarea casi terminada (ticks_remaining = 2, total_ticks = 10 -> progress = 0.8)
    # threshold = 0.4 * (1.0 - 0.8 * 0.5) = 0.4 * 0.6 = 0.24
    # need value = 0.3 > 0.24 -> ya no califica como crítica
    exec_late = _make_execution(p, needs={"need_crit": 0.3})
    exec_late.agent.performing = state.PerformingAction(
        action=ontology.ActionId("act_work"),
        ticks_remaining=2,
        total_ticks=10,
    )
    assert decide.critical_needs(exec_late, p) == []

