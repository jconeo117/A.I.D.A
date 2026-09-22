"""Tests para el panel de salud de un content pack (HealthReport, measure, check).

Cubre las especificaciones de §6.1 a §6.6 de 16-plan-u5c.md:
- §6.1: measure cuenta completadas y detecta acciones dominadas en unused.
- §6.2: crisis_time mide presión real entre dos packs contrastados.
- §6.3: abort_ratio maneja el caso sin datos (sin división por cero) y abandonos reales.
- §6.4: differentiation es None con una ficha, > 0 con dos fichas distintas.
- §6.5: stalls cuenta instantes en que el agente queda libre y sin opciones.
- §6.6: check dispara el mensaje correcto para cada banda y universal invariant.
"""

from __future__ import annotations

import pytest

import aida.engine.ontology as ontology
import aida.engine.pack as pack
from aida.health import HealthReport, check, measure

# ---------------------------------------------------------------------------
# Funciones auxiliares para construir packs sintéticos en tests
# ---------------------------------------------------------------------------


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
    agents: list[ontology.AgentSpec] | None = None,
    traits: list[ontology.TraitSpec] | None = None,
    critical_need_threshold: float = 0.15,
    switch_cost: float = 0.0,
    candidate_pool: int | None = None,
    temperature: float = 0.01,
    health: ontology.HealthSpec | None = None,
) -> pack.ContentPack:
    tuning = ontology.TuningSpec(
        critical_need_threshold=critical_need_threshold,
        switch_cost=switch_cost,
        candidate_pool=candidate_pool if candidate_pool is not None else (len(actions) or 1),
        temperature=temperature,
    )
    if agents is None:
        agents = [
            ontology.AgentSpec(
                id=ontology.AgentId("ag1"),
                label="Ag 1",
                initial_needs={n.id: n.initial for n in needs},
            )
        ]
    return pack.ContentPack(
        id="pack_health_test",
        label="Pack Health Test",
        needs=needs,
        actions=actions,
        agents=agents,
        traits=traits or [],
        tuning=tuning,
        health=health or ontology.HealthSpec(),
    )


# ---------------------------------------------------------------------------
# §6.1 — measure cuenta lo que de verdad pasó
# ---------------------------------------------------------------------------


def test_measure_completed_count() -> None:
    """§6.1 Test 1: 10 instantes con acción de 1 tick completan exactamente 9 veces.

    El instante 1 delibera e inicia la acción; se completa en el instante 2.
    Diez step() resultan en 9 completadas, y unused está vacío.
    Ancla Mutación 1: si completions[event.action] += 1 no incrementa, completed es 0.
    """
    n1 = _make_need("n1", initial=0.5, decay_per_tick=0.01)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.3})
    p = _make_pack([n1], [act1])

    report = measure(p, days=10, seeds=(1,))

    assert report.completed == 9
    assert report.unused == frozenset()
    assert report.stalls == 0


def test_measure_detects_unused_dominated_action() -> None:
    """§6.1 Test 2: una acción dominada en utilidad nunca se completa y aparece en unused.

    act1 (satisfies 0.4) domina a act_pobre (satisfies 0.01)
    bajo candidate_pool=1 y temperature=0.01.
    act_pobre nunca se elige, por lo que nunca se completa y report.unused contiene act_pobre.
    Ancla Mutación 2: si unused = frozenset(), report.unused saldría vacío y falla.
    """
    n1 = _make_need("n1", initial=0.5, decay_per_tick=0.01)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.4})
    act_pobre = _make_action("act_pobre", duration_ticks=1, satisfies={"n1": 0.01})
    p = _make_pack(
        [n1],
        [act1, act_pobre],
        candidate_pool=1,
        temperature=0.01,
    )

    report = measure(p, days=10, seeds=(1,))

    assert ontology.ActionId("act_pobre") in report.unused
    assert ontology.ActionId("act1") not in report.unused
    assert report.completed == 9
    assert report.stalls == 0


# ---------------------------------------------------------------------------
# §6.2 — crisis_time mide presión real, no una constante
# ---------------------------------------------------------------------------


def test_crisis_time_reflects_pressure_difference() -> None:
    """§6.2: dos packs idénticos salvo decay_per_tick de n_crit contrastan crisis_time.

    Pack A (decay=0.2) se hunde rápidamente por debajo del umbral crítico (0.15).
    Pack B (decay=0.0) se mantiene en 0.5 > 0.15 sin ninguna crisis.
    Ancla Mutación 3: si critical_needs se ignora (if False:), ambos dan 0.0 y la comparación falla.
    """
    n_crit_a = _make_need("n_crit", initial=0.5, decay_per_tick=0.2)
    n_crit_b = _make_need("n_crit", initial=0.5, decay_per_tick=0.0)
    n_otra = _make_need("n_otra", initial=0.5, decay_per_tick=0.01)
    act_otra = _make_action("act_otra", duration_ticks=1, satisfies={"n_otra": 0.3})

    pack_a = _make_pack([n_crit_a, n_otra], [act_otra], critical_need_threshold=0.15)
    pack_b = _make_pack([n_crit_b, n_otra], [act_otra], critical_need_threshold=0.15)

    report_a = measure(pack_a, days=20, seeds=(1,))
    report_b = measure(pack_b, days=20, seeds=(1,))

    assert report_a.crisis_time > report_b.crisis_time
    assert report_b.crisis_time == 0.0
    assert report_a.crisis_time > 0.0


# ---------------------------------------------------------------------------
# §6.3 — abort_ratio y el caso sin datos
# ---------------------------------------------------------------------------


def test_abort_ratio_no_data_zero_division_guard() -> None:
    """§6.3 Test 1: un pack sin acciones mide abort_ratio=0.0 sin ZeroDivisionError.

    Con actions=[], completed=0 y aborted=0, la división abort_total / denom
    está protegida por guarda.
    Ancla Mutación 4: sin la guarda denom else 0.0, este test lanza ZeroDivisionError.
    """
    n_calm = _make_need("n_calm", initial=0.5, decay_per_tick=0.01)
    p = _make_pack([n_calm], actions=[])

    report = measure(p, days=5, seeds=(1,))

    assert report.abort_ratio == 0.0
    assert report.completed == 0
    assert report.stalls == 5


def test_abort_ratio_real_aborts() -> None:
    """§6.3 Test 2: acción interrumpible larga se aborta ante necesidad crítica emergente.

    act_long dura 5 ticks; n_urg decae 0.25 por tick cruzando el umbral crítico 0.40,
    provocando abortos reales registrados en abort_ratio.
    """
    n_calm = _make_need("n_calm", initial=0.5, decay_per_tick=0.01)
    n_urg = _make_need("n_urg", initial=0.8, decay_per_tick=0.25)
    act_long = _make_action(
        "act_long",
        duration_ticks=5,
        interruptible=True,
        satisfies={"n_calm": 0.2},
    )
    act_urg = _make_action(
        "act_urg",
        duration_ticks=1,
        interruptible=True,
        satisfies={"n_urg": 0.5},
    )
    p = _make_pack(
        [n_calm, n_urg],
        [act_long, act_urg],
        critical_need_threshold=0.4,
    )

    report = measure(p, days=20, seeds=(1,))

    assert report.abort_ratio > 0.0
    assert report.completed > 0


# ---------------------------------------------------------------------------
# §6.4 — differentiation: None con una ficha, número con dos distintas
# ---------------------------------------------------------------------------


def test_differentiation_none_for_single_agent() -> None:
    """§6.4 Test 1: con un solo agente en el pack, differentiation es None.

    No hay con qué comparar (no aplica); es conceptualmente distinto a un 0.0 medido.
    Ancla Mutación 5: si len(profiles) >= 1, differentiation dejaría de ser None y falla.
    """
    n1 = _make_need("n1", initial=0.5, decay_per_tick=0.01)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.3})
    p = _make_pack([n1], [act1])

    report = measure(p, days=10, seeds=(1,))

    assert report.differentiation is None


def _make_pack_a(
    agents: list[ontology.AgentSpec] | None = None,
) -> pack.ContentPack:
    """Pack A (simple): dos necesidades sin desgaste, dos acciones simétricas,
    dos rasgos de peso x5 y candidate_pool=2, temperature=0.01.
    """
    n_a = ontology.NeedSpec(
        id=ontology.NeedId("n_a"),
        label="Need A",
        decay_per_tick=0.05,
        initial=0.5,
        cap=1.0,
    )
    n_b = ontology.NeedSpec(
        id=ontology.NeedId("n_b"),
        label="Need B",
        decay_per_tick=0.05,
        initial=0.5,
        cap=1.0,
    )
    act_a = ontology.ActionSpec(
        id=ontology.ActionId("act_a"),
        label="Action A",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={ontology.NeedId("n_a"): 0.3},
        effects={ontology.NeedId("n_a"): 0.3},
    )
    act_b = ontology.ActionSpec(
        id=ontology.ActionId("act_b"),
        label="Action B",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={ontology.NeedId("n_b"): 0.3},
        effects={ontology.NeedId("n_b"): 0.3},
    )
    t_a = ontology.TraitSpec(
        id=ontology.TraitId("t_a"),
        label="Trait A",
        need_weight_multipliers={ontology.NeedId("n_a"): 5.0},
    )
    t_b = ontology.TraitSpec(
        id=ontology.TraitId("t_b"),
        label="Trait B",
        need_weight_multipliers={ontology.NeedId("n_b"): 5.0},
    )
    tuning = ontology.TuningSpec(candidate_pool=2, temperature=0.01)
    if agents is None:
        agents = [
            ontology.AgentSpec(
                id=ontology.AgentId("ag1"),
                label="Agente 1",
                initial_needs={ontology.NeedId("n_a"): 0.5, ontology.NeedId("n_b"): 0.5},
            ),
            ontology.AgentSpec(
                id=ontology.AgentId("ag2"),
                label="Agente 2",
                initial_needs={ontology.NeedId("n_a"): 0.5, ontology.NeedId("n_b"): 0.5},
            ),
        ]
    return pack.ContentPack(
        id="pack_a",
        label="Pack A",
        needs=[n_a, n_b],
        actions=[act_a, act_b],
        traits=[t_a, t_b],
        agents=agents,
        tuning=tuning,
    )


def _make_pack_b(
    agents: list[ontology.AgentSpec] | None = None,
) -> pack.ContentPack:
    """Pack B (con desgaste): igual que Pack A pero necesidades con strain_weight=1.0,
    act_b cuesta 0.05 de n_a, y tuning añade loss_aversion=1.0 y strain_multipliers.
    """
    n_a = ontology.NeedSpec(
        id=ontology.NeedId("n_a"),
        label="Need A",
        decay_per_tick=0.05,
        initial=0.5,
        cap=1.0,
        strain_weight=1.0,
    )
    n_b = ontology.NeedSpec(
        id=ontology.NeedId("n_b"),
        label="Need B",
        decay_per_tick=0.05,
        initial=0.5,
        cap=1.0,
        strain_weight=1.0,
    )
    act_a = ontology.ActionSpec(
        id=ontology.ActionId("act_a"),
        label="Action A",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={ontology.NeedId("n_a"): 0.3},
        effects={ontology.NeedId("n_a"): 0.3},
    )
    act_b = ontology.ActionSpec(
        id=ontology.ActionId("act_b"),
        label="Action B",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={ontology.NeedId("n_b"): 0.3},
        costs={ontology.NeedId("n_a"): 0.05},
        effects={ontology.NeedId("n_b"): 0.3, ontology.NeedId("n_a"): -0.05},
    )
    t_a = ontology.TraitSpec(
        id=ontology.TraitId("t_a"),
        label="Trait A",
        need_weight_multipliers={ontology.NeedId("n_a"): 5.0},
    )
    t_b = ontology.TraitSpec(
        id=ontology.TraitId("t_b"),
        label="Trait B",
        need_weight_multipliers={ontology.NeedId("n_b"): 5.0},
    )
    tuning = ontology.TuningSpec(
        candidate_pool=2,
        temperature=0.01,
        loss_aversion=1.0,
        strain_multipliers={ontology.TemperamentKnob.LOSS_AVERSION: 4.0},
    )
    if agents is None:
        agents = [
            ontology.AgentSpec(
                id=ontology.AgentId("ag_a"),
                label="Ag A",
                traits={ontology.TraitId("t_a"): 1.0},
                initial_needs={ontology.NeedId("n_a"): 0.5, ontology.NeedId("n_b"): 0.5},
            ),
            ontology.AgentSpec(
                id=ontology.AgentId("ag_b"),
                label="Ag B",
                traits={ontology.TraitId("t_b"): 1.0},
                initial_needs={ontology.NeedId("n_a"): 0.5, ontology.NeedId("n_b"): 0.5},
            ),
        ]
    return pack.ContentPack(
        id="pack_b",
        label="Pack B",
        needs=[n_a, n_b],
        actions=[act_a, act_b],
        traits=[t_a, t_b],
        agents=agents,
        tuning=tuning,
    )


def test_differentiation_zero_for_identical_twins() -> None:
    """§6.1 Test 1 (Gemelas): Pack A, days=20, seeds=(1,).

    Dos fichas idénticas en todo salvo identificador arrojan exactamente 0.0.
    Sin azar residual en la métrica; evalúa carácter y no suerte.
    """
    p = _make_pack_a()
    report = measure(p, days=20, seeds=(1,))
    assert report.differentiation == 0.0
    assert type(report.differentiation) is float


def test_differentiation_for_distinct_agents_pack_b() -> None:
    """§6.1 Test 2 (Fichas distintas): Pack B, days=10, seeds=(1,).

    Dos fichas con rasgos distintos (ag_a pondera n_a, ag_b pondera n_b) sobre Pack B
    arrojan differentiation medido con exactitud (0.49613649177154817).
    """
    p = _make_pack_b()
    report = measure(p, days=10, seeds=(1,))
    assert report.differentiation == pytest.approx(0.49613649177154817)


def test_differentiation_invariant_to_seeds_over_pack_b() -> None:
    """§6.1 Test 4 (Invarianza ante semillas): sobre Pack B,
    medir con seeds=(1,) vs seeds=(1, 2) produce exactamente el mismo número.
    """
    p = _make_pack_b()
    report1 = measure(p, days=10, seeds=(1,))
    report2 = measure(p, days=10, seeds=(1, 2))
    assert report1.differentiation == pytest.approx(0.49613649177154817)
    assert report2.differentiation == pytest.approx(0.49613649177154817)
    assert report1.differentiation == report2.differentiation



def test_differentiation_positive_for_distinct_agents() -> None:
    """§6.4 Test 2: dos agentes con rasgos opuestos arrojan differentiation > 0.0.

    Agente A pondera n_a mediante t_a; Agente B pondera n_b mediante t_b.
    Sus perfiles de actividad difieren, produciendo distancia L1 estrictamente positiva.
    Ancla Mutación 6: si _l1_distance devuelve 0.0 fijo, report.differentiation saldría 0.0 y falla.
    """
    n_a = _make_need("n_a", initial=0.5, decay_per_tick=0.01)
    n_b = _make_need("n_b", initial=0.5, decay_per_tick=0.01)
    act_a = _make_action("act_a", duration_ticks=1, satisfies={"n_a": 0.3})
    act_b = _make_action("act_b", duration_ticks=1, satisfies={"n_b": 0.3})
    t_a = ontology.TraitSpec(
        id=ontology.TraitId("t_a"),
        label="Trait A",
        need_weight_multipliers={ontology.NeedId("n_a"): 2.0},
    )
    t_b = ontology.TraitSpec(
        id=ontology.TraitId("t_b"),
        label="Trait B",
        need_weight_multipliers={ontology.NeedId("n_b"): 2.0},
    )
    ag_a = ontology.AgentSpec(
        id=ontology.AgentId("ag_a"),
        label="Ag A",
        traits={ontology.TraitId("t_a"): 1.0},
    )
    ag_b = ontology.AgentSpec(
        id=ontology.AgentId("ag_b"),
        label="Ag B",
        traits={ontology.TraitId("t_b"): 1.0},
    )
    p = _make_pack(
        [n_a, n_b],
        [act_a, act_b],
        agents=[ag_a, ag_b],
        traits=[t_a, t_b],
    )

    report = measure(p, days=20, seeds=(1,))

    assert report.differentiation is not None
    assert report.differentiation > 0.0


# ---------------------------------------------------------------------------
# §6.5 — stalls: cuando el agente queda libre y sin nada que hacer
# ---------------------------------------------------------------------------


def test_stalls_recorded_when_no_actions_available() -> None:
    """§6.5: pack sin acciones registra un bloqueo en cada instante simulado.

    Para days=5, decide() devuelve None en cada tick y stalls es exactamente 5.
    Ancla Mutación 7: si loop.execution.agent.performing is None se salta (if False:), stalls es 0.
    """
    n1 = _make_need("n1", initial=0.5, decay_per_tick=0.01)
    p = _make_pack([n1], actions=[])

    report = measure(p, days=5, seeds=(1,))

    assert report.stalls == 5


# ---------------------------------------------------------------------------
# §6.6 — check: cada banda e invariante universal dispara su propio mensaje
# ---------------------------------------------------------------------------


def test_check_flags_zero_completions_universal_invariant() -> None:
    """§6.6 Invariante 1: completed == 0 genera el mensaje universal de fallo.

    Ancla Mutación 13: si if report.completed == 0 se comenta o pone if False:,
    el mensaje no aparece.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", satisfies={"n1": 0.3})
    p = _make_pack([n1], [act1])

    report_zero = HealthReport(
        days=10,
        seeds=(1,),
        agents=1,
        crisis_time=0.1,
        abort_ratio=0.0,
        completed=0,
        stalls=0,
    )
    problems_zero = check(p, report_zero)
    assert any("nadie completó ninguna acción en toda la corrida" in msg for msg in problems_zero)

    report_ok = HealthReport(
        days=10,
        seeds=(1,),
        agents=1,
        crisis_time=0.1,
        abort_ratio=0.0,
        completed=5,
        stalls=0,
    )
    problems_ok = check(p, report_ok)
    assert not any("nadie completó ninguna acción" in msg for msg in problems_ok)


def test_check_flags_stalls_universal_invariant() -> None:
    """§6.6 Invariante 2: stalls > 0 genera el mensaje universal de bloqueo de opciones.

    Ancla Mutación 14: si if report.stalls se comenta o pone if False:, el mensaje no aparece.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", satisfies={"n1": 0.3})
    p = _make_pack([n1], [act1])

    report_stalls = HealthReport(
        days=10,
        seeds=(1,),
        agents=1,
        crisis_time=0.1,
        abort_ratio=0.0,
        completed=5,
        stalls=3,
    )
    problems_stalls = check(p, report_stalls)
    assert any(
        "3 instantes sin ninguna opción disponible" in msg
        and "revisa `requires`/`requires_traits`" in msg
        for msg in problems_stalls
    )

    report_ok = HealthReport(
        days=10,
        seeds=(1,),
        agents=1,
        crisis_time=0.1,
        abort_ratio=0.0,
        completed=5,
        stalls=0,
    )
    problems_ok = check(p, report_ok)
    assert not any("sin ninguna opción disponible" in msg for msg in problems_ok)


def test_check_flags_min_crisis_violation() -> None:
    """§6.6 Banda 1: report.crisis_time < spec.min_crisis dispara mensaje específico.

    Ancla Mutación 8: si if spec.min_crisis is not None and ... se pone if False:,
    el mensaje no aparece.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", satisfies={"n1": 0.3})
    p_with_band = _make_pack([n1], [act1], health=ontology.HealthSpec(min_crisis=0.20))
    p_without_band = _make_pack([n1], [act1], health=ontology.HealthSpec(min_crisis=None))

    report_low = HealthReport(
        days=10,
        seeds=(1,),
        agents=1,
        crisis_time=0.02,
        abort_ratio=0.0,
        completed=5,
        stalls=0,
    )

    problems = check(p_with_band, report_low)
    assert any(
        "tiempo en crisis" in msg
        and "por debajo del mínimo" in msg
        and "sin presión, el carácter deja de notarse" in msg
        for msg in problems
    )

    # Caso contrario: banda no declarada en pack
    problems_no_band = check(p_without_band, report_low)
    assert not any("por debajo del mínimo" in msg for msg in problems_no_band)


def test_check_flags_max_crisis_violation() -> None:
    """§6.6 Banda 2: report.crisis_time > spec.max_crisis dispara mensaje específico.

    Ancla Mutación 9: si if spec.max_crisis is not None and ... se pone if False:,
    el mensaje no aparece.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", satisfies={"n1": 0.3})
    p_with_band = _make_pack([n1], [act1], health=ontology.HealthSpec(max_crisis=0.50))
    p_without_band = _make_pack([n1], [act1], health=ontology.HealthSpec(max_crisis=None))

    report_high = HealthReport(
        days=10,
        seeds=(1,),
        agents=1,
        crisis_time=0.85,
        abort_ratio=0.0,
        completed=5,
        stalls=0,
    )

    problems = check(p_with_band, report_high)
    assert any(
        "tiempo en crisis" in msg
        and "por encima del máximo" in msg
        and "la emergencia aplasta al carácter" in msg
        for msg in problems
    )

    # Caso contrario: banda no declarada
    problems_no_band = check(p_without_band, report_high)
    assert not any("por encima del máximo" in msg and "crisis" in msg for msg in problems_no_band)


def test_check_flags_max_abort_ratio_violation() -> None:
    """§6.6 Banda 3: report.abort_ratio > spec.max_abort_ratio dispara mensaje específico.

    Ancla Mutación 10: si if spec.max_abort_ratio is not None and ... se pone if False:,
    el mensaje no aparece.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", satisfies={"n1": 0.3})
    p_with_band = _make_pack([n1], [act1], health=ontology.HealthSpec(max_abort_ratio=0.30))
    p_without_band = _make_pack([n1], [act1], health=ontology.HealthSpec(max_abort_ratio=None))

    report_high = HealthReport(
        days=10,
        seeds=(1,),
        agents=1,
        crisis_time=0.10,
        abort_ratio=0.60,
        completed=5,
        stalls=0,
    )

    problems = check(p_with_band, report_high)
    assert any(
        "abandonos" in msg and "por encima del máximo" in msg
        for msg in problems
    )

    # Caso contrario: banda no declarada
    problems_no_band = check(p_without_band, report_high)
    assert not any("abandonos" in msg for msg in problems_no_band)


def test_check_flags_min_differentiation_violation() -> None:
    """§6.6 Banda 4: report.differentiation < spec.min_differentiation dispara mensaje específico.

    Ancla Mutación 11: si el bloque de min_differentiation se pone if False:, el mensaje no aparece.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", satisfies={"n1": 0.3})
    p_with_band = _make_pack([n1], [act1], health=ontology.HealthSpec(min_differentiation=0.40))
    p_without_band = _make_pack([n1], [act1], health=ontology.HealthSpec(min_differentiation=None))

    report_low = HealthReport(
        days=10,
        seeds=(1,),
        agents=2,
        crisis_time=0.10,
        abort_ratio=0.0,
        completed=5,
        stalls=0,
        differentiation=0.05,
    )

    problems = check(p_with_band, report_low)
    assert any(
        "diferenciación" in msg
        and "por debajo del mínimo" in msg
        and "dos fichas de personalidad distintas están decidiendo casi igual" in msg
        for msg in problems
    )

    # Caso contrario: banda no declarada
    problems_no_band = check(p_without_band, report_low)
    assert not any("diferenciación" in msg for msg in problems_no_band)

    # Con 1 agente (differentiation is None), la comprobación no aplica ni falla
    report_none = HealthReport(
        days=10,
        seeds=(1,),
        agents=1,
        crisis_time=0.10,
        abort_ratio=0.0,
        completed=5,
        stalls=0,
        differentiation=None,
    )
    assert not any("diferenciación" in msg for msg in check(p_with_band, report_none))


def test_check_flags_orphan_unused_action_violation() -> None:
    """§6.6 Banda 5: acciones no completadas fuera de may_go_unused disparan mensaje huérfano.

    Ancla Mutación 12: si el bloque orphans se pone if False:, el mensaje de huérfano no aparece.
    """
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", satisfies={"n1": 0.3})
    act_pobre = _make_action("act_pobre", satisfies={"n1": 0.01})
    p_without_exempt = _make_pack(
        [n1],
        [act1, act_pobre],
        health=ontology.HealthSpec(may_go_unused=frozenset()),
    )
    p_with_exempt = _make_pack(
        [n1],
        [act1, act_pobre],
        health=ontology.HealthSpec(may_go_unused=frozenset([ontology.ActionId("act_pobre")])),
    )

    report_with_unused = HealthReport(
        days=10,
        seeds=(1,),
        agents=1,
        crisis_time=0.10,
        abort_ratio=0.0,
        completed=5,
        stalls=0,
        unused=frozenset([ontology.ActionId("act_pobre")]),
    )

    problems = check(p_without_exempt, report_with_unused)
    assert any(
        "acciones declaradas que nunca se completan" in msg
        and "act_pobre" in msg
        and "`health.may_go_unused`" in msg
        for msg in problems
    )

    # Caso contrario: la acción está exenta en may_go_unused
    problems_exempt = check(p_with_exempt, report_with_unused)
    assert not any("acciones declaradas que nunca se completan" in msg for msg in problems_exempt)


def test_check_negative_control_clean_report() -> None:
    """§6.6 Control negativo: un reporte que cumple todas las bandas devuelve lista vacía []."""
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", satisfies={"n1": 0.3})
    act_pobre = _make_action("act_pobre", satisfies={"n1": 0.01})
    p_clean = _make_pack(
        [n1],
        [act1, act_pobre],
        health=ontology.HealthSpec(
            min_crisis=0.05,
            max_crisis=0.80,
            max_abort_ratio=0.50,
            min_differentiation=0.05,
            may_go_unused=frozenset([ontology.ActionId("act_pobre")]),
        ),
    )
    compliant_report = HealthReport(
        days=10,
        seeds=(1,),
        agents=2,
        crisis_time=0.25,
        abort_ratio=0.10,
        completed=15,
        stalls=0,
        unused=frozenset([ontology.ActionId("act_pobre")]),
        differentiation=0.30,
    )

    problems = check(p_clean, compliant_report)
    assert problems == []


# ---------------------------------------------------------------------------
# Cobertura complementaria de HealthReport.lines y valores por omisión
# ---------------------------------------------------------------------------


def test_health_report_lines_formatting() -> None:
    """Verifica el formato legible de HealthReport.lines()."""
    # Sin diferenciación y con unused
    r1 = HealthReport(
        days=10,
        seeds=(1, 2),
        agents=1,
        crisis_time=0.152,
        abort_ratio=0.05,
        completed=20,
        stalls=0,
        unused=frozenset([ontology.ActionId("act_x")]),
        differentiation=None,
    )
    lines1 = r1.lines()
    assert len(lines1) == 6
    assert "10 instantes · 1 ficha(s) · semillas [1, 2]" in lines1[0]
    assert "tiempo en crisis         15.2%" in lines1[2]
    assert "abandonos                 5.0% (20 completadas)" in lines1[3]
    assert "bloqueos                     0" in lines1[4]
    assert "acciones sin usar       ['act_x']" in lines1[5]

    # Con diferenciación y sin unused
    r2 = HealthReport(
        days=20,
        seeds=(42,),
        agents=2,
        crisis_time=0.0,
        abort_ratio=0.0,
        completed=10,
        stalls=2,
        unused=frozenset(),
        differentiation=0.105,
    )
    lines2 = r2.lines()
    assert len(lines2) == 7
    assert "acciones sin usar       —" in lines2[5]
    assert "diferenciación           0.105" in lines2[6]


def test_measure_and_check_default_parameters() -> None:
    """Verifica que measure usa pack.health si no se pasan días/semillas,
    y check mide automáticamente."""
    n1 = _make_need("n1", initial=0.5, decay_per_tick=0.01)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.3})
    p = _make_pack(
        [n1],
        [act1],
        health=ontology.HealthSpec(days=6, seeds=(99,)),
    )

    # measure sin argumentos explícitos
    report = measure(p)
    assert report.days == 6
    assert report.seeds == (99,)
    assert report.completed == 5  # 6 ticks -> 5 completadas

    # check sin reporte invoca measure internamente
    problems = check(p)
    assert problems == []


def test_measure_days_zero_no_divide_por_cero_y_differentiation_es_none() -> None:
    """I-047: measure(pack, days=0) no lanza ZeroDivisionError y reporta
    crisis_time == 0.0 y differentiation is None (ejerce 'if total_days else 0.0'
    y 'if len(fichas) >= 2 and sampled_states:')."""
    n1 = _make_need("n1", initial=0.5)
    act1 = _make_action("act1", duration_ticks=1, satisfies={"n1": 0.3})
    ag_a = ontology.AgentSpec(id=ontology.AgentId("ag_a"), label="A")
    ag_b = ontology.AgentSpec(id=ontology.AgentId("ag_b"), label="B")
    p = _make_pack([n1], [act1], agents=[ag_a, ag_b])

    report = measure(p, days=0)
    assert report.days == 0
    assert report.crisis_time == 0.0
    assert report.differentiation is None


def test_measure_orden_agentes_invertido_muestrea_fichas_cero_consistentemente() -> None:
    """I-047: measure con dos agentes en orden invertido en pack.agents comprueba que
    sorted(pack.agents, key=lambda a: a.id) asegura que el muestreo de estados siempre
    toma fichas[0] (el agente canónico de menor id)."""
    n1 = _make_need("n1", initial=0.5, decay_per_tick=0.05)
    act1 = _make_action("act1", satisfies={"n1": 0.3}, costs={})
    act2 = _make_action("act2", satisfies={"n1": 0.1}, costs={})

    ag_a = ontology.AgentSpec(
        id=ontology.AgentId("ag_a"),
        label="A",
        initial_needs={ontology.NeedId("n1"): 0.1},
        tastes=(ontology.TasteSpec(target=ontology.ActionId("act1"), multiplier=2.0),),
    )
    ag_b = ontology.AgentSpec(
        id=ontology.AgentId("ag_b"),
        label="B",
        initial_needs={ontology.NeedId("n1"): 0.9},
        tastes=(ontology.TasteSpec(target=ontology.ActionId("act2"), multiplier=2.0),),
    )

    p_ord = _make_pack([n1], [act1, act2], agents=[ag_a, ag_b])
    p_inv = _make_pack([n1], [act1, act2], agents=[ag_b, ag_a])

    rep_ord = measure(p_ord, days=5, seeds=(1,))
    rep_inv = measure(p_inv, days=5, seeds=(1,))

    assert rep_ord.differentiation is not None
    assert rep_inv.differentiation is not None
    assert rep_inv.differentiation == rep_ord.differentiation
    assert rep_inv.differentiation == pytest.approx(0.41142857)


def test_utility_profile_utilidad_cero_devuelve_ceros_sin_zero_division() -> None:
    """I-047: _utility_profile cuando todas las decisiones tienen utilidad 0.0
    devuelve 0.0 para todas las acciones sin dividir por cero."""
    from aida.health import _utility_profile

    n1 = _make_need("n1")
    act = _make_action("act_zero", satisfies={}, costs={})
    spec = ontology.AgentSpec(id=ontology.AgentId("ag1"), label="Ag 1")
    p = _make_pack([n1], [act], agents=[spec])

    profile = _utility_profile(p, spec, {ontology.NeedId("n1"): 0.5})
    assert profile == {ontology.ActionId("act_zero"): 0.0}
