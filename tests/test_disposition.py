"""Pruebas de la disposición y composición de personalidad (`aida.engine.disposition`).

Cubre la unidad U3 (§8 de 06-plan-u3.md):
- Intensidad como exponente: 0 es neutro (M**0 == 1), -1 es inverso exacto (M**-1 == 1/M).
- Determinismo frente al orden de inserción de rasgos en el diccionario (bit a bit idéntico).
- Independencia estricta de aptitud (gains) y temple (costs) (las 4 combinaciones).
- Los 4 compose_* devuelven todos los ejes del pack.
- compose_caps arranca del cap declarado en el pack y se recorta a 1.0.
- compose_need_weights incorpora metas activas (pursuing) en orden de identificador.
- Unicidad de urgency en todo el motor: state.urgency is disposition.urgency y no queda
  _urgency en state.py.
- Conservación del no-filtrado de deuda: la deuda no altera urgency ni compute_strain.
- Convexidad de compute_strain: necesidades en 0.55 dan desgaste marcadamente menor que 0.45.
- Cotas de temperamento: temperature suelo 1e-6, cotas superiores en 1.0, loss_aversion sin techo.
- Perilla del pack en 0 no es resucitada por ningún rasgo.
- compose_tastes: colapso multiplicativo por objetivo.
"""

from __future__ import annotations

import dataclasses

import pytest

import aida.engine.disposition as disposition
import aida.engine.state as state
from aida.engine.disposition import (
    Temperament,
    compose_caps,
    compose_costs,
    compose_gains,
    compose_need_weights,
    compose_tastes,
    compose_temperament,
    compute_strain,
    urgency,
)
from aida.engine.ontology import (
    ActionId,
    ActionSpec,
    AgentId,
    AgentSpec,
    GoalId,
    GoalSpec,
    NeedId,
    NeedSpec,
    Reversibility,
    SituationId,
    SituationSpec,
    TasteSpec,
    TemperamentKnob,
    TraitId,
    TraitSpec,
    TuningSpec,
)
from aida.engine.pack import ContentPack

_AGENTE_ID = AgentId("agente_prueba")


def _make_pack(
    *,
    needs: list[NeedSpec] | None = None,
    traits: list[TraitSpec] | None = None,
    goals: list[GoalSpec] | None = None,
    situations: list[SituationSpec] | None = None,
    tuning: TuningSpec | None = None,
) -> ContentPack:
    """Crea un ContentPack mínimo válido para pruebas de disposición."""
    pack_needs = tuple(
        needs
        if needs is not None
        else [
            NeedSpec(
                id=NeedId("n1"),
                label="Eje 1",
                decay_per_tick=0.01,
                base_weight=1.0,
                cap=1.0,
                urgency_exponent=2.0,
                strain_weight=1.0,
            ),
            NeedSpec(
                id=NeedId("n2"),
                label="Eje 2",
                decay_per_tick=0.02,
                base_weight=2.0,
                cap=0.8,
                urgency_exponent=1.5,
                strain_weight=0.5,
            ),
        ]
    )
    pack_actions = tuple(
        [
            ActionSpec(
                id=ActionId(f"act_{need.id}"),
                label=f"Accion {need.id}",
                duration_ticks=1,
                reversibility=Reversibility.REVERSIBLE,
                satisfies={need.id: 0.5},
                effects={need.id: 0.5},
            )
            for need in pack_needs
        ]
    )
    return ContentPack(
        id="pack_disp_test",
        label="Pack de prueba para disposición",
        needs=pack_needs,
        actions=pack_actions,
        agents=(
            AgentSpec(
                id=_AGENTE_ID,
                label="Agente de prueba",
                initial_needs={need.id: 0.5 for need in pack_needs},
            ),
        ),
        traits=tuple(traits or ()),
        goals=tuple(goals or ()),
        situations=tuple(situations or ()),
        tuning=tuning or TuningSpec(),
    )


# -----------------------------------------------------------------------------
# 1. Intensidad como exponente
# -----------------------------------------------------------------------------


def test_intensidad_cero_es_neutra_en_todos_los_canales() -> None:
    """Con intensidad 0.0 el factor multiplicativo debe ser exactamente 1.0 (M**0 == 1)."""
    t1 = TraitSpec(
        id=TraitId("t1"),
        label="Rasgo neutro",
        need_weight_multipliers={NeedId("n1"): 2.5},
        gain_multipliers={NeedId("n1"): 3.0},
        cost_multipliers={NeedId("n1"): 4.0},
        cap_multipliers={NeedId("n1"): 1.5},
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 2.0},
    )
    pack = _make_pack(traits=[t1])
    traits = {TraitId("t1"): 0.0}

    weights = compose_need_weights(pack, traits)
    gains = compose_gains(pack, traits)
    costs = compose_costs(pack, traits)
    caps = compose_caps(pack, traits)
    temperament = compose_temperament(pack, traits)

    # Base weight de n1 es 1.0
    assert weights[NeedId("n1")] == pytest.approx(1.0)
    assert gains[NeedId("n1")] == pytest.approx(1.0)
    assert costs[NeedId("n1")] == pytest.approx(1.0)
    assert caps[NeedId("n1")] == pytest.approx(1.0)
    assert temperament.loss_aversion == pytest.approx(pack.tuning.loss_aversion)


def test_intensidad_menos_uno_invierte_exactamente_en_todos_los_canales() -> None:
    """Con intensidad -1.0 el factor debe ser el inverso exacto (M**-1 == 1/M)."""
    t1 = TraitSpec(
        id=TraitId("t1"),
        label="Rasgo inverso",
        need_weight_multipliers={NeedId("n1"): 4.0},
        gain_multipliers={NeedId("n1"): 2.0},
        cost_multipliers={NeedId("n1"): 5.0},
        cap_multipliers={NeedId("n1"): 2.0},
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 4.0},
    )
    pack = _make_pack(traits=[t1])
    traits = {TraitId("t1"): -1.0}

    weights = compose_need_weights(pack, traits)
    gains = compose_gains(pack, traits)
    costs = compose_costs(pack, traits)
    caps = compose_caps(pack, traits)
    temperament = compose_temperament(pack, traits)

    assert weights[NeedId("n1")] == pytest.approx(1.0 * 0.25)
    assert gains[NeedId("n1")] == pytest.approx(0.5)
    assert costs[NeedId("n1")] == pytest.approx(0.2)
    assert caps[NeedId("n1")] == pytest.approx(0.5)
    assert temperament.loss_aversion == pytest.approx(pack.tuning.loss_aversion * 0.25)


def test_intensidad_como_exponente_descarta_interpolacion_lineal() -> None:
    """La interpolación lineal (1 + (M - 1) * int) daría valores negativos para M > 2
    cuando la intensidad es negativa (-1), lo cual carece de sentido. La potencia
    M ** intensidad garantiza positividad estricta y composición logarítmica limpia.
    """
    m = 3.0
    intensity = -1.0
    t1 = TraitSpec(
        id=TraitId("t1"),
        label="Rasgo fuerte",
        gain_multipliers={NeedId("n1"): m},
    )
    pack = _make_pack(traits=[t1])
    gains = compose_gains(pack, {TraitId("t1"): intensity})

    # Si fuera interpolación lineal: 1.0 + (3.0 - 1.0) * (-1.0) = -1.0 (inválido/negativo)
    # Como es exponente: 3.0 ** (-1.0) = 1/3 > 0
    assert gains[NeedId("n1")] > 0.0
    assert gains[NeedId("n1")] == pytest.approx(1.0 / 3.0)


# -----------------------------------------------------------------------------
# 2. Determinismo en orden de inserción de rasgos
# -----------------------------------------------------------------------------


def test_orden_de_insercion_de_rasgos_no_altera_resultado_bit_a_bit() -> None:
    """La iteración ordenada (sorted) sobre las claves de rasgos garantiza que el orden
    de inserción en el diccionario no afecte en ningún bit el resultado en coma flotante.
    """
    t_a = TraitSpec(
        id=TraitId("t_alpha"),
        label="Alpha",
        need_weight_multipliers={NeedId("n1"): 1.13},
        gain_multipliers={NeedId("n1"): 1.41},
        cost_multipliers={NeedId("n1"): 0.87},
        cap_multipliers={NeedId("n1"): 1.05},
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 1.25},
    )
    t_b = TraitSpec(
        id=TraitId("t_beta"),
        label="Beta",
        need_weight_multipliers={NeedId("n1"): 0.77},
        gain_multipliers={NeedId("n1"): 1.23},
        cost_multipliers={NeedId("n1"): 1.62},
        cap_multipliers={NeedId("n1"): 0.95},
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 0.85},
    )
    t_c = TraitSpec(
        id=TraitId("t_gamma"),
        label="Gamma",
        need_weight_multipliers={NeedId("n1"): 1.55},
        gain_multipliers={NeedId("n1"): 0.65},
        cost_multipliers={NeedId("n1"): 1.18},
        cap_multipliers={NeedId("n1"): 1.11},
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 1.45},
    )
    pack = _make_pack(traits=[t_a, t_b, t_c])

    orden1 = {TraitId("t_alpha"): 0.6, TraitId("t_beta"): -0.3, TraitId("t_gamma"): 0.9}
    orden2 = {TraitId("t_gamma"): 0.9, TraitId("t_beta"): -0.3, TraitId("t_alpha"): 0.6}
    orden3 = {TraitId("t_beta"): -0.3, TraitId("t_gamma"): 0.9, TraitId("t_alpha"): 0.6}

    # weights
    w1 = compose_need_weights(pack, orden1)
    w2 = compose_need_weights(pack, orden2)
    w3 = compose_need_weights(pack, orden3)
    assert w1 == w2 == w3

    # gains
    g1 = compose_gains(pack, orden1)
    g2 = compose_gains(pack, orden2)
    g3 = compose_gains(pack, orden3)
    assert g1 == g2 == g3

    # costs
    c1 = compose_costs(pack, orden1)
    c2 = compose_costs(pack, orden2)
    c3 = compose_costs(pack, orden3)
    assert c1 == c2 == c3

    # caps
    cp1 = compose_caps(pack, orden1)
    cp2 = compose_caps(pack, orden2)
    cp3 = compose_caps(pack, orden3)
    assert cp1 == cp2 == cp3

    # temperament
    temp1 = compose_temperament(pack, orden1)
    temp2 = compose_temperament(pack, orden2)
    temp3 = compose_temperament(pack, orden3)
    assert temp1 == temp2 == temp3


def test_orden_de_metas_en_need_weights_no_altera_resultado_bit_a_bit() -> None:
    """Las metas activas recibidas en `pursuing` se ordenan por su id antes de multiplicarse."""
    g1 = GoalSpec(
        id=GoalId("g_alpha"),
        label="Meta Alpha",
        need=NeedId("n1"),
        at_least=0.8,
        weight_multiplier=1.33,
    )
    g2 = GoalSpec(
        id=GoalId("g_beta"),
        label="Meta Beta",
        need=NeedId("n1"),
        at_least=0.9,
        weight_multiplier=1.67,
    )
    pack = _make_pack(goals=[g1, g2])

    w_orden1 = compose_need_weights(pack, {}, pursuing=[g1, g2])
    w_orden2 = compose_need_weights(pack, {}, pursuing=[g2, g1])

    assert w_orden1[NeedId("n1")] == w_orden2[NeedId("n1")]


def test_metas_modifican_peso_absoluto_multiplicativamente() -> None:
    """Una meta modifica el peso absoluto multiplicando (no sumando ni quedando inerte)."""
    n1 = NeedSpec(
        id=NeedId("n1"),
        label="Eje 1",
        decay_per_tick=0.01,
        base_weight=2.0,
    )
    g1 = GoalSpec(
        id=GoalId("g1"),
        label="Meta 1",
        need=NeedId("n1"),
        at_least=0.8,
        weight_multiplier=3.0,
    )
    pack = _make_pack(needs=[n1], goals=[g1])

    w_sin = compose_need_weights(pack, {}, pursuing=[])
    w_con = compose_need_weights(pack, {}, pursuing=[g1])

    # El peso base sin metas es 2.0
    assert w_sin[NeedId("n1")] == pytest.approx(2.0)
    # Con la meta activa, el peso se multiplica por 3.0 => 6.0
    assert w_con[NeedId("n1")] == pytest.approx(6.0)
    # Si fuera suma (2.0 + 3.0 = 5.0) o inerte (2.0), esta aserción falla
    assert w_con[NeedId("n1")] != pytest.approx(w_sin[NeedId("n1")] + g1.weight_multiplier)


def test_metas_efecto_proporcional_al_peso_previo() -> None:
    """Una meta sobre un eje con peso alto produce un incremento mayor que sobre
    uno con peso bajo (I-015).
    """
    n_alta = NeedSpec(
        id=NeedId("n_alta"),
        label="Eje prioritario",
        decay_per_tick=0.01,
        base_weight=10.0,
    )
    n_baja = NeedSpec(
        id=NeedId("n_baja"),
        label="Eje secundario",
        decay_per_tick=0.01,
        base_weight=1.0,
    )
    g_alta = GoalSpec(
        id=GoalId("g_alta"),
        label="Meta alta",
        need=NeedId("n_alta"),
        at_least=0.8,
        weight_multiplier=2.0,
    )
    g_baja = GoalSpec(
        id=GoalId("g_baja"),
        label="Meta baja",
        need=NeedId("n_baja"),
        at_least=0.8,
        weight_multiplier=2.0,
    )
    pack = _make_pack(needs=[n_alta, n_baja], goals=[g_alta, g_baja])

    w_sin = compose_need_weights(pack, {}, pursuing=[])
    w_con = compose_need_weights(pack, {}, pursuing=[g_alta, g_baja])

    delta_alta = w_con[NeedId("n_alta")] - w_sin[NeedId("n_alta")]
    delta_baja = w_con[NeedId("n_baja")] - w_sin[NeedId("n_baja")]

    # Con multiplicación: delta_alta = 20 - 10 = 10; delta_baja = 2 - 1 = 1. Ratio = 10.0
    assert delta_alta == pytest.approx(10.0)
    assert delta_baja == pytest.approx(1.0)
    assert delta_alta > delta_baja
    assert (delta_alta / delta_baja) == pytest.approx(10.0)


def test_orden_de_situaciones_en_compose_temperament_no_altera_resultado_bit_a_bit() -> None:
    """Las situaciones activas se ordenan por su id antes de multiplicarse."""
    s1 = SituationSpec(
        id=SituationId("sit_alpha"),
        label="Sit Alpha",
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 1.25},
    )
    s2 = SituationSpec(
        id=SituationId("sit_beta"),
        label="Sit Beta",
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 1.75},
    )
    pack = _make_pack(situations=[s1, s2])

    t1 = compose_temperament(
        pack,
        {},
        active_situations=[SituationId("sit_alpha"), SituationId("sit_beta")],
    )
    t2 = compose_temperament(
        pack,
        {},
        active_situations=[SituationId("sit_beta"), SituationId("sit_alpha")],
    )

    assert t1.loss_aversion == t2.loss_aversion


# -----------------------------------------------------------------------------
# 3. Independencia de aptitud (`gains`) y temple (`costs`)
# -----------------------------------------------------------------------------


def test_independencia_canales_aptitud_y_temple_las_cuatro_combinaciones() -> None:
    """Aptitud (gains) y temple (costs) son canales totalmente independientes.
    Las 4 combinaciones describen perfiles de agente distintos y mutuamente excluyentes.
    """
    t_apt = TraitSpec(
        id=TraitId("t_aptitud"),
        label="Aptitud alta",
        gain_multipliers={NeedId("n1"): 2.0},
    )
    t_tmp = TraitSpec(
        id=TraitId("t_temple"),
        label="Temple alto",
        cost_multipliers={NeedId("n1"): 3.0},
    )
    pack = _make_pack(traits=[t_apt, t_tmp])

    # 1. Alta aptitud, alto temple
    g1 = compose_gains(pack, {TraitId("t_aptitud"): 1.0, TraitId("t_temple"): 1.0})[NeedId("n1")]
    c1 = compose_costs(pack, {TraitId("t_aptitud"): 1.0, TraitId("t_temple"): 1.0})[NeedId("n1")]

    # 2. Alta aptitud, bajo temple
    g2 = compose_gains(pack, {TraitId("t_aptitud"): 1.0, TraitId("t_temple"): -1.0})[NeedId("n1")]
    c2 = compose_costs(pack, {TraitId("t_aptitud"): 1.0, TraitId("t_temple"): -1.0})[NeedId("n1")]

    # 3. Baja aptitud, alto temple
    g3 = compose_gains(pack, {TraitId("t_aptitud"): -1.0, TraitId("t_temple"): 1.0})[NeedId("n1")]
    c3 = compose_costs(pack, {TraitId("t_aptitud"): -1.0, TraitId("t_temple"): 1.0})[NeedId("n1")]

    # 4. Baja aptitud, bajo temple
    g4 = compose_gains(pack, {TraitId("t_aptitud"): -1.0, TraitId("t_temple"): -1.0})[NeedId("n1")]
    c4 = compose_costs(pack, {TraitId("t_aptitud"): -1.0, TraitId("t_temple"): -1.0})[NeedId("n1")]

    assert (g1, c1) == pytest.approx((2.0, 3.0))
    assert (g2, c2) == pytest.approx((2.0, 1.0 / 3.0))
    assert (g3, c3) == pytest.approx((0.5, 3.0))
    assert (g4, c4) == pytest.approx((0.5, 1.0 / 3.0))

    # Las 4 tuplas deben ser distintas entre sí
    tuplas = {(g1, c1), (g2, c2), (g3, c3), (g4, c4)}
    assert len(tuplas) == 4


def test_modificar_ganancias_no_afecta_costes_y_viceversa() -> None:
    """Modificar un rasgo con solo gain_multipliers deja costs en 1.0 y viceversa."""
    t_apt = TraitSpec(
        id=TraitId("t_apt"),
        label="Solo ganancias",
        gain_multipliers={NeedId("n1"): 5.0},
    )
    t_cst = TraitSpec(
        id=TraitId("t_cst"),
        label="Solo costes",
        cost_multipliers={NeedId("n1"): 7.0},
    )
    pack = _make_pack(traits=[t_apt, t_cst])

    # Solo aptitud activa
    g_apt = compose_gains(pack, {TraitId("t_apt"): 1.0})
    c_apt = compose_costs(pack, {TraitId("t_apt"): 1.0})
    assert g_apt[NeedId("n1")] == pytest.approx(5.0)
    assert c_apt[NeedId("n1")] == pytest.approx(1.0)

    # Solo coste activo
    g_cst = compose_gains(pack, {TraitId("t_cst"): 1.0})
    c_cst = compose_costs(pack, {TraitId("t_cst"): 1.0})
    assert g_cst[NeedId("n1")] == pytest.approx(1.0)
    assert c_cst[NeedId("n1")] == pytest.approx(7.0)


# -----------------------------------------------------------------------------
# 4. Los 4 compose_* devuelven todos los ejes del pack
# -----------------------------------------------------------------------------


def test_los_cuatro_compose_devuelven_todos_los_ejes_incluso_no_tocados() -> None:
    """Todos los compose_* deben contener todas las necesidades del pack,
    evitando que consumidores inventen valores por defecto dispersos.
    """
    t1 = TraitSpec(
        id=TraitId("t1"),
        label="Solo toca n1",
        need_weight_multipliers={NeedId("n1"): 2.0},
        gain_multipliers={NeedId("n1"): 2.0},
        cost_multipliers={NeedId("n1"): 2.0},
        cap_multipliers={NeedId("n1"): 0.9},
    )
    pack = _make_pack(traits=[t1])
    traits = {TraitId("t1"): 1.0}

    todos_los_ejes = {NeedId("n1"), NeedId("n2")}

    weights = compose_need_weights(pack, traits)
    gains = compose_gains(pack, traits)
    costs = compose_costs(pack, traits)
    caps = compose_caps(pack, traits)

    for dicc in (weights, gains, costs, caps):
        assert set(dicc.keys()) == todos_los_ejes

    # n2 no fue tocado por t1, conserva sus valores base
    assert weights[NeedId("n2")] == pytest.approx(2.0)  # base_weight de n2
    assert gains[NeedId("n2")] == pytest.approx(1.0)
    assert costs[NeedId("n2")] == pytest.approx(1.0)
    assert caps[NeedId("n2")] == pytest.approx(0.8)  # cap declarado de n2


def test_agente_sin_rasgos_obtiene_valores_de_identidad_del_pack() -> None:
    """Un agente con diccionario vacío de rasgos obtiene exactamente la identidad
    para cada canal.
    """
    pack = _make_pack()
    traits: dict[TraitId, float] = {}

    weights = compose_need_weights(pack, traits)
    gains = compose_gains(pack, traits)
    costs = compose_costs(pack, traits)
    caps = compose_caps(pack, traits)

    for need in pack.needs:
        assert weights[need.id] == pytest.approx(need.base_weight)
        assert gains[need.id] == pytest.approx(1.0)
        assert costs[need.id] == pytest.approx(1.0)
        assert caps[need.id] == pytest.approx(need.cap)


# -----------------------------------------------------------------------------
# 5. compose_caps arranca del cap declarado y se recorta a 1.0
# -----------------------------------------------------------------------------


def test_compose_caps_arranca_en_need_cap_no_en_uno() -> None:
    """compose_caps es el único canal que no arranca en 1.0: arranca en need.cap."""
    n_baja = NeedSpec(
        id=NeedId("n_baja"),
        label="Cap bajo",
        decay_per_tick=0.01,
        cap=0.4,
    )
    pack = _make_pack(needs=[n_baja])
    caps = compose_caps(pack, {})
    assert caps[NeedId("n_baja")] == pytest.approx(0.4)


def test_compose_caps_se_recorta_estrictamente_a_uno() -> None:
    """Un don extremo nunca debe superar 1.0 en compose_caps."""
    n1 = NeedSpec(
        id=NeedId("n1"),
        label="Cap alto",
        decay_per_tick=0.01,
        cap=0.8,
    )
    t_don = TraitSpec(
        id=TraitId("t_don"),
        label="Gran don",
        cap_multipliers={NeedId("n1"): 3.0},
    )
    pack = _make_pack(needs=[n1], traits=[t_don])

    # 0.8 * 3.0 = 2.4, debe recortarse a 1.0
    caps = compose_caps(pack, {TraitId("t_don"): 1.0})
    assert caps[NeedId("n1")] == 1.0


def test_compose_caps_reduce_correctamente_con_multiplicador_menor_que_uno() -> None:
    """Un rasgo reductor disminuye el tope adecuadamente por debajo de need.cap."""
    n1 = NeedSpec(
        id=NeedId("n1"),
        label="Cap normal",
        decay_per_tick=0.01,
        cap=0.8,
    )
    t_tara = TraitSpec(
        id=TraitId("t_tara"),
        label="Reductor de tope",
        cap_multipliers={NeedId("n1"): 0.5},
    )
    pack = _make_pack(needs=[n1], traits=[t_tara])

    caps = compose_caps(pack, {TraitId("t_tara"): 1.0})
    assert caps[NeedId("n1")] == pytest.approx(0.4)


# -----------------------------------------------------------------------------
# 6. Unicidad de urgency en todo el motor
# -----------------------------------------------------------------------------


def test_urgency_es_la_misma_funcion_en_disposition_y_state() -> None:
    """Invariante de una sola puerta por mecanismo: state.urgency DEBE ser la misma
    función que disposition.urgency por introspección de identidad de objeto.
    """
    assert state.urgency is disposition.urgency


def test_state_no_conserva_implementacion_privada_urgency() -> None:
    """Garantiza que _urgency haya sido eliminada de state.py y no quede código duplicado."""
    assert not hasattr(state, "_urgency")


def test_urgency_valores_matematicos_y_casos_borde() -> None:
    """Comprobación de la fórmula (1 - value/cap)**exponent y sus límites numéricos."""
    # En 0.0, urgencia máxima
    assert urgency(0.0, exponent=2.0, cap=1.0) == pytest.approx(1.0)

    # En cap, urgencia nula
    assert urgency(1.0, exponent=2.0, cap=1.0) == pytest.approx(0.0)

    # Por encima de cap, urgencia recortada a 0.0
    assert urgency(1.2, exponent=2.0, cap=1.0) == 0.0

    # Con exponente 1.0 (lineal)
    assert urgency(0.4, exponent=1.0, cap=1.0) == pytest.approx(0.6)

    # Con exponente 2.0 (cuadrática)
    assert urgency(0.5, exponent=2.0, cap=1.0) == pytest.approx(0.25)

    # Con cap menor que 1.0
    assert urgency(0.25, exponent=2.0, cap=0.5) == pytest.approx(0.25)
    assert urgency(0.5, exponent=2.0, cap=0.5) == pytest.approx(0.0)

    # Con cap <= 0.0 devuelve 0.0
    assert urgency(0.5, exponent=2.0, cap=0.0) == 0.0
    assert urgency(0.5, exponent=2.0, cap=-0.5) == 0.0

    # Conversión explícita a float
    val = urgency(0.3, 1.5, 1.0)
    assert isinstance(val, float)




# -----------------------------------------------------------------------------
# 8. Convexidad de `compute_strain`
# -----------------------------------------------------------------------------


def test_compute_strain_es_convexa() -> None:
    """Un sistema sano con necesidades en torno a 0.55 y exponente convexo produce un
    desgaste marcadamente menor que el 0.45 que daría un modelo lineal.
    """
    n1 = NeedSpec(
        id=NeedId("n1"),
        label="Eje 1",
        decay_per_tick=0.01,
        urgency_exponent=2.5,
        strain_weight=1.0,
    )
    pack = _make_pack(needs=[n1])

    # Lineal daría 1.0 - 0.55 = 0.45
    strain_sano = compute_strain(pack, {NeedId("n1"): 0.55})
    # Convexa da (1.0 - 0.55) ** 2.5 ~= 0.1357
    assert strain_sano < 0.20
    assert strain_sano < (1.0 - 0.55)


def test_compute_strain_ejes_al_maximo_da_cero() -> None:
    """Con todas las necesidades al 100%, el desgaste es exactamente 0.0."""
    pack = _make_pack()
    assert compute_strain(pack, {NeedId("n1"): 1.0, NeedId("n2"): 1.0}) == pytest.approx(0.0)


def test_compute_strain_ejes_a_cero_da_uno() -> None:
    """Con todas las necesidades en 0.0, el desgaste es 1.0."""
    pack = _make_pack()
    assert compute_strain(pack, {NeedId("n1"): 0.0, NeedId("n2"): 0.0}) == pytest.approx(1.0)


def test_compute_strain_sin_ejes_desgastantes_devuelve_cero() -> None:
    """Si ningún eje declara strain_weight > 0, compute_strain devuelve 0.0 sin fallar."""
    n_tranquila = NeedSpec(
        id=NeedId("n_tranquila"),
        label="No desgastante",
        decay_per_tick=0.01,
        strain_weight=0.0,
    )
    pack = _make_pack(needs=[n_tranquila])
    assert compute_strain(pack, {NeedId("n_tranquila"): 0.0}) == 0.0


def test_compute_strain_normaliza_por_techos_efectivos() -> None:
    """compute_strain normaliza cada eje respecto a su techo efectivo (caps),
    evitando desgaste falso (I-014).
    """
    n1 = NeedSpec(
        id=NeedId("n1"),
        label="Eje 1",
        decay_per_tick=0.01,
        cap=1.0,
        urgency_exponent=2.0,
        strain_weight=1.0,
    )
    pack = _make_pack(needs=[n1])

    # Un agente con cap=0.5 que está a 0.5 ha alcanzado su máximo: urgencia y strain 0.0
    caps_agente = {NeedId("n1"): 0.5}
    needs_agente = {NeedId("n1"): 0.5}

    strain_con_caps = compute_strain(pack, needs_agente, caps=caps_agente)
    # Sin caps usa need.cap = 1.0 => (1 - 0.5/1.0)**2 = 0.25
    strain_sin_caps = compute_strain(pack, needs_agente)

    assert strain_con_caps == pytest.approx(0.0)
    assert strain_sin_caps == pytest.approx(0.25)


# -----------------------------------------------------------------------------
# 9. Temperamento y cotas
# -----------------------------------------------------------------------------


def test_temperament_estructura_inmutable_con_slots() -> None:
    """Temperament debe ser una dataclass frozen con slots."""
    assert dataclasses.is_dataclass(Temperament)
    t = Temperament(
        temperature=0.1,
        loss_aversion=1.0,
        critical_need_threshold=0.15,
        switch_cost=0.0,
        novelty_weight=0.6,
        learning_rate=0.05,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.temperature = 0.5  # type: ignore[misc]


def test_temperament_temperature_suelo_estricto() -> None:
    """La perilla de temperature no puede bajar de 1e-6 para evitar división por cero."""
    t_frio = TraitSpec(
        id=TraitId("t_frio"),
        label="Frío extremo",
        temperament_multipliers={TemperamentKnob.TEMPERATURE: 1e-10},
    )
    pack = _make_pack(traits=[t_frio])
    temp = compose_temperament(pack, {TraitId("t_frio"): 1.0})
    assert temp.temperature >= 1e-6
    assert temp.temperature == pytest.approx(1e-6)


def test_temperament_cotas_superiores_en_uno() -> None:
    """critical_need_threshold, switch_cost, novelty_weight y learning_rate tienen techo en 1.0."""
    t_extremo = TraitSpec(
        id=TraitId("t_extremo"),
        label="Multiplicadores gigantes",
        temperament_multipliers={
            TemperamentKnob.CRITICAL_THRESHOLD: 100.0,
            TemperamentKnob.SWITCH_COST: 100.0,
            TemperamentKnob.NOVELTY: 100.0,
            TemperamentKnob.LEARNING_RATE: 100.0,
        },
    )
    tuning = TuningSpec(
        critical_need_threshold=0.5,
        switch_cost=0.5,
        novelty_weight=0.5,
        learning_rate=0.5,
    )
    pack = _make_pack(traits=[t_extremo], tuning=tuning)
    temp = compose_temperament(pack, {TraitId("t_extremo"): 1.0})

    assert temp.critical_need_threshold <= 1.0
    assert temp.switch_cost <= 1.0
    assert temp.novelty_weight <= 1.0
    assert temp.learning_rate <= 1.0


def test_temperament_loss_aversion_sin_techo() -> None:
    """loss_aversion no tiene techo superior (permite agentes paralizados por el coste)."""
    t_avaro = TraitSpec(
        id=TraitId("t_avaro"),
        label="Avaricia",
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 10.0},
    )
    tuning = TuningSpec(loss_aversion=3.0)
    pack = _make_pack(traits=[t_avaro], tuning=tuning)
    temp = compose_temperament(pack, {TraitId("t_avaro"): 1.0})

    assert temp.loss_aversion == pytest.approx(30.0)
    assert temp.loss_aversion > 1.0


def test_perilla_a_cero_en_pack_no_resucita_por_ningun_rasgo() -> None:
    """Si el pack pone una perilla a 0.0, ningún rasgo multiplicativo la resucita (0 * M = 0)."""
    tuning = TuningSpec(
        learning_rate=0.0,
        switch_cost=0.0,
        loss_aversion=0.0,
        novelty_weight=0.0,
        critical_need_threshold=0.0,
    )
    t_grande = TraitSpec(
        id=TraitId("t_grande"),
        label="Intento de resurrección",
        temperament_multipliers={
            TemperamentKnob.LEARNING_RATE: 5.0,
            TemperamentKnob.SWITCH_COST: 5.0,
            TemperamentKnob.LOSS_AVERSION: 5.0,
            TemperamentKnob.NOVELTY: 5.0,
            TemperamentKnob.CRITICAL_THRESHOLD: 5.0,
        },
    )
    pack = _make_pack(traits=[t_grande], tuning=tuning)
    temp = compose_temperament(pack, {TraitId("t_grande"): 1.0})

    assert temp.learning_rate == 0.0
    assert temp.switch_cost == 0.0
    assert temp.loss_aversion == 0.0
    assert temp.novelty_weight == 0.0
    assert temp.critical_need_threshold == 0.0


def test_temperament_composicion_tres_capas_desgaste_y_situacion() -> None:
    """Verifica la composición completa de las 3 capas:
    mundo * rasgo^int * desgaste^strain * situación.
    """
    tuning = TuningSpec(
        loss_aversion=1.0,
        strain_multipliers={TemperamentKnob.LOSS_AVERSION: 2.0},
    )
    trait = TraitSpec(
        id=TraitId("t_cauteloso"),
        label="Cauteloso",
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 1.5},
    )
    situation = SituationSpec(
        id=SituationId("sit_alerta"),
        label="Alerta roja",
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 1.2},
    )
    pack = _make_pack(traits=[trait], situations=[situation], tuning=tuning)

    # Base: 1.0
    # Rasgo: 1.5 ** 2.0 = 2.25
    # Desgaste: 2.0 ** 0.5 ~= 1.41421356
    # Situación activa: 1.2 ** 1.0 = 1.2
    # Total: 1.0 * 2.25 * (2.0 ** 0.5) * 1.2
    esperado = 1.0 * (1.5**2.0) * (2.0**0.5) * 1.2

    temp = compose_temperament(
        pack,
        traits={TraitId("t_cauteloso"): 2.0},
        strain=0.5,
        active_situations=[SituationId("sit_alerta")],
    )
    assert temp.loss_aversion == pytest.approx(esperado)


# -----------------------------------------------------------------------------
# 10. compose_tastes
# -----------------------------------------------------------------------------


def test_compose_tastes_colapso_multiplicativo_por_objetivo() -> None:
    """Dos gustos hacia el mismo objetivo se multiplican en vez de pisarse."""
    g1 = TasteSpec(target=ActionId("act_a"), multiplier=0.5)
    g2 = TasteSpec(target=ActionId("act_a"), multiplier=1.5)
    g3 = TasteSpec(target=ActionId("act_b"), multiplier=2.0)

    tastes = compose_tastes([g1, g2, g3])
    assert tastes["act_a"] == pytest.approx(0.75)
    assert tastes["act_b"] == pytest.approx(2.0)


def test_compose_tastes_vacio() -> None:
    """Sin gustos devuelve un diccionario vacío."""
    assert compose_tastes([]) == {}
