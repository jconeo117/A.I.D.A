"""Batería de pruebas adversariales y de estrés para la Unidad U3.

Diseñada por Challenger para someter a prueba empírica:
1. Determinismo bit a bit bajo diversas semillas PYTHONHASHSEED y permutaciones aleatorias
   en el orden de inserción de rasgos, metas, situaciones y efectos de acciones.
2. Cotas y límites matemáticos extremos en `urgency` (cap <= 0.0, value > cap, value < 0.0,
   exponent = 0.0, valores gigantescos y minúsculos) y `signed_saturation`.
3. Extremos de intensidad en rasgos: neutro (0.0), inverso exacto (-1.0), identidad (1.0),
   y cancelación composicional exacta entre rasgos opuestos.
4. Fronteras y cotas estrictas de temperamento: suelo estricto de temperatura (1e-6), techos
   en 1.0 para critical_threshold, novelty_weight, switch_cost, learning_rate; ausencia de
   techo en loss_aversion; y resistencia contra resurrección de perillas en cero.
5. Convexidad multi-eje en `compute_strain`: desigualdad de Jensen en espacios multidimensionales
   y casos extremos de distribución de necesidades.
6. Aislamiento y robustez de `learn`: exclusión estricta de techo, suelo y colaterales en
   fronteras exactas (room == promised, before == spend), resolución determinista de desempate,
   y desacoplamiento de almacenes gain_beliefs y cost_beliefs.

Convención de pruebas (I-002, E-011): no se importan nombres de ontología en cabecera para
evitar interrupciones en tiempo de colección. Se accede vía `ontology.<Nombre>`.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys

import pytest

import aida.engine.disposition as disposition
import aida.engine.ontology as ontology
import aida.engine.outcome as outcome
import aida.engine.pack as pack
import aida.engine.state as state

DIVERSE_SEEDS = [
    "0",
    "1",
    "2",
    "7",
    "42",
    "137",
    "1337",
    "9999",
    "65535",
    "99999",
    "123456",
    "999999",
]


def _make_adversarial_pack(
    num_needs: int = 5,
    num_traits: int = 5,
    num_goals: int = 3,
    num_situations: int = 3,
) -> pack.ContentPack:
    """Construye un ContentPack con múltiples entidades para pruebas de estrés."""
    needs = [
        ontology.NeedSpec(
            id=ontology.NeedId(f"eje_{i}"),
            label=f"Eje abstracto {i}",
            decay_per_tick=0.01 * (i + 1),
            base_weight=1.0 + 0.2 * i,
            cap=0.6 + 0.08 * i,
            urgency_exponent=1.5 + 0.5 * (i % 3),
            strain_weight=1.0 if i % 2 == 0 else 0.5,
        )
        for i in range(num_needs)
    ]
    actions = [
        ontology.ActionSpec(
            id=ontology.ActionId(f"act_{i}"),
            label=f"Accion {i}",
            duration_ticks=1,
            reversibility=ontology.Reversibility.REVERSIBLE,
            satisfies={needs[i % num_needs].id: 0.3},
            effects={needs[i % num_needs].id: 0.3},
        )
        for i in range(num_needs)
    ]
    traits = [
        ontology.TraitSpec(
            id=ontology.TraitId(f"rasgo_{j}"),
            label=f"Rasgo {j}",
            need_weight_multipliers={
                needs[k % num_needs].id: 1.1 + 0.1 * ((j + k) % 5) for k in range(3)
            },
            gain_multipliers={
                needs[k % num_needs].id: 0.8 + 0.15 * ((j + 2 * k) % 4) for k in range(2)
            },
            cost_multipliers={needs[k % num_needs].id: 0.9 + 0.2 * ((j + k) % 3) for k in range(2)},
            cap_multipliers={needs[k % num_needs].id: 1.05 + 0.05 * (j % 3) for k in range(2)},
            temperament_multipliers={
                ontology.TemperamentKnob.LOSS_AVERSION: 1.2 + 0.1 * (j % 4),
                ontology.TemperamentKnob.TEMPERATURE: 0.9 + 0.05 * (j % 3),
                ontology.TemperamentKnob.LEARNING_RATE: 1.1 + 0.05 * (j % 2),
            },
        )
        for j in range(num_traits)
    ]
    goals = [
        ontology.GoalSpec(
            id=ontology.GoalId(f"meta_{g}"),
            label=f"Meta {g}",
            need=needs[g % num_needs].id,
            at_least=0.7,
            weight_multiplier=1.25 + 0.1 * g,
        )
        for g in range(num_goals)
    ]
    situations = [
        ontology.SituationSpec(
            id=ontology.SituationId(f"sit_{s}"),
            label=f"Situacion {s}",
            temperament_multipliers={
                ontology.TemperamentKnob.LOSS_AVERSION: 1.15 + 0.05 * s,
                ontology.TemperamentKnob.TEMPERATURE: 0.85 + 0.05 * s,
            },
        )
        for s in range(num_situations)
    ]
    return pack.ContentPack(
        id="pack_adversarial_u3",
        label="Pack para pruebas de estres U3",
        needs=tuple(needs),
        actions=tuple(actions),
        agents=(
            ontology.AgentSpec(
                id=ontology.AgentId("agente_estres"),
                label="Agente de estres",
                initial_needs={n.id: 0.5 for n in needs},
            ),
        ),
        traits=tuple(traits),
        goals=tuple(goals),
        situations=tuple(situations),
        tuning=ontology.TuningSpec(
            loss_aversion=1.5,
            temperature=0.2,
            critical_need_threshold=0.25,
            switch_cost=0.1,
            novelty_weight=0.5,
            learning_rate=0.1,
            strain_multipliers={ontology.TemperamentKnob.LOSS_AVERSION: 1.5},
        ),
    )


# ===========================================================================
# 1. Determinismo estricto entre subprocesos (PYTHONHASHSEED) y permutaciones
# ===========================================================================


def test_adversarial_determinismo_composicion_rasgos_muchos_hash_seeds() -> None:
    """Verifica que compose_* produce resultados bit a bit idénticos bajo diversas
    semillas de hash (PYTHONHASHSEED) en subprocesos independientes.
    """
    worker_code = """
import sys
import aida.engine.disposition as disposition
import aida.engine.ontology as ontology
import aida.engine.pack as pack

# Construir pack y rasgos
needs = [
    ontology.NeedSpec(
        id=ontology.NeedId(f"eje_{i}"),
        label=f"Eje {i}",
        decay_per_tick=0.01,
        base_weight=1.0 + 0.1 * i,
        cap=0.8,
        urgency_exponent=2.0,
        strain_weight=1.0,
    )
    for i in range(8)
]
actions = [
    ontology.ActionSpec(
        id=ontology.ActionId(f"act_{i}"),
        label=f"Accion {i}",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={needs[i].id: 0.2},
        effects={needs[i].id: 0.2},
    )
    for i in range(8)
]
traits_spec = [
    ontology.TraitSpec(
        id=ontology.TraitId(f"t_{j}"),
        label=f"Rasgo {j}",
        need_weight_multipliers={needs[k].id: 1.05 + 0.05 * ((j + k) % 5) for k in range(8)},
        gain_multipliers={needs[k].id: 0.95 + 0.03 * ((j + k) % 4) for k in range(8)},
        cost_multipliers={needs[k].id: 1.02 + 0.02 * ((j + k) % 3) for k in range(8)},
        cap_multipliers={needs[k].id: 1.01 + 0.01 * (j % 4) for k in range(8)},
        temperament_multipliers={ontology.TemperamentKnob.LOSS_AVERSION: 1.1 + 0.05 * j},
    )
    for j in range(12)
]
content_pack = pack.ContentPack(
    id="pack_hash_test",
    label="Pack hash",
    needs=tuple(needs),
    actions=tuple(actions),
    agents=(
        ontology.AgentSpec(
            id=ontology.AgentId("ag"),
            label="Ag",
            initial_needs={n.id: 0.5 for n in needs},
        ),
    ),
    traits=tuple(traits_spec),
    tuning=ontology.TuningSpec(),
)

traits_map = {ontology.TraitId(f"t_{j}"): 0.3 * ((-1) ** j) for j in range(12)}
weights = disposition.compose_need_weights(content_pack, traits_map)
gains = disposition.compose_gains(content_pack, traits_map)
costs = disposition.compose_costs(content_pack, traits_map)
caps = disposition.compose_caps(content_pack, traits_map)
temp = disposition.compose_temperament(content_pack, traits_map)

# Serializar valores en formato float.hex para comparar bit a bit exacto
rep = [
    [k, weights[k].hex()] for k in sorted(weights)
] + [
    [k, gains[k].hex()] for k in sorted(gains)
] + [
    [k, costs[k].hex()] for k in sorted(costs)
] + [
    [k, caps[k].hex()] for k in sorted(caps)
] + [
    ["temp_loss_av", temp.loss_aversion.hex()]
]
sys.stdout.write(str(rep))
"""
    results: set[str] = set()
    for seed in DIVERSE_SEEDS:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        res = subprocess.run(
            [sys.executable, "-c", worker_code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        results.add(res.stdout)

    assert len(results) == 1, (
        f"Composición de rasgos divergió entre semillas: {len(results)} salidas distintas"
    )


def test_adversarial_permutaciones_orden_insercion_rasgos_invarianza_hex() -> None:
    """Prueba 50 permutaciones aleatorias del orden de claves en el diccionario de rasgos
    y verifica que el valor en coma flotante es idéntico bit a bit (hex string idéntico).
    """
    cp = _make_adversarial_pack(num_needs=6, num_traits=10)
    base_trait_items = [(t.id, 0.25 * ((i % 4) - 1.5)) for i, t in enumerate(cp.traits)]

    # Cálculo de referencia
    ref_dict = dict(base_trait_items)
    ref_weights = disposition.compose_need_weights(cp, ref_dict)
    ref_gains = disposition.compose_gains(cp, ref_dict)
    ref_costs = disposition.compose_costs(cp, ref_dict)
    ref_caps = disposition.compose_caps(cp, ref_dict)
    ref_temp = disposition.compose_temperament(cp, ref_dict)

    rng = random.Random(42)
    for _ in range(50):
        permuted_items = list(base_trait_items)
        rng.shuffle(permuted_items)
        perm_dict = dict(permuted_items)

        w = disposition.compose_need_weights(cp, perm_dict)
        g = disposition.compose_gains(cp, perm_dict)
        c = disposition.compose_costs(cp, perm_dict)
        caps = disposition.compose_caps(cp, perm_dict)
        temp = disposition.compose_temperament(cp, perm_dict)

        for nid in cp.needs:
            assert w[nid.id].hex() == ref_weights[nid.id].hex()
            assert g[nid.id].hex() == ref_gains[nid.id].hex()
            assert c[nid.id].hex() == ref_costs[nid.id].hex()
            assert caps[nid.id].hex() == ref_caps[nid.id].hex()

        assert temp.loss_aversion.hex() == ref_temp.loss_aversion.hex()
        assert temp.temperature.hex() == ref_temp.temperature.hex()


def test_adversarial_permutaciones_orden_metas_y_situaciones() -> None:
    """Verifica que permutaciones de metas y situaciones producen resultados bit a bit exactos."""
    cp = _make_adversarial_pack(num_needs=4, num_goals=6, num_situations=6)
    ref_goals = list(cp.goals)
    ref_sits = [s.id for s in cp.situations]

    ref_w = disposition.compose_need_weights(cp, {}, pursuing=ref_goals)
    ref_temp = disposition.compose_temperament(cp, {}, active_situations=ref_sits)

    rng = random.Random(1337)
    for _ in range(30):
        shuffled_goals = list(ref_goals)
        rng.shuffle(shuffled_goals)
        w = disposition.compose_need_weights(cp, {}, pursuing=shuffled_goals)
        for nid in cp.needs:
            assert w[nid.id].hex() == ref_w[nid.id].hex()

        shuffled_sits = list(ref_sits)
        rng.shuffle(shuffled_sits)
        temp = disposition.compose_temperament(cp, {}, active_situations=shuffled_sits)
        assert temp.loss_aversion.hex() == ref_temp.loss_aversion.hex()


def test_adversarial_determinismo_learn_muchos_hash_seeds() -> None:
    """outcome.learn produce idéntica sorpresa y mutaciones bajo múltiples semillas hash."""
    worker_code = """
import sys
import aida.engine.ontology as ontology
import aida.engine.outcome as outcome

action = ontology.ActionSpec(
    id=ontology.ActionId("act_det"),
    label="Accion determinista",
    duration_ticks=1,
    reversibility=ontology.Reversibility.REVERSIBLE,
    satisfies={ontology.NeedId(f"n_sat_{i}"): 0.2 + 0.05 * i for i in range(6)},
    costs={ontology.NeedId(f"n_cst_{i}"): 0.15 + 0.05 * i for i in range(6)},
    effects={
        **{ontology.NeedId(f"n_sat_{i}"): 0.2 + 0.05 * i for i in range(6)},
        **{ontology.NeedId(f"n_cst_{i}"): -(0.15 + 0.05 * i) for i in range(6)},
    },
)
gain_beliefs = {ontology.NeedId(f"n_sat_{i}"): 1.0 for i in range(6)}
cost_beliefs = {ontology.NeedId(f"n_cst_{i}"): 1.0 for i in range(6)}
before = {
    **{ontology.NeedId(f"n_sat_{i}"): 0.2 for i in range(6)},
    **{ontology.NeedId(f"n_cst_{i}"): 0.8 for i in range(6)},
}
received = {
    **{ontology.NeedId(f"n_sat_{i}"): 0.25 + 0.04 * i for i in range(6)},
    **{ontology.NeedId(f"n_cst_{i}"): -(0.2 + 0.03 * i) for i in range(6)},
}
sorpresa = outcome.learn(gain_beliefs, cost_beliefs, action, received, before, rate=0.2)
out = [
    sorpresa.hex(),
    [[k, gain_beliefs[k].hex()] for k in sorted(gain_beliefs)],
    [[k, cost_beliefs[k].hex()] for k in sorted(cost_beliefs)],
]
sys.stdout.write(str(out))
"""
    results: set[str] = set()
    for seed in DIVERSE_SEEDS:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        res = subprocess.run(
            [sys.executable, "-c", worker_code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        results.add(res.stdout)

    assert len(results) == 1, (
        f"outcome.learn divergió entre semillas PYTHONHASHSEED: {len(results)} salidas distintas"
    )


# ===========================================================================
# 2. Cotas y límites matemáticos extremos (`urgency`, `signed_saturation`)
# ===========================================================================


def test_adversarial_urgency_cap_menor_o_igual_a_cero() -> None:
    """urgency con cap <= 0.0 devuelve estrictamente 0.0 para cualquier valor y exponente."""
    for cap_val in [0.0, -0.0, -1e-15, -1.0, -100.0, -1e9]:
        for val in [-1.0, 0.0, 0.5, 1.0, 10.0]:
            for exp in [0.5, 1.0, 2.0, 5.0]:
                res = disposition.urgency(val, exp, cap=cap_val)
                assert res == 0.0
                assert isinstance(res, float)


def test_adversarial_urgency_valor_mayor_que_cap() -> None:
    """urgency con valor superior al cap devuelve estrictamente 0.0."""
    cap = 0.8
    for delta in [1e-12, 0.01, 0.5, 10.0, 1e6]:
        res = disposition.urgency(cap + delta, exponent=2.0, cap=cap)
        assert res == 0.0
        assert math.copysign(1.0, res) == 1.0  # Asegurar no -0.0 extraño


def test_adversarial_urgency_valor_negativo() -> None:
    """urgency con valor negativo (eje por debajo de 0) calcula (1 - val/cap)**exp sin reventar."""
    cap = 1.0
    val = -0.5
    exp = 2.0
    # (1 - (-0.5)/1.0) = 1.5, 1.5 ** 2.0 = 2.25
    assert disposition.urgency(val, exp, cap) == pytest.approx(2.25)


def test_adversarial_urgency_exponente_cero() -> None:
    """urgency con exponente 0.0: para cualquier base positiva da 1.0, y con cap <= 0.0 da 0.0."""
    # Base positiva
    assert disposition.urgency(0.5, exponent=0.0, cap=1.0) == 1.0
    # En cap <= 0 devuelve 0.0
    assert disposition.urgency(0.5, exponent=0.0, cap=0.0) == 0.0
    assert disposition.urgency(0.5, exponent=0.0, cap=-1.0) == 0.0


def test_adversarial_urgency_exponentes_extremos() -> None:
    """urgency con exponentes muy grandes o muy pequeños."""
    cap = 1.0
    val = 0.5
    # Exponente muy grande: (0.5)**100.0 -> extremadamente cercano a 0 pero positivo
    res_large = disposition.urgency(val, exponent=100.0, cap=cap)
    assert 0.0 <= res_large < 1e-20

    # Exponente minúsculo: (0.5)**1e-6 -> cercano a 1.0
    res_small = disposition.urgency(val, exponent=1e-6, cap=cap)
    assert pytest.approx(res_small, rel=1e-4) == 1.0


def test_adversarial_signed_saturation_asymptotics_near_limits() -> None:
    """signed_saturation se aproxima asintóticamente a +1 y -1 sin excederlos."""
    # Dentro de la resolución de mantisa (mag < 2**53): estrictamente < 1.0 y > -1.0
    for mag in [1e2, 1e4, 1e8, 1e12, 1e15]:
        val_pos = outcome.signed_saturation(mag, scale=1.0)
        val_neg = outcome.signed_saturation(-mag, scale=1.0)
        assert 0.0 < val_pos < 1.0
        assert -1.0 < val_neg < 0.0
        assert val_pos == pytest.approx(-val_neg)

    # Para magnitudes extremas (mag >= 2**53), la suma mag + 1.0 satura numéricamente
    # a mag en float64, alcanzando exactamente 1.0 y -1.0 sin jamás excederlos ni generar NaN.
    for mag in [1e16, 1e50, 1e200]:
        val_pos = outcome.signed_saturation(mag, scale=1.0)
        val_neg = outcome.signed_saturation(-mag, scale=1.0)
        assert 0.0 < val_pos <= 1.0
        assert -1.0 <= val_neg < 0.0
        assert not math.isnan(val_pos)
        assert not math.isnan(val_neg)


# ===========================================================================
# 3. Extremos de intensidad en rasgos
# ===========================================================================


def test_adversarial_intensidad_extremos_cancelacion_exacta() -> None:
    """Dos rasgos con multiplicadores recíprocos o con intensidades +1.0 y -1.0
    se cancelan mutuamente devolviendo el valor neutro 1.0 (dentro de precisión flotante).
    """
    n1 = ontology.NeedId("n1")
    t_pos = ontology.TraitSpec(
        id=ontology.TraitId("t_pos"),
        label="Multiplicador positivo",
        gain_multipliers={n1: 2.5},
        cost_multipliers={n1: 3.2},
    )
    t_neg = ontology.TraitSpec(
        id=ontology.TraitId("t_neg"),
        label="Multiplicador a cancelar",
        gain_multipliers={n1: 2.5},
        cost_multipliers={n1: 3.2},
    )
    pack_inst = pack.ContentPack(
        id="pack_cancel",
        label="Pack cancelacion",
        needs=(
            ontology.NeedSpec(id=n1, label="N1", decay_per_tick=0.01, base_weight=1.0, cap=1.0),
        ),
        actions=(
            ontology.ActionSpec(
                id=ontology.ActionId("act"),
                label="Act",
                duration_ticks=1,
                reversibility=ontology.Reversibility.REVERSIBLE,
                satisfies={n1: 0.1},
                effects={n1: 0.1},
            ),
        ),
        agents=(
            ontology.AgentSpec(id=ontology.AgentId("ag"), label="Ag", initial_needs={n1: 0.5}),
        ),
        traits=(t_pos, t_neg),
        tuning=ontology.TuningSpec(),
    )

    # Con t_pos a +1.0 y t_neg a -1.0: 2.5**1 * 2.5**-1 = 1.0
    traits = {ontology.TraitId("t_pos"): 1.0, ontology.TraitId("t_neg"): -1.0}
    gains = disposition.compose_gains(pack_inst, traits)
    costs = disposition.compose_costs(pack_inst, traits)

    assert gains[n1] == pytest.approx(1.0, abs=1e-14)
    assert costs[n1] == pytest.approx(1.0, abs=1e-14)


def test_adversarial_drift_trait_acotacion_y_recomposicion() -> None:
    """Verifica que AgentState.drift_trait satura correctamente en [-1.0, 1.0]
    sin desbordar ante deltas arbitrariamente grandes.
    """
    cp = _make_adversarial_pack(num_needs=2, num_traits=1)
    trait_id = cp.traits[0].id
    ag = state.AgentState(agent_id=ontology.AgentId("ag"), needs={n.id: 0.5 for n in cp.needs})

    # Empieza en 0.0 tras recompose
    ag.recompose(cp)
    assert ag.traits.get(trait_id, 0.0) == 0.0

    # Intento de deriva extremo positivo: +1000.0
    ag.drift_trait(cp, trait_id, 1000.0)
    assert ag.traits[trait_id] == 1.0
    assert ag.gains[cp.needs[0].id] == pytest.approx(
        cp.traits[0].gain_multipliers.get(cp.needs[0].id, 1.0) ** 1.0
    )

    # Intento de deriva extremo negativo: -2000.0
    ag.drift_trait(cp, trait_id, -2000.0)
    assert ag.traits[trait_id] == -1.0
    assert ag.gains[cp.needs[0].id] == pytest.approx(
        cp.traits[0].gain_multipliers.get(cp.needs[0].id, 1.0) ** -1.0
    )


# ===========================================================================
# 4. Fronteras y cotas estrictas de temperamento
# ===========================================================================


def test_adversarial_temperament_suelo_temperatura_multiplicadores_minusculos() -> None:
    """La temperatura jamás baja de 1e-6 incluso ante productos extremos de factores < 1."""
    t_ultra_frio = ontology.TraitSpec(
        id=ontology.TraitId("t_ultra_frio"),
        label="Ultra frio",
        temperament_multipliers={ontology.TemperamentKnob.TEMPERATURE: 1e-15},
    )
    sit_fria = ontology.SituationSpec(
        id=ontology.SituationId("sit_fria"),
        label="Sit ultra fria",
        temperament_multipliers={ontology.TemperamentKnob.TEMPERATURE: 1e-15},
    )
    tuning = ontology.TuningSpec(
        temperature=1e-5,
        strain_multipliers={ontology.TemperamentKnob.TEMPERATURE: 1e-15},
    )
    cp = _make_adversarial_pack(num_needs=2, num_traits=0, num_situations=0)
    cp = pack.ContentPack(
        id=cp.id,
        label=cp.label,
        needs=cp.needs,
        actions=cp.actions,
        agents=cp.agents,
        traits=(t_ultra_frio,),
        situations=(sit_fria,),
        tuning=tuning,
    )

    temp = disposition.compose_temperament(
        cp,
        traits={ontology.TraitId("t_ultra_frio"): 1.0},
        strain=1.0,
        active_situations=[ontology.SituationId("sit_fria")],
    )

    assert temp.temperature >= 1e-6
    assert temp.temperature == 1e-6


def test_adversarial_temperament_techos_estrictos_a_uno() -> None:
    """Las 4 perillas acotadas superiormente se mantienen estrictamente <= 1.0
    frente a factores masivos provenientes de rasgos, tuning, desgaste y situaciones.
    """
    t_masivo = ontology.TraitSpec(
        id=ontology.TraitId("t_masivo"),
        label="Factores masivos",
        temperament_multipliers={
            ontology.TemperamentKnob.CRITICAL_THRESHOLD: 100.0,
            ontology.TemperamentKnob.SWITCH_COST: 100.0,
            ontology.TemperamentKnob.NOVELTY: 100.0,
            ontology.TemperamentKnob.LEARNING_RATE: 100.0,
        },
    )
    sit_masiva = ontology.SituationSpec(
        id=ontology.SituationId("sit_masiva"),
        label="Sit masiva",
        temperament_multipliers={
            ontology.TemperamentKnob.CRITICAL_THRESHOLD: 50.0,
            ontology.TemperamentKnob.SWITCH_COST: 50.0,
            ontology.TemperamentKnob.NOVELTY: 50.0,
            ontology.TemperamentKnob.LEARNING_RATE: 50.0,
        },
    )
    tuning = ontology.TuningSpec(
        critical_need_threshold=0.9,
        switch_cost=0.9,
        novelty_weight=0.9,
        learning_rate=0.9,
        strain_multipliers={
            ontology.TemperamentKnob.CRITICAL_THRESHOLD: 10.0,
            ontology.TemperamentKnob.SWITCH_COST: 10.0,
            ontology.TemperamentKnob.NOVELTY: 10.0,
            ontology.TemperamentKnob.LEARNING_RATE: 10.0,
        },
    )
    cp = pack.ContentPack(
        id="pack_masivo",
        label="Pack masivo",
        needs=(),
        actions=(),
        agents=(ontology.AgentSpec(id=ontology.AgentId("ag"), label="Ag"),),
        traits=(t_masivo,),
        situations=(sit_masiva,),
        tuning=tuning,
    )

    temp = disposition.compose_temperament(
        cp,
        traits={ontology.TraitId("t_masivo"): 1.0},
        strain=1.0,
        active_situations=[ontology.SituationId("sit_masiva")],
    )

    assert temp.critical_need_threshold == 1.0
    assert temp.switch_cost == 1.0
    assert temp.novelty_weight == 1.0
    assert temp.learning_rate == 1.0


def test_adversarial_temperament_loss_aversion_crecimiento_sin_cota() -> None:
    """loss_aversion crece libremente sin techo (10.0, 100.0, 1000.0) según especificación."""
    t_avaro = ontology.TraitSpec(
        id=ontology.TraitId("t_avaro"),
        label="Avaricia",
        temperament_multipliers={ontology.TemperamentKnob.LOSS_AVERSION: 10.0},
    )
    tuning = ontology.TuningSpec(
        loss_aversion=5.0,
        strain_multipliers={ontology.TemperamentKnob.LOSS_AVERSION: 2.0},
    )
    cp = pack.ContentPack(
        id="pack_avaro",
        label="Pack avaro",
        needs=(),
        actions=(),
        agents=(ontology.AgentSpec(id=ontology.AgentId("ag"), label="Ag"),),
        traits=(t_avaro,),
        tuning=tuning,
    )

    # Base 5.0 * (10.0 ** 2.0 = 100.0) * (2.0 ** 1.0 = 2.0) = 1000.0
    temp = disposition.compose_temperament(
        cp,
        traits={ontology.TraitId("t_avaro"): 2.0},
        strain=1.0,
    )
    assert temp.loss_aversion == pytest.approx(1000.0)


def test_adversarial_perilla_cero_inmune_a_resurreccion() -> None:
    """Si el pack calibra perillas a cero, ningún rasgo, desgaste o situación las levanta."""
    tuning = ontology.TuningSpec(
        learning_rate=0.0,
        switch_cost=0.0,
        loss_aversion=0.0,
        novelty_weight=0.0,
        critical_need_threshold=0.0,
        strain_multipliers={
            ontology.TemperamentKnob.LEARNING_RATE: 10.0,
            ontology.TemperamentKnob.SWITCH_COST: 10.0,
            ontology.TemperamentKnob.LOSS_AVERSION: 10.0,
            ontology.TemperamentKnob.NOVELTY: 10.0,
            ontology.TemperamentKnob.CRITICAL_THRESHOLD: 10.0,
        },
    )
    trait = ontology.TraitSpec(
        id=ontology.TraitId("t_res"),
        label="Resucitar",
        temperament_multipliers={
            ontology.TemperamentKnob.LEARNING_RATE: 10.0,
            ontology.TemperamentKnob.SWITCH_COST: 10.0,
            ontology.TemperamentKnob.LOSS_AVERSION: 10.0,
            ontology.TemperamentKnob.NOVELTY: 10.0,
            ontology.TemperamentKnob.CRITICAL_THRESHOLD: 10.0,
        },
    )
    sit = ontology.SituationSpec(
        id=ontology.SituationId("sit_res"),
        label="Sit resucitar",
        temperament_multipliers={
            ontology.TemperamentKnob.LEARNING_RATE: 10.0,
            ontology.TemperamentKnob.SWITCH_COST: 10.0,
            ontology.TemperamentKnob.LOSS_AVERSION: 10.0,
            ontology.TemperamentKnob.NOVELTY: 10.0,
            ontology.TemperamentKnob.CRITICAL_THRESHOLD: 10.0,
        },
    )
    cp = pack.ContentPack(
        id="pack_cero",
        label="Pack ceros",
        needs=(),
        actions=(),
        agents=(ontology.AgentSpec(id=ontology.AgentId("ag"), label="Ag"),),
        traits=(trait,),
        situations=(sit,),
        tuning=tuning,
    )

    temp = disposition.compose_temperament(
        cp,
        traits={ontology.TraitId("t_res"): 1.0},
        strain=1.0,
        active_situations=[ontology.SituationId("sit_res")],
    )

    assert temp.learning_rate == 0.0
    assert temp.switch_cost == 0.0
    assert temp.loss_aversion == 0.0
    assert temp.novelty_weight == 0.0
    assert temp.critical_need_threshold == 0.0


# ===========================================================================
# 5. Convexidad multi-eje en `compute_strain`
# ===========================================================================


def test_adversarial_compute_strain_desigualdad_de_jensen() -> None:
    """Verifica empíricamente la convexidad de compute_strain:
    Para cualesquiera estados u y v en [0, 1]^N y lambda in (0, 1):
        strain(lambda * u + (1 - lambda) * v) <= lambda * strain(u) + (1 - lambda) * strain(v)
    Dado que urgency_exponent >= 1.0 para cada eje desgastante.
    """
    cp = _make_adversarial_pack(num_needs=6)
    need_ids = [n.id for n in cp.needs]

    rng = random.Random(999)
    for _ in range(40):
        # Generar dos estados de necesidades u y v
        u = {nid: rng.uniform(0.0, 1.0) for nid in need_ids}
        v = {nid: rng.uniform(0.0, 1.0) for nid in need_ids}
        lam = rng.uniform(0.1, 0.9)

        comb = {nid: lam * u[nid] + (1.0 - lam) * v[nid] for nid in need_ids}

        strain_comb = disposition.compute_strain(cp, comb)
        strain_u = disposition.compute_strain(cp, u)
        strain_v = disposition.compute_strain(cp, v)

        expected_bound = lam * strain_u + (1.0 - lam) * strain_v
        # Permitir tolerancia mínima por imprecisión numérica
        assert strain_comb <= expected_bound + 1e-12, (
            f"Fallo de convexidad: {strain_comb} > {expected_bound} (u={u}, v={v}, lam={lam})"
        )


def test_adversarial_compute_strain_casos_borde_extremos() -> None:
    """Verifica compute_strain en configuraciones de necesidades en el borde."""
    cp = _make_adversarial_pack(num_needs=4)
    need_ids = [n.id for n in cp.needs]

    # Todos al tope -> 0.0
    assert disposition.compute_strain(cp, {nid: 1.0 for nid in need_ids}) == pytest.approx(0.0)

    # Todos a cero -> 1.0
    assert disposition.compute_strain(cp, {nid: 0.0 for nid in need_ids}) == pytest.approx(1.0)

    # Pack sin ejes con strain_weight > 0 -> 0.0
    cp_no_strain = pack.ContentPack(
        id="pack_nostrain",
        label="No strain",
        needs=tuple(
            ontology.NeedSpec(
                id=ontology.NeedId(f"n_{i}"),
                label=f"N {i}",
                decay_per_tick=0.01,
                strain_weight=0.0,
            )
            for i in range(3)
        ),
        actions=(),
        agents=(ontology.AgentSpec(id=ontology.AgentId("ag"), label="Ag"),),
        tuning=ontology.TuningSpec(),
    )
    assert (
        disposition.compute_strain(cp_no_strain, {ontology.NeedId(f"n_{i}"): 0.0 for i in range(3)})
        == 0.0
    )


# ===========================================================================
# 6. Aprendizaje y sorpresa en `outcome.learn`
# ===========================================================================


def test_adversarial_learn_fronteras_exactas_de_exclusion() -> None:
    """Comprueba las fronteras exactas de las exclusiones:
    - room == promised debe APRENDER (el efecto cabe justo entero).
    - room < promised debe EXCLUIR (recorte por techo).
    - before == spend debe APRENDER (se puede pagar justo el coste).
    - before < spend debe EXCLUIR (recorte por suelo).
    """
    n_sat = ontology.NeedId("n_sat")
    n_cst = ontology.NeedId("n_cst")
    action = ontology.ActionSpec(
        id=ontology.ActionId("act_border"),
        label="Accion frontera",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={n_sat: 0.3},
        costs={n_cst: 0.4},
        effects={n_sat: 0.3, n_cst: -0.4},
    )

    # Caso 1a: room == promised (before = 0.7 => room = 0.3 == promised 0.3) -> APRENDE
    gb1: dict[ontology.NeedId, float] = {n_sat: 1.0}
    cb1: dict[ontology.NeedId, float] = {n_cst: 1.0}
    # Solo testeamos sat, poniendo cst sin saldo para que cst sea ignorado
    s1 = outcome.learn(
        gb1,
        cb1,
        action,
        received={n_sat: 0.45, n_cst: -0.1},
        before={n_sat: 0.7, n_cst: 0.2},  # cst excluido (before 0.2 < spend 0.4)
        rate=0.5,
    )
    assert gb1[n_sat] == pytest.approx(1.0 + 0.5 * (1.5 - 1.0))
    assert s1 == pytest.approx(0.5)

    # Caso 1b: room < promised (before = 0.70001 => room = 0.29999 < 0.3) -> EXCLUIDO
    gb2: dict[ontology.NeedId, float] = {n_sat: 1.0}
    cb2: dict[ontology.NeedId, float] = {n_cst: 1.0}
    s2 = outcome.learn(
        gb2,
        cb2,
        action,
        received={n_sat: 0.45, n_cst: -0.1},
        before={n_sat: 0.71, n_cst: 0.2},
        rate=0.5,
    )
    assert gb2[n_sat] == 1.0  # Sin mutación
    assert s2 == 0.0

    # Caso 2a: before == spend (before = 0.4 == spend 0.4) -> APRENDE
    gb3: dict[ontology.NeedId, float] = {n_sat: 1.0}
    cb3: dict[ontology.NeedId, float] = {n_cst: 1.0}
    # Sat excluido (before = 0.9 => room = 0.1 < 0.3)
    s3 = outcome.learn(
        gb3,
        cb3,
        action,
        received={n_sat: 0.1, n_cst: -0.6},
        before={n_sat: 0.9, n_cst: 0.4},
        rate=0.5,
    )
    # realized = 0.6 / 0.4 = 1.5, error = +0.5, cb = 1.0 + 0.5 * 0.5 = 1.25
    assert cb3[n_cst] == pytest.approx(1.25)
    # Sorpresa normalizada para costes: +0.5 * (-1.0) = -0.5
    assert s3 == pytest.approx(-0.5)

    # Caso 2b: before < spend (before = 0.39 < spend 0.4) -> EXCLUIDO
    gb4: dict[ontology.NeedId, float] = {n_sat: 1.0}
    cb4: dict[ontology.NeedId, float] = {n_cst: 1.0}
    s4 = outcome.learn(
        gb4,
        cb4,
        action,
        received={n_sat: 0.1, n_cst: -0.6},
        before={n_sat: 0.9, n_cst: 0.39},
        rate=0.5,
    )
    assert cb4[n_cst] == 1.0
    assert s4 == 0.0


def test_adversarial_learn_desempate_determinista_sorpresa() -> None:
    """Si una ganancia y un coste producen idéntico error absoluto,
    el desempate se resuelve deterministamente por el primer eje en sorted(action.effects),
    fijando el signo de la sorpresa independientemente del orden de inserción de las claves (I-019).
    """
    n_a = ontology.NeedId("n_alpha")
    n_b = ontology.NeedId("n_beta")

    # n_a es ganancia (+1.0), n_b es coste (-1.0).
    # Con sorted(effects), n_a < n_b, gana la ganancia => sorpresa +0.5.
    # Se pasa con effects insertando n_b primero.
    action_ab = ontology.ActionSpec(
        id=ontology.ActionId("act_ab"),
        label="Accion empate AB",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={n_a: 0.5},
        costs={n_b: 0.5},
        effects={n_b: -0.5, n_a: 0.5},
    )
    gb1: dict[ontology.NeedId, float] = {n_a: 1.0}
    cb1: dict[ontology.NeedId, float] = {n_b: 1.0}
    before = {n_a: 0.2, n_b: 0.8}
    received = {n_a: 0.75, n_b: -0.75}

    sorpresa1 = outcome.learn(gb1, cb1, action_ab, received, before, rate=0.1)
    assert sorpresa1 == pytest.approx(0.5)

    # Invertimos roles: n_a es coste (-1.0), n_b es ganancia (+1.0).
    # Con sorted(effects), n_a < n_b, gana el coste => sorpresa -0.5.
    # Se pasa con effects insertando n_b primero.
    action_ba = ontology.ActionSpec(
        id=ontology.ActionId("act_ba"),
        label="Accion empate BA",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={n_b: 0.5},
        costs={n_a: 0.5},
        effects={n_b: 0.5, n_a: -0.5},
    )
    gb2: dict[ontology.NeedId, float] = {n_b: 1.0}
    cb2: dict[ontology.NeedId, float] = {n_a: 1.0}
    before2 = {n_a: 0.8, n_b: 0.2}
    received2 = {n_a: -0.75, n_b: 0.75}

    sorpresa2 = outcome.learn(gb2, cb2, action_ba, received2, before2, rate=0.1)
    assert sorpresa2 == pytest.approx(-0.5)


def test_adversarial_metas_multiplicativas_valor_absoluto_y_no_suma() -> None:
    """Verifica que las metas modulan pesos multiplicativamente y no aditivamente
    bajo múltiples configuraciones (I-015).
    """
    pack = _make_adversarial_pack(num_needs=1, num_traits=0, num_goals=2)
    w_sin = disposition.compose_need_weights(pack, {}, pursuing=[])
    w_con = disposition.compose_need_weights(pack, {}, pursuing=list(pack.goals))
    for g in pack.goals:
        assert w_con[g.need] > 0.0
        # Multiplicación estricta frente a suma
        assert w_con[g.need] != pytest.approx(w_sin[g.need] + g.weight_multiplier)


def test_adversarial_learn_rate_cero_o_negativo_no_toca_nada() -> None:
    """Tasa de aprendizaje <= 0.0 devuelve 0.0 y no muta los diccionarios."""
    n1 = ontology.NeedId("n1")
    action = ontology.ActionSpec(
        id=ontology.ActionId("act_zero"),
        label="Accion",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        satisfies={n1: 0.5},
        effects={n1: 0.5},
    )
    for r in [0.0, -0.0, -0.1, -1.0]:
        gb: dict[ontology.NeedId, float] = {n1: 1.0}
        cb: dict[ontology.NeedId, float] = {}
        s = outcome.learn(gb, cb, action, received={n1: 1.0}, before={n1: 0.2}, rate=r)
        assert s == 0.0
        assert gb == {n1: 1.0}
        assert cb == {}
