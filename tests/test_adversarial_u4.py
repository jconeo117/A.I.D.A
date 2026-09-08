"""Batería de pruebas adversariales y de estrés para la Unidad U4.

Diseñada por Challenger 1 para someter a prueba empírica:
1. Determinismo entre procesos: salidas byte a byte idénticas bajo diversas
   semillas PYTHONHASHSEED tanto para la CLI `--explain` como para la
   serialización JSONL de `DecisionTrace.to_json()`.
2. Estabilidad de la clave de flujo RNG:
   - Identidad (semilla, agente, tick) produce idénticos resultados.
   - Variación de agente o tick altera el flujo de manera reproducible.
   - Resistencia a colisiones por concatenación mediante separador ASCII.
3. Estabilidad numérica de softmax:
   - Desafío con utilidades extremas (diferencia > 1000, > 50000, 1e20, 1e100, 1e300).
   - Ausencia absoluta de `OverflowError` en `math.exp`.
   - Garantía de que la suma de pesos es finita y estrictamente positiva.
4. Desempate estricto e independiente del orden del pack:
   - Múltiples acciones candidatas con utilidades idénticas bit a bit.
   - Invarianza total frente a 50 permutaciones de orden de inserción en ContentPack.
5. Casos frontera y condiciones de borde:
   - `must_address` apuntando a eje inexistente o imposible -> strictly None sin fallback.
   - `critical_needs` con todas las necesidades por encima del umbral -> lista vacía.
   - `critical_needs` con `urgent_when_low = False` -> excluido siempre.
   - `critical_needs` en frontera exacta (igualdad estricta al umbral).
   - Triaje de `critical_need` durante ejecución de acción en curso.
   - Estado no recompuesto (`agent.temperament = None`) en `decide`, `candidates` y
     `critical_needs` -> `UncomposedAgentStateError`.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import aida.agents.decide as decide_mod
import aida.engine.explain as explain_mod
import aida.engine.ontology as ontology
import aida.engine.pack as pack_mod
import aida.engine.rng as rng_mod
import aida.engine.state as state_mod

TEMPLATE_PACK_PATH = Path(__file__).resolve().parents[1] / "packs" / "_plantilla" / "pack.yaml"

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
    num_actions: int = 5,
    candidate_pool: int = 3,
    temperature: float = 0.15,
    actions_order: list[int] | None = None,
) -> pack_mod.ContentPack:
    """Crea un ContentPack controlado para pruebas de estrés."""
    needs = [
        ontology.NeedSpec(
            id=ontology.NeedId("eje_primario"),
            label="Eje Primario",
            initial=0.5,
            urgency_exponent=1.0,
            decay_per_tick=0.01,
            urgent_when_low=True,
        ),
        ontology.NeedSpec(
            id=ontology.NeedId("eje_secundario"),
            label="Eje Secundario",
            initial=0.5,
            urgency_exponent=1.0,
            decay_per_tick=0.01,
            urgent_when_low=False,
        ),
    ]

    indices = list(range(num_actions)) if actions_order is None else actions_order
    actions = [
        ontology.ActionSpec(
            id=ontology.ActionId(f"accion_{i:02d}"),
            label=f"Accion {i}",
            duration_ticks=2,
            reversibility=ontology.Reversibility.REVERSIBLE,
            satisfies={ontology.NeedId("eje_primario"): 0.2},
            effects={ontology.NeedId("eje_primario"): 0.2},
        )
        for i in indices
    ]

    agent_spec = ontology.AgentSpec(
        id=ontology.AgentId("agente_prueba"),
        label="Agente de Prueba",
        initial_needs={
            ontology.NeedId("eje_primario"): 0.5,
            ontology.NeedId("eje_secundario"): 0.5,
        },
        traits={},
    )

    tuning = ontology.TuningSpec(candidate_pool=candidate_pool, temperature=temperature)

    return pack_mod.ContentPack(
        id="pack_adversarial",
        label="Pack Adversarial",
        needs=needs,
        actions=actions,
        agents=[agent_spec],
        tuning=tuning,
    )


# =============================================================================
# 1. Determinismo entre procesos (PYTHONHASHSEED)
# =============================================================================


class TestProcessDeterminismAcrossHashSeeds:
    """Verifica que la salida de deliberación y traza sea byte a byte idéntica."""

    def test_cli_explain_byte_identical_across_12_seeds(self) -> None:
        """La CLI `aida --pack ... --explain` produce la misma salida con cualquier seed."""
        outputs: list[str] = []
        for seed in DIVERSE_SEEDS:
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            cmd = [
                sys.executable,
                "-m",
                "aida.run",
                "--pack",
                str(TEMPLATE_PACK_PATH),
                "--explain",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            outputs.append(result.stdout)

        first_output = outputs[0]
        for i, out in enumerate(outputs[1:], start=1):
            assert out == first_output, (
                f"Salida CLI con seed={DIVERSE_SEEDS[i]} difiere de {DIVERSE_SEEDS[0]}"
            )

    def test_jsonl_trace_byte_identical_across_12_seeds(self) -> None:
        """La serialización `DecisionTrace.to_json()` es byte a byte idéntica."""
        outputs: list[str] = []
        code = (
            "import pathlib; "
            "from aida.engine.pack import load_pack; "
            "from aida.engine.state import AgentState, ExecutionState; "
            "from aida.engine.explain import explain_deliberation; "
            f"pack = load_pack(pathlib.Path({str(TEMPLATE_PACK_PATH)!r})); "
            "ag_spec = pack.agents[0]; "
            "needs = {n.id: ag_spec.initial_needs.get(n.id, n.initial) for n in pack.needs}; "
            "agent = AgentState("
            "    agent_id=ag_spec.id, needs=needs, traits=dict(ag_spec.traits), goals={}"
            "); "
            "agent.recompose(pack); "
            "exec_st = ExecutionState(agent=agent, tick=0); "
            "d, tr = explain_deliberation(exec_st, pack); "
            "assert tr is not None; "
            "print(tr.to_json())"
        )
        for seed in DIVERSE_SEEDS:
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            cmd = [sys.executable, "-c", code]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            outputs.append(result.stdout.strip())

        first_json = outputs[0]
        for i, json_str in enumerate(outputs[1:], start=1):
            assert json_str == first_json, (
                f"El JSONL con PYTHONHASHSEED={DIVERSE_SEEDS[i]} difiere de seed {DIVERSE_SEEDS[0]}"
            )


# =============================================================================
# 2. Estabilidad de la clave de flujo RNG
# =============================================================================


class TestRngStreamKeyStability:
    """Verifica el aislamiento y reproducibilidad del flujo derivado por clave."""

    def test_identical_seed_agent_tick_produces_identical_random_sequence(self) -> None:
        """Una misma terna (semilla, agente, tick) produce exactamente la misma secuencia."""
        source1 = rng_mod.RngSource(seed=42)
        source2 = rng_mod.RngSource(seed=42)

        stream1 = source1.stream("decide", "centinela_01", 10)
        stream2 = source2.stream("decide", "centinela_01", 10)

        seq1 = [stream1.random() for _ in range(100)]
        seq2 = [stream2.random() for _ in range(100)]
        assert seq1 == seq2

    def test_changing_tick_alters_stream_predictably(self) -> None:
        """Alterar el tick cambia el flujo generado."""
        source = rng_mod.RngSource(seed=42)
        s0 = source.stream("decide", "centinela_01", 0)
        s1 = source.stream("decide", "centinela_01", 1)

        seq0 = [s0.random() for _ in range(50)]
        seq1 = [s1.random() for _ in range(50)]
        assert seq0 != seq1

    def test_changing_agent_alters_stream(self) -> None:
        """Alterar el agent_id genera flujos mutuamente independientes."""
        source = rng_mod.RngSource(seed=42)
        s_a = source.stream("decide", "agente_a", 0)
        s_b = source.stream("decide", "agente_b", 0)

        seq_a = [s_a.random() for _ in range(50)]
        seq_b = [s_b.random() for _ in range(50)]
        assert seq_a != seq_b

    def test_separator_prevents_concatenation_collision(self) -> None:
        """El separador ASCII evita colisiones entre ('ab', 'c') y ('a', 'bc')."""
        source = rng_mod.RngSource(seed=123)
        s1 = source.stream("decide", "ab", "c")
        s2 = source.stream("decide", "a", "bc")

        assert [s1.random() for _ in range(30)] != [s2.random() for _ in range(30)]

    def test_decide_reproducibility_with_stream_key(self) -> None:
        """`decide` invocado en estados idénticos devuelve la misma decisión."""
        pack = _make_adversarial_pack(num_actions=4, candidate_pool=4, temperature=1.0)
        agent1 = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={ontology.NeedId("eje_primario"): 0.5},
            traits={},
            goals={},
        )
        agent1.recompose(pack)
        exec1 = state_mod.ExecutionState(
            agent=agent1, tick=7, rng=rng_mod.RngSource(seed=999)
        )

        agent2 = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={ontology.NeedId("eje_primario"): 0.5},
            traits={},
            goals={},
        )
        agent2.recompose(pack)
        exec2 = state_mod.ExecutionState(
            agent=agent2, tick=7, rng=rng_mod.RngSource(seed=999)
        )

        d1 = decide_mod.decide(exec1, pack)
        d2 = decide_mod.decide(exec2, pack)
        assert d1 is not None and d2 is not None
        assert d1.action == d2.action
        assert d1.utility == d2.utility


# =============================================================================
# 3. Estabilidad numérica de Softmax
# =============================================================================


class TestSoftmaxNumericalStability:
    """Verifica que softmax no desborde con valores de utilidad extremos."""

    @pytest.mark.parametrize(
        "utility_values",
        [
            [1000.0, 0.0],
            [10000.0, 100.0, 0.0],
            [50000.0, 10.0, -1000.0],
            [1e6, 0.0, -1e6],
            [1e20, 1e10, 0.0],
            [1e100, 1e50, 0.0],
            [1e300, 0.0],
            [-1000.0, -2000.0],
            [-1e50, -1e100],
            [100000.0, 99999.0],  # valores enormes pero cercanos
        ],
    )
    def test_extreme_utility_difference_never_raises_overflow_error(
        self, utility_values: list[float]
    ) -> None:
        """La sustracción del máximo previene `OverflowError` en `math.exp`."""
        pack = _make_adversarial_pack(
            num_actions=len(utility_values),
            candidate_pool=len(utility_values),
            temperature=0.15,
        )
        agent = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={ontology.NeedId("eje_primario"): 0.5},
            traits={},
            goals={},
        )
        agent.recompose(pack)
        exec_st = state_mod.ExecutionState(
            agent=agent, tick=0, rng=rng_mod.RngSource(seed=123)
        )

        mock_pool = [
            decide_mod.Decision(
                action=ontology.ActionId(f"accion_{i:02d}"),
                utility=u,
                benefit=max(0.0, u),
                cost=max(0.0, -u),
                novelty=1.0,
                phase_fit=1.0,
                situation_fit=1.0,
            )
            for i, u in enumerate(utility_values)
        ]

        with patch("aida.agents.decide.candidates", return_value=(mock_pool, {})):
            chosen = decide_mod.decide(exec_st, pack)
            assert chosen is not None
            # El de mayor utilidad debe ser seleccionado cuando los demás se desvanecen a peso 0
            assert chosen.utility == max(utility_values)

    @pytest.mark.parametrize("temp", [1e-6, 1e-4, 0.01, 1.0, 100.0, 1000.0])
    def test_softmax_temperature_boundaries(self, temp: float) -> None:
        """Verifica estabilidad aritmética con temperaturas muy bajas o muy altas."""
        pack = _make_adversarial_pack(num_actions=3, candidate_pool=3, temperature=temp)
        agent = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={ontology.NeedId("eje_primario"): 0.5},
            traits={},
            goals={},
        )
        agent.recompose(pack)
        exec_st = state_mod.ExecutionState(
            agent=agent, tick=0, rng=rng_mod.RngSource(seed=42)
        )

        mock_pool = [
            decide_mod.Decision(
                action=ontology.ActionId(f"accion_{i:02d}"),
                utility=float(100 - i * 10),
                benefit=100.0,
                cost=float(i * 10),
                novelty=1.0,
                phase_fit=1.0,
                situation_fit=1.0,
            )
            for i in range(3)
        ]

        with patch("aida.agents.decide.candidates", return_value=(mock_pool, {})):
            chosen = decide_mod.decide(exec_st, pack)
            assert chosen is not None
            assert chosen.action in [d.action for d in mock_pool]

    def test_at_least_one_weight_is_strictly_one(self) -> None:
        """El mejor finalista tiene peso `math.exp(0.0) == 1.0`, impidiendo suma cero."""
        finalists = [
            decide_mod.Decision(
                action=ontology.ActionId("act_a"),
                utility=10.0,
                benefit=10.0,
                cost=0.0,
                novelty=1.0,
                phase_fit=1.0,
                situation_fit=1.0,
            ),
            decide_mod.Decision(
                action=ontology.ActionId("act_b"),
                utility=-999999.0,
                benefit=0.0,
                cost=999999.0,
                novelty=1.0,
                phase_fit=1.0,
                situation_fit=1.0,
            ),
        ]
        best = finalists[0].utility
        temperature = 0.15
        weights = [math.exp((d.utility - best) / temperature) for d in finalists]
        assert weights[0] == 1.0
        assert weights[1] == 0.0
        assert sum(weights) >= 1.0


# =============================================================================
# 4. Desempate estricto e independiente del orden del pack
# =============================================================================


class TestTieBreakingStressTest:
    """Verifica que el desempate por identificador sea absolutamente determinista."""

    def test_tie_breaking_invariant_across_50_pack_permutations(self) -> None:
        """50 permutaciones aleatorias de orden de inserción seleccionan
        exactamente la misma acción.
        """
        num_actions = 10
        actions_base = list(range(num_actions))
        rnd = random.Random(888)

        chosen_actions: list[ontology.ActionId] = []

        for _ in range(50):
            shuffled_order = list(actions_base)
            rnd.shuffle(shuffled_order)

            pack = _make_adversarial_pack(
                num_actions=num_actions,
                candidate_pool=5,
                temperature=0.5,
                actions_order=shuffled_order,
            )
            agent = state_mod.AgentState(
                agent_id=ontology.AgentId("agente_prueba"),
                needs={ontology.NeedId("eje_primario"): 0.5},
                traits={},
                goals={},
            )
            agent.recompose(pack)
            exec_st = state_mod.ExecutionState(
                agent=agent, tick=42, rng=rng_mod.RngSource(seed=1337)
            )

            d = decide_mod.decide(exec_st, pack)
            assert d is not None
            chosen_actions.append(d.action)

        unique_choices = set(chosen_actions)
        assert len(unique_choices) == 1, (
            f"El desempate divergente entre permutaciones: {unique_choices}"
        )

    def test_candidate_pool_one_takes_lexicographical_first_without_softmax(self) -> None:
        """Con candidate_pool=1, la acción con menor ID gana directamente entre empates."""
        for order in [[2, 1, 0], [0, 1, 2], [1, 2, 0]]:
            pack = _make_adversarial_pack(
                num_actions=3,
                candidate_pool=1,
                temperature=0.5,
                actions_order=order,
            )
            agent = state_mod.AgentState(
                agent_id=ontology.AgentId("agente_prueba"),
                needs={ontology.NeedId("eje_primario"): 0.5},
                traits={},
                goals={},
            )
            agent.recompose(pack)
            exec_st = state_mod.ExecutionState(
                agent=agent, tick=0, rng=rng_mod.RngSource(seed=1)
            )

            d = decide_mod.decide(exec_st, pack)
            assert d is not None
            assert d.action == "accion_00"


# =============================================================================
# 5. Casos frontera y condiciones de borde
# =============================================================================


class TestEdgeCases:
    """Verifica casos extremos en decide, critical_needs y explain."""

    def test_must_address_targeting_impossible_need_strictly_returns_none(self) -> None:
        """`must_address` a necesidad no atendida retorna None sin degradación libre."""
        pack = _make_adversarial_pack(num_actions=3)
        agent = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={ontology.NeedId("eje_primario"): 0.5},
            traits={},
            goals={},
        )
        agent.recompose(pack)
        exec_st = state_mod.ExecutionState(
            agent=agent, tick=0, rng=rng_mod.RngSource(seed=42)
        )

        # Necesidad inexistente
        res_nonexistent = decide_mod.decide(
            exec_st, pack, must_address=ontology.NeedId("eje_fantasma")
        )
        assert res_nonexistent is None

        # Necesidad existente pero no atendida por ninguna acción
        res_unserved = decide_mod.decide(
            exec_st, pack, must_address=ontology.NeedId("eje_secundario")
        )
        assert res_unserved is None

    def test_critical_needs_when_all_above_threshold_returns_empty_list(self) -> None:
        """`critical_needs` devuelve lista vacía si todas están sobre el umbral."""
        pack = _make_adversarial_pack()
        agent = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={
                ontology.NeedId("eje_primario"): 0.9,
                ontology.NeedId("eje_secundario"): 0.9,
            },
            traits={},
            goals={},
        )
        agent.recompose(pack)
        exec_st = state_mod.ExecutionState(agent=agent, tick=0)

        urgent = decide_mod.critical_needs(exec_st, pack)
        assert urgent == []

    def test_critical_needs_with_urgent_when_low_false_excluded(self) -> None:
        """Un eje con `urgent_when_low = False` jamás entra en `critical_needs`."""
        pack = _make_adversarial_pack()
        # eje_secundario tiene urgent_when_low = False
        agent = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={
                ontology.NeedId("eje_primario"): 0.9,  # por encima de umbral
                ontology.NeedId("eje_secundario"): 0.01,  # crítico pero acumulativo
            },
            traits={},
            goals={},
        )
        agent.recompose(pack)
        exec_st = state_mod.ExecutionState(agent=agent, tick=0)

        urgent = decide_mod.critical_needs(exec_st, pack)
        assert urgent == []

    def test_critical_needs_exact_boundary_comparison(self) -> None:
        """La comparación es `< threshold`: exactamente en el umbral no es crítica."""
        pack = _make_adversarial_pack()
        agent = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={
                ontology.NeedId("eje_primario"): 0.5,
                ontology.NeedId("eje_secundario"): 0.5,
            },
            traits={},
            goals={},
        )
        agent.recompose(pack)
        threshold = agent.temperament.critical_need_threshold  # type: ignore[union-attr]

        # Exactamente en el umbral
        agent.needs[ontology.NeedId("eje_primario")] = threshold
        exec_at = state_mod.ExecutionState(agent=agent, tick=0)
        assert decide_mod.critical_needs(exec_at, pack) == []

        # Un épsilon por debajo del umbral
        agent.needs[ontology.NeedId("eje_primario")] = threshold - 1e-6
        exec_below = state_mod.ExecutionState(agent=agent, tick=0)
        assert decide_mod.critical_needs(exec_below, pack) == [ontology.NeedId("eje_primario")]

    def test_critical_need_triage_during_in_progress_action(self) -> None:
        """Si la acción en curso atiende la peor emergencia, `critical_need` retorna None."""
        pack = _make_adversarial_pack()
        agent = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={ontology.NeedId("eje_primario"): 0.05},
            traits={},
            goals={},
        )
        agent.recompose(pack)
        exec_st = state_mod.ExecutionState(agent=agent, tick=0)

        # Sin acción en curso -> devuelve eje_primario
        assert decide_mod.critical_need(exec_st, pack) == ontology.NeedId("eje_primario")

        # Con acción en curso que SÍ satisface eje_primario -> devuelve None (no interrumpir)
        agent.performing = state_mod.PerformingAction(
            action=ontology.ActionId("accion_00"),
            ticks_remaining=2,
            total_ticks=2,
        )
        assert decide_mod.critical_need(exec_st, pack) is None

    def test_uncomposed_agent_state_raises_readable_exception(self) -> None:
        """Invocar funciones de decisión sin recompose lanza UncomposedAgentStateError."""
        pack = _make_adversarial_pack()
        agent = state_mod.AgentState(
            agent_id=ontology.AgentId("agente_prueba"),
            needs={ontology.NeedId("eje_primario"): 0.5},
            traits={},
            goals={},
        )
        assert agent.temperament is None
        exec_st = state_mod.ExecutionState(agent=agent, tick=0)

        with pytest.raises(decide_mod.UncomposedAgentStateError) as exc_decide:
            decide_mod.decide(exec_st, pack)
        assert "temperament=None" in str(exc_decide.value)
        assert "recompose" in str(exc_decide.value)

        with pytest.raises(decide_mod.UncomposedAgentStateError):
            decide_mod.candidates(exec_st, pack)

        with pytest.raises(decide_mod.UncomposedAgentStateError):
            decide_mod.critical_needs(exec_st, pack)

    def test_hard_doors_filter_and_empty_pool_returns_none(self) -> None:
        """Si todas las acciones quedan vetadas por puertas duras, decide retorna None."""
        needs = [
            ontology.NeedSpec(
                id=ontology.NeedId("eje"),
                label="Eje",
                initial=0.1,
                urgency_exponent=1.0,
                decay_per_tick=0.01,
            )
        ]
        actions = [
            ontology.ActionSpec(
                id=ontology.ActionId("act_vetada"),
                label="Vetada",
                duration_ticks=1,
                reversibility=ontology.Reversibility.REVERSIBLE,
                requires={ontology.NeedId("eje"): 0.9},  # requiere 0.9, agente tiene 0.1
                satisfies={ontology.NeedId("eje"): 0.5},
                effects={ontology.NeedId("eje"): 0.5},
            )
        ]
        agent_spec = ontology.AgentSpec(
            id=ontology.AgentId("agente"),
            label="Agente",
            initial_needs={ontology.NeedId("eje"): 0.1},
            traits={},
        )
        pack = pack_mod.ContentPack(
            id="p",
            label="P",
            needs=needs,
            actions=actions,
            agents=[agent_spec],
            tuning=ontology.TuningSpec(),
        )

        agent = state_mod.AgentState(
            agent_id=ontology.AgentId("agente"),
            needs={ontology.NeedId("eje"): 0.1},
            traits={},
            goals={},
        )
        agent.recompose(pack)
        exec_st = state_mod.ExecutionState(agent=agent, tick=0)

        pool, discarded = decide_mod.candidates(exec_st, pack)
        assert pool == []
        assert discarded == {ontology.ActionId("act_vetada"): "requires"}

        d = decide_mod.decide(exec_st, pack)
        assert d is None

        dec, tr = explain_mod.explain_deliberation(exec_st, pack)
        assert dec is None
        assert tr is None
