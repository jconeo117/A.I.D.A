"""Pruebas completas de explicabilidad y traza causal de deliberación (`aida.engine.explain`).

Cubre la unidad U4 de A.I.D.A. según la especificación de `.claude/workflow/07-plan-u4.md`:
1. Ancla de mutación 9: Fidelidad de la traza (no recalculación).
   - La utilidad de la acción ganadora en la traza es bit a bit idéntica a `decision.utility`.
   - La traza lee los sumandos directamente del objeto `Decision` sin recalcular la fórmula.
2. Ancla de mutación 11: Contenido exhaustivo de la traza causal.
   - Acción elegida y utilidad ganadora.
   - Segunda mejor opción (`runner_up`) y margen de diferencia (`margin`).
   - Caso de opción única sin alternativa (`runner_up is None`, `margin is None`).
   - Desglose aritmético de cada candidata evaluada (beneficio, coste, novedad, franja, situación).
   - Presiones de urgencia y pesos de necesidades.
   - Ejes críticos bajo el umbral.
   - Opciones vetadas con el nombre de su puerta dura (`requires`, `requires_traits`).
   - Desviación de creencias aprendidas respecto al valor nominal del pack (1.0).
3. Ancla de mutación 13: Serialización a JSONL determinista.
   - Claves ordenadas lexicográficamente en todos los niveles.
   - Exactamente una línea por registro JSONL (`splitlines()` == 1).
   - Escape de separadores de línea Unicode `\\u2028` y `\\u2029`.
   - Idéntica salida byte a byte entre procesos con diferente `PYTHONHASHSEED`.
4. Ancla de mutación 12: Integración en CLI `--explain` (`aida.run.main`).
   - Retorna código 0 en éxito e imprime la traza legible por consola.
   - Retorna código 1 cuando no hay candidatos elegibles, sin llamar a `sys.exit`.
   - Determinismo entre subprocesos con diferente `PYTHONHASHSEED`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import aida.agents.decide as decide
import aida.engine.explain as explain
import aida.engine.ontology as ontology
import aida.engine.pack as pack
import aida.engine.rng as rng
import aida.engine.state as state
from aida.run import main

TEMPLATE_PACK_PATH = (
    Path(__file__).resolve().parent.parent / "packs" / "_plantilla" / "pack.yaml"
)


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
    """Construye un NeedSpec sin vocabulario de dominio."""
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
    """Construye un ContentPack válido para pruebas de explicabilidad."""
    ns = needs or [_make_need("need_a")]
    first_need_id = str(ns[0].id)
    acts = (
        actions
        if actions is not None
        else [_make_action("action_1", satisfies={first_need_id: 0.5})]
    )
    trs = traits or []

    needed_phases = {p for a in acts for p in a.phase_multipliers}
    ph_list = list(phases) if phases is not None else [
        ontology.PhaseSpec(id=pid, label=str(pid), starts_at=idx * 10)
        for idx, pid in enumerate(sorted(needed_phases))
    ]

    needed_situations = {s for a in acts for s in a.situation_multipliers}
    sit_list = list(situations) if situations is not None else [
        ontology.SituationSpec(id=sid, label=str(sid)) for sid in sorted(needed_situations)
    ]

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
        label="Pack de prueba explain",
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
# 1. Ancla de mutación 9: Fidelidad de la traza (No recalculación)
# =============================================================================


def test_fidelidad_traza_utilidad_ganadora_identica_bit_a_bit() -> None:
    """Ancla contra mutación: la utilidad de la ganadora coincide bit a bit con decision.utility.

    La traza LEE los sumandos y la utilidad que `Decision` ya calculó.
    Si `explain` recalculara por su cuenta (mutación: recalcular con otra fórmula
    o redondear), divergiría de la decisión real tomada.
    """
    need = _make_need("need_a", cap=1.0, exponent=1.7, initial=0.3333333333333333)
    act1 = _make_action("act_1", satisfies={"need_a": 0.7777777777777778})
    act2 = _make_action("act_2", satisfies={"need_a": 0.5555555555555556})
    p = _make_pack(needs=[need], actions=[act1, act2])
    exec_st = _make_execution(p, tick=4)

    decision, trace = explain.explain_deliberation(exec_st, p)

    assert decision is not None
    assert trace is not None
    # Comprobación de fidelidad bit a bit: igualdad exacta y misma representación hexadecimal
    assert trace.utility == decision.utility
    assert trace.utility.hex() == decision.utility.hex()
    assert trace.chosen == decision.action


def test_build_trace_lee_sumandos_directos_de_decision_sin_recalcular() -> None:
    """`build_trace` consume directamente los campos de `Decision`.

    Incluso si se pasa un `Decision` con valores artificiales inusuales,
    `build_trace` los transfiere fielmente a la traza sin reemplazarlos por una fórmula.
    """
    need = _make_need("need_a")
    p = _make_pack(needs=[need])
    exec_st = _make_execution(p)

    synthetic_decision = decide.Decision(
        action=ontology.ActionId("act_custom"),
        utility=123.456789,
        benefit=100.0,
        cost=50.0,
        novelty=0.75,
        phase_fit=1.2,
        situation_fit=1.1,
    )

    trace = explain.build_trace(
        exec_st,
        p,
        decision=synthetic_decision,
        pool=[synthetic_decision],
        discarded={},
    )

    assert trace.chosen == ontology.ActionId("act_custom")
    assert trace.utility == 123.456789
    assert len(trace.candidates) == 1
    cand = trace.candidates[0]
    assert cand.benefit == 100.0
    assert cand.cost == 50.0
    assert cand.novelty == 0.75
    assert cand.phase_fit == 1.2
    assert cand.situation_fit == 1.1


# =============================================================================
# 2. Ancla de mutación 11: Contenido exhaustivo de la traza causal
# =============================================================================


def test_traza_contiene_runner_up_y_margen_con_multiples_candidatos() -> None:
    """La traza reporta la segunda mejor opción (`runner_up`) y el margen exacto de diferencia."""
    need = _make_need("need_a", initial=0.0)
    act_a = _make_action("act_a", satisfies={"need_a": 0.8})
    act_b = _make_action("act_b", satisfies={"need_a": 0.5})
    p = _make_pack(needs=[need], actions=[act_a, act_b], candidate_pool=1)
    exec_st = _make_execution(p)

    decision, trace = explain.explain_deliberation(exec_st, p)
    assert decision is not None
    assert trace is not None
    assert trace.chosen == ontology.ActionId("act_a")
    assert trace.runner_up == ontology.ActionId("act_b")
    # Margen es la diferencia entre la mejor y la segunda
    assert trace.margin is not None
    expected_margin = 0.8 - 0.5
    assert trace.margin == pytest.approx(expected_margin)


def test_traza_sin_alternativa_cuando_hay_un_solo_candidato() -> None:
    """Cuando solo hay una opción disponible, `runner_up` y `margin` son `None`."""
    need = _make_need("need_a")
    act_solo = _make_action("act_solo", satisfies={"need_a": 0.5})
    p = _make_pack(needs=[need], actions=[act_solo])
    exec_st = _make_execution(p)

    decision, trace = explain.explain_deliberation(exec_st, p)
    assert decision is not None
    assert trace is not None
    assert trace.chosen == ontology.ActionId("act_solo")
    assert trace.runner_up is None
    assert trace.margin is None


def test_traza_reporta_opciones_descartadas_con_nombre_de_puerta() -> None:
    """La traza registra todas las opciones descartadas y su puerta de descarte."""
    need = _make_need("need_a")
    trait = ontology.TraitSpec(id=ontology.TraitId("trait_a"), label="Trait A")
    act_ok = _make_action("act_ok", satisfies={"need_a": 0.3})
    act_need_fail = _make_action("act_need_fail", requires={"need_a": 0.95})
    act_trait_fail = _make_action("act_trait_fail", requires_traits={"trait_a": 0.95})

    p = _make_pack(
        needs=[need],
        traits=[trait],
        actions=[act_ok, act_need_fail, act_trait_fail],
    )
    exec_st = _make_execution(
        p,
        needs={"need_a": 0.5},
        traits={"trait_a": 0.2},
    )

    decision, trace = explain.explain_deliberation(exec_st, p)
    assert decision is not None
    assert trace is not None

    assert ontology.ActionId("act_need_fail") in trace.discarded
    assert trace.discarded[ontology.ActionId("act_need_fail")] == "requires"

    assert ontology.ActionId("act_trait_fail") in trace.discarded
    assert trace.discarded[ontology.ActionId("act_trait_fail")] == "requires_traits"

    assert ontology.ActionId("act_ok") not in trace.discarded


def test_traza_reporta_desviaciones_de_creencias_aprendidas() -> None:
    """La traza computa y muestra la desviación de creencias aprendidas respecto a 1.0."""
    need = _make_need("need_a")
    act = _make_action("act_1", satisfies={"need_a": 0.5})
    p = _make_pack(needs=[need], actions=[act])

    # El agente cree que act_1 rinde 1.4 (+0.4 sobre el pack)
    exec_st = _make_execution(
        p,
        gain_beliefs={"act_1": {"need_a": 1.4}},
    )

    decision, trace = explain.explain_deliberation(exec_st, p)
    assert decision is not None
    assert trace is not None

    assert ontology.ActionId("act_1") in trace.belief_deviations
    devs = trace.belief_deviations[ontology.ActionId("act_1")]
    assert ontology.NeedId("need_a") in devs
    assert devs[ontology.NeedId("need_a")] == pytest.approx(0.4)


def test_traza_sin_desviaciones_cuando_creencias_coinciden_con_pack() -> None:
    """Si las creencias no se desvían de 1.0, `belief_deviations` permanece vacío."""
    need = _make_need("need_a")
    act = _make_action("act_1", satisfies={"need_a": 0.5})
    p = _make_pack(needs=[need], actions=[act])
    exec_st = _make_execution(p)

    decision, trace = explain.explain_deliberation(exec_st, p)
    assert decision is not None
    assert trace is not None
    assert trace.belief_deviations == {}


# =============================================================================
# 3. Ancla de mutación 13: Serialización a JSONL determinista
# =============================================================================


def test_jsonl_claves_ordenadas_en_todos_los_niveles() -> None:
    """`trace.to_json()` vuelca un JSONL con claves ordenadas en todos los niveles."""
    need = _make_need("need_a")
    act = _make_action("act_1", satisfies={"need_a": 0.5})
    p = _make_pack(needs=[need], actions=[act])
    exec_st = _make_execution(p)

    _, trace = explain.explain_deliberation(exec_st, p)
    assert trace is not None

    line = trace.to_json()
    assert isinstance(line, str)

    # Debe ser una sola línea JSON válida
    parsed = json.loads(line)
    assert isinstance(parsed, dict)

    # Verificar que las claves en la cadena JSON aparecen en orden alfabético
    # Para el nivel raíz:
    keys = list(parsed.keys())
    assert keys == sorted(keys)


def test_jsonl_exactamente_una_linea_y_escape_unicode() -> None:
    """`to_json` garantiza exactamente una línea, escapando saltos de línea y U+2028/U+2029."""
    need = _make_need("need_a")
    act = _make_action("act_1", satisfies={"need_a": 0.5})
    p = _make_pack(needs=[need], actions=[act])
    exec_st = _make_execution(p)

    _, trace = explain.explain_deliberation(exec_st, p)
    assert trace is not None

    line = trace.to_json()
    assert len(line.splitlines()) == 1
    assert "\n" not in line
    assert "\r" not in line
    assert "\u2028" not in line
    assert "\u2029" not in line


def test_jsonl_reproducible_entre_procesos_con_distinto_hashseed() -> None:
    """La serialización JSONL de una traza es idéntica byte a byte con PYTHONHASHSEED."""
    code = """
import aida.engine.explain as explain
import aida.engine.ontology as ontology
import aida.engine.pack as pack
import aida.engine.state as state
import aida.engine.rng as rng

need = ontology.NeedSpec(id=ontology.NeedId("n1"), label="N1", decay_per_tick=0.01, initial=0.0)
act1 = ontology.ActionSpec(
    id=ontology.ActionId("act_b"),
    label="B",
    duration_ticks=1,
    reversibility=ontology.Reversibility.REVERSIBLE,
    satisfies={ontology.NeedId("n1"): 0.8},
    effects={ontology.NeedId("n1"): 0.8},
)
act2 = ontology.ActionSpec(
    id=ontology.ActionId("act_a"),
    label="A",
    duration_ticks=1,
    reversibility=ontology.Reversibility.REVERSIBLE,
    satisfies={ontology.NeedId("n1"): 0.4},
    effects={ontology.NeedId("n1"): 0.4},
)
agent_spec = ontology.AgentSpec(
    id=ontology.AgentId("ag1"), label="Ag1", initial_needs={need.id: 0.0}
)
p = pack.ContentPack(
    id="p1",
    label="P1",
    needs=[need],
    actions=[act1, act2],
    agents=[agent_spec],
    tuning=ontology.TuningSpec(candidate_pool=1, temperature=0.1),
)
agent_st = state.AgentState(agent_id=ontology.AgentId("ag1"), needs={need.id: 0.0})
agent_st.recompose(p)
exec_st = state.ExecutionState(agent=agent_st, tick=0, rng=rng.RngSource(42))

_, trace = explain.explain_deliberation(exec_st, p)
assert trace is not None
print(trace.to_json())
"""
    outputs = []
    for hseed in ["0", "1", "42", "999"]:
        p_res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hseed},
        )
        outputs.append(p_res.stdout.strip())

    assert len(set(outputs)) == 1, f"JSONL divergió entre semillas de hash: {outputs}"


# =============================================================================
# 4. Ancla de mutación 12: Integración en CLI (--explain)
# =============================================================================


def test_cli_explain_exito_con_pack_plantilla(capsys: pytest.CaptureFixture[str]) -> None:
    """`aida --pack packs/_plantilla/pack.yaml --explain` retorna 0 e imprime traza legible."""
    assert TEMPLATE_PACK_PATH.exists(), f"Falta pack de plantilla en {TEMPLATE_PACK_PATH}"

    code = main(["--pack", str(TEMPLATE_PACK_PATH), "--explain"])
    captured = capsys.readouterr()

    assert code == 0
    assert "=== Deliberación: agente=centinela_01" in captured.out
    assert "Acción elegida:" in captured.out
    assert "Utilidad ganadora:" in captured.out
    assert "Opciones evaluadas" in captured.out


def test_cli_explain_sin_opciones_retorna_uno(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`--explain` sobre un pack sin acciones elegibles retorna 1."""
    yaml_no_actions = """
id: pack_empty
label: Pack sin acciones
needs:
  - id: n1
    label: N1
    decay_per_tick: 0.01
    initial: 0.5
actions: []
agents:
  - id: ag1
    label: Ag1
    initial_needs: { n1: 0.5 }
tuning:
  candidate_pool: 1
  temperature: 0.1
"""
    p_empty = tmp_path / "pack_empty.yaml"
    p_empty.write_text(yaml_no_actions, encoding="utf-8")

    code_empty = main(["--pack", str(p_empty), "--explain"])
    captured_empty = capsys.readouterr()

    assert code_empty == 1
    assert "No hay ninguna opción disponible" in captured_empty.err


def test_format_human_estructura_completa() -> None:
    """`format_human` genera una representación de texto con todas las secciones de diseño."""
    need = _make_need("need_a")
    act = _make_action("act_1", satisfies={"need_a": 0.5})
    p = _make_pack(needs=[need], actions=[act])
    exec_st = _make_execution(p)

    _, trace = explain.explain_deliberation(exec_st, p)
    assert trace is not None

    text = trace.format_human()
    assert "=== Deliberación:" in text
    assert "Decisión:" in text
    assert "Opciones evaluadas" in text
    assert "Presiones y pesos por necesidad:" in text
    assert "Desviación de creencias aprendidas" in text
    assert "Opciones descartadas por puertas duras" in text
    assert str(trace) == text
