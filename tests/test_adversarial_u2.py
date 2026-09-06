"""Batería de pruebas adversariales y de estrés para la Unidad U2.

Diseñada por Challenger 1 para someter a prueba:
1. Determinismo byte a byte entre subprocesos independientes con diversas semillas PYTHONHASHSEED.
2. Coherencia de signos en ActionSpec y casos de esquina (ceros, flotantes extremos, NaNs).
3. Comprobación rigurosa de cotas en _check_declared_bounds.
4. Consolidación de fallos múltiples (10+ fallos simultáneos sin cortocircuito).
5. Entradas hostiles, inyecciones en YAML y tipos inválidos.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from aida.engine.ontology import (
    ActionSpec,
    AgentSpec,
    GoalSpec,
    HealthSpec,
    NeedId,
    NeedSpec,
    Reversibility,
    TasteSpec,
    TraitSpec,
)
from aida.engine.pack import ContentPack, PackError, load_pack

# ===========================================================================
# 1. Determinismo entre subprocesos (PYTHONHASHSEED)
# ===========================================================================

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


def test_adversarial_determinismo_serializacion_tags_muchos_seeds() -> None:
    """Verifica que 20 tags serializan a JSON exactamente igual en 12+ semillas hash."""
    worker_code = """
import json, sys
from aida.engine.ontology import ActionSpec, Reversibility

tags = frozenset([
    "whiskey", "alpha", "tango", "bravo", "foxtrot",
    "charlie", "zulu", "delta", "kilo", "echo",
    "golf", "hotel", "india", "juliet", "lima",
    "mike", "november", "oscar", "quebec", "romeo"
])
action = ActionSpec(
    id="test_action",
    label="Acción de prueba",
    reversibility=Reversibility.REVERSIBLE,
    duration_ticks=1,
    tags=tags,
)
dumped = action.model_dump_json()
data = json.loads(dumped)
sys.stdout.write(json.dumps(data["tags"]))
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

    assert len(results) == 1, f"Serialización divergió entre semillas: {results}"
    expected_order = sorted(
        [
            "whiskey",
            "alpha",
            "tango",
            "bravo",
            "foxtrot",
            "charlie",
            "zulu",
            "delta",
            "kilo",
            "echo",
            "golf",
            "hotel",
            "india",
            "juliet",
            "lima",
            "mike",
            "november",
            "oscar",
            "quebec",
            "romeo",
        ]
    )
    assert json.loads(next(iter(results))) == expected_order


def test_adversarial_determinismo_mensaje_error_pack_muchos_seeds() -> None:
    """Verifica error de pack malformado byte a byte idéntico en 12+ semillas."""
    worker_code = """
import sys
from pydantic import ValidationError
from aida.engine.pack import ContentPack

data = {
    "id": "broken_pack",
    "label": "Pack Roto",
    "needs": [
        {"id": "n_omega", "label": "Omega", "initial": 0.5, "decay_per_tick": 0.01},
        {"id": "n_alpha", "label": "Alpha", "initial": 0.5, "decay_per_tick": 0.01},
        {"id": "n_omega", "label": "Omega Duplicado", "initial": 0.5, "decay_per_tick": 0.01},
        {"id": "n_beta", "label": "Beta", "initial": 0.5, "decay_per_tick": 0.01},
    ],
    "actions": [
        {
            "id": "act_shared",
            "label": "Accion Compartida",
            "reversibility": "irreversible",
            "duration_ticks": 1,
            "effects": {"n_alpha": -0.4, "n_inexistente": 0.2},
            "satisfies": {"n_inexistente": 0.2},
        },
    ],
    "agents": [
        {
            "id": "act_shared",
            "label": "Agente con id colisionante",
            "initial_needs": {"n_alpha": 1.5, "n_fantasma": 0.5},
            "tastes": [{"target": "entidad_desconocida", "multiplier": 0.5}],
        },
    ],
    "health": {
        "may_go_unused": [
            "invalida_z", "invalida_a", "invalida_m", "invalida_d", "invalida_k",
            "invalida_b", "invalida_x", "invalida_c", "invalida_w", "invalida_p",
        ],
    },
}

try:
    ContentPack.model_validate(data)
except ValidationError as exc:
    sys.stdout.write(str(exc))
except Exception as e:
    sys.stderr.write(f"Unexpected: {e}")
    sys.exit(2)
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

    assert len(results) == 1, f"Mensaje de error divergió entre semillas! {results}"


# ===========================================================================
# 2. Coherencia de signos en ActionSpec
# ===========================================================================


def test_adversarial_sign_coherence_satisfies_zero_delta_rejected() -> None:
    """satisfies con delta 0.0 en effects debe ser rechazado."""
    with pytest.raises(
        ValidationError,
        match=r"satisfies.*debe aparecer en `effects` con signo positivo",
    ):
        ActionSpec(
            id="act_zero_sat",
            label="Zero satisfies delta",
            reversibility=Reversibility.REVERSIBLE,
            duration_ticks=1,
            satisfies={"n1": 0.5},
            effects={"n1": 0.0},
        )


def test_adversarial_sign_coherence_costs_zero_delta_rejected() -> None:
    """costs con delta 0.0 en effects debe ser rechazado."""
    with pytest.raises(
        ValidationError,
        match=r"costs.*debe aparecer en `effects` con signo negativo",
    ):
        ActionSpec(
            id="act_zero_cost",
            label="Zero cost delta",
            reversibility=Reversibility.REVERSIBLE,
            duration_ticks=1,
            costs={"n1": 0.5},
            effects={"n1": 0.0},
        )


def test_adversarial_sign_coherence_costs_negative_zero_delta_rejected() -> None:
    """costs con delta -0.0 en effects debe ser rechazado (-0.0 >= 0.0 en Python)."""
    with pytest.raises(
        ValidationError,
        match=r"costs.*debe aparecer en `effects` con signo negativo",
    ):
        ActionSpec(
            id="act_neg_zero_cost",
            label="Neg zero cost delta",
            reversibility=Reversibility.REVERSIBLE,
            duration_ticks=1,
            costs={"n1": 0.5},
            effects={"n1": -0.0},
        )


def test_adversarial_sign_coherence_satisfies_key_missing_from_effects() -> None:
    """Clave en satisfies que no existe en effects debe ser rechazada."""
    with pytest.raises(ValidationError, match=r"satisfies.*debe aparecer en `effects`"):
        ActionSpec(
            id="act_missing_sat",
            label="Missing satisfies in effects",
            reversibility=Reversibility.REVERSIBLE,
            duration_ticks=1,
            satisfies={"n1": 0.5},
            effects={},
        )


def test_adversarial_sign_coherence_costs_key_missing_from_effects() -> None:
    """Clave en costs que no existe en effects debe ser rechazada."""
    with pytest.raises(ValidationError, match=r"costs.*debe aparecer en `effects`"):
        ActionSpec(
            id="act_missing_cost",
            label="Missing cost in effects",
            reversibility=Reversibility.REVERSIBLE,
            duration_ticks=1,
            costs={"n1": 0.5},
            effects={},
        )


def test_adversarial_sign_coherence_key_in_both_satisfies_and_costs_impossible() -> None:
    """Una misma clave en satisfies y costs no puede tener signos opuestos a la vez."""
    with pytest.raises(ValidationError):
        ActionSpec(
            id="act_both",
            label="Key in both satisfies and costs",
            reversibility=Reversibility.REVERSIBLE,
            duration_ticks=1,
            satisfies={"n1": 0.3},
            costs={"n1": 0.3},
            effects={"n1": 0.2},
        )


def test_adversarial_sign_coherence_explicit_zero_cost_with_negative_effect() -> None:
    """Un coste declarado como 0.0 explícito para ignorar efecto negativo es válido."""
    action = ActionSpec(
        id="act_explicit_zero_cost",
        label="Efecto negativo ignorado a propósito",
        reversibility=Reversibility.REVERSIBLE_WITH_COST,
        duration_ticks=1,
        costs={"n1": 0.0},
        effects={"n1": -0.2},
    )
    assert action.costs[NeedId("n1")] == 0.0
    assert action.effects[NeedId("n1")] == -0.2


def test_adversarial_sign_coherence_extreme_float_boundaries() -> None:
    """Valores flotantes extremadamente pequeños cerca de 0.0 respetan los signos."""
    action = ActionSpec(
        id="act_tiny_floats",
        label="Flotantes diminutos",
        reversibility=Reversibility.REVERSIBLE,
        duration_ticks=1,
        satisfies={"n1": 0.01},
        costs={"n2": 0.01},
        effects={"n1": 1e-12, "n2": -1e-12},
    )
    assert action.effects[NeedId("n1")] == 1e-12
    assert action.effects[NeedId("n2")] == -1e-12


def test_adversarial_requires_traits_exact_boundaries() -> None:
    """requires_traits en -1.0 y 1.0 son válidos; fuera por épsilon son rechazados."""
    ActionSpec(
        id="act_bound_ok",
        label="Bounds OK",
        reversibility=Reversibility.REVERSIBLE,
        duration_ticks=1,
        requires_traits={"t1": -1.0, "t2": 1.0},
    )

    with pytest.raises(ValidationError, match=r"requires_traits.*fuera de \[-1, 1\]"):
        ActionSpec(
            id="act_bound_low",
            label="Bound Low",
            reversibility=Reversibility.REVERSIBLE,
            duration_ticks=1,
            requires_traits={"t1": -1.000001},
        )

    with pytest.raises(ValidationError, match=r"requires_traits.*fuera de \[-1, 1\]"):
        ActionSpec(
            id="act_bound_high",
            label="Bound High",
            reversibility=Reversibility.REVERSIBLE,
            duration_ticks=1,
            requires_traits={"t1": 1.000001},
        )


# ===========================================================================
# 3. Comprobación de cotas en _check_declared_bounds
# ===========================================================================


def test_adversarial_bounds_initial_needs_exact_boundaries() -> None:
    """AgentSpec.initial_needs en 0 y 1 son válidos; fuera son detectados por cotas."""
    need = NeedSpec(id="n1", label="N1", initial=0.5, decay_per_tick=0.01)
    act = ActionSpec(
        id="a1",
        label="A1",
        reversibility=Reversibility.REVERSIBLE,
        duration_ticks=1,
        effects={"n1": 0.1},
        satisfies={"n1": 0.1},
    )

    # Válido en 0.0 y 1.0
    pack_ok = ContentPack(
        id="p_ok",
        label="Pack OK",
        needs=(need,),
        actions=(act,),
        agents=(
            AgentSpec(id="ag1", label="Agente 1", initial_needs={"n1": 0.0}),
            AgentSpec(id="ag2", label="Agente 2", initial_needs={"n1": 1.0}),
        ),
    )
    assert len(pack_ok.agents) == 2

    # Inválido en -1e-6
    with pytest.raises(
        ValueError,
        match=r"agente 'ag_low'\.initial_needs\['n1'\] = -1e-06, fuera de \[0\.0, 1\.0\]",
    ):
        ContentPack(
            id="p_bad_low",
            label="Pack Bad Low",
            needs=(need,),
            actions=(act,),
            agents=(AgentSpec(id="ag_low", label="Ag Low", initial_needs={"n1": -1e-6}),),
        )

    # Inválido en 1.000001
    with pytest.raises(
        ValueError,
        match=r"agente 'ag_high'\.initial_needs\['n1'\] = 1\.000001, fuera de \[0\.0, 1\.0\]",
    ):
        ContentPack(
            id="p_bad_high",
            label="Pack Bad High",
            needs=(need,),
            actions=(act,),
            agents=(AgentSpec(id="ag_high", label="Ag High", initial_needs={"n1": 1.000001}),),
        )


def test_adversarial_bounds_at_traits_exact_boundaries() -> None:
    """GoalSpec.at_traits en -2.0 y 2.0 son válidos; fuera son detectados en agente y en pack."""
    need = NeedSpec(id="n1", label="N1", initial=0.5, decay_per_tick=0.01)
    trait = TraitSpec(id="t1", label="T1")
    act = ActionSpec(
        id="a1",
        label="A1",
        reversibility=Reversibility.REVERSIBLE,
        duration_ticks=1,
        effects={"n1": 0.1},
        satisfies={"n1": 0.1},
    )

    # Válido en -2.0 y 2.0 en agente y pack con meta helper para alcanzar cota
    goal_helper = GoalSpec(
        id="g_helper", label="Helper", need="n1", at_least=0.5, on_success={"t1": 1.0}
    )
    goal_min = GoalSpec(id="g_min", label="G Min", need="n1", at_traits={"t1": -2.0})
    goal_max = GoalSpec(id="g_max", label="G Max", need="n1", at_traits={"t1": 2.0})

    goal_helper_pack = GoalSpec(
        id="g_helper_p", label="Helper P", need="n1", at_least=0.5, on_success={"t1": 2.0}
    )

    pack_ok = ContentPack(
        id="p_ok_traits",
        label="Pack OK Traits",
        needs=(need,),
        traits=(trait,),
        actions=(act,),
        agents=(
            AgentSpec(
                id="ag1",
                label="Ag1",
                traits={"t1": 1.0},
                goals=(goal_helper, goal_min, goal_max),
            ),
        ),
        goals=(goal_helper_pack, goal_min, goal_max),
    )
    assert len(pack_ok.goals) == 3

    # Inválido en agente: at_traits < -2.0
    goal_toolow = GoalSpec(id="g_toolow", label="G Low", need="n1", at_traits={"t1": -2.0001})
    pattern_low = (
        r"agente 'ag1', meta 'g_toolow'\.at_traits\['t1'\] = -2\.0001, "
        r"fuera de \[-2\.0, 2\.0\]"
    )
    with pytest.raises(ValueError, match=pattern_low):
        ContentPack(
            id="p_bad_ag_trait_low",
            label="Pack Bad",
            needs=(need,),
            traits=(trait,),
            actions=(act,),
            agents=(
                AgentSpec(
                    id="ag1",
                    label="Ag1",
                    traits={"t1": 0.0},
                    goals=(goal_toolow,),
                ),
            ),
        )

    # Inválido en pack: at_traits > 2.0
    goal_toohigh = GoalSpec(id="g_toohigh", label="G High", need="n1", at_traits={"t1": 2.0001})
    with pytest.raises(
        ValueError,
        match=r"meta 'g_toohigh'\.at_traits\['t1'\] = 2\.0001, fuera de \[-2\.0, 2\.0\]",
    ):
        ContentPack(
            id="p_bad_pack_trait_high",
            label="Pack Bad High",
            needs=(need,),
            traits=(trait,),
            actions=(act,),
            agents=(AgentSpec(id="ag1", label="Ag1"),),
            goals=(goal_toohigh,),
        )


# ===========================================================================
# 4. Consolidación de múltiples fallos (10+ fallos de diversas clases)
# ===========================================================================


def test_adversarial_consolidacion_fallos_masivos() -> None:
    """Un pack con 13 violaciones simultáneas reporta TODOS los fallos sin cortocircuito."""
    with pytest.raises(ValidationError) as exc_info:
        ContentPack(
            id="pack_caotico",
            label="Pack con 13 violaciones",
            needs=(
                NeedSpec(id="n1", label="N1", initial=0.5, decay_per_tick=0.01),
                NeedSpec(id="n1", label="N1 Dup", initial=0.5, decay_per_tick=0.01),
            ),
            traits=(
                TraitSpec(id="t_valido", label="T1"),
                TraitSpec(id="t_fantasma_shift", label="Shift sin franjas", phase_shift=1),
            ),
            phases=(),
            situations=(),
            actions=(
                ActionSpec(
                    id="clash_id",
                    label="Accion Clash",
                    reversibility=Reversibility.IRREVERSIBLE,
                    duration_ticks=1,
                    requires={"n_fantasma": 0.5},
                    requires_traits={"t_desconocido": 0.5},
                    phase_multipliers={"franja_inexistente": 1.5},
                    situation_multipliers={"sit_inexistente": 1.2},
                    effects={"n1": -0.4},
                ),
            ),
            agents=(
                AgentSpec(
                    id="clash_id",
                    label="Agente Clash",
                    initial_needs={"n1": 3.0},
                    traits={"t_valido": 0.0},
                    tastes=(TasteSpec(target="entidad_inexistente", multiplier=0.8),),
                    goals=(
                        GoalSpec(
                            id="g_ciclo_1",
                            label="G1",
                            need="n1",
                            requires_goal="g_ciclo_2",
                            at_traits={"t_valido": 5.0},
                        ),
                        GoalSpec(
                            id="g_ciclo_2",
                            label="G2",
                            need="n1",
                            at_least=0.5,
                            requires_goal="g_ciclo_1",
                        ),
                    ),
                ),
            ),
            health=HealthSpec(
                may_go_unused=frozenset(["accion_no_registrada"]),
            ),
        )

    msg = str(exc_info.value)

    expected_checks = [
        "necesidades: id duplicado 'n1'",
        "rasgo 't_fantasma_shift': declara `phase_shift` pero el pack no tiene franjas",
        "id 'clash_id' usado a la vez por un/a acción y un/a agente",
        "acción 'clash_id'.requires: necesidad desconocida/o 'n_fantasma'",
        "acción 'clash_id'.requires_traits: rasgo desconocida/o 't_desconocido'",
        "acción 'clash_id': franja desconocida/o 'franja_inexistente'",
        "acción 'clash_id': situación desconocida/o 'sit_inexistente'",
        "agente 'clash_id': ciclo de metas g_ciclo_1 -> g_ciclo_2 -> g_ciclo_1",
        "agente 'clash_id': gusto hacia 'entidad_inexistente', que no es ninguna acción ni agente",
        "acción 'clash_id': el efecto sobre 'n1' es negativo (-0.4) pero no declara coste",
        "health.may_go_unused: acción desconocida 'accion_no_registrada'",
        "agente 'clash_id'.initial_needs['n1'] = 3.0, fuera de [0.0, 1.0]",
        "agente 'clash_id', meta 'g_ciclo_1'.at_traits['t_valido'] = 5.0, fuera de [-2.0, 2.0]",
    ]

    missing = [c for c in expected_checks if c not in msg]
    assert not missing, "Los siguientes errores no aparecieron en el reporte:\n" + "\n".join(
        missing
    )


# ===========================================================================
# 5. Entradas hostiles y seguridad de carga
# ===========================================================================


def test_adversarial_load_pack_yaml_bomba_o_inyeccion_objeto(tmp_path: Path) -> None:
    """load_pack rechaza inyección de tipos python y etiquetas de ejecución de código."""
    malicious_yaml = tmp_path / "malicious.yaml"
    malicious_yaml.write_text(
        """
id: exploit
label: Exploit
needs: !!python/object/apply:os.system ["echo PWNED"]
actions: []
agents: []
""",
        encoding="utf-8",
    )
    with pytest.raises(PackError, match=r"no se pudo cargar"):
        load_pack(malicious_yaml)


def test_adversarial_content_pack_rechaza_campos_adicionales() -> None:
    """ContentPack hereda de Spec y debe rechazar claves adicionales por extra='forbid'."""
    need = NeedSpec(id="n1", label="N1", initial=0.5, decay_per_tick=0.01)
    act = ActionSpec(
        id="a1",
        label="A1",
        reversibility=Reversibility.REVERSIBLE,
        duration_ticks=1,
    )
    agent = AgentSpec(id="ag1", label="Ag1")

    with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
        ContentPack.model_validate(
            {
                "id": "pack_extra",
                "label": "Pack Extra",
                "needs": [need.model_dump()],
                "actions": [act.model_dump()],
                "agents": [agent.model_dump()],
                "campo_no_permitido": "malicioso",
            }
        )


def test_adversarial_actionspec_rechaza_reversibility_invalida_o_ausente() -> None:
    """ActionSpec rechaza strings no pertenecientes a Reversibility o la ausencia del campo."""
    # Ausente
    with pytest.raises(ValidationError, match=r"Field required"):
        ActionSpec.model_validate(
            {
                "id": "act_sin_rev",
                "label": "Sin rev",
                "duration_ticks": 1,
            }
        )

    # String inválido
    with pytest.raises(
        ValidationError,
        match=r"Input should be 'reversible', 'reversible_with_cost' or 'irreversible'",
    ):
        ActionSpec.model_validate(
            {
                "id": "act_rev_invalida",
                "label": "Rev invalida",
                "reversibility": "semi_reversible",
                "duration_ticks": 1,
            }
        )
