"""Pruebas completas de carga y validación de packs (`aida.engine.pack`).

Cubre la unidad U2:
- Contrato de `ContentPack`, `PackError`, `load_pack` y `describe_pack`.
- Las 14 comprobaciones deterministas `_check_*`.
- Reporte consolidado de múltiples errores simultáneos (4+ clases de error).
- Determinismo del mensaje de error entre subprocesos con diferente `PYTHONHASHSEED`.
- Índices cacheados (`need_by_id`, `action_by_id`, etc.).
- Validación del pack de plantilla en `packs/_plantilla/pack.yaml`.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import aida.engine.ontology as ontology
import aida.engine.pack as pack


def base_pack_dict() -> dict[str, Any]:

    """Retorna la definición de un ContentPack mínimo válido."""
    return {
        "id": "pack_test",
        "label": "Pack de prueba",
        "needs": [
            {
                "id": "cpu",
                "label": "Uso de CPU",
                "decay_per_tick": 0.02,
                "initial": 0.5,
            },
            {
                "id": "memoria",
                "label": "Uso de memoria",
                "decay_per_tick": 0.01,
                "initial": 0.3,
            },
        ],
        "traits": [
            {
                "id": "cauteloso",
                "label": "Comportamiento cauteloso",
                "need_weight_multipliers": {"cpu": 1.2},
            }
        ],
        "actions": [
            {
                "id": "optimizar_cpu",
                "label": "Optimizar procesos de CPU",
                "duration_ticks": 2,
                "reversibility": "reversible",
                "satisfies": {"cpu": 0.4},
                "costs": {"memoria": 0.1},
                "effects": {"cpu": 0.4, "memoria": -0.1},
                "requires": {"memoria": 0.2},
            },
            {
                "id": "reiniciar_servicio",
                "label": "Reiniciar servicio crítico",
                "duration_ticks": 5,
                "reversibility": "reversible_with_cost",
                "satisfies": {"memoria": 0.5},
                "effects": {"memoria": 0.5},
            },
        ],
        "agents": [
            {
                "id": "agente_monitor",
                "label": "Monitor de sistemas",
                "traits": {"cauteloso": 0.5},
                "initial_needs": {"cpu": 0.4, "memoria": 0.5},
                "tastes": [
                    {"target": "optimizar_cpu", "multiplier": 1.2},
                ],
            }
        ],
        "phases": [
            {
                "id": "pico_laboral",
                "label": "Horas pico",
                "starts_at": 100,
            }
        ],
        "situations": [
            {
                "id": "alta_demanda",
                "label": "Carga inusual de peticiones",
            }
        ],
        "goals": [
            {
                "id": "mantener_rendimiento",
                "label": "Mantener CPU bajo control",
                "need": "cpu",
                "at_least": 0.8,
            }
        ],
        "health": {
            "may_go_unused": ["reiniciar_servicio"],
        },
    }


def build_pack(**overrides: Any) -> pack.ContentPack:
    """Construye un ContentPack a partir de base_pack_dict con modificaciones."""
    data = base_pack_dict()
    data.update(copy.deepcopy(overrides))
    return pack.ContentPack.model_validate(data)


def expect_rejection(fragment: str, **overrides: Any) -> None:
    """Verifica que el pack sea rechazado y que el mensaje contenga `fragment`."""
    data = base_pack_dict()
    data.update(copy.deepcopy(overrides))
    with pytest.raises(ValidationError) as caught:
        pack.ContentPack.model_validate(data)
    assert fragment in str(caught.value), (
        f"Rechazado, pero el mensaje no menciona {fragment!r}:\n{caught.value}"
    )


# ---------------------------------------------------------------------------
# Pack válido e índices cacheados
# ---------------------------------------------------------------------------


def test_pack_base_es_valido() -> None:
    """El pack base cumple todas las validaciones sin lanzar error."""
    p = build_pack()
    assert p.id == "pack_test"
    assert len(p.needs) == 2
    assert len(p.actions) == 2
    assert len(p.agents) == 1


def test_indices_cached_property_devuelven_mapeo_correcto_y_son_estables() -> None:
    """Los 7 índices son `@cached_property`, mapean id a Spec y conservan identidad."""
    p = build_pack()

    # Mapeos correctos
    assert p.need_by_id[ontology.NeedId("cpu")].label == "Uso de CPU"
    assert p.trait_by_id[ontology.TraitId("cauteloso")].label == "Comportamiento cauteloso"
    assert p.action_by_id[ontology.ActionId("optimizar_cpu")].duration_ticks == 2
    assert p.agent_by_id[ontology.AgentId("agente_monitor")].label == "Monitor de sistemas"
    assert p.phase_by_id[ontology.PhaseId("pico_laboral")].starts_at == 100
    assert p.situation_by_id[ontology.SituationId("alta_demanda")].id == "alta_demanda"
    assert p.goal_by_id[ontology.GoalId("mantener_rendimiento")].need == "cpu"

    # Identidad cached: la segunda llamada devuelve el mismo objeto
    assert p.need_by_id is p.need_by_id
    assert p.action_by_id is p.action_by_id
    assert p.agent_by_id is p.agent_by_id


# ---------------------------------------------------------------------------
# 1. _check_unique_ids
# ---------------------------------------------------------------------------


def test_check_unique_ids_detecta_necesidad_duplicada() -> None:
    """Dos necesidades con el mismo id son rechazadas."""
    expect_rejection(
        "necesidades",
        needs=[
            {"id": "cpu", "label": "CPU 1", "decay_per_tick": 0.01},
            {"id": "cpu", "label": "CPU 2", "decay_per_tick": 0.02},
        ],
    )


def test_check_unique_ids_detecta_accion_duplicada() -> None:
    """Dos acciones con el mismo id son rechazadas."""
    expect_rejection(
        "acciones",
        actions=[
            {"id": "act1", "label": "a1", "duration_ticks": 1, "reversibility": "reversible"},
            {"id": "act1", "label": "a2", "duration_ticks": 1, "reversibility": "reversible"},
        ],
    )


def test_check_unique_ids_detecta_agente_duplicado() -> None:
    """Dos agentes con el mismo id son rechazados."""
    expect_rejection(
        "agentes",
        agents=[
            {"id": "agente_monitor", "label": "M1"},
            {"id": "agente_monitor", "label": "M2"},
        ],
    )


def test_check_unique_ids_detecta_rasgo_duplicado() -> None:
    """Dos rasgos con el mismo id son rechazados (I-003: rama sin cubrir)."""
    expect_rejection(
        "rasgos: id duplicado 'cauteloso'",
        traits=[
            {"id": "cauteloso", "label": "Cauteloso 1"},
            {"id": "cauteloso", "label": "Cauteloso 2"},
        ],
    )


def test_check_unique_ids_detecta_franja_duplicada() -> None:
    """Dos franjas con el mismo id son rechazadas (I-003: rama sin cubrir)."""
    expect_rejection(
        "franjas: id duplicado 'pico_laboral'",
        phases=[
            {"id": "pico_laboral", "label": "Pico 1", "starts_at": 100},
            {"id": "pico_laboral", "label": "Pico 2", "starts_at": 200},
        ],
    )


def test_check_unique_ids_detecta_situacion_duplicada() -> None:
    """Dos situaciones con el mismo id son rechazadas (I-003: rama sin cubrir)."""
    expect_rejection(
        "situaciones: id duplicado 'alta_demanda'",
        situations=[
            {"id": "alta_demanda", "label": "Alta demanda 1"},
            {"id": "alta_demanda", "label": "Alta demanda 2"},
        ],
    )


def test_check_unique_ids_detecta_meta_de_pack_duplicada() -> None:
    """Dos metas del pack con el mismo id son rechazadas (I-003: rama sin cubrir)."""
    expect_rejection(
        "metas: id duplicado 'meta_dup'",
        goals=[
            {"id": "meta_dup", "label": "Meta 1", "need": "cpu", "at_least": 0.5},
            {"id": "meta_dup", "label": "Meta 2", "need": "cpu", "at_least": 0.6},
        ],
    )


def test_check_unique_ids_detecta_meta_duplicada_dentro_de_un_agente() -> None:
    """Dos metas dentro del mismo agente con igual id son rechazadas (I-003: rama sin cubrir)."""
    expect_rejection(
        "metas de 'agente_monitor': id duplicado 'meta_dup'",
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "goals": [
                    {"id": "meta_dup", "label": "Meta 1", "need": "cpu", "at_least": 0.5},
                    {"id": "meta_dup", "label": "Meta 2", "need": "cpu", "at_least": 0.6},
                ],
            }
        ],
    )


# ---------------------------------------------------------------------------
# 2. _check_global_id_collisions
# ---------------------------------------------------------------------------


def test_check_global_id_collisions_detecta_colision_accion_agente() -> None:
    """Un id compartido entre una acción y un agente es ambiguo para gustos y se rechaza."""
    expect_rejection(
        "usado a la vez",
        actions=[
            {
                "id": "agente_monitor",
                "label": "Acción que choca con agente",
                "duration_ticks": 1,
                "reversibility": "reversible",
            },
            {
                "id": "reiniciar_servicio",
                "label": "Reiniciar",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "satisfies": {"cpu": 0.1},
                "effects": {"cpu": 0.1},
            },
        ],
    )


# ---------------------------------------------------------------------------
# 3. _check_need_refs
# ---------------------------------------------------------------------------


def test_check_need_refs_detecta_necesidad_desconocida_en_traits() -> None:
    """Un rasgo que referencia una necesidad inexistente es rechazado."""
    expect_rejection(
        "necesidad desconocida",
        traits=[
            {
                "id": "cauteloso",
                "label": "cauteloso",
                "need_weight_multipliers": {"inexistente": 1.5},
            }
        ],
    )


def test_check_need_refs_detecta_necesidad_desconocida_en_actions_requires() -> None:
    """Una acción que requiere una necesidad inexistente es rechazada."""
    expect_rejection(
        "necesidad desconocida",
        actions=[
            {
                "id": "optimizar_cpu",
                "label": "opt",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "requires": {"inexistente": 0.5},
            }
        ],
    )


def test_check_need_refs_detecta_necesidad_desconocida_en_actions_satisfies() -> None:
    """Una acción que satisface una necesidad inexistente es rechazada."""
    expect_rejection(
        "necesidad desconocida",
        actions=[
            {
                "id": "optimizar_cpu",
                "label": "opt",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "satisfies": {"fantasma": 0.5},
                "effects": {"fantasma": 0.5},
            }
        ],
    )


def test_check_need_refs_detecta_necesidad_desconocida_en_agents_initial_needs() -> None:
    """Un agente con initial_needs hacia un eje inexistente es rechazado."""
    expect_rejection(
        "necesidad desconocida",
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "initial_needs": {"inexistente": 0.5},
            }
        ],
    )


def test_check_need_refs_detecta_necesidad_desconocida_en_goals() -> None:
    """Una meta sobre una necesidad inexistente es rechazada."""
    expect_rejection(
        "necesidad desconocida",
        goals=[
            {
                "id": "meta_invalida",
                "label": "Meta invalida",
                "need": "fantasma",
                "at_least": 0.5,
            }
        ],
    )


def test_check_need_refs_detecta_necesidad_desconocida_en_situations() -> None:
    """Una situación que afecta a una necesidad inexistente es rechazada."""
    expect_rejection(
        "necesidad desconocida",
        situations=[
            {
                "id": "alta_demanda",
                "label": "Alta demanda",
                "effects": {"fantasma": -0.1},
            }
        ],
    )


# ---------------------------------------------------------------------------
# 4. _check_trait_refs
# ---------------------------------------------------------------------------


def test_check_trait_refs_detecta_rasgo_desconocido_en_agents() -> None:
    """Un agente con un rasgo no declarado en traits es rechazado."""
    expect_rejection(
        "rasgo desconocida/o",
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "traits": {"rasgo_inventado": 0.5},
            }
        ],
    )


def test_check_trait_refs_detecta_rasgo_desconocido_en_actions_requires_traits() -> None:
    """Una acción que exige un rasgo inexistente es rechazada."""
    expect_rejection(
        "rasgo desconocida/o",
        actions=[
            {
                "id": "optimizar_cpu",
                "label": "opt",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "requires_traits": {"rasgo_inventado": 0.5},
            }
        ],
    )


def test_check_trait_refs_detecta_rasgo_desconocido_en_goals_at_traits() -> None:
    """Una meta con at_traits hacia un rasgo inexistente es rechazada."""
    expect_rejection(
        "rasgo desconocida/o",
        goals=[
            {
                "id": "mantener_rendimiento",
                "label": "Meta",
                "need": "cpu",
                "at_traits": {"rasgo_inexistente": 0.5},
            }
        ],
    )


# ---------------------------------------------------------------------------
# 5. _check_phase_refs
# ---------------------------------------------------------------------------


def test_check_phase_refs_detecta_franja_desconocida_en_actions() -> None:
    """Una acción con phase_multipliers apuntando a una franja inexistente es rechazada."""
    expect_rejection(
        "franja desconocida/o",
        actions=[
            {
                "id": "optimizar_cpu",
                "label": "opt",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "phase_multipliers": {"franja_no_declarada": 1.5},
            }
        ],
    )


# ---------------------------------------------------------------------------
# 6. _check_situation_refs
# ---------------------------------------------------------------------------


def test_check_situation_refs_detecta_situacion_desconocida_en_actions() -> None:
    """Una acción con situation_multipliers hacia una situación inexistente es rechazada."""
    expect_rejection(
        "situación desconocida/o",
        actions=[
            {
                "id": "optimizar_cpu",
                "label": "opt",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "situation_multipliers": {"situacion_inexistente": 1.2},
            }
        ],
    )


# ---------------------------------------------------------------------------
# 7. _check_goal_refs
# ---------------------------------------------------------------------------


def test_check_goal_refs_detecta_prerrequisito_de_meta_inexistente() -> None:
    """Una meta cuyo requires_goal no existe es rechazada."""
    expect_rejection(
        "espera a 'meta_fantasma'",
        goals=[
            {
                "id": "mantener_rendimiento",
                "label": "Meta",
                "need": "cpu",
                "at_least": 0.5,
                "requires_goal": "meta_fantasma",
            }
        ],
    )


def test_check_goal_refs_detecta_ciclo_en_prerrequisitos_de_metas() -> None:
    """Un ciclo en requires_goal (g1 espera a g2 y g2 espera a g1) es rechazado."""
    expect_rejection(
        "ciclo de metas",
        goals=[
            {
                "id": "g1",
                "label": "Meta 1",
                "need": "cpu",
                "at_least": 0.5,
                "requires_goal": "g2",
            },
            {
                "id": "g2",
                "label": "Meta 2",
                "need": "cpu",
                "at_least": 0.5,
                "requires_goal": "g1",
            },
        ],
    )


def test_check_goal_refs_detecta_requires_goal_que_no_pertenece_al_agente() -> None:
    """`requires_goal` de una meta de agente que apunta a una meta que no es
    suya (aunque exista, a nivel de pack) es rechazado (I-003: rama sin cubrir,
    `pack.py:318`).
    """
    expect_rejection(
        "agente 'agente_monitor', meta 'meta_local': espera a 'meta_del_pack', "
        "que no es una meta suya",
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "goals": [
                    {
                        "id": "meta_local",
                        "label": "Meta local",
                        "need": "cpu",
                        "at_least": 0.5,
                        "requires_goal": "meta_del_pack",
                    }
                ],
            }
        ],
        goals=[
            {
                "id": "meta_del_pack",
                "label": "Meta del pack",
                "need": "cpu",
                "at_least": 0.5,
            }
        ],
    )


# ---------------------------------------------------------------------------
# 8. _check_goals_are_reachable
# ---------------------------------------------------------------------------


def test_check_goals_are_reachable_detecta_meta_inalcanzable_sin_accion_que_alivie() -> None:
    """Una meta sobre una necesidad que ninguna acción del pack alivia es rechazada."""
    expect_rejection(
        "ninguna acción sube 'memoria'",
        goals=[
            {
                "id": "meta_memoria",
                "label": "Meta memoria",
                "need": "memoria",
                "at_least": 0.5,
            }
        ],
        actions=[
            {
                "id": "solo_cpu",
                "label": "Solo CPU",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "satisfies": {"cpu": 0.5},
                "effects": {"cpu": 0.5},
            }
        ],
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
            }
        ],
        health={},
    )


def test_check_goals_are_reachable_detecta_meta_de_agente_sobre_necesidad_que_nadie_sube() -> None:
    """Una meta DENTRO de un agente sobre un eje que ninguna acción sube es
    rechazada (I-003: rama de agente sin cubrir; la de pack ya estaba cubierta).
    """
    expect_rejection(
        "agente 'agente_monitor', meta 'meta_agente_inalcanzable': ninguna acción "
        "sube 'disco_temporal'",
        needs=[
            {"id": "cpu", "label": "Uso de CPU", "decay_per_tick": 0.02, "initial": 0.5},
            {"id": "memoria", "label": "Uso de memoria", "decay_per_tick": 0.01, "initial": 0.3},
            {"id": "disco_temporal", "label": "Disco temporal", "decay_per_tick": 0.01},
        ],
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "goals": [
                    {
                        "id": "meta_agente_inalcanzable",
                        "label": "Meta inalcanzable",
                        "need": "disco_temporal",
                        "at_least": 0.5,
                    }
                ],
            }
        ],
    )


def test_check_goals_are_reachable_detecta_at_traits_de_meta_de_agente_inalcanzable() -> None:
    """`at_traits` de una meta DE AGENTE que ninguna otra meta suya sube es
    rechazado (I-003: rama sin cubrir).
    """
    expect_rejection(
        "agente 'agente_monitor', meta 'meta_trait': pide 'cauteloso' a 0.9, "
        "tiene 0.5 y ninguna de sus metas se lo sube",
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "traits": {"cauteloso": 0.5},
                "goals": [
                    {
                        "id": "meta_trait",
                        "label": "Meta trait",
                        "need": "cpu",
                        "at_traits": {"cauteloso": 0.9},
                    }
                ],
            }
        ],
    )


def test_check_goals_are_reachable_detecta_at_traits_de_meta_de_pack_inalcanzable() -> None:
    """`at_traits` de una meta DE PACK que ninguna otra meta del pack sube es
    rechazado (I-003: rama sin cubrir).
    """
    expect_rejection(
        "meta 'meta_trait_pack': pide 'cauteloso' a 0.9 y ninguna meta se lo sube",
        goals=[
            {
                "id": "meta_trait_pack",
                "label": "Meta trait pack",
                "need": "cpu",
                "at_traits": {"cauteloso": 0.9},
            }
        ],
    )


# ---------------------------------------------------------------------------
# 9. _check_taste_refs
# ---------------------------------------------------------------------------


def test_check_taste_refs_detecta_gusto_hacia_destino_inexistente() -> None:
    """Un gusto hacia un identificador que no es acción ni agente es rechazado."""
    expect_rejection(
        "gusto hacia 'destino_inexistente'",
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "tastes": [{"target": "destino_inexistente", "multiplier": 1.5}],
            }
        ],
    )


def test_check_taste_refs_admite_gusto_hacia_accion_o_agente() -> None:
    """Un gusto hacia una acción o hacia un agente declarado es válido."""
    p = build_pack(
        agents=[
            {
                "id": "agente_1",
                "label": "Agente 1",
                "tastes": [
                    {"target": "optimizar_cpu", "multiplier": 1.5},
                    {"target": "agente_2", "multiplier": 0.8},
                ],
            },
            {"id": "agente_2", "label": "Agente 2"},
        ]
    )
    assert len(p.agents) == 2


# ---------------------------------------------------------------------------
# 10. _check_costs_are_declared
# ---------------------------------------------------------------------------


def test_check_costs_are_declared_detecta_efecto_negativo_sin_coste() -> None:
    """Un efecto negativo sin declaración en `costs` es rechazado."""
    expect_rejection(
        "no declara coste",
        actions=[
            {
                "id": "accion_tramposa",
                "label": "Acción tramposa",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "effects": {"cpu": -0.3},
            }
        ],
    )


def test_check_costs_are_declared_permite_coste_cero_explicito() -> None:
    """Declarar coste 0.0 explícito permite efectos negativos no previstos conscientemente."""
    data = base_pack_dict()
    data["actions"].append(
        {
            "id": "accion_imprudente",
            "label": "Imprudente intencional",
            "duration_ticks": 1,
            "reversibility": "reversible",
            "costs": {"cpu": 0.0},
            "effects": {"cpu": -0.3},
        }
    )
    p = pack.ContentPack.model_validate(data)
    accion = p.action_by_id[ontology.ActionId("accion_imprudente")]
    assert accion.costs[ontology.NeedId("cpu")] == 0.0




# ---------------------------------------------------------------------------
# 11. _check_requirements_cover_what_they_spend
# ---------------------------------------------------------------------------


def test_check_requirements_cover_what_they_spend_detecta_requisito_menor_al_gasto() -> None:
    """Un requisito inferior a lo que la acción consume truncaría el efecto y es rechazado."""
    expect_rejection(
        "exige 0.1 de 'memoria' pero puede consumir hasta 0.2",
        actions=[
            {
                "id": "accion_deficitaria",
                "label": "Gasta más de lo que exige",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "requires": {"memoria": 0.1},
                "costs": {"memoria": 0.2},
                "effects": {"memoria": -0.2},
            }
        ],
    )


# ---------------------------------------------------------------------------
# 12. _check_actions_are_reachable
# ---------------------------------------------------------------------------


def test_check_actions_are_reachable_detecta_requisito_mayor_a_uno() -> None:
    """Una acción que exige un nivel de necesidad > 1.0 es inalcanzable y se rechaza."""
    expect_rejection(
        "el valor máximo de una necesidad es 1.0; jamás será ejecutable",
        actions=[
            {
                "id": "accion_imposible",
                "label": "Imposible",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "requires": {"cpu": 1.5},
            }
        ],
    )


def test_check_actions_are_reachable_detecta_requisito_que_nadie_alcanza() -> None:
    """Una acción que exige un nivel que ningún agente tiene y ninguna acción sube se rechaza."""
    expect_rejection(
        "ningún agente empieza con ese nivel y ninguna acción lo sube",
        actions=[
            {
                "id": "accion_bloqueada",
                "label": "Bloqueada",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "requires": {"memoria": 0.95},
            }
        ],
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "initial_needs": {"memoria": 0.1},
            }
        ],
    )


def test_check_actions_are_reachable_detecta_rasgo_requerido_que_ningun_agente_alcanza() -> None:
    """Una acción que exige un rasgo a un nivel que ningún agente tiene ni puede
    llegar a tener por sus metas es inalcanzable y se rechaza (I-004: la rama
    `if not alcanza_alguno` de `_check_actions_are_reachable`, alcanzable y sin
    test hasta ahora — no confundir con `if req_val > 1.0`, que es rama muerta
    porque `ActionSpec.requires_traits` ya está acotado a `[-1, 1]`).
    """
    expect_rejection(
        "acción 'requiere_prudencia': exige rasgo 'cauteloso' a 0.9, pero ningún "
        "agente lo alcanza ni tiene metas que lo suban",
        actions=[
            {
                "id": "optimizar_cpu",
                "label": "Optimizar procesos de CPU",
                "duration_ticks": 2,
                "reversibility": "reversible",
                "satisfies": {"cpu": 0.4},
                "costs": {"memoria": 0.1},
                "effects": {"cpu": 0.4, "memoria": -0.1},
                "requires": {"memoria": 0.2},
            },
            {
                "id": "reiniciar_servicio",
                "label": "Reiniciar servicio crítico",
                "duration_ticks": 5,
                "reversibility": "reversible_with_cost",
                "satisfies": {"memoria": 0.5},
                "effects": {"memoria": 0.5},
            },
            {
                "id": "requiere_prudencia",
                "label": "Exige mucha prudencia",
                "duration_ticks": 1,
                "reversibility": "reversible",
                "requires_traits": {"cauteloso": 0.9},
            },
        ],
    )


# ---------------------------------------------------------------------------
# 13. _check_health_refs
# ---------------------------------------------------------------------------


def test_check_health_refs_detecta_accion_desconocida_en_may_go_unused() -> None:
    """health.may_go_unused apuntando a una acción inexistente se rechaza."""
    expect_rejection(
        "health.may_go_unused: acción desconocida 'accion_inexistente'",
        health={"may_go_unused": ["accion_inexistente"]},
    )


# ---------------------------------------------------------------------------
# 14. _check_declared_bounds
# ---------------------------------------------------------------------------


def test_check_declared_bounds_detecta_initial_needs_fuera_de_rango() -> None:
    """AgentSpec.initial_needs fuera de [0.0, 1.0] es rechazado."""
    # Mayor a 1.0
    expect_rejection(
        "initial_needs['cpu'] = 1.25, fuera de [0.0, 1.0]",
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "initial_needs": {"cpu": 1.25},
            }
        ],
    )

    # Menor a 0.0
    expect_rejection(
        "initial_needs['cpu'] = -0.1, fuera de [0.0, 1.0]",
        agents=[
            {
                "id": "agente_monitor",
                "label": "Monitor",
                "initial_needs": {"cpu": -0.1},
            }
        ],
    )


def test_check_declared_bounds_detecta_goal_at_traits_fuera_de_rango() -> None:
    """GoalSpec.at_traits fuera de [-2.0, 2.0] es rechazado."""
    expect_rejection(
        "at_traits['cauteloso'] = 2.5, fuera de [-2.0, 2.0]",
        goals=[
            {
                "id": "mantener_rendimiento",
                "label": "Meta",
                "need": "cpu",
                "at_traits": {"cauteloso": 2.5},
            }
        ],
    )


# ---------------------------------------------------------------------------
# Reporte consolidado de errores múltiples
# ---------------------------------------------------------------------------


def test_pack_malformado_con_multiples_errores_reporta_todos_simultaneamente() -> None:
    """Un pack con 4+ fallos de distintas clases reporta TODOS los problemas
    en una sola excepción consolidated, en lugar de fallar solo por el primero.
    """
    data = base_pack_dict()
    # 1. Id de necesidad duplicado
    data["needs"].append({"id": "cpu", "label": "CPU Duplicada", "decay_per_tick": 0.01})
    # 2. Rasgo desconocido en agente
    data["agents"][0]["traits"] = {"rasgo_desconocido": 0.5}
    # 3. Efecto negativo sin coste
    data["actions"].append({
        "id": "accion_sin_coste",
        "label": "Sin coste",
        "duration_ticks": 1,
        "reversibility": "reversible",
        "effects": {"cpu": -0.4},
    })
    # 4. Cota violada en initial_needs
    data["agents"][0]["initial_needs"] = {"cpu": 3.5}

    with pytest.raises(ValidationError) as excinfo:
        pack.ContentPack.model_validate(data)

    msg = str(excinfo.value)
    # Comprobar que los 4 errores están simultáneamente en el mensaje
    assert "necesidades: id duplicado 'cpu'" in msg
    assert "rasgo desconocida/o 'rasgo_desconocido'" in msg
    assert "acción 'accion_sin_coste': el efecto sobre 'cpu' es negativo" in msg
    assert "initial_needs['cpu'] = 3.5, fuera de [0.0, 1.0]" in msg


# ---------------------------------------------------------------------------
# describe_pack y load_pack
# ---------------------------------------------------------------------------


def test_describe_pack_retorna_conteo_exacto_por_coleccion() -> None:
    """describe_pack devuelve un mapeo con el recuento exacto de cada colección."""
    p = build_pack()
    conteo = pack.describe_pack(p)
    assert conteo == {
        "needs": 2,
        "traits": 1,
        "actions": 2,
        "agents": 1,
        "phases": 1,
        "situations": 1,
        "goals": 1,
    }


def test_load_pack_carga_pack_valido_desde_archivo(tmp_path: Path) -> None:
    """`load_pack` carga un archivo YAML válido y retorna la instancia de ContentPack."""
    ruta = tmp_path / "pack.yaml"
    import yaml
    ruta.write_text(yaml.dump(base_pack_dict()), encoding="utf-8")

    cargado = pack.load_pack(ruta)
    assert cargado.id == "pack_test"
    assert len(cargado.actions) == 2


def test_load_pack_falla_con_pack_error_en_archivo_invalido(tmp_path: Path) -> None:
    """`load_pack` encapsula ValidationError en PackError al fallar."""
    ruta = tmp_path / "invalido.yaml"
    datos = base_pack_dict()
    datos["needs"].append({"id": "cpu", "label": "dup", "decay_per_tick": 0.01})
    import yaml
    ruta.write_text(yaml.dump(datos), encoding="utf-8")

    with pytest.raises(pack.PackError) as excinfo:
        pack.load_pack(ruta)
    assert "no se pudo cargar" in str(excinfo.value)
    assert "id duplicado 'cpu'" in str(excinfo.value)


def test_load_pack_usa_safe_load_rechaza_constructores_python(tmp_path: Path) -> None:
    """`load_pack` rechaza YAML malicioso o con constructores Python arbitrarios."""
    ruta = tmp_path / "malicioso.yaml"
    ruta.write_text("!!python/object/apply:os.system ['echo pwned']", encoding="utf-8")

    with pytest.raises(pack.PackError):
        pack.load_pack(ruta)


# ---------------------------------------------------------------------------
# Determinismo del mensaje de error entre subprocesos (PYTHONHASHSEED)
# ---------------------------------------------------------------------------


def test_error_mensaje_determinista_entre_subprocesos_con_distinto_hashseed() -> None:
    """El mensaje de error de un pack malformado es byte a byte idéntico
    entre ejecuciones en subprocesos con diferente PYTHONHASHSEED (E-001, E-012).
    """
    code = """
import aida.engine.pack as pack
from pydantic import ValidationError

data = {
    "id": "pack_determinismo",
    "label": "Pack determinismo",
    "needs": [
        {"id": "n1", "label": "N1", "decay_per_tick": 0.01},
        {"id": "n2", "label": "N2", "decay_per_tick": 0.01},
        {"id": "n1", "label": "N1 dup", "decay_per_tick": 0.01},
    ],
    "traits": [
        {
            "id": "t1",
            "label": "T1",
            "need_weight_multipliers": {"desconocida_a": 1.2, "desconocida_b": 1.4},
        },
    ],
    "actions": [
        {
            "id": "a1",
            "label": "A1",
            "duration_ticks": 1,
            "reversibility": "reversible",
            "effects": {"n1": -0.2, "n2": -0.5},
        },
    ],

    "agents": [
        {"id": "ag1", "label": "AG1", "initial_needs": {"n1": 2.5, "n2": -0.5}},
    ],
    "health": {
        "may_go_unused": [
            "accion_invalida_z",
            "accion_invalida_a",
            "accion_invalida_m",
            "accion_invalida_d",
            "accion_invalida_k",
            "accion_invalida_b",
            "accion_invalida_x",
            "accion_invalida_c",
            "accion_invalida_w",
            "accion_invalida_p",
        ],
    },
}

try:
    pack.ContentPack.model_validate(data)
except ValidationError as exc:
    # Imprimir el mensaje de error exactamente
    print(str(exc))
"""
    seeds = ["0", "1", "2", "42", "1337"]
    outputs: list[str] = []

    for seed in seeds:
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        outputs.append(res.stdout.strip())

    # Todos los mensajes de error entre subprocesos con distintas semillas deben ser idénticos
    assert len(set(outputs)) == 1, (
        f"Divergencia de mensaje de error entre semillas: {outputs}"
    )


def test_determinismo_health_may_go_unused_identico_en_40_semillas() -> None:
    """El mensaje de `health.may_go_unused` es idéntico entre 40 semillas de
    `PYTHONHASHSEED` cuando el motor ordena antes de recorrer el `frozenset`.

    I-008: esta función tenía una segunda mitad que construía un `frozenset` a
    pelo, sin importar el motor, y comprobaba que CPython lo itera en orden
    variable entre procesos. Es una propiedad del intérprete, no del código
    bajo prueba: mutar `pack.py` —incluso borrar el `sorted()` real— no puede
    ponerla en rojo. Se retira y queda solo la mitad que sí muerde: la que
    ejercita `ContentPack.model_validate` de verdad.
    """
    code_with_sorted = """
import sys
from pydantic import ValidationError
import aida.engine.pack as pack

data = {
    "id": "pack_det_health",
    "label": "Pack Health",
    "needs": [{"id": "n1", "label": "N1", "decay_per_tick": 0.01}],
    "actions": [{"id": "a1", "label": "A1", "duration_ticks": 1, "reversibility": "reversible"}],
    "agents": [{"id": "ag1", "label": "AG1"}],
    "health": {
        "may_go_unused": [
            "accion_z", "accion_a", "accion_m", "accion_d", "accion_k",
            "accion_b", "accion_x", "accion_c", "accion_w", "accion_p",
        ],
    },
}
try:
    pack.ContentPack.model_validate(data)
except ValidationError as exc:
    sys.stdout.write(str(exc))
"""
    # 40 semillas consecutivas
    seeds = [str(i) for i in range(40)]
    outputs_with_sorted: set[str] = set()

    for seed in seeds:
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run(
            [sys.executable, "-c", code_with_sorted],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        outputs_with_sorted.add(res.stdout.strip())

    # Con sorted() implementado en el motor, todas las 40 semillas deben producir el mismo texto
    assert len(outputs_with_sorted) == 1, (
        f"El mensaje de health.may_go_unused divergió entre semillas con sorted(): "
        f"{len(outputs_with_sorted)} variantes"
    )


# ---------------------------------------------------------------------------
# Pack de plantilla oficial (`packs/_plantilla/pack.yaml`)
# ---------------------------------------------------------------------------


def test_plantilla_pack_yaml_carga_y_valida_limpiamente() -> None:
    """`packs/_plantilla/pack.yaml` carga sin errores vía `load_pack` y valida."""
    ruta_plantilla = Path(__file__).resolve().parents[1] / "packs" / "_plantilla" / "pack.yaml"
    assert ruta_plantilla.exists(), f"Falta el archivo de plantilla en {ruta_plantilla}"

    p = pack.load_pack(ruta_plantilla)
    assert p.id == "infraestructura_base"
    assert len(p.needs) >= 3
    assert len(p.actions) >= 4
    assert len(p.agents) >= 1
    assert len(p.traits) >= 2


def test_plantilla_pack_yaml_ejercita_requisitos_de_u2() -> None:
    """El pack de plantilla ejercita el vertical de infraestructura, max_debt,
    escalates_after, los 3 niveles de reversibilidad y salud del sistema.
    """
    ruta_plantilla = Path(__file__).resolve().parents[1] / "packs" / "_plantilla" / "pack.yaml"
    p = pack.load_pack(ruta_plantilla)

    # Vertical de infraestructura
    need_ids = {n.id for n in p.needs}
    assert "espacio_libre" in need_ids
    assert "certificados_vigentes" in need_ids
    assert "replicas_al_dia" in need_ids

    # max_debt y escalates_after
    assert any(n.max_debt > 0.0 for n in p.needs), (
        "Debe haber al menos un eje con max_debt > 0"
    )
    assert any(n.escalates_after > 0 for n in p.needs), (
        "Debe haber al menos un eje con escalates_after > 0"
    )


    # Los 3 niveles de reversibilidad en acciones
    reversibilidades = {a.reversibility for a in p.actions}
    assert ontology.Reversibility.REVERSIBLE in reversibilidades
    assert ontology.Reversibility.REVERSIBLE_WITH_COST in reversibilidades
    assert ontology.Reversibility.IRREVERSIBLE in reversibilidades

    # Salud
    assert p.health is not None
