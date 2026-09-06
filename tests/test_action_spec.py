"""Pruebas de `ActionSpec` y `Reversibility` en `aida.engine.ontology`.

Cubren la unidad U2:
- Contrato de `Reversibility`: enum con 3 niveles:
  "reversible", "reversible_with_cost", "irreversible".
- `reversibility` obligatorio en `ActionSpec` sin valor por defecto.

- Validador de coherencia de signos: `satisfies` exige delta > 0.0 en `effects`,
  `costs` exige delta < 0.0 en `effects`.
- Cotas de `requires_traits` en [-1.0, 1.0], `duration_ticks` >= 1, `saturation_gain` >= 0.0.
- Ausencia de los 17 campos de simulación social/espacial descartados de Agora.
- Determinismo de serialización JSON de `tags` (frozenset) entre subprocesos
  con distintas semillas de PYTHONHASHSEED y 12+ etiquetas (E-012).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

import aida.engine.ontology as ontology

# Lista derivada de .claude/workflow/05-plan-u2.md §4 (no derivada de dir(ActionSpec) per E-010)
DISCARDED_FIELDS = [
    "elastic",
    "requires_affordance",
    "requires_rank",
    "requires_faccion",
    "requires_prestigio",
    "requires_lealtad",
    "lealtad",
    "forbids_faccion",
    "funda_faccion",
    "requires_edad",
    "forbids_condition",
    "requires_company",
    "effects_per_company",
    "requires_relation",
    "target",
    "inflicts",
    "cures",
]

# 12+ tags para evitar falsos verdes casuales per E-012
TWELVE_TAGS = [
    "seguridad", "mantenimiento", "red", "almacenamiento",
    "auditoria", "backup", "optimizacion", "limpieza",
    "sincronizacion", "verificacion", "notificacion", "archivado",
]


def _action(**overrides: Any) -> ontology.ActionSpec:
    """Construye un `ActionSpec` mínimo válido."""
    base: dict[str, Any] = {
        "id": ontology.ActionId("inspeccionar"),
        "label": "Inspeccionar estado",
        "duration_ticks": 1,
        "reversibility": ontology.Reversibility.REVERSIBLE,
    }
    base.update(overrides)
    return ontology.ActionSpec(**base)


# ---------------------------------------------------------------------------
# Reversibility y obligatoriedad en ActionSpec
# ---------------------------------------------------------------------------


def test_reversibility_enum_valores_esperados() -> None:
    """Reversibility declara exactamente los 3 niveles estipulados."""
    assert ontology.Reversibility.REVERSIBLE.value == "reversible"
    assert ontology.Reversibility.REVERSIBLE_WITH_COST.value == "reversible_with_cost"
    assert ontology.Reversibility.IRREVERSIBLE.value == "irreversible"
    assert len(ontology.Reversibility) == 3


def test_actionspec_admite_los_tres_niveles_de_reversibilidad() -> None:
    """Una acción se construye con cualquiera de los tres valores válidos."""
    for rev in (
        ontology.Reversibility.REVERSIBLE,
        ontology.Reversibility.REVERSIBLE_WITH_COST,
        ontology.Reversibility.IRREVERSIBLE,
    ):
        act = _action(reversibility=rev)
        assert act.reversibility == rev


def test_actionspec_rechaza_instanciacion_sin_reversibility() -> None:
    """`reversibility` es obligatorio y no tiene valor por defecto."""
    datos: dict[str, Any] = {
        "id": ontology.ActionId("act_sin_rev"),
        "label": "Sin reversibilidad",
        "duration_ticks": 1,
    }
    with pytest.raises(ValidationError) as excinfo:
        ontology.ActionSpec.model_validate(datos)
    assert "reversibility" in str(excinfo.value)



def test_actionspec_rechaza_reversibilidad_invalida() -> None:
    """Un valor arbitrario de reversibilidad es rechazado."""
    with pytest.raises(ValidationError):
        _action(reversibility="invalida")


# ---------------------------------------------------------------------------
# Validador de coherencia de signos (satisfies y costs vs effects)
# ---------------------------------------------------------------------------


def test_sign_coherence_accion_valida_con_satisfies_y_costs() -> None:
    """Acción con satisfies (>0) y costs (<0) coherentes en effects es válida."""
    act = _action(
        satisfies={ontology.NeedId("n1"): 0.8},
        costs={ontology.NeedId("n2"): 0.3},
        effects={
            ontology.NeedId("n1"): 0.6,
            ontology.NeedId("n2"): -0.3,
        },
    )
    assert act.satisfies[ontology.NeedId("n1")] == 0.8
    assert act.costs[ontology.NeedId("n2")] == 0.3
    assert act.effects[ontology.NeedId("n1")] == 0.6
    assert act.effects[ontology.NeedId("n2")] == -0.3


def test_sign_coherence_satisfies_requiere_aparicion_en_effects() -> None:
    """Una clave en `satisfies` debe existir en `effects`."""
    with pytest.raises(ValidationError) as excinfo:
        _action(
            satisfies={ontology.NeedId("n1"): 0.5},
            effects={},
        )
    assert "satisfies" in str(excinfo.value)
    assert "n1" in str(excinfo.value)


def test_sign_coherence_satisfies_requiere_efecto_positivo() -> None:
    """Una clave en `satisfies` con efecto <= 0.0 es rechazada."""
    # Efecto cero
    with pytest.raises(ValidationError) as excinfo:
        _action(
            satisfies={ontology.NeedId("n1"): 0.5},
            effects={ontology.NeedId("n1"): 0.0},
        )
    assert "satisfies" in str(excinfo.value)

    # Efecto negativo
    with pytest.raises(ValidationError) as excinfo:
        _action(
            satisfies={ontology.NeedId("n1"): 0.5},
            effects={ontology.NeedId("n1"): -0.2},
        )
    assert "satisfies" in str(excinfo.value)


def test_sign_coherence_costs_requiere_aparicion_en_effects() -> None:
    """Una clave en `costs` debe existir en `effects`."""
    with pytest.raises(ValidationError) as excinfo:
        _action(
            costs={ontology.NeedId("n1"): 0.2},
            effects={},
        )
    assert "costs" in str(excinfo.value)
    assert "n1" in str(excinfo.value)


def test_sign_coherence_costs_requiere_efecto_negativo() -> None:
    """Una clave en `costs` con efecto >= 0.0 es rechazada."""
    # Efecto cero
    with pytest.raises(ValidationError) as excinfo:
        _action(
            costs={ontology.NeedId("n1"): 0.2},
            effects={ontology.NeedId("n1"): 0.0},
        )
    assert "costs" in str(excinfo.value)

    # Efecto positivo
    with pytest.raises(ValidationError) as excinfo:
        _action(
            costs={ontology.NeedId("n1"): 0.2},
            effects={ontology.NeedId("n1"): 0.4},
        )
    assert "costs" in str(excinfo.value)


def test_costs_con_valor_negativo_es_rechazado_por_non_negative_factor() -> None:
    """costs es NonNegativeFactor: un coste negativo (ganancia) es rechazado."""
    with pytest.raises(ValidationError):
        _action(
            costs={ontology.NeedId("n1"): -0.5},
            effects={ontology.NeedId("n1"): -0.5},
        )


# ---------------------------------------------------------------------------
# Cotas y validación de campos numéricos
# ---------------------------------------------------------------------------


def test_duration_ticks_minimo_uno() -> None:
    """duration_ticks debe ser >= 1."""
    assert _action(duration_ticks=1).duration_ticks == 1
    assert _action(duration_ticks=5).duration_ticks == 5

    with pytest.raises(ValidationError):
        _action(duration_ticks=0)
    with pytest.raises(ValidationError):
        _action(duration_ticks=-1)


def test_saturation_gain_no_negativo() -> None:
    """saturation_gain debe ser >= 0.0."""
    assert _action(saturation_gain=0.0).saturation_gain == 0.0
    assert _action(saturation_gain=0.75).saturation_gain == 0.75

    with pytest.raises(ValidationError):
        _action(saturation_gain=-0.1)


def test_requires_traits_acotado_a_menos_uno_y_uno() -> None:
    """requires_traits debe estar en [-1.0, 1.0]."""
    # Valores límite válidos
    assert _action(requires_traits={ontology.TraitId("t1"): -1.0}).requires_traits == {
        ontology.TraitId("t1"): -1.0
    }
    assert _action(requires_traits={ontology.TraitId("t1"): 1.0}).requires_traits == {
        ontology.TraitId("t1"): 1.0
    }

    # Valores fuera de cota
    with pytest.raises(ValidationError):
        _action(requires_traits={ontology.TraitId("t1"): 1.01})
    with pytest.raises(ValidationError):
        _action(requires_traits={ontology.TraitId("t1"): -1.01})


# ---------------------------------------------------------------------------
# Campos descartados de Agora
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", DISCARDED_FIELDS)
def test_los_17_campos_descartados_no_existen_en_actionspec(field_name: str) -> None:
    """Ninguno de los 17 campos descartados de simulación social/espacial
    debe figurar en el modelo de ActionSpec.
    """
    assert field_name not in ontology.ActionSpec.model_fields


@pytest.mark.parametrize("field_name", DISCARDED_FIELDS)
def test_actionspec_rechaza_campos_descartados_al_instanciar(field_name: str) -> None:
    """Spec tiene extra='forbid': pasar cualquier campo descartado levanta ValidationError."""
    with pytest.raises(ValidationError):
        _action(**{field_name: "prohibido"})


# ---------------------------------------------------------------------------
# Determinismo de serialización de tags (frozenset)
# ---------------------------------------------------------------------------


def test_actionspec_tags_serializa_ordenado_en_json() -> None:
    """field_serializer('tags', when_used='json') ordena alfabéticamente."""
    act = _action(tags=frozenset(ontology.Tag(t) for t in TWELVE_TAGS))
    raw_json = act.model_dump_json()
    data = json.loads(raw_json)
    assert data["tags"] == sorted(TWELVE_TAGS)


def test_actionspec_tags_serializacion_estable_entre_subprocesos() -> None:
    """La serialización de tags de ActionSpec es idéntica cruzando el límite
    del proceso con distintos valores de PYTHONHASHSEED (E-001, E-012).
    """
    code = """
import json
import aida.engine.ontology as ontology

tags = frozenset(ontology.Tag(t) for t in [
    "seguridad", "mantenimiento", "red", "almacenamiento",
    "auditoria", "backup", "optimizacion", "limpieza",
    "sincronizacion", "verificacion", "notificacion", "archivado"
])
act = ontology.ActionSpec(
    id=ontology.ActionId("a1"),
    label="a1",
    duration_ticks=1,
    reversibility=ontology.Reversibility.REVERSIBLE,
    tags=tags,
)
print(act.model_dump_json())
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

    # Todas las salidas entre subprocesos con distinta semilla deben ser idénticas
    assert len(set(outputs)) == 1, (
        f"Divergencia de serialización detectada entre semillas: {outputs}"
    )
