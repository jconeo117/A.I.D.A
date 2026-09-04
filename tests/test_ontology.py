"""Pruebas de la ontología invariante del motor (`aida.engine.ontology`).

Cubren la unidad U1a. Tres frentes:

- El contrato de `Spec`: inmutable, estricto con campos desconocidos y con
  los rangos declarados por cada campo.
- Los validadores propios de `NeedSpec`, `GoalSpec` y `SituationSpec`.
- El corte: lo que se descartó al copiar de Agora no debe reaparecer, y las
  dos piezas de la deuda de necesidad (`max_debt`, `escalates_after`) deben
  seguir declaradas con la semántica acordada.

No se prueba comportamiento de ejecución (eso vive en `state.py`, U1c):
aquí solo se prueba la declaración y su validación al construir.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import aida.engine.ontology as ontology
from aida.engine.ontology import (
    AgentSpec,
    GoalId,
    GoalSpec,
    HealthSpec,
    NeedId,
    NeedSpec,
    PhaseSpec,
    SituationId,
    SituationSpec,
    TasteSpec,
    TemperamentKnob,
    TraitId,
    TraitSpec,
    TuningSpec,
)


def _need(**overrides: object) -> NeedSpec:
    """Construye un `NeedSpec` mínimo válido, con overrides puntuales."""
    base: dict[str, object] = {
        "id": NeedId("necesidad"),
        "label": "Necesidad de prueba",
        "decay_per_tick": 0.05,
    }
    base.update(overrides)
    return NeedSpec(**base)


def _goal(**overrides: object) -> GoalSpec:
    """Construye un `GoalSpec` mínimo válido, con overrides puntuales."""
    base: dict[str, object] = {
        "id": GoalId("meta"),
        "label": "Meta de prueba",
        "need": NeedId("necesidad"),
        "at_least": 0.8,
    }
    base.update(overrides)
    return GoalSpec(**base)


def _situation(**overrides: object) -> SituationSpec:
    """Construye un `SituationSpec` mínimo válido, con overrides puntuales."""
    base: dict[str, object] = {
        "id": SituationId("situacion"),
        "label": "Situación de prueba",
    }
    base.update(overrides)
    return SituationSpec(**base)


# ---------------------------------------------------------------------------
# 1. El contrato de `Spec`: inmutable y estricto.
# ---------------------------------------------------------------------------
def test_un_spec_construido_no_se_puede_mutar() -> None:
    """Asignar a un campo de un `Spec` ya construido levanta `ValidationError`
    porque `Spec.model_config` fija `frozen=True`.
    """
    # Arrange
    necesidad = _need()

    # Act / Assert
    with pytest.raises(ValidationError):
        necesidad.label = "otro nombre"  # type: ignore[misc]


def test_un_campo_desconocido_falla_en_need_spec() -> None:
    """Un campo que no existe en `NeedSpec` levanta `ValidationError`: la
    restricción `extra="forbid"` vive en la base `Spec`.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        NeedSpec(
            id=NeedId("necesidad"),
            label="Necesidad",
            decay_per_tick=0.05,
            campo_inventado=1,  # type: ignore[call-arg]
        )


def test_un_campo_desconocido_falla_en_tuning_spec() -> None:
    """La misma restricción de campo desconocido se cumple en `TuningSpec`,
    una clase distinta de `NeedSpec`: prueba de que vive en la base y no está
    repetida a mano en cada pieza.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        TuningSpec(campo_inventado=1)  # type: ignore[call-arg]


def test_decay_per_tick_fuera_de_rango_falla() -> None:
    """`decay_per_tick` está acotado a `[0, 1]`; un valor por encima levanta
    `ValidationError`.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        _need(decay_per_tick=1.5)


def test_urgency_exponent_no_positivo_falla() -> None:
    """`urgency_exponent` es estrictamente positivo; cero no es un valor
    válido.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        _need(urgency_exponent=0.0)


# ---------------------------------------------------------------------------
# 2. Validadores propios de cada pieza.
# ---------------------------------------------------------------------------
def test_need_spec_de_progresion_sin_mecanicas_de_carencia_es_valida() -> None:
    """Un eje de progresión que no declara decaimiento, deuda ni desgaste es
    coherente y se construye sin error.
    """
    # Arrange / Act
    necesidad = _need(
        progression=True,
        decay_per_tick=0.0,
        max_debt=0.0,
        strain_weight=0.0,
    )

    # Assert
    assert necesidad.progression is True


def test_need_spec_de_progresion_con_decaimiento_falla() -> None:
    """Un eje de progresión no admite `decay_per_tick` distinto de cero: no
    "se pierde" progreso con el tiempo, así que la combinación es prohibida.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        _need(progression=True, decay_per_tick=0.1)


def test_goal_spec_con_exactamente_una_condicion_at_least_es_valida() -> None:
    """Declarar solo `at_least` satisface el validador: exactamente una
    condición de logro.
    """
    # Arrange / Act
    meta = _goal(at_least=0.9, at_traits={})

    # Assert
    assert meta.at_least == 0.9


def test_goal_spec_con_exactamente_una_condicion_at_traits_es_valida() -> None:
    """Declarar solo `at_traits` también satisface el validador."""
    # Arrange / Act
    meta = _goal(at_least=None, at_traits={TraitId("valiente"): 0.5})

    # Assert
    assert meta.at_traits == {TraitId("valiente"): 0.5}


def test_goal_spec_sin_ninguna_condicion_falla() -> None:
    """Sin `at_least` ni `at_traits` la meta no se podría cerrar jamás, así
    que el validador lo rechaza.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        _goal(at_least=None, at_traits={})


def test_goal_spec_con_las_dos_condiciones_a_la_vez_falla() -> None:
    """Con `at_least` y `at_traits` a la vez habría que inventar una
    convención sobre si basta una u obligan las dos; el validador exige
    exactamente una.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        _goal(at_least=0.5, at_traits={TraitId("valiente"): 0.5})


def test_situation_spec_periodica_bien_formada_es_valida() -> None:
    """Una situación periódica con `every` y `lasts < every` es coherente:
    vuelve y también se apaga.
    """
    # Arrange / Act
    situacion = _situation(every=10, lasts=3)

    # Assert
    assert situacion.periodic is True


def test_situation_spec_que_vuelve_pero_dura_cero_falla() -> None:
    """`every` sin `lasts` describe una situación que nunca llega a ocurrir:
    el validador lo rechaza.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        _situation(every=10, lasts=0)


# ---------------------------------------------------------------------------
# 3. Lo que no debe existir: protege el corte de 02-corte-ontologia.md.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nombre",
    [
        "PlaceSpec",
        "RelationSpec",
        "FactionKindSpec",
        "TargetSpec",
        "BirthSpec",
        "TriggerSpec",
        "PropertySpec",
        "PropertyKind",
        "DebtSpec",
        "AflictionSpec",
        "ElasticSpec",
        "ActionSpec",
        "ClockSpec",
    ],
)
def test_pieza_descartada_o_diferida_no_existe_en_el_modulo(nombre: str) -> None:
    """Ninguna de las piezas descartadas por el corte, ni las diferidas a U3
    (`ActionSpec`) o U4 (`ClockSpec`), está definida en `ontology`.
    """
    # Arrange: el nombre bajo prueba llega parametrizado.

    # Act / Assert
    assert not hasattr(ontology, nombre)


@pytest.mark.parametrize(
    "nombre",
    ["PlaceId", "RelationId", "FactionId", "FactionKindId", "PropertyId", "Affordance"],
)
def test_identificador_descartado_no_existe_en_el_modulo(nombre: str) -> None:
    """Los identificadores propios de piezas descartadas (lugar, relación,
    facción, propiedad) no están definidos en `ontology`.
    """
    # Arrange: el nombre bajo prueba llega parametrizado.

    # Act / Assert
    assert not hasattr(ontology, nombre)


def test_need_spec_no_tiene_campo_rank_slowdown() -> None:
    """`rank_slowdown` no transfirió: progresión por rangos numerados no
    tiene consumidor previsto en AIDA.
    """
    # Arrange / Act / Assert
    assert "rank_slowdown" not in NeedSpec.model_fields


def test_goal_spec_no_tiene_campo_at_rank() -> None:
    """`at_rank` corre la misma suerte que `rank_slowdown`: sin consumidor,
    fuera del corte."""
    # Arrange / Act / Assert
    assert "at_rank" not in GoalSpec.model_fields


def test_tuning_spec_no_tiene_campo_travel_discount() -> None:
    """`travel_discount` es el único campo de `TuningSpec` que asumía
    geografía; el corte lo descarta explícitamente."""
    # Arrange / Act / Assert
    assert "travel_discount" not in TuningSpec.model_fields


def test_need_spec_no_tiene_campo_lethal_after() -> None:
    """`lethal_after` no sobrevive con ese nombre: se renombró a
    `escalates_after` porque cruzar el umbral ya no mata, escala."""
    # Arrange / Act / Assert
    assert "lethal_after" not in NeedSpec.model_fields


# ---------------------------------------------------------------------------
# 4. `TemperamentKnob`: exactamente seis miembros.
# ---------------------------------------------------------------------------
def test_temperament_knob_tiene_exactamente_seis_miembros() -> None:
    """El conjunto completo de nombres es el que se decidió tras la
    corrección del worker: ni faltan `SWITCH_COST`, `NOVELTY` y
    `LEARNING_RATE`, ni sobra ninguna perilla espacial o social.
    """
    # Arrange
    esperados = {
        "TEMPERATURE",
        "LOSS_AVERSION",
        "CRITICAL_THRESHOLD",
        "SWITCH_COST",
        "NOVELTY",
        "LEARNING_RATE",
    }

    # Act
    encontrados = {miembro.name for miembro in TemperamentKnob}

    # Assert
    assert encontrados == esperados
    assert len(TemperamentKnob) == 6


@pytest.mark.parametrize(
    "knob",
    list(TemperamentKnob),
    ids=[miembro.name for miembro in TemperamentKnob],
)
def test_cada_temperament_knob_corresponde_a_un_campo_de_tuning_spec(
    knob: TemperamentKnob,
) -> None:
    """El valor de cada miembro coincide con el nombre de un campo real de
    `TuningSpec`, que es lo que permite que el rasgo module el valor que el
    pack fija para el mundo entero.
    """
    # Arrange: el miembro bajo prueba llega parametrizado.

    # Act / Assert
    assert knob.value in TuningSpec.model_fields


# ---------------------------------------------------------------------------
# 5. Los dos mecanismos de deuda: solo la declaración.
# ---------------------------------------------------------------------------
def test_max_debt_existe_con_valor_por_defecto_cero() -> None:
    """`NeedSpec.max_debt` por defecto es `0.0`, lo que desactiva la deuda
    para un eje que no la declara explícitamente.
    """
    # Arrange / Act
    necesidad = _need()

    # Assert
    assert necesidad.max_debt == 0.0


def test_max_debt_negativo_falla() -> None:
    """`max_debt` no admite valores negativos: es un tope, no un balance."""
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        _need(max_debt=-0.1)


def test_escalates_after_existe_con_valor_por_defecto_cero() -> None:
    """`NeedSpec.escalates_after` por defecto es `0`, que significa "nunca
    escala".
    """
    # Arrange / Act
    necesidad = _need()

    # Assert
    assert necesidad.escalates_after == 0


def test_escalates_after_negativo_falla() -> None:
    """`escalates_after` cuenta ticks seguidos hundido; un valor negativo no
    tiene interpretación.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValidationError):
        _need(escalates_after=-1)


# ---------------------------------------------------------------------------
# Superficie pública mínima: las piezas del corte existen e instancian.
# ---------------------------------------------------------------------------
def test_las_once_piezas_del_corte_son_importables_y_construibles() -> None:
    """Cada clase que el corte decidió transferir existe en el módulo y
    admite al menos una construcción válida con los campos obligatorios.

    No repite los validadores ya cubiertos arriba: solo confirma que la
    superficie pública completa está presente.
    """
    # Arrange / Act
    necesidad = _need()
    rasgo = TraitSpec(id=TraitId("valiente"), label="Valiente")
    gusto = TasteSpec(target=ontology.ActionId("accion"), multiplier=1.5)
    meta = _goal()
    agente = AgentSpec(id=ontology.AgentId("aida"), label="AIDA")
    franja = PhaseSpec(id=ontology.PhaseId("manana"), label="Mañana", starts_at=0)
    situacion = _situation()
    salud = HealthSpec()
    calibracion = TuningSpec()

    # Assert
    assert necesidad.id == NeedId("necesidad")
    assert rasgo.id == TraitId("valiente")
    assert gusto.multiplier == 1.5
    assert meta.id == GoalId("meta")
    assert agente.id == ontology.AgentId("aida")
    assert franja.id == ontology.PhaseId("manana")
    assert situacion.id == SituationId("situacion")
    assert salud.days == 7
    assert calibracion.candidate_pool == 3
