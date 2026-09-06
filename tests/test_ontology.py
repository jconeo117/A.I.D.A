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

**Nota sobre el estilo de importación (arreglo de `30-validation.md`).** Este
archivo NO importa las piezas de `ontology` por nombre en la cabecera —ni
`NeedSpec`, ni `TraitId`, ni ninguna de las demás—. Se accede siempre como
`ontology.NeedSpec`, `ontology.TraitId`, etc., dentro del CUERPO de cada
función. La razón: un `from aida.engine.ontology import (NeedSpec, ...)` en
la cabecera falla con `ImportError` en cuanto CUALQUIERA de esos nombres
desaparece, y eso ocurre al COLECCIONAR el módulo, antes de que se ejecute ni
un test. El resultado es que los 60+ tests de este archivo se marcan en
error de golpe, sin importar cuál de ellos habría detectado la regresión de
verdad — y si dos nombres desaparecen a la vez, el mensaje solo delata el
primero de la tupla, ocultando el segundo. Referenciar por atributo de
módulo, en cambio, difiere la comprobación al momento en que CADA test se
ejecuta: falta un nombre y solo fallan, con un `AttributeError` legible que
dice exactamente cuál, los tests que de verdad lo usan. El resto de la
batería sigue corriendo y sigue informando.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import aida.engine.ontology as ontology


def _need(**overrides: object) -> ontology.NeedSpec:
    """Construye un `NeedSpec` mínimo válido, con overrides puntuales."""
    base: dict[str, object] = {
        "id": ontology.NeedId("necesidad"),
        "label": "Necesidad de prueba",
        "decay_per_tick": 0.05,
    }
    base.update(overrides)
    return ontology.NeedSpec(**base)


def _goal(**overrides: object) -> ontology.GoalSpec:
    """Construye un `GoalSpec` mínimo válido, con overrides puntuales."""
    base: dict[str, object] = {
        "id": ontology.GoalId("meta"),
        "label": "Meta de prueba",
        "need": ontology.NeedId("necesidad"),
        "at_least": 0.8,
    }
    base.update(overrides)
    return ontology.GoalSpec(**base)


def _situation(**overrides: object) -> ontology.SituationSpec:
    """Construye un `SituationSpec` mínimo válido, con overrides puntuales."""
    base: dict[str, object] = {
        "id": ontology.SituationId("situacion"),
        "label": "Situación de prueba",
    }
    base.update(overrides)
    return ontology.SituationSpec(**base)


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
        ontology.NeedSpec(
            id=ontology.NeedId("necesidad"),
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
        ontology.TuningSpec(campo_inventado=1)  # type: ignore[call-arg]


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
    meta = _goal(at_least=None, at_traits={ontology.TraitId("valiente"): 0.5})

    # Assert
    assert meta.at_traits == {ontology.TraitId("valiente"): 0.5}


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
        _goal(at_least=0.5, at_traits={ontology.TraitId("valiente"): 0.5})


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
    assert "rank_slowdown" not in ontology.NeedSpec.model_fields


def test_goal_spec_no_tiene_campo_at_rank() -> None:
    """`at_rank` corre la misma suerte que `rank_slowdown`: sin consumidor,
    fuera del corte."""
    # Arrange / Act / Assert
    assert "at_rank" not in ontology.GoalSpec.model_fields


def test_tuning_spec_no_tiene_campo_travel_discount() -> None:
    """`travel_discount` es el único campo de `TuningSpec` que asumía
    geografía; el corte lo descarta explícitamente."""
    # Arrange / Act / Assert
    assert "travel_discount" not in ontology.TuningSpec.model_fields


def test_need_spec_no_tiene_campo_lethal_after() -> None:
    """`lethal_after` no sobrevive con ese nombre: se renombró a
    `escalates_after` porque cruzar el umbral ya no mata, escala."""
    # Arrange / Act / Assert
    assert "lethal_after" not in ontology.NeedSpec.model_fields


# ---------------------------------------------------------------------------
# 3b. Lo que SÍ debe existir: los identificadores y las piezas que el corte
# conserva.
#
# `test_pieza_descartada_o_diferida_no_existe_en_el_modulo` y
# `test_identificador_descartado_no_existe_en_el_modulo` solo protegen que lo
# descartado no vuelva. Sin esto, borrar uno de los identificadores
# conservados (`02-corte-ontologia.md`, tabla "Se copia, adaptando") no lo
# detectaría ningún test: es el hallazgo 1 de `30-validation.md`.
#
# Ambos parametrizados acceden por `hasattr(ontology, nombre)`: cada
# variante es un test independiente que solo falla por sí mismo si ESE
# nombre desaparece, sin arrastrar al resto de la batería.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nombre",
    [
        "NeedId",
        "TraitId",
        "GoalId",
        "PhaseId",
        "SituationId",
        "ActionId",
        "AgentId",
        "Tag",
    ],
)
def test_identificador_conservado_existe_en_el_modulo(nombre: str) -> None:
    """Cada uno de los ocho identificadores que el corte decidió conservar
    sigue definido en `ontology`. Complementa, en sentido inverso, a
    `test_identificador_descartado_no_existe_en_el_modulo`.
    """
    # Arrange: el nombre bajo prueba llega parametrizado.

    # Act / Assert
    assert hasattr(ontology, nombre)


@pytest.mark.parametrize(
    "nombre",
    [
        "NeedSpec",
        "TraitSpec",
        "TasteSpec",
        "GoalSpec",
        "AgentSpec",
        "PhaseSpec",
        "SituationSpec",
        "HealthSpec",
        "TuningSpec",
        "TemperamentKnob",
        "GoalStatus",
    ],
)
def test_pieza_conservada_existe_en_el_modulo(nombre: str) -> None:
    """Ancla, con un `hasattr` propio e independiente de cualquier otro uso
    incidental en el resto del archivo, cada una de las clases y enums que
    el corte conserva. Es el arreglo de C1 (`30-validation.md`): borrar
    cualquiera de estos nombres solo hace fallar ESTE caso concreto, con un
    mensaje que dice exactamente cuál falta — no interrumpe la colección de
    los demás 60+ tests del archivo, como sí hacía el `ImportError` de un
    `from aida.engine.ontology import (...)` en la cabecera.
    """
    # Arrange: el nombre bajo prueba llega parametrizado.

    # Act / Assert
    assert hasattr(ontology, nombre)


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
    encontrados = {miembro.name for miembro in ontology.TemperamentKnob}

    # Assert
    assert encontrados == esperados
    assert len(ontology.TemperamentKnob) == 6


def test_cada_temperament_knob_corresponde_a_un_campo_de_tuning_spec() -> None:
    """El valor de cada miembro coincide con el nombre de un campo real de
    `TuningSpec`, que es lo que permite que el rasgo module el valor que el
    pack fija para el mundo entero.

    Recorre los miembros en un bucle dentro del cuerpo del test, no vía
    `@pytest.mark.parametrize(..., list(TemperamentKnob), ...)`: esa lista se
    evaluaría al DECORAR la función, es decir, al coleccionar el módulo — si
    el enum desapareciera, la falta reventaría ahí, arrastrando otra vez toda
    la colección del archivo, exactamente el defecto que el arreglo de C1
    corrige en el resto de este archivo. Con el bucle dentro del cuerpo, la
    ausencia se nota al EJECUTAR este test concreto, y solo este.
    """
    # Arrange / Act / Assert
    for knob in ontology.TemperamentKnob:
        assert knob.value in ontology.TuningSpec.model_fields


# ---------------------------------------------------------------------------
# 4b. `GoalStatus`: exactamente cuatro estados, ni más ni menos.
#
# Es un enum, no un `Spec`, y por eso no lo cubría
# `test_las_once_piezas_del_corte_son_importables_y_construibles`: la mutación
# B de `30-validation.md` lo borró entero y la batería no se enteró.
# ---------------------------------------------------------------------------
def test_goal_status_tiene_exactamente_cuatro_estados() -> None:
    """El conjunto completo de estados es `LOCKED`, `ACTIVE`, `ACHIEVED` y
    `FAILED`: ni falta ninguno, ni sobra uno que confunda logrado con
    perdido, que el corte distingue a propósito porque mueven rasgos
    opuestos.
    """
    # Arrange
    esperados = {"LOCKED", "ACTIVE", "ACHIEVED", "FAILED"}

    # Act
    encontrados = {miembro.name for miembro in ontology.GoalStatus}

    # Assert
    assert encontrados == esperados
    assert len(ontology.GoalStatus) == 4


# ---------------------------------------------------------------------------
# 4c. Los campos que `02-corte-ontologia.md` nombra uno a uno en `NeedSpec`
# siguen declarados, con el defecto que corresponde. `max_debt` y
# `escalates_after` ya tienen su propia prueba en la sección 5; aquí van el
# resto, y por introspección de `model_fields`, nunca por lectura del código:
# un campo que sigue existiendo pero cuyo defecto cambió en silencio es el
# mismo problema con otra cara.
# ---------------------------------------------------------------------------
def test_need_spec_decay_per_tick_sigue_siendo_obligatorio() -> None:
    """`decay_per_tick` no tiene defecto: calibrarlo es trabajo del pack y no
    hay un valor razonable que suponer en su ausencia.
    """
    # Arrange / Act
    campo = ontology.NeedSpec.model_fields["decay_per_tick"]

    # Assert
    assert campo.is_required()


@pytest.mark.parametrize(
    "campo,defecto",
    [
        ("progression", False),
        ("urgent_when_low", True),
        ("base_weight", 1.0),
        ("strain_weight", 0.0),
        ("initial", 1.0),
        ("urgency_exponent", 2.0),
        ("cap", 1.0),
    ],
)
def test_need_spec_campo_del_corte_conserva_su_valor_por_defecto(
    campo: str, defecto: object
) -> None:
    """El campo sigue declarado en `NeedSpec` y su valor por defecto no
    cambió en silencio.
    """
    # Arrange: el campo y su defecto esperado llegan parametrizados.

    # Act / Assert
    assert campo in ontology.NeedSpec.model_fields
    assert ontology.NeedSpec.model_fields[campo].default == defecto


def test_health_spec_may_go_unused_existe_con_defecto_frozenset_vacio() -> None:
    """`HealthSpec.may_go_unused` — una de las tres piezas que borró la
    mutación B de `30-validation.md` — sigue declarado, con un `frozenset`
    vacío como defecto: ningún pack está obligado a marcar redes de
    seguridad.
    """
    # Arrange / Act
    salud = ontology.HealthSpec()

    # Assert
    assert "may_go_unused" in ontology.HealthSpec.model_fields
    assert salud.may_go_unused == frozenset()


def test_health_spec_may_go_unused_serializa_en_orden_alfabetico() -> None:
    """`model_dump_json` vuelca `may_go_unused` siempre en orden alfabético,
    aunque el `frozenset` se construya con los elementos en otro orden: es
    lo que ancla el arreglo del criterio G6 (`field_serializer` de
    `_may_go_unused_sorted`). Sin esto, el orden de volcado dependería de
    `PYTHONHASHSEED` y el mismo `HealthSpec` produciría un JSON distinto en
    cada proceso.

    Arreglo de C2 (`30-validation.md`): con solo cuatro elementos, un barrido
    de las 200 semillas de hash 0-199 encontró que en 4 de ellas -136, 169,
    179, 181- el `frozenset` de entrada itera por casualidad en el mismo
    orden alfabético que se está comprobando, así que perder el
    `field_serializer` de verdad pasaría inadvertido en un 2 % de las
    ejecuciones: un falso verde. Se cubre aquí con DOS conjuntos, de doce y
    de ocho elementos y con contenido distinto entre sí -para no compartir
    una tabla de hash con casualidades correlacionadas-, y un barrido de 400
    semillas de `PYTHONHASHSEED` (0 a 399) no encontró ni una sola en la que
    cualquiera de los dos iterase en orden alfabético por casualidad: con
    doce elementos hay 12! aprox 4.79e8 reordenaciones posibles frente a una
    sola que cuenta como acierto casual, así que la probabilidad de falso
    verde queda despreciable, y exigir que AMBOS coincidan a la vez la deja
    todavía más pequeña.
    """
    # Arrange: construidos a propósito en un orden que no es el alfabético.
    acciones_a = [
        "zeta",
        "eme",
        "alfa",
        "beta",
        "gama",
        "delta",
        "kappa",
        "iota",
        "theta",
        "sigma",
        "omega",
        "rho",
    ]
    acciones_b = ["wolf", "tango", "sierra", "romeo", "quebec", "papa", "oscar", "november"]
    salud_a = ontology.HealthSpec(
        may_go_unused=frozenset(ontology.ActionId(a) for a in acciones_a)
    )
    salud_b = ontology.HealthSpec(
        may_go_unused=frozenset(ontology.ActionId(a) for a in acciones_b)
    )

    # Act
    volcado_a = json.loads(salud_a.model_dump_json())
    volcado_b = json.loads(salud_b.model_dump_json())

    # Assert
    assert volcado_a["may_go_unused"] == sorted(acciones_a)
    assert volcado_b["may_go_unused"] == sorted(acciones_b)


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
    """Cada una de las once piezas de la tabla "Se copia, adaptando" de
    `02-corte-ontologia.md` existe en el módulo: las nueve `Spec` admiten al
    menos una construcción válida con los campos obligatorios, y
    `TemperamentKnob`/`GoalStatus` —los dos enums de la tabla, no `Spec`— se
    construyen a partir de uno de sus miembros.

    No repite los validadores ya cubiertos arriba: solo confirma que la
    superficie pública completa está presente. `GoalStatus` es justamente lo
    que la mutación B de `30-validation.md` borró sin que este test se
    enterase antes de incluirlo aquí.
    """
    # Arrange / Act
    necesidad = _need()
    rasgo = ontology.TraitSpec(id=ontology.TraitId("valiente"), label="Valiente")
    gusto = ontology.TasteSpec(target=ontology.ActionId("accion"), multiplier=1.5)
    meta = _goal()
    agente = ontology.AgentSpec(id=ontology.AgentId("aida"), label="AIDA")
    franja = ontology.PhaseSpec(id=ontology.PhaseId("manana"), label="Mañana", starts_at=0)
    situacion = _situation()
    salud = ontology.HealthSpec()
    calibracion = ontology.TuningSpec()
    perilla = ontology.TemperamentKnob("temperature")
    estado_de_meta = ontology.GoalStatus("active")

    # Assert
    assert necesidad.id == ontology.NeedId("necesidad")
    assert rasgo.id == ontology.TraitId("valiente")
    assert gusto.multiplier == 1.5
    assert meta.id == ontology.GoalId("meta")
    assert agente.id == ontology.AgentId("aida")
    assert franja.id == ontology.PhaseId("manana")
    assert situacion.id == ontology.SituationId("situacion")
    assert salud.days == 7
    assert calibracion.candidate_pool == 3
    assert perilla is ontology.TemperamentKnob.TEMPERATURE
    assert estado_de_meta is ontology.GoalStatus.ACTIVE
