"""Pruebas del estado mutable de una ejecución (`aida.engine.state`).

Cubren la unidad U1c. Cuatro frentes:

- Los DOS mecanismos de deuda (`04-corte-estado.md`, «la razón de que U1c
  exista»): `debt` —cuán hundido— y `floor_streak` —cuánto llevas—, que son
  independientes y se comportan distinto a propósito.
- El resto de `AgentState`: `decay`, `apply`, `endure`, `gain`, `restore`,
  `deficit`, `saturate`, `recover_novelty`, `pursuing`.
- El `Ledger`, con su distinción entre lo activo (llega por actuar) y lo
  pasivo (llega por estar ahí), y `PerformingAction` con su validación.
- `ExecutionState`, incluida la comprobación por introspección de que no
  expone nada que el corte descartó (censo, facciones, propiedad, lugares,
  ocupantes).

Lo que este archivo NO prueba, a propósito, porque `04-corte-estado.md` lo
deja fuera de U1c: `NEED_CRITICAL` (depende de U2/U3), transiciones de metas
o de situaciones (U3/U4), `recompose` y `drift_trait` (movidos a U2).
"""

from __future__ import annotations

import dataclasses

import pytest

import aida.engine.state as state
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import (
    ActionId,
    AgentId,
    GoalId,
    GoalStatus,
    NeedId,
    NeedSpec,
    SituationId,
    SituationSpec,
)
from aida.engine.rng import RngSource
from aida.engine.state import AgentState, ExecutionState, Ledger, PerformingAction

_ALICIA = AgentId("alicia")


def _need_spec(id_: str = "necesidad", **overrides: object) -> NeedSpec:
    """Construye un `NeedSpec` mínimo válido, con overrides puntuales."""
    base: dict[str, object] = {
        "id": NeedId(id_),
        "label": "Necesidad de prueba",
        "decay_per_tick": 0.1,
    }
    base.update(overrides)
    return NeedSpec(**base)


def _agent(
    needs: dict[NeedId, float],
    *,
    debt: dict[NeedId, float] | None = None,
    goals: dict[GoalId, GoalStatus] | None = None,
    saturation: dict[ActionId, float] | None = None,
    ledger: Ledger | None = None,
) -> AgentState:
    """Construye un `AgentState` mínimo, con overrides puntuales."""
    return AgentState(
        agent_id=_ALICIA,
        needs=dict(needs),
        debt=dict(debt) if debt is not None else {},
        goals=dict(goals) if goals is not None else {},
        saturation=dict(saturation) if saturation is not None else {},
        ledger=ledger,
    )


# ---------------------------------------------------------------------------
# Bloque A.1 — Reponer paga la deuda antes de subir el valor.
# ---------------------------------------------------------------------------
def test_restore_con_deuda_pendiente_paga_la_deuda_y_no_sube_el_eje() -> None:
    """Una reposición menor que la deuda pendiente se va entera en saldarla:
    el eje no sube nada, aunque la cantidad entregada sea positiva.
    """
    # Arrange
    necesidad = _need_spec(max_debt=1.0)
    agente = _agent({necesidad.id: 0.0}, debt={necesidad.id: 0.6})

    # Act
    entregado = agente.restore(necesidad.id, 0.4, necesidad)

    # Assert
    assert entregado == 0.0
    assert agente.needs[necesidad.id] == 0.0
    assert agente.debt[necesidad.id] == pytest.approx(0.2)


def test_restore_que_termina_de_saldar_la_deuda_solo_entonces_sube_el_eje() -> None:
    """Segundo paso del mismo mecanismo: con solo 0.2 de deuda pendiente
    —lo que quedó tras el test anterior—, una reposición de 0.5 la termina
    de saldar y reparte el resto (0.3) al eje.
    """
    # Arrange
    necesidad = _need_spec(max_debt=1.0)
    agente = _agent({necesidad.id: 0.0}, debt={necesidad.id: 0.2})

    # Act
    entregado = agente.restore(necesidad.id, 0.5, necesidad)

    # Assert
    assert entregado == pytest.approx(0.3)
    assert agente.needs[necesidad.id] == pytest.approx(0.3)
    assert necesidad.id not in agente.debt


def test_restore_sin_deuda_se_recorta_contra_el_cap() -> None:
    """Sin deuda pendiente, `restore` sube el eje pero nunca por encima del
    `cap` declarado: lo que se pierde contra el techo no se devuelve —el
    retorno refleja lo que de verdad se notó, no la cantidad bruta.
    """
    # Arrange
    necesidad = _need_spec(cap=1.0)
    agente = _agent({necesidad.id: 0.9})

    # Act
    entregado = agente.restore(necesidad.id, 0.5, necesidad)

    # Assert
    assert entregado == pytest.approx(0.1)
    assert agente.needs[necesidad.id] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bloque A.2 — La deuda no se filtra a la urgencia ni al desgaste.
# ---------------------------------------------------------------------------
def test_urgency_no_se_entera_de_la_deuda() -> None:
    """Dos agentes con el MISMO valor de necesidad y distinta deuda dan la
    MISMA urgencia: `debt` no es un argumento de la fórmula. Si se filtrara,
    habría dos formas de expresar "esto va mal" y ninguna se podría calibrar
    por separado.
    """
    # Arrange
    necesidad = _need_spec(cap=1.0, urgency_exponent=2.0)
    con_poca_deuda = _agent({necesidad.id: 0.4}, debt={necesidad.id: 0.05})
    con_mucha_deuda = _agent({necesidad.id: 0.4}, debt={necesidad.id: 0.9})

    # Act
    urgencia_poca = con_poca_deuda.urgency(necesidad)
    urgencia_mucha = con_mucha_deuda.urgency(necesidad)

    # Assert
    assert urgencia_poca == urgencia_mucha


def test_strain_no_se_entera_de_la_deuda() -> None:
    """La misma propiedad, un nivel más arriba: `refresh_state` recalcula
    `strain` a partir de la urgencia de cada eje, así que hereda su ceguera
    a la deuda. Mismo valor de necesidad, distinta deuda, mismo `strain`.
    """
    # Arrange
    necesidad = _need_spec(cap=1.0, urgency_exponent=2.0, strain_weight=1.0)
    specs = {necesidad.id: necesidad}
    sin_deuda = _agent({necesidad.id: 0.4})
    con_deuda = _agent({necesidad.id: 0.4}, debt={necesidad.id: 0.9})

    # Act
    sin_deuda.refresh_state(specs)
    con_deuda.refresh_state(specs)

    # Assert
    assert sin_deuda.strain == con_deuda.strain


# ---------------------------------------------------------------------------
# Bloque A.3 y A.4 — La deuda respeta `max_debt`; `max_debt == 0` la
# desactiva por completo.
# ---------------------------------------------------------------------------
def test_spend_topa_la_deuda_en_max_debt() -> None:
    """Un recorte que hundiría el eje muy por debajo de cero deja la deuda
    TOPADA en `max_debt`, no creciendo sin límite.
    """
    # Arrange: decae 1.0 sobre un eje que solo tiene 0.1, así que el recorte
    # "querría" comerse 0.9 de deuda — muy por encima del tope de 0.3.
    necesidad = _need_spec(decay_per_tick=1.0, max_debt=0.3, cap=1.0)
    agente = _agent({necesidad.id: 0.1})

    # Act
    agente.decay({necesidad.id: necesidad})

    # Assert
    assert agente.needs[necesidad.id] == 0.0
    assert agente.debt[necesidad.id] == pytest.approx(0.3)


def test_max_debt_cero_desactiva_la_deuda_y_no_deja_entrada() -> None:
    """Con `max_debt == 0.0` —el defecto—, un recorte por debajo del suelo
    se pierde sin más: el eje se recorta contra cero y NO aparece ninguna
    entrada en `debt`, ni siquiera a 0.0.
    """
    # Arrange
    necesidad = _need_spec(decay_per_tick=1.0, max_debt=0.0, cap=1.0)
    agente = _agent({necesidad.id: 0.1})

    # Act
    agente.decay({necesidad.id: necesidad})

    # Assert
    assert agente.needs[necesidad.id] == 0.0
    assert necesidad.id not in agente.debt


# ---------------------------------------------------------------------------
# Bloque A.5 — El contador de hundimiento se reinicia de golpe, a
# diferencia de la deuda, que se paga a plazos.
# ---------------------------------------------------------------------------
def test_floor_streak_sube_de_uno_en_uno_mientras_el_eje_esta_al_suelo() -> None:
    """El contador cuenta ticks SEGUIDOS con el eje en 0.0, de uno en uno."""
    # Arrange
    necesidad = _need_spec(escalates_after=5)
    agente = _agent({necesidad.id: 0.0})
    log = EventLog()
    specs = {necesidad.id: necesidad}

    # Act
    for tick in range(1, 4):
        agente.advance_floor_streak(specs, event_log=log, owner=_ALICIA, tick=tick)

    # Assert
    assert agente.floor_streak[necesidad.id] == 3


def test_floor_streak_se_reinicia_de_golpe_al_levantar_el_eje() -> None:
    """En cuanto el eje deja de estar al suelo, el contador DESAPARECE del
    diccionario entero en la siguiente llamada — no baja a 2, ni a 1: se
    borra de golpe.
    """
    # Arrange
    necesidad = _need_spec(escalates_after=5)
    agente = _agent({necesidad.id: 0.0})
    log = EventLog()
    specs = {necesidad.id: necesidad}
    for tick in range(1, 4):
        agente.advance_floor_streak(specs, event_log=log, owner=_ALICIA, tick=tick)
    assert agente.floor_streak[necesidad.id] == 3, "arranque del test no es el esperado"

    # Act: el eje se levanta del suelo.
    agente.needs[necesidad.id] = 0.5
    agente.advance_floor_streak(specs, event_log=log, owner=_ALICIA, tick=4)

    # Assert
    assert necesidad.id not in agente.floor_streak


def test_floor_streak_se_reinicia_de_golpe_a_diferencia_de_la_deuda_que_se_paga_a_plazos() -> None:
    """Las dos primitivas de U1c, lado a lado, sobre el MISMO eje: levantar
    el valor de la necesidad borra el contador de hundimiento por completo,
    pero no toca la deuda ni un céntimo — la deuda solo la paga `restore`,
    y a plazos (ver el bloque A.1, arriba). Son independientes: se puede
    tocar fondo un instante o llevar mucho tiempo raspando el suelo, y son
    preguntas distintas.
    """
    # Arrange
    necesidad = _need_spec(escalates_after=2, max_debt=1.0)
    agente = _agent({necesidad.id: 0.0}, debt={necesidad.id: 0.6})
    log = EventLog()
    specs = {necesidad.id: necesidad}
    agente.advance_floor_streak(specs, event_log=log, owner=_ALICIA, tick=1)
    agente.advance_floor_streak(specs, event_log=log, owner=_ALICIA, tick=2)
    assert agente.floor_streak[necesidad.id] == 2

    # Act: levantar el eje de golpe, SIN pasar por `restore`.
    agente.needs[necesidad.id] = 0.5
    agente.advance_floor_streak(specs, event_log=log, owner=_ALICIA, tick=3)

    # Assert: el contador desaparece; la deuda, en cambio, sigue intacta.
    assert necesidad.id not in agente.floor_streak
    assert agente.debt[necesidad.id] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Bloque A.6 — `NEED_ESCALATED` se emite en la transición, no en el estado
# sostenido.
# ---------------------------------------------------------------------------
def test_need_escalated_se_emite_exactamente_en_el_tick_de_transicion() -> None:
    """El evento se emite en el tick EXACTO en que el contador alcanza
    `escalates_after` — ni antes, ni en los ticks siguientes en que el eje
    siga hundido: es la transición la que se anuncia, no el estado
    sostenido.
    """
    # Arrange
    necesidad = _need_spec(escalates_after=3)
    agente = _agent({necesidad.id: 0.0})
    log = EventLog()
    specs = {necesidad.id: necesidad}

    # Act: dos ticks por debajo del umbral, el que lo cruza, y dos más ya
    # escalado.
    for tick in range(1, 6):
        agente.advance_floor_streak(specs, event_log=log, owner=_ALICIA, tick=tick)

    # Assert: un único NEED_ESCALATED, exactamente en el tick 3.
    escalados = log.of_kind(EventKind.NEED_ESCALATED)
    assert len(escalados) == 1
    assert escalados[0].tick == 3
    assert escalados[0].data == {"need": necesidad.id}
    # U1c no emite NEED_CRITICAL: depende de la composición de temperamento
    # (U2) y de si la acción en curso ya atiende el eje (U3).
    assert log.of_kind(EventKind.NEED_CRITICAL) == []


# ---------------------------------------------------------------------------
# Bloque A.7 — `escalates_after == 0` desactiva el mecanismo por completo.
# ---------------------------------------------------------------------------
def test_escalates_after_cero_desactiva_el_contador_y_no_emite() -> None:
    """Con `escalates_after == 0` el eje no cuenta ni deja entrada en
    `floor_streak`, por muchos ticks que pase en el suelo, y nunca emite.
    """
    # Arrange
    necesidad = _need_spec(escalates_after=0)
    agente = _agent({necesidad.id: 0.0})
    log = EventLog()
    specs = {necesidad.id: necesidad}

    # Act
    for tick in range(1, 4):
        agente.advance_floor_streak(specs, event_log=log, owner=_ALICIA, tick=tick)

    # Assert
    assert necesidad.id not in agente.floor_streak
    assert len(log) == 0


# ---------------------------------------------------------------------------
# Bloque B — `decay`.
# ---------------------------------------------------------------------------
def test_decay_baja_el_eje_segun_su_propia_tasa() -> None:
    """`decay` resta `decay_per_tick` de cada eje que aparece en el mapa de
    especificaciones."""
    # Arrange
    necesidad = _need_spec(decay_per_tick=0.1)
    agente = _agent({necesidad.id: 0.5})

    # Act
    agente.decay({necesidad.id: necesidad})

    # Assert
    assert agente.needs[necesidad.id] == pytest.approx(0.4)


def test_decay_anota_en_el_libro_de_demanda() -> None:
    """Con el `Ledger` activo, `decay` anota lo perdido en `demand` — nunca
    en `active` ni en `passive`: no llega por actuar ni por estar bajo una
    situación, es simplemente el paso del tiempo."""
    # Arrange
    necesidad = _need_spec(decay_per_tick=0.1)
    agente = _agent({necesidad.id: 0.5}, ledger=Ledger())

    # Act
    agente.decay({necesidad.id: necesidad})

    # Assert
    assert agente.ledger is not None
    assert agente.ledger.demand[necesidad.id] == pytest.approx(0.1)
    assert necesidad.id not in agente.ledger.active
    assert necesidad.id not in agente.ledger.passive


# ---------------------------------------------------------------------------
# Bloque B — `apply` y `endure`: el `Ledger` distingue activo de pasivo.
# ---------------------------------------------------------------------------
def test_apply_con_delta_positivo_sube_el_eje_y_anota_como_activo() -> None:
    """Un delta positivo de `apply` —lo que un agente acaba de completar—
    entra por `gain`: sube el eje y, con el libro activo, se anota en
    `active`."""
    # Arrange
    necesidad = _need_spec(cap=1.0)
    agente = _agent({necesidad.id: 0.3}, ledger=Ledger())

    # Act
    agente.apply({necesidad.id: 0.2}, {necesidad.id: necesidad})

    # Assert
    assert agente.needs[necesidad.id] == pytest.approx(0.5)
    assert agente.ledger is not None
    assert agente.ledger.active[necesidad.id] == pytest.approx(0.2)
    assert necesidad.id not in agente.ledger.passive


def test_apply_con_delta_negativo_baja_el_eje_y_anota_como_demanda() -> None:
    """Un delta negativo de `apply` gasta: baja el eje y se anota en
    `demand`, como cualquier coste."""
    # Arrange
    necesidad = _need_spec(cap=1.0)
    agente = _agent({necesidad.id: 0.5}, ledger=Ledger())

    # Act
    agente.apply({necesidad.id: -0.2}, {necesidad.id: necesidad})

    # Assert
    assert agente.needs[necesidad.id] == pytest.approx(0.3)
    assert agente.ledger is not None
    assert agente.ledger.demand[necesidad.id] == pytest.approx(0.2)


def test_endure_con_delta_positivo_anota_como_pasivo_no_activo() -> None:
    """Lo que llega por `endure` —estar bajo el alcance de una situación,
    no por decidir nada— se anota en `passive`. Contrástese con
    `test_apply_con_delta_positivo_sube_el_eje_y_anota_como_activo`: la
    MISMA cantidad, sobre el MISMO eje, pero `apply` y `endure` la archivan
    en libros distintos — exactamente la distinción que el `Ledger` existe
    para no perder.
    """
    # Arrange
    necesidad = _need_spec(cap=1.0)
    agente = _agent({necesidad.id: 0.3}, ledger=Ledger())

    # Act
    agente.endure({necesidad.id: 0.2}, {necesidad.id: necesidad})

    # Assert
    assert agente.needs[necesidad.id] == pytest.approx(0.5)
    assert agente.ledger is not None
    assert agente.ledger.passive[necesidad.id] == pytest.approx(0.2)
    assert necesidad.id not in agente.ledger.active


def test_endure_con_delta_negativo_anota_como_demanda_igual_que_apply() -> None:
    """El lado negativo de `endure` no distingue de `apply`: ambos son
    coste y se anotan en `demand`, no en `passive`."""
    # Arrange
    necesidad = _need_spec(cap=1.0)
    agente = _agent({necesidad.id: 0.5}, ledger=Ledger())

    # Act
    agente.endure({necesidad.id: -0.2}, {necesidad.id: necesidad})

    # Assert
    assert agente.needs[necesidad.id] == pytest.approx(0.3)
    assert agente.ledger is not None
    assert agente.ledger.demand[necesidad.id] == pytest.approx(0.2)
    assert necesidad.id not in agente.ledger.passive


# ---------------------------------------------------------------------------
# Bloque B — `gain` y `deficit`.
# ---------------------------------------------------------------------------
def test_gain_es_restore_anotado_como_activo() -> None:
    """`gain` es `restore` con una única diferencia: anota lo entregado en
    el libro `active` si el `Ledger` está activo."""
    # Arrange
    necesidad = _need_spec(cap=1.0)
    agente = _agent({necesidad.id: 0.3}, ledger=Ledger())

    # Act
    entregado = agente.gain(necesidad.id, 0.2, necesidad)

    # Assert
    assert entregado == pytest.approx(0.2)
    assert agente.needs[necesidad.id] == pytest.approx(0.5)
    assert agente.ledger is not None
    assert agente.ledger.active[necesidad.id] == pytest.approx(0.2)


def test_deficit_cuenta_lo_debido_ademas_de_lo_que_falta() -> None:
    """`deficit` es `1.0 - needs + debt`: cuenta también lo que se debe, no
    solo lo que el eje muestra ahora mismo."""
    # Arrange
    necesidad = _need_spec()
    agente = _agent({necesidad.id: 0.3}, debt={necesidad.id: 0.2})

    # Act
    deficit = agente.deficit(necesidad.id)

    # Assert
    assert deficit == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Bloque B — hastío: `saturate` y `recover_novelty`.
# ---------------------------------------------------------------------------
def test_saturate_con_ganancia_no_positiva_no_deja_rastro() -> None:
    """`gain <= 0` no anota nada: el hastío no se registra sin una
    finalización de verdad."""
    # Arrange
    accion = ActionId("accion")
    agente = _agent({})

    # Act
    agente.saturate(accion, 0.0)

    # Assert
    assert accion not in agente.saturation


def test_saturate_acumula_y_se_recorta_en_uno() -> None:
    """Completar la misma acción varias veces acumula hastío, recortado a
    1.0: no se puede estar "más que saturado"."""
    # Arrange
    accion = ActionId("accion")
    agente = _agent({})

    # Act
    agente.saturate(accion, 0.7)
    agente.saturate(accion, 0.7)

    # Assert
    assert agente.saturation[accion] == 1.0


def test_recover_novelty_con_tasa_no_positiva_no_hace_nada() -> None:
    """`rate <= 0` desactiva la recuperación: el hastío se queda tal cual."""
    # Arrange
    accion = ActionId("accion")
    agente = _agent({}, saturation={accion: 0.5})

    # Act
    agente.recover_novelty(0.0)

    # Assert
    assert agente.saturation[accion] == 0.5


def test_recover_novelty_olvida_lo_que_llega_a_cero() -> None:
    """El hastío que baja de cero no se queda en el diccionario en
    negativo: la entrada desaparece por completo, no ocupa nada."""
    # Arrange
    accion = ActionId("accion")
    agente = _agent({}, saturation={accion: 0.05})

    # Act
    agente.recover_novelty(0.1)

    # Assert
    assert accion not in agente.saturation


# ---------------------------------------------------------------------------
# Bloque B — `pursuing`.
# ---------------------------------------------------------------------------
def test_pursuing_devuelve_solo_las_metas_activas_ordenadas() -> None:
    """`pursuing` filtra las metas `ACTIVE` y descarta las demás —logradas,
    perdidas o bloqueadas—, en orden estable de identificador."""
    # Arrange
    agente = _agent(
        {},
        goals={
            GoalId("zeta"): GoalStatus.ACTIVE,
            GoalId("alfa"): GoalStatus.ACTIVE,
            GoalId("logrado"): GoalStatus.ACHIEVED,
            GoalId("perdido"): GoalStatus.FAILED,
            GoalId("bloqueado"): GoalStatus.LOCKED,
        },
    )

    # Act
    activas = agente.pursuing()

    # Assert
    assert activas == (GoalId("alfa"), GoalId("zeta"))


# ---------------------------------------------------------------------------
# Bloque B — `Ledger.record`.
# ---------------------------------------------------------------------------
def test_ledger_record_acumula_sobre_el_mismo_eje() -> None:
    """Dos anotaciones sobre el mismo eje en el mismo libro se SUMAN, no se
    sustituyen: es lo que permite acumular varias fuentes en un solo tick."""
    # Arrange
    libro: dict[NeedId, float] = {}
    necesidad_id = NeedId("necesidad")
    ledger = Ledger()

    # Act
    ledger.record(libro, necesidad_id, 0.3)
    ledger.record(libro, necesidad_id, 0.4)

    # Assert
    assert libro[necesidad_id] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Bloque B — `PerformingAction` y su validación en `__post_init__`.
# ---------------------------------------------------------------------------
def test_performing_action_valida_construye_sin_error() -> None:
    """Con `ticks_remaining` y `total_ticks` no negativos, se construye sin
    problema."""
    # Arrange / Act
    accion = PerformingAction(action=ActionId("accion"), ticks_remaining=2, total_ticks=5)

    # Assert
    assert accion.ticks_remaining == 2
    assert accion.total_ticks == 5


def test_performing_action_con_ticks_remaining_negativo_falla() -> None:
    """`ticks_remaining` negativo no tiene interpretación: no se puede
    deber un tiempo negativo."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="ticks_remaining"):
        PerformingAction(action=ActionId("accion"), ticks_remaining=-1)


def test_performing_action_con_total_ticks_negativo_falla() -> None:
    """`total_ticks` negativo corre la misma suerte."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="total_ticks"):
        PerformingAction(action=ActionId("accion"), ticks_remaining=0, total_ticks=-1)


# ---------------------------------------------------------------------------
# Bloque B — `ExecutionState`.
# ---------------------------------------------------------------------------
def test_execution_state_seed_expone_la_semilla_del_rng() -> None:
    """`seed` es la del `RngSource` que la ejecución lleva dentro, no un
    campo separado que pudiera desincronizarse de él."""
    # Arrange
    ejecucion = ExecutionState(agent=_agent({}), rng=RngSource(1312))

    # Assert
    assert ejecucion.seed == 1312


def test_execution_state_active_situations_incluye_periodica_y_eventual_encendida() -> None:
    """Una situación PERIÓDICA activa según el tick, y una EVENTUAL que el
    diccionario de la ejecución marca como encendida, aparecen las dos —una
    tercera, ni periódica en este tick ni encendida, no aparece—, y el
    resultado llega ORDENADO."""
    # Arrange
    periodica = SituationSpec(id=SituationId("periodica"), label="Periódica", every=4, lasts=2)
    eventual_encendida = SituationSpec(id=SituationId("encendida"), label="Encendida")
    eventual_apagada = SituationSpec(id=SituationId("apagada"), label="Apagada")
    ejecucion = ExecutionState(
        agent=_agent({}),
        tick=5,
        situations={SituationId("encendida"): None},
    )

    # Act
    activas = ejecucion.active_situations([periodica, eventual_encendida, eventual_apagada])

    # Assert
    assert activas == (SituationId("encendida"), SituationId("periodica"))


# ---------------------------------------------------------------------------
# Lo que no debe existir en `state.py`: protege el corte de
# `04-corte-estado.md`, verificado por introspección.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nombre",
    ["Traveling", "Bond", "Faccion", "Propiedad", "Condicion", "WorldState"],
)
def test_pieza_descartada_no_existe_en_el_modulo(nombre: str) -> None:
    """Ninguna pieza descartada por el corte reaparece: ni el viaje, ni el
    aprecio entre pares, ni facciones, propiedad o condiciones corporales, ni
    un `WorldState` sin renombrar."""
    # Arrange: el nombre bajo prueba llega parametrizado.

    # Act / Assert
    assert not hasattr(state, nombre)


@pytest.mark.parametrize(
    "nombre",
    [
        "regard_at",
        "regard_for",
        "acercar",
        "meet",
        "forget_bonds",
        "remember_company",
        "effective_company",
        "afligir",
        "aliviar",
        "expirar_condiciones",
        "elegir_destinatario",
        "afinidad",
        "disposicion",
        "observe",
        "perceived_crowd",
        "place_of",
        "occupants",
        "others_at",
        "situations_at",
        "edad",
        "ritmo_de_rango",
        "subir_de_rango",
        "rango",
        "animos",
        "recompose",
        "drift_trait",
    ],
)
def test_metodo_descartado_no_existe_en_agent_state(nombre: str) -> None:
    """Ninguno de los métodos descartados cuelga de `AgentState` — incluidos
    `recompose` y `drift_trait`, que la corrección de `04-corte-estado.md`
    movió a U2 porque dependen de `disposition.py`, todavía inexistente."""
    # Arrange: el nombre bajo prueba llega parametrizado.

    # Act / Assert
    assert not hasattr(AgentState, nombre)


def test_agent_state_no_declara_campos_que_dependen_de_disposition() -> None:
    """`need_weights`, `gains`, `costs`, `caps`, `temperament`,
    `phase_shift`, `gain_beliefs` y `cost_beliefs` no tienen quién los llene
    todavía: aprobado explícitamente en la corrección de
    `04-corte-estado.md`. Un campo sin escritor ni consumidor es código
    muerto."""
    # Arrange
    prohibidos = {
        "need_weights",
        "gains",
        "costs",
        "caps",
        "temperament",
        "phase_shift",
        "gain_beliefs",
        "cost_beliefs",
    }

    # Act
    campos = {campo.name for campo in dataclasses.fields(AgentState)}

    # Assert
    assert campos.isdisjoint(prohibidos)


def test_execution_state_no_expone_censo_ni_geografia_ni_faccion() -> None:
    """Recortado de lo que en la referencia era el estado de un mundo
    entero: no queda censo de agentes, ni lugares, ni facciones, ni
    propiedad, ni ocupantes — solo hay un agente."""
    # Arrange
    prohibidos = {
        "agents",
        "census",
        "places",
        "factions",
        "properties",
        "occupants",
        "bonds",
    }

    # Act
    campos = {campo.name for campo in dataclasses.fields(ExecutionState)}

    # Assert
    assert campos.isdisjoint(prohibidos)
