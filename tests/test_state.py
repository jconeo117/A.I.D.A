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

import ast
import dataclasses
import inspect
import textwrap

import pytest

import aida.engine.disposition as disposition
import aida.engine.ontology as ontology
import aida.engine.state as state
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import (
    ActionId,
    ActionSpec,
    AgentId,
    AgentSpec,
    GoalId,
    GoalSpec,
    GoalStatus,
    NeedId,
    NeedSpec,
    Reversibility,
    SituationId,
    SituationSpec,
    TemperamentKnob,
    TraitId,
    TraitSpec,
    TuningSpec,
)
from aida.engine.outcome import learn
from aida.engine.pack import ContentPack
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


def test_restore_respeta_techo_efectivo_del_agente_y_no_supera_caps() -> None:
    """`restore` recorta contra el techo efectivo del agente (`cap_of`), no contra `need.cap`.

    Si el agente tiene un techo reducido por personalidad (`caps={'n1': 0.5}`),
    reponer cuando está en 0.4 una cantidad de 0.5 debe llevarlo como máximo a 0.5
    (entregando 0.1), no a 0.9 (entregando 0.5) (I-021, mutante S2b).
    """
    necesidad = _need_spec(cap=1.0)
    agente = _agent({necesidad.id: 0.4})
    agente.caps[necesidad.id] = 0.5

    entregado = agente.restore(necesidad.id, 0.5, necesidad)

    assert agente.needs[necesidad.id] == pytest.approx(0.5)
    assert entregado == pytest.approx(0.1)


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


def test_refresh_state_coincide_numericando_con_compute_strain() -> None:
    """AgentState.refresh_state y disposition.compute_strain producen el mismo resultado
    sin divergencia (I-014).
    """
    necesidad = _need_spec(cap=1.0, urgency_exponent=2.0, strain_weight=1.0)
    specs = {necesidad.id: necesidad}
    agente = _agent({necesidad.id: 0.5})
    agente.caps = {necesidad.id: 0.5}

    agente.refresh_state(specs)
    strain_directo = disposition.compute_strain(specs, agente.needs, agente.caps)

    # Con cap=0.5 y need=0.5, el agente está lleno en su cota personal => strain es 0.0
    assert agente.strain == pytest.approx(0.0)
    assert agente.strain == pytest.approx(strain_directo)


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


def test_spend_recorta_contra_techo_efectivo_del_agente() -> None:
    """`_spend` recorta contra el techo efectivo del agente (`cap_of`), no contra `need.cap`.

    Si un agente tiene un valor de necesidad en 0.8 y su techo efectivo se reduce a 0.5
    (por ejemplo, tras recomposición de rasgos), un gasto de 0.1 que deja un valor
    bruto de 0.7 debe recortarse contra el techo del agente (0.5), no contra el
    nominal del pack (1.0) (mutante S2a).
    """
    necesidad = _need_spec(cap=1.0)
    agente = _agent({necesidad.id: 0.8})
    agente.caps[necesidad.id] = 0.5

    agente._spend(0.1, necesidad)

    # 0.8 - 0.1 = 0.7; con cap_of(necesidad)=0.5, debe recortarse a 0.5
    # Si mutara a cap = need.cap (1.0), el valor quedaría en 0.7.
    assert agente.needs[necesidad.id] == pytest.approx(0.5)


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
    ],
)
def test_metodo_descartado_no_existe_en_agent_state(nombre: str) -> None:
    """Ninguno de los métodos descartados cuelga de `AgentState`.
    Nota: `recompose` y `drift_trait` pertenecían a este listado antes de U3,
    pero fueron incorporados como métodos oficiales al llegar `disposition.py`.
    """
    # Arrange: el nombre bajo prueba llega parametrizado.

    # Act / Assert
    assert not hasattr(AgentState, nombre)


def test_agent_state_declara_los_siete_campos_derivados_de_u3() -> None:
    """En U3, AgentState incorpora los siete campos derivados:
    need_weights, gains, costs, caps, temperament, gain_beliefs, cost_beliefs.
    """
    requeridos = {
        "need_weights",
        "gains",
        "costs",
        "caps",
        "temperament",
        "gain_beliefs",
        "cost_beliefs",
    }
    campos = {campo.name for campo in dataclasses.fields(AgentState)}
    assert requeridos.issubset(campos)


def test_agent_state_sigue_sin_declarar_phase_shift_diferido_a_reloj() -> None:
    """phase_shift se difiere al reloj (U5) y no debe existir en AgentState."""
    campos = {campo.name for campo in dataclasses.fields(AgentState)}
    assert "phase_shift" not in campos


# ---------------------------------------------------------------------------
# Bloque E — U3: recomposición y deriva de rasgos
# ---------------------------------------------------------------------------


def _pack_para_recomposicion(**overrides: object) -> ContentPack:
    n1 = NeedSpec(
        id=NeedId("n1"),
        label="Eje 1",
        decay_per_tick=0.01,
        base_weight=1.0,
        cap=0.8,
        urgency_exponent=2.0,
        strain_weight=1.0,
    )
    t1 = TraitSpec(
        id=TraitId("t1"),
        label="Rasgo 1",
        need_weight_multipliers={NeedId("n1"): 2.0},
        gain_multipliers={NeedId("n1"): 1.5},
        cost_multipliers={NeedId("n1"): 0.5},
        cap_multipliers={NeedId("n1"): 1.25},
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 2.0},
    )
    act1 = ActionSpec(
        id=ActionId("act1"),
        label="Accion 1",
        duration_ticks=1,
        reversibility=Reversibility.REVERSIBLE,
        satisfies={NeedId("n1"): 0.5},
        effects={NeedId("n1"): 0.5},
    )
    base: dict[str, object] = {
        "id": "pack_recomp",
        "label": "Pack para recomposicion",
        "needs": (n1,),
        "actions": (act1,),
        "agents": (AgentSpec(id=_ALICIA, label="Alicia", initial_needs={"n1": 0.5}),),
        "traits": (t1,),
        "tuning": TuningSpec(loss_aversion=1.0),
    }
    base.update(overrides)
    return ContentPack(**base)


def test_recompose_recalcula_todos_los_campos_derivados() -> None:
    """recompose(pack) actualiza need_weights, gains, costs, caps y temperament
    a partir de los rasgos actuales del agente. Preserva las creencias existentes.
    """
    pack = _pack_para_recomposicion()
    agente = _agent({NeedId("n1"): 0.4})
    agente.traits = {TraitId("t1"): 1.0}
    # Creencias previas (indexadas por ActionId y NeedId)
    agente.gain_beliefs = {ActionId("act1"): {NeedId("n1"): 1.2}}
    agente.cost_beliefs = {ActionId("act1"): {NeedId("n1"): 0.8}}

    agente.recompose(pack)

    assert agente.need_weights[NeedId("n1")] == pytest.approx(2.0)
    assert agente.gains[NeedId("n1")] == pytest.approx(1.5)
    assert agente.costs[NeedId("n1")] == pytest.approx(0.5)
    assert agente.caps[NeedId("n1")] == pytest.approx(1.0)  # min(1.0, 0.8 * 1.25 = 1.0)
    assert agente.temperament is not None
    assert agente.temperament.loss_aversion == pytest.approx(2.0)

    # Creencias no deben reiniciarse
    assert agente.gain_beliefs[ActionId("act1")][NeedId("n1")] == pytest.approx(1.2)
    assert agente.cost_beliefs[ActionId("act1")][NeedId("n1")] == pytest.approx(0.8)


def test_recompose_pasa_metas_activas_a_need_weights() -> None:
    """recompose traslada efectivamente las metas activas a compose_need_weights (I-015)."""
    n1 = NeedSpec(
        id=NeedId("n1"),
        label="Eje 1",
        decay_per_tick=0.01,
        base_weight=2.0,
    )
    g1 = GoalSpec(
        id=GoalId("g1"),
        label="Meta activa",
        need=NeedId("n1"),
        at_least=0.8,
        weight_multiplier=3.0,
    )
    pack = _pack_para_recomposicion(needs=(n1,), goals=(g1,))
    agente = _agent({NeedId("n1"): 0.5}, goals={GoalId("g1"): GoalStatus.ACTIVE})
    agente.traits = {TraitId("t1"): 0.0}
    agente.recompose(pack)

    # Si recompose pasara () en vez de las metas activas, daría 2.0 en vez de 6.0
    assert agente.need_weights[NeedId("n1")] == pytest.approx(6.0)

    # Metas alcanzadas o fallidas no se consideran activas en pursuing()
    agente.goals[GoalId("g1")] = GoalStatus.ACHIEVED
    agente.recompose(pack)
    assert agente.need_weights[NeedId("n1")] == pytest.approx(2.0)


def test_recompose_modula_temperamento_con_strain_del_agente() -> None:
    """recompose traslada efectivamente self.strain a compose_temperament (I-018)."""
    pack = _pack_para_recomposicion(
        tuning=TuningSpec(
            loss_aversion=1.0,
            strain_multipliers={TemperamentKnob.LOSS_AVERSION: 4.0},
        ),
    )
    agente = _agent({NeedId("n1"): 0.5})
    agente.traits = {TraitId("t1"): 0.0}
    # Con strain=0.5: 1.0 * (4.0 ** 0.5) = 2.0. Con strain ignorado/0.0 daría 1.0.
    agente.strain = 0.5
    agente.recompose(pack)

    assert agente.temperament is not None
    assert agente.temperament.loss_aversion == pytest.approx(2.0)


def test_recompose_modula_temperamento_con_situaciones_activas() -> None:
    """recompose traslada efectivamente active_situations a compose_temperament (I-018)."""
    s1 = SituationSpec(
        id=SituationId("s1"),
        label="Situación crítica",
        temperament_multipliers={TemperamentKnob.LOSS_AVERSION: 3.0},
    )
    pack = _pack_para_recomposicion(situations=(s1,))
    agente = _agent({NeedId("n1"): 0.5})
    agente.traits = {TraitId("t1"): 0.0}
    agente.recompose(pack, active_situations=(SituationId("s1"),))

    assert agente.temperament is not None
    # Con situación activa: 1.0 * 3.0 = 3.0. Si fuera ignorada o pasada como (), daría 1.0.
    assert agente.temperament.loss_aversion == pytest.approx(3.0)


def test_beliefs_estan_indexadas_por_accion_y_necesidad_sin_compartirse() -> None:
    """Las creencias en AgentState se indexan por [ActionId][NeedId] y dos acciones
    no comparten creencias sobre el mismo eje al aprender (I-013, I-022).
    """
    import typing

    # Verificamos la declaración canónica del tipo: indexado por acción y necesidad
    hints = typing.get_type_hints(AgentState)
    assert hints["gain_beliefs"] == dict[ActionId, dict[NeedId, float]]
    assert hints["cost_beliefs"] == dict[ActionId, dict[NeedId, float]]

    n1 = NeedId("n1")
    act_a = ActionSpec(
        id=ActionId("act_a"),
        label="Acción A",
        duration_ticks=1,
        reversibility=Reversibility.REVERSIBLE,
        satisfies={n1: 0.5},
        costs={},
        effects={n1: 0.5},
    )
    act_b = ActionSpec(
        id=ActionId("act_b"),
        label="Acción B",
        duration_ticks=1,
        reversibility=Reversibility.REVERSIBLE,
        satisfies={n1: 0.5},
        costs={},
        effects={n1: 0.5},
    )

    agente = _agent({n1: 0.2})

    # Arrancan vacías
    assert agente.gain_beliefs == {}
    assert agente.cost_beliefs == {}

    # Ambas acciones resuelven por defecto a 1.0
    assert agente.gain_beliefs.get(act_a.id, {}).get(n1, 1.0) == 1.0
    assert agente.gain_beliefs.get(act_b.id, {}).get(n1, 1.0) == 1.0

    # Ejecutamos aprendizaje real con learn sobre act_a.
    # Prometía 0.5, pero solo rindió 0.25 (la mitad).
    # Con rate=0.4: creencia = 1.0 + 0.4 * (0.25/0.5 - 1.0) = 1.0 - 0.2 = 0.8
    gb_a = agente.gain_beliefs.setdefault(act_a.id, {})
    cb_a = agente.cost_beliefs.setdefault(act_a.id, {})
    learn(
        gain_beliefs=gb_a,
        cost_beliefs=cb_a,
        action=act_a,
        received={n1: 0.25},
        before={n1: 0.2},
        rate=0.4,
    )

    # act_a actualizó su creencia
    assert agente.gain_beliefs[act_a.id][n1] == pytest.approx(0.8)

    # act_b NO se movió ni comparte el aprendizaje sobre n1
    assert act_b.id not in agente.gain_beliefs
    assert agente.gain_beliefs.get(act_b.id, {}).get(n1, 1.0) == 1.0


def test_drift_trait_muta_rasgo_y_recompone_en_la_misma_llamada() -> None:
    """drift_trait altera la intensidad del rasgo y vuelve a componer de inmediato."""
    pack = _pack_para_recomposicion()
    agente = _agent({NeedId("n1"): 0.4})
    agente.traits = {TraitId("t1"): 0.0}
    agente.recompose(pack)

    assert agente.gains[NeedId("n1")] == pytest.approx(1.0)

    # Aplicamos deriva de +1.0
    agente.drift_trait(pack, TraitId("t1"), 1.0)

    assert agente.traits[TraitId("t1")] == pytest.approx(1.0)
    # Los derivados deben haberse recalculado sin invocar recompose por separado
    assert agente.gains[NeedId("n1")] == pytest.approx(1.5)
    assert agente.costs[NeedId("n1")] == pytest.approx(0.5)


def test_drift_trait_acota_intensidad_a_menos_uno_y_uno() -> None:
    """drift_trait no permite que la intensidad se salga del intervalo [-1.0, 1.0]."""
    pack = _pack_para_recomposicion()
    agente = _agent({NeedId("n1"): 0.4})
    agente.traits = {TraitId("t1"): 0.5}

    agente.drift_trait(pack, TraitId("t1"), 2.0)
    assert agente.traits[TraitId("t1")] == 1.0

    agente.drift_trait(pack, TraitId("t1"), -5.0)
    assert agente.traits[TraitId("t1")] == -1.0


def test_drift_trait_con_rasgo_inexistente_lanza_keyerror() -> None:
    """drift_trait sobre un rasgo que no está en el pack levanta KeyError con
    mensaje propio que nombra el rasgo.
    """
    pack = _pack_para_recomposicion()
    agente = _agent({NeedId("n1"): 0.4})

    with pytest.raises(KeyError, match=r"el rasgo 'fantasma' no existe en el pack"):
        agente.drift_trait(pack, TraitId("fantasma"), 0.1)


def test_agent_state_urgency_delega_en_disposition_urgency_usando_caps() -> None:
    """AgentState.urgency delega en disposition.urgency leyendo el techo derivado de caps."""
    necesidad = _need_spec(cap=1.0, urgency_exponent=2.0)
    agente = _agent({necesidad.id: 0.5})
    # Sin caps personalizadas, usa need.cap = 1.0 => (1 - 0.5/1.0)**2 = 0.25
    assert agente.urgency(necesidad) == pytest.approx(0.25)

    # Con cap personalizada en caps => ej. cap=0.5, valor=0.5 => 0.0 (alcanzó su tope)
    agente.caps = {necesidad.id: 0.5}
    assert agente.urgency(necesidad) == pytest.approx(0.0)


def test_state_urgency_es_identica_a_disposition_urgency() -> None:
    """Introspección: state.urgency es exactamente disposition.urgency."""
    assert state.urgency is disposition.urgency
    assert not hasattr(state, "_urgency")


def test_agent_state_urgency_invoca_la_funcion_urgency_sin_reimplementarla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AgentState.urgency invoca la función canónica de urgency y no re-implementa
    la fórmula (I-017).
    """
    llamadas: list[tuple[float, float, float]] = []

    def spy_urgency(value: float, exponent: float, cap: float = 1.0) -> float:
        llamadas.append((value, exponent, cap))
        return 777.0

    monkeypatch.setattr(state, "urgency", spy_urgency)
    necesidad = _need_spec(cap=1.0, urgency_exponent=2.0)
    agente = _agent({necesidad.id: 0.5})
    agente.caps = {necesidad.id: 0.8}

    res = agente.urgency(necesidad)
    # Si AgentState.urgency contuviera una fórmula local, devolvería (1 - 0.5/0.8)**2 != 777.0
    assert res == 777.0
    assert len(llamadas) == 1
    assert llamadas[0] == (0.5, 2.0, 0.8)


def test_agent_state_urgency_no_contiene_operaciones_aritmeticas_locales_de_urgencia() -> None:
    """Verifica por AST que AgentState.urgency no reintroduce la fórmula de potencia
    localmente (I-017).
    """
    source = textwrap.dedent(inspect.getsource(AgentState.urgency))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)), (
            "AgentState.urgency no debe reimplementar la fórmula con operador **"
        )


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


# =============================================================================
# Bloque Remediación I-028 (§5.5): Aptitud (gains) y Temple (costs) en apply y endure
# =============================================================================


def test_apply_con_delta_positivo_escala_por_aptitud_gains() -> None:
    """apply con delta positivo multiplica la ganancia por self.gains (I-028).

    Ancla contra mutación: quitar '* self.gains.get(need_id, 1.0)' hace que el
    incremento sea igual al delta sin amplificar por la aptitud del agente.
    """
    pack = _pack_para_recomposicion()
    agente = _agent({NeedId("n1"): 0.2}, ledger=Ledger())
    agente.traits = {TraitId("t1"): 1.0}
    agente.recompose(pack)
    assert agente.gains[NeedId("n1")] == pytest.approx(1.5)

    need_spec = pack.need_by_id[NeedId("n1")]
    agente.apply({NeedId("n1"): 0.2}, {NeedId("n1"): need_spec})

    # Delta 0.2 * aptitud 1.5 = 0.3 ganancia efectiva
    # Nivel: 0.2 + 0.3 = 0.5
    assert agente.needs[NeedId("n1")] == pytest.approx(0.5)
    assert agente.ledger is not None
    assert agente.ledger.active[NeedId("n1")] == pytest.approx(0.3)


def test_apply_con_delta_negativo_escala_por_temple_costs() -> None:
    """apply con delta negativo multiplica el gasto por self.costs (I-028).

    Ancla contra mutación: quitar '* self.costs.get(need_id, 1.0)' hace que el
    coste sea igual a abs(delta) sin amortiguar por el temple del agente.
    """
    pack = _pack_para_recomposicion()
    agente = _agent({NeedId("n1"): 0.8}, ledger=Ledger())
    agente.traits = {TraitId("t1"): 1.0}
    agente.recompose(pack)
    assert agente.costs[NeedId("n1")] == pytest.approx(0.5)

    need_spec = pack.need_by_id[NeedId("n1")]
    agente.apply({NeedId("n1"): -0.4}, {NeedId("n1"): need_spec})

    # Gasto nominal 0.4 * temple 0.5 = 0.2 gasto efectivo
    # Nivel: 0.8 - 0.2 = 0.6
    assert agente.needs[NeedId("n1")] == pytest.approx(0.6)
    assert agente.ledger is not None
    assert agente.ledger.demand[NeedId("n1")] == pytest.approx(0.2)


def test_endure_con_delta_negativo_escala_por_temple_costs() -> None:
    """endure con delta negativo multiplica el gasto por self.costs (I-028).

    Ancla contra mutación: quitar '* self.costs.get(need_id, 1.0)' en endure
    hace que el coste ambiental no se amortigüe por el temple del agente.
    """
    pack = _pack_para_recomposicion()
    agente = _agent({NeedId("n1"): 0.8}, ledger=Ledger())
    agente.traits = {TraitId("t1"): 1.0}
    agente.recompose(pack)
    assert agente.costs[NeedId("n1")] == pytest.approx(0.5)

    need_spec = pack.need_by_id[NeedId("n1")]
    agente.endure({NeedId("n1"): -0.4}, {NeedId("n1"): need_spec})

    # Gasto nominal 0.4 * temple 0.5 = 0.2 gasto efectivo
    # Nivel: 0.8 - 0.2 = 0.6
    assert agente.needs[NeedId("n1")] == pytest.approx(0.6)
    assert agente.ledger is not None
    assert agente.ledger.demand[NeedId("n1")] == pytest.approx(0.2)


def test_endure_con_delta_positivo_no_escala_por_aptitud_gains() -> None:
    """endure con delta positivo NO escala por self.gains (I-028).

    La aptitud mide rendimiento al actuar deliberadamente; los efectos ambientales
    pasivos entran directamente sin amplificación por aptitud.

    Ancla contra mutación: añadir '* self.gains.get(need_id, 1.0)' en el lado
    positivo de endure amplifica la entrada pasiva (0.2 * 1.5 = 0.3), haciendo
    fallar la comprobación de que el nivel queda exactamente en 0.4.
    """
    pack = _pack_para_recomposicion()
    agente = _agent({NeedId("n1"): 0.2}, ledger=Ledger())
    agente.traits = {TraitId("t1"): 1.0}
    agente.recompose(pack)
    assert agente.gains[NeedId("n1")] == pytest.approx(1.5)

    need_spec = pack.need_by_id[NeedId("n1")]
    agente.endure({NeedId("n1"): 0.2}, {NeedId("n1"): need_spec})

    # Efecto pasivo puro: exactamente 0.2, sin multiplicar por gains (1.5)
    # Nivel: 0.2 + 0.2 = 0.4 (si multiplicara daría 0.5)
    assert agente.needs[NeedId("n1")] == pytest.approx(0.4)
    assert agente.ledger is not None
    assert agente.ledger.passive[NeedId("n1")] == pytest.approx(0.2)
    assert NeedId("n1") not in agente.ledger.active


# =====================================================================
# Unidad U5a (§6.1 y §6.2 de 12-plan-u5a.md)
# =====================================================================


def test_from_spec_initial_needs_override_pack_default() -> None:
    """AgentState.from_spec aplica initial_needs sobre el valor por defecto de NeedSpec.

    Ancla de mutación 1: omitir 'needs.update(spec.initial_needs)' provoca que
    el agente conserve el default del pack (0.2) en vez del override (0.85).
    """
    n1 = ontology.NeedSpec(
        id=ontology.NeedId("n1"),
        label="Necesidad 1",
        decay_per_tick=0.1,
        initial=0.2,
    )
    n2 = ontology.NeedSpec(
        id=ontology.NeedId("n2"),
        label="Necesidad 2",
        decay_per_tick=0.1,
        initial=0.5,
    )
    pack = _pack_para_recomposicion(needs=(n1, n2))

    spec = ontology.AgentSpec(
        id=ontology.AgentId("agente_u5a"),
        label="Agente U5a",
        initial_needs={"n1": 0.85},
    )

    agente = state.AgentState.from_spec(pack, spec)
    # n1 fue sobreescrito a 0.85 (no el 0.2 default del pack)
    assert agente.needs[ontology.NeedId("n1")] == pytest.approx(0.85)
    # n2 conservó el default del pack (0.5)
    assert agente.needs[ontology.NeedId("n2")] == pytest.approx(0.5)


def test_from_spec_goals_initialization_locked_vs_active() -> None:
    """AgentState.from_spec inicia metas dependientes en LOCKED e independientes en ACTIVE.

    Ancla de mutación 2: invertir la condición LOCKED/ACTIVE provoca que la meta independiente
    nazca LOCKED y la dependiente nazca ACTIVE.
    """
    g_raiz = ontology.GoalSpec(
        id=ontology.GoalId("meta_raiz"),
        label="Meta Raíz",
        need=ontology.NeedId("n1"),
        at_least=0.8,
        requires_goal=None,
    )
    g_dep = ontology.GoalSpec(
        id=ontology.GoalId("meta_dependiente"),
        label="Meta Dependiente",
        need=ontology.NeedId("n1"),
        at_least=0.8,
        requires_goal=ontology.GoalId("meta_raiz"),
    )
    pack = _pack_para_recomposicion()
    spec = ontology.AgentSpec(
        id=ontology.AgentId("agente_metas"),
        label="Agente Metas",
        goals=(g_raiz, g_dep),
    )

    agente = state.AgentState.from_spec(pack, spec)
    assert agente.goals[ontology.GoalId("meta_raiz")] is ontology.GoalStatus.ACTIVE
    assert agente.goals[ontology.GoalId("meta_dependiente")] is ontology.GoalStatus.LOCKED


def test_from_spec_calls_recompose_and_initializes_temperament_and_tastes() -> None:
    """AgentState.from_spec llama a recompose y copia diccionarios defensivamente.

    Ancla de mutación 3: omitir 'agent.recompose(pack)' deja 'temperament is None'.
    """
    pack = _pack_para_recomposicion()
    spec = ontology.AgentSpec(
        id=ontology.AgentId("agente_tastes"),
        label="Agente Tastes",
        traits={"t1": 0.5},
        tastes=(ontology.TasteSpec(target="act1", multiplier=1.8),),
    )

    agente = state.AgentState.from_spec(pack, spec)
    # Recompose fue invocado
    assert agente.temperament is not None
    assert agente.temperament.loss_aversion is not None
    assert agente.need_weights != {}
    assert agente.gains != {}
    assert agente.costs != {}
    assert agente.caps != {}

    # Gustos compuestos correctamente
    assert agente.tastes == {"act1": 1.8}

    # Copia defensiva de rasgos
    assert agente.traits == {ontology.TraitId("t1"): 0.5}
    agente.traits[ontology.TraitId("t1")] = 0.9
    assert spec.traits[ontology.TraitId("t1")] == 0.5


def test_advance_decrements_ticks_remaining() -> None:
    """AgentState.advance consume 1 tick de la acción en curso preservando performing.

    Ancla de mutación 4: sustituir ticks_remaining - 1 por ticks_remaining - 2
    produce ticks_remaining == 1 en lugar de 2.
    """
    agente = _agent({ontology.NeedId("n1"): 0.5})
    agente.performing = state.PerformingAction(
        action=ontology.ActionId("act1"),
        ticks_remaining=3,
        total_ticks=3,
    )

    agente.advance()

    assert agente.performing is not None
    assert agente.performing.ticks_remaining == 2
    assert agente.performing.total_ticks == 3
    assert agente.performing.action == ontology.ActionId("act1")


def test_advance_guard_at_zero_preserves_performing_without_crashing() -> None:
    """AgentState.advance no revienta y preserva performing cuando ticks_remaining == 0.

    Ancla de mutación 5: omitir 'or performing.ticks_remaining == 0' de la guarda
    intenta decrementar a 0 - 1 = -1, provocando fallo de validación en PerformingAction.
    """
    agente = _agent({ontology.NeedId("n1"): 0.5})
    agente.performing = state.PerformingAction(
        action=ontology.ActionId("act1"),
        ticks_remaining=0,
        total_ticks=3,
    )

    agente.advance()

    # No debe levantarse excepción y performing debe permanecer intacto con ticks_remaining=0
    assert agente.performing is not None
    assert agente.performing.ticks_remaining == 0
    assert agente.performing.action == ontology.ActionId("act1")


def test_advance_with_performing_none_does_nothing() -> None:
    """AgentState.advance cuando performing es None no revienta y mantiene performing=None."""
    agente = _agent({ontology.NeedId("n1"): 0.5})
    assert agente.performing is None

    agente.advance()

    assert agente.performing is None

