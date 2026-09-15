"""Pruebas de la preparación del briefing y hechos de cognición.

Cubre §6.1 de `.claude/workflow/24-plan-u7.md`:
1. Traducción exacta de los 9 EventKind reales a hechos con etiquetas del pack.
2. Evento sin traducción propia (SITUATION_BEGAN) cae en mensaje genérico defensivo.
3. Evento INPUT_OBSERVED se omite y resulta en aviso de no novedades.
4. Lista de eventos vacía resulta en aviso de no novedades.
5. Inclusión/omisión de contexto de estilo previo en build_briefing.
6. Concatenación determinista de hechos en fallback_reply.
7. Validación de SpokenReply (rechazo de texto vacío y campos extra).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import aida.engine.events as events
import aida.engine.ontology as ontology
import aida.engine.pack as pack
from aida.cognition.briefing import (
    SpokenReply,
    build_briefing,
    build_facts,
    fallback_reply,
)


def _crear_pack_prueba() -> pack.ContentPack:
    """Construye un content pack mínimo para pruebas de hechos sin vocabulario de negocio."""
    necesidad = ontology.NeedSpec(
        id=ontology.NeedId("n1"),
        label="Necesidad Beta",
        decay_per_tick=0.01,
        initial=0.5,
        cap=1.0,
    )
    accion = ontology.ActionSpec(
        id=ontology.ActionId("act1"),
        label="Acción Alfa",
        duration_ticks=1,
        reversibility=ontology.Reversibility.REVERSIBLE,
        interruptible=True,
        satisfies={necesidad.id: 0.5},
        effects={necesidad.id: 0.5},
    )
    intencion = ontology.IntentSpec(
        id=ontology.IntentId("int1"),
        label="Petición Gamma",
        need=necesidad.id,
        examples=("peticion gamma",),
    )
    agente = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agente Alfa",
        initial_needs={necesidad.id: 0.5},
    )
    tuning = ontology.TuningSpec(candidate_pool=1, temperature=0.01)
    return pack.ContentPack(
        id="pack_briefing_test",
        label="Pack Briefing Test",
        needs=[necesidad],
        actions=[accion],
        agents=[agente],
        tuning=tuning,
        intents=[intencion],
    )


@pytest.mark.parametrize(
    ("kind", "action_id", "data", "frase_esperada"),
    [
        (
            events.EventKind.ACTION_STARTED,
            ontology.ActionId("act1"),
            events.NO_DATA,
            "Empezó a hacer: Acción Alfa.",
        ),
        (
            events.EventKind.ACTION_COMPLETED,
            ontology.ActionId("act1"),
            events.NO_DATA,
            "Terminó de hacer: Acción Alfa.",
        ),
        (
            events.EventKind.ACTION_AUTHORIZED,
            ontology.ActionId("act1"),
            events.NO_DATA,
            "Se autorizó: Acción Alfa.",
        ),
        (
            events.EventKind.ACTION_DENIED,
            ontology.ActionId("act1"),
            events.NO_DATA,
            "Se denegó: Acción Alfa.",
        ),
        (
            events.EventKind.ACTION_ABORTED,
            ontology.ActionId("act1"),
            {"need": "n1"},
            "Interrumpió Acción Alfa porque Necesidad Beta necesitaba atención urgente.",
        ),
        (
            events.EventKind.ACTION_PROPOSED,
            ontology.ActionId("act1"),
            events.NO_DATA,
            "Propuso Acción Alfa y está esperando autorización.",
        ),
        (
            events.EventKind.ACTION_EXPIRED,
            ontology.ActionId("act1"),
            events.NO_DATA,
            "Nadie respondió a tiempo a la propuesta de Acción Alfa, y caducó.",
        ),
        (
            events.EventKind.NEED_ESCALATED,
            None,
            {"need": "n1"},
            "Necesidad Beta lleva mucho tiempo sin atención.",
        ),
        (
            events.EventKind.INTENT_CLASSIFIED,
            None,
            {"intent": "int1"},
            "Se reconoció la petición: Petición Gamma.",
        ),
        (
            events.EventKind.INTENT_CLASSIFIED,
            None,
            events.NO_DATA,
            "No se reconoció ninguna petición concreta en lo que se dijo.",
        ),
    ],
)
def test_build_facts_traduce_los_nueve_event_kinds_reales(
    kind: events.EventKind,
    action_id: ontology.ActionId | None,
    data: events.Payload,
    frase_esperada: str,
) -> None:
    """Caso 1: cada uno de los 9 tipos de eventos se traduce a una frase exacta con etiquetas."""
    test_pack = _crear_pack_prueba()
    evento = events.Event(
        seq=1,
        tick=1,
        kind=kind,
        owner=ontology.AgentId("agent_01"),
        action=action_id,
        data=data,
    )
    hechos = build_facts(test_pack, [evento])
    assert hechos == [frase_esperada]


def test_build_facts_evento_sin_traduccion_propia_cae_en_generico() -> None:
    """Caso 2: un EventKind no emitido (SITUATION_BEGAN) cae en el formato genérico."""
    test_pack = _crear_pack_prueba()
    evento = events.Event(
        seq=1,
        tick=1,
        kind=events.EventKind.SITUATION_BEGAN,
        owner=ontology.AgentId("agent_01"),
    )
    hechos = build_facts(test_pack, [evento])
    assert hechos == [f"Ocurrió un evento interno ({events.EventKind.SITUATION_BEGAN.value})."]


def test_build_facts_omite_input_observed() -> None:
    """Caso 3: INPUT_OBSERVED no se traduce como hecho observable y da mensaje por defecto."""
    test_pack = _crear_pack_prueba()
    evento = events.Event(
        seq=1,
        tick=1,
        kind=events.EventKind.INPUT_OBSERVED,
        owner=ontology.AgentId("agent_01"),
        content="mensaje de entrada del usuario",
    )
    hechos = build_facts(test_pack, [evento])
    assert hechos == ["No hay novedades desde el último turno."]


def test_build_facts_lista_vacia_da_mensaje_por_defecto() -> None:
    """Caso 4: una lista vacía de eventos resulta en el mensaje determinista por defecto."""
    test_pack = _crear_pack_prueba()
    hechos = build_facts(test_pack, [])
    assert hechos == ["No hay novedades desde el último turno."]


def test_build_briefing_con_y_sin_previous_reply() -> None:
    """Caso 5: build_briefing omite la sección de tono si previous_reply es None, y la incluye
    si existe.
    """
    hechos = ["Empezó a hacer: Acción Alfa."]

    briefing_sin_previo = build_briefing(hechos, previous_reply=None)
    assert "Hechos nuevos desde el último turno:" in briefing_sin_previo
    assert "- Empezó a hacer: Acción Alfa." in briefing_sin_previo
    assert "Lo último que le dijiste a esta persona" not in briefing_sin_previo

    respuesta_previa = "Ya estoy en ello, revisando el estado."
    briefing_con_previo = build_briefing(hechos, previous_reply=respuesta_previa)
    assert "Lo último que le dijiste a esta persona" in briefing_con_previo
    assert f'"{respuesta_previa}"' in briefing_con_previo


def test_fallback_reply_concatena_hechos_con_espacio() -> None:
    """Caso 6: fallback_reply une los hechos con espacio simple de manera determinista."""
    hechos = ["Hecho alfa.", "Hecho beta."]
    assert fallback_reply(hechos) == "Hecho alfa. Hecho beta."
    assert fallback_reply([]) == ""


def test_spoken_reply_rechaza_texto_vacio_y_campo_extra() -> None:
    """Caso 7: SpokenReply exige min_length=1 y rechaza cualquier campo no
    declarado (extra=forbid).
    """
    valido = SpokenReply(text="Todo en orden.")
    assert valido.text == "Todo en orden."

    with pytest.raises(ValidationError):
        SpokenReply(text="")

    with pytest.raises(ValidationError):
        SpokenReply.model_validate({"text": "Texto válido", "campo_extra": "prohibido"})
