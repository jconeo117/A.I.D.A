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
    FREE_CHAT_INSTRUCTIONS,
    OPERATIONAL_INSTRUCTIONS,
    SpokenReply,
    build_briefing,
    build_facts,
    build_free_chat_briefing,
    describe_temperament,
    fallback_free_chat_reply,
    fallback_reply,
)
from aida.engine.disposition import Temperament


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
    assert OPERATIONAL_INSTRUCTIONS in briefing_sin_previo
    assert "Hechos nuevos desde el último turno:" in briefing_sin_previo
    assert "- Empezó a hacer: Acción Alfa." in briefing_sin_previo
    assert "Lo último que le dijiste a esta persona" not in briefing_sin_previo

    respuesta_previa = "Ya estoy en ello, revisando el estado."
    briefing_con_previo = build_briefing(hechos, previous_reply=respuesta_previa)
    assert OPERATIONAL_INSTRUCTIONS in briefing_con_previo
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


def test_describe_temperament_valores_neutros_sin_rasgos_marcados() -> None:
    """Caso 1: describe_temperament con valores neutros da el texto 'sin rasgos marcados'."""
    temperament = Temperament(
        temperature=0.2,
        loss_aversion=1.0,
        critical_need_threshold=0.2,
        switch_cost=0.4,
        novelty_weight=0.5,
        learning_rate=0.0,
    )
    resultado = describe_temperament(temperament)
    assert "sin rasgos marcados" in resultado
    assert resultado == "Carácter de quien habla: sin rasgos marcados, tono neutro."


def test_describe_temperament_temperature_baja_incluye_metodico() -> None:
    """Caso 2: temperature < 0.10 incluye 'metódico'."""
    temperament = Temperament(
        temperature=0.05,
        loss_aversion=1.0,
        critical_need_threshold=0.2,
        switch_cost=0.4,
        novelty_weight=0.5,
        learning_rate=0.0,
    )
    resultado = describe_temperament(temperament)
    assert "metódico" in resultado


def test_describe_temperament_temperature_alta_incluye_espontaneo() -> None:
    """Caso 3: temperature > 0.50 incluye 'espontáneo'."""
    temperament = Temperament(
        temperature=0.8,
        loss_aversion=1.0,
        critical_need_threshold=0.2,
        switch_cost=0.4,
        novelty_weight=0.5,
        learning_rate=0.0,
    )
    resultado = describe_temperament(temperament)
    assert "espontáneo" in resultado


def test_describe_temperament_perillas_marcadas_orden_fijo() -> None:
    """Caso 4: Temperament con perillas fuera de rango neutro genera los fragmentos en orden."""
    temperament = Temperament(
        temperature=0.05,
        loss_aversion=3.0,
        critical_need_threshold=0.40,
        switch_cost=0.70,
        novelty_weight=0.80,
        learning_rate=0.5,
    )
    resultado = describe_temperament(temperament)
    esperados = [
        "metódico, casi nunca improvisa",
        "cauteloso con lo que cuesta cualquier cosa",
        "se alarma pronto",
        "prefiere terminar lo que empieza",
        "busca variedad, se cansa de lo repetido",
    ]
    for fragmento in esperados:
        assert fragmento in resultado

    posiciones = [resultado.index(f) for f in esperados]
    assert posiciones == sorted(posiciones)
    assert resultado == "Carácter de quien habla: " + ", ".join(esperados) + "."


def test_describe_temperament_limite_exacto_switch_cost() -> None:
    """Caso 5: switch_cost=0.60 (umbral exacto) no activa 'prefiere terminar lo que empieza'."""
    temperament = Temperament(
        temperature=0.2,
        loss_aversion=1.0,
        critical_need_threshold=0.2,
        switch_cost=0.60,
        novelty_weight=0.5,
        learning_rate=0.0,
    )
    resultado = describe_temperament(temperament)
    assert "prefiere terminar lo que empieza" not in resultado
    assert resultado == "Carácter de quien habla: sin rasgos marcados, tono neutro."


def test_build_free_chat_briefing_con_y_sin_previous_reply() -> None:
    """Caso 6: build_free_chat_briefing omite la sección previa si previous_reply es None,
    y la incluye con el texto literal cuando existe.
    """
    sin_previo = build_free_chat_briefing(
        "Hola", voice_guide="Carácter: neutro", previous_reply=None
    )
    assert "Lo último que le dijiste" not in sin_previo

    con_previo = build_free_chat_briefing(
        "Hola",
        voice_guide="Carácter: neutro",
        previous_reply="Buenas tardes, ¿en qué puedo ayudarte?",
    )
    assert "Lo último que le dijiste (mantén el tono, no lo repitas):" in con_previo
    assert '"Buenas tardes, ¿en qué puedo ayudarte?"' in con_previo


def test_build_free_chat_briefing_incluye_instrucciones_y_utterance_literal() -> None:
    """Caso 7: build_free_chat_briefing incluye FREE_CHAT_INSTRUCTIONS y utterance literal."""
    utterance = "Cuéntame una historia interesante."
    briefing = build_free_chat_briefing(
        utterance, voice_guide="Carácter de prueba", previous_reply=None
    )
    assert FREE_CHAT_INSTRUCTIONS in briefing
    assert f'La persona dijo: "{utterance}"' in briefing


def test_fallback_free_chat_reply_devuelve_frase_fija_determinista() -> None:
    """Caso 8: fallback_free_chat_reply devuelve siempre la misma frase fija sin argumentos."""
    esperado = "Ahora mismo no puedo charlar con naturalidad: no tengo el modelo disponible."
    r1 = fallback_free_chat_reply()
    r2 = fallback_free_chat_reply()
    assert r1 == esperado
    assert r2 == esperado

