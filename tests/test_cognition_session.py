"""Pruebas de la orquestación de sesión conversacional (Session).

Cubre §6.3 de `.claude/workflow/24-plan-u7.md`:
1. Session.turn() con proveedor fijo devuelve exactamente su texto.
2. Session.turn() con provider=None devuelve fallback_reply (hechos concatenados).
3. Segundo turn() no repite hechos ya contados (avance de last_reported_seq).
4. Segundo turn() incluye la respuesta anterior en el briefing como contexto de estilo.
5. Turno sin eventos contables produce el hecho por defecto de no novedades.
6. Session.turn() avanza el tick del bucle exactamente en 1 por llamada y encola
   hacia el tick entrante (tick + 1).
7. Cobertura defensiva de _redact: proveedor que devuelve un objeto no-SpokenReply
   cae a fallback_reply sin reventar (protege M10).
"""

from __future__ import annotations

from typing import cast

import pytest

import aida.engine.events as events
import aida.engine.loop as loop
import aida.engine.ontology as ontology
import aida.engine.pack as pack
from aida.cognition.briefing import (
    FREE_CHAT_INSTRUCTIONS,
    OPERATIONAL_INSTRUCTIONS,
    fallback_free_chat_reply,
    fallback_reply,
)
from aida.cognition.session import Session
from aida.engine.ports import IntentClassifier
from aida.engine.queue import EventQueue, IncomingEvent
from aida.intent.classify import build_classifier


class _ProveedorFijo:
    """Proveedor de prueba con respuesta constante conforme a SpokenReply."""

    def __init__(self, texto: str) -> None:
        self._texto = texto

    def think(self, briefing: str, schema: type) -> object | None:
        return cast(object, schema(text=self._texto))


class _ProveedorEspia:
    """Proveedor espía que registra los briefings recibidos para verificar el estado."""

    def __init__(self, texto: str = "Respuesta del modelo") -> None:
        self._texto = texto
        self.briefings: list[str] = []

    def think(self, briefing: str, schema: type) -> object | None:
        self.briefings.append(briefing)
        return cast(object, schema(text=self._texto))


class _ProveedorNoSpokenReply:
    """Proveedor que devuelve un objeto ajeno a SpokenReply (dict)."""

    def think(self, briefing: str, schema: type) -> object | None:
        return {"text": "un diccionario en vez de instancia SpokenReply"}


def _crear_pack_sesion(
    *,
    con_acciones: bool = True,
    duration_ticks: int = 1,
    initial_need: float = 0.5,
    critical_need_threshold: float = 0.15,
    intents: list[ontology.IntentSpec] | None = None,
) -> pack.ContentPack:
    """Construye un content pack abstracto para pruebas de Session."""
    necesidad = ontology.NeedSpec(
        id=ontology.NeedId("n1"),
        label="Necesidad Alfa",
        decay_per_tick=0.01,
        initial=initial_need,
        cap=1.0,
    )
    acciones: list[ontology.ActionSpec] = []
    if con_acciones:
        acciones.append(
            ontology.ActionSpec(
                id=ontology.ActionId("act1"),
                label="Acción Alfa",
                duration_ticks=duration_ticks,
                reversibility=ontology.Reversibility.REVERSIBLE,
                interruptible=True,
                satisfies={necesidad.id: 0.5},
                effects={necesidad.id: 0.5},
            )
        )

    agente = ontology.AgentSpec(
        id=ontology.AgentId("agent_01"),
        label="Agente Alfa",
        initial_needs={necesidad.id: initial_need},
    )
    tuning = ontology.TuningSpec(
        candidate_pool=max(len(acciones), 1),
        temperature=0.01,
        critical_need_threshold=critical_need_threshold,
    )
    return pack.ContentPack(
        id="pack_session_test",
        label="Pack Session Test",
        needs=[necesidad],
        actions=acciones,
        agents=[agente],
        tuning=tuning,
        intents=intents or [],
    )


def _iniciar_loop(
    p: pack.ContentPack,
    intent: IntentClassifier | None = None,
) -> loop.Loop:
    return loop.Loop.start(p, p.agents[0], seed=42, inbox_capacity=32, intent=intent)


def test_session_turn_con_proveedor_fijo_devuelve_su_texto() -> None:
    """Caso 1: Session.turn() devuelve el texto redactado por el proveedor."""
    p = _crear_pack_sesion()
    lp = _iniciar_loop(p)
    proveedor = _ProveedorFijo(texto="Mensaje generado por el proveedor.")
    sesion = Session(loop=lp, provider=proveedor)

    respuesta = sesion.turn("Hola")
    assert respuesta == "Mensaje generado por el proveedor."


def test_session_turn_sin_proveedor_devuelve_fallback() -> None:
    """Caso 2: Session.turn() con provider=None devuelve fallback_reply (hechos enlazados)."""
    p = _crear_pack_sesion(con_acciones=True)
    lp = _iniciar_loop(p)
    sesion = Session(loop=lp, provider=None)

    respuesta = sesion.turn("Iniciar tarea")
    assert respuesta == "Empezó a hacer: Acción Alfa."


def test_session_segundo_turn_no_repite_hechos_anteriores() -> None:
    """Caso 3: last_reported_seq avanza y el segundo turno no repite eventos ya reportados."""
    # duration_ticks=3 asegura que la acción sigue en curso en el turno 2 y no recomienza
    p = _crear_pack_sesion(con_acciones=True, duration_ticks=3)
    lp = _iniciar_loop(p)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia)

    # Primer turno: inicia act1
    sesion.turn("Primer turno")
    assert len(espia.briefings) == 1
    assert "Empezó a hacer: Acción Alfa." in espia.briefings[0]

    # Segundo turno: la acción sigue en curso, no hay nuevo evento de inicio
    sesion.turn("Segundo turno")
    assert len(espia.briefings) == 2
    assert "Empezó a hacer: Acción Alfa." not in espia.briefings[1]


def test_session_segundo_turn_incluye_contexto_de_estilo_anterior() -> None:
    """Caso 4: el segundo turno incluye la respuesta del primer turno en el briefing."""
    p = _crear_pack_sesion()
    lp = _iniciar_loop(p)
    espia = _ProveedorEspia(texto="Respuesta previa de turno 1.")
    sesion = Session(loop=lp, provider=espia)

    sesion.turn("Hola 1")
    assert "Lo último que le dijiste a esta persona" not in espia.briefings[0]

    sesion.turn("Hola 2")
    assert "Lo último que le dijiste a esta persona" in espia.briefings[1]
    assert '"Respuesta previa de turno 1."' in espia.briefings[1]


def test_session_turno_sin_eventos_contables_da_mensaje_por_defecto() -> None:
    """Caso 5: un turno sin eventos nuevos observables produce el aviso por defecto."""
    p = _crear_pack_sesion(con_acciones=False)
    lp = _iniciar_loop(p)
    sesion = Session(loop=lp, provider=None)

    respuesta = sesion.turn("Mensaje neutro sin efecto")
    assert respuesta == "No hay novedades desde el último turno."


def test_session_turn_avanza_tick_exactamente_uno_por_llamada(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 6: Session.turn() avanza loop.execution.tick exactamente en 1 por turno y encola
    hacia tick + 1 (protege M11).
    """
    p = _crear_pack_sesion()
    lp = _iniciar_loop(p)
    sesion = Session(loop=lp, provider=None)

    ticks_empujados: list[int] = []
    push_original = EventQueue.push

    def _push_espia(
        self: EventQueue, *, tick: int, owner: ontology.AgentId, content: str
    ) -> IncomingEvent:
        ticks_empujados.append(tick)
        return push_original(self, tick=tick, owner=owner, content=content)

    monkeypatch.setattr(EventQueue, "push", _push_espia)

    assert sesion.loop.execution.tick == 0
    sesion.turn("Mensaje 1")
    assert sesion.loop.execution.tick == 1

    sesion.turn("Mensaje 2")
    assert sesion.loop.execution.tick == 2

    # Verifica que el mensaje se programó para el tick entrante y no para un tick ya pasado
    assert ticks_empujados == [1, 2]


def test_session_proveedor_no_spoken_reply_cae_a_fallback() -> None:
    """Caso 7 (M10): si el proveedor devuelve un objeto que no es SpokenReply, _redact
    cae defensivamente a fallback_reply sin levantar error ni asumir formato.
    """
    p = _crear_pack_sesion(con_acciones=True)
    lp = _iniciar_loop(p)
    sesion = Session(loop=lp, provider=_ProveedorNoSpokenReply())

    respuesta = sesion.turn("Iniciar acción")
    assert respuesta == "Empezó a hacer: Acción Alfa."


class _FailingProvider:
    """Proveedor simulado cuyo think() lanza RuntimeError no controlado."""

    def think(self, briefing: str, schema: type) -> object | None:
        raise RuntimeError("Simulated LLM failure")


def test_session_turn_with_failing_provider_falls_back_without_raising() -> None:
    """I-061 / V2: un fallo no controlado del proveedor LLM externo nunca debe interrumpir
    la interacción con el usuario; Session.turn debe degradar de forma transparente a la
    frase literal determinista producida por fallback_reply.
    """
    pack_spec = _crear_pack_sesion(con_acciones=True)
    engine_loop = _iniciar_loop(pack_spec)
    session = Session(loop=engine_loop, provider=_FailingProvider())

    response = session.turn("Iniciar acción")
    assert response == "Empezó a hacer: Acción Alfa."


def test_session_free_chat_activado_sin_urgencia_ni_intencion_usa_redact_free_chat() -> None:
    """Caso 1 (§6.3): free_chat_enabled=True, sin intención ni necesidad crítica
    usa _redact_free_chat.
    """
    p = _crear_pack_sesion(con_acciones=False, initial_need=0.5, critical_need_threshold=0.15)
    lp = _iniciar_loop(p)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia, free_chat_enabled=True)

    sesion.turn("Hola, ¿cómo estás?")
    assert len(espia.briefings) == 1
    assert FREE_CHAT_INSTRUCTIONS in espia.briefings[0]
    assert OPERATIONAL_INSTRUCTIONS not in espia.briefings[0]


def test_session_free_chat_con_necesidad_critica_fuerza_modo_operativo() -> None:
    """Caso 2 (§6.3): free_chat_enabled=True, pero necesidad por debajo del umbral crítico
    usa _redact_operational.
    """
    p = _crear_pack_sesion(con_acciones=False, initial_need=0.05, critical_need_threshold=0.50)
    lp = _iniciar_loop(p)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia, free_chat_enabled=True)

    sesion.turn("Hola, ¿cómo estás?")
    assert len(espia.briefings) == 1
    assert OPERATIONAL_INSTRUCTIONS in espia.briefings[0]
    assert FREE_CHAT_INSTRUCTIONS not in espia.briefings[0]


def test_session_free_chat_con_intencion_reconocida_fuerza_modo_operativo() -> None:
    """Caso 3 (§6.3): free_chat_enabled=True, pero la entrada dispara un IntentSpec
    usa _redact_operational.
    """
    necesidad_id = ontology.NeedId("n1")
    intencion = ontology.IntentSpec(
        id=ontology.IntentId("int_alfa"),
        label="Peticion Alfa",
        need=necesidad_id,
        examples=("peticion alfa", "ejecutar accion"),
    )
    p = _crear_pack_sesion(con_acciones=True, intents=[intencion])
    clasificador = build_classifier(p)
    lp = _iniciar_loop(p, intent=clasificador)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia, free_chat_enabled=True)

    sesion.turn("ejecutar accion")
    assert len(espia.briefings) == 1
    assert OPERATIONAL_INSTRUCTIONS in espia.briefings[0]
    assert FREE_CHAT_INSTRUCTIONS not in espia.briefings[0]


def test_session_free_chat_emite_reply_given_modo_libre() -> None:
    """Caso 4 (§6.3): free_chat_enabled=True en charla libre emite EventKind.REPLY_GIVEN
    con mode='libre'.
    """
    p = _crear_pack_sesion(con_acciones=False)
    lp = _iniciar_loop(p)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia, free_chat_enabled=True)

    sesion.turn("Charla informal sin tareas")
    eventos_reply = sesion.loop.log.of_kind(events.EventKind.REPLY_GIVEN)
    assert len(eventos_reply) == 1
    assert eventos_reply[0].data == {"mode": "libre"}


def test_session_free_chat_con_urgencia_o_intencion_emite_reply_given_modo_operativo() -> None:
    """Caso 5 (§6.3): free_chat_enabled=True en turno operativo emite EventKind.REPLY_GIVEN
    con mode='operativo'.
    """
    p = _crear_pack_sesion(con_acciones=False, initial_need=0.05, critical_need_threshold=0.50)
    lp = _iniciar_loop(p)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia, free_chat_enabled=True)

    sesion.turn("Hola")
    eventos_reply = sesion.loop.log.of_kind(events.EventKind.REPLY_GIVEN)
    assert len(eventos_reply) == 1
    assert eventos_reply[0].data == {"mode": "operativo"}


def test_session_free_chat_desactivado_no_emite_reply_given() -> None:
    """Caso 6 (§6.3): free_chat_enabled=False (default) no emite ningún REPLY_GIVEN en el log."""
    p = _crear_pack_sesion(con_acciones=False)
    lp = _iniciar_loop(p)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia)

    sesion.turn("Hola, no deberia emitir reply_given")
    eventos_reply = sesion.loop.log.of_kind(events.EventKind.REPLY_GIVEN)
    assert len(eventos_reply) == 0


def test_session_sin_proveedor_free_chat_activado_devuelve_fallback_free_chat() -> None:
    """Caso 7 (§6.3): sin proveedor y free_chat_enabled=True devuelve fallback_free_chat_reply
    exacto.
    """
    p = _crear_pack_sesion(con_acciones=False)
    lp = _iniciar_loop(p)
    sesion = Session(loop=lp, provider=None, free_chat_enabled=True)

    respuesta = sesion.turn("Hola")
    assert respuesta == fallback_free_chat_reply()
    assert respuesta != fallback_reply(["No hay novedades desde el último turno."])


def test_session_intencion_no_reconocida_permite_charla_libre() -> None:
    """Protección M7: un clasificador activo con intención no reconocida emite INTENT_CLASSIFIED
    sin 'intent' en data, lo que NO debe abortar la charla libre."""
    necesidad_id = ontology.NeedId("n1")
    intencion = ontology.IntentSpec(
        id=ontology.IntentId("int_alfa"),
        label="Peticion Alfa",
        need=necesidad_id,
        examples=("peticion alfa",),
    )
    p = _crear_pack_sesion(con_acciones=True, intents=[intencion])
    clasificador = build_classifier(p)
    lp = _iniciar_loop(p, intent=clasificador)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia, free_chat_enabled=True)

    sesion.turn("texto que no tiene relacion con la intencion")
    eventos_intent = sesion.loop.log.of_kind(events.EventKind.INTENT_CLASSIFIED)
    assert len(eventos_intent) == 1
    assert "intent" not in eventos_intent[0].data

    assert len(espia.briefings) == 1
    assert FREE_CHAT_INSTRUCTIONS in espia.briefings[0]
    assert OPERATIONAL_INSTRUCTIONS not in espia.briefings[0]
    eventos_reply = sesion.loop.log.of_kind(events.EventKind.REPLY_GIVEN)
    assert len(eventos_reply) == 1
    assert eventos_reply[0].data == {"mode": "libre"}


def test_session_free_chat_incluye_guia_de_caracter_en_briefing() -> None:
    """I-083: Session.turn() en modo charla libre debe incluir el temperamento traducido
    a guía de estilo en el briefing del proveedor (ancla que describe_temperament llegue
    efectivamente al LLM).
    """
    p = _crear_pack_sesion(con_acciones=False)
    lp = _iniciar_loop(p)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia, free_chat_enabled=True)

    sesion.turn("Hola, ¿cómo estás?")
    assert len(espia.briefings) == 1
    assert "Carácter de quien habla" in espia.briefings[0]
    assert "metódico, casi nunca improvisa" in espia.briefings[0]


def test_session_free_chat_incluye_utterance_literal_en_briefing() -> None:
    """I-084: Session.turn() en modo charla libre debe incluir el texto literal del usuario
    (utterance) en el briefing del proveedor.
    """
    p = _crear_pack_sesion(con_acciones=False)
    lp = _iniciar_loop(p)
    espia = _ProveedorEspia()
    sesion = Session(loop=lp, provider=espia, free_chat_enabled=True)

    entrada = "cuentame un chiste sobre pinguinos"
    sesion.turn(entrada)
    assert len(espia.briefings) == 1
    assert entrada in espia.briefings[0]

