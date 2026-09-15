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

import aida.engine.loop as loop
import aida.engine.ontology as ontology
import aida.engine.pack as pack
from aida.cognition.session import Session
from aida.engine.queue import EventQueue, IncomingEvent


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
) -> pack.ContentPack:
    """Construye un content pack abstracto para pruebas de Session."""
    necesidad = ontology.NeedSpec(
        id=ontology.NeedId("n1"),
        label="Necesidad Alfa",
        decay_per_tick=0.01,
        initial=0.5,
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
        initial_needs={necesidad.id: 0.5},
    )
    tuning = ontology.TuningSpec(candidate_pool=max(len(acciones), 1), temperature=0.01)
    return pack.ContentPack(
        id="pack_session_test",
        label="Pack Session Test",
        needs=[necesidad],
        actions=acciones,
        agents=[agente],
        tuning=tuning,
        intents=[],
    )


def _iniciar_loop(p: pack.ContentPack) -> loop.Loop:
    return loop.Loop.start(p, p.agents[0], seed=42, inbox_capacity=32)


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
