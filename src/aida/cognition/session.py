"""Un turno de conversación: empuja al buzón, avanza el bucle un tick,
junta lo nuevo, y lo convierte en una frase.

El hilo de la conversación no vive en el modelo: `Session` guarda el `seq`
del último evento ya contado y la última respuesta dada, y eso es TODO el
estado de continuidad. Cambiar de proveedor a mitad de conversación no
pierde el hilo, porque el hilo nunca estuvo en el proveedor.
"""

from __future__ import annotations

from dataclasses import dataclass

from aida.agents.decide import critical_needs
from aida.cognition.briefing import (
    SpokenReply,
    build_briefing,
    build_facts,
    build_free_chat_briefing,
    describe_temperament,
    fallback_free_chat_reply,
    fallback_reply,
)
from aida.engine.events import Event, EventKind
from aida.engine.loop import Loop
from aida.engine.ports import CognitionProvider


@dataclass(slots=True)
class Session:
    loop: Loop
    provider: CognitionProvider | None
    last_reported_seq: int = -1
    last_reply: str | None = None
    free_chat_enabled: bool = False
    """Apagado por defecto: con `False`, `turn()` se comporta exactamente
    igual que en U7 — ni un evento nuevo, ni una rama nueva. Se enciende
    explícitamente para el agente que sí debe poder charlar sin delegar
    una tarea."""

    def turn(self, utterance: str) -> str:
        """Un mensaje entra, una respuesta sale. Avanza el `Loop` UN tick —
        el de este mensaje; no hay reloj de fondo corriendo entre turnos,
        ver §1 del plan de U7."""
        agent = self.loop.execution.agent
        tick = self.loop.execution.tick + 1
        self.loop.inbox.push(tick=tick, owner=agent.agent_id, content=utterance)
        self.loop.step()

        new_events = [e for e in self.loop.log if e.seq > self.last_reported_seq]
        if new_events:
            self.last_reported_seq = new_events[-1].seq

        use_free_chat = self.free_chat_enabled and not self._something_to_report(new_events)
        if use_free_chat:
            reply = self._redact_free_chat(utterance)
        else:
            reply = self._redact_operational(new_events)

        if self.free_chat_enabled:
            self.loop.log.emit(
                tick=self.loop.execution.tick,
                kind=EventKind.REPLY_GIVEN,
                owner=agent.agent_id,
                data={"mode": "libre" if use_free_chat else "operativo"},
            )

        self.last_reply = reply
        return reply

    def _something_to_report(self, new_events: list[Event]) -> bool:
        """Verdadero si hay algo urgente ahora mismo, o si el motor
        reconoció la entrada como una petición concreta este turno — en
        cualquiera de los dos casos, la charla libre queda descartada
        aunque esté activada.

        Lee `INTENT_CLASSIFIED` del log en vez de volver a llamar al
        clasificador: `loop.step()` ya clasificó este turno si había
        clasificador, y preguntar dos veces sería la misma comprobación en
        dos sitios que la regla de determinismo prohíbe."""
        if critical_needs(self.loop.execution, self.loop.pack):
            return True
        return any(
            e.kind == EventKind.INTENT_CLASSIFIED and "intent" in e.data for e in new_events
        )

    def _redact_operational(self, new_events: list[Event]) -> str:
        """Pide al proveedor una redacción oral coherente de lo que pasó.
        Si el proveedor falla o no existe, degrada al redactor determinista
        para que la conversación nunca se interrumpa por un fallo remoto."""
        facts = build_facts(self.loop.pack, new_events)
        briefing = build_briefing(facts, previous_reply=self.last_reply)
        if self.provider is not None:
            try:
                result = self.provider.think(briefing, SpokenReply)
            except Exception:
                result = None
            if isinstance(result, SpokenReply):
                return result.text
        return fallback_reply(facts)

    def _redact_free_chat(self, utterance: str) -> str:
        """Charla libre: sin hechos que reportar, con el carácter del
        agente. Mismo respaldo determinista si no hay proveedor — aquí no
        es una concatenación de hechos, es una frase honesta sobre no
        poder charlar sin modelo."""
        temperament = self.loop.execution.agent.temperament
        guide = describe_temperament(temperament) if temperament is not None else ""
        briefing = build_free_chat_briefing(
            utterance, voice_guide=guide, previous_reply=self.last_reply
        )
        if self.provider is not None:
            try:
                result = self.provider.think(briefing, SpokenReply)
            except Exception:
                result = None
            if isinstance(result, SpokenReply):
                return result.text
        return fallback_free_chat_reply()
