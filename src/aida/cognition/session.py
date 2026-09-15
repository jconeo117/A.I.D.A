"""Un turno de conversación: empuja al buzón, avanza el bucle un tick,
junta lo nuevo, y lo convierte en una frase.

El hilo de la conversación no vive en el modelo: `Session` guarda el `seq`
del último evento ya contado y la última respuesta dada, y eso es TODO el
estado de continuidad. Cambiar de proveedor a mitad de conversación no
pierde el hilo, porque el hilo nunca estuvo en el proveedor.
"""

from __future__ import annotations

from dataclasses import dataclass

from aida.cognition.briefing import SpokenReply, build_briefing, build_facts, fallback_reply
from aida.engine.loop import Loop
from aida.engine.ports import CognitionProvider


@dataclass(slots=True)
class Session:
    loop: Loop
    provider: CognitionProvider | None
    last_reported_seq: int = -1
    last_reply: str | None = None

    def turn(self, utterance: str) -> str:
        """Un mensaje entra, una respuesta sale. Avanza el `Loop` UN tick —
        el de este mensaje; no hay reloj de fondo corriendo entre turnos,
        ver §1 del plan."""
        agent = self.loop.execution.agent
        tick = self.loop.execution.tick + 1
        self.loop.inbox.push(tick=tick, owner=agent.agent_id, content=utterance)
        self.loop.step()

        new_events = [e for e in self.loop.log if e.seq > self.last_reported_seq]
        facts = build_facts(self.loop.pack, new_events)
        reply = self._redact(facts)

        if new_events:
            self.last_reported_seq = new_events[-1].seq
        self.last_reply = reply
        return reply

    def _redact(self, facts: list[str]) -> str:
        """Pide al proveedor una redacción oral coherente. Si el proveedor
        falla o no existe, degrada al redactor determinista para que la
        conversación nunca se interrumpa por un fallo remoto."""
        briefing = build_briefing(facts, previous_reply=self.last_reply)
        if self.provider is not None:
            try:
                result = self.provider.think(briefing, SpokenReply)
            except Exception:
                result = None
            if isinstance(result, SpokenReply):
                return result.text
        return fallback_reply(facts)
