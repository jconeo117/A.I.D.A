"""Hechos que el motor manda al modelo, y la forma de lo que puede volver.

Un hecho es UNA frase que describe lo que ya ocurrió, con las etiquetas que
el pack declara (`Spec.label`) — nunca un id técnico, nunca un número de
utilidad. `build_facts` es exhaustivo sobre lo que emite el motor hoy: cubre
los nueve `EventKind` que algún módulo llega a emitir (comprobado por
`grep -rn "EventKind\\." src/aida/engine/*.py`). Los seis restantes
(`GOAL_*`, `NEED_CRITICAL`, `SITUATION_*`) están declarados en `events.py`
pero ningún módulo los produce todavía: inventarles una traducción sería
construir sobre un `data` que nadie ha decidido. Caen en el genérico.

El modelo nunca ve estos eventos en crudo: solo la frase ya redactada. Así
"cambiar de proveedor no cambia lo que se cuenta" se cumple por
construcción, no por disciplina de prompt.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from aida.engine.events import Event, EventKind
from aida.engine.ontology import ActionId, IntentId, NeedId
from aida.engine.pack import ContentPack

SYSTEM_PROMPT = (
    "Redactas en español lo que un asistente ya decidió hacer. No decides "
    "nada y no propones nada nuevo: solo conviertes los HECHOS que te dan en "
    "una respuesta natural y breve, para la persona que te está leyendo. No "
    "menciones ningún hecho que no esté en la lista. Si no hay hechos "
    "nuevos, dilo con naturalidad, sin inventar que pasó algo."
)
"""Instrucción fija, igual para los tres proveedores. Vive aquí y no en cada
adaptador para que "cambiar de proveedor" no pueda, ni por descuido, cambiar
también la regla que los ata a los hechos."""


class SpokenReply(BaseModel):
    """Lo único que un proveedor puede devolver. `extra="forbid"`: un campo
    de más en la respuesta del modelo se rechaza, no se ignora en silencio.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)


def _fact_for(event: Event, pack: ContentPack) -> str:
    """Un evento, una frase. Genérico si el tipo no tiene traducción propia."""
    if event.kind in (
        EventKind.ACTION_STARTED,
        EventKind.ACTION_COMPLETED,
        EventKind.ACTION_AUTHORIZED,
        EventKind.ACTION_DENIED,
    ):
        label = pack.action_by_id[ActionId(event.action)].label if event.action else "?"
        verbo = {
            EventKind.ACTION_STARTED: "Empezó a hacer",
            EventKind.ACTION_COMPLETED: "Terminó de hacer",
            EventKind.ACTION_AUTHORIZED: "Se autorizó",
            EventKind.ACTION_DENIED: "Se denegó",
        }[event.kind]
        return f"{verbo}: {label}."

    if event.kind == EventKind.ACTION_ABORTED:
        action_label = pack.action_by_id[ActionId(event.action)].label if event.action else "?"
        need_id = event.data.get("need")
        need_label = pack.need_by_id[NeedId(str(need_id))].label if need_id else "algo urgente"
        return f"Interrumpió {action_label} porque {need_label} necesitaba atención urgente."

    if event.kind == EventKind.ACTION_PROPOSED:
        label = pack.action_by_id[ActionId(event.action)].label if event.action else "?"
        return f"Propuso {label} y está esperando autorización."

    if event.kind == EventKind.ACTION_EXPIRED:
        label = pack.action_by_id[ActionId(event.action)].label if event.action else "?"
        return f"Nadie respondió a tiempo a la propuesta de {label}, y caducó."

    if event.kind == EventKind.NEED_ESCALATED:
        need_id = event.data.get("need")
        need_label = pack.need_by_id[NeedId(str(need_id))].label if need_id else "algo"
        return f"{need_label} lleva mucho tiempo sin atención."

    if event.kind == EventKind.INTENT_CLASSIFIED:
        intent_id = event.data.get("intent")
        if intent_id is None:
            return "No se reconoció ninguna petición concreta en lo que se dijo."
        label = pack.intent_by_id[IntentId(str(intent_id))].label
        return f"Se reconoció la petición: {label}."

    return f"Ocurrió un evento interno ({event.kind.value})."


def build_facts(pack: ContentPack, events: Iterable[Event]) -> list[str]:
    """Un hecho por evento relevante, en el orden en que ocurrieron.

    `INPUT_OBSERVED` se omite a propósito: es el propio mensaje del usuario,
    contarle de vuelta lo que acaba de escribir no es un hecho nuevo.
    """
    facts = [
        _fact_for(event, pack) for event in events if event.kind != EventKind.INPUT_OBSERVED
    ]
    if not facts:
        facts = ["No hay novedades desde el último turno."]
    return facts


def build_briefing(facts: list[str], *, previous_reply: str | None) -> str:
    """El texto que recibe `CognitionProvider.think()`. Hechos primero,
    contexto de estilo después — nunca al revés, para que un modelo con
    poca ventana de contexto no trunque justo los hechos."""
    lines = ["Hechos nuevos desde el último turno:"]
    lines += [f"- {f}" for f in facts]
    if previous_reply is not None:
        lines += [
            "",
            "Lo último que le dijiste a esta persona (mantén el tono, no lo repitas):",
            f'"{previous_reply}"',
        ]
    return "\n".join(lines)


def fallback_reply(facts: list[str]) -> str:
    """Redacción sin modelo: siempre disponible, siempre determinista.
    Es lo que sale si no hay proveedor o si el proveedor falla."""
    return " ".join(facts)
