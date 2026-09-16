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

from aida.engine.disposition import Temperament
from aida.engine.events import Event, EventKind
from aida.engine.ontology import ActionId, IntentId, NeedId
from aida.engine.pack import ContentPack

UNIVERSAL_SYSTEM_PROMPT = (
    "Sigues al pie de la letra las instrucciones que llegan al principio "
    "de este texto. Nunca inventas datos, hechos o acciones que no se te "
    "hayan dado explícitamente. Respondes siempre en el formato pedido."
)
"""Lo único fijo que lleva cada proveedor (`cognition/providers.py`): una
disciplina universal, nunca una instrucción de tarea. Qué se redacta y con
qué reglas —operativo, charla libre, lo que venga después— lo decide quien
arma el `briefing`, nunca el adaptador. Así un modo nuevo no exige volver a
tocar los tres proveedores."""

OPERATIONAL_INSTRUCTIONS = (
    "Redactas en español lo que un asistente ya decidió hacer. No decides "
    "nada y no propones nada nuevo: solo conviertes los HECHOS que te dan en "
    "una respuesta natural y breve, para la persona que te está leyendo. No "
    "menciones ningún hecho que no esté en la lista. Si no hay hechos "
    "nuevos, dilo con naturalidad, sin inventar que pasó algo."
)
"""La instrucción de U7, con el mismo texto — solo cambia dónde vive: antes
en `providers.py` como `SYSTEM_PROMPT`, ahora dentro del `briefing` que
arma `build_briefing`."""

FREE_CHAT_INSTRUCTIONS = (
    "Charlas de forma natural con quien te habla — no hay ninguna tarea "
    "que reportar ahora mismo. Puedes bromear, comentar, tener opiniones "
    "sin consecuencia y llevar una conversación natural. Lo único que no "
    "puedes hacer es afirmar que hiciste, decidiste o vas a hacer algo "
    "concreto: eso solo lo cuenta el motor, nunca la charla libre. Sigue "
    "la guía de carácter que se te da, no un tono genérico."
)


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
    """El texto que recibe `CognitionProvider.think()`. La instrucción de
    tarea va primero, los hechos después — nunca al revés, para que un
    modelo con poca ventana de contexto no trunque justo los hechos."""
    lines = [OPERATIONAL_INSTRUCTIONS, "", "Hechos nuevos desde el último turno:"]
    lines += [f"- {f}" for f in facts]
    if previous_reply is not None:
        lines += [
            "",
            "Lo último que le dijiste a esta persona (mantén el tono, no lo repitas):",
            f'"{previous_reply}"',
        ]
    return "\n".join(lines)


def describe_temperament(temperament: Temperament) -> str:
    """Traduce las seis perillas ya compuestas del agente a una guía de
    estilo en prosa — nunca los números crudos. El motor decide el
    carácter; esto solo lo pone en palabras para quien redacta.

    Los umbrales son juicio de diseño, no calibración del pack: separan
    "marcado" de "normal" para cada perilla, con los valores por defecto
    de `TuningSpec` como referencia de lo neutro.
    """
    rasgos: list[str] = []

    if temperament.temperature < 0.10:
        rasgos.append("metódico, casi nunca improvisa")
    elif temperament.temperature > 0.50:
        rasgos.append("espontáneo, disfruta variar")

    if temperament.loss_aversion > 2.0:
        rasgos.append("cauteloso con lo que cuesta cualquier cosa")
    elif temperament.loss_aversion < 0.5:
        rasgos.append("vive el momento, no le pesa gastar")

    if temperament.critical_need_threshold > 0.30:
        rasgos.append("se alarma pronto")
    elif temperament.critical_need_threshold < 0.10:
        rasgos.append("aguanta mucho antes de alarmarse")

    if temperament.switch_cost > 0.60:
        rasgos.append("prefiere terminar lo que empieza")
    elif temperament.switch_cost < 0.20:
        rasgos.append("cambia de tema con facilidad")

    if temperament.novelty_weight > 0.70:
        rasgos.append("busca variedad, se cansa de lo repetido")
    elif temperament.novelty_weight < 0.30:
        rasgos.append("no le importa repetir lo mismo")

    if not rasgos:
        return "Carácter de quien habla: sin rasgos marcados, tono neutro."
    return "Carácter de quien habla: " + ", ".join(rasgos) + "."


def build_free_chat_briefing(
    utterance: str, *, voice_guide: str, previous_reply: str | None
) -> str:
    """El texto que recibe `CognitionProvider.think()` en modo charla
    libre. Sin hechos que reportar — la instrucción, el carácter, y lo que
    dijo la persona."""
    lines = [FREE_CHAT_INSTRUCTIONS, "", voice_guide, "", f'La persona dijo: "{utterance}"']
    if previous_reply is not None:
        lines += [
            "",
            "Lo último que le dijiste (mantén el tono, no lo repitas):",
            f'"{previous_reply}"',
        ]
    return "\n".join(lines)


def fallback_free_chat_reply() -> str:
    """Sin proveedor no hay con qué charlar de verdad. No se inventa una
    broma ni una respuesta que finja naturalidad — se dice la verdad."""
    return "Ahora mismo no puedo charlar con naturalidad: no tengo el modelo disponible."


def fallback_reply(facts: list[str]) -> str:
    """Redacción sin modelo: siempre disponible, siempre determinista.
    Es lo que sale si no hay proveedor o si el proveedor falla."""
    return " ".join(facts)
