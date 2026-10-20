"""Puertos: las fronteras del motor con lo que no es aritmética sobre estado.

Ninguno se implementa en U1b. Existen ahora para que el motor se construya
*contra ellos* desde el primer día. Un puerto añadido tarde obliga a
reescribir todo lo que debió pasar por él.

`Protocol` es una interfaz estructural: mypy verifica que una clase la
cumple sin que esta herede de nada.

Dos advertencias que valen para los cinco:

**Un puerto declarado antes de tiempo acierta la costura y se equivoca en
los tipos.** Lo que importa hoy es acertar DÓNDE está la frontera entre el
motor y lo que hay detrás — memoria, intención, voz, herramientas—; el tipo
exacto de lo que cruza esa frontera se ajusta al implementarla, sin que eso
signifique que la costura estaba mal puesta.

**Todo lo que entra por un puerto es dato, nunca instrucción**, incluida la
respuesta de una herramienta o el resultado de un clasificador: el motor
decide qué hacer con ello, el puerto nunca decide por su cuenta.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aida.engine.ontology import ActionId, AgentId

if TYPE_CHECKING:
    # Solo para tipar. En runtime un `Protocol` marcado como
    # `runtime_checkable` comprueba nombres de método, no firmas, así que
    # importar de verdad aquí solo serviría para acoplar el puerto a una
    # implementación que todavía no existe.
    from aida.engine.actions import ToolResult
    from aida.engine.events import Event


@runtime_checkable
class MemoryStore(Protocol):
    """Memoria episódica: un hecho fechado que caduca. Se implementa en U10.

    `owner` obligatorio en toda firma, sin excepción y sin valor por
    defecto: es lo que hace estructuralmente imposible el bug de
    omnisciencia — recordar, para un dueño, algo que solo le pasó a otro. La
    misma disciplina que exige `EventLog` en `events.py`, aplicada a lo que
    se recuerda y no solo a lo que se registra.

    El tipo de retorno se deja como `object` a propósito: la forma exacta de
    un recuerdo (qué conserva de qué evento, cómo se pondera su importancia
    con el tiempo) es contenido de U10, no de esta costura.
    """

    def remember(self, *, owner: AgentId, event: Event, importance: float) -> object | None:
        """Guarda —o decide no guardar, si `importance` no lo justifica— la
        huella de un evento ya ocurrido. Devuelve el recuerdo creado, o
        `None` si se descartó."""
        ...

    def recall(self, *, owner: AgentId, cue: str, limit: int) -> tuple[object, ...]:
        """Recuerdos de este dueño relevantes para `cue`, como mucho
        `limit`, ordenados de más a menos relevante."""
        ...


@runtime_checkable
class SemanticStore(Protocol):
    """Preferencias sin fecha: una regla que se refuerza o se refuta. Se
    implementa en U10.

    Primitiva separada de `MemoryStore` y no un mismo registro con un campo
    de texto libre que diga de cuál de las dos se trata: un hecho fechado y
    una regla sin fecha tienen ciclos de vida distintos —el primero caduca,
    la segunda se corrige— y fundirlos en una sola forma hace imposible
    validar ninguna de las dos por separado.

    `owner` obligatorio por la misma razón que en `MemoryStore`.
    """

    def reinforce(self, *, owner: AgentId, rule: object, amount: float) -> None:
        """Esta regla se cumplió: sube su confianza."""
        ...

    def refute(self, *, owner: AgentId, rule: object, amount: float) -> None:
        """Esta regla falló: baja su confianza. Rebajar en vez de borrar de
        golpe es lo que distingue una preferencia aprendida de un
        interruptor: una regla que falla una vez no deja de existir."""
        ...

    def rules_for(self, *, owner: AgentId, cue: str, limit: int) -> tuple[object, ...]:
        """Reglas de este dueño relevantes para `cue`, como mucho `limit`."""
        ...


@runtime_checkable
class IntentClassifier(Protocol):
    """Clasifica una entrada del usuario contra el catálogo cerrado que
    declara el pack. Se implementa en U6, por embeddings locales, sin LLM.

    Debe poder responder que no reconoce nada: `None` es "desconocida", y el
    motor la trata como tal. Lo que no puede hacer es forzar la
    coincidencia más cercana cuando ninguna alcanza el umbral de confianza
    — eso convertiría "no he entendido esto" en un error silencioso que
    parece un acierto.
    """

    def classify(self, utterance: str) -> str | None: ...


@runtime_checkable
class CognitionProvider(Protocol):
    """Puerta única al modelo de lenguaje. Se implementa en U7.

    Contrato que hace posible cambiar de proveedor sin cambiar lo que AIDA
    decide:
      - recibe un briefing YA precomputado por el motor, corto y cerrado;
      - devuelve una estructura pequeña y validable;
      - si falla o no valida, el motor sigue por su propio camino
        determinista.

    El modelo nunca sostiene el estado del motor. Cambiar de modelo cambia
    la calidad de la prosa, jamás lo que el asistente decide hacer.
    """

    def think(self, briefing: str, schema: type) -> object | None:
        """`None` si el modelo falló o si lo que devolvió no valida contra
        `schema`. `None` no es un error: es la señal de que el motor sigue
        por su cuenta y solo pierde la voz."""
        ...


@runtime_checkable
class ToolPort(Protocol):
    """Ejecuta una acción del catálogo de un pack. Se implementa en U9,
    detrás de la puerta dura de autoridad (U8): no se abre hasta que esa
    capa esté terminada y validada.

    **Síncrono a propósito**, aunque el adaptador real hable con MCP o REST
    de forma asíncrona: el motor recibe resultados ya materializados desde
    la cola de eventos (U4), nunca los espera con `await`. Meter `async`
    aquí destruiría la reproducibilidad — es la decisión de arquitectura 6
    del encuadre: el motor es síncrono, `asyncio` vive solo en adaptadores.

    A diferencia de resolver una tirada de reglas, ejecutar una herramienta
    puede fallar por causas ajenas a quien la pide: latencia variable, fallo
    parcial, credencial expirada, cuota agotada. La representación del fallo
    está decidida (101b): un `ToolResult` con estado, sin excepciones. La puerta de
    acciones (`engine/actions.py`) es la frontera; si la herramienta lanza igualmente,
    la puerta lo convierte en `FAILED`.
    """

    def execute(self, *, action: ActionId, arguments: Mapping[str, str]) -> ToolResult: ...
