"""Registro de eventos: la única ventana al estado.

Regla dura: **nada muta el estado sin emitir un evento**. Es la fuente de la
que se alimentará la memoria (U10) y el rastro de auditoría que exige la
§5.5 del README: quién pidió qué, qué decidió el motor y por qué, qué
herramienta se llamó con qué argumentos, qué devolvió.

Si un cambio no deja evento, para las capas de arriba ese cambio nunca
ocurrió.

El log es append-only y cada entrada lleva un `seq` monótono. Ese `seq` es el
identificador con el que la memoria de U10 referenciará recuerdos, así que no
se reordena ni se reutiliza jamás.

Todo evento tiene un `owner`, sin excepción y sin `None`. Es el criterio G7
del encuadre y la razón está en la §5.5 del README: un asistente
multiusuario que filtra el contexto de un dueño a otro es un incidente de
seguridad, y el motor debe hacerlo estructuralmente imposible en vez de
confiar en que nadie se equivoque. Si el dueño pudiera ser `None`, existiría
un camino por el que un evento no pertenece a nadie, y ese camino acaba
siendo por donde se fuga un dato ajeno. Ni siquiera un latido de reloj o una
respuesta de herramienta quedan exentos: ocurren siempre por cuenta de
alguien.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import TextIO

from aida.engine.ontology import ActionId, AgentId

type Payload = Mapping[str, str | int | float | bool]
"""Datos extra del evento. Deliberadamente plano y sin objetos: el log debe
poder serializarse a JSONL sin ceremonia, y un briefing para un modelo
pequeño (U7) se construye mucho mejor desde valores escalares."""

NO_DATA: Payload = MappingProxyType({})
"""El payload de los eventos que no llevan ninguno, que son la mayoría.

Un `default_factory=dict` alojaría un diccionario vacío nuevo por evento.
Compartir uno inmutable cuesta cero y no se puede mutar por accidente."""

_CONTENT_DIGEST_SIZE = 16
"""Bytes del digest de una entrada del exterior. Más que los 8 de `rng.py`:
ahí basta con dispersión para sembrar un generador, aquí el digest tiene que
resistir que alguien intente colar contenido distinto sin que se note al
comparar solo el resumen, no solo variar entre claves."""


def digest_of_content(content: str) -> str:
    """Huella estable del contenido exacto de una entrada del exterior.

    Existe para que una repetición pueda detectar que divergió sin comparar
    el texto entero: dos digests iguales certifican el mismo contenido, byte
    a byte, sin diferenciar megabytes de texto a mano. `blake2b` y no
    `hash()` por la misma razón que en `rng.py`: `hash()` varía entre
    procesos salvo que se fije `PYTHONHASHSEED`, y esto tiene que comparar
    igual entre procesos distintos.
    """
    return hashlib.blake2b(content.encode("utf-8"), digest_size=_CONTENT_DIGEST_SIZE).hexdigest()


class EventKind(StrEnum):
    """Vocabulario de eventos. Abstracto: ninguno nombra dominio.

    Conjunto cerrado que arranca mínimo y crece con sus consumidores. Añadir
    un valor en una unidad posterior es correcto y esperado; declarar hoy el
    de una unidad futura "por si acaso" no lo es.
    """

    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    ACTION_ABORTED = "action_aborted"
    """La acción dejó de ser válida a mitad. Sin esto, depurar algo atascado
    es adivinar."""

    ACTION_PROPOSED = "action_proposed"
    """El asistente quiso empezar algo que su techo de autoridad no le deja
    empezar solo. No es un fallo: es la mitad asíncrona de la confirmación."""

    ACTION_AUTHORIZED = "action_authorized"
    """Alguien con autoridad dijo que sí. Evento propio y no una nota dentro
    de `ACTION_STARTED`: el permiso y la ejecución ocurren en instantes
    distintos, y auditar quién autorizó qué exige que consten por separado."""

    ACTION_DENIED = "action_denied"
    ACTION_EXPIRED = "action_expired"
    """Nadie contestó a tiempo. Se anota como hecho: el silencio también es
    una respuesta, y tiene que poder rastrearse."""

    NEED_CRITICAL = "need_critical"
    """Un eje cruzó su umbral hacia abajo (`TuningSpec.critical_need_threshold`)."""

    NEED_ESCALATED = "need_escalated"
    """Un eje lleva `NeedSpec.escalates_after` ticks seguidos hundido.

    Convierte "esto va mal" en "hay que avisar ahora", sin que el motor sepa
    ni le importe qué significa avisar — eso lo decide quien consuma el
    evento, nunca esta pieza."""

    GOAL_ACHIEVED = "goal_achieved"
    GOAL_FAILED = "goal_failed"

    GOAL_UNLOCKED = "goal_unlocked"
    """Se abrió el siguiente eslabón: la meta que lo precedía se logró y esta
    pasa de `LOCKED` a `ACTIVE`. Evento propio y no una nota dentro de
    `GOAL_ACHIEVED` porque son dos cosas distintas: una es cerrar algo, la
    otra es que se abra un camino."""

    SITUATION_BEGAN = "situation_began"
    """Algo empezó a estar activo. Solo lo emiten las eventuales: una
    periódica es una función del reloj, y anunciarla sería anunciar que un
    ciclo volvió a empezar."""

    SITUATION_ENDED = "situation_ended"

    INPUT_OBSERVED = "input_observed"
    """El journal: una entrada procedente del exterior se registró con su
    contenido exacto. Es lo que hace verificable el determinismo — ver
    `Event.content` y `Event.digest`."""

    INTENT_CLASSIFIED = "intent_classified"
    """Se intentó reconocer una entrada del buzón contra el catálogo del
    pack. `data={"intent": ...}` si se reconoció; sin `data` si quedó
    desconocida — el silencio en el payload es la señal, igual que
    `ACTION_EXPIRED` anota el silencio de quien no contesta a una
    propuesta."""

    CATALOG_MATCHED = "catalog_matched"
    """Un texto se cotejó contra un catálogo cerrado externo al pack —el
    de intenciones tiene su propio tipo, `INTENT_CLASSIFIED`, porque
    mezclar los dos audits impediría distinguirlos después. `data={"codes":
    "..."}` con lo encontrado, separado por comas; sin `data` si nada
    alcanzó el umbral."""

    REPLY_GIVEN = "reply_given"
    """La voz produjo una respuesta para quien habla. `data={"mode":
    "operativo"|"libre"}` — qué clase de turno fue, nunca el texto: el
    contenido exacto de una respuesta es trabajo de la memoria (U10), no
    del journal de auditoría mínima. Solo lo emite `Session` cuando la
    charla libre está activada (`free_chat_enabled=True`) — con la
    bandera apagada, el comportamiento de U7 no cambia ni un evento."""


@dataclass(frozen=True, slots=True)
class Event:
    """Un hecho ocurrido. Inmutable: el pasado no se reescribe.

    **No es un modelo de pydantic, y la distinción es de fondo:** pydantic
    existe para validar lo que escribe un humano —el pack, que puede estar
    mal—. Un evento lo construye el motor con datos que él mismo acaba de
    calcular. Validarlo es pagar por comprobar lo que ya se sabe.

    Medido en Agora sobre doscientos mil eventos de cada uno:

        pydantic (frozen)      1176 B/evento    6.84 us/emisión
        dataclass + slots       136 B/evento    3.83 us/emisión

    Un `BaseModel` congelado arrastra `__dict__`, `__pydantic_extra__`,
    `__pydantic_private__` y `__pydantic_fields_set__` —este último un
    *set*, que solo él ya pesa más que el evento entero—. Con `slots` no hay
    ninguno de los cuatro.
    """

    seq: int
    tick: int
    kind: EventKind
    owner: AgentId
    action: ActionId | None = None
    data: Payload = NO_DATA

    content: str | None = None
    """El contenido exacto de una entrada del exterior, sin resumir ni
    normalizar. Solo lo llevan los eventos `INPUT_OBSERVED`; el resto lo deja
    en `None`.

    Va en un campo propio y no dentro de `data`: `data` es un mapa de
    escalares cortos pensado para un briefing (U7), y una entrada de fuera
    puede ser texto arbitrario y grande. Mezclarlos arriesgaría que un
    consumidor de `data` trunque o normalice sin querer el único registro
    que tiene que sobrevivir intacto.
    """

    digest: str | None = None
    """`digest_of_content(content)` para todo evento que entra por `emit`,
    la única puerta que lo garantiza: es quien calcula el digest y nunca
    deja que quien llama lo componga por su cuenta (ver el docstring de
    `emit`). Un `Event` construido a mano, sin pasar por `emit`, puede
    llevar un `content` y un `digest` que no se correspondan — el
    `dataclass` no lo impide, y añadir esa validación aquí lo convertiría en
    un modelo de pydantic por la puerta de atrás, exactamente lo que la
    medición de la cabecera de esta clase justifica no hacer.

    Se guarda en vez de recalcularse en cada lectura porque comparar dos
    digests tiene que poder hacerse sin volver a tener el contenido a mano
    — por ejemplo, para confirmar que un journal no se corrompió sin
    transportar el texto entero de vuelta.
    """

    def __deepcopy__(self, memo: dict[int, Any]) -> Event:
        """Se devuelve a sí mismo: es la consecuencia de ser inmutable.
        Copiar un hecho pasado no puede producir nada distinto."""
        return self

    def __copy__(self) -> Event:
        return self

    def to_json(self) -> str:
        """Una línea JSONL. Se omite lo que no lleva valor, incluido el
        payload vacío: en la mayoría de los eventos `"data":{}` sería ruido
        puro.

        El payload se vuelca con sus claves ORDENADAS, y no es cosmético:
        `dict(self.data)` conservaría el orden de inserción de quien
        construyó el evento, así que dos eventos lógicamente idénticos
        construidos con las claves en distinto orden producirían dos líneas
        JSON distintas — exactamente el fallo que ya mordió una vez a
        `HealthSpec.may_go_unused` con un `frozenset`, aquí con un `dict`
        en vez de con el orden de hash de un `PYTHONHASHSEED` variable.

        `ensure_ascii=False` es deliberado —el log tiene que ser legible por
        un humano, y escapar todo no-ASCII lo convertiría en `\\uXXXX`
        ilegible—, pero `U+2028` y `U+2029` (separador de línea y de
        párrafo Unicode) son un caso aparte: no son `\\n`, así que JSON los
        deja crudos dentro de la cadena, pero sí cuentan como salto de línea
        para `str.splitlines()`. `content` almacena texto arbitrario que
        viene de fuera —la entrada menos confiable del sistema—, y un solo
        `U+2028` colado ahí bastaría para que "una línea por evento" sea
        cierto contando por `\\n` y falso contando por `splitlines()`, que
        es justo lo que usan los consumidores del journal. Se escapan a mano
        después de `json.dumps`, que no los toca por tratarse de caracteres
        válidos dentro de una cadena JSON.
        """
        fields: dict[str, Any] = {
            "seq": self.seq,
            "tick": self.tick,
            "kind": self.kind.value,
            "owner": self.owner,
        }
        if self.action is not None:
            fields["action"] = self.action
        if self.data:
            fields["data"] = {key: self.data[key] for key in sorted(self.data)}
        if self.content is not None:
            fields["content"] = self.content
        if self.digest is not None:
            fields["digest"] = self.digest
        line = json.dumps(fields, separators=(",", ":"), ensure_ascii=False)
        return line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


class EvictedEventError(LookupError):
    """Se pidió un evento que la ventana ya descartó.

    Revienta a propósito. Con ventana, indexar directamente sobre la lista
    interna devolvería un evento DISTINTO en silencio, y ese es exactamente
    el tipo de fallo que este proyecto no se puede permitir: un log que
    miente sin avisar.
    """


PURGE_BATCH_SIZE = 4096
"""Cada cuántos eventos se mira si hay que desalojar.

No se purga en cada emisión porque `del lista[:k]` mueve el resto: hacerlo
de uno en uno sería cuadrático. En lotes es amortizado O(1), a cambio de que
la ventana retenga un poco más de lo pedido, nunca menos."""


class EventLog:
    """Append-only, con un índice por dueño para la consulta que el motor
    hará de verdad: "¿qué le ha pasado a este dueño?". Es la base del
    aislamiento entre dueños y de la memoria episódica de U10. Construirlo al
    insertar cuesta nada; reconstruirlo después obliga a recorrer el log
    entero en cada consulta.

    **La ventana.** Con `window` en ticks, el log conserva solo lo reciente.
    No es solo un apaño de memoria: la memoria de U10 va a leer de aquí, y
    nada se recuerda indefinidamente. Se saca lo saliente de una ventana, no
    de un archivo perpetuo.

    Lo que la ventana NO toca: `seq` sigue siendo global y monótono, como
    promete su docstring. Nunca se reutiliza; simplemente deja de haber
    evento detrás, y pedirlo revienta.

    **El vertido.** Con `sink`, cada evento se escribe según se genera. Es lo
    que hace inocua a la ventana: la historia entera queda en el fichero y
    en memoria solo vive lo reciente. Al vertido va TODO lo que ocurre, se
    retenga o no — es lo que permite descartar en memoria sin perder el
    rastro de auditoría.

    Sin `window` ni `sink`, nada cambia: se guarda todo, como siempre.
    """

    __slots__ = (
        "_by_owner",
        "_events",
        "_evicted",
        "_since_purge_check",
        "_sink",
        "_window",
    )

    def __init__(self, window: int | None = None, sink: TextIO | None = None) -> None:
        self._events: list[Event] = []
        # Deque y no lista: el desalojo va en orden de `seq`, así que el que
        # se va siempre es la cabeza, y `popleft` en un deque es O(1).
        self._by_owner: dict[AgentId, deque[int]] = {}
        self._window = window
        self._sink = sink
        self._evicted = 0
        """Cuántos se descartaron. Es también el `seq` del primero que
        queda, y con eso se traduce un `seq` a su posición actual en la
        lista."""
        self._since_purge_check = 0

    def emit(
        self,
        *,
        tick: int,
        kind: EventKind,
        owner: AgentId,
        action: ActionId | None = None,
        data: Payload | None = None,
        content: str | None = None,
    ) -> Event:
        """Registra un hecho y devuelve el evento ya sellado con su `seq`.

        Solo por argumentos con nombre: una llamada con varios posicionales
        del mismo tipo (`tick`, `seq`...) es un error esperando a ocurrir.

        `content` es la puerta única del journal: si se pasa, `digest` se
        calcula aquí mismo con `digest_of_content` y nunca lo compone quien
        llama. Que el digest no sea un parámetro propio es intencionado —
        un llamante no puede construir un evento con contenido y digest que
        no se correspondan, porque no tiene forma de pasar el segundo por su
        cuenta.
        """
        event = Event(
            # `_evicted +` y no `len(self._events)` a secas: lo segundo
            # reiniciaría la numeración cada vez que la ventana descarta, y
            # `seq` es la clave con la que la memoria de U10 referenciará
            # recuerdos.
            seq=self._evicted + len(self._events),
            tick=tick,
            kind=kind,
            owner=owner,
            action=action,
            data=data if data else NO_DATA,
            content=content,
            digest=digest_of_content(content) if content is not None else None,
        )
        self._events.append(event)
        # Antes de indexar y antes de purgar: al vertido va TODO lo que
        # ocurre, se retenga o no.
        if self._sink is not None:
            self._sink.write(event.to_json())
            self._sink.write("\n")

        self._by_owner.setdefault(event.owner, deque()).append(event.seq)

        self._since_purge_check += 1
        if self._window is not None and self._since_purge_check >= PURGE_BATCH_SIZE:
            self._evict(tick)

        return event

    def _evict(self, current_tick: int) -> None:
        """Descarta lo que quedó fuera de la ventana, en un solo corte.

        El índice guarda `seq`, no eventos: si no se poda también, la
        memoria sigue creciendo aunque los eventos se vayan. Y como el
        desalojo va en orden de `seq` y el índice los tiene ascendentes, el
        que se va siempre es la cabeza — de ahí el deque.
        """
        assert self._window is not None
        self._since_purge_check = 0
        limit = current_tick - self._window

        count = 0
        for event in self._events:
            if event.tick >= limit:
                break
            count += 1
        if count == 0:
            return

        for event in self._events[:count]:
            self._by_owner[event.owner].popleft()

        del self._events[:count]
        self._evicted += count

    def __len__(self) -> int:
        """Los RETENIDOS, que son los que se pueden recorrer.

        Con ventana esto deja de ser lo mismo que `emitted`, y se elige que
        `len` case con la iteración: un `len` que no coincide con lo que
        sale al iterar es peor mentira que un número más pequeño del
        esperado.
        """
        return len(self._events)

    @property
    def emitted(self) -> int:
        """Cuántos ocurrieron en total, se conserven o no."""
        return self._evicted + len(self._events)

    @property
    def evicted(self) -> int:
        return self._evicted

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __getitem__(self, seq: int) -> Event:
        position = seq - self._evicted
        if position < 0 or position >= len(self._events):
            raise EvictedEventError(
                f"el evento {seq} ya no está: la ventana conserva "
                f"de {self._evicted} a {self.emitted - 1}"
            )
        return self._events[position]

    def by_owner(self, owner: AgentId) -> list[Event]:
        """Todo lo que le ha pasado a este dueño. Base de la memoria
        episódica (U10) y del aislamiento entre dueños."""
        return [self[seq] for seq in self._by_owner.get(owner, ())]

    def since(self, tick: int) -> list[Event]:
        """Eventos desde `tick` inclusive. El log está ordenado por tick,
        así que basta recorrer desde el final hacia atrás."""
        cut = len(self._events)
        while cut > 0 and self._events[cut - 1].tick >= tick:
            cut -= 1
        return self._events[cut:]

    def of_kind(self, *kinds: EventKind) -> list[Event]:
        wanted = frozenset(kinds)
        return [event for event in self._events if event.kind in wanted]

    def to_jsonl(self) -> Iterable[str]:
        """Serialización estable de lo RETENIDO, para diffear dos corridas.

        El criterio de determinismo (G6) compara exactamente esto: mismo
        estado y mismas entradas, mismo JSONL, byte a byte. Con ventana solo
        cubre el tramo conservado; para una corrida larga, lo que hay que
        diffear es el fichero del `sink`, que sí lleva todo.
        """
        return (event.to_json() for event in self._events)
