"""La puerta de acciones: la única forma de que una petición cambie algo fuera del motor.

Por qué existe: la agenda (y, después, WhatsApp y el asistente) cambian datos reales. Si cada
brazo los cambiara por su cuenta, no habría un solo sitio donde quedara escrito quién pidió
qué, ni un evento de cierre que dijera cómo acabó. Aquí se registra la petición antes de
ejecutarla y se emite exactamente un evento de cierre, siempre.

El motor no sabe qué acciones existen: el catálogo lo declara quien monta el brazo. Esta
puerta no importa `arms/` ni `platform/`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import ActionId, AgentId
from aida.engine.ports import ToolPort


class ToolStatus(StrEnum):
    DONE = "done"
    REJECTED = "rejected"
    """Una regla del negocio dice que no (hueco ocupado, fuera de horario). Se explica a
    quien pidió; no es un fallo."""
    FAILED = "failed"
    """Algo ajeno a la regla (disco, red, una excepción). Se reintenta o se avisa."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    status: ToolStatus
    reason: str = ""
    """Código estable del rechazo o del fallo (`overlap`, `outside_hours`…): el texto lo pone
    el pack, no la herramienta."""
    data: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionRequest:
    action: ActionId
    arguments: Mapping[str, str]
    requested_by: str
    """El `id` del perfil identificado. Una petición sin autor no entra (`I-156`)."""


class ActionGate:
    def __init__(
        self,
        *,
        actions: frozenset[ActionId],
        tool: ToolPort,
        log: EventLog,
        owner: AgentId,
        tick: Callable[[], int],
    ) -> None:
        self._actions = actions
        self._tool = tool
        self._log = log
        self._owner = owner
        self._tick = tick

    def submit(self, request: ActionRequest) -> ToolResult:
        """Deja la petición en el registro, la ejecuta y emite el cierre. Es lo único que la
        herramienta ve: la acción, sus argumentos ya cerrados, y nunca quién pidió."""
        if not request.requested_by:
            raise ValueError("una petición sin autor no se registra (I-156)")
        argumentos = dict(sorted(request.arguments.items()))
        self._log.emit(
            tick=self._tick(),
            kind=EventKind.ACTION_REQUESTED,
            owner=self._owner,
            action=request.action,
            data={"by": request.requested_by, **argumentos},
        )
        if request.action not in self._actions:
            resultado = ToolResult(ToolStatus.REJECTED, "unknown_action")
        else:
            try:
                resultado = self._tool.execute(action=request.action, arguments=argumentos)
            except Exception as exc:  # frontera: la herramienta puede fallar por causas ajenas
                resultado = ToolResult(ToolStatus.FAILED, type(exc).__name__)
        clase = {
            ToolStatus.DONE: EventKind.ACTION_COMPLETED,
            ToolStatus.REJECTED: EventKind.ACTION_REJECTED,
            ToolStatus.FAILED: EventKind.ACTION_FAILED,
        }[resultado.status]
        self._log.emit(
            tick=self._tick(),
            kind=clase,
            owner=self._owner,
            action=request.action,
            data={"reason": resultado.reason, **dict(resultado.data)},
        )
        return resultado
