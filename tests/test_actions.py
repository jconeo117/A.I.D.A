from __future__ import annotations

from collections.abc import Mapping

import pytest

from aida.engine.actions import ActionGate, ActionRequest, ToolResult, ToolStatus
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import ActionId, AgentId
from aida.engine.ports import ToolPort


class EspiaTool(ToolPort):
    def __init__(
        self,
        resultado: ToolResult | None = None,
        excepcion: Exception | None = None,
        log: EventLog | None = None,
    ) -> None:
        self.llamadas: list[tuple[ActionId, Mapping[str, str]]] = []
        self.resultado = resultado or ToolResult(ToolStatus.DONE, data={"appointment": "1"})
        self.excepcion = excepcion
        self.log = log
        self.eventos_previos_al_ejecutar = 0

    def execute(self, *, action: ActionId, arguments: Mapping[str, str]) -> ToolResult:
        if self.log is not None:
            self.eventos_previos_al_ejecutar = len(list(self.log))
        self.llamadas.append((action, arguments))
        if self.excepcion is not None:
            raise self.excepcion
        return self.resultado


def _crear_puerta(
    tool: ToolPort, acciones: frozenset[ActionId] | None = None
) -> tuple[ActionGate, EventLog]:
    log = EventLog()
    if isinstance(tool, EspiaTool):
        tool.log = log
    reloj = [0]

    def tick() -> int:
        reloj[0] += 1
        return reloj[0]

    acts = (
        acciones
        if acciones is not None
        else frozenset({ActionId("agendar"), ActionId("cancelar")})
    )
    puerta = ActionGate(
        actions=acts,
        tool=tool,
        log=log,
        owner=AgentId("agenda"),
        tick=tick,
    )
    return puerta, log


def test_submit_emite_requested_y_completed_en_orden() -> None:
    espia = EspiaTool(resultado=ToolResult(ToolStatus.DONE, data={"appointment": "42"}))
    puerta, log = _crear_puerta(espia)
    req = ActionRequest(
        action=ActionId("agendar"),
        arguments={"phone": "3001234567", "name": "Ana", "service": "consulta"},
        requested_by="prof-1",
    )
    res = puerta.submit(req)
    assert res.status == ToolStatus.DONE
    assert len(espia.llamadas) == 1
    assert espia.eventos_previos_al_ejecutar == 1
    eventos = list(log)
    assert len(eventos) == 2
    e_req, e_close = eventos[0], eventos[1]
    assert e_req.kind == EventKind.ACTION_REQUESTED
    assert e_req.action == ActionId("agendar")
    assert e_req.data == {
        "by": "prof-1",
        "name": "Ana",
        "phone": "3001234567",
        "service": "consulta",
    }
    assert e_close.kind == EventKind.ACTION_COMPLETED
    assert e_close.action == ActionId("agendar")
    assert e_close.data == {"reason": "", "appointment": "42"}


def test_submit_accion_desconocida_emite_rejected_sin_llamar_herramienta() -> None:
    espia = EspiaTool()
    puerta, log = _crear_puerta(espia, acciones=frozenset({ActionId("agendar")}))
    req = ActionRequest(
        action=ActionId("accion_inexistente"),
        arguments={"foo": "bar"},
        requested_by="prof-1",
    )
    res = puerta.submit(req)
    assert res.status == ToolStatus.REJECTED
    assert res.reason == "unknown_action"
    assert len(espia.llamadas) == 0
    eventos = list(log)
    assert len(eventos) == 2
    assert eventos[0].kind == EventKind.ACTION_REQUESTED
    assert eventos[1].kind == EventKind.ACTION_REJECTED
    assert eventos[1].data["reason"] == "unknown_action"


def test_submit_herramienta_lanza_excepcion_emite_failed_y_sigue() -> None:
    espia = EspiaTool(excepcion=RuntimeError("fallo de disco simulado"))
    puerta, log = _crear_puerta(espia)
    req = ActionRequest(
        action=ActionId("agendar"),
        arguments={"name": "Carlos"},
        requested_by="prof-1",
    )
    res = puerta.submit(req)
    assert res.status == ToolStatus.FAILED
    assert res.reason == "RuntimeError"
    eventos = list(log)
    assert len(eventos) == 2
    assert eventos[0].kind == EventKind.ACTION_REQUESTED
    assert eventos[1].kind == EventKind.ACTION_FAILED
    assert eventos[1].data["reason"] == "RuntimeError"

    espia.excepcion = None
    espia.resultado = ToolResult(ToolStatus.DONE)
    res2 = puerta.submit(req)
    assert res2.status == ToolStatus.DONE


def test_submit_sin_requested_by_lanza_value_error_y_ningun_evento() -> None:
    espia = EspiaTool()
    puerta, log = _crear_puerta(espia)
    req = ActionRequest(
        action=ActionId("agendar"),
        arguments={"name": "Sin Autor"},
        requested_by="",
    )
    with pytest.raises(ValueError, match="sin autor"):
        puerta.submit(req)
    assert len(espia.llamadas) == 0
    assert len(list(log)) == 0


def test_submit_argumentos_ordenados_alfabeticamente() -> None:
    espia = EspiaTool()
    puerta, log = _crear_puerta(espia)
    req = ActionRequest(
        action=ActionId("agendar"),
        arguments={"z_arg": "ultimo", "a_arg": "primero", "m_arg": "medio"},
        requested_by="prof-1",
    )
    puerta.submit(req)
    args_recibidos = espia.llamadas[0][1]
    assert list(args_recibidos.keys()) == ["a_arg", "m_arg", "z_arg"]
    e_req = next(iter(log))
    claves_evento = [k for k in e_req.data if k != "by"]
    assert claves_evento == ["a_arg", "m_arg", "z_arg"]
