"""La herramienta de la agenda: las reglas de cada acción, y los huecos y el resumen de semana
que pinta la pantalla (101b).

La herramienta no registra nada ni audita: eso es de la puerta de acciones y del brazo. Devuelve
un `ToolResult` con estado y no lanza por una regla del negocio. Cuál es la primera regla que
falla importa, porque es lo que se le explica a quien pidió: por eso el orden va fijado aquí.

Quien pide llega en `arguments["by"]`, que pone el brazo a partir de la identidad del armazón.
El contacto no lo elige: el brazo lo sobrescribe siempre.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timedelta

from aida.arms.agenda.setup import DIAS, AgendaSetup, ResourceSpec
from aida.arms.agenda.store import AgendaStore, Appointment, AppointmentState
from aida.engine.actions import ToolResult, ToolStatus

ACCIONES = frozenset({"agendar", "editar", "reprogramar", "cancelar", "atendida", "no_asistio"})


def _hecho(cita: int) -> ToolResult:
    return ToolResult(ToolStatus.DONE, data={"appointment": str(cita)})


def _rechazo(razon: str) -> ToolResult:
    return ToolResult(ToolStatus.REJECTED, razon)


def _fecha(texto: str | None) -> datetime | None:
    """Una fecha con zona. Sin zona no se sabe qué hora es, y no se agenda."""
    if not texto:
        return None
    try:
        valor = datetime.fromisoformat(texto)
    except ValueError:
        return None
    return valor if valor.tzinfo is not None else None


def _tramos_cubren(recurso: ResourceSpec, inicio: datetime, minutos: int) -> bool:
    """La cita entera cae dentro de **un** tramo del recurso para ese día (A.3)."""
    fin = inicio + timedelta(minutes=minutos)
    for ta, tb in recurso.hours.get(DIAS[inicio.weekday()], ()):
        desde = datetime.combine(inicio.date(), ta, tzinfo=inicio.tzinfo)
        hasta = datetime.combine(inicio.date(), tb, tzinfo=inicio.tzinfo)
        if desde <= inicio and fin <= hasta:
            return True
    return False


def _solapa(inicio: datetime, minutos: int, otra: Appointment) -> bool:
    """Dos citas solapan si cada una empieza antes de que acabe la otra: un hueco que acaba
    justo donde empieza otro no solapa."""
    fin = inicio + timedelta(minutes=minutos)
    return inicio < otra.start + timedelta(minutes=otra.minutes) and otra.start < fin


def motivo_hueco(
    store: AgendaStore,
    recurso: ResourceSpec,
    inicio: datetime,
    minutos: int,
    *,
    excepto: int | None,
) -> str | None:
    """La primera regla de horario o de solape que falla para un hueco. Es la única puerta
    de esta comprobación: la usan la herramienta y la pantalla de huecos, y con dos copias
    podrían decir cosas distintas."""
    if not _tramos_cubren(recurso, inicio, minutos):
        return "outside_hours"
    vecinas = store.between(inicio - timedelta(days=1), inicio + timedelta(days=1))
    for otra in vecinas:
        if otra.id == excepto or otra.resource != recurso.id:
            continue
        if otra.state == AppointmentState.CANCELLED:
            continue
        if _solapa(inicio, minutos, otra):
            return "overlap"
    return None


class AgendaTool:
    """Implementa `ToolPort`: la puerta de acciones la llama con la acción y sus argumentos."""

    def __init__(
        self,
        store: AgendaStore,
        setup: AgendaSetup,
        now: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._setup = setup
        self._now = now

    def execute(self, *, action: str, arguments: Mapping[str, str]) -> ToolResult:
        if action == "agendar":
            return self._agendar(arguments)
        if action == "editar":
            return self._editar(arguments)
        if action == "reprogramar":
            return self._reprogramar(arguments)
        if action == "cancelar":
            return self._cerrar(arguments, AppointmentState.CANCELLED, con_hora=False)
        if action == "atendida":
            return self._cerrar(arguments, AppointmentState.ATTENDED, con_hora=True)
        if action == "no_asistio":
            return self._cerrar(arguments, AppointmentState.NO_SHOW, con_hora=True)
        return _rechazo("unknown_action")

    def _agendar(self, args: Mapping[str, str]) -> ToolResult:
        nombre = args.get("name", "").strip()
        telefono = args.get("phone", "").strip()
        if not nombre or not telefono:
            return _rechazo("missing_client")
        servicio = self._setup.service(args.get("service", ""))
        if servicio is None:
            return _rechazo("unknown_service")
        inicio = _fecha(args.get("start"))
        if inicio is None:
            return _rechazo("bad_argument")
        pedido = args.get("resource", "")
        if pedido and self._setup.resource(pedido) is None:
            return _rechazo("unknown_resource")
        candidatos = [
            r
            for r in self._setup.resources
            if (not pedido or r.id == pedido) and servicio.id in r.services
        ]
        if not candidatos:
            return _rechazo("service_not_offered")
        if inicio < self._now():
            return _rechazo("past")
        primer_motivo: str | None = None
        for recurso in candidatos:
            motivo = motivo_hueco(self._store, recurso, inicio, servicio.minutes, excepto=None)
            if motivo is None:
                return self._crear(
                    args["by"], nombre, telefono, servicio.id, recurso.id, inicio, servicio.minutes
                )
            primer_motivo = primer_motivo or motivo
        return _rechazo(primer_motivo or "outside_hours")

    def _crear(
        self,
        por: str,
        nombre: str,
        telefono: str,
        servicio: str,
        recurso: str,
        inicio: datetime,
        minutos: int,
    ) -> ToolResult:
        contacto = self._store.client_by_phone(telefono)
        if contacto is None:
            contacto = self._store.add_client(name=nombre, phone=telefono)
        elif contacto.name != nombre:
            self._store.update_client(contacto.id, name=nombre, phone=telefono)
        cita = self._store.add(
            client=contacto.id,
            service=servicio,
            resource=recurso,
            start=inicio,
            minutes=minutos,
            by=por,
        )
        return _hecho(cita.id)

    def _editar(self, args: Mapping[str, str]) -> ToolResult:
        cita = self._cita(args)
        if isinstance(cita, ToolResult):
            return cita
        nombre = args.get("name")
        telefono = args.get("phone")
        if (nombre is not None and not nombre.strip()) or (
            telefono is not None and not telefono.strip()
        ):
            return _rechazo("missing_client")
        servicio, minutos = cita.service, cita.minutes
        if args.get("service") and args["service"] != cita.service:
            nuevo = self._setup.service(args["service"])
            if nuevo is None:
                return _rechazo("unknown_service")
            servicio, minutos = nuevo.id, nuevo.minutes
        recurso_id = cita.resource
        if args.get("resource") and args["resource"] != cita.resource:
            if self._setup.resource(args["resource"]) is None:
                return _rechazo("unknown_resource")
            recurso_id = args["resource"]
        recurso = self._setup.resource(recurso_id)
        if recurso is None or servicio not in recurso.services:
            return _rechazo("service_not_offered")
        if recurso_id != cita.resource or minutos != cita.minutes:
            motivo = motivo_hueco(self._store, recurso, cita.start, minutos, excepto=cita.id)
            if motivo is not None:
                return _rechazo(motivo)
        if nombre is not None or telefono is not None:
            actual = self._store.client(cita.client)
            self._store.update_client(
                cita.client,
                name=nombre if nombre is not None else actual.name,
                phone=telefono if telefono is not None else actual.phone,
            )
        self._store.update(
            cita.id,
            by=args["by"],
            service=servicio,
            resource=recurso_id,
            minutes=minutos,
        )
        return _hecho(cita.id)

    def _reprogramar(self, args: Mapping[str, str]) -> ToolResult:
        cita = self._cita(args)
        if isinstance(cita, ToolResult):
            return cita
        inicio = _fecha(args.get("start"))
        if inicio is None:
            return _rechazo("bad_argument")
        if inicio < self._now():
            return _rechazo("past")
        pedido = args.get("resource", "")
        if pedido and self._setup.resource(pedido) is None:
            return _rechazo("unknown_resource")
        recurso = self._setup.resource(pedido or cita.resource)
        if recurso is None or cita.service not in recurso.services:
            return _rechazo("service_not_offered")
        motivo = motivo_hueco(self._store, recurso, inicio, cita.minutes, excepto=cita.id)
        if motivo is not None:
            return _rechazo(motivo)
        self._store.update(
            cita.id,
            by=args["by"],
            start=inicio,
            resource=recurso.id if pedido else None,
        )
        return _hecho(cita.id)

    def _cerrar(
        self, args: Mapping[str, str], estado: AppointmentState, *, con_hora: bool
    ) -> ToolResult:
        cita = self._cita(args)
        if isinstance(cita, ToolResult):
            return cita
        if con_hora and cita.start > self._now():
            return _rechazo("not_yet")
        self._store.update(cita.id, by=args["by"], state=estado)
        return _hecho(cita.id)

    def _cita(self, args: Mapping[str, str]) -> Appointment | ToolResult:
        """La cita del argumento, si existe y sigue agendada. Las acciones que la cambian
        empiezan igual."""
        try:
            identificador = int(args.get("appointment", ""))
        except ValueError:
            return _rechazo("unknown_appointment")
        cita = self._store.get(identificador)
        if cita is None:
            return _rechazo("unknown_appointment")
        if cita.state != AppointmentState.BOOKED:
            return _rechazo("not_booked")
        return cita


def huecos_libres(
    store: AgendaStore,
    setup: AgendaSetup,
    dia: date,
    servicio: str,
    ahora: datetime,
) -> tuple[tuple[str, datetime], ...]:
    """Los `(recurso, inicio)` en los que `agendar` daría `DONE` para ese servicio, ordenados
    por inicio y, a igualdad, por recurso."""
    duracion = setup.service(servicio)
    if duracion is None:
        return ()
    huecos: list[tuple[str, datetime]] = []
    for recurso in setup.resources:
        if servicio not in recurso.services:
            continue
        for ta, tb in recurso.hours.get(DIAS[dia.weekday()], ()):
            cursor = datetime.combine(dia, ta, tzinfo=ahora.tzinfo)
            fin_tramo = datetime.combine(dia, tb, tzinfo=ahora.tzinfo)
            while cursor + timedelta(minutes=duracion.minutes) <= fin_tramo:
                if cursor >= ahora and (
                    motivo_hueco(store, recurso, cursor, duracion.minutes, excepto=None) is None
                ):
                    huecos.append((recurso.id, cursor))
                cursor += timedelta(minutes=setup.slot_minutes)
    return tuple(sorted(huecos, key=lambda h: (h[1], h[0])))


def resumen_semana(
    store: AgendaStore, setup: AgendaSetup, lunes: date, ahora: datetime
) -> tuple[tuple[date, int, int], ...]:
    """`(día, ocupados, capacidad)` de lunes a sábado, en huecos de `slot_minutes`. `ahora`
    solo da la zona horaria: los días se leen en la del consultorio."""
    slot = setup.slot_minutes
    filas: list[tuple[date, int, int]] = []
    for desplazamiento in range(6):
        dia = lunes + timedelta(days=desplazamiento)
        capacidad = 0
        for recurso in setup.resources:
            for ta, tb in recurso.hours.get(DIAS[dia.weekday()], ()):
                capacidad += (tb.hour * 60 + tb.minute - ta.hour * 60 - ta.minute) // slot
        inicio_dia = datetime.combine(dia, time(0, 0), tzinfo=ahora.tzinfo)
        ocupados = 0
        for cita in store.between(inicio_dia, inicio_dia + timedelta(days=1)):
            if cita.state != AppointmentState.CANCELLED:
                ocupados += -(-cita.minutes // slot)
        filas.append((dia, ocupados, capacidad))
    return tuple(filas)


free_slots = huecos_libres
week_summary = resumen_semana
