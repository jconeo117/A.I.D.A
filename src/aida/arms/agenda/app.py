"""La pantalla de la agenda: el segundo brazo de la plataforma (101b).

Cada cambio entra por `POST accion`, que pasa por la puerta de acciones del motor. Aquí no hay
ninguna ruta por botón que pueda saltarse la puerta (decisión 8). La pantalla lee el día con
`GET dia/<fecha>` y se pinta en el navegador: nada de lo que se muestra sale del servidor sin
pasar por quien está identificado.
"""

from __future__ import annotations

import html
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from aida.arms.agenda.script import AGENDA_JS
from aida.arms.agenda.setup import WEEKDAYS, AgendaSetup, load_setup
from aida.arms.agenda.store import AgendaStore, AppointmentState, SqliteAgendaStore
from aida.arms.agenda.style import AGENDA_CSS
from aida.arms.agenda.tool import (
    ACTIONS,
    AgendaTool,
    find_clients,
    free_slots,
    week_summary,
)
from aida.engine.actions import ActionGate, ActionRequest, ToolStatus
from aida.engine.events import EventLog
from aida.engine.ontology import AgentId
from aida.keys import cache_key
from aida.platform.identity import Professional
from aida.platform.license import LicenseState
from aida.platform.web.pages import JS_EXIT, page_frame, user_chip
from aida.platform.web.shell import Response, ShellContext, json_response
from aida.resources import data_root, user_dir

_UI = data_root() / "packs" / "agenda" / "ui.yaml"
_SUBJECTS = ("appointment", "block", "client")
"""Lo que una acción aplicada devuelve como sujeto. La acción dice de qué tipo es."""


@dataclass(frozen=True, slots=True)
class AgendaWording:
    texts: Mapping[str, str]

    def __getitem__(self, clave: str) -> str:
        return self.texts[clave]

    def reason(self, razon: str) -> str:
        return self.texts.get(f"reason_{razon}", "")


def load_agenda_wording(path: Path | None = None) -> AgendaWording:
    raw = yaml.safe_load((path or _UI).read_text(encoding="utf-8"))
    seccion = raw.get("agenda") if isinstance(raw, dict) else None
    if not isinstance(seccion, dict) or not all(isinstance(v, str) for v in seccion.values()):
        raise ValueError(f"falta la sección 'agenda' de textos en {path or _UI}")
    return AgendaWording(texts=dict(seccion))


def _names(profesionales: tuple[Professional, ...]) -> dict[str, str]:
    return {p.id: p.name for p in profesionales}


class AgendaArm:
    """El brazo de la agenda, para el armazón (`ArmApp`)."""

    def __init__(
        self,
        *,
        arm_id: str,
        name: str,
        setup_path: Path | None,
        store: AgendaStore,
        now: Callable[[], datetime],
        wording: AgendaWording,
        lock: threading.Lock,
    ) -> None:
        self.id = arm_id
        self.name = name
        self._setup_path = setup_path
        self._store = store
        self._now = now
        self._wording = wording
        self._lock = lock
        self._sin_guardar = False
        self._mount()

    def _mount(self) -> None:
        """La herramienta y la puerta se montan con la configuración del momento. Al volver al
        lanzador se recargan: un horario cambiado en el archivo se nota sin reiniciar."""
        self._setup: AgendaSetup = load_setup(self._setup_path)
        self._herramienta = AgendaTool(self._store, self._setup, self._now)
        self._log = EventLog()
        self._reloj = 0
        self._puerta = ActionGate(
            actions=ACTIONS,
            tool=self._herramienta,
            log=self._log,
            owner=AgentId(self.id),
            tick=self._next,
        )

    def _next(self) -> int:
        self._reloj += 1
        return self._reloj

    # Ciclo de vida: lo que el armazón necesita de este brazo.

    def at_rest(self) -> bool:
        return not self._sin_guardar

    def in_settings(self) -> bool:
        return False

    def reset(self) -> None:
        self._sin_guardar = False
        self._mount()

    def rebuild(self) -> None:
        self.reset()

    def on_identified(self, actor: Professional) -> None:
        return None

    def on_unlocked(self) -> None:
        return None

    def settings_cards(self, ctx: ShellContext) -> str:
        """Una tarjeta en Ajustes: qué configuración usa y cuántos recursos tiene."""
        e = html.escape
        ruta = self._setup_path or user_dir("config") / "agenda.yaml"
        texto = self._wording.texts["settings_card_info"].format(
            self._setup.sector, str(ruta), len(self._setup.resources)
        )
        return (
            '<div class="setting"><div class="setting-head"><svg><use href="#i-calendar"/></svg>'
            f"<h3>{e(self._wording['title'])}</h3></div><p>{e(texto)}</p></div>"
        )

    def readonly_items(self, ctx: ShellContext) -> tuple[str, ...]:
        return (self._wording["readonly_view"],)

    # Páginas y rutas

    def page(self, ctx: ShellContext) -> str:
        e = html.escape
        w = self._wording
        cuerpo = f"""<section class="screen" id="s-agenda">
  <div class="stage">
    <div class="agenda-wrap">
      <div class="agenda-head">
        <h2 id="day-title"></h2>
        <div class="day-nav">
          <button type="button" id="prev">&lsaquo;</button>
          <button type="button" id="today">{e(w["today"])}</button>
          <button type="button" id="next">&rsaquo;</button>
        </div>
        <span class="grow"></span>
        <input class="input search" id="search" type="search"
          placeholder="{e(w["search_placeholder"])}">
        <button type="button" class="btn" id="block-btn">{e(w["block_button"])}</button>
        <button type="button" class="btn btn-primary" id="new-btn">
          <svg><use href="#i-plus"/></svg>{e(w["new_button"])}</button>
      </div>
      <div class="week" id="week"></div>
      <div class="agenda">
        <div class="grid" id="grid"></div>
        <aside class="side"><div class="panel-card" id="side"></div></aside>
      </div>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="exit">
      <svg><use href="#i-exit"/></svg>{e(ctx.wording["exit_label"])}</button>
  </footer>
</section>"""
        datos = {
            "token": ctx.token,
            "dia0": ctx.now.date().isoformat(),
            "w": dict(w.texts),
            "contacto": self._setup.client_label,
        }
        script = (
            f"const CONFIG = {json.dumps(datos, ensure_ascii=False)};\n"
            f"const TOKEN = {json.dumps(ctx.token)};\n{AGENDA_JS}{JS_EXIT}"
        )
        return page_frame(
            brand=ctx.brand,
            title=w["title"],
            place=ctx.wording["header_place"],
            moment=ctx.now,
            pill=ctx.wording["pill_ready"],
            pill_class="pill",
            body=cuerpo,
            script=script,
            head_link=user_chip(ctx.wording, ctx.professional),
            extra_css=AGENDA_CSS,
        )

    def get(self, path: str, ctx: ShellContext) -> Response | None:
        if path.startswith("contacto/"):
            try:
                contacto_id = int(path[len("contacto/") :])
            except ValueError:
                return None
            with self._lock:
                ficha = self._card(contacto_id, ctx)
            return json_response(200, ficha) if ficha is not None else None
        if not path.startswith("dia/"):
            return None
        try:
            dia = date.fromisoformat(path[len("dia/") :])
        except ValueError:
            return None
        with self._lock:
            return json_response(200, self._day(dia, ctx))

    def _day(self, dia: date, ctx: ShellContext) -> dict[str, Any]:
        ahora = self._now()
        nombres = _names(ctx.profiles)
        setup = self._setup
        store = self._store
        inicio = datetime.combine(dia, datetime.min.time(), tzinfo=ahora.tzinfo)
        citas = []
        for c in store.between(inicio, inicio + timedelta(days=1)):
            contacto = store.client(c.client)
            servicio = setup.service(c.service)
            citas.append(
                {
                    "id": c.id,
                    "client_id": c.client,
                    "client": contacto.name,
                    "phone": contacto.phone,
                    "service": c.service,
                    "service_label": servicio.label if servicio else c.service,
                    "resource": c.resource,
                    "start": c.start.isoformat(),
                    "minutes": c.minutes,
                    "state": c.state.value,
                    "created_by": nombres.get(c.created_by, c.created_by),
                    "updated_by": nombres.get(c.updated_by, c.updated_by),
                }
            )
        lunes = dia - timedelta(days=dia.weekday())
        return {
            "day": dia.isoformat(),
            "slot_minutes": setup.slot_minutes,
            "offset": ahora.isoformat()[-6:],
            "services": [
                {"id": s.id, "label": s.label, "minutes": s.minutes} for s in setup.services
            ],
            "resources": [
                {
                    "id": r.id,
                    "label": r.label,
                    "place": r.place,
                    "services": list(r.services),
                    "hours": [
                        [ta.strftime("%H:%M"), tb.strftime("%H:%M")]
                        for ta, tb in r.hours.get(WEEKDAYS[dia.weekday()], ())
                    ],
                }
                for r in setup.resources
            ],
            "appointments": citas,
            "blocks": [
                {
                    "id": b.id,
                    "resource": b.resource,
                    "start": b.start.isoformat(),
                    "end": b.end.isoformat(),
                    "reason": b.reason,
                    "created_by": nombres.get(b.created_by, b.created_by),
                }
                for b in store.blocks_between(inicio, inicio + timedelta(days=1))
            ],
            "week": [
                {"day": d.isoformat(), "booked": ocupados, "capacity": capacidad}
                for d, ocupados, capacidad in week_summary(store, setup, lunes, ahora)
            ],
            "free": {
                s.id: [
                    {"resource": r, "start": h.isoformat()}
                    for r, h in free_slots(store, setup, dia, s.id, ahora)
                ]
                for s in setup.services
            },
        }

    def _card(self, contacto_id: int, ctx: ShellContext) -> dict[str, Any] | None:
        """La ficha: las cifras se cuentan aquí y en ningún otro sitio."""
        try:
            contacto = self._store.client(contacto_id)
        except LookupError:
            return None
        ahora = self._now()
        nombres = _names(ctx.profiles)
        citas = self._store.appointments_for(contacto_id)
        return {
            "id": contacto.id,
            "name": contacto.name,
            "phone": contacto.phone,
            "upcoming": sum(
                1 for c in citas if c.state == AppointmentState.BOOKED and c.start >= ahora
            ),
            "attended": sum(1 for c in citas if c.state == AppointmentState.ATTENDED),
            "no_show": sum(1 for c in citas if c.state == AppointmentState.NO_SHOW),
            "appointments": [
                {
                    "id": c.id,
                    "service_label": (
                        s.label if (s := self._setup.service(c.service)) else c.service
                    ),
                    "resource": c.resource,
                    "start": c.start.isoformat(),
                    "minutes": c.minutes,
                    "state": c.state.value,
                    "created_by": nombres.get(c.created_by, c.created_by),
                }
                for c in reversed(citas)
            ],
        }

    def _search(self, body: bytes | None) -> Response:
        """Por `POST` y no en la dirección: lo que se busca es un nombre o un teléfono, y una
        dirección acaba en registros e historiales."""
        try:
            pedido = json.loads(body or b"{}")
        except ValueError:
            pedido = None
        if not isinstance(pedido, dict) or not isinstance(pedido.get("query"), str):
            return json_response(400, {"error": "cuerpo inválido"})
        with self._lock:
            hallados = find_clients(self._store, pedido["query"])
        return json_response(
            200, {"clients": [{"id": c.id, "name": c.name, "phone": c.phone} for c in hallados]}
        )

    def post(
        self,
        path: str,
        body: bytes | None,
        headers: Mapping[str, str],
        ctx: ShellContext,
    ) -> Response | None:
        if path == "actividad":
            return self._activity(body)
        if path == "accion":
            return self._action(body, ctx)
        if path == "buscar":
            return self._search(body)
        return None

    def _activity(self, body: bytes | None) -> Response:
        try:
            pedido = json.loads(body or b"{}")
        except ValueError:
            pedido = None
        if not isinstance(pedido, dict) or not isinstance(pedido.get("dirty"), bool):
            return json_response(400, {"error": "cuerpo inválido"})
        self._sin_guardar = pedido["dirty"]
        return json_response(200, {"ok": True})

    def _action(self, body: bytes | None, ctx: ShellContext) -> Response:
        """Una petición de cambio: se pasa por la puerta. Solo las citas aplicadas se auditan:
        un rechazo no cambió nada, y lo dice su respuesta."""
        if ctx.professional is None:
            return json_response(401, {"error": ""})
        try:
            pedido = json.loads(body or b"{}")
        except ValueError:
            pedido = None
        if not isinstance(pedido, dict) or not isinstance(pedido.get("action"), str):
            return json_response(400, {"error": "cuerpo inválido"})
        argumentos = pedido.get("arguments", {})
        if not isinstance(argumentos, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in argumentos.items()
        ):
            return json_response(400, {"error": "argumentos inválidos"})
        accion = pedido["action"]
        # Quien pide lo pone la plataforma: el contacto no elige su propio nombre.
        argumentos = {**argumentos, "by": ctx.professional.id}
        with self._lock:
            resultado = self._puerta.submit(
                ActionRequest(
                    action=accion,
                    arguments=argumentos,
                    requested_by=ctx.professional.id,
                )
            )
            if resultado.status == ToolStatus.DONE:
                sujeto = next((resultado.data[k] for k in _SUBJECTS if k in resultado.data), "")
                ctx.audit.record(
                    "agenda_changed",
                    actor=ctx.professional,
                    owner=sujeto,
                    data={"action": accion, "start": argumentos.get("start", "")},
                )
            conflictos: list[dict[str, str]] = []
            if resultado.reason == "has_appointments":
                for texto in resultado.data.get("appointments", "").split(","):
                    cita = self._store.get(int(texto))
                    if cita is not None:
                        conflictos.append(
                            {
                                "id": str(cita.id),
                                "client": self._store.client(cita.client).name,
                                "start": cita.start.isoformat(),
                            }
                        )
        return json_response(
            200,
            {
                "status": resultado.status.value,
                "reason": resultado.reason,
                "reason_text": self._wording.reason(resultado.reason),
                "data": dict(resultado.data),
                "conflicts": conflictos,
            },
        )


def build_arm(
    args: Any,
    license: LicenseState,
    manifest: Any,
    lock: threading.Lock,
) -> AgendaArm:
    """El punto de entrada del brazo (`packs/agenda/arm.yaml`): la configuración, la base cifrada
    con la clave del equipo, y el brazo listo para el armazón. Sin clave del equipo no hay
    agenda: los datos de las citas no se guardan en claro (101b)."""
    clave = cache_key()
    if clave is None:
        raise ValueError("sin clave del equipo no se puede abrir la agenda")
    ruta_base = user_dir("data") / "agenda.sqlite"
    store = SqliteAgendaStore(ruta_base, clave.encode())
    return AgendaArm(
        arm_id=manifest.id,
        name=manifest.name,
        setup_path=None,
        store=store,
        now=lambda: datetime.now().astimezone(),
        wording=load_agenda_wording(),
        lock=lock,
    )
