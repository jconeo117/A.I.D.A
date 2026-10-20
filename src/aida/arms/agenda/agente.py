"""El agente de mensajes: entiende lo que escribe una persona y decide qué hacer con la agenda
(101b, demo del agente de WhatsApp).

Por qué es el motor el que decide y no un modelo de lenguaje: cada respuesta del agente sale de
una regla determinista: una intención por frases del pack, un diálogo con pasos fijos, y una
acción que solo ocurre a través de la puerta de acciones. Un modelo podría redactar más bonito,
pero no podría agendar ni cancelar por su cuenta. Las respuestas, las frases y los días son datos
del pack: el código no sabe qué es una cita.

Lo que el agente no hace todavía: no habla con WhatsApp (eso es el canal, 102) ni guarda los
diálogos en curso al cerrar AIDA. Un diálogo a medias se pierde al reiniciar; el historial de
mensajes sí queda.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import yaml

from aida.arms.agenda.setup import AgendaSetup
from aida.arms.agenda.store import AgendaStore, Appointment, AppointmentState
from aida.arms.agenda.tool import huecos_libres
from aida.engine.actions import ToolResult, ToolStatus
from aida.resources import data_root
from aida.textmatch import cosine, fold, vectorize

_POR_DEFECTO = data_root() / "packs" / "agenda" / "agente.yaml"
_PALABRAS = re.compile(r"[a-z0-9:/]+")


@dataclass(frozen=True, slots=True)
class AgentePack:
    intents: Mapping[str, tuple[str, ...]]
    dias: Mapping[str, int]
    palabras_hoy: tuple[str, ...]
    palabras_manana: tuple[str, ...]
    umbral: float
    recordatorio_horas: int
    textos: Mapping[str, str]

    def t(self, clave: str, **datos: str) -> str:
        return self.textos[clave].format(**datos)


def load_agente_pack(path: Path | None = None) -> AgentePack:
    """Revienta si falta algo: un agente que no sabe decir una respuesta se nota con alguien
    escribiendo al otro lado."""
    raw = yaml.safe_load((path or _POR_DEFECTO).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path or _POR_DEFECTO} debe ser un mapeo YAML")
    intenciones = {
        str(nombre): tuple(fold(str(f)) for f in frases)
        for nombre, frases in dict(raw["intents"]).items()
    }
    dias = {fold(str(k)): int(v) for k, v in dict(raw["dias"]).items()}
    return AgentePack(
        intents=intenciones,
        dias=dias,
        palabras_hoy=tuple(fold(str(p)) for p in raw["palabras_hoy"]),
        palabras_manana=tuple(fold(str(p)) for p in raw["palabras_manana"]),
        umbral=float(raw["umbral"]),
        recordatorio_horas=int(raw["recordatorio_horas"]),
        textos={str(k): str(v) for k, v in dict(raw["textos"]).items()},
    )


@dataclass(frozen=True, slots=True)
class Respuesta:
    texto: str
    accion: str = ""
    """La acción que se ejecutó a través de la puerta, si alguna (`agendar`, `cancelar`…)."""


@dataclass
class _Dialogo:
    paso: str
    objetivo: str
    servicio: str | None = None
    dia: date | None = None
    hora: str | None = None
    recurso: str = ""
    nombre: str | None = None
    cita: int | None = None


Ejecutor = Callable[[str, Mapping[str, str]], ToolResult]


class Agente:
    def __init__(
        self,
        *,
        store: AgendaStore,
        setup: Callable[[], AgendaSetup],
        pack: AgentePack,
        ahora: Callable[[], datetime],
        motivos: Mapping[str, str],
    ) -> None:
        self._store = store
        self._setup = setup
        self._pack = pack
        self._ahora = ahora
        self._motivos = motivos
        self._dialogos: dict[str, _Dialogo] = {}
        self._vectores = {
            nombre: [vectorize(f) for f in frases] for nombre, frases in pack.intents.items()
        }

    # Lectura del texto: intención, servicio, día y hora.

    def _intencion(self, texto: str, solo: tuple[str, ...] = ()) -> str | None:
        vector = vectorize(fold(texto))
        mejor: str | None = None
        puntaje = 0.0
        for nombre in sorted(self._vectores):
            if solo and nombre not in solo:
                continue
            for frase in self._vectores[nombre]:
                valor = cosine(vector, frase)
                if valor > puntaje:
                    mejor, puntaje = nombre, valor
        return mejor if puntaje >= self._pack.umbral else None

    def _servicio(self, texto: str) -> str | None:
        t = f" {fold(texto)} "
        for servicio in self._setup().services:
            if f" {fold(servicio.label)} " in t or f" {servicio.id.replace('_', ' ')} " in t:
                return servicio.id
        return None

    def _dia(self, texto: str) -> date | None:
        hoy = self._ahora().date()
        palabras = set(_PALABRAS.findall(fold(texto)))
        if palabras & set(self._pack.palabras_hoy):
            return hoy
        if palabras & set(self._pack.palabras_manana):
            return hoy + timedelta(days=1)
        for nombre, dia_semana in self._pack.dias.items():
            if nombre in palabras:
                return hoy + timedelta(days=(dia_semana - hoy.weekday()) % 7)
        coincide = re.search(r"\b(\d{1,2})/(\d{1,2})\b", fold(texto))
        if coincide:
            try:
                dia = date(hoy.year, int(coincide.group(2)), int(coincide.group(1)))
            except ValueError:
                return None
            return dia if dia >= hoy else dia.replace(year=hoy.year + 1)
        return None

    def _hora(self, texto: str) -> str | None:
        coincide = re.search(r"\b(\d{1,2})(?::(\d{2}))?\b", fold(texto))
        if not coincide:
            return None
        horas, minutos = int(coincide.group(1)), int(coincide.group(2) or 0)
        if horas > 23 or minutos > 59:
            return None
        return f"{horas:02d}:{minutos:02d}"

    def _horas_libres(self, servicio: str, dia: date) -> list[tuple[str, datetime]]:
        return list(huecos_libres(self._store, self._setup(), dia, servicio, self._ahora()))

    def _nombre_servicio(self, servicio: str) -> str:
        return next((s.label for s in self._setup().services if s.id == servicio), servicio)

    def _lista_servicios(self) -> str:
        return ", ".join(s.label for s in self._setup().services)

    def _lista_horas(self, huecos: list[tuple[str, datetime]]) -> str:
        return ", ".join(sorted({h.strftime("%H:%M") for _r, h in huecos}))

    # Diálogo: cada respuesta es un paso fijo.

    def recibir(
        self,
        *,
        telefono: str,
        nombre: str,
        texto: str,
        ejecutar: Ejecutor,
    ) -> Respuesta:
        """Una mensaje entrante: se guarda, se responde y la respuesta también se guarda."""
        ahora = self._ahora()
        self._store.add_message(
            phone=telefono, direction="in", text=texto, at=ahora, kind="entrante"
        )
        respuesta = self._responder(telefono, nombre, texto, ejecutar)
        self._store.add_message(
            phone=telefono, direction="out", text=respuesta.texto, at=ahora, kind="respuesta"
        )
        return respuesta

    def _responder(self, telefono: str, nombre: str, texto: str, ejecutar: Ejecutor) -> Respuesta:
        dialogo = self._dialogos.get(telefono)
        if dialogo is not None:
            return self._paso(telefono, nombre, dialogo, texto, ejecutar)
        intencion = self._intencion(texto)
        if intencion == "ayuda":
            return Respuesta(self._pack.t("ayuda"))
        if intencion == "mis_citas":
            return Respuesta(self._mis_citas(telefono))
        if intencion == "disponibilidad":
            return self._ver_disponibilidad(telefono, texto)
        if intencion == "cancelar":
            return self._pedir_cancelar(telefono)
        if intencion == "agendar":
            dialogo = _Dialogo(paso="servicio", objetivo="agendar")
            dialogo.servicio = self._servicio(texto)
            dialogo.dia = self._dia(texto)
            self._dialogos[telefono] = dialogo
            return self._avanzar(telefono, nombre, dialogo)
        return Respuesta(self._pack.t("no_entiendo"))

    def _avanzar(self, telefono: str, nombre: str, dialogo: _Dialogo) -> Respuesta:
        """Pregunta lo que falta, en el orden fijo: servicio, día, hora, nombre, confirmación."""
        if dialogo.servicio is None:
            dialogo.paso = "servicio"
            return Respuesta(self._pack.t("pide_servicio", servicios=self._lista_servicios()))
        if dialogo.dia is None:
            dialogo.paso = "dia"
            return Respuesta(self._pack.t("pide_dia"))
        huecos = self._horas_libres(dialogo.servicio, dialogo.dia)
        if not huecos:
            dialogo.dia = None
            dialogo.paso = "dia"
            return Respuesta(
                self._pack.t("sin_huecos", servicio=self._nombre_servicio(dialogo.servicio))
            )
        if dialogo.hora is None:
            dialogo.paso = "hora"
            return Respuesta(
                self._pack.t(
                    "ofrece_horas",
                    servicio=self._nombre_servicio(dialogo.servicio),
                    dia=dialogo.dia.strftime("%d/%m"),
                    horas=self._lista_horas(huecos),
                )
            )
        if dialogo.nombre is None:
            dialogo.paso = "nombre"
            if nombre.strip():
                dialogo.nombre = nombre.strip()[:80]
                return self._avanzar(telefono, nombre, dialogo)
            return Respuesta(self._pack.t("pide_nombre"))
        dialogo.paso = "confirmar"
        return Respuesta(
            self._pack.t(
                "confirma",
                servicio=self._nombre_servicio(dialogo.servicio),
                nombre=dialogo.nombre,
                dia=dialogo.dia.strftime("%d/%m"),
                hora=dialogo.hora,
            )
        )

    def _paso(
        self,
        telefono: str,
        nombre: str,
        dialogo: _Dialogo,
        texto: str,
        ejecutar: Ejecutor,
    ) -> Respuesta:
        if dialogo.paso == "servicio":
            servicio = self._servicio(texto)
            if servicio is None:
                return Respuesta(self._pack.t("servicio_no", servicios=self._lista_servicios()))
            dialogo.servicio = servicio
            return self._avanzar(telefono, nombre, dialogo)
        if dialogo.paso == "dia":
            dia = self._dia(texto)
            if dia is None:
                return Respuesta(self._pack.t("dia_no"))
            if dialogo.objetivo == "ver":
                del self._dialogos[telefono]
                return self._ver_dia(dialogo.servicio or self._default_servicio(), dia)
            dialogo.dia = dia
            return self._avanzar(telefono, nombre, dialogo)
        if dialogo.paso == "hora":
            hora = self._hora(texto)
            servicio = dialogo.servicio or ""
            dia = dialogo.dia or self._ahora().date()
            huecos = self._horas_libres(servicio, dia)
            libres = [(r, h) for r, h in huecos if h.strftime("%H:%M") == hora]
            if hora is None or not libres:
                return Respuesta(self._pack.t("hora_no", horas=self._lista_horas(huecos)))
            dialogo.hora = hora
            dialogo.recurso = libres[0][0]
            return self._avanzar(telefono, nombre, dialogo)
        if dialogo.paso == "nombre":
            dialogo.nombre = texto.strip()[:80] or nombre.strip()[:80] or "sin nombre"
            return self._avanzar(telefono, nombre, dialogo)
        if dialogo.paso == "confirmar":
            respuesta = self._intencion(texto, solo=("si", "no"))
            if respuesta == "no":
                del self._dialogos[telefono]
                return Respuesta(self._pack.t("cancelado_no_se"))
            if respuesta != "si":
                return self._avanzar(telefono, nombre, dialogo)
            return self._agendar(telefono, dialogo, ejecutar)
        if dialogo.paso == "cancelar":
            respuesta = self._intencion(texto, solo=("si", "no"))
            if respuesta == "si" and dialogo.cita is not None:
                del self._dialogos[telefono]
                resultado = ejecutar("cancelar", {"appointment": str(dialogo.cita)})
                if resultado.status == ToolStatus.DONE:
                    return Respuesta(self._pack.t("cancelado"), accion="cancelar")
                return Respuesta(self._motivo(resultado))
            if respuesta == "no":
                del self._dialogos[telefono]
                return Respuesta(self._pack.t("cita_sigue"))
            return Respuesta(self._pack.t("responde_si_no"))
        del self._dialogos[telefono]
        return Respuesta(self._pack.t("no_entiendo"))

    def _agendar(self, telefono: str, dialogo: _Dialogo, ejecutar: Ejecutor) -> Respuesta:
        assert dialogo.servicio is not None and dialogo.dia is not None and dialogo.hora is not None
        inicio = datetime.combine(
            dialogo.dia,
            time(int(dialogo.hora[:2]), int(dialogo.hora[3:])),
            tzinfo=self._ahora().tzinfo,
        )
        resultado = ejecutar(
            "agendar",
            {
                "service": dialogo.servicio,
                "resource": dialogo.recurso,
                "start": inicio.isoformat(),
                "name": dialogo.nombre or "",
                "phone": telefono,
            },
        )
        del self._dialogos[telefono]
        if resultado.status == ToolStatus.DONE:
            return Respuesta(
                self._pack.t(
                    "agendado",
                    servicio=self._nombre_servicio(dialogo.servicio),
                    dia=dialogo.dia.strftime("%d/%m"),
                    hora=dialogo.hora,
                ),
                accion="agendar",
            )
        return Respuesta(self._motivo(resultado))

    def _motivo(self, resultado: ToolResult) -> str:
        return self._pack.t(
            "no_agendado",
            motivo=self._motivos.get(f"reason_{resultado.reason}", resultado.reason or "error"),
        )

    def _default_servicio(self) -> str:
        servicios = self._setup().services
        return servicios[0].id if servicios else ""

    def _ver_disponibilidad(self, telefono: str, texto: str) -> Respuesta:
        servicio = self._servicio(texto) or self._default_servicio()
        dia = self._dia(texto)
        if dia is None:
            self._dialogos[telefono] = _Dialogo(paso="dia", objetivo="ver", servicio=servicio)
            return Respuesta(self._pack.t("disponibilidad_pide_dia"))
        return self._ver_dia(servicio, dia)

    def _ver_dia(self, servicio: str, dia: date) -> Respuesta:
        huecos = self._horas_libres(servicio, dia)
        if not huecos:
            return Respuesta(self._pack.t("sin_huecos", servicio=self._nombre_servicio(servicio)))
        return Respuesta(
            self._pack.t(
                "disponibilidad",
                servicio=self._nombre_servicio(servicio),
                dia=dia.strftime("%d/%m"),
                horas=self._lista_horas(huecos),
            )
        )

    def _proximas(self, telefono: str) -> list[Appointment]:
        contacto = self._store.client_by_phone(telefono)
        if contacto is None:
            return []
        ahora = self._ahora()
        return [
            c
            for c in self._store.appointments_for_client(contacto.id)
            if c.state == AppointmentState.BOOKED and c.start >= ahora
        ]

    def _mis_citas(self, telefono: str) -> str:
        citas = self._proximas(telefono)
        if not citas:
            return self._pack.t("sin_citas")
        lineas = [
            self._pack.t(
                "linea_cita",
                dia=c.start.strftime("%d/%m"),
                hora=c.start.strftime("%H:%M"),
                servicio=self._nombre_servicio(c.service),
            )
            for c in citas
        ]
        return self._pack.t("mis_citas", citas="; ".join(lineas))

    def _pedir_cancelar(self, telefono: str) -> Respuesta:
        citas = self._proximas(telefono)
        if not citas:
            return Respuesta(self._pack.t("sin_cita_cancelar"))
        cita = citas[0]
        self._dialogos[telefono] = _Dialogo(paso="cancelar", objetivo="cancelar", cita=cita.id)
        return Respuesta(
            self._pack.t(
                "pide_cancelar",
                dia=cita.start.strftime("%d/%m"),
                hora=cita.start.strftime("%H:%M"),
            )
        )

    # Lo que el agente dice sin que nadie le escriba: recordatorios y mensajes posteriores.

    def recordatorios(self, *, registrar: Callable[[int, str], None]) -> int:
        """Manda los recordatorios de las citas de las próximas horas y los mensajes de después de
        la cita. Cada uno se manda una sola vez: el registro de recordatorios lo impide. Devuelve
        cuántos mensajes salieron."""
        ahora = self._ahora()
        enviados = 0
        proximas = self._store.between(
            ahora, ahora + timedelta(hours=self._pack.recordatorio_horas)
        )
        for cita in proximas:
            if cita.state != AppointmentState.BOOKED:
                continue
            if not self._store.mark_reminder(cita.id, "recordatorio", ahora):
                continue
            telefono = self._store.client(cita.client).phone
            texto = self._pack.t(
                "recordatorio",
                servicio=self._nombre_servicio(cita.service),
                hora=cita.start.strftime("%H:%M"),
                dia=cita.start.strftime("%d/%m"),
            )
            self._store.add_message(
                phone=telefono, direction="out", text=texto, at=ahora, kind="recordatorio"
            )
            registrar(cita.id, "recordatorio")
            enviados += 1
        for cita in self._store.between(ahora - timedelta(days=7), ahora):
            if cita.state not in (AppointmentState.ATTENDED, AppointmentState.NO_SHOW):
                continue
            if not self._store.mark_reminder(cita.id, "post", ahora):
                continue
            telefono = self._store.client(cita.client).phone
            clave = (
                "post_atendida" if cita.state == AppointmentState.ATTENDED else "post_no_asistio"
            )
            self._store.add_message(
                phone=telefono, direction="out", text=self._pack.t(clave), at=ahora, kind="post"
            )
            registrar(cita.id, "post")
            enviados += 1
        return enviados
