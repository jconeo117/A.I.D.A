"""La configuración del contacto de la agenda: servicios, recursos y horarios (101b).

Es un archivo y no código: un consultorio y una peluquería no comparten horarios, y el
contacto lo ajusta a mano en su instalación (el editor llega en la 101c). Al primer arranque se
copia la plantilla de su sector desde `packs/agenda/<sector>/agenda.yaml`.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

import yaml

from aida.resources import data_root, user_dir

DIAS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
"""Los días en el mismo orden que `datetime.weekday()`."""


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    id: str
    label: str
    minutes: int


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    id: str
    label: str
    place: str
    services: tuple[str, ...]
    hours: Mapping[str, tuple[tuple[time, time], ...]]
    """Día de la semana (`mon`…`sun`) a tramos de atención."""


@dataclass(frozen=True, slots=True)
class AgendaSetup:
    sector: str
    slot_minutes: int
    client_label: str
    services: tuple[ServiceSpec, ...]
    resources: tuple[ResourceSpec, ...]

    def service(self, id_: str) -> ServiceSpec | None:
        return next((s for s in self.services if s.id == id_), None)

    def resource(self, id_: str) -> ResourceSpec | None:
        return next((r for r in self.resources if r.id == id_), None)


def _hora(texto: object, ruta: Path) -> time:
    if not isinstance(texto, str):
        raise ValueError(f"una hora debe ser texto «HH:MM» en {ruta}")
    try:
        h, m = (int(p) for p in texto.split(":"))
        return time(h, m)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"hora inválida {texto!r} en {ruta}") from exc


def _parsear(raw: Any, ruta: Path) -> AgendaSetup:
    if not isinstance(raw, dict):
        raise ValueError(f"{ruta} debe ser un mapeo YAML")
    slot = raw.get("slot_minutes")
    if not isinstance(slot, int) or isinstance(slot, bool) or slot <= 0:
        raise ValueError(f"'slot_minutes' debe ser un entero positivo en {ruta}")
    servicios: list[ServiceSpec] = []
    for s in raw.get("services", []):
        minutos = s.get("minutes")
        if not isinstance(minutos, int) or isinstance(minutos, bool) or minutos <= 0:
            raise ValueError(f"el servicio {s.get('id')!r} necesita minutos positivos en {ruta}")
        servicios.append(ServiceSpec(id=str(s["id"]), label=str(s["label"]), minutes=minutos))
    ids_servicios = {s.id for s in servicios}
    recursos: list[ResourceSpec] = []
    for r in raw.get("resources", []):
        ofrece = tuple(str(x) for x in r.get("services", []))
        desconocidos = [x for x in ofrece if x not in ids_servicios]
        if desconocidos:
            raise ValueError(
                f"el recurso {r['id']!r} ofrece servicios que no existen: {desconocidos}"
            )
        horario: dict[str, tuple[tuple[time, time], ...]] = {}
        for dia, tramos in (r.get("hours") or {}).items():
            if dia not in DIAS:
                raise ValueError(f"día desconocido {dia!r} en el recurso {r['id']!r}")
            parseados: list[tuple[time, time]] = []
            for par in tramos:
                inicio, fin = _hora(par[0], ruta), _hora(par[1], ruta)
                minutos = (fin.hour * 60 + fin.minute) - (inicio.hour * 60 + inicio.minute)
                if minutos <= 0:
                    raise ValueError(
                        f"un tramo está al revés en {r['id']!r}: {par[0]} a {par[1]}"
                    )
                if minutos % slot != 0:
                    raise ValueError(
                        f"un tramo no es múltiplo de {slot} minutos en {r['id']!r}: "
                        f"{par[0]} a {par[1]}"
                    )
                parseados.append((inicio, fin))
            horario[dia] = tuple(parseados)
        recursos.append(
            ResourceSpec(
                id=str(r["id"]),
                label=str(r["label"]),
                place=str(r["place"]),
                services=ofrece,
                hours=horario,
            )
        )
    return AgendaSetup(
        sector=str(raw["sector"]),
        slot_minutes=slot,
        client_label=str(raw["client_label"]),
        services=tuple(servicios),
        resources=tuple(recursos),
    )


def _copiar_plantilla(destino: Path) -> None:
    """La plantilla del sector que dice `packs/agenda/arm.yaml`: la primera vez, nadie escribe
    el archivo a mano."""
    manifiesto = yaml.safe_load(
        (data_root() / "packs" / "agenda" / "arm.yaml").read_text(encoding="utf-8")
    )
    sector = str(manifiesto["default_sector"])
    origen = data_root() / "packs" / "agenda" / sector / "agenda.yaml"
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origen, destino)


def load_setup(path: Path | None = None) -> AgendaSetup:
    """Revienta si el archivo no cuadra: una agenda con un horario al revés se nota al
    agendar, cuando ya hay alguien esperando."""
    ruta = path if path is not None else user_dir("config") / "agenda.yaml"
    if not ruta.exists():
        _copiar_plantilla(ruta)
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    return _parsear(raw, ruta)
