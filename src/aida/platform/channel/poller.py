"""Recoger del relé: el latido lento del reloj híbrido para lo que llega por el canal.

Un paso (`poll_once`) es síncrono y se prueba sin hilos ni red. El hilo (`Poller`) solo lo
repite. Un fallo de red no para el latido: se anota y se reintenta en la vuelta siguiente,
porque lo no confirmado sigue en el relé.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.platform.channel.config import ChannelConfig
from aida.platform.channel.store import ChannelStore
from aida.relay.seal import open_sealed

OWNER = AgentId("whatsapp")


@dataclass(frozen=True, slots=True)
class PollReport:
    stored: int
    duplicates: int
    unreadable: int


def poll_once(
    *,
    http: httpx.Client,
    config: ChannelConfig,
    token: str,
    private_key: bytes,
    store: ChannelStore,
    log: EventLog,
    tick: Callable[[], int],
) -> PollReport:
    """Recoge, abre, guarda y confirma. Se confirma lo guardado y lo repetido; lo que no abre,
    no (decisión 8)."""
    cabeceras = {"Authorization": f"Bearer {token}"}
    respuesta = http.get(f"{config.relay_url}/pull", headers=cabeceras)
    respuesta.raise_for_status()
    guardadas = repetidas = ilegibles = 0
    confirmar: list[int] = []
    for item in respuesta.json()["items"]:
        try:
            datos = json.loads(open_sealed(private_key, base64.b64decode(item["sealed"])))
        except ValueError:
            ilegibles += 1
            continue
        entrada = store.add_inbound(
            wamid=str(datos["wamid"]),
            phone=str(datos["from"]),
            name=str(datos["name"]),
            kind=str(datos["type"]),
            text=str(datos["text"]),
            at=datetime.fromtimestamp(int(datos["timestamp"]), tz=UTC),
        )
        if entrada is None:
            repetidas += 1
        else:
            guardadas += 1
            log.emit(
                tick=tick(),
                kind=EventKind.INPUT_OBSERVED,
                owner=OWNER,
                data={"message": str(entrada.id), "kind": entrada.kind},
                content=entrada.text,
            )
        confirmar.append(int(item["id"]))
    if confirmar:
        http.post(
            f"{config.relay_url}/ack", json={"ids": confirmar}, headers=cabeceras
        ).raise_for_status()
    return PollReport(stored=guardadas, duplicates=repetidas, unreadable=ilegibles)


class Poller:
    def __init__(self, step: Callable[[], object], seconds: int) -> None:
        self._step = step
        self._seconds = seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error = ""
        """El tipo del último fallo, o vacío. Lo enseñará Ajustes (102b)."""

    def _run(self) -> None:
        while True:
            try:
                self._step()
                self.last_error = ""
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                self.last_error = type(exc).__name__
            if self._stop.wait(self._seconds):
                return

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="aida-channel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
