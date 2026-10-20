"""El canal de WhatsApp en el equipo (102a): núcleo de la plataforma, no un brazo. Lo activa
la licencia (`whatsapp` en `arms`) y no importa ningún brazo; la pantalla la monta la agenda
(102b)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import httpx

from aida.engine.events import EventLog
from aida.keys import cache_key, channel_private_key, relay_token, whatsapp_token
from aida.platform.channel.config import ChannelConfig, load_channel_config
from aida.platform.channel.poller import Poller, poll_once
from aida.platform.channel.send import ChannelTool
from aida.platform.channel.store import SqliteChannelStore
from aida.platform.license import LicenseState
from aida.resources import user_dir

LICENSE_ID = "whatsapp"


@dataclass(slots=True)
class Channel:
    config: ChannelConfig
    store: SqliteChannelStore
    tool: ChannelTool
    poller: Poller
    log: EventLog


def open_channel(
    license: LicenseState,
    *,
    http: httpx.Client | None = None,
    now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> Channel | None:
    """`None` si la licencia no lo da o falta algo: el archivo `channel.yaml`, el token de Meta,
    el del relé o la clave del equipo. Sin canal, AIDA trabaja igual."""
    if not license.allows(LICENSE_ID):
        return None
    config = load_channel_config()
    token = whatsapp_token()
    recogida = relay_token()
    clave = cache_key()
    privada = channel_private_key()
    if config is None or token is None or recogida is None or clave is None or privada is None:
        return None
    conexion = http if http is not None else httpx.Client(timeout=15.0)
    store = SqliteChannelStore(user_dir("data") / "channel.sqlite", clave.encode())
    log = EventLog()
    reloj = [0]
    candado = threading.Lock()

    def tick() -> int:
        with candado:
            reloj[0] += 1
            return reloj[0]

    def paso() -> object:
        return poll_once(
            http=conexion, config=config, token=recogida, private_key=privada,
            store=store, log=log, tick=tick,
        )

    return Channel(
        config=config,
        store=store,
        tool=ChannelTool(store=store, config=config, token=token, http=conexion, now=now),
        poller=Poller(paso, config.poll_seconds),
        log=log,
    )
