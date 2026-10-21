"""El canal de WhatsApp en el equipo (102a): núcleo de la plataforma, no un brazo. Lo activa
la licencia (`whatsapp` en `arms`) y no importa ningún brazo; la pantalla la monta la agenda
(102b), que lo recibe por `ShellContext.channel`."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import httpx

from aida.engine.actions import ActionGate
from aida.engine.events import EventLog
from aida.keys import cache_key, channel_private_key, relay_token, whatsapp_token
from aida.platform.channel.config import ChannelConfig, load_channel_config
from aida.platform.channel.poller import OWNER, Poller, poll_once
from aida.platform.channel.send import CHANNEL_ACTIONS, ChannelTool
from aida.platform.channel.store import SqliteChannelStore
from aida.platform.license import LicenseState
from aida.resources import user_dir

LICENSE_ID = "whatsapp"


@dataclass(slots=True)
class Channel:
    config: ChannelConfig
    store: SqliteChannelStore
    tool: ChannelTool
    gate: ActionGate
    """La única puerta del canal: la usan la pantalla (102b) y el agente (103)."""
    poller: Poller
    log: EventLog
    lock: threading.Lock
    """El que comparten la recogida y la puerta: las dos escriben en `log` (decisión 7)."""


def missing_parts(license: LicenseState) -> tuple[str, ...]:
    """Qué falta para abrir el canal, en orden fijo: `license`, `config`, `whatsapp_token`,
    `relay_token`. Una sola puerta: la usan `open_channel` y el panel «Canal»."""
    if not license.allows(LICENSE_ID):
        return ("license",)
    faltan: list[str] = []
    if load_channel_config() is None:
        faltan.append("config")
    if whatsapp_token() is None:
        faltan.append("whatsapp_token")
    if relay_token() is None:
        faltan.append("relay_token")
    return tuple(faltan)


def open_channel(
    license: LicenseState,
    *,
    http: httpx.Client | None = None,
    now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> Channel | None:
    """`None` si falta algo (`missing_parts`). Sin canal, AIDA trabaja igual."""
    if missing_parts(license):
        return None
    config = load_channel_config()
    token = whatsapp_token()
    recogida = relay_token()
    clave = cache_key()
    privada = channel_private_key()
    if config is None or token is None or recogida is None or clave is None or privada is None:
        return None
    conexion = http if http is not None else httpx.Client(timeout=15.0)
    store = SqliteChannelStore(
        user_dir("data") / "channel.sqlite", clave.encode(), country=config.country_code
    )
    log = EventLog()
    reloj = [0]
    candado_reloj = threading.Lock()
    cerrojo = threading.Lock()

    def tick() -> int:
        with candado_reloj:
            reloj[0] += 1
            return reloj[0]

    def paso() -> object:
        with cerrojo:
            return poll_once(
                http=conexion, config=config, token=recogida, private_key=privada,
                store=store, log=log, tick=tick,
            )

    herramienta = ChannelTool(store=store, config=config, token=token, http=conexion, now=now)
    return Channel(
        config=config,
        store=store,
        tool=herramienta,
        gate=ActionGate(
            actions=CHANNEL_ACTIONS, tool=herramienta, log=log, owner=OWNER, tick=tick
        ),
        poller=Poller(paso, config.poll_seconds),
        log=log,
        lock=cerrojo,
    )
