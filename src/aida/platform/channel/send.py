"""Enviar y marcar como leída: la herramienta del canal para la puerta de acciones.

Enviar es irreversible. La autoridad es el clic de una persona identificada (decisión 3 de la
101b); su `id` llega en `arguments["by"]`. La ventana de 24 h la impone Meta: fuera de ella un
texto libre no sale, y se explica aquí antes de intentarlo.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta

import httpx

from aida.engine.actions import ToolResult, ToolStatus
from aida.engine.ontology import ActionId
from aida.platform.channel.config import ChannelConfig
from aida.platform.channel.store import ChannelStore, to_whatsapp

CHANNEL_ACTIONS = frozenset({ActionId("enviar"), ActionId("marcar_leida")})
WINDOW = timedelta(hours=24)


def window_open(last_inbound: datetime | None, now: datetime) -> bool:
    """La única puerta de la ventana: la herramienta la usa para rechazar y la pantalla para
    deshabilitar la caja. Con dos copias, la caja podría ofrecer lo que la herramienta niega."""
    return last_inbound is not None and now - last_inbound < WINDOW


def check_whatsapp_token(
    token: str, config: ChannelConfig, *, http: httpx.Client | None = None
) -> tuple[bool, str]:
    """La petición más barata que confirma token y número a la vez: leer el número. Un token
    caducado se ve al guardarlo, no cuando alguien espera respuesta."""
    conexion = http if http is not None else httpx.Client(timeout=10.0)
    try:
        respuesta = conexion.get(
            f"{config.graph_url}/{config.phone_number_id}",
            headers={"Authorization": f"Bearer {token.strip()}"},
        )
    except httpx.HTTPError as exc:
        return False, type(exc).__name__
    return respuesta.status_code == 200, str(respuesta.status_code)


class ChannelTool:
    """Implementa `ToolPort`."""

    def __init__(
        self,
        *,
        store: ChannelStore,
        config: ChannelConfig,
        token: str,
        http: httpx.Client,
        now: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._config = config
        self._token = token
        self._http = http
        self._now = now

    def execute(self, *, action: str, arguments: Mapping[str, str]) -> ToolResult:
        if action == "enviar":
            return self._send(arguments)
        if action == "marcar_leida":
            return self._mark_read(arguments)
        return ToolResult(ToolStatus.REJECTED, "unknown_action")

    def _send(self, arguments: Mapping[str, str]) -> ToolResult:
        numero = to_whatsapp(arguments.get("phone", ""), self._config.country_code)
        texto = arguments.get("text", "").strip()
        if not numero or not texto:
            return ToolResult(ToolStatus.REJECTED, "missing_text")
        ahora = self._now()
        if not window_open(self._store.last_inbound(numero), ahora):
            return ToolResult(ToolStatus.REJECTED, "window_closed")
        respuesta = self._http.post(
            f"{self._config.graph_url}/{self._config.phone_number_id}/messages",
            headers={"Authorization": f"Bearer {self._token}"},
            json={
                "messaging_product": "whatsapp",
                "to": numero,
                "type": "text",
                "text": {"body": texto},
            },
        )
        if respuesta.status_code != 200:
            return ToolResult(ToolStatus.FAILED, f"meta_{respuesta.status_code}")
        wamid = str(respuesta.json()["messages"][0]["id"])
        enviado = self._store.add_outbound(
            wamid=wamid, phone=numero, text=texto, at=ahora, by=arguments["by"]
        )
        return ToolResult(ToolStatus.DONE, data={"message": str(enviado.id)})

    def _mark_read(self, arguments: Mapping[str, str]) -> ToolResult:
        try:
            hasta = int(arguments.get("up_to", ""))
        except ValueError:
            return ToolResult(ToolStatus.REJECTED, "bad_argument")
        if not self._store.mark_read(
            arguments.get("phone", ""), up_to=hasta, by=arguments["by"], at=self._now()
        ):
            return ToolResult(ToolStatus.REJECTED, "unknown_conversation")
        return ToolResult(ToolStatus.DONE)
