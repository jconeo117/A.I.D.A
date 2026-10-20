"""Enviar: la herramienta del canal para la puerta de acciones.

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
from aida.platform.channel.store import ChannelStore, normalize_number

CHANNEL_ACTIONS = frozenset({ActionId("enviar")})
WINDOW = timedelta(hours=24)


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
        if action != "enviar":
            return ToolResult(ToolStatus.REJECTED, "unknown_action")
        numero = normalize_number(arguments.get("phone", ""))
        texto = arguments.get("text", "").strip()
        if not numero or not texto:
            return ToolResult(ToolStatus.REJECTED, "missing_text")
        ultima = self._store.last_inbound(numero)
        ahora = self._now()
        if ultima is None or ahora - ultima >= WINDOW:
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
