from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from cryptography.fernet import Fernet

from aida.engine.actions import ToolStatus
from aida.platform.channel.config import ChannelConfig
from aida.platform.channel.send import ChannelTool
from aida.platform.channel.store import SqliteChannelStore


def test_enviar_done_y_cuerpo_a_meta(tmp_path: Path) -> None:
    store = SqliteChannelStore(tmp_path / "ch.sqlite", Fernet.generate_key())
    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    tel = "573001112233"

    store.add_inbound(
        wamid="wamid.IN1",
        phone=tel,
        name="Ana",
        kind="text",
        text="Hola",
        at=t0,
    )

    t_envio = t0 + timedelta(hours=1)
    config = ChannelConfig(
        relay_url="http://relay.local",
        phone_number_id="num-meta-1",
        graph_url="https://graph.facebook.com/v21.0",
        poll_seconds=10,
    )

    peticiones_recibidas: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        peticiones_recibidas.append(request)
        return httpx.Response(
            200,
            json={"messaging_product": "whatsapp", "messages": [{"id": "wamid.OUT_OK"}]},
        )

    cliente_http = httpx.Client(transport=httpx.MockTransport(handler))
    tool = ChannelTool(
        store=store,
        config=config,
        token="meta-token-xyz",
        http=cliente_http,
        now=lambda: t_envio,
    )

    res = tool.execute(
        action="enviar",
        arguments={"phone": tel, "text": "Sí, confirmada su cita", "by": "prof-1"},
    )
    assert res.status == ToolStatus.DONE
    assert res.data["message"] != ""

    assert len(peticiones_recibidas) == 1
    req = peticiones_recibidas[0]
    assert req.url == "https://graph.facebook.com/v21.0/num-meta-1/messages"
    assert req.headers["Authorization"] == "Bearer meta-token-xyz"
    cuerpo = json.loads(req.content)
    assert cuerpo["to"] == "573001112233"
    assert cuerpo["text"]["body"] == "Sí, confirmada su cita"

    # Verificado en store
    mensajes = store.messages(tel)
    assert len(mensajes) == 2
    assert mensajes[1].wamid == "wamid.OUT_OK"
    assert mensajes[1].by == "prof-1"


def test_enviar_ventana_cerrada_en_24h_exactas(tmp_path: Path) -> None:
    store = SqliteChannelStore(tmp_path / "ch.sqlite", Fernet.generate_key())
    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    tel = "573001112233"

    store.add_inbound(
        wamid="wamid.IN1",
        phone=tel,
        name="Ana",
        kind="text",
        text="Hola",
        at=t0,
    )

    # Exactamente 24 horas después
    t_24h = t0 + timedelta(hours=24)
    config = ChannelConfig(
        relay_url="http://relay.local",
        phone_number_id="num-1",
        graph_url="https://graph.local",
        poll_seconds=10,
    )
    tool = ChannelTool(
        store=store,
        config=config,
        token="tok",
        http=httpx.Client(),
        now=lambda: t_24h,
    )

    res = tool.execute(
        action="enviar",
        arguments={"phone": tel, "text": "Respuesta tardía", "by": "prof-1"},
    )
    assert res.status == ToolStatus.REJECTED
    assert res.reason == "window_closed"


def test_enviar_sin_entrada_previa_window_closed(tmp_path: Path) -> None:
    store = SqliteChannelStore(tmp_path / "ch.sqlite", Fernet.generate_key())
    config = ChannelConfig(
        relay_url="http://relay.local",
        phone_number_id="num-1",
        graph_url="https://graph.local",
        poll_seconds=10,
    )
    tool = ChannelTool(
        store=store,
        config=config,
        token="tok",
        http=httpx.Client(),
        now=lambda: datetime(2026, 10, 20, 10, 0, tzinfo=UTC),
    )

    res = tool.execute(
        action="enviar",
        arguments={"phone": "573009998877", "text": "Hola nuevo", "by": "prof-1"},
    )
    assert res.status == ToolStatus.REJECTED
    assert res.reason == "window_closed"


def test_enviar_meta_error_failed(tmp_path: Path) -> None:
    store = SqliteChannelStore(tmp_path / "ch.sqlite", Fernet.generate_key())
    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    tel = "573001112233"

    store.add_inbound(
        wamid="wamid.IN1",
        phone=tel,
        name="Ana",
        kind="text",
        text="Hola",
        at=t0,
    )

    def handler_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Invalid param"}})

    config = ChannelConfig(
        relay_url="http://relay.local",
        phone_number_id="num-1",
        graph_url="https://graph.local",
        poll_seconds=10,
    )
    tool = ChannelTool(
        store=store,
        config=config,
        token="tok",
        http=httpx.Client(transport=httpx.MockTransport(handler_error)),
        now=lambda: t0 + timedelta(minutes=5),
    )

    res = tool.execute(
        action="enviar",
        arguments={"phone": tel, "text": "Respuesta", "by": "prof-1"},
    )
    assert res.status == ToolStatus.FAILED
    assert res.reason == "meta_400"
    assert len(store.messages(tel)) == 1  # Solo la entrada


def test_enviar_texto_vacio_missing_text(tmp_path: Path) -> None:
    store = SqliteChannelStore(tmp_path / "ch.sqlite", Fernet.generate_key())
    config = ChannelConfig(
        relay_url="http://relay.local",
        phone_number_id="num-1",
        graph_url="https://graph.local",
        poll_seconds=10,
    )
    tool = ChannelTool(
        store=store,
        config=config,
        token="tok",
        http=httpx.Client(),
        now=lambda: datetime(2026, 10, 20, 10, 0, tzinfo=UTC),
    )

    r1 = tool.execute(action="enviar", arguments={"phone": "", "text": "Hola", "by": "u"})
    assert r1.status == ToolStatus.REJECTED and r1.reason == "missing_text"

    r2 = tool.execute(action="enviar", arguments={"phone": "123", "text": "   ", "by": "u"})
    assert r2.status == ToolStatus.REJECTED and r2.reason == "missing_text"

    r3 = tool.execute(action="desconocida", arguments={})
    assert r3.status == ToolStatus.REJECTED and r3.reason == "unknown_action"
