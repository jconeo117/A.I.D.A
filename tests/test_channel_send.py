from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from cryptography.fernet import Fernet

from aida.engine.actions import ToolStatus
from aida.platform.channel.config import ChannelConfig
from aida.platform.channel.send import (
    ChannelTool,
    check_whatsapp_token,
    window_open,
)
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


def test_window_open_bordes() -> None:
    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    assert window_open(None, t0) is False
    # 23:59:59 después -> abierta
    t_casi_24h = t0 + timedelta(hours=24) - timedelta(seconds=1)
    assert window_open(t0, t_casi_24h) is True
    # 24:00:00 exactas -> cerrada
    t_24h = t0 + timedelta(hours=24)
    assert window_open(t0, t_24h) is False


def test_enviar_pone_prefijo_en_to(tmp_path: Path) -> None:
    store = SqliteChannelStore(tmp_path / "ch.sqlite", Fernet.generate_key(), country="57")
    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    # Entrada llega con 573001112233
    store.add_inbound(
        wamid="w.in", phone="573001112233", name="Ana", kind="text", text="Hola", at=t0
    )
    peticiones: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        peticiones.append(request)
        return httpx.Response(200, json={"messages": [{"id": "w.out"}]})

    cliente = httpx.Client(transport=httpx.MockTransport(handler))
    config = ChannelConfig(
        relay_url="http://relay.local",
        phone_number_id="num-1",
        graph_url="https://graph.local",
        poll_seconds=10,
        country_code="57",
    )
    tool = ChannelTool(
        store=store,
        config=config,
        token="tok",
        http=cliente,
        now=lambda: t0 + timedelta(minutes=1),
    )
    # Envío con número sin prefijo nacional
    res = tool.execute(
        action="enviar",
        arguments={"phone": "3001112233", "text": "Respuesta", "by": "prof-1"},
    )
    assert res.status == ToolStatus.DONE
    assert len(peticiones) == 1
    cuerpo = json.loads(peticiones[0].content)
    assert cuerpo["to"] == "573001112233"


def test_marcar_leida_done_unknown_y_bad_argument(tmp_path: Path) -> None:
    store = SqliteChannelStore(tmp_path / "ch.sqlite", Fernet.generate_key(), country="57")
    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    m1 = store.add_inbound(
        wamid="w.1", phone="573001112233", name="Ana", kind="text", text="Hola", at=t0
    )
    assert m1 is not None

    config = ChannelConfig(
        relay_url="http://relay.local",
        phone_number_id="num-1",
        graph_url="https://graph.local",
        poll_seconds=10,
        country_code="57",
    )
    tool = ChannelTool(
        store=store,
        config=config,
        token="tok",
        http=httpx.Client(),
        now=lambda: t0 + timedelta(minutes=1),
    )

    # 1. bad_argument (up_to no es número entero)
    r_bad = tool.execute(
        action="marcar_leida", arguments={"phone": "573001112233", "up_to": "xyz", "by": "p1"}
    )
    assert r_bad.status == ToolStatus.REJECTED
    assert r_bad.reason == "bad_argument"

    # 2. unknown_conversation (teléfono que no existe)
    r_unk = tool.execute(
        action="marcar_leida", arguments={"phone": "573009999999", "up_to": "1", "by": "p1"}
    )
    assert r_unk.status == ToolStatus.REJECTED
    assert r_unk.reason == "unknown_conversation"

    # 3. done (marca leída exitosa)
    r_ok = tool.execute(
        action="marcar_leida",
        arguments={"phone": "573001112233", "up_to": str(m1.id), "by": "p1"},
    )
    assert r_ok.status == ToolStatus.DONE


def test_check_whatsapp_token_200_401_y_red() -> None:
    config = ChannelConfig(
        relay_url="http://relay.local",
        phone_number_id="num-123",
        graph_url="https://graph.local",
        poll_seconds=10,
    )

    # 1. 200 OK
    def handler_200(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "num-123"})

    c_200 = httpx.Client(transport=httpx.MockTransport(handler_200))
    ok, motivo = check_whatsapp_token("tok-valido", config, http=c_200)
    assert ok is True
    assert motivo == "200"

    # 2. 401 Unauthorized
    def handler_401(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    c_401 = httpx.Client(transport=httpx.MockTransport(handler_401))
    ok, motivo = check_whatsapp_token("tok-invalido", config, http=c_401)
    assert ok is False
    assert motivo == "401"

    # 3. Error de red
    def handler_err(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=request)

    c_err = httpx.Client(transport=httpx.MockTransport(handler_err))
    ok, motivo = check_whatsapp_token("tok", config, http=c_err)
    assert ok is False
    assert motivo == "ConnectError"
