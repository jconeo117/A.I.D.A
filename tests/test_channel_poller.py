from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from cryptography.fernet import Fernet

from aida.engine.events import EventKind, EventLog
from aida.platform.channel.config import ChannelConfig
from aida.platform.channel.poller import OWNER, Poller, poll_once
from aida.platform.channel.store import SqliteChannelStore
from aida.relay.config import RelayConfig, Tenant
from aida.relay.seal import new_keypair, seal
from aida.relay.server import build_server
from aida.relay.store import SqliteRelayStore
from tests.conftest import _ORIGINAL_HANDLE_REQUEST

CanalEnv = tuple[str, str, bytes, SqliteRelayStore, SqliteChannelStore, EventLog, ChannelConfig]


@pytest.fixture
def canal_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> CanalEnv:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)
    relay_db = tmp_path / "relay.sqlite"
    r_store = SqliteRelayStore(relay_db)

    priv, pub = new_keypair()
    token = "token-secreto"
    t = Tenant(
        id="t1",
        phone_number_id="num-1",
        public_key=pub,
        token_sha256=httpx.URL("")._uri_reference.encode().hex() if False else "",
    )
    import hashlib
    t = Tenant(
        id="t1",
        phone_number_id="num-1",
        public_key=pub,
        token_sha256=hashlib.sha256(token.encode()).hexdigest(),
    )
    r_config = RelayConfig(app_secret="sec", verify_token="vt", tenants=(t,))

    srv = build_server("127.0.0.1", 0, r_config, r_store)
    puerto = srv.server_port
    hilo = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo.start()

    relay_url = f"http://127.0.0.1:{puerto}"
    ch_config = ChannelConfig(
        relay_url=relay_url,
        phone_number_id="num-1",
        graph_url="http://graph.local",
        poll_seconds=1,
    )

    ch_store = SqliteChannelStore(tmp_path / "ch.sqlite", Fernet.generate_key())
    log = EventLog()

    return relay_url, token, priv, r_store, ch_store, log, ch_config


def test_poll_once_guarda_emite_y_confirma(canal_env: CanalEnv) -> None:
    _, token, priv, r_store, ch_store, log, ch_config = canal_env

    # Meter un mensaje sellado en el relé
    _, pub = new_keypair()  # solo para referencia
    from aida.relay.seal import public_of
    pub = public_of(priv)

    datos = {
        "wamid": "wamid.POLL1",
        "from": "573001112233",
        "name": "Ana",
        "type": "text",
        "text": "Hola poller",
        "timestamp": "1729440000",
    }
    sellado = seal(pub, json.dumps(datos).encode())
    r_store.put(tenant="t1", sealed=sellado)

    reloj = [0]
    with httpx.Client(trust_env=False) as http:
        reporte = poll_once(
            http=http,
            config=ch_config,
            token=token,
            private_key=priv,
            store=ch_store,
            log=log,
            tick=lambda: reloj[0] + 1,
        )

    assert reporte.stored == 1
    assert reporte.duplicates == 0
    assert reporte.unreadable == 0

    # Guardado en channel store
    msgs = ch_store.messages("573001112233")
    assert len(msgs) == 1
    assert msgs[0].text == "Hola poller"

    # Emitido en log
    eventos = list(log)
    assert len(eventos) == 1
    assert eventos[0].kind == EventKind.INPUT_OBSERVED
    assert eventos[0].owner == OWNER
    assert eventos[0].content == "Hola poller"

    # Confirmado en relé (borrado)
    assert len(r_store.pending(tenant="t1", limit=10)) == 0


def test_poll_once_repetido_confirma_sin_evento(canal_env: CanalEnv) -> None:
    _, token, priv, r_store, ch_store, log, ch_config = canal_env
    from aida.relay.seal import public_of
    pub = public_of(priv)

    datos = {
        "wamid": "wamid.POLL_REP",
        "from": "573001112233",
        "name": "Ana",
        "type": "text",
        "text": "Mensaje repetido",
        "timestamp": "1729440000",
    }
    # Pre-insertar en store local
    ch_store.add_inbound(
        wamid=datos["wamid"],
        phone=datos["from"],
        name=datos["name"],
        kind="text",
        text=datos["text"],
        at=datetime.fromtimestamp(1729440000, tz=UTC),
    )

    # Meter en relé
    sellado = seal(pub, json.dumps(datos).encode())
    r_store.put(tenant="t1", sealed=sellado)

    reloj = [0]
    with httpx.Client(trust_env=False) as http:
        reporte = poll_once(
            http=http,
            config=ch_config,
            token=token,
            private_key=priv,
            store=ch_store,
            log=log,
            tick=lambda: reloj[0] + 1,
        )

    assert reporte.stored == 0
    assert reporte.duplicates == 1
    assert reporte.unreadable == 0

    # No emite evento nuevo
    assert len(list(log)) == 0

    # Pero sí confirma en relé
    assert len(r_store.pending(tenant="t1", limit=10)) == 0


def test_poll_once_ilegible_no_confirma(canal_env: CanalEnv) -> None:
    _, token, priv, r_store, ch_store, log, ch_config = canal_env

    # Sellar con OTRA clave ajena
    _, pub_ajena = new_keypair()
    sellado_ajeno = seal(pub_ajena, b'{"wamid":"x"}')
    r_store.put(tenant="t1", sealed=sellado_ajeno)

    reloj = [0]
    with httpx.Client(trust_env=False) as http:
        reporte = poll_once(
            http=http,
            config=ch_config,
            token=token,
            private_key=priv,
            store=ch_store,
            log=log,
            tick=lambda: reloj[0] + 1,
        )

    assert reporte.unreadable == 1
    assert reporte.stored == 0
    assert reporte.duplicates == 0

    # No se borra del relé (decisión 8)
    assert len(r_store.pending(tenant="t1", limit=10)) == 1


def test_poller_sigue_tras_un_fallo_de_red() -> None:
    intentos = [0]
    poller: Poller | None = None

    def paso_falla_luego_ok() -> None:
        intentos[0] += 1
        if intentos[0] == 1:
            raise httpx.ConnectError("fallo simulado")
        if poller is not None:
            poller._stop.set()

    poller = Poller(paso_falla_luego_ok, seconds=0)
    poller.start()
    if poller._thread is not None:
        poller._thread.join(timeout=2.0)

    assert intentos[0] >= 2
    assert poller.last_error == ""
