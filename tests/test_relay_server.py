from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
from pathlib import Path

import httpx
import pytest

from aida.relay.config import RelayConfig, Tenant, load_relay_config
from aida.relay.seal import new_keypair
from aida.relay.server import MAX_BODY, build_server
from aida.relay.store import SqliteRelayStore
from tests.conftest import _ORIGINAL_HANDLE_REQUEST


@pytest.fixture
def relay_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, RelayConfig, SqliteRelayStore, str, Tenant]:
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)
    db_path = tmp_path / "relay.sqlite"
    store = SqliteRelayStore(db_path)

    _, pub1 = new_keypair()
    token1 = "token-inquilino-1"
    token1_hash = hashlib.sha256(token1.encode()).hexdigest()
    t1 = Tenant(
        id="t1",
        phone_number_id="num-101",
        public_key=pub1,
        token_sha256=token1_hash,
    )

    _, pub2 = new_keypair()
    token2 = "token-inquilino-2"
    token2_hash = hashlib.sha256(token2.encode()).hexdigest()
    t2 = Tenant(
        id="t2",
        phone_number_id="num-102",
        public_key=pub2,
        token_sha256=token2_hash,
    )

    config = RelayConfig(
        app_secret="secreto-meta",
        verify_token="token-verificacion",
        tenants=(t1, t2),
    )

    srv = build_server("127.0.0.1", 0, config, store)
    puerto = srv.server_port
    hilo = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo.start()

    base_url = f"http://127.0.0.1:{puerto}"
    return base_url, config, store, token1, t1


def _meta_body(phone_number_id: str, text: str, wamid: str = "wamid.1") -> bytes:
    return json.dumps({
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "field": "messages",
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"phone_number_id": phone_number_id},
                    "contacts": [{"wa_id": "573001112233", "profile": {"name": "Test"}}],
                    "messages": [{
                        "from": "573001112233",
                        "id": wamid,
                        "timestamp": "1729440000",
                        "type": "text",
                        "text": {"body": text},
                    }],
                },
            }],
        }],
    }).encode("utf-8")


def _signature(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_verificacion_webhook(
    relay_app: tuple[str, RelayConfig, SqliteRelayStore, str, Tenant],
) -> None:
    base_url, config, _, _, _ = relay_app
    with httpx.Client(trust_env=False) as client:
        r_ok = client.get(
            f"{base_url}/webhook?hub.mode=subscribe&hub.verify_token={config.verify_token}&hub.challenge=reto123"
        )
        assert r_ok.status_code == 200
        assert r_ok.text == "reto123"

        r_bad = client.get(
            f"{base_url}/webhook?hub.mode=subscribe&hub.verify_token=malo&hub.challenge=reto123"
        )
        assert r_bad.status_code == 403


def test_webhook_sin_firma_401_y_no_guarda(
    relay_app: tuple[str, RelayConfig, SqliteRelayStore, str, Tenant],
) -> None:
    base_url, _, store, _, t1 = relay_app
    cuerpo = _meta_body(t1.phone_number_id, "Hola")
    with httpx.Client(trust_env=False) as client:
        # Sin firma
        r1 = client.post(f"{base_url}/webhook", content=cuerpo)
        assert r1.status_code == 401

        # Firma inválida
        r2 = client.post(
            f"{base_url}/webhook",
            content=cuerpo,
            headers={"X-Hub-Signature-256": "sha256=invalida"},
        )
        assert r2.status_code == 401

        assert len(store.pending(tenant=t1.id, limit=10)) == 0


def test_webhook_numero_desconocido_no_guarda(
    relay_app: tuple[str, RelayConfig, SqliteRelayStore, str, Tenant],
) -> None:
    base_url, config, store, _, t1 = relay_app
    cuerpo = _meta_body("num-fantasma", "Hola")
    firma = _signature(config.app_secret, cuerpo)
    with httpx.Client(trust_env=False) as client:
        r = client.post(
            f"{base_url}/webhook",
            content=cuerpo,
            headers={"X-Hub-Signature-256": firma},
        )
        assert r.status_code == 200
        assert r.json() == {"stored": 0}
        assert len(store.pending(tenant=t1.id, limit=10)) == 0


def test_pull_solo_del_inquilino_y_ack_no_toca_ajenos(
    relay_app: tuple[str, RelayConfig, SqliteRelayStore, str, Tenant],
) -> None:
    base_url, config, store, token1, t1 = relay_app
    t2 = config.tenants[1]
    token2 = "token-inquilino-2"

    cuerpo1 = _meta_body(t1.phone_number_id, "Mensaje 1", "wamid.1")
    cuerpo2 = _meta_body(t2.phone_number_id, "Mensaje 2", "wamid.2")

    with httpx.Client(trust_env=False) as client:
        client.post(
            f"{base_url}/webhook",
            content=cuerpo1,
            headers={"X-Hub-Signature-256": _signature(config.app_secret, cuerpo1)},
        )
        client.post(
            f"{base_url}/webhook",
            content=cuerpo2,
            headers={"X-Hub-Signature-256": _signature(config.app_secret, cuerpo2)},
        )

        # Pull de t1 solo trae items de t1
        r_pull1 = client.get(f"{base_url}/pull", headers={"Authorization": f"Bearer {token1}"})
        assert r_pull1.status_code == 200
        items1 = r_pull1.json()["items"]
        assert len(items1) == 1
        id_t1 = items1[0]["id"]

        # Pull de t2 solo trae items de t2
        r_pull2 = client.get(f"{base_url}/pull", headers={"Authorization": f"Bearer {token2}"})
        assert r_pull2.status_code == 200
        items2 = r_pull2.json()["items"]
        assert len(items2) == 1
        id_t2 = items2[0]["id"]

        # t1 intenta hacer ack del id de t2 -> no debe borrarlo (acked: 0)
        r_ack_ajeno = client.post(
            f"{base_url}/ack",
            headers={"Authorization": f"Bearer {token1}"},
            json={"ids": [id_t2]},
        )
        assert r_ack_ajeno.status_code == 200
        assert r_ack_ajeno.json() == {"acked": 0}
        assert len(store.pending(tenant=t2.id, limit=10)) == 1

        # t1 hace ack de su propio id -> acked: 1
        r_ack_propio = client.post(
            f"{base_url}/ack",
            headers={"Authorization": f"Bearer {token1}"},
            json={"ids": [id_t1]},
        )
        assert r_ack_propio.status_code == 200
        assert r_ack_propio.json() == {"acked": 1}
        assert len(store.pending(tenant=t1.id, limit=10)) == 0


def test_almacen_del_rele_no_tiene_claro(
    tmp_path: Path,
    relay_app: tuple[str, RelayConfig, SqliteRelayStore, str, Tenant],
) -> None:
    base_url, config, _, _, t1 = relay_app
    texto_secreto = "TextoSuperSecretoClaro123"
    cuerpo = _meta_body(t1.phone_number_id, texto_secreto)
    firma = _signature(config.app_secret, cuerpo)

    with httpx.Client(trust_env=False) as client:
        r = client.post(
            f"{base_url}/webhook",
            content=cuerpo,
            headers={"X-Hub-Signature-256": firma},
        )
        assert r.status_code == 200

    ruta_sqlite = tmp_path / "relay.sqlite"
    assert ruta_sqlite.is_file()
    contenido = ruta_sqlite.read_bytes()
    assert texto_secreto.encode("utf-8") not in contenido


def test_cuerpo_demasiado_grande_413(
    relay_app: tuple[str, RelayConfig, SqliteRelayStore, str, Tenant],
) -> None:
    base_url, _, _, _, _ = relay_app
    cuerpo_enorme = b"X" * (MAX_BODY + 100)
    with httpx.Client(trust_env=False) as client:
        r = client.post(f"{base_url}/webhook", content=cuerpo_enorme)
        assert r.status_code == 413


def test_config_rechaza_repetidos_y_clave_corta(tmp_path: Path) -> None:
    _, pub = new_keypair()
    pub_b64 = base64.b64encode(pub).decode()
    env = {
        "AIDA_RELAY_APP_SECRET": "sec",
        "AIDA_RELAY_VERIFY_TOKEN": "vt",
    }

    # Clave corta
    f_corta = tmp_path / "corta.json"
    pub_corta = base64.b64encode(b"corta").decode()
    t_corta = [{"id": "t1", "phone_number_id": "1", "public_key": pub_corta, "token_sha256": "abc"}]
    f_corta.write_text(json.dumps({"tenants": t_corta}))
    with pytest.raises(ValueError, match="32 bytes"):
        load_relay_config(f_corta, env)

    # Id repetido
    f_rep_id = tmp_path / "rep_id.json"
    f_rep_id.write_text(json.dumps({
        "tenants": [
            {"id": "t1", "phone_number_id": "1", "public_key": pub_b64, "token_sha256": "abc"},
            {"id": "t1", "phone_number_id": "2", "public_key": pub_b64, "token_sha256": "def"},
        ]
    }))
    with pytest.raises(ValueError, match="'id' repetido"):
        load_relay_config(f_rep_id, env)

    # Faltan variables de entorno
    with pytest.raises(ValueError, match="faltan AIDA_RELAY_APP_SECRET"):
        load_relay_config(f_rep_id, {})
