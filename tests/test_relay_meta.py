from __future__ import annotations

import hashlib
import hmac
import json

from aida.relay.meta import incoming, valid_signature


def test_firma_valida_e_invalida() -> None:
    secreto = "secreto_meta_123"
    cuerpo = b'{"object":"whatsapp_business_account"}'
    valida = "sha256=" + hmac.new(secreto.encode(), cuerpo, hashlib.sha256).hexdigest()

    assert valid_signature(secreto, cuerpo, valida) is True
    assert valid_signature("otro_secreto", cuerpo, valida) is False
    assert valid_signature(secreto, cuerpo, valida.replace("sha256=", "")) is False
    assert valid_signature(secreto, cuerpo, None) is False
    assert valid_signature(secreto, cuerpo, "") is False


def test_incoming_texto_y_nombre() -> None:
    cuerpo = json.dumps({
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
            "changes": [{
                "field": "messages",
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"phone_number_id": "100200300"},
                    "contacts": [{"wa_id": "573001112233", "profile": {"name": "Ana Gómez"}}],
                    "messages": [{
                        "from": "573001112233",
                        "id": "wamid.TEST1",
                        "timestamp": "1729440000",
                        "type": "text",
                        "text": {"body": "Hola, ¿hay citas?"},
                    }],
                },
            }],
        }],
    }).encode("utf-8")

    entradas = incoming(cuerpo)
    assert len(entradas) == 1
    assert entradas[0].phone_number_id == "100200300"
    datos = json.loads(entradas[0].payload)
    assert datos["wamid"] == "wamid.TEST1"
    assert datos["from"] == "573001112233"
    assert datos["name"] == "Ana Gómez"
    assert datos["type"] == "text"
    assert datos["text"] == "Hola, ¿hay citas?"
    assert datos["timestamp"] == "1729440000"


def test_incoming_ignora_estados_y_cuerpo_raro() -> None:
    cuerpo_estados = json.dumps({
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "field": "messages",
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"phone_number_id": "100200300"},
                    "statuses": [{
                        "id": "wamid.OUT1",
                        "status": "delivered",
                        "recipient_id": "57300",
                    }],
                },
            }],
        }],
    }).encode("utf-8")
    assert incoming(cuerpo_estados) == ()

    assert incoming(b"no json") == ()
    assert incoming(b"{}") == ()
    assert incoming(b"[]") == ()


def test_incoming_tipo_no_texto_lleva_texto_vacio() -> None:
    cuerpo_audio = json.dumps({
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "field": "messages",
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"phone_number_id": "100200300"},
                    "messages": [{
                        "from": "573001112233",
                        "id": "wamid.AUDIO1",
                        "timestamp": "1729440000",
                        "type": "audio",
                        "audio": {"id": "media-1"},
                    }],
                },
            }],
        }],
    }).encode("utf-8")

    entradas = incoming(cuerpo_audio)
    assert len(entradas) == 1
    datos = json.loads(entradas[0].payload)
    assert datos["type"] == "audio"
    assert datos["text"] == ""
