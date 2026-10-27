from __future__ import annotations

import stat
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest
import yaml

from aida.keys import channel_private_key, relay_token
from aida.platform.channel import missing_parts, open_channel
from aida.platform.license import License, LicenseState
from aida.run import _channel_setup


def test_channel_private_key_se_crea_una_vez_y_0600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))

    k1 = channel_private_key()
    assert k1 is not None
    assert len(k1) == 32

    archivo = config_dir / "channel-x25519.key"
    assert archivo.is_file()
    assert sys.platform == "win32" or stat.S_IMODE(archivo.stat().st_mode) == 0o600

    k2 = channel_private_key()
    assert k1 == k2


def test_relay_token_create(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))

    assert relay_token(create=False) is None

    t1 = relay_token(create=True)
    assert t1 is not None
    assert len(t1) >= 32

    archivo = config_dir / "relay.key"
    assert archivo.is_file()
    assert sys.platform == "win32" or stat.S_IMODE(archivo.stat().st_mode) == 0o600

    t2 = relay_token(create=True)
    assert t1 == t2


def test_open_channel_none_sin_licencia_o_sin_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AIDA_DATA_DIR", str(data_dir))

    # Preparamos configuración y claves válidas
    ch_yaml = config_dir / "channel.yaml"
    ch_yaml.write_text(
        yaml.dump({
            "relay_url": "http://relay.local",
            "phone_number_id": "123",
            "graph_url": "https://graph.local",
            "poll_seconds": 10,
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIDA_WHATSAPP_TOKEN", "meta-tok")
    monkeypatch.setenv("AIDA_RELAY_TOKEN", "relay-tok")
    (config_dir / "cache.key").write_text(
        "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=", encoding="utf-8"
    )
    channel_private_key(create=True)

    # 1. Licencia SIN whatsapp -> devuelve None (para que caiga M13)
    lic_sin_wa = LicenseState(
        status="valid",
        license=License(
            id="lic-test",
            client="Test",
            arms=("agenda", "health"),
            issued=date.today(),
            expires=date.today() + timedelta(days=365),
        ),
    )
    assert open_channel(lic_sin_wa) is None

    # 2. Licencia CON whatsapp -> devuelve Channel
    lic_con_wa = LicenseState(
        status="valid",
        license=License(
            id="lic-test",
            client="Test",
            arms=("agenda", "whatsapp"),
            issued=date.today(),
            expires=date.today() + timedelta(days=365),
        ),
    )
    canal = open_channel(lic_con_wa)
    assert canal is not None

    # 3. Sin channel.yaml -> devuelve None
    ch_yaml.unlink()
    assert open_channel(lic_con_wa) is None


def test_channel_setup_imprime_publica_y_huella_sin_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))

    ret = _channel_setup()
    assert ret == 0

    salida = capsys.readouterr().out
    assert "public_key:" in salida
    assert "token_sha256:" in salida

    # Verificar que el token secreto en claro NO está en la salida
    tok = relay_token()
    assert tok is not None
    assert tok not in salida


def test_missing_parts_orden_y_licencia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))

    lic_sin_wa = LicenseState(
        status="valid",
        license=License(
            id="lic-test",
            client="Test",
            arms=("agenda",),
            issued=date.today(),
            expires=date.today() + timedelta(days=365),
        ),
    )
    # Sin licencia whatsapp -> ("license",)
    assert missing_parts(lic_sin_wa) == ("license",)

    lic_con_wa = LicenseState(
        status="valid",
        license=License(
            id="lic-test",
            client="Test",
            arms=("agenda", "whatsapp"),
            issued=date.today(),
            expires=date.today() + timedelta(days=365),
        ),
    )
    # Sin nada configurado -> orden ("config", "whatsapp_token", "relay_token")
    assert missing_parts(lic_con_wa) == ("config", "whatsapp_token", "relay_token")

    # Con channel.yaml
    ch_yaml = config_dir / "channel.yaml"
    ch_yaml.write_text(
        yaml.dump({
            "relay_url": "http://relay.local",
            "phone_number_id": "123",
            "graph_url": "https://graph.local",
            "poll_seconds": 10,
        }),
        encoding="utf-8",
    )
    assert missing_parts(lic_con_wa) == ("whatsapp_token", "relay_token")

    # Con relay_token
    monkeypatch.setenv("AIDA_RELAY_TOKEN", "relay-tok")
    assert missing_parts(lic_con_wa) == ("whatsapp_token",)

    # Con whatsapp_token
    monkeypatch.setenv("AIDA_WHATSAPP_TOKEN", "meta-tok")
    assert missing_parts(lic_con_wa) == ()


def test_open_channel_trae_puerta_y_candado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AIDA_DATA_DIR", str(data_dir))

    ch_yaml = config_dir / "channel.yaml"
    ch_yaml.write_text(
        yaml.dump({
            "relay_url": "http://relay.local",
            "phone_number_id": "123",
            "graph_url": "https://graph.local",
            "poll_seconds": 10,
            "country_code": "57",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIDA_WHATSAPP_TOKEN", "meta-tok")
    monkeypatch.setenv("AIDA_RELAY_TOKEN", "relay-tok")
    (config_dir / "cache.key").write_text(
        "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=", encoding="utf-8"
    )
    channel_private_key(create=True)

    lic = LicenseState(
        status="valid",
        license=License(
            id="lic-test",
            client="Test",
            arms=("agenda", "whatsapp"),
            issued=date.today(),
            expires=date.today() + timedelta(days=365),
        ),
    )
    cliente = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    canal = open_channel(lic, http=cliente)
    assert canal is not None
    assert canal.gate is not None
    assert canal.lock is not None
    assert canal.tool is not None
    assert canal.store is not None
    assert canal.poller is not None
    assert canal.log is not None
    assert canal.config.country_code == "57"
