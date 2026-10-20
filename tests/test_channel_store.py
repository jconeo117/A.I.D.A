from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet

from aida.platform.channel.store import Direction, SqliteChannelStore


def test_add_inbound_repetido_devuelve_none(tmp_path: Path) -> None:
    clave = Fernet.generate_key()
    store = SqliteChannelStore(tmp_path / "channel.sqlite", clave)

    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    m1 = store.add_inbound(
        wamid="wamid.123",
        phone="573001112233",
        name="Ana",
        kind="text",
        text="Hola",
        at=t0,
    )
    assert m1 is not None
    assert m1.wamid == "wamid.123"
    assert m1.direction == Direction.IN
    assert m1.phone == "573001112233"
    assert m1.name == "Ana"
    assert m1.text == "Hola"
    assert m1.at == t0

    # Segundo intento con el mismo wamid devuelve None
    m_rep = store.add_inbound(
        wamid="wamid.123",
        phone="573001112233",
        name="Ana",
        kind="text",
        text="Hola otra vez",
        at=t0,
    )
    assert m_rep is None


def test_texto_y_telefono_sellados_en_disco(tmp_path: Path) -> None:
    clave = Fernet.generate_key()
    db_file = tmp_path / "channel.sqlite"
    store = SqliteChannelStore(db_file, clave)

    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    tel = "573001112233"
    nombre = "Paciente Secreto"
    texto = "Mi texto confidencial"

    store.add_inbound(
        wamid="wamid.456",
        phone=tel,
        name=nombre,
        kind="text",
        text=texto,
        at=t0,
    )

    contenido = db_file.read_bytes()
    assert texto.encode("utf-8") not in contenido
    assert tel.encode("utf-8") not in contenido
    assert nombre.encode("utf-8") not in contenido


def test_messages_y_last_inbound_por_telefono_normalizado(tmp_path: Path) -> None:
    clave = Fernet.generate_key()
    store = SqliteChannelStore(tmp_path / "channel.sqlite", clave)

    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=15)
    t2 = t0 + timedelta(minutes=30)

    # Entran con distintos formatos de número (con espacios, +)
    store.add_inbound(
        wamid="wamid.A",
        phone="+57 300 111 2233",
        name="Ana",
        kind="text",
        text="Pregunta 1",
        at=t0,
    )
    store.add_outbound(
        wamid="wamid.B",
        phone="573001112233",
        text="Respuesta 1",
        at=t1,
        by="prof-1",
    )
    store.add_inbound(
        wamid="wamid.C",
        phone="57-300-111-2233",
        name="Ana",
        kind="text",
        text="Pregunta 2",
        at=t2,
    )

    # Buscar con cualquier variante encuentra los 3 mensajes en orden
    msgs = store.messages("573001112233")
    assert len(msgs) == 3
    assert [m.wamid for m in msgs] == ["wamid.A", "wamid.B", "wamid.C"]

    # last_inbound devuelve t2 (ignora la salida t1)
    ult = store.last_inbound("+57 300 111 2233")
    assert ult == t2
