from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet

from aida.platform.channel.store import (
    Direction,
    SqliteChannelStore,
    national,
    to_whatsapp,
)


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


def test_to_whatsapp_y_national() -> None:
    # con y sin prefijo, con + y espacios, país vacío
    assert to_whatsapp("3001112233", "57") == "573001112233"
    assert to_whatsapp("+57 300 111 2233", "57") == "573001112233"
    assert to_whatsapp("573001112233", "57") == "573001112233"
    assert to_whatsapp("3001112233", "") == "3001112233"
    assert to_whatsapp("+57 3001112233", "") == "573001112233"

    assert national("573001112233", "57") == "3001112233"
    assert national("+57 300 111 2233", "57") == "3001112233"
    assert national("3001112233", "57") == "3001112233"
    assert national("573001112233", "") == "573001112233"


def test_mismo_contacto_con_y_sin_prefijo_es_una_conversacion(tmp_path: Path) -> None:
    clave = Fernet.generate_key()
    store = SqliteChannelStore(tmp_path / "ch.sqlite", clave, country="57")
    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)

    # Entra con prefijo nacional desde Meta
    store.add_inbound(
        wamid="wamid.1",
        phone="573001112233",
        name="Ana",
        kind="text",
        text="Hola desde Meta",
        at=t0,
    )

    # El mostrador busca sin prefijo: lo encuentra
    msgs = store.messages("3001112233")
    assert len(msgs) == 1
    assert msgs[0].wamid == "wamid.1"

    # Una sola conversación agrupada
    convos = store.conversations()
    assert len(convos) == 1
    assert convos[0].phone == "573001112233"


def test_conversations_orden_nombre_y_no_leidos(tmp_path: Path) -> None:
    clave = Fernet.generate_key()
    store = SqliteChannelStore(tmp_path / "ch.sqlite", clave, country="57")
    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=10)
    t2 = t0 + timedelta(minutes=20)

    # Contacto 1: primero con nombre, luego otro mensaje sin nombre
    m1 = store.add_inbound(
        wamid="w.c1.1", phone="573001112233", name="Ana Gómez", kind="text", text="Hola", at=t0
    )
    assert m1 is not None
    store.add_outbound(
        wamid="w.c1.out", phone="573001112233", text="Respuesta", at=t1, by="prof-1"
    )
    store.add_inbound(
        wamid="w.c1.2", phone="573001112233", name="", kind="text", text="Gracias", at=t1
    )

    # Contacto 2: mensaje más reciente a t2
    store.add_inbound(
        wamid="w.c2.1", phone="573009998877", name="Beto", kind="text", text="Consulta", at=t2
    )

    convos = store.conversations()
    assert len(convos) == 2

    # Más reciente primero (Contacto 2 at t2)
    assert convos[0].phone == "573009998877"
    assert convos[0].name == "Beto"
    assert convos[0].unread == 1

    # Contacto 1: conserva el último nombre no vacío y unread solo cuenta IN (2 inbounds)
    assert convos[1].phone == "573001112233"
    assert convos[1].name == "Ana Gómez"
    assert convos[1].unread == 2


def test_mark_read_no_retrocede_y_conversacion_inexistente(tmp_path: Path) -> None:
    clave = Fernet.generate_key()
    store = SqliteChannelStore(tmp_path / "ch.sqlite", clave, country="57")
    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)

    # Conversación inexistente da False
    assert not store.mark_read("573000000000", up_to=10, by="prof-1", at=t0)

    # Conversación con 3 mensajes entrantes
    m1 = store.add_inbound(
        wamid="w.m1", phone="573001112233", name="Ana", kind="text", text="1", at=t0
    )
    assert m1 is not None
    m2 = store.add_inbound(
        wamid="w.m2",
        phone="573001112233",
        name="Ana",
        kind="text",
        text="2",
        at=t0 + timedelta(seconds=1),
    )
    assert m2 is not None
    m3 = store.add_inbound(
        wamid="w.m3",
        phone="573001112233",
        name="Ana",
        kind="text",
        text="3",
        at=t0 + timedelta(seconds=2),
    )
    assert m3 is not None

    assert store.conversations()[0].unread == 3

    # Marcar leída hasta m2 (id de m2)
    ok = store.mark_read("573001112233", up_to=m2.id, by="prof-1", at=t0)
    assert ok is True
    assert store.conversations()[0].unread == 1  # Solo queda m3

    # Intentar retroceder a m1 no tiene efecto (no retrocede)
    store.mark_read("573001112233", up_to=m1.id, by="prof-1", at=t0)
    assert store.conversations()[0].unread == 1

