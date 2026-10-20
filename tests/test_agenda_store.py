from __future__ import annotations

import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from aida.arms.agenda.store import AgendaStore, AppointmentState, SqliteAgendaStore


@pytest.fixture
def clave_fernet() -> bytes:
    return Fernet.generate_key()


@pytest.fixture
def store(tmp_path: Path, clave_fernet: bytes) -> AgendaStore:
    return SqliteAgendaStore(tmp_path / "agenda.sqlite", clave_fernet)


def test_store_cumple_protocolo() -> None:
    # Verificamos estáticamente y por métodos que SqliteAgendaStore cumple AgendaStore
    st: AgendaStore = SqliteAgendaStore(Path("/tmp/dummy_test_store.sqlite"), Fernet.generate_key())
    for metodo in ("add_client", "client_by_phone", "client", "add", "update", "get", "between"):
        assert hasattr(st, metodo)
        assert callable(getattr(st, metodo))


def test_ida_y_vuelta_cliente_y_citas(store: AgendaStore) -> None:
    ahora = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    cliente = store.add_client(name="María Pérez", phone="3005550142")
    cita = store.add(
        client=cliente.id,
        service="consulta",
        resource="box_1",
        start=ahora,
        minutes=30,
        by="prof-1",
    )
    recuperado = store.client(cliente.id)
    assert recuperado is not None
    assert recuperado.id == cliente.id
    assert recuperado.name == "María Pérez"
    assert recuperado.phone == "3005550142"

    cita_rec = store.get(cita.id)
    assert cita_rec is not None
    assert cita_rec.id == cita.id
    assert cita_rec.service == "consulta"
    assert cita_rec.resource == "box_1"
    assert cita_rec.start == ahora
    assert cita_rec.minutes == 30
    assert cita_rec.state == AppointmentState.BOOKED
    assert cita_rec.created_by == "prof-1"


def test_client_by_phone_con_distinto_formato(store: AgendaStore) -> None:
    store.add_client(name="Juan Gómez", phone="300 555 0142")
    c1 = store.client_by_phone("3005550142")
    c2 = store.client_by_phone("300-555-0142")
    c3 = store.client_by_phone("300.555.0142")
    assert c1 is not None and c1.name == "Juan Gómez"
    assert c2 is not None and c2.name == "Juan Gómez"
    assert c3 is not None and c3.name == "Juan Gómez"


def test_between_ordenado_por_inicio_e_id(store: AgendaStore) -> None:
    c = store.add_client(name="Test", phone="1234567890")
    t1 = datetime(2026, 10, 20, 9, 0, tzinfo=UTC)
    t2 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    t3 = datetime(2026, 10, 20, 11, 0, tzinfo=UTC)

    cita_10_b = store.add(client=c.id, service="s", resource="r2", start=t2, minutes=30, by="u")
    cita_9 = store.add(client=c.id, service="s", resource="r1", start=t1, minutes=30, by="u")
    cita_10_a = store.add(client=c.id, service="s", resource="r1", start=t2, minutes=30, by="u")
    cita_11 = store.add(client=c.id, service="s", resource="r1", start=t3, minutes=30, by="u")

    rango = store.between(t1, t3)
    assert [x.id for x in rango] == [cita_9.id, cita_10_b.id, cita_10_a.id]
    assert cita_11.id not in [x.id for x in rango]


def test_archivo_sqlite_no_contiene_datos_en_claro(tmp_path: Path, clave_fernet: bytes) -> None:
    ruta = tmp_path / "agenda_cifrada.sqlite"
    st = SqliteAgendaStore(ruta, clave_fernet)
    nombre = "Paciente Super Secreto"
    telefono = "3119876543"
    st.add_client(name=nombre, phone=telefono)

    contenido_bytes = ruta.read_bytes()
    assert nombre.encode("utf-8") not in contenido_bytes
    assert telefono.encode("utf-8") not in contenido_bytes

    conn = sqlite3.connect(ruta)
    filas = conn.execute("SELECT name, phone FROM clients").fetchall()
    conn.close()
    assert len(filas) == 1
    blob_nombre, _blob_telefono = filas[0]
    assert isinstance(blob_nombre, bytes)
    assert nombre.encode("utf-8") != blob_nombre


def test_permisos_archivo_sqlite_son_0600(tmp_path: Path, clave_fernet: bytes) -> None:
    ruta = tmp_path / "agenda_permisos.sqlite"
    SqliteAgendaStore(ruta, clave_fernet)
    assert stat.S_IMODE(ruta.stat().st_mode) == 0o600


def test_clave_distinta_no_encuentra_cliente(tmp_path: Path, clave_fernet: bytes) -> None:
    ruta = tmp_path / "agenda_claves.sqlite"
    st1 = SqliteAgendaStore(ruta, clave_fernet)
    st1.add_client(name="Lucía", phone="3120001122")

    otra_clave = Fernet.generate_key()
    st2 = SqliteAgendaStore(ruta, otra_clave)
    assert st2.client_by_phone("3120001122") is None


def test_clave_indice_derivada_distinta_de_clave_fernet(
    tmp_path: Path, clave_fernet: bytes
) -> None:
    ruta = tmp_path / "agenda_derivacion.sqlite"
    st = SqliteAgendaStore(ruta, clave_fernet)
    assert st._clave_indice != clave_fernet
