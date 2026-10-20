from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from aida.arms.agenda.setup import AgendaSetup, ResourceSpec, ServiceSpec, load_setup
from aida.arms.agenda.store import SqliteAgendaStore
from aida.arms.agenda.tool import AgendaTool, huecos_libres, resumen_semana
from aida.engine.actions import ToolStatus
from aida.resources import data_root

TZ = timezone(timedelta(hours=-5))


@pytest.fixture
def setup_prueba() -> AgendaSetup:
    servicios = (
        ServiceSpec(id="general", label="Consulta General", minutes=30),
        ServiceSpec(id="especial", label="Consulta Especial", minutes=60),
    )
    horario = {
        dia: ((time(8, 0), time(12, 0)),) for dia in ("mon", "tue", "wed", "thu", "fri")
    }
    horario["sat"] = ((time(9, 0), time(12, 0)),)
    recursos = (
        ResourceSpec(
            id="medico_1", label="Dr. Uno", place="Box 1",
            services=("general", "especial"), hours=horario,
        ),
        ResourceSpec(
            id="medico_2", label="Dra. Dos", place="Box 2",
            services=("general",), hours=horario,
        ),
    )
    return AgendaSetup(
        sector="salud", slot_minutes=30, client_label="Paciente",
        services=servicios, resources=recursos,
    )


@pytest.fixture
def agenda_tool(
    tmp_path: Path, setup_prueba: AgendaSetup
) -> tuple[AgendaTool, SqliteAgendaStore, list[datetime]]:
    store = SqliteAgendaStore(tmp_path / "agenda.sqlite", Fernet.generate_key())
    reloj = [datetime(2026, 10, 20, 7, 0, tzinfo=TZ)]
    tool = AgendaTool(store=store, setup=setup_prueba, now=lambda: reloj[0])
    return tool, store, reloj


def test_agendar_missing_client(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    r1 = tool.execute(
        action="agendar", arguments={"name": "", "phone": "123", "service": "general"}
    )
    assert r1.status == ToolStatus.REJECTED and r1.reason == "missing_client"


def test_agendar_unknown_service(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    r = tool.execute(
        action="agendar",
        arguments={"name": "Ana", "phone": "123", "service": "inexistente", "by": "u"},
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "unknown_service"


def test_agendar_bad_argument_fecha_invalida_o_sin_zona(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    r = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana", "phone": "123", "service": "general",
            "start": "2026-10-20T08:00:00", "by": "u",
        },
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "bad_argument"


def test_agendar_unknown_resource(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    inicio = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana", "phone": "123", "service": "general",
            "resource": "fantasma", "start": inicio, "by": "u",
        },
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "unknown_resource"


def test_agendar_service_not_offered(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    inicio = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana", "phone": "123", "service": "especial",
            "resource": "medico_2", "start": inicio, "by": "u",
        },
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "service_not_offered"


def test_agendar_past(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 10, 0, tzinfo=TZ)
    pasado = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r = tool.execute(
        action="agendar",
        arguments={"name": "Ana", "phone": "123", "service": "general", "start": pasado, "by": "u"},
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "past"


def test_agendar_outside_hours(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    fuera = datetime(2026, 10, 20, 12, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={"name": "Ana", "phone": "123", "service": "general", "start": fuera, "by": "u"},
    )
    assert r1.status == ToolStatus.REJECTED and r1.reason == "outside_hours"


def test_solapes_en_los_bordes_y_overlap(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana", "phone": "123", "service": "general",
            "resource": "medico_1", "start": t_0900, "by": "u",
        },
    )
    assert r1.status == ToolStatus.DONE

    t_0930 = datetime(2026, 10, 20, 9, 30, tzinfo=TZ).isoformat()
    r2 = tool.execute(
        action="agendar",
        arguments={
            "name": "Beto", "phone": "456", "service": "general",
            "resource": "medico_1", "start": t_0930, "by": "u",
        },
    )
    assert r2.status == ToolStatus.DONE

    t_0830 = datetime(2026, 10, 20, 8, 30, tzinfo=TZ).isoformat()
    r3 = tool.execute(
        action="agendar",
        arguments={
            "name": "Carlos", "phone": "789", "service": "general",
            "resource": "medico_1", "start": t_0830, "by": "u",
        },
    )
    assert r3.status == ToolStatus.DONE

    r4 = tool.execute(
        action="agendar",
        arguments={
            "name": "Diana", "phone": "000", "service": "general",
            "resource": "medico_1", "start": t_0900, "by": "u",
        },
    )
    assert r4.status == ToolStatus.REJECTED and r4.reason == "overlap"


def test_cancelada_no_genera_solape(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana", "phone": "123", "service": "general",
            "resource": "medico_1", "start": t_0900, "by": "u",
        },
    )
    cita_id = r1.data["appointment"]

    tool.execute(action="cancelar", arguments={"appointment": cita_id, "by": "u"})

    r2 = tool.execute(
        action="agendar",
        arguments={
            "name": "Beto", "phone": "456", "service": "general",
            "resource": "medico_1", "start": t_0900, "by": "u",
        },
    )
    assert r2.status == ToolStatus.DONE


def test_transiciones_de_estado_solo_desde_booked(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana", "phone": "123", "service": "general",
            "resource": "medico_1", "start": t_0900, "by": "u",
        },
    )
    cita_id = r1.data["appointment"]

    tool.execute(action="cancelar", arguments={"appointment": cita_id, "by": "u"})

    rc = tool.execute(action="cancelar", arguments={"appointment": cita_id, "by": "u"})
    assert rc.status == ToolStatus.REJECTED and rc.reason == "not_booked"

    re = tool.execute(
        action="editar",
        arguments={"appointment": cita_id, "name": "Otro", "by": "u"},
    )
    assert re.status == ToolStatus.REJECTED and re.reason == "not_booked"

    t_1000 = datetime(2026, 10, 20, 10, 0, tzinfo=TZ).isoformat()
    rm = tool.execute(
        action="reprogramar",
        arguments={"appointment": cita_id, "start": t_1000, "by": "u"},
    )
    assert rm.status == ToolStatus.REJECTED and rm.reason == "not_booked"


def test_not_yet_antes_de_la_hora(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana", "phone": "123", "service": "general",
            "resource": "medico_1", "start": t_0900, "by": "u",
        },
    )
    cita_id = r1.data["appointment"]

    ra = tool.execute(action="atendida", arguments={"appointment": cita_id, "by": "u"})
    assert ra.status == ToolStatus.REJECTED and ra.reason == "not_yet"

    reloj[0] = datetime(2026, 10, 20, 9, 5, tzinfo=TZ)
    ra2 = tool.execute(action="atendida", arguments={"appointment": cita_id, "by": "u"})
    assert ra2.status == ToolStatus.DONE


def test_reprogramar_no_se_cuenta_a_si_misma(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, _, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana", "phone": "123", "service": "general",
            "resource": "medico_1", "start": t_0900, "by": "u",
        },
    )
    cita_id = r1.data["appointment"]

    r_rep = tool.execute(
        action="reprogramar",
        arguments={"appointment": cita_id, "start": t_0900, "resource": "medico_1", "by": "u"},
    )
    assert r_rep.status == ToolStatus.DONE


def test_agendar_resource_vacio_asigna_primer_libre(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]
) -> None:
    tool, store, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana", "phone": "123", "service": "general",
            "resource": "", "start": t_0900, "by": "u",
        },
    )
    assert r.status == ToolStatus.DONE
    cita = store.get(int(r.data["appointment"]))
    assert cita is not None and cita.resource == "medico_1"


def test_plantillas_consultorio_y_peluqueria(tmp_path: Path) -> None:
    pack_cons = data_root() / "packs" / "agenda" / "consultorio" / "agenda.yaml"
    setup_cons = load_setup(pack_cons)
    st_cons = SqliteAgendaStore(tmp_path / "cons.sqlite", Fernet.generate_key())
    ahora_cons = datetime(2026, 10, 20, 7, 0, tzinfo=TZ)

    huecos_pv = huecos_libres(st_cons, setup_cons, date(2026, 10, 20), "primera_vez", ahora_cons)
    recursos_pv = {r for r, _ in huecos_pv}
    assert "consultorio_2" not in recursos_pv
    assert "consultorio_1" in recursos_pv

    pack_pelu = data_root() / "packs" / "agenda" / "peluqueria" / "agenda.yaml"
    setup_pelu = load_setup(pack_pelu)
    assert setup_pelu.slot_minutes == 15
    st_pelu = SqliteAgendaStore(tmp_path / "pelu.sqlite", Fernet.generate_key())
    ahora_pelu = datetime(2026, 10, 24, 7, 0, tzinfo=TZ)

    huecos_sab = huecos_libres(st_pelu, setup_pelu, date(2026, 10, 24), "corte", ahora_pelu)
    assert len(huecos_sab) > 0

    lunes = date(2026, 10, 19)
    res_sem = resumen_semana(st_pelu, setup_pelu, lunes, ahora_pelu)
    assert len(res_sem) == 6
