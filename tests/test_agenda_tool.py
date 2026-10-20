from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from aida.arms.agenda.setup import AgendaSetup, ResourceSpec, ServiceSpec, load_setup
from aida.arms.agenda.store import SqliteAgendaStore
from aida.arms.agenda.tool import AgendaTool, find_clients, free_slots, week_summary
from aida.engine.actions import ToolStatus
from aida.resources import data_root

TZ = timezone(timedelta(hours=-5))


@pytest.fixture
def setup_prueba() -> AgendaSetup:
    servicios = (
        ServiceSpec(id="general", label="Consulta General", minutes=30),
        ServiceSpec(id="especial", label="Consulta Especial", minutes=60),
    )
    horario = {dia: ((time(8, 0), time(12, 0)),) for dia in ("mon", "tue", "wed", "thu", "fri")}
    horario["sat"] = ((time(9, 0), time(12, 0)),)
    recursos = (
        ResourceSpec(
            id="medico_1",
            label="Dr. Uno",
            place="Box 1",
            services=("general", "especial"),
            hours=horario,
        ),
        ResourceSpec(
            id="medico_2",
            label="Dra. Dos",
            place="Box 2",
            services=("general",),
            hours=horario,
        ),
    )
    return AgendaSetup(
        sector="salud",
        slot_minutes=30,
        client_label="Paciente",
        services=servicios,
        resources=recursos,
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
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    r1 = tool.execute(
        action="agendar", arguments={"name": "", "phone": "123", "service": "general"}
    )
    assert r1.status == ToolStatus.REJECTED and r1.reason == "missing_client"


def test_agendar_unknown_service(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    r = tool.execute(
        action="agendar",
        arguments={"name": "Ana", "phone": "123", "service": "inexistente", "by": "u"},
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "unknown_service"


def test_agendar_bad_argument_fecha_invalida_o_sin_zona(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    r = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "start": "2026-10-20T08:00:00",
            "by": "u",
        },
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "bad_argument"


def test_agendar_unknown_resource(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    inicio = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "fantasma",
            "start": inicio,
            "by": "u",
        },
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "unknown_resource"


def test_agendar_service_not_offered(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    inicio = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "especial",
            "resource": "medico_2",
            "start": inicio,
            "by": "u",
        },
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "service_not_offered"


def test_agendar_past(agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]]) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 10, 0, tzinfo=TZ)
    pasado = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r = tool.execute(
        action="agendar",
        arguments={"name": "Ana", "phone": "123", "service": "general", "start": pasado, "by": "u"},
    )
    assert r.status == ToolStatus.REJECTED and r.reason == "past"


def test_agendar_outside_hours(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    fuera = datetime(2026, 10, 20, 12, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={"name": "Ana", "phone": "123", "service": "general", "start": fuera, "by": "u"},
    )
    assert r1.status == ToolStatus.REJECTED and r1.reason == "outside_hours"


def test_solapes_en_los_bordes_y_overlap(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    assert r1.status == ToolStatus.DONE

    t_0930 = datetime(2026, 10, 20, 9, 30, tzinfo=TZ).isoformat()
    r2 = tool.execute(
        action="agendar",
        arguments={
            "name": "Beto",
            "phone": "456",
            "service": "general",
            "resource": "medico_1",
            "start": t_0930,
            "by": "u",
        },
    )
    assert r2.status == ToolStatus.DONE

    t_0830 = datetime(2026, 10, 20, 8, 30, tzinfo=TZ).isoformat()
    r3 = tool.execute(
        action="agendar",
        arguments={
            "name": "Carlos",
            "phone": "789",
            "service": "general",
            "resource": "medico_1",
            "start": t_0830,
            "by": "u",
        },
    )
    assert r3.status == ToolStatus.DONE

    r4 = tool.execute(
        action="agendar",
        arguments={
            "name": "Diana",
            "phone": "000",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    assert r4.status == ToolStatus.REJECTED and r4.reason == "overlap"


def test_cancelada_no_genera_solape(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    cita_id = r1.data["appointment"]

    tool.execute(action="cancelar", arguments={"appointment": cita_id, "by": "u"})

    r2 = tool.execute(
        action="agendar",
        arguments={
            "name": "Beto",
            "phone": "456",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    assert r2.status == ToolStatus.DONE


def test_transiciones_de_estado_solo_desde_booked(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
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
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    cita_id = r1.data["appointment"]

    ra = tool.execute(action="atendida", arguments={"appointment": cita_id, "by": "u"})
    assert ra.status == ToolStatus.REJECTED and ra.reason == "not_yet"

    reloj[0] = datetime(2026, 10, 20, 9, 5, tzinfo=TZ)
    ra2 = tool.execute(action="atendida", arguments={"appointment": cita_id, "by": "u"})
    assert ra2.status == ToolStatus.DONE


def test_reprogramar_no_se_cuenta_a_si_misma(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    cita_id = r1.data["appointment"]

    r_rep = tool.execute(
        action="reprogramar",
        arguments={"appointment": cita_id, "start": t_0900, "resource": "medico_1", "by": "u"},
    )
    assert r_rep.status == ToolStatus.DONE


def test_agendar_resource_vacio_asigna_primer_libre(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, store, _ = agenda_tool
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    r = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "",
            "start": t_0900,
            "by": "u",
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

    huecos_pv = free_slots(st_cons, setup_cons, date(2026, 10, 20), "primera_vez", ahora_cons)
    recursos_pv = {r for r, _ in huecos_pv}
    assert "consultorio_2" not in recursos_pv
    assert "consultorio_1" in recursos_pv

    pack_pelu = data_root() / "packs" / "agenda" / "peluqueria" / "agenda.yaml"
    setup_pelu = load_setup(pack_pelu)
    assert setup_pelu.slot_minutes == 15
    st_pelu = SqliteAgendaStore(tmp_path / "pelu.sqlite", Fernet.generate_key())
    ahora_pelu = datetime(2026, 10, 24, 7, 0, tzinfo=TZ)

    huecos_sab = free_slots(st_pelu, setup_pelu, date(2026, 10, 24), "corte", ahora_pelu)
    assert len(huecos_sab) > 0

    lunes = date(2026, 10, 19)
    res_sem = week_summary(st_pelu, setup_pelu, lunes, ahora_pelu)
    assert len(res_sem) == 6


def test_agendar_en_bloqueo_de_su_recurso_blocked(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    t_1000 = datetime(2026, 10, 20, 10, 0, tzinfo=TZ).isoformat()

    rb = tool.execute(
        action="bloquear",
        arguments={
            "resource": "medico_1",
            "start": t_0900,
            "end": t_1000,
            "reason": "Junta",
            "by": "u",
        },
    )
    assert rb.status == ToolStatus.DONE

    ra = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    assert ra.status == ToolStatus.REJECTED and ra.reason == "blocked"


def test_bloqueo_de_todos_bloquea_cada_recurso(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    t_1000 = datetime(2026, 10, 20, 10, 0, tzinfo=TZ).isoformat()

    rb = tool.execute(
        action="bloquear",
        arguments={"resource": "", "start": t_0900, "end": t_1000, "reason": "Feriado", "by": "u"},
    )
    assert rb.status == ToolStatus.DONE

    ra1 = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    assert ra1.status == ToolStatus.REJECTED and ra1.reason == "blocked"

    ra2 = tool.execute(
        action="agendar",
        arguments={
            "name": "Beto",
            "phone": "456",
            "service": "general",
            "resource": "medico_2",
            "start": t_0900,
            "by": "u",
        },
    )
    assert ra2.status == ToolStatus.REJECTED and ra2.reason == "blocked"


def test_bloqueo_de_otro_recurso_no_afecta(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    t_1000 = datetime(2026, 10, 20, 10, 0, tzinfo=TZ).isoformat()

    rb = tool.execute(
        action="bloquear",
        arguments={
            "resource": "medico_2",
            "start": t_0900,
            "end": t_1000,
            "reason": "Permiso",
            "by": "u",
        },
    )
    assert rb.status == ToolStatus.DONE

    ra = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    assert ra.status == ToolStatus.DONE


def test_orden_outside_hours_antes_que_blocked(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, store, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 6, 0, tzinfo=TZ)
    t_0600 = datetime(2026, 10, 20, 6, 0, tzinfo=TZ)
    t_0700 = datetime(2026, 10, 20, 7, 0, tzinfo=TZ)

    store.add_block(resource="medico_1", start=t_0600, end=t_0700, reason="Madrugada", by="u")

    ra = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0600.isoformat(),
            "by": "u",
        },
    )
    assert ra.status == ToolStatus.REJECTED and ra.reason == "outside_hours"


def test_reprogramar_a_bloqueo_blocked(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0800 = datetime(2026, 10, 20, 8, 0, tzinfo=TZ).isoformat()
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    t_1000 = datetime(2026, 10, 20, 10, 0, tzinfo=TZ).isoformat()

    ra = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0800,
            "by": "u",
        },
    )
    cita_id = ra.data["appointment"]

    tool.execute(
        action="bloquear",
        arguments={
            "resource": "medico_1",
            "start": t_0900,
            "end": t_1000,
            "reason": "Junta",
            "by": "u",
        },
    )

    rr = tool.execute(
        action="reprogramar",
        arguments={"appointment": cita_id, "start": t_0900, "resource": "medico_1", "by": "u"},
    )
    assert rr.status == ToolStatus.REJECTED and rr.reason == "blocked"


def test_free_slots_excluye_lo_bloqueado(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
    setup_prueba: AgendaSetup,
) -> None:
    _, store, reloj = agenda_tool
    dia = date(2026, 10, 20)
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)

    huecos_antes = free_slots(store, setup_prueba, dia, "general", reloj[0])
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ)
    t_0930 = datetime(2026, 10, 20, 9, 30, tzinfo=TZ)
    assert ("medico_1", t_0900) in huecos_antes
    assert ("medico_1", t_0930) in huecos_antes

    store.add_block(
        resource="medico_1",
        start=t_0900,
        end=datetime(2026, 10, 20, 10, 0, tzinfo=TZ),
        reason="Pausa",
        by="u",
    )

    huecos_despues = free_slots(store, setup_prueba, dia, "general", reloj[0])
    assert ("medico_1", t_0900) not in huecos_despues
    assert ("medico_1", t_0930) not in huecos_despues
    assert ("medico_2", t_0900) in huecos_despues


def test_week_summary_resta_capacidad_bloqueada(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
    setup_prueba: AgendaSetup,
) -> None:
    _, store, _ = agenda_tool
    lunes = date(2026, 10, 19)
    ahora = datetime(2026, 10, 19, 7, 0, tzinfo=TZ)

    res_antes = week_summary(store, setup_prueba, lunes, ahora)
    assert res_antes[0] == (lunes, 0, 16)

    store.add_block(
        resource="",
        start=datetime(2026, 10, 19, 8, 0, tzinfo=TZ),
        end=datetime(2026, 10, 19, 9, 0, tzinfo=TZ),
        reason="Reunión general",
        by="u",
    )

    res_despues = week_summary(store, setup_prueba, lunes, ahora)
    assert res_despues[0] == (lunes, 0, 12)


def test_bloquear_con_cita_dentro_has_appointments(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, store, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    ra = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    cita_id = ra.data["appointment"]

    t_0830 = datetime(2026, 10, 20, 8, 30, tzinfo=TZ).isoformat()
    t_0930 = datetime(2026, 10, 20, 9, 30, tzinfo=TZ).isoformat()
    rb = tool.execute(
        action="bloquear",
        arguments={
            "resource": "medico_1",
            "start": t_0830,
            "end": t_0930,
            "reason": "Junta",
            "by": "u",
        },
    )
    assert rb.status == ToolStatus.REJECTED
    assert rb.reason == "has_appointments"
    assert rb.data["appointments"] == cita_id
    assert (
        len(
            store.blocks_between(
                datetime(2026, 10, 20, 8, 0, tzinfo=TZ), datetime(2026, 10, 20, 12, 0, tzinfo=TZ)
            )
        )
        == 0
    )


def test_bloquear_con_cita_cancelada_dentro_se_permite(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    ra = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    cita_id = ra.data["appointment"]
    tool.execute(action="cancelar", arguments={"appointment": cita_id, "by": "u"})

    t_0830 = datetime(2026, 10, 20, 8, 30, tzinfo=TZ).isoformat()
    t_0930 = datetime(2026, 10, 20, 9, 30, tzinfo=TZ).isoformat()
    rb = tool.execute(
        action="bloquear",
        arguments={
            "resource": "medico_1",
            "start": t_0830,
            "end": t_0930,
            "reason": "Junta",
            "by": "u",
        },
    )
    assert rb.status == ToolStatus.DONE


def test_bloquear_cita_que_empezo_antes_cuenta(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0930 = datetime(2026, 10, 20, 9, 30, tzinfo=TZ).isoformat()
    ra = tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "123",
            "service": "especial",
            "resource": "medico_1",
            "start": t_0930,
            "by": "u",
        },
    )
    assert ra.status == ToolStatus.DONE
    cita_id = ra.data["appointment"]

    t_1000 = datetime(2026, 10, 20, 10, 0, tzinfo=TZ).isoformat()
    t_1100 = datetime(2026, 10, 20, 11, 0, tzinfo=TZ).isoformat()
    rb = tool.execute(
        action="bloquear",
        arguments={
            "resource": "medico_1",
            "start": t_1000,
            "end": t_1100,
            "reason": "Junta",
            "by": "u",
        },
    )
    assert rb.status == ToolStatus.REJECTED
    assert rb.reason == "has_appointments"
    assert rb.data["appointments"] == cita_id


def test_bloquear_past_bad_argument_unknown_resource(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, _, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 10, 0, tzinfo=TZ)
    t_0800 = datetime(2026, 10, 20, 8, 0, tzinfo=TZ).isoformat()
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    t_1000 = datetime(2026, 10, 20, 10, 0, tzinfo=TZ).isoformat()
    t_1100 = datetime(2026, 10, 20, 11, 0, tzinfo=TZ).isoformat()

    r_past = tool.execute(
        action="bloquear",
        arguments={"start": t_0800, "end": t_0900, "by": "u"},
    )
    assert r_past.status == ToolStatus.REJECTED and r_past.reason == "past"

    r_bad = tool.execute(
        action="bloquear",
        arguments={"start": t_1100, "end": t_0900, "by": "u"},
    )
    assert r_bad.status == ToolStatus.REJECTED and r_bad.reason == "bad_argument"

    r_unk = tool.execute(
        action="bloquear",
        arguments={"resource": "fantasma", "start": t_1000, "end": t_1100, "by": "u"},
    )
    assert r_unk.status == ToolStatus.REJECTED and r_unk.reason == "unknown_resource"


def test_desbloquear_libera_y_unknown_block(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, store, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()
    t_1000 = datetime(2026, 10, 20, 10, 0, tzinfo=TZ).isoformat()

    rb = tool.execute(
        action="bloquear",
        arguments={
            "resource": "medico_1",
            "start": t_0900,
            "end": t_1000,
            "reason": "Junta",
            "by": "u",
        },
    )
    block_id = rb.data["block"]
    assert store.block(int(block_id)) is not None

    ru = tool.execute(action="desbloquear", arguments={"block": block_id})
    assert ru.status == ToolStatus.DONE and ru.data["block"] == block_id
    assert store.block(int(block_id)) is None

    ru_again = tool.execute(action="desbloquear", arguments={"block": block_id})
    assert ru_again.status == ToolStatus.REJECTED and ru_again.reason == "unknown_block"

    ru_inval = tool.execute(action="desbloquear", arguments={"block": "abc"})
    assert ru_inval.status == ToolStatus.REJECTED and ru_inval.reason == "unknown_block"


def test_editar_cliente_done_unknown_missing_phone_taken(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, store, _ = agenda_tool
    c1 = store.add_client(name="Ana", phone="3001112233")
    store.add_client(name="Beto", phone="3002223344")

    r_ok = tool.execute(
        action="editar_contacto",
        arguments={"client": str(c1.id), "name": "Ana María", "phone": "3001119999"},
    )
    assert r_ok.status == ToolStatus.DONE and r_ok.data["client"] == str(c1.id)
    assert store.client(c1.id).name == "Ana María"
    assert store.client(c1.id).phone == "3001119999"

    r_unk = tool.execute(
        action="editar_contacto",
        arguments={"client": "99999", "name": "X", "phone": "123"},
    )
    assert r_unk.status == ToolStatus.REJECTED and r_unk.reason == "unknown_client"

    r_mis = tool.execute(
        action="editar_contacto",
        arguments={"client": str(c1.id), "name": "", "phone": "3001119999"},
    )
    assert r_mis.status == ToolStatus.REJECTED and r_mis.reason == "missing_client"

    r_taken = tool.execute(
        action="editar_contacto",
        arguments={"client": str(c1.id), "name": "Ana", "phone": "3002223344"},
    )
    assert r_taken.status == ToolStatus.REJECTED and r_taken.reason == "phone_taken"


def test_editar_con_telefono_ajeno_phone_taken_sin_mutar(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    tool, store, reloj = agenda_tool
    reloj[0] = datetime(2026, 10, 20, 8, 0, tzinfo=TZ)
    t_0900 = datetime(2026, 10, 20, 9, 0, tzinfo=TZ).isoformat()

    tool.execute(
        action="agendar",
        arguments={
            "name": "Ana",
            "phone": "3001112233",
            "service": "general",
            "resource": "medico_1",
            "start": t_0900,
            "by": "u",
        },
    )
    r2 = tool.execute(
        action="agendar",
        arguments={
            "name": "Beto",
            "phone": "3002223344",
            "service": "general",
            "resource": "medico_2",
            "start": t_0900,
            "by": "u",
        },
    )
    cita2_id = r2.data["appointment"]

    r_edit = tool.execute(
        action="editar",
        arguments={
            "appointment": cita2_id,
            "name": "Beto Editado",
            "phone": "3001112233",
            "by": "u",
        },
    )
    assert r_edit.status == ToolStatus.REJECTED
    assert r_edit.reason == "phone_taken"

    cita2 = store.get(int(cita2_id))
    assert cita2 is not None
    cliente2 = store.client(cita2.client)
    assert cliente2.name == "Beto"
    assert cliente2.phone == "3002223344"


def test_find_clients_por_nombre_sin_tildes_y_por_telefono(
    agenda_tool: tuple[AgendaTool, SqliteAgendaStore, list[datetime]],
) -> None:
    _, store, _ = agenda_tool
    c1 = store.add_client(name="Gómez", phone="3005550101")
    c2 = store.add_client(name="gomez", phone="3005550102")
    store.add_client(name="Carlos Giménez", phone="3005550103")

    hallados_gomez = find_clients(store, "gomez")
    assert len(hallados_gomez) == 2
    assert {c.id for c in hallados_gomez} == {c1.id, c2.id}

    hallados_tel = find_clients(store, "555")
    assert len(hallados_tel) == 3

    assert find_clients(store, "   ") == ()
