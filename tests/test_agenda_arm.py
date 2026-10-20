from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from aida.arms.agenda.app import AgendaArm, load_agenda_wording
from aida.arms.agenda.store import AppointmentState, SqliteAgendaStore
from aida.platform.audit import AuditLog
from aida.platform.identity import Professional
from aida.platform.license import License, LicenseState
from aida.platform.web.shell import ShellContext
from aida.platform.web.wording import load_shell_wording

TZ = timezone(timedelta(hours=-5))


@pytest.fixture
def arm_contexto(tmp_path: Path) -> tuple[AgendaArm, ShellContext, AuditLog, list[datetime]]:
    store = SqliteAgendaStore(tmp_path / "agenda.sqlite", Fernet.generate_key())
    reloj = [datetime(2026, 10, 20, 8, 0, tzinfo=TZ)]
    lock = threading.Lock()
    wording = load_agenda_wording()
    arm = AgendaArm(
        arm_id="agenda",
        name="Agenda",
        setup_path=None,
        store=store,
        now=lambda: reloj[0],
        wording=wording,
        lock=lock,
    )
    prof = Professional(id="p1", name="Dra. Gómez", registry="12345")
    auditoria = AuditLog(tmp_path / "audit.jsonl")
    lic = LicenseState(
        status="valid",
        license=License(
            id="lic-test",
            client="Test",
            arms=("agenda", "health"),
            issued=date.today(),
            expires=date.today() + timedelta(days=365),
        ),
    )
    shell_words = load_shell_wording()
    ctx = ShellContext(
        token="tok123",
        brand="AIDA",
        now=reloj[0],
        professional=prof,
        license=lic,
        audit=auditoria,
        wording=shell_words,
        lock=lock,
        profiles=(prof,),
        launcher_link="",
    )
    return arm, ctx, auditoria, reloj


def test_get_dia_devuelve_estructura_literal(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, _ = arm_contexto
    resp = arm.get("dia/2026-10-20", ctx)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["day"] == "2026-10-20"
    for k in ("slot_minutes", "offset", "services", "resources", "appointments", "week", "free"):
        assert k in data


def test_post_accion_invalida_400_y_sin_evento_en_log(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, _ = arm_contexto
    r1 = arm.post("accion", b"{roto", {}, ctx)
    assert r1 is not None and r1.status == 400
    r2 = arm.post("accion", json.dumps({"arguments": {}}).encode(), {}, ctx)
    assert r2 is not None and r2.status == 400
    assert len(list(arm._log)) == 0


def test_post_accion_done_audita_agenda_changed(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, audit, reloj = arm_contexto
    inicio = (reloj[0] + timedelta(hours=1)).isoformat()
    body = {
        "action": "agendar",
        "arguments": {
            "name": "Carlos Ruiz",
            "phone": "3001112233",
            "service": arm._setup.services[0].id,
            "resource": arm._setup.resources[0].id,
            "start": inicio,
        },
    }
    resp = arm.post("accion", json.dumps(body).encode(), {}, ctx)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["status"] == "done"

    lineas_audit = audit.lines()
    assert len(lineas_audit) == 1
    assert lineas_audit[0]["kind"] == "agenda_changed"
    assert lineas_audit[0]["actor"] == "p1"
    assert json.loads(lineas_audit[0]["data"])["action"] == "agendar"


def test_post_accion_rejected_sin_auditoria_y_con_reason_text(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, audit, _ = arm_contexto
    body = {"action": "agendar", "arguments": {"name": "", "phone": "", "service": "invalido"}}
    resp = arm.post("accion", json.dumps(body).encode(), {}, ctx)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["status"] == "rejected"
    assert data["reason"] == "missing_client"
    assert data["reason_text"] != ""
    assert len(audit.lines()) == 0


def test_post_actividad_marca_at_rest(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, _ = arm_contexto
    assert arm.at_rest() is True
    arm.post("actividad", json.dumps({"dirty": True}).encode(), {}, ctx)
    assert arm.at_rest() is False
    arm.post("actividad", json.dumps({"dirty": False}).encode(), {}, ctx)
    assert arm.at_rest() is True


def test_get_dia_trae_blocks_y_client_id(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, reloj = arm_contexto
    inicio = (reloj[0] + timedelta(hours=1)).isoformat()
    fin = (reloj[0] + timedelta(hours=2)).isoformat()

    r_cita = arm.post(
        "accion",
        json.dumps(
            {
                "action": "agendar",
                "arguments": {
                    "name": "Ana",
                    "phone": "3001112233",
                    "service": arm._setup.services[0].id,
                    "resource": arm._setup.resources[0].id,
                    "start": inicio,
                },
            }
        ).encode(),
        {},
        ctx,
    )
    assert r_cita is not None and r_cita.status == 200

    r_block = arm.post(
        "accion",
        json.dumps(
            {
                "action": "bloquear",
                "arguments": {
                    "resource": arm._setup.resources[1].id,
                    "start": inicio,
                    "end": fin,
                    "reason": "Capacitación",
                },
            }
        ).encode(),
        {},
        ctx,
    )
    assert r_block is not None and r_block.status == 200

    resp = arm.get("dia/2026-10-20", ctx)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert "blocks" in data
    assert len(data["blocks"]) == 1
    assert data["blocks"][0]["reason"] == "Capacitación"
    assert data["blocks"][0]["resource"] == arm._setup.resources[1].id

    assert len(data["appointments"]) == 1
    assert "client_id" in data["appointments"][0]
    assert isinstance(data["appointments"][0]["client_id"], int)


def test_bloquear_audita_con_owner_del_bloqueo(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, audit, reloj = arm_contexto
    inicio = (reloj[0] + timedelta(hours=1)).isoformat()
    fin = (reloj[0] + timedelta(hours=2)).isoformat()

    resp = arm.post(
        "accion",
        json.dumps(
            {
                "action": "bloquear",
                "arguments": {
                    "resource": arm._setup.resources[0].id,
                    "start": inicio,
                    "end": fin,
                    "reason": "Mantenimiento",
                },
            }
        ).encode(),
        {},
        ctx,
    )
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["status"] == "done"
    block_id = data["data"]["block"]

    lineas = audit.lines()
    assert len(lineas) == 1
    assert lineas[0]["kind"] == "agenda_changed"
    assert ctx.professional is not None
    assert lineas[0]["actor"] == ctx.professional.id
    assert lineas[0]["owner"] == block_id
    payload = json.loads(lineas[0]["data"])
    assert payload["action"] == "bloquear"
    assert payload["start"] == inicio


def test_has_appointments_devuelve_conflicts_con_nombre(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, reloj = arm_contexto
    inicio = (reloj[0] + timedelta(hours=1)).isoformat()
    fin = (reloj[0] + timedelta(hours=2)).isoformat()

    arm.post(
        "accion",
        json.dumps(
            {
                "action": "agendar",
                "arguments": {
                    "name": "Carlos Ruiz",
                    "phone": "3001112233",
                    "service": arm._setup.services[0].id,
                    "resource": arm._setup.resources[0].id,
                    "start": inicio,
                },
            }
        ).encode(),
        {},
        ctx,
    )

    resp = arm.post(
        "accion",
        json.dumps(
            {
                "action": "bloquear",
                "arguments": {
                    "resource": arm._setup.resources[0].id,
                    "start": inicio,
                    "end": fin,
                    "reason": "Junta",
                },
            }
        ).encode(),
        {},
        ctx,
    )
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["status"] == "rejected"
    assert data["reason"] == "has_appointments"
    assert "conflicts" in data
    assert len(data["conflicts"]) == 1
    assert data["conflicts"][0]["client"] == "Carlos Ruiz"
    assert data["conflicts"][0]["start"] == inicio


def test_buscar_post_200_y_cuerpo_invalido_400(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, _ = arm_contexto
    arm._store.add_client(name="Carlos Gómez", phone="3001234567")

    r_ok = arm.post("buscar", json.dumps({"query": "Carlos"}).encode(), {}, ctx)
    assert r_ok is not None and r_ok.status == 200
    data = json.loads(r_ok.body)
    assert len(data["clients"]) == 1
    assert data["clients"][0]["name"] == "Carlos Gómez"

    r_bad1 = arm.post("buscar", b"no-json", {}, ctx)
    assert r_bad1 is not None and r_bad1.status == 400
    r_bad2 = arm.post("buscar", json.dumps({"otra_clave": 123}).encode(), {}, ctx)
    assert r_bad2 is not None and r_bad2.status == 400


def test_get_cliente_ficha_con_cifras(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, reloj = arm_contexto
    ahora = reloj[0]
    cliente = arm._store.add_client(name="Lucía Morales", phone="3009998877")

    arm._store.add(
        client=cliente.id,
        service=arm._setup.services[0].id,
        resource=arm._setup.resources[0].id,
        start=ahora + timedelta(hours=2),
        minutes=30,
        by="prof-1",
    )
    arm._store.add(
        client=cliente.id,
        service=arm._setup.services[0].id,
        resource=arm._setup.resources[0].id,
        start=ahora - timedelta(days=1),
        minutes=30,
        by="prof-1",
    )
    c_atendida = arm._store.add(
        client=cliente.id,
        service=arm._setup.services[0].id,
        resource=arm._setup.resources[0].id,
        start=ahora - timedelta(days=2),
        minutes=30,
        by="prof-1",
    )
    arm._store.update(c_atendida.id, by="prof-1", state=AppointmentState.ATTENDED)

    c_noshow = arm._store.add(
        client=cliente.id,
        service=arm._setup.services[0].id,
        resource=arm._setup.resources[0].id,
        start=ahora - timedelta(days=3),
        minutes=30,
        by="prof-1",
    )
    arm._store.update(c_noshow.id, by="prof-1", state=AppointmentState.NO_SHOW)

    resp = arm.get(f"contacto/{cliente.id}", ctx)
    assert resp is not None and resp.status == 200
    ficha = json.loads(resp.body)
    assert ficha["id"] == cliente.id
    assert ficha["name"] == "Lucía Morales"
    assert ficha["phone"] == "3009998877"
    assert ficha["upcoming"] == 1
    assert ficha["attended"] == 1
    assert ficha["no_show"] == 1
    assert len(ficha["appointments"]) == 4


def test_get_cliente_inexistente_none(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, _ = arm_contexto
    assert arm.get("contacto/99999", ctx) is None
    assert arm.get("contacto/invalido", ctx) is None
