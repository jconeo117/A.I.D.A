from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from aida.arms.agenda.app import AgendaArm, load_agenda_wording
from aida.arms.agenda.store import SqliteAgendaStore
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
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]]
) -> None:
    arm, ctx, _, _ = arm_contexto
    resp = arm.get("dia/2026-10-20", ctx)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["day"] == "2026-10-20"
    for k in ("slot_minutes", "offset", "services", "resources", "appointments", "week", "free"):
        assert k in data


def test_post_accion_invalida_400_y_sin_evento_en_log(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]]
) -> None:
    arm, ctx, _, _ = arm_contexto
    r1 = arm.post("accion", b"{roto", {}, ctx)
    assert r1 is not None and r1.status == 400
    r2 = arm.post("accion", json.dumps({"arguments": {}}).encode(), {}, ctx)
    assert r2 is not None and r2.status == 400
    assert len(list(arm._log)) == 0


def test_post_accion_done_audita_agenda_changed(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]]
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
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]]
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
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]]
) -> None:
    arm, ctx, _, _ = arm_contexto
    assert arm.at_rest() is True
    arm.post("actividad", json.dumps({"dirty": True}).encode(), {}, ctx)
    assert arm.at_rest() is False
    arm.post("actividad", json.dumps({"dirty": False}).encode(), {}, ctx)
    assert arm.at_rest() is True
