from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import yaml
from cryptography.fernet import Fernet

from aida.arms.agenda.app import AgendaArm, load_agenda_wording
from aida.arms.agenda.store import AppointmentState, SqliteAgendaStore
from aida.engine.actions import ActionGate
from aida.engine.events import EventLog
from aida.keys import whatsapp_token
from aida.platform.audit import AuditLog
from aida.platform.channel import LICENSE_ID, Channel
from aida.platform.channel.config import ChannelConfig
from aida.platform.channel.poller import OWNER, Poller
from aida.platform.channel.send import CHANNEL_ACTIONS, ChannelTool
from aida.platform.channel.store import SqliteChannelStore
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


def _crear_canal_prueba(tmp_path: Path, reloj: list[datetime]) -> Channel:
    key = Fernet.generate_key()
    store = SqliteChannelStore(tmp_path / "canal_test.sqlite", key, country="57")
    config = ChannelConfig(
        relay_url="http://relay.local",
        phone_number_id="num-meta-1",
        graph_url="https://graph.local",
        poll_seconds=10,
        country_code="57",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"messaging_product": "whatsapp", "messages": [{"id": "wamid.OUT_TEST"}]}
        )

    cliente = httpx.Client(transport=httpx.MockTransport(handler))
    tool = ChannelTool(
        store=store, config=config, token="tok-meta", http=cliente, now=lambda: reloj[0]
    )
    log = EventLog()
    gate = ActionGate(actions=CHANNEL_ACTIONS, tool=tool, log=log, owner=OWNER, tick=lambda: 1)
    poller = Poller(lambda: None, seconds=10)
    lock = threading.Lock()
    return Channel(
        config=config,
        store=store,
        tool=tool,
        gate=gate,
        poller=poller,
        log=log,
        lock=lock,
    )


def test_get_conversaciones_con_estado_del_canal(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
    tmp_path: Path,
) -> None:
    arm, ctx, _, reloj = arm_contexto
    canal = _crear_canal_prueba(tmp_path, reloj)
    canal.store.add_inbound(
        wamid="w.in1",
        phone="573001112233",
        name="Carlos",
        kind="text",
        text="Hola",
        at=reloj[0],
    )
    ctx_canal = replace(ctx, channel=canal)
    resp = arm.get("conversaciones", ctx_canal)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["running"] is True
    assert data["poll_error"] == ""
    assert len(data["conversations"]) == 1
    c = data["conversations"][0]
    assert c["phone"] == "573001112233"
    assert c["name"] == "Carlos"
    assert c["unread"] == 1
    assert c["last_text"] == "Hola"
    assert c["last_direction"] == "in"


def test_get_conversacion_enlaza_ficha_sin_prefijo(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
    tmp_path: Path,
) -> None:
    arm, ctx, _, reloj = arm_contexto
    canal = _crear_canal_prueba(tmp_path, reloj)

    # Cliente registrado en agenda con número nacional (sin prefijo 57)
    cliente = arm._store.add_client(name="Ana Ruiz", phone="3001112233")

    # Mensaje en canal guardado con prefijo internacional
    canal.store.add_inbound(
        wamid="w.in1",
        phone="573001112233",
        name="Ana",
        kind="text",
        text="Hola",
        at=reloj[0],
    )

    ctx_canal = replace(ctx, channel=canal)
    resp = arm.get("conversacion/573001112233", ctx_canal)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["phone"] == "573001112233"
    assert data["client_id"] == cliente.id
    assert data["window_open"] is True


def test_get_conversacion_hora_en_zona_local(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
    tmp_path: Path,
) -> None:
    arm, ctx, _, reloj = arm_contexto
    canal = _crear_canal_prueba(tmp_path, reloj)

    t_utc = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)
    canal.store.add_inbound(
        wamid="w.in1",
        phone="573001112233",
        name="Ana",
        kind="text",
        text="Hola",
        at=t_utc,
    )

    ctx_canal = replace(ctx, channel=canal)
    resp = arm.get("conversacion/573001112233", ctx_canal)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    msg_at = data["messages"][0]["at"]
    assert "-05:00" in msg_at
    assert "+00:00" not in msg_at
    assert msg_at == t_utc.astimezone(TZ).isoformat()


def test_ficha_trae_conversation_phone(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
    tmp_path: Path,
) -> None:
    arm, ctx, _, reloj = arm_contexto
    canal = _crear_canal_prueba(tmp_path, reloj)

    # Cliente registrado en agenda con número nacional
    cliente = arm._store.add_client(name="Beto", phone="3009998877")

    # Inbound en canal
    canal.store.add_inbound(
        wamid="w.b1",
        phone="573009998877",
        name="Beto",
        kind="text",
        text="Mensaje Beto",
        at=reloj[0],
    )

    # Con canal
    ctx_canal = replace(ctx, channel=canal)
    resp = arm.get(f"contacto/{cliente.id}", ctx_canal)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["conversation_phone"] == "573009998877"

    # Sin canal
    resp_sin = arm.get(f"contacto/{cliente.id}", ctx)
    assert resp_sin is not None and resp_sin.status == 200
    assert json.loads(resp_sin.body)["conversation_phone"] == ""


def test_enviar_por_accion_audita_message_sent_con_texto_cifrado(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
    tmp_path: Path,
) -> None:
    arm, ctx, audit, reloj = arm_contexto
    canal = _crear_canal_prueba(tmp_path, reloj)

    # Abrir ventana de conversación
    canal.store.add_inbound(
        wamid="w.in1",
        phone="573001112233",
        name="Ana",
        kind="text",
        text="Hola",
        at=reloj[0],
    )

    ctx_canal = replace(ctx, channel=canal)
    post_body = json.dumps({
        "action": "enviar",
        "arguments": {"phone": "573001112233", "text": "TextoSuperSecreto123"},
    }).encode()
    resp = arm.post("accion", post_body, {}, ctx_canal)
    assert resp is not None and resp.status == 200
    data = json.loads(resp.body)
    assert data["status"] == "done"

    # Auditoría contiene message_sent firmado por ctx.professional
    assert ctx.professional is not None
    lineas = [x for x in audit.lines() if x["kind"] == "message_sent"]
    assert len(lineas) == 1
    assert lineas[0]["actor"] == ctx.professional.id
    assert lineas[0]["owner"] == data["data"]["message"]

    # Texto secreto cifrado aparte en disco (no aparece en crudo en audit.jsonl)
    assert audit._chain._path is not None
    crudo = audit._chain._path.read_bytes()
    assert b"TextoSuperSecreto123" not in crudo


def test_enviar_sin_canal_409(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, _ = arm_contexto
    post_body = json.dumps({
        "action": "enviar",
        "arguments": {"phone": "573001112233", "text": "Hola"},
    }).encode()
    resp = arm.post("accion", post_body, {}, ctx)
    assert resp is not None and resp.status == 409


def test_marcar_leida_no_audita(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
    tmp_path: Path,
) -> None:
    arm, ctx, audit, reloj = arm_contexto
    canal = _crear_canal_prueba(tmp_path, reloj)
    canal.store.add_inbound(
        wamid="w.in1",
        phone="573001112233",
        name="Ana",
        kind="text",
        text="Hola",
        at=reloj[0],
    )

    ctx_canal = replace(ctx, channel=canal)
    post_body = json.dumps({
        "action": "marcar_leida",
        "arguments": {"phone": "573001112233", "up_to": "1"},
    }).encode()
    resp = arm.post("accion", post_body, {}, ctx_canal)
    assert resp is not None and resp.status == 200
    assert json.loads(resp.body)["status"] == "done"

    # Marcar leída no genera evento de auditoría
    assert not any(x["kind"] == "marcar_leida" for x in audit.lines())
    assert not any(x["kind"] == "message_sent" for x in audit.lines())


def test_canal_token_guarda_solo_si_meta_acepta(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm, ctx, _, _ = arm_contexto
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(cfg_dir))

    ch_yaml = cfg_dir / "channel.yaml"
    ch_yaml.write_text(
        yaml.dump({
            "relay_url": "http://relay.local",
            "phone_number_id": "123",
            "graph_url": "https://graph.local",
            "poll_seconds": 10,
        }),
        encoding="utf-8",
    )

    lic_wa = LicenseState(
        status="valid",
        license=License(
            id="l1",
            client="C",
            arms=("agenda", "whatsapp"),
            issued=date.today(),
            expires=date.today() + timedelta(days=1),
        ),
    )
    ctx_wa = replace(ctx, license=lic_wa)

    # 1. Meta rechaza -> no guarda
    monkeypatch.setattr(
        "aida.arms.agenda.app.check_whatsapp_token", lambda tok, cfg: (False, "401")
    )
    r_bad = arm.post("canal-token", json.dumps({"token": "tok_malo"}).encode(), {}, ctx_wa)
    assert r_bad is not None and r_bad.status == 200
    d_bad = json.loads(r_bad.body)
    assert d_bad["ok"] is False
    assert d_bad["reason"] == "401"
    assert whatsapp_token() != "tok_malo"

    # 2. Meta acepta -> sí guarda
    monkeypatch.setattr(
        "aida.arms.agenda.app.check_whatsapp_token", lambda tok, cfg: (True, "200")
    )
    r_ok = arm.post("canal-token", json.dumps({"token": "tok_bueno"}).encode(), {}, ctx_wa)
    assert r_ok is not None and r_ok.status == 200
    d_ok = json.loads(r_ok.body)
    assert d_ok["ok"] is True
    assert d_ok["reason"] == "200"
    assert whatsapp_token() == "tok_bueno"


def test_pagina_sin_licencia_whatsapp_no_trae_pestanas(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, _ = arm_contexto
    # ctx por defecto solo tiene arms=("agenda", "health")
    assert not ctx.license.allows(LICENSE_ID)
    pagina = arm.page(ctx)
    assert 'id="tab-convo"' not in pagina
    assert 'id="pane-convo"' not in pagina
    assert "CONVERSATIONS_JS" not in pagina


def test_pagina_con_licencia_trae_pestanas(
    arm_contexto: tuple[AgendaArm, ShellContext, AuditLog, list[datetime]],
) -> None:
    arm, ctx, _, _ = arm_contexto
    lic_wa = LicenseState(
        status="valid",
        license=License(
            id="l1",
            client="C",
            arms=("agenda", "whatsapp"),
            issued=date.today(),
            expires=date.today() + timedelta(days=1),
        ),
    )
    ctx_wa = replace(ctx, license=lic_wa)
    pagina = arm.page(ctx_wa)
    assert 'id="tab-convo"' in pagina
    assert 'id="pane-convo"' in pagina
    assert "cargarConversaciones" in pagina
