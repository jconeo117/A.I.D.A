"""La aplicación local: AIDA abierta, una sesión de trabajo tras otra, sin terminal.

A diferencia de la pantalla de un solo uso (`review_ui.collect_review`), esto es un
servicio, y lo es a propósito (plan 77): quien trabaja con AIDA no lanza un comando
por cada sesión. Sigue escuchando solo en 127.0.0.1 y detrás de un token, y atiende
una sesión cada vez.

Ningún texto de la pantalla vive aquí: sale de `ui.yaml`, sección `app` (`I-073`).
"""

from __future__ import annotations

import contextlib
import html
import json
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

import yaml
from cryptography.fernet import Fernet

from aida.arms.health.audio import to_wav
from aida.arms.health.capture import ModelUnavailableError, TranscriptionError
from aida.arms.health.consent import ConsentRecord, ConsentTexts, load_consent_texts
from aida.arms.health.consultation import (
    AbsenceUnresolvedError,
    Consultation,
    ConsultationResult,
    PreparedConsultation,
)
from aida.arms.health.live import (
    RECORDING_NAME,
    LiveSession,
    LiveTranscriber,
    line_kind,
    recording_wav,
)
from aida.arms.health.note import RenderedNote
from aida.arms.health.recovery import (
    Interrupted,
    expired,
    interrupted,
    purge_unmarked,
    sessions_root,
    started_at,
    write_marker,
)
from aida.arms.health.review import ReviewItem, review_items
from aida.arms.health.review_ui import (
    _ICONS_CONSENT,
    HistoryView,
    ReviewSubmission,
    UISettings,
    _render_page,
    history_view,
    parse_submission,
    reviewer_from,
)
from aida.arms.health.sections import SectionSpec, load_sections
from aida.arms.health.traceability import ReviewTrace
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import Event, EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.platform.audit import AuditLog
from aida.platform.identity import IdentityError, Professional, ProfileStore
from aida.platform.license import LicenseState
from aida.platform.sealed import (
    plain_copy,
    plain_directory,
    purge_plain_leftovers,
    seal_bytes,
)
from aida.platform.web.pages import (
    JS_TO_SETTINGS,
    identity_script,
    license_block,
    license_chip,
    license_page,
    page_frame,
    profiles_block,
    readonly_page,
    settings_frame,
    user_chip,
)
from aida.platform.web.shell import Response, ShellContext, json_response, run_shell
from aida.platform.web.wording import SHELL_KEYS, load_shell_wording
from aida.resources import data_root, user_dir

_RAIZ = data_root()
_UI_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "ui.yaml"

_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
"""Un código, no un nombre: sin espacios ni tildes. Aparece en la caché, en las
métricas y en el nombre de los archivos exportados, y lo que no es un nombre no es
un dato que haya que proteger."""



_WORKDIR_PREFIX = "aida-sesion-"
"""Las sesiones viven en carpetas con este prefijo, dentro de `sessions_root()`. Al
arrancar se borran las que no tienen marcador (subidas a medias), y las grabaciones que
sí lo tienen se conservan para recuperarlas (87). Las que quedaran en `/tmp` de
versiones anteriores se siguen borrando: nadie las va a recuperar."""

_CLAVES_APP = (
    "title_start",
    "title_working",
    "title_problem",
    "lead_upload",
    "lead_record",
    "owner_label",
    "owner_placeholder",
    "owner_hint",
    "owner_invalid",
    "mode_upload",
    "mode_record",
    "audio_title",
    "audio_hint",
    "document_title",
    "document_hint",
    "change_file",
    "mic_title",
    "mic_hint",
    "mic_denied",
    "check_cloud",
    "check_local_only",
    "check_local",
    "check_cache",
    "check_writer",
    "check_no_writer",
    "cloud_note",
    "cloud_needs_local",
    "check_local_setting",
    "where_title",
    "where_lead",
    "where_toggle",
    "where_out",
    "where_local",
    "where_transcription",
    "where_note",
    "where_fallback",
    "where_reader",
    "where_here",
    "recover_title",
    "recover_meta",
    "recover_expiry",
    "recover_go",
    "recover_drop",
    "recover_confirm",
    "recover_back",
    "recover_yes",
    "recover_gone",
    "missing_owner",
    "missing_audio",
    "missing_document",
    "missing_consent",
    "consent_version",
    "ready",
    "start_button",
    "record_button",
    "recording_pill",
    "stop_button",
    "working_lead",
    "working_count",
    "pill_working",
    "retry_label",
    "nothing_saved",
    "problem_capture_title",
    "problem_capture_detail",
    "problem_capture_command",
    "problem_cache_key_title",
    "problem_cache_key_detail",
    "problem_cache_key_command",
    "problem_transcription_title",
    "problem_transcription_detail",
    "problem_transcription_command",
    "problem_processing_title",
    "problem_processing_detail",
    "problem_processing_command",
    "live_title",
    "live_hint",
    "live_tab",
    "state_ai",
    "state_writing",
    "state_mine",
    "state_waiting",
    "waiting_placeholder",
    "banner_live",
    "banner_offline",
    "listening",
    "listening_offline",
    "strip_recording",
    "strip_paused",
    "pause_label",
    "resume_label",
    "pill_paused",
    "pause_mark",
    "new_label",
    "new_count",
    "key_title",
    "key_text",
    "key_status_file",
    "key_status_env",
    "key_status_none",
    "key_status_bad",
    "key_test",
    "key_save",
    "key_ok",
    "key_fail",
    "key_saved",
    "key_env_note",
    "model_title",
    "model_text",
    "model_ok",
    "model_missing",
    "model_download",
    "model_downloading",
    "model_progress",
    "model_failed",
    "model_keep_open",
    "model_lead",
    "model_to_settings",
    "reader_title",
    "reader_text",
    "reader_ok",
    "reader_missing",
    "cipher_title",
    "cipher_text",
    "cipher_env",
    "cipher_ok",
    "check_no_local",
    "pill_preparing",
    "problem_model_title",
    "readonly_item_consents",
    "readonly_item_audit",
    "readonly_item_texts",
)

_CSS_EXTRA = """.mode {
  display: inline-flex; gap: 2px; padding: 3px; background: #dfe4ea; border-radius: 10px;
  margin: 1.5rem 0 0;
}
.mode button {
  border: 0; background: transparent; padding: .45rem .9rem; border-radius: 8px; font-weight: 600;
  font-size: .86rem; color: var(--muted); display: inline-flex; align-items: center; gap: .4rem;
}
.mode button[aria-pressed="true"] {
  background: var(--surface); color: var(--ink); box-shadow: var(--e1);
}
.rec-hint {
  display: flex; align-items: center; gap: .6rem; min-height: 150px; padding: 1rem 1.2rem;
  border-radius: var(--r); background: var(--surface-2); border: 1.5px dashed var(--line-2);
  color: var(--muted); font-size: .88rem;
}
.rec-hint svg { width: 26px; height: 26px; color: var(--brand); }
.consent {
  display: flex; gap: .55rem; align-items: flex-start; margin: 1rem 0 0; font-size: .85rem;
  color: var(--ink-2);
}
.consent input { margin-top: .2rem; accent-color: var(--brand); }
.rec {
  display: flex; flex-direction: column; align-items: center; text-align: center; gap: .9rem;
  padding: 1rem 0 .5rem;
}
.rec-dot {
  width: 84px; height: 84px; border-radius: 50%; display: grid; place-items: center;
  background: var(--off-soft); position: relative;
}
.rec-dot::before {
  content: ""; width: 30px; height: 30px; border-radius: 50%; background: #dc2626;
  box-shadow: 0 0 0 0 rgba(220,38,38,.5); animation: pulse 1.6s ease-out infinite;
}
@keyframes pulse { to { box-shadow: 0 0 0 26px rgba(220,38,38,0); } }
.rec-time {
  font: 600 2.4rem/1 ui-monospace, "JetBrains Mono", SFMono-Regular, Menlo, monospace;
  font-variant-numeric: tabular-nums; letter-spacing: .02em;
}
.rec-state { color: var(--muted); font-size: .9rem; }
.meter { display: flex; align-items: flex-end; gap: 3px; height: 34px; }
.meter span { width: 5px; border-radius: 3px; background: var(--brand); opacity: .85; }
.pill.rec-pill .dot { background: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,.25); }
.btn-stop { background: #b91c1c; box-shadow: 0 2px 6px rgba(185,28,28,.35); }
.btn-stop:hover { background: #991b1b; }
/* Pantallas de la 77: inicio, progreso, error */
.stage {
  flex: 1; display: flex; justify-content: center; align-items: flex-start;
  padding: 2.5rem 1.75rem; overflow-y: auto;
}
.panel-card {
  width: 100%; max-width: 780px; background: var(--surface); border: 1px solid var(--line);
  border-radius: 16px; box-shadow: var(--e2); padding: 1.75rem 2rem;
}
.panel-card h2 { margin: 0; font-size: 1.35rem; font-weight: 650; letter-spacing: -.01em; }
.panel-card .lead { margin: .25rem 0 1.5rem; color: var(--muted); font-size: .92rem; }
.label {
  display: block; font-size: .72rem; font-weight: 700; letter-spacing: .09em;
  text-transform: uppercase; color: var(--brand-ink); margin-bottom: .4rem;
}
.input {
  width: 100%; font: inherit; font-size: 1rem; padding: .65rem .8rem;
  border: 1px solid var(--line-2);
  border-radius: var(--r-s); background: var(--surface); color: var(--ink); outline: none;
}
.input:focus { border-color: var(--brand); box-shadow: 0 0 0 4px var(--brand-ring); }
.hint { margin: .4rem 0 0; font-size: .8rem; color: var(--muted); }
.drops { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0 1.25rem; }
.drop {
  position: relative; display: flex; flex-direction: column; align-items: center;
  justify-content: center;
  gap: .4rem; min-height: 150px; padding: 1rem; text-align: center;
  border: 1.5px dashed var(--line-2);
  border-radius: var(--r); background: var(--surface-2); color: var(--muted); cursor: pointer;
  transition: border-color .15s, background .15s;
}
.drop:hover, .drop.over {
  border-color: var(--brand); background: var(--brand-soft); color: var(--brand-ink);
}
.drop svg { width: 26px; height: 26px; color: var(--brand); }
.drop strong { color: var(--ink); font-size: .95rem; }
.drop small { font-size: .78rem; }
.drop input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop.filled { border-style: solid; border-color: var(--brand); background: var(--surface); }
.drop .file { font-size: .85rem; color: var(--brand-ink); font-weight: 600; word-break: break-all; }
.checks {
  display: flex; flex-wrap: wrap; gap: .5rem; padding-top: 1.25rem;
  border-top: 1px solid var(--line);
}
.check {
  display: inline-flex; align-items: center; gap: .4rem; font-size: .8rem; color: var(--ink-2);
  padding: .3rem .65rem; border-radius: 999px; background: var(--surface-2);
  border: 1px solid var(--line);
}
.check .dot { width: 7px; height: 7px; border-radius: 50%; background: #14b8a6; }
.check.warn .dot { background: #f59e0b; } .check.fail .dot { background: #dc2626; }
.steps { list-style: none; margin: 1.5rem 0 0; padding: 0; }
.steps li {
  display: flex; align-items: center; gap: .9rem; padding: .85rem 0;
  border-top: 1px solid var(--line); color: var(--faint);
}
.steps li:first-child { border-top: 0; }
.steps .mark-step {
  width: 28px; height: 28px; flex: none; border-radius: 50%; display: grid; place-items: center;
  border: 2px solid var(--line-2); background: var(--surface);
}
.steps .mark-step svg { width: 14px; height: 14px; }
.steps li.done { color: var(--ink-2); }
.steps li.done .mark-step { background: var(--brand); border-color: var(--brand); color: white; }
.steps li.now { color: var(--ink); font-weight: 600; }
.steps li.now .mark-step {
  border-color: var(--brand); border-top-color: transparent; animation: spin 1s linear infinite;
}
.steps .time {
  margin-left: auto; font-size: .8rem; color: var(--muted); font-variant-numeric: tabular-nums;
  font-weight: 400;
}
@keyframes spin { to { transform: rotate(360deg); } }
.bar-progress {
  height: 6px; border-radius: 999px; background: var(--line); overflow: hidden;
  margin-top: 1.25rem;
}
.bar-progress span {
  display: block; height: 100%; background: linear-gradient(90deg, #14b8a6, var(--brand));
  border-radius: 999px;
}
.alert-head { display: flex; align-items: center; gap: .8rem; }
.alert-icon {
  width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center;
  background: var(--off-soft); color: var(--off); flex: none;
}
.alert-icon svg { width: 22px; height: 22px; }
.alert-icon.warn { background: var(--warn-soft); color: var(--warn); }
.cmd {
  display: flex; align-items: center; gap: .75rem; margin: 1.25rem 0; padding: .75rem 1rem;
  background: var(--head); color: var(--head-ink); border-radius: var(--r-s);
  font: 500 .9rem ui-monospace, "JetBrains Mono", SFMono-Regular, Menlo, monospace;
}
.cmd code { flex: 1; }
.cmd .icon-btn { opacity: 1; color: var(--head-muted); margin-left: 0; }
.cmd .icon-btn:hover { color: white; background: rgba(255,255,255,.08); border-color: transparent; }
.btn-quiet { border: 0; background: transparent; color: var(--muted); box-shadow: none; }
.btn-quiet:hover { color: var(--ink); border: 0; }
.btn-primary:disabled { background: #9fb8b5; box-shadow: none; cursor: not-allowed; }
.drop[hidden], .rec-hint[hidden], .consent[hidden] { display: none; }
.screen[hidden] { display: none; }
.screen { flex: 1; display: flex; flex-direction: column; min-height: 0; }"""

_ICONS_EXTRA = """  <symbol id="i-audio" viewBox="0 0 24 24">
    <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
  </symbol>
  <symbol id="i-upload" viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5M4 20h16"/></symbol>
  <symbol id="i-alert" viewBox="0 0 24 24">
    <path d="M12 9v4M12 17h.01"/>
    <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>
  </symbol>
  <symbol id="i-play" viewBox="0 0 24 24"><path d="M6 4l14 8-14 8Z"/></symbol>
  <symbol id="i-mic" viewBox="0 0 24 24">
    <rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10a7 7 0 0 0 14 0M12 17v5M8 22h8"/>
  </symbol>
  <symbol id="i-stop" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></symbol>
  <symbol id="i-exit" viewBox="0 0 24 24">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>
  </symbol>"""



_CSS_AJUSTES_SALUD = """.key-row { display: flex; gap: .6rem; margin-top: .8rem; }
.key-row .input {
  flex: 1; font-family: ui-monospace, "JetBrains Mono", SFMono-Regular, Menlo, monospace;
  font-size: .9rem;
}
.result { display: flex; align-items: center; gap: .45rem; margin-top: .6rem; font-size: .85rem; }
.result svg { width: 15px; height: 15px; }
.result.ok { color: #1a7f4b; }
.result.fail { color: var(--off); }
.result[hidden] { display: none; }
.download-bar {
  height: 8px; border-radius: 999px; background: var(--line); overflow: hidden;
  margin: 1rem 0 .4rem;
}
.download-bar span {
  display: block; height: 100%; background: linear-gradient(90deg, #14b8a6, var(--brand));
}
.download-meta {
  display: flex; justify-content: space-between; font-size: .8rem; color: var(--muted);
  font-variant-numeric: tabular-nums;
}"""

_ICONS_SETTINGS_EXTRA = """  <symbol id="i-gear" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="3"/>
    <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3"/>
    <path d="M15 19.4a1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5"/>
    <path d="M8.9 19.4a1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8"/>
    <path d="M4.6 15a1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1"/>
    <path d="M4.6 8.9a1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3"/>
    <path d="M9 4.6a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5"/>
    <path d="M15.1 4.6a1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8"/>
    <path d="M19.4 9a1.7 1.7 0 0 0 1.5 1h.1a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1"/>
  </symbol>
  <symbol id="i-key" viewBox="0 0 24 24">
    <circle cx="7.5" cy="15.5" r="4.5"/><path d="M10.7 12.3 21 2M16 7l3 3M14 9l2 2"/>
  </symbol>
  <symbol id="i-download" viewBox="0 0 24 24"><path d="M12 4v12M7 11l5 5 5-5M4 20h16"/></symbol>
  <symbol id="i-back" viewBox="0 0 24 24"><path d="M15 18l-6-6 6-6"/></symbol>"""



_CSS_CONSENT = """/* La 82: la autorización, antes de grabar o subir */
.consent-card {
  margin-top: 1.25rem; border: 1px solid var(--line); border-radius: var(--r);
  background: var(--surface-2); padding: .9rem 1.1rem 1rem;
}
.consent-head { display: flex; align-items: center; gap: .55rem; }
.consent-head svg { width: 18px; height: 18px; color: var(--brand); }
.consent-head strong { font-size: .9rem; }
.consent-head .version { margin-left: auto; font-size: .75rem; color: var(--muted); }
.consent-text {
  margin: .6rem 0 .2rem; padding: .7rem .9rem; border-left: 3px solid var(--brand);
  background: var(--surface); border-radius: 0 8px 8px 0; font-size: .88rem;
  line-height: 1.55; color: var(--ink-2);
}
.consent-card .consent { margin-top: .75rem; font-weight: 600; color: var(--ink); }
.consent-card.done { border-color: color-mix(in srgb, var(--brand) 45%, var(--line)); }

/* La 85b: la nube, con su propia autorización */
.consent-card .consent + .consent { margin-top: .45rem; font-weight: 500; color: var(--ink-2); }
.consent-note { margin: .35rem 0 0 1.6rem; font-size: .8rem; color: var(--muted); }
.consent-note[hidden] { display: none; }

/* La 85b: dónde se procesan los datos */
.toggle-row { display: flex; align-items: center; gap: .8rem; margin-top: .8rem; }
.toggle {
  width: 42px; height: 24px; border-radius: 999px; background: var(--line-2); position: relative;
  border: 0; cursor: pointer; flex: none;
}
.toggle::after {
  content: ""; position: absolute; top: 3px; left: 3px; width: 18px; height: 18px;
  border-radius: 50%;
  background: #fff; transition: left .15s;
}
.toggle[aria-checked="true"] { background: var(--brand); }
.toggle[aria-checked="true"]::after { left: 21px; }
.where {
  margin: .8rem 0 0; padding: 0; list-style: none; display: grid; gap: .4rem;
  font-size: .86rem;
}
.where li { display: flex; justify-content: space-between; gap: 1rem; }
.where span { color: var(--muted); }
.where b { font-weight: 600; }
.where .off { color: var(--muted); text-decoration: line-through; }

/* La 87: una grabación interrumpida, solo para quien la grabó */
.recover {
  margin: 0 0 1.25rem; padding: 1rem 1.1rem; border-radius: var(--r-s);
  border: 1px solid color-mix(in srgb, var(--warn) 40%, var(--line));
  background: color-mix(in srgb, var(--warn-soft) 55%, white);
}
.recover-head { display: flex; align-items: center; gap: .6rem; }
.recover-head svg { width: 18px; height: 18px; color: var(--warn); flex: none; }
.recover-head strong { font-size: .95rem; }
.recover-meta {
  margin: .45rem 0 0 1.75rem; font-size: .88rem; color: var(--ink-2);
  font-variant-numeric: tabular-nums;
}
.recover-meta b { color: var(--ink); }
.recover-expiry { margin: .25rem 0 0 1.75rem; font-size: .8rem; color: var(--muted); }
.recover-actions { display: flex; gap: .6rem; margin: .85rem 0 0 1.75rem; flex-wrap: wrap; }
.recover-confirm { margin: .85rem 0 0 1.75rem; font-size: .88rem; color: var(--ink-2); }
.recover[hidden], .recover-confirm[hidden], .recover-actions[hidden] { display: none; }
.btn-danger { color: var(--off); border-color: color-mix(in srgb, var(--off) 35%, var(--line)); }
.btn-danger:hover { color: white; background: var(--off); border-color: var(--off); }"""













_CSS_LIVE = """/* La 77b: la consulta en vivo */
.field-state {
  display: inline-flex; align-items: center; gap: .3rem; font-size: .72rem; color: var(--faint);
  font-weight: 500;
}
.field-state svg { width: 12px; height: 12px; }
.field-state.writing { color: var(--brand); font-weight: 600; }
.field-state.mine { color: var(--warn); font-weight: 650; }
.field.writing textarea {
  background: linear-gradient(90deg, transparent 0%, var(--brand-soft) 50%, transparent 100%);
  background-size: 200% 100%; animation: shimmer 1.8s linear infinite;
}
@keyframes shimmer { from { background-position: 100% 0; } to { background-position: -100% 0; } }
.field.waiting { background: var(--surface-2); box-shadow: none; border-style: dashed; }
.live-banner {
  display: flex; align-items: center; gap: .45rem; margin-bottom: .8rem; padding: .45rem .75rem;
  border-radius: var(--r-s); background: var(--brand-soft); color: var(--brand-ink);
  font-size: .8rem;
}
.live-banner svg { width: 14px; height: 14px; }
.live-banner.offline { background: var(--warn-soft); color: var(--warn); }
.live-banner[hidden], .listening[hidden], .pause-mark[hidden] { display: none; }
.chat .a.fresh { animation: arrive 2.4s ease-out; border-color: var(--brand); }
@keyframes arrive { from { background: var(--brand-soft); } to { background: var(--surface); } }
.listening {
  display: flex; align-items: center; gap: .5rem; margin-top: .6rem; color: var(--muted);
  font-size: .82rem;
}
.listening .dot-live {
  width: 8px; height: 8px; border-radius: 50%; background: var(--brand);
  animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink { 50% { opacity: .25; } }
.pause-mark {
  align-self: center; font-size: .75rem; color: var(--warn); letter-spacing: .08em;
  text-transform: uppercase; margin: .4rem 0;
}
.badge.new { background: var(--brand-soft); color: var(--brand-ink); }
.code.fresh { border-color: var(--brand); box-shadow: 0 0 0 4px var(--brand-ring); }
.tab .count.new { background: var(--brand); color: white; }
.rec-strip { display: flex; align-items: center; gap: .9rem; }
.rec-strip .rec-small {
  width: 12px; height: 12px; border-radius: 50%; background: #dc2626;
  box-shadow: 0 0 0 0 rgba(220,38,38,.5); animation: pulse 1.6s ease-out infinite;
}
.rec-strip .rec-clock {
  font: 600 1.05rem ui-monospace, "JetBrains Mono", SFMono-Regular, Menlo, monospace;
  font-variant-numeric: tabular-nums;
}
.rec-strip .meter { height: 22px; gap: 2px; }
.rec-strip .meter span { width: 4px; }
.rec-strip .note { font-size: .8rem; color: var(--muted); }
.btn-pause { background: var(--surface); }
footer .btn-pause { margin-left: auto; }
footer .btn-pause + .btn-primary { margin-left: 0; }
body.paused .rec-strip .rec-small { background: #f59e0b; animation: none; }
body.paused .rec-strip .rec-clock { color: var(--warn); }
body.paused .rec-strip .meter span { opacity: .25; }
.pill.pause-pill .dot { background: #fbbf24; box-shadow: 0 0 0 3px rgba(251,191,36,.2); }"""

_ICONS_LIVE_EXTRA = """  <symbol id="i-lock" viewBox="0 0 24 24">
    <rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>
  </symbol>
  <symbol id="i-pause" viewBox="0 0 24 24"><path d="M9 5v14M15 5v14"/></symbol>
  <symbol id="i-spark" viewBox="0 0 24 24">
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4"/>
    <path d="M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/>
  </symbol>
  <symbol id="i-offline" viewBox="0 0 24 24">
    <path d="M2 8.8a15 15 0 0 1 4.2-2.6M10.7 5.1A15 15 0 0 1 22 8.8M5 12.9a10 10 0 0 1 3.1-2"/>
    <path d="M12.5 10.1a10 10 0 0 1 6.5 2.8M8.5 16.4a5 5 0 0 1 7 0M12 20h.01M2 2l20 20"/>
  </symbol>"""




@dataclass(frozen=True, slots=True)
class AppWording:
    """Los textos de la aplicación, validados al cargar: una pantalla a medias se
    descubre con alguien esperando delante."""

    texts: Mapping[str, str]
    steps: tuple[str, ...]
    idle_lock_minutes: float = 10.0
    """Minutos sin uso tras los que AIDA pide el PIN otra vez (plan 83). En el pack:
    cuánto se tolera un equipo desatendido es una decisión de cada consultorio."""

    def __getitem__(self, clave: str) -> str:
        return self.texts[clave]





@dataclass(frozen=True, slots=True)
class AppProblem:
    """Por qué no se puede empezar o no se pudo terminar. `kind` elige los textos en
    `ui.yaml` (`problem_<kind>_title`…); `detail` es lo que dijo la excepción, si la
    hubo, y se enseña tal cual: es dato, no instrucción."""

    kind: str
    detail: str = ""


class DocumentReader(Protocol):
    def read(self, image_path: Path) -> str: ...


@dataclass(frozen=True, slots=True)
class AppReady:
    """Todo lo que la aplicación necesita para atender sesiones. Se construye una
    vez —el catálogo y el modelo local tardan en cargar— y se reutiliza."""

    consultation: Consultation
    reader: DocumentReader
    log: EventLog
    cloud: bool
    writer: str | None
    live_stt: LiveTranscriber | None = None
    """El transcriptor del vivo (plan 77b): solo la nube. `None` sin clave de nube, y
    entonces grabar sigue funcionando, pero sin vivo: todo se procesa al terminar."""
    local_model: bool = True
    """Si hay modelo local de respaldo (plan 78). Sin él se trabaja con la nube, y el
    inicio lo avisa: una consulta sin conexión no se podría transcribir."""
    seal_key: bytes | None = None
    """Con qué se sella lo que se guarda durante una sesión (plan 85a): la clave del
    equipo. Sin ella, una de esta ejecución, que se pierde al cerrar."""
    local_only: bool = False
    """«Procesar solo en este equipo» (plan 85b): con él no hay casilla de nube ni vivo
    en la nube, aunque haya clave."""


@dataclass(frozen=True, slots=True)
class AppTools:
    """Lo que Ajustes consulta y cambia en el equipo (plan 78). Inyectado, como
    `build`: la aplicación no sabe dónde se guardan las claves ni cómo se descarga el
    modelo, y la batería lo sustituye sin red ni disco."""

    cloud_key_source: Callable[[], str | None]
    """`env`, `file` o `None`."""
    check_cloud_key: Callable[[str], tuple[bool, str]]
    save_cloud_key: Callable[[str], None]
    model_name: str
    model_total_mb: int
    model_available: Callable[[], bool]
    model_bytes: Callable[[], int]
    download_model: Callable[[], None]
    reader_version: Callable[[], str | None]
    """La versión del lector de documentos, o `None` si no lee español. Puede
    reventar si el lector no está."""
    cache_key_since: Callable[[], float | None]
    """Cuándo se creó la clave de la caché; `None` si viene del entorno."""
    local_only: Callable[[], bool]
    save_local_only: Callable[[bool], None]
    destinations: tuple[tuple[str, str], ...]
    """(qué, dónde) de lo que sale en la nube: [("transcription", "OpenAI · Estados
    Unidos"), ("note", "openai · Estados Unidos")]. Lo monta quien lanza: la aplicación
    no sabe qué proveedores hay."""


@dataclass(frozen=True, slots=True)
class ReviewDraft:
    prepared: PreparedConsultation
    note: RenderedNote
    items: tuple[ReviewItem, ...]
    trace: ReviewTrace | None = None
    """Las ausencias sin respaldo y la historia con su fecha (plan 84), para la pantalla."""


def draft_for_screen(consultation: Consultation, prepared: PreparedConsultation) -> ReviewDraft:
    """Lo que la pantalla necesita para abrirse: la nota redactada y solo los ítems
    de código. Una sola puerta para la CLI (`--ui`) y la aplicación: con dos, una
    acabaría ofreciendo al profesional otros ítems que la otra."""
    if prepared.grouped_sections is None or prepared.encounter is None:
        raise ValueError("la pantalla necesita la segmentación por apartados")
    wording = default_summary_wording()
    note = consultation.render_note(prepared)
    items = review_items(
        grouped_sections=prepared.grouped_sections,
        encounter=prepared.encounter,
        matches=prepared.matches,
        known_code_suffix=wording.known_code_suffix,
        evidence_template=wording.evidence_template,
        negated_suffix=wording.negated_suffix,
        review_findings=False,
    )
    ajustes = consultation.traceability
    trace = (
        ReviewTrace(
            flags=consultation.absence_flags(prepared, note),
            profile=consultation.history_profile(prepared),
            settings=ajustes,
        )
        if ajustes is not None
        else None
    )
    return ReviewDraft(prepared=prepared, note=note, items=items, trace=trace)


def close_from_screen(
    consultation: Consultation,
    draft: ReviewDraft,
    submission: ReviewSubmission,
    *,
    actor: Professional | None,
) -> ConsultationResult:
    return consultation.review(
        draft.prepared,
        reviewer=reviewer_from(submission),
        elapsed_seconds=submission.elapsed_seconds,
        review_findings=False,
        note=draft.note,
        note_edits=submission.note_fields,
        actor=actor,
        absences_confirmed=submission.absences_confirmed,
    )


def valid_owner(text: str) -> bool:
    return _OWNER.fullmatch(text) is not None


def progress(events: Sequence[Event], *, drafted: bool) -> tuple[bool, ...]:
    """Qué pasos están hechos, leído de los eventos de esta sesión y no de un
    contador aparte: una sola puerta, y lo que no deja evento no ocurrió.

    El orden lo fija `Consultation.prepare`: la primera observación es la
    transcripción y la segunda, el documento. La redacción no deja evento: el último
    paso lo marca quien redacta (`drafted`)."""
    observados = sum(1 for e in events if e.kind is EventKind.INPUT_OBSERVED)
    tipos = {e.kind for e in events}
    return (
        observados >= 1,
        EventKind.TEXT_SEGMENTED in tipos,
        observados >= 2,
        EventKind.CATALOG_MATCHED in tipos,
        drafted,
    )





def _safe_suffix(name: str) -> str:
    sufijo = Path(name).suffix.lower()
    return sufijo if re.fullmatch(r"\.[a-z0-9]{1,8}", sufijo) else ""








def _duracion(delta: timedelta) -> str:
    total = max(int(delta.total_seconds()), 0)
    horas, resto = divmod(total, 3600)
    minutos, segundos = divmod(resto, 60)
    return f"{horas}:{minutos:02d}:{segundos:02d}" if horas else f"{minutos:02d}:{segundos:02d}"


def _recover_card(
    w: AppWording, pending: Interrupted, expires: datetime | None
) -> str:
    """La tarjeta de una grabación que AIDA no llegó a terminar (87). Solo se llama para
    quien la grabó: la ver es ya un permiso."""
    e = html.escape
    titular = f"<b>{e(pending.owner)}</b>"
    inicio = pending.started
    meta = w["recover_meta"].format(
        titular,
        inicio.strftime("%d/%m/%Y"),
        inicio.strftime("%H:%M"),
        f"<b>{_duracion(pending.duration())}</b>",
    )
    caduca = (
        f'<p class="recover-expiry">'
        f'{e(w["recover_expiry"].format(expires.strftime("%d/%m/%Y"), expires.strftime("%H:%M")))}'
        "</p>"
        if expires is not None
        else ""
    )
    return f"""<div class="recover" id="recover">
        <div class="recover-head"><svg><use href="#i-alert"/></svg>
          <strong>{e(w["recover_title"])}</strong></div>
        <p class="recover-meta">{meta}</p>
        {caduca}
        <div class="recover-actions" id="recover-actions">
          <button type="button" class="btn btn-primary" id="recover-go"
            data-recording="{e(pending.folder.name)}">
            <svg><use href="#i-play"/></svg>{e(w["recover_go"])}</button>
          <button type="button" class="btn btn-danger" id="recover-drop">
            {e(w["recover_drop"])}</button>
        </div>
        <div class="recover-confirm" id="recover-confirm" hidden>
          <p style="margin:0 0 .6rem">{w["recover_confirm"].format(titular)}</p>
          <div class="recover-actions" style="margin-left:0">
            <button type="button" class="btn" id="recover-back">
              {e(w["recover_back"])}</button>
            <button type="button" class="btn btn-danger" id="recover-yes">
              {e(w["recover_yes"])}</button>
          </div>
        </div>
      </div>"""


_JS_RECOVER = """
const recuperable = document.getElementById('recover');
if (recuperable) {
  const id = document.getElementById('recover-go').dataset.recording;
  const acciones = document.getElementById('recover-actions');
  const confirmar = document.getElementById('recover-confirm');
  const enviar = async (path) => {
    const cabeceras = { 'Content-Type': 'application/json' };
    const sent = await post(path, JSON.stringify({ id: id }), cabeceras);
    if (!sent.ok) { fail((await sent.json()).error); return; }
    location.reload();
  };
  document.getElementById('recover-drop').addEventListener('click', () => {
    acciones.hidden = true;
    confirmar.hidden = false;
  });
  document.getElementById('recover-back').addEventListener('click', () => {
    confirmar.hidden = true;
    acciones.hidden = false;
  });
  document.getElementById('recover-yes').addEventListener('click', () => enviar('/descartar'));
  document.getElementById('recover-go').addEventListener('click', () => enviar('/recuperar'));
}
"""


def _start_page(
    w: AppWording,
    brand: str,
    token: str,
    ready: AppReady,
    moment: datetime,
    *,
    settings: bool = False,
    consent: ConsentTexts | None = None,
    professional: Professional | None = None,
    license: LicenseState | None = None,
    pending: Interrupted | None = None,
    pending_expires: datetime | None = None,
) -> str:
    e = html.escape
    textos = consent if consent is not None else load_consent_texts()
    chips = (
        (w["check_local_setting"], True)
        if ready.local_only
        else (w["check_cloud"] if ready.cloud else w["check_local_only"], ready.cloud),
        (w["check_local"] if ready.local_model else w["check_no_local"], ready.local_model),
        (w["check_cache"], True),
        (
            w["check_writer"].format(ready.writer) if ready.writer else w["check_no_writer"],
            ready.writer is not None,
        ),
    )
    checks = "".join(
        f'<span class="check{"" if ok else " warn"}"><span class="dot"></span>{e(t)}</span>'
        for t, ok in chips
    )
    casilla_nube = (
        f'<label class="consent" id="cloud-box"><input type="checkbox" id="cloud">\n'
        f'          <span id="cloud-check"></span></label>\n'
        f'        <p class="consent-note" id="cloud-note">{e(w["cloud_note"])}</p>'
        if ready.cloud
        else ""
    )
    tarjeta = _recover_card(w, pending, pending_expires) if pending is not None else ""
    body = f"""<section class="screen" id="s-start">
  <div class="stage">
    <div class="panel-card">
      {tarjeta}
      <h2>{e(w["title_start"])}</h2>
      <p class="lead" id="lead">{e(w["lead_upload"])}</p>
      <label class="label" for="owner">{e(w["owner_label"])}</label>
      <input class="input" id="owner" placeholder="{e(w["owner_placeholder"])}"
        autocomplete="off">
      <p class="hint">{e(w["owner_hint"])}</p>
      <div class="mode" role="group">
        <button type="button" id="mode-upload" aria-pressed="true">
          <svg><use href="#i-upload"/></svg>{e(w["mode_upload"])}</button>
        <button type="button" id="mode-record" aria-pressed="false">
          <svg><use href="#i-mic"/></svg>{e(w["mode_record"])}</button>
      </div>
      <div class="drops">
        <label class="drop" id="drop-audio">
          <svg><use href="#i-audio"/></svg>
          <strong>{e(w["audio_title"])}</strong>
          <span class="file" hidden></span>
          <small>{e(w["audio_hint"])}</small>
          <input type="file" id="file-audio" accept="audio/*">
        </label>
        <div class="rec-hint" id="mic-box" hidden><svg><use href="#i-mic"/></svg>
          <div><strong>{e(w["mic_title"])}</strong><br>{e(w["mic_hint"])}</div></div>
        <label class="drop" id="drop-doc">
          <svg><use href="#i-doc"/></svg>
          <strong>{e(w["document_title"])}</strong>
          <span class="file" hidden></span>
          <small>{e(w["document_hint"])}</small>
          <input type="file" id="file-doc" accept="application/pdf,image/*">
        </label>
      </div>
      <div class="consent-card" id="consent-box">
        <div class="consent-head"><svg><use href="#i-shield"/></svg>
          <strong id="consent-title"></strong>
          <span class="version">{e(w["consent_version"].format(textos.version))}</span></div>
        <p class="consent-text" id="consent-text"></p>
        <label class="consent"><input type="checkbox" id="consent">
          <span id="consent-check"></span></label>
        {casilla_nube}
      </div>
      <div class="checks">{checks}</div>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="exit">
      <svg><use href="#i-exit"/></svg>{e(w["exit_label"])}</button>
    <span class="stat" id="missing"></span>
    <span class="error" id="start-error" hidden></span>
    <button type="button" class="btn btn-primary" id="go" disabled>
      <svg><use href="#i-play"/></svg><span>{e(w["start_button"])}</span></button>
  </footer>
</section>"""
    claves = (
        "lead_upload", "lead_record", "owner_invalid", "change_file", "mic_denied",
        "missing_owner", "missing_audio", "missing_document", "missing_consent", "ready",
        "start_button", "record_button", "closed_text",
    )
    palabras = json.dumps({c: w[c] for c in claves}, ensure_ascii=False)
    # Los dos textos van a la página y el modo elige cuál se ve: el que se lee al
    # paciente sale del pack, nunca del código.
    autorizaciones = json.dumps(
        {
            modo: {
                "title": t.title,
                "text": t.text,
                "check": t.check,
                "cloud": {"text": t.cloud.text, "check": t.cloud.check},
            }
            for modo, t in sorted(textos.modes.items())
        },
        ensure_ascii=False,
    )
    script = (
        f"const TOKEN = {json.dumps(token)};\nconst W = {palabras};\n"
        f"const CONSENT = {autorizaciones};\n{_JS_START}"
    )
    enlace = license_chip(w, license) + user_chip(w, professional)
    script += identity_script(w)
    if pending is not None:
        script += _JS_RECOVER
    if settings:
        enlace += (
            f'\n    <button type="button" class="head-link" id="to-settings">'
            f'<svg><use href="#i-gear"/></svg>{e(w["settings_label"])}</button>'
        )
        script += JS_TO_SETTINGS
    return _page(
        brand=brand, title=w["title_start"], place=w["header_place"], moment=moment,
        pill=w["pill_ready"], pill_class="pill", body=body, script=script, head_link=enlace,
    )


def _working_page(w: AppWording, brand: str, token: str, owner: str, moment: datetime) -> str:
    e = html.escape
    pasos = "\n".join(
        f'        <li><span class="mark-step"></span>{e(paso)}</li>' for paso in w.steps
    )
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card">
      <h2>{e(w["title_working"])}</h2>
      <p class="lead">{e(w["working_lead"].format(owner))}</p>
      <div class="bar-progress"><span id="progress-bar" style="width: 0%"></span></div>
      <ol class="steps">
{pasos}
      </ol>
    </div>
  </div>
  <footer class="bar">
    <span class="stat"><b id="done-count">0</b>{e(w["working_count"].format(len(w.steps)))}</span>
  </footer>
</section>"""
    script = f"const TOKEN = {json.dumps(token)};\n{_JS_WORKING}"
    return _page(
        brand=brand, title=w["title_working"], place=w["header_place"], moment=moment,
        pill=w["pill_working"], pill_class="pill", body=body, script=script,
    )


def _problem_page(
    w: AppWording, brand: str, token: str, problem: AppProblem, moment: datetime
) -> str:
    e = html.escape
    clave = problem.kind if problem.kind in (
        "capture", "cache_key", "transcription", "processing"
    ) else "processing"
    comando = w[f"problem_{clave}_command"]
    bloque_comando = (
        f'<div class="cmd"><code id="cmd">{e(comando)}</code>'
        f'<button type="button" class="icon-btn" id="copy-cmd">'
        f'<svg><use href="#i-copy"/></svg></button></div>'
        if comando
        else ""
    )
    detalle = f'<p class="hint">{e(problem.detail)}</p>' if problem.detail else ""
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card">
      <div class="alert-head">
        <span class="alert-icon"><svg><use href="#i-alert"/></svg></span>
        <div><h2>{e(w[f"problem_{clave}_title"])}</h2></div>
      </div>
      <p>{e(w[f"problem_{clave}_detail"])}</p>
      {bloque_comando}
      {detalle}
      <p class="hint">{e(w["nothing_saved"])}</p>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="back">{e(w["back_label"])}</button>
    <button type="button" class="btn btn-primary" id="retry">{e(w["retry_label"])}</button>
  </footer>
</section>"""
    script = f"const TOKEN = {json.dumps(token)};\n{_JS_PROBLEM}"
    return _page(
        brand=brand, title=w["title_problem"], place=w["header_place"], moment=moment,
        pill=w["pill_problem"], pill_class="pill fallback", body=body, script=script,
    )


_JS_START = """
const owner = document.getElementById('owner');
const ownerOk = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
let mode = 'upload';
let audioFile = null;
let docFile = null;

function pick(dropId, inputId, set) {
  const drop = document.getElementById(dropId);
  const input = document.getElementById(inputId);
  const show = file => {
    set(file);
    drop.classList.add('filled');
    const label = drop.querySelector('.file');
    const megas = (file.size / 1048576).toFixed(1).replace('.', ',');
    label.textContent = file.name + ' · ' + megas + ' MB';
    label.hidden = false;
    drop.querySelector('small').textContent = W.change_file;
    update();
  };
  input.addEventListener('change', () => { if (input.files[0]) { show(input.files[0]); } });
  drop.addEventListener('dragover', ev => { ev.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', ev => {
    ev.preventDefault();
    drop.classList.remove('over');
    if (ev.dataTransfer.files[0]) { show(ev.dataTransfer.files[0]); }
  });
}
pick('drop-audio', 'file-audio', file => { audioFile = file; });
pick('drop-doc', 'file-doc', file => { docFile = file; });

function missing() {
  const value = owner.value.trim();
  if (!value) { return W.missing_owner; }
  if (!ownerOk.test(value)) { return W.owner_invalid; }
  if (mode === 'upload' && !audioFile) { return W.missing_audio; }
  if (!docFile) { return W.missing_document; }
  if (!document.getElementById('consent').checked) {
    return W.missing_consent;
  }
  return '';
}

function update() {
  const text = missing();
  document.getElementById('missing').textContent = text || W.ready;
  document.getElementById('go').disabled = text !== '';
}
owner.addEventListener('input', update);
document.getElementById('consent').addEventListener('change', () => {
  document.getElementById('consent-box').classList.toggle(
    'done', document.getElementById('consent').checked);
  update();
});
const casillaNube = document.getElementById('cloud');
if (casillaNube) {
  casillaNube.addEventListener('change', () => {
    document.getElementById('cloud-box').classList.toggle('done', casillaNube.checked);
    document.getElementById('cloud-note').hidden = casillaNube.checked;
  });
}

function setMode(next) {
  mode = next;
  document.getElementById('mode-upload').setAttribute('aria-pressed', String(mode === 'upload'));
  document.getElementById('mode-record').setAttribute('aria-pressed', String(mode === 'record'));
  document.getElementById('drop-audio').hidden = mode !== 'upload';
  document.getElementById('mic-box').hidden = mode !== 'record';
  // Otro modo es otro texto: lo marcado para uno no vale para el otro.
  const texto = CONSENT[mode === 'record' ? 'recorded' : 'uploaded'];
  document.getElementById('consent-title').textContent = texto.title;
  document.getElementById('consent-text').textContent = casillaNube
    ? texto.text + ' ' + texto.cloud.text : texto.text;
  document.getElementById('consent-check').textContent = texto.check;
  document.getElementById('consent').checked = false;
  document.getElementById('consent-box').classList.remove('done');
  if (casillaNube) {
    document.getElementById('cloud-check').textContent = texto.cloud.check;
    casillaNube.checked = false;
    document.getElementById('cloud-box').classList.remove('done');
    document.getElementById('cloud-note').hidden = false;
  }
  document.getElementById('lead').textContent = mode === 'upload' ? W.lead_upload : W.lead_record;
  document.querySelector('#go span').textContent =
    mode === 'upload' ? W.start_button : W.record_button;
  update();
}
document.getElementById('mode-upload').addEventListener('click', () => setMode('upload'));
document.getElementById('mode-record').addEventListener('click', () => setMode('record'));

function post(path, body, headers) {
  return fetch('/' + TOKEN + path, { method: 'POST', headers: headers || {}, body: body });
}

function fail(text) {
  const box = document.getElementById('start-error');
  box.textContent = text;
  box.hidden = false;
}

async function send(audio, name) {
  if (audio) {
    const sent = await post('/audio', audio, { 'X-Name': name });
    if (!sent.ok) { fail((await sent.json()).error); return; }
  }
  const doc = await post('/documento', docFile, { 'X-Name': docFile.name });
  if (!doc.ok) { fail((await doc.json()).error); return; }
  const body = JSON.stringify({
    owner: owner.value.trim(),
    recorded: mode === 'record',
    consent: document.getElementById('consent').checked,
    cloud: !!document.getElementById('cloud')?.checked,
  });
  const go = await post('/empezar', body, { 'Content-Type': 'application/json' });
  if (!go.ok) { fail((await go.json()).error); return; }
  location.reload();
}

document.getElementById('go').addEventListener('click', () => {
  if (mode === 'upload') { send(audioFile, audioFile.name); } else { send(null, ''); }
});
document.getElementById('exit').addEventListener('click', async () => {
  await post('/salir', '');
  document.body.textContent = W.closed_text;
});
setMode('upload');
"""

_JS_WORKING = """
const CHECK = '<svg><use href="#i-check"/></svg>';
async function poll() {
  const answer = await fetch('/' + TOKEN + '/estado');
  const status = await answer.json();
  if (status.state !== 'working') { location.reload(); return; }
  const items = document.querySelectorAll('.steps li');
  let done = 0;
  status.steps.forEach((ok, i) => {
    items[i].className = ok ? 'done' : '';
    items[i].querySelector('.mark-step').innerHTML = ok ? CHECK : '';
    if (ok) { done += 1; }
  });
  const next = status.steps.indexOf(false);
  if (next >= 0) { items[next].className = 'now'; }
  document.getElementById('progress-bar').style.width = (done / status.steps.length * 100) + '%';
  document.getElementById('done-count').textContent = String(done);
  setTimeout(poll, 1000);
}
poll();
"""




def _where_block(w: AppWording, tools: AppTools) -> str:
    """Qué sale de este equipo y a qué país (plan 85b). El país es lo último de «dónde»
    —«OpenAI · Estados Unidos»—: lo monta quien lanza, y así la aplicación no conoce
    los proveedores."""
    e = html.escape
    solo_aqui = tools.local_only()
    destinos = () if solo_aqui else tools.destinations
    paises = list(dict.fromkeys(donde.split(" · ")[-1] for _que, donde in tools.destinations))
    if destinos:
        saliente = e(w["where_out"].format(", ".join(paises)))
        estado = f'<span class="status warn" id="where-status">{saliente}</span>'
    else:
        estado = f'<span class="status ok" id="where-status">{e(w["where_local"])}</span>'
    etiquetas = {"transcription": w["where_transcription"], "note": w["where_note"]}
    clase = "cloud off" if solo_aqui else "cloud"
    filas = "".join(
        f'<li><span>{e(etiquetas.get(que, que))}</span><b class="{clase}">{e(donde)}</b></li>'
        for que, donde in tools.destinations
    )
    return f"""<div class="setting">
          <div class="setting-head"><svg><use href="#i-key"/></svg><h3>{e(w["where_title"])}</h3>
            {estado}</div>
          <p>{e(w["where_lead"])}</p>
          <div class="toggle-row"><button type="button" class="toggle" role="switch"
            aria-checked="{"true" if solo_aqui else "false"}" id="local-only"></button>
            <strong>{e(w["where_toggle"])}</strong></div>
          <ul class="where" id="where">
            {filas}
            <li><span>{e(w["where_fallback"])}</span><b>{e(w["where_here"])}</b></li>
            <li><span>{e(w["where_reader"])}</span><b>{e(w["where_here"])}</b></li>
          </ul>
        </div>"""


def _download_block(w: AppWording, download: str) -> str:
    e = html.escape
    corriendo = download == "running"
    boton = w["model_downloading"] if corriendo else w["model_download"]
    return f"""<div class="download-bar"><span id="dl-bar" style="width: 0%"></span></div>
          <div class="download-meta"><span id="dl-text"></span>
            <span>{e(w["model_keep_open"])}</span></div>
          <div class="key-row"><button type="button" class="btn btn-primary" id="dl-button"
            {"disabled" if corriendo else ""}><svg><use href="#i-download"/></svg>
            <span>{e(boton)}</span></button></div>"""


def _settings_script(w: AppWording, token: str, tools: AppTools) -> str:
    claves = (
        "key_ok", "key_fail", "key_saved", "key_status_bad", "key_status_file",
        "model_progress", "model_downloading", "model_download", "model_failed",
    )
    textos = json.dumps({c: w[c] for c in claves}, ensure_ascii=False)
    return (
        f"const TOKEN = {json.dumps(token)};\nconst W = {textos};\n{_JS_SETTINGS}"
        f"\n{_JS_LOCAL_ONLY}"
    )


_JS_LOCAL_ONLY = """
document.getElementById('local-only').addEventListener('click', async () => {
  const on = document.getElementById('local-only').getAttribute('aria-checked') !== 'true';
  await fetch('/' + TOKEN + '/solo-local', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ on: on }),
  });
  location.reload();
});
"""


def _model_page(
    w: AppWording, brand: str, token: str, tools: AppTools, moment: datetime, download: str
) -> str:
    """El primer arranque sin modelo y sin clave de nube: no hay con qué transcribir.
    Dos salidas, descargar o ir a poner la clave; ninguna obliga a la otra."""
    e = html.escape
    corriendo = download == "running"
    boton = w["model_downloading"] if corriendo else w["model_download"]
    entrada = w["model_lead"].format(tools.model_total_mb)
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card">
      <div class="alert-head">
        <span class="alert-icon warn"><svg><use href="#i-download"/></svg></span>
        <div><h2>{e(w["problem_model_title"])}</h2>
        <p class="lead" style="margin:.2rem 0 0">{e(entrada)}</p></div>
      </div>
      <div class="download-bar"><span id="dl-bar" style="width: 0%"></span></div>
      <div class="download-meta"><span id="dl-text"></span>
        <span>{e(w["model_keep_open"])}</span></div>
      <p class="hint">{e(w["model_text"])}</p>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="to-settings">
      {e(w["model_to_settings"])}</button>
    <button type="button" class="btn btn-primary" id="dl-button" {"disabled" if corriendo else ""}>
      <svg><use href="#i-download"/></svg><span>{e(boton)}</span></button>
  </footer>
</section>"""
    return _page(
        brand=brand, title=w["title_first_run"], place=w["header_place"], moment=moment,
        pill=w["pill_preparing"], pill_class="pill", body=body,
        script=_settings_script(w, token, tools) + JS_TO_SETTINGS,
    )



































@contextlib.contextmanager
def _plain_audio(audio: Path, key: bytes) -> Iterator[Path]:
    """El audio de la sesión, como WAV de 16 kHz en un temporal propio. La grabación del
    vivo ya es PCM a 16 kHz; lo subido se descifra y se convierte."""
    with plain_directory() as carpeta:
        destino = carpeta / "audio-16k.wav"
        if audio.name == RECORDING_NAME:
            recording_wav(audio, key, destino)
            yield destino
            return
        with plain_copy(audio, key) as original:
            to_wav(original, destino)
        yield destino


def _read_sealed(reader: DocumentReader, document: Path, key: bytes) -> str:
    with plain_copy(document, key) as historia:
        return reader.read(historia)



























_JS_SETTINGS = """
function post(path, body) {
  return fetch('/' + TOKEN + path, { method: 'POST', body: body || '' });
}
const back = document.getElementById('back');
if (back) {
  back.addEventListener('click', async () => { await post('/volver'); location.reload(); });
}
function show(ok, text) {
  const good = document.getElementById('key-ok');
  const bad = document.getElementById('key-fail');
  good.hidden = !ok; bad.hidden = ok;
  (ok ? good : bad).querySelector('span').textContent = text;
}
async function key(path) {
  const value = document.getElementById('key-input').value.trim();
  if (!value) return;
  const r = await (await post(path, JSON.stringify({ key: value }))).json();
  const status = document.getElementById('key-status');
  if (r.ok) {
    show(true, path === '/clave-guardar' ? W.key_saved : W.key_ok);
    if (path === '/clave-guardar') {
      status.className = 'status ok'; status.textContent = W.key_status_file;
    }
  } else {
    show(false, W.key_fail.replace('{0}', r.reason));
    status.className = 'status fail'; status.textContent = W.key_status_bad;
  }
}
const test = document.getElementById('key-test');
if (test) {
  test.addEventListener('click', () => key('/clave-probar'));
  document.getElementById('key-save').addEventListener('click', () => key('/clave-guardar'));
}
const dl = document.getElementById('dl-button');
async function poll() {
  const r = await (await fetch('/' + TOKEN + '/modelo')).json();
  const mb = (n) => Math.round(n / 1048576);
  const bar = document.getElementById('dl-bar');
  const text = document.getElementById('dl-text');
  if (r.total > 0) bar.style.width = Math.min(100, 100 * r.bytes / r.total) + '%';
  if (r.state === 'running') {
    text.textContent = W.model_progress.replace('{0}', mb(r.bytes)).replace('{1}', mb(r.total));
    setTimeout(poll, 1000);
  } else if (r.state === 'done') {
    await post('/reintentar');
    location.reload();
  } else if (r.state === 'failed') {
    text.textContent = W.model_failed.replace('{0}', r.detail);
    dl.disabled = false; dl.querySelector('span').textContent = W.model_download;
  }
}
if (dl) {
  dl.addEventListener('click', async () => {
    dl.disabled = true; dl.querySelector('span').textContent = W.model_downloading;
    await post('/modelo');
    poll();
  });
  if (dl.disabled) poll();
}
"""


_JS_PROBLEM = """
function post(path) { return fetch('/' + TOKEN + path, { method: 'POST', body: '' }); }
document.getElementById('back').addEventListener('click', async () => {
  await post('/volver');
  location.reload();
});
document.getElementById('retry').addEventListener('click', async () => {
  await post('/reintentar');
  location.reload();
});
const copyButton = document.getElementById('copy-cmd');
if (copyButton) {
  copyButton.addEventListener('click', () => {
    navigator.clipboard.writeText(document.getElementById('cmd').textContent);
  });
}
"""


def _keep_mine(
    draft: ReviewDraft, mine: Mapping[str, str], note_sections: Sequence[SectionSpec]
) -> ReviewDraft:
    """Un campo que el profesional escribió en vivo y que el análisis final no trae
    no puede perderse: se añade vacío del lado del modelo, y la revisión lo registra
    como editado (`review_note`), que es lo que fue."""
    presentes = {spec.section_id for spec, _texto in draft.note.fields}
    faltan = tuple(
        (spec, "")
        for spec in note_sections
        if spec.section_id in mine and spec.section_id not in presentes
    )
    if not faltan:
        return draft
    return replace(draft, note=replace(draft.note, fields=draft.note.fields + faltan))


def _live_status(session: _Session, path: str) -> dict[str, Any]:
    if session.state != "live" or session.live is None:
        return {"state": session.state}
    try:
        desde = max(0, int(parse_qs(urlsplit(path).query).get("desde", ["0"])[0]))
    except ValueError:
        desde = 0
    estado = session.live.snapshot(desde)
    for linea in estado["lines"]:
        if linea["kind"] != "pause":
            linea["kind"] = line_kind(linea["text"])
    return {"state": "live", **estado}


def _live_page(
    w: AppWording,
    settings: UISettings,
    token: str,
    owner: str,
    moment: datetime,
    note_sections: Sequence[SectionSpec],
    view: HistoryView | None,
) -> str:
    e = html.escape
    r = settings.wording
    copiar = (
        f'<button type="button" class="icon-btn copy" data-label="{e(r.copy_label)}">'
        f'<svg><use href="#i-copy"/></svg><span>{e(r.copy_label)}</span></button>'
    )
    campos = "\n".join(
        f'      <article class="field waiting" data-field-id="{e(spec.section_id)}">'
        f'<div class="field-head"><h3>{e(spec.label)}</h3>'
        f'<span class="field-state">{e(w["state_waiting"])}</span>{copiar}</div>\n'
        f'        <textarea placeholder="{e(w["waiting_placeholder"])}"></textarea></article>'
        for spec in note_sections
    )
    if view is not None and view.content_type == "application/pdf":
        historia = (
            f'<iframe class="history-frame" src="/{token}/historia" '
            f'title="{e(r.history_tab)}"></iframe>'
        )
    elif view is not None and view.content_type is not None:
        historia = f'<img class="history-image" src="/{token}/historia" alt="{e(r.history_tab)}">'
    else:
        historia = ""
    barras = "".join("<span></span>" for _ in range(12))
    body = f"""<main>
  <section class="column">
    <div class="column-head">
      <div><h2>{e(r.note_heading)}</h2><p>{e(w["live_hint"])}</p></div>
      <button type="button" class="btn" id="copy-all" data-label="{e(r.copy_all_label)}">
        <svg><use href="#i-copy"/></svg><span>{e(r.copy_all_label)}</span>
      </button>
    </div>
    <div class="scroll">
{campos}
    </div>
  </section>
  <section class="column">
    <div class="tabs" role="tablist">
      <button type="button" class="tab" data-pane="live" aria-selected="true">
        <svg><use href="#i-chat"/></svg>{e(w["live_tab"])}</button>
      <button type="button" class="tab" data-pane="codes" aria-selected="false">
        <svg><use href="#i-codes"/></svg>{e(r.codes_tab)}
        <span class="count" id="codes-count">0</span>
        <span class="count new" id="codes-new" hidden></span></button>
      <button type="button" class="tab" data-pane="history" aria-selected="false">
        <svg><use href="#i-doc"/></svg>{e(r.history_tab)}</button>
    </div>
    <div class="pane scroll" id="pane-live">
      <div class="live-banner" id="banner-live">
        <svg><use href="#i-spark"/></svg>{e(w["banner_live"])}</div>
      <div class="live-banner offline" id="banner-offline" hidden>
        <svg><use href="#i-offline"/></svg>{e(w["banner_offline"])}</div>
      <div class="chat" id="chat"></div>
      <div class="listening" id="listening"><span class="dot-live"></span>
        <span id="listening-text">{e(w["listening"])}</span></div>
    </div>
    <div class="pane scroll" id="pane-codes" hidden>
      <div class="legend"><span>{e(r.origin_today_label)}</span>
        <span class="h">{e(r.origin_history_label)}</span>
        <span class="b">{e(r.origin_both_label)}</span></div>
      <div id="codes"></div>
    </div>
    <div class="pane" id="pane-history" hidden>
{historia}
    </div>
  </section>
</main>
<footer class="bar">
  <div class="rec-strip">
    <span class="rec-small"></span>
    <span class="rec-clock" id="clock">00:00</span>
    <div class="meter">{barras}</div>
    <span class="note" id="strip-note">{e(w["strip_recording"])}</span>
  </div>
  <span class="error" id="live-error" hidden></span>
  <button type="button" class="btn btn-pause" id="pause">
    <svg><use href="#i-pause"/></svg><span>{e(w["pause_label"])}</span></button>
  <button type="button" class="btn btn-primary btn-stop" id="stop">
    <svg><use href="#i-stop"/></svg>{e(w["stop_button"])}</button>
</footer>"""
    claves = (
        "state_ai", "state_writing", "state_mine", "state_waiting", "listening",
        "listening_offline", "strip_recording", "strip_paused", "pause_label",
        "resume_label", "pill_paused", "recording_pill", "pause_mark", "new_label",
        "new_count", "mic_denied",
    )
    textos = {c: w[c] for c in claves}
    textos.update(
        {
            "copied_label": r.copied_label,
            "include_label": r.include_label,
            "today": r.origin_today_label,
            "history": r.origin_history_label,
            "both": r.origin_both_label,
        }
    )
    script = (
        f"const TOKEN = {json.dumps(token)};\n"
        f"const W = {json.dumps(textos, ensure_ascii=False)};\n{_JS_LIVE}"
    )
    return _page(
        brand=r.brand, title=w["live_title"], place=f'{owner} · {w["header_place"]}',
        moment=moment, pill=w["recording_pill"], pill_class="pill rec-pill",
        body=body, script=script, live=True,
    )


_JS_LIVE = """
const grow = t => { t.style.height = 'auto'; t.style.height = t.scrollHeight + 'px'; };
const WORKLET = 'class Tap extends AudioWorkletProcessor {' +
  ' process(inputs) { const ch = inputs[0][0]; if (ch) { this.port.postMessage(ch.slice(0)); }' +
  ' return true; } } registerProcessor("aida-tap", Tap);';
let since = 0;
let version = -1;
let paused = false;
let pending = [];
let context = null;
let elapsed = 0;
const timers = new Map();

function post(path, body, json) {
  const headers = json ? { 'Content-Type': 'application/json' } : {};
  return fetch('/' + TOKEN + path, { method: 'POST', headers: headers, body: body });
}

function showError(text) {
  const box = document.getElementById('live-error');
  box.textContent = text;
  box.hidden = false;
}

function age(seconds) {
  return seconds < 60 ? seconds + ' s' : Math.floor(seconds / 60) + ' min';
}

function setState(field, kind, text) {
  const state = field.querySelector('.field-state');
  state.className = 'field-state' + (kind ? ' ' + kind : '');
  state.replaceChildren();
  const icon = kind === 'writing' ? 'i-spark' : kind === 'mine' ? 'i-lock' : '';
  if (icon) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', '#' + icon);
    svg.appendChild(use);
    state.appendChild(svg);
  }
  state.appendChild(document.createTextNode(text));
}

document.querySelectorAll('.field').forEach(field => {
  const text = field.querySelector('textarea');
  text.addEventListener('input', () => {
    grow(text);
    field.dataset.mine = '1';
    field.classList.remove('waiting', 'writing');
    field.classList.add('edited');
    setState(field, 'mine', W.state_mine);
    clearTimeout(timers.get(field));
    timers.set(field, setTimeout(() => {
      post('/campo', JSON.stringify({ id: field.dataset.fieldId, text: text.value }), true);
    }, 1200));
  });
  const button = field.querySelector('.copy');
  button.addEventListener('click', () => copy(text.value, button));
});

function copy(text, button) {
  navigator.clipboard.writeText(text).then(() => {
    const label = button.querySelector('span');
    label.textContent = W.copied_label;
    setTimeout(() => { label.textContent = button.dataset.label; }, 1500);
  });
}

document.getElementById('copy-all').addEventListener('click', event => {
  const parts = [];
  document.querySelectorAll('.field').forEach(field => {
    const value = field.querySelector('textarea').value.trim();
    if (value) { parts.push(field.querySelector('h3').textContent + '\\n' + value); }
  });
  copy(parts.join('\\n\\n'), event.currentTarget);
});

document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(t => {
    t.setAttribute('aria-selected', String(t === tab));
  });
  document.querySelectorAll('.pane').forEach(p => {
    p.hidden = p.id !== 'pane-' + tab.dataset.pane;
  });
}));

function updateField(f) {
  const field = document.querySelector('.field[data-field-id="' + f.id + '"]');
  if (!field || field.dataset.mine === '1') { return; }
  const text = field.querySelector('textarea');
  if (document.activeElement === text) { return; }
  if (text.value !== f.text) { text.value = f.text; grow(text); }
  field.classList.toggle('waiting', !f.text);
  field.classList.toggle('writing', f.state === 'writing');
  if (f.state === 'writing') {
    setState(field, 'writing', W.state_writing);
  } else if (f.text) {
    setState(field, '', W.state_ai.replace('{}', age(f.age)));
  } else {
    setState(field, '', W.state_waiting);
  }
}

function addLine(line) {
  const chat = document.getElementById('chat');
  chat.querySelectorAll('.fresh').forEach(x => x.classList.remove('fresh'));
  const p = document.createElement('p');
  if (line.kind === 'pause') {
    p.className = 'pause-mark';
    p.textContent = W.pause_mark;
  } else {
    p.className = line.kind + (line.kind === 'a' ? ' fresh' : '');
    p.textContent = line.text;
  }
  chat.appendChild(p);
  const pane = document.getElementById('pane-live');
  pane.scrollTop = pane.scrollHeight;
}

function renderCodes(codes) {
  const box = document.getElementById('codes');
  box.replaceChildren();
  let fresh = 0;
  codes.forEach(c => {
    const card = document.createElement('article');
    card.className = 'code' + (c.included ? '' : ' off') + (c.fresh ? ' fresh' : '');
    card.dataset.origin = c.origin;
    const head = document.createElement('div');
    head.className = 'code-head';
    const id = document.createElement('span');
    id.className = 'code-id';
    id.textContent = c.code;
    const origin = document.createElement('span');
    origin.className = 'origin';
    origin.textContent = W[c.origin];
    head.append(id, origin);
    if (c.fresh) {
      fresh += 1;
      const badge = document.createElement('span');
      badge.className = 'badge new';
      badge.textContent = W.new_label;
      head.appendChild(badge);
    }
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'switch';
    toggle.setAttribute('aria-label', W.include_label);
    toggle.addEventListener('click', () => {
      card.classList.toggle('off');
      const body = JSON.stringify({ code: c.code, included: !card.classList.contains('off') });
      post('/codigo', body, true);
    });
    head.appendChild(toggle);
    const desc = document.createElement('p');
    desc.className = 'code-desc';
    desc.textContent = c.description;
    card.append(head, desc);
    if (c.evidence) {
      const ev = document.createElement('p');
      ev.className = 'code-evidence';
      ev.textContent = c.evidence;
      card.appendChild(ev);
    }
    box.appendChild(card);
  });
  document.getElementById('codes-count').textContent = String(codes.length);
  const badge = document.getElementById('codes-new');
  badge.hidden = fresh === 0;
  badge.textContent = W.new_count.replace('{}', fresh);
}

async function poll() {
  try {
    const answer = await fetch('/' + TOKEN + '/vivo?desde=' + since);
    const s = await answer.json();
    if (s.state !== 'live') { location.reload(); return; }
    since = s.line_count;
    s.lines.forEach(addLine);
    document.getElementById('banner-live').hidden = s.offline;
    document.getElementById('banner-offline').hidden = !s.offline;
    document.getElementById('listening-text').textContent =
      s.offline ? W.listening_offline : W.listening;
    if (s.version !== version) {
      version = s.version;
      s.fields.forEach(updateField);
      renderCodes(s.codes);
    }
  } catch (err) {
    // un sondeo perdido no importa: el siguiente trae todo lo que falte
  }
  setTimeout(poll, 1000);
}

function flush() {
  if (!pending.length || !context) { return; }
  const total = pending.reduce((n, a) => n + a.length, 0);
  const all = new Float32Array(total);
  let offset = 0;
  pending.forEach(a => { all.set(a, offset); offset += a.length; });
  pending = [];
  const ratio = context.sampleRate / 16000;
  const n = Math.floor(all.length / ratio);
  const out = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * ratio);
    const b = Math.min(all.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = a; j < b; j++) { sum += all[j]; }
    const v = sum / Math.max(1, b - a);
    out[i] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
  }
  post('/pcm', out.buffer, false);
}

async function startAudio() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  } catch (err) {
    showError(W.mic_denied);
    return;
  }
  context = new AudioContext();
  const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
  await context.audioWorklet.addModule(url);
  const source = context.createMediaStreamSource(stream);
  const tap = new AudioWorkletNode(context, 'aida-tap');
  tap.port.onmessage = ev => { if (!paused) { pending.push(ev.data); } };
  source.connect(tap);
  const analyser = context.createAnalyser();
  analyser.fftSize = 64;
  source.connect(analyser);
  const bins = new Uint8Array(analyser.frequencyBinCount);
  const bars = document.querySelectorAll('.meter span');
  const draw = () => {
    analyser.getByteFrequencyData(bins);
    bars.forEach((bar, i) => { bar.style.height = Math.max(8, bins[i + 2] / 2.55) + '%'; });
    requestAnimationFrame(draw);
  };
  draw();
  setInterval(flush, 1000);
  setInterval(() => {
    if (paused) { return; }
    elapsed += 1;
    const clock = String(Math.floor(elapsed / 60)).padStart(2, '0') + ':' +
      String(elapsed % 60).padStart(2, '0');
    document.getElementById('clock').textContent = clock;
    document.getElementById('pill-text').textContent = W.recording_pill + ' ' + clock;
  }, 1000);
}

document.getElementById('pause').addEventListener('click', () => {
  paused = !paused;
  if (paused) { pending = []; }
  post('/pausa', JSON.stringify({ paused: paused }), true);
  document.body.classList.toggle('paused', paused);
  const pill = document.getElementById('head-pill');
  pill.classList.toggle('rec-pill', !paused);
  pill.classList.toggle('pause-pill', paused);
  if (paused) { document.getElementById('pill-text').textContent = W.pill_paused; }
  document.getElementById('strip-note').textContent = paused ? W.strip_paused : W.strip_recording;
  document.querySelector('#pause span').textContent = paused ? W.resume_label : W.pause_label;
  document.querySelector('#pause use').setAttribute('href', paused ? '#i-play' : '#i-pause');
  document.getElementById('listening').hidden = paused;
});

document.getElementById('stop').addEventListener('click', async () => {
  flush();
  await post('/terminar', '', false);
  location.reload();
});

startAudio();
poll();
"""


_EXTRA_CSS_SALUD = f"{_CSS_EXTRA}\n{_CSS_AJUSTES_SALUD}\n{_CSS_CONSENT}"
_EXTRA_ICONOS_SALUD = f"{_ICONS_EXTRA}\n{_ICONS_SETTINGS_EXTRA}\n{_ICONS_CONSENT}"
_EXTRA_ICONOS_VIVO = f"{_EXTRA_ICONOS_SALUD}\n{_ICONS_LIVE_EXTRA}"


def _page(
    *,
    brand: str,
    title: str,
    place: str,
    moment: datetime,
    pill: str,
    pill_class: str,
    body: str,
    script: str,
    live: bool = False,
    head_link: str = "",
) -> str:
    """Las pantallas de la consulta sobre el marco del armazón (101a): el CSS y los iconos
    de salud entran detrás de los de la plataforma. `live` añade lo que la grabación en
    vivo necesita, y nada más."""
    if live:
        return page_frame(
            brand=brand, title=title, place=place, moment=moment, pill=pill,
            pill_class=pill_class, body=body, script=script, head_link=head_link,
            extra_css=f"{_CSS_EXTRA}\n{_CSS_LIVE}", extra_icons=_EXTRA_ICONOS_VIVO,
        )
    return page_frame(
        brand=brand, title=title, place=place, moment=moment, pill=pill,
        pill_class=pill_class, body=body, script=script, head_link=head_link,
        extra_css=_EXTRA_CSS_SALUD, extra_icons=_EXTRA_ICONOS_SALUD,
    )


def _settings_cards(
    w: AppWording,
    tools: AppTools,
    download: str,
    profiles: Sequence[Professional],
    license: LicenseState | None,
) -> str:
    """Las tarjetas de Ajustes que son de la consulta. Cada una se consulta al abrir: lo que
    enseña tiene que ser lo que hay ahora. Los perfiles y la licencia son de la plataforma:
    se piden donde toca, para que el orden de la página sea el de siempre."""
    e = html.escape
    origen = tools.cloud_key_source()
    estado_clave = {
        "env": ("ok", w["key_status_env"]),
        "file": ("ok", w["key_status_file"]),
    }.get(origen or "", ("warn", w["key_status_none"]))
    nota_entorno = f'<p>{e(w["key_env_note"])}</p>' if origen == "env" else ""
    modelo_ok = tools.model_available()
    if modelo_ok:
        instalado = w["model_ok"].format(tools.model_name)
        estado_modelo = f'<span class="status ok">{e(instalado)}</span>'
        descarga = ""
    else:
        estado_modelo = f'<span class="status warn">{e(w["model_missing"])}</span>'
        descarga = _download_block(w, download)
    try:
        version = tools.reader_version()
        lector = (
            ("ok", w["reader_ok"].format(version)) if version else ("fail", w["reader_missing"])
        )
    except Exception:
        lector = ("fail", w["reader_missing"])
    desde = tools.cache_key_since()
    cifrado = (
        w["cipher_text"].format(datetime.fromtimestamp(desde).strftime("%d/%m/%Y"))
        if desde is not None
        else w["cipher_env"]
    )
    return f"""        <div class="setting">
          <div class="setting-head"><svg><use href="#i-key"/></svg><h3>{e(w["key_title"])}</h3>
            <span class="status {estado_clave[0]}" id="key-status">{e(estado_clave[1])}</span></div>
          <p>{e(w["key_text"])}</p>
          {nota_entorno}
          <div class="key-row">
            <input class="input" id="key-input" type="password" autocomplete="off">
            <button type="button" class="btn" id="key-test">{e(w["key_test"])}</button>
            <button type="button" class="btn btn-primary" id="key-save">{e(w["key_save"])}</button>
          </div>
          <div class="result ok" id="key-ok" hidden><svg><use href="#i-check"/></svg>
            <span></span></div>
          <div class="result fail" id="key-fail" hidden><svg><use href="#i-alert"/></svg>
            <span></span></div>
        </div>
        {_where_block(w, tools)}
        <div class="setting">
          <div class="setting-head"><svg><use href="#i-download"/></svg>
            <h3>{e(w["model_title"])}</h3>{estado_modelo}</div>
          <p>{e(w["model_text"])}</p>
          {descarga}
        </div>
        <div class="setting">
          <div class="setting-head"><svg><use href="#i-doc"/></svg><h3>{e(w["reader_title"])}</h3>
            <span class="status {lector[0]}">{e(lector[1])}</span></div>
          <p>{e(w["reader_text"])}</p>
        </div>
        {profiles_block(w, profiles)}
        {license_block(w, license)}
        <div class="setting">
          <div class="setting-head"><svg><use href="#i-key"/></svg><h3>{e(w["cipher_title"])}</h3>
            <span class="status ok">{e(w["cipher_ok"])}</span></div>
          <p>{e(cifrado)}</p>
        </div>"""


def _settings_page(
    w: AppWording,
    brand: str,
    token: str,
    tools: AppTools,
    moment: datetime,
    download: str,
    *,
    profiles: Sequence[Professional] = (),
    professional: Professional | None = None,
    license: LicenseState | None = None,
) -> str:
    """Ajustes completos, con su marco de la plataforma. Se conserva la firma de antes: las
    pruebas y la aplicación la llaman así (101a, decisión 3)."""
    return settings_frame(
        w, brand, token, moment,
        cards=_settings_cards(w, tools, download, profiles, license),
        script=_settings_script(w, token, tools),
        professional=professional,
        license=license,
        extra_css=_EXTRA_CSS_SALUD,
        extra_icons=_EXTRA_ICONOS_SALUD,
    )


@dataclass
class _Session:
    """El estado de la consulta: lo que hace AIDA con una grabación, una subida o una
    revisión. Conserva professional y last_activity por compatibilidad con la firma
    preexistente y los tests que inspeccionan la sesión (101a, decisión 3)."""

    ready: AppReady | AppProblem
    professional: Professional | None = None
    last_activity: float = 0.0
    state: str = "start"
    workdir: Path | None = None
    audio: Path | None = None
    document: Path | None = None
    owner: str = ""
    start_index: int = 0
    drafted: bool = False
    draft: ReviewDraft | None = None
    view: HistoryView | None = None
    problem: AppProblem | None = None
    started: float = 0.0
    live: LiveSession | None = None
    mine: Mapping[str, str] = field(default_factory=dict)
    excluded: frozenset[str] = frozenset()
    consent: ConsentRecord | None = None
    """La autorización de la sesión en curso (plan 82): la revisión la enseña."""
    history_seen: bool = False
    download: str = "idle"
    """`idle`, `running`, `done` o `failed`: la descarga del modelo sobrevive a cambiar
    de pantalla, y `clean` no la toca."""
    download_detail: str = ""

    def clean(self) -> None:
        """Lo subido o grabado no se queda en disco: el audio en claro es lo más sensible de
        una sesión, y la transcripción ya vive en la caché cifrada."""
        if self.live is not None:
            self.live.stop()
        self.live = None
        self.mine = {}
        self.excluded = frozenset()
        self.consent = None
        self.history_seen = False
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
        self.workdir = None
        self.audio = None
        self.document = None
        self.draft = None
        self.view = None
        self.drafted = False


_AJUSTES_RUTAS = ("ajustes", "clave-probar", "clave-guardar", "modelo", "solo-local")


class HealthArm:
    """El brazo de salud dentro del armazón (101a): su estado de consulta, sus rutas y sus
    pantallas. Quién atiende, la licencia y Ajustes son del armazón, que se los presta en
    `ShellContext`; el brazo no guarda nada de identidad."""

    def __init__(
        self,
        build: Callable[[], AppReady | AppProblem],
        *,
        arm_id: str,
        arm_name: str,
        ui: UISettings,
        wording: AppWording,
        tools: AppTools | None,
        consent_texts: ConsentTexts,
        auditoria: AuditLog,
        lock: threading.Lock,
        clock: Callable[[], float],
        now: Callable[[], datetime],
        raiz: Path,
        secciones: Sequence[SectionSpec],
        crear_vad: Callable[[], Callable[[Any], Any]],
        retention_days: int | None,
        interrupted_hours: int | None,
    ) -> None:
        self.id = arm_id
        self.name = arm_name
        self._build = build
        self._ui = ui
        self._wording = wording
        self._tools = tools
        self._textos = consent_texts
        self._auditoria = auditoria
        self._lock = lock
        self._clock = clock
        self._now = now
        self._raiz = raiz
        self._secciones = secciones
        self._crear_vad = crear_vad
        self._retention_days = retention_days
        self._interrupted_hours = interrupted_hours
        # Sin clave del equipo, una de esta ejecución: tampoco queda nada en claro.
        self._efimera = Fernet.generate_key()
        self._session = _Session(ready=build())

    # Estado y ciclo de vida: lo que el armazón necesita saber del brazo.

    def at_rest(self) -> bool:
        return self._session.state in ("start", "problem", "settings")

    def in_settings(self) -> bool:
        return self._session.state == "settings"

    def reset(self) -> None:
        self._session.professional = None
        self._session.clean()

    def rebuild(self) -> None:
        """Vuelve a montar todo: una clave guardada o un modelo descargado se notan sin cerrar
        AIDA (`I-152`)."""
        self._session.clean()
        self._session.ready = self._build()
        self._fijar_reposo()

    def on_identified(self, actor: Professional) -> None:
        """Tras abrir sesión: borra lo que cumplió su plazo, firmado por quien abre (85a, 87)."""
        self._session.professional = actor
        self._retener(actor)
        self._fijar_reposo()

    def on_unlocked(self) -> None:
        # Quien desbloquea es el mismo profesional: se recoge en la siguiente petición
        # (`_sincronizar`), que es la que trae el contexto del armazón.
        self._fijar_reposo()

    def _sincronizar(self, ctx: ShellContext) -> None:
        """Toma la identidad del armazón si el brazo no la tiene (tras desbloquear). Sin
        candado: el armazón ya lo tiene tomado al entregar el contexto."""
        if self._session.professional is None:
            self._session.professional = ctx.professional

    def _actor(self) -> Professional:
        """Quien firma lo que se registra, leído del estado del brazo (101a-fix, I-160). Sin
        profesional, revienta antes de escribir nada (`I-156`)."""

        def actor() -> Professional:
            if self._session.professional is None:
                raise IdentityError("no_actor", "no hay profesional en la sesión")
            return self._session.professional

        return actor()

    def _fijar_reposo(self) -> None:
        if isinstance(self._session.ready, AppProblem):
            self._session.state = "problem"
            self._session.problem = self._session.ready
        else:
            self._session.state = "start"
            self._session.problem = None

    def _clave(self, ready: AppReady | None) -> bytes:
        if ready is not None and ready.seal_key is not None:
            return ready.seal_key
        return self._efimera

    def _pendiente(self, ctx: ShellContext) -> Interrupted | None:
        """La grabación interrumpida más antigua de quien atiende, si la hay (87)."""
        ready = self._session.ready
        if ctx.professional is None or not isinstance(ready, AppReady) or ready.seal_key is None:
            return None
        for interrumpida in interrupted(self._raiz, ready.seal_key):
            if interrumpida.professional == ctx.professional.id:
                return interrumpida
        return None

    def _retener(self, actor: Professional) -> None:
        """Lo que cumplió su plazo: los textos de auditoría (85a) y las grabaciones
        interrumpidas (87), de cualquiera. Lo que nadie recuperó no se queda para siempre."""
        ahora = self._now().astimezone()
        if self._retention_days is not None:
            self._auditoria.purge_texts(
                ahora - timedelta(days=self._retention_days), actor=actor
            )
        if self._interrupted_hours is None:
            return
        llave = self._clave(
            self._session.ready if isinstance(self._session.ready, AppReady) else None
        )
        legibles = {i.folder: i for i in interrupted(self._raiz, llave)}
        for carpeta in expired(self._raiz, llave, self._interrupted_hours, ahora):
            empieza, legible = started_at(carpeta, llave)
            duenio = legibles.get(carpeta)
            shutil.rmtree(carpeta, ignore_errors=True)
            self._auditoria.record(
                "recording_purged",
                actor=actor,
                owner=duenio.owner if duenio is not None else "",
                data={
                    "started": empieza.isoformat(timespec="seconds"),
                    "readable": "true" if legible else "false",
                },
            )

    def _descargar(self, herramientas: AppTools) -> None:
        try:
            herramientas.download_model()
        except Exception as exc:  # frontera: una descarga fallida no tumba el http_server
            with self._lock:
                self._session.download = "failed"
                self._session.download_detail = f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            self._session.download = "done"
            self._session.download_detail = ""

    def _procesar(
        self,
        ready: AppReady,
        owner: str,
        audio: Path,
        document: Path,
        consent: ConsentRecord | None,
        profesional: Professional,
        mine: Mapping[str, str] | None = None,
        excluded: frozenset[str] = frozenset(),
    ) -> None:
        sesion = self._session
        try:
            # En claro solo aquí dentro, y en temporales propios que se borran al salir del
            # `with`, haya ido bien o no (plan 85a).
            with (
                _plain_audio(audio, self._clave(ready)) as wav,
                plain_copy(document, self._clave(ready)) as historia,
            ):
                prepared = ready.consultation.prepare(
                    owner=AgentId(owner),
                    audio_path=wav,
                    history_text=lambda: ready.reader.read(historia),
                    consent=consent,
                )
                draft = _keep_mine(
                    draft_for_screen(ready.consultation, prepared), mine or {}, self._secciones
                )
                vista = history_view(historia, prepared.history)
        except (TranscriptionError, ModelUnavailableError) as exc:
            with self._lock:
                sesion.clean()
                sesion.state = "problem"
                sesion.problem = AppProblem("transcription", str(exc))
            return
        except Exception as exc:  # frontera del adaptador: un fallo no tumba el http_server
            with self._lock:
                sesion.clean()
                sesion.state = "problem"
                sesion.problem = AppProblem("processing", f"{type(exc).__name__}: {exc}")
            return
        with self._lock:
            # Lo que propuso AIDA, sin los textos (van cifrados con la revisión guardada):
            # qué códigos y qué campos con contenido.
            self._auditoria.record(
                "draft_generated",
                actor=profesional,
                owner=owner,
                data={
                    "codes": ",".join(sorted(m.diagnostic.code for m in prepared.matches)),
                    "fields": ",".join(
                        sorted(spec.section_id for spec, texto in draft.note.fields if texto)
                    ),
                },
            )
            sesion.drafted = True
            sesion.mine = dict(mine or {})
            sesion.excluded = excluded
            sesion.draft = draft
            sesion.view = vista
            sesion.started = self._clock()
            sesion.state = "review"

    # Páginas: el armazón llama con el candado tomado, como antes de la mudanza.

    def page(self, ctx: ShellContext) -> str:
        self._sincronizar(ctx)
        s = self._session
        w = self._wording
        token = ctx.token
        marca = ctx.brand
        momento = ctx.now
        if s.state == "review" and s.draft is not None and s.view:
            borrador = s.draft
            return _render_page(
                borrador.prepared.matches,
                borrador.note,
                borrador.prepared.transcript,
                s.view,
                s.owner,
                momento,
                self._ui.wording,
                token,
                overrides=s.mine,
                excluded=s.excluded,
                consent_line=self._textos.line(s.consent) if s.consent else None,
                trace=borrador.trace,
            )
        if s.state == "live" and s.live is not None:
            return _live_page(
                w, self._ui, token, s.owner, momento, self._secciones, s.view
            )
        if s.state == "working":
            return _working_page(w, marca, token, s.owner, momento)
        if s.state == "settings" and self._tools is not None:
            return _settings_page(
                w, marca, token, self._tools, momento, s.download,
                profiles=ctx.profiles, professional=ctx.professional, license=ctx.license,
            )
        # La licencia (plan 100): sin una válida, la pantalla de cargarla; caducada, solo
        # lectura. Las dos, después de identificarse: cargar una licencia es un registro y
        # tiene autor.
        if ctx.license.status in ("missing", "invalid"):
            return license_page(w, marca, token, ctx.license, momento, ctx.professional)
        if not ctx.license.allows(self.id) and s.state == "start":
            return readonly_page(
                w, marca, token, ctx.license, momento, ctx.professional, self.name,
                self.readonly_items(ctx),
            )
        if s.state == "problem" and s.problem is not None:
            if s.problem.kind == "model" and self._tools is not None:
                return _model_page(w, marca, token, self._tools, momento, s.download)
            return _problem_page(w, marca, token, s.problem, momento)
        if isinstance(s.ready, AppReady):
            pendiente = self._pendiente(ctx)
            return _start_page(
                w, marca, token, s.ready, momento,
                settings=self._tools is not None, consent=self._textos,
                professional=ctx.professional, license=ctx.license,
                pending=pendiente,
                pending_expires=(
                    pendiente.expires(self._interrupted_hours)
                    if pendiente is not None and self._interrupted_hours is not None
                    else None
                ),
            )
        return _problem_page(w, marca, token, AppProblem("processing"), momento)

    def settings_cards(self, ctx: ShellContext) -> str:
        if self._tools is None:
            return ""
        return _settings_cards(
            self._wording, self._tools, self._session.download, ctx.profiles, ctx.license
        )

    def readonly_items(self, ctx: ShellContext) -> tuple[str, ...]:
        return tuple(
            self._wording[clave]
            for clave in ("readonly_item_consents", "readonly_item_audit", "readonly_item_texts")
        )

    # Rutas: el armazón ya ha comprobado la identidad; aquí solo lo de la consulta.

    def get(self, path: str, ctx: ShellContext) -> Response | None:
        self._sincronizar(ctx)
        s = self._session
        if path.startswith("estado"):
            with self._lock:
                pasos: tuple[bool, ...] = ()
                if isinstance(s.ready, AppReady):
                    eventos = list(s.ready.log)[s.start_index :]
                    pasos = progress(eventos, drafted=s.drafted)
                return json_response(200, {"state": s.state, "steps": list(pasos)})
        if path.startswith("vivo"):
            with self._lock:
                return json_response(200, _live_status(s, f"/{ctx.token}/{path}"))
        if path == "modelo" and self._tools is not None:
            with self._lock:
                return json_response(200, {
                    "state": s.download,
                    "bytes": self._tools.model_bytes(),
                    "total": self._tools.model_total_mb * 1024 * 1024,
                    "detail": s.download_detail,
                })
        if path == "historia":
            with self._lock:
                vista = s.view
                if (
                    s.state in ("review", "live")
                    and vista is not None
                    and vista.content_type is not None
                ):
                    # Ver la historia original es un acceso a datos de la persona atendida:
                    # queda en la auditoría, una vez por consulta.
                    if not s.history_seen:
                        s.history_seen = True
                        self._auditoria.record(
                            "history_viewed", actor=self._actor(), owner=s.owner
                        )
                    return Response(200, vista.data, vista.content_type)
        return None

    def post(
        self,
        path: str,
        body: bytes | None,
        headers: Mapping[str, str],
        ctx: ShellContext,
    ) -> Response | None:
        self._sincronizar(ctx)
        if path == "actividad":
            return json_response(200, {"ok": True})
        if path in ("audio", "documento"):
            return self._upload(path, body, headers.get("x-name", ""), ctx)
        if path == "empezar":
            return self._empezar(body, ctx)
        if path in ("recuperar", "descartar"):
            return self._recover(path, body, ctx)
        if path == "guardar":
            return self._guardar(body, ctx)
        if path in ("pcm", "campo", "codigo", "pausa"):
            return self._live(path, body)
        if path == "terminar":
            return self._terminar(ctx)
        if path in _AJUSTES_RUTAS and self._tools is not None:
            return self._ajustes(path, body, self._tools)
        if path == "volver":
            with self._lock:
                if self._session.state == "settings":
                    self.rebuild()
                else:
                    self._session.clean()
                    if isinstance(self._session.ready, AppReady):
                        self._session.state = "start"
                        self._session.problem = None
            return json_response(200, {"ok": True})
        if path == "reintentar":
            with self._lock:
                self.rebuild()
            return json_response(200, {"ok": True})
        return None

    def _upload(
        self, accion: str, datos: bytes | None, nombre: str, ctx: ShellContext
    ) -> Response:
        if datos is None:
            return json_response(413, {"error": "archivo demasiado grande"})
        sesion = self._session
        with self._lock:
            if sesion.state != "start":
                return json_response(409, {"error": "hay una sesión en curso"})
            if sesion.workdir is None:
                sesion.workdir = Path(
                    tempfile.mkdtemp(prefix=_WORKDIR_PREFIX, dir=self._raiz)
                )
            sufijo = _safe_suffix(nombre)
            # Sellado al escribirlo (plan 85a): lo subido nunca está en claro en disco.
            destino = sesion.workdir / f"{accion}{sufijo}.sealed"
            ready = sesion.ready
            seal_bytes(destino, datos, self._clave(ready if isinstance(ready, AppReady) else None))
            if accion == "audio":
                sesion.audio = destino
            else:
                sesion.document = destino
        return json_response(200, {"ok": True})

    def _empezar(self, body: bytes | None, ctx: ShellContext) -> Response:
        datos = body or b"{}"
        try:
            pedido = json.loads(datos)
        except ValueError:
            pedido = None
        if not isinstance(pedido, dict):
            return json_response(400, {"error": "cuerpo inválido"})
        owner = pedido.get("owner")
        sesion = self._session
        with self._lock:
            ready = sesion.ready
            if not isinstance(ready, AppReady) or sesion.state != "start":
                return json_response(409, {"error": "no se puede empezar ahora"})
            if not isinstance(owner, str) or not valid_owner(owner):
                return json_response(400, {"error": self._wording["owner_invalid"]})
            # Caducada o sin licencia, nada nuevo (plan 100). Una sola puerta: todo lo que
            # empieza trabajo en la aplicación pasa por aquí.
            if not ctx.license.allows(self.id):
                return json_response(403, {"error": self._wording["license_readonly"]})
            grabando = pedido.get("recorded") is True
            if sesion.document is None or (sesion.audio is None and not grabando):
                return json_response(400, {"error": self._wording["missing_document"]})
            # Al grabar y al subir (plan 82): sin autorización no se procesa nada.
            if pedido.get("consent") is not True:
                return json_response(400, {"error": self._wording["missing_consent"]})
            # La nube solo si la casilla está marcada y la hay (plan 85b).
            nube = pedido.get("cloud") is True and ready.cloud
            if not nube and not ready.local_model:
                return json_response(400, {"error": self._wording["cloud_needs_local"]})
            sesion.owner = owner
            registro = ready.consultation.record_consent(
                owner=AgentId(owner),
                mode="recorded" if grabando else "uploaded",
                actor=self._actor(),
                cloud=nube,
            )
            sesion.consent = registro
            if grabando and sesion.workdir is not None:
                documento = sesion.document
                sesion.live = LiveSession(
                    consultation=ready.consultation,
                    owner=owner,
                    note_sections=self._secciones,
                    history_text=lambda: _read_sealed(ready.reader, documento, self._clave(ready)),
                    workdir=sesion.workdir,
                    transcriber=ready.live_stt if nube else None,
                    vad=self._crear_vad(),
                    seal_key=self._clave(ready),
                    consent=registro,
                    clock=self._clock,
                )
                # Sin clave del equipo no hay marcador (87).
                if ready.seal_key is not None:
                    write_marker(
                        sesion.workdir,
                        owner=owner,
                        consent_digest=registro.digest,
                        professional=self._actor().id,
                        started=self._now().astimezone(),
                        document=documento.name,
                        key=ready.seal_key,
                    )
                sesion.live.start()
                with plain_copy(documento, self._clave(ready)) as historia:
                    sesion.view = history_view(historia, "")
                sesion.state = "live"
                return json_response(200, {"ok": True})
            sesion.start_index = len(ready.log)
            sesion.drafted = False
            sesion.state = "working"
            audio, document = sesion.audio, sesion.document
            assert audio is not None and document is not None
        threading.Thread(
            target=self._procesar,
            args=(ready, owner, audio, document, registro, self._actor()),
            daemon=True,
        ).start()
        return json_response(200, {"ok": True})

    def _recover(self, accion: str, body: bytes | None, ctx: ShellContext) -> Response:
        """Recuperar o descartar una grabación interrumpida. Solo la ve quien la grabó: el
        filtro por profesional va antes que cualquier otra cosa, y con él un id ajeno es
        simplemente «no existe» (87)."""
        try:
            pedido = json.loads(body or b"{}")
        except ValueError:
            pedido = None
        if not isinstance(pedido, dict) or not isinstance(pedido.get("id"), str):
            return json_response(400, {"error": "cuerpo inválido"})
        nombre = pedido["id"]
        sesion = self._session
        with self._lock:
            ready = sesion.ready
            if (
                not isinstance(ready, AppReady)
                or ready.seal_key is None
                or sesion.state != "start"
                or ctx.professional is None
            ):
                return json_response(409, {"error": "no se puede recuperar ahora"})
            if not ctx.license.allows(self.id):
                return json_response(403, {"error": self._wording["license_readonly"]})
            elegidas = [
                i
                for i in interrupted(self._raiz, ready.seal_key)
                if i.professional == ctx.professional.id and i.folder.name == nombre
            ]
            if not elegidas:
                return json_response(404, {"error": "no hay esa grabación"})
            elegida = elegidas[0]
            if accion == "descartar":
                shutil.rmtree(elegida.folder, ignore_errors=True)
                self._auditoria.record(
                    "recording_discarded",
                    actor=ctx.professional,
                    owner=elegida.owner,
                    data={"started": elegida.started.isoformat(timespec="seconds")},
                )
                return json_response(200, {"ok": True})
            registro = ready.consultation.consent_by_digest(elegida.consent_digest)
            if registro is None:
                return json_response(409, {"error": self._wording["recover_gone"]})
            sesion.workdir = elegida.folder
            sesion.owner = elegida.owner
            sesion.document = elegida.folder / elegida.document
            sesion.consent = registro
            sesion.start_index = len(ready.log)
            sesion.drafted = False
            sesion.state = "working"
            self._auditoria.record(
                "recording_recovered",
                actor=ctx.professional,
                owner=elegida.owner,
                data={"started": elegida.started.isoformat(timespec="seconds")},
            )
            audio = elegida.folder / RECORDING_NAME
            documento = elegida.folder / elegida.document
        threading.Thread(
            target=self._procesar,
            args=(ready, elegida.owner, audio, documento, registro, self._actor()),
            daemon=True,
        ).start()
        return json_response(200, {"ok": True})

    def _guardar(self, body: bytes | None, ctx: ShellContext) -> Response:
        datos = body or b"{}"
        try:
            pedido = json.loads(datos)
        except ValueError:
            pedido = None
        if not isinstance(pedido, dict):
            return json_response(400, {"error": "cuerpo inválido"})
        sesion = self._session
        with self._lock:
            ready = sesion.ready
            borrador = sesion.draft
            if (
                sesion.state != "review"
                or borrador is None
                or not isinstance(ready, AppReady)
            ):
                return json_response(409, {"error": "no hay nada que guardar"})
            envio, error = parse_submission(
                pedido,
                items=borrador.items,
                note=borrador.note,
                measured_elapsed=self._clock() - sesion.started,
            )
            if envio is None:
                return json_response(400, {"error": error})
            try:
                close_from_screen(ready.consultation, borrador, envio, actor=self._actor())
            except AbsenceUnresolvedError:
                # La pantalla no deja llegar aquí; si llega, el servidor manda.
                ajustes = ready.consultation.traceability
                texto = ajustes.absences_unresolved if ajustes is not None else ""
                return json_response(400, {"error": texto})
            sesion.clean()
            sesion.state = "start"
        return json_response(200, {"ok": True, "next": f"/{ctx.token}"})

    def _live(self, accion: str, datos: bytes | None) -> Response:
        with self._lock:
            vivo = self._session.live if self._session.state == "live" else None
        if vivo is None or datos is None:
            return json_response(409, {"error": "no hay una grabación en curso"})
        if accion == "pcm":
            vivo.feed(datos)
            return json_response(200, {"ok": True})
        try:
            pedido = json.loads(datos or b"{}")
        except ValueError:
            pedido = None
        if not isinstance(pedido, dict):
            return json_response(400, {"error": "cuerpo inválido"})
        if accion == "campo":
            ok = isinstance(pedido.get("id"), str) and isinstance(pedido.get("text"), str)
            if not ok or not vivo.edit_field(pedido["id"], pedido["text"]):
                return json_response(400, {"error": "campo desconocido"})
        elif accion == "codigo":
            if not isinstance(pedido.get("code"), str):
                return json_response(400, {"error": "código inválido"})
            vivo.set_code(pedido["code"], pedido.get("included") is not False)
        else:
            vivo.pause(pedido.get("paused") is True)
        return json_response(200, {"ok": True})

    def _terminar(self, ctx: ShellContext) -> Response:
        sesion = self._session
        with self._lock:
            ready = sesion.ready
            vivo = sesion.live
            document = sesion.document
            if (
                sesion.state != "live"
                or vivo is None
                or document is None
                or not isinstance(ready, AppReady)
            ):
                return json_response(409, {"error": "no hay una grabación en curso"})
            sesion.state = "working"
            sesion.start_index = len(ready.log)
            sesion.drafted = False
        audio = vivo.stop()
        mine, excluded = vivo.mine(), vivo.rejected()
        with self._lock:
            sesion.live = None
        threading.Thread(
            target=self._procesar,
            args=(
                ready, sesion.owner, audio, document, vivo.consent,
                self._actor(), mine, excluded,
            ),
            daemon=True,
        ).start()
        return json_response(200, {"ok": True})

    def _ajustes(self, accion: str, body: bytes | None, herramientas: AppTools) -> Response:
        """Ajustes solo se abre desde el inicio o desde un problema: nunca con una consulta a
        medias, que se perdería al reconstruir."""
        sesion = self._session
        if accion == "ajustes":
            with self._lock:
                if sesion.state not in ("start", "problem"):
                    return json_response(409, {"ok": False})
                sesion.clean()
                sesion.state = "settings"
            return json_response(200, {"ok": True})
        if accion == "modelo":
            with self._lock:
                if sesion.download == "running":
                    return json_response(200, {"ok": True})
                sesion.download = "running"
                sesion.download_detail = ""
            threading.Thread(target=self._descargar, args=(herramientas,), daemon=True).start()
            return json_response(200, {"ok": True})
        if accion == "solo-local":
            try:
                on = json.loads(body or b"{}").get("on")
            except (ValueError, AttributeError):
                on = None
            if not isinstance(on, bool):
                return json_response(400, {"ok": False})
            herramientas.save_local_only(on)
            return json_response(200, {"ok": True})
        try:
            clave = str(json.loads(body or b"{}").get("key", "")).strip()
        except (ValueError, AttributeError):
            clave = ""
        if not clave:
            return json_response(400, {"ok": False, "reason": ""})
        # Sin el candado: probar la clave sale a la red y puede tardar segundos.
        sirve, motivo = herramientas.check_cloud_key(clave)
        if accion == "clave-guardar" and sirve:
            herramientas.save_cloud_key(clave)
        return json_response(200, {"ok": sirve, "reason": motivo})


def load_app_wording(path: Path | None = None) -> AppWording:
    """Los textos de salud, con los de la plataforma por detrás. Una clave en los dos
    archivos es un error de carga: dos verdades para la misma pantalla (101a)."""
    ruta = path if path is not None else _UI_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    seccion = raw.get("app") if isinstance(raw, dict) else None
    if not isinstance(seccion, dict):
        raise ValueError(f"falta la sección 'app' en {ruta}")
    repetidas = [clave for clave in seccion if clave in SHELL_KEYS]
    if repetidas:
        raise ValueError(f"claves de plataforma en la sección 'app' de {ruta}: {repetidas}")
    faltan = [clave for clave in _CLAVES_APP if not isinstance(seccion.get(clave), str)]
    if faltan:
        raise ValueError(f"faltan claves en la sección 'app' de {ruta}: {faltan}")
    pasos = seccion.get("steps")
    if not isinstance(pasos, list) or len(pasos) != 5 or not all(
        isinstance(p, str) for p in pasos
    ):
        raise ValueError(f"'app.steps' debe ser una lista de cinco cadenas en {ruta}")
    shell = load_shell_wording()
    return AppWording(
        texts={**{clave: seccion[clave] for clave in _CLAVES_APP}, **dict(shell.texts)},
        steps=tuple(pasos),
        idle_lock_minutes=shell.idle_lock_minutes,
    )


def serve(
    build: Callable[[], AppReady | AppProblem],
    *,
    settings: UISettings,
    wording: AppWording,
    open_browser: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = datetime.now,
    vad_factory: Callable[[], Callable[[Any], Any]] | None = None,
    note_sections: Sequence[SectionSpec] | None = None,
    tools: AppTools | None = None,
    consent_texts: ConsentTexts | None = None,
    profiles: ProfileStore | None = None,
    audit: AuditLog | None = None,
    license: LicenseState,
    license_path: Path | None = None,
    arm_id: str = "health",
    arm_name: str = "",
    today: Callable[[], date] = date.today,
    retention_days: int | None = None,
    interrupted_hours: int | None = None,
    sessions: Path | None = None,
) -> None:
    """Levanta la aplicación de salud y bloquea hasta que alguien pulsa «Salir» (o Ctrl-C).
    La firma es la de siempre: quien la llama no sabe que hay un armazón debajo (101a)."""
    raiz = sessions if sessions is not None else sessions_root()
    raiz.mkdir(parents=True, exist_ok=True, mode=0o700)
    purge_plain_leftovers()
    for resto in sorted(Path(tempfile.gettempdir()).glob(f"{_WORKDIR_PREFIX}*")):
        shutil.rmtree(resto, ignore_errors=True)
    purge_unmarked(raiz)
    secciones: Sequence[SectionSpec] = (
        note_sections
        if note_sections is not None
        else tuple(s for s in load_sections().sections if s.in_note)
    )
    if vad_factory is None:
        from aida.voice.listen import vad_from_faster_whisper

        crear_vad: Callable[[], Callable[[Any], Any]] = vad_from_faster_whisper
    else:
        crear_vad = vad_factory
    textos = consent_texts if consent_texts is not None else load_consent_texts()
    perfiles = (
        profiles if profiles is not None else ProfileStore(user_dir("config") / "profiles.json")
    )
    auditoria = audit if audit is not None else AuditLog()
    lock = threading.Lock()
    brazo = HealthArm(
        build,
        arm_id=arm_id,
        arm_name=arm_name,
        ui=settings,
        wording=wording,
        tools=tools,
        consent_texts=textos,
        auditoria=auditoria,
        lock=lock,
        clock=clock,
        now=now,
        raiz=raiz,
        secciones=secciones,
        crear_vad=crear_vad,
        retention_days=retention_days,
        interrupted_hours=interrupted_hours,
    )
    run_shell(
        [brazo],
        brand=settings.wording.brand,
        wording=wording,
        profiles=perfiles,
        audit=auditoria,
        license=license,
        license_path=license_path,
        lock=lock,
        open_browser=open_browser,
        clock=clock,
        now=now,
        today=today,
    )
