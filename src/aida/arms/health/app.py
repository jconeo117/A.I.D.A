"""La aplicación local: AIDA abierta, una sesión de trabajo tras otra, sin terminal.

A diferencia de la pantalla de un solo uso (`review_ui.collect_review`), esto es un
servicio, y lo es a propósito (plan 77): quien trabaja con AIDA no lanza un comando
por cada sesión. Sigue escuchando solo en 127.0.0.1 y detrás de un token, y atiende
una sesión cada vez.

Ningún texto de la pantalla vive aquí: sale de `ui.yaml`, sección `app` (`I-073`).
"""

from __future__ import annotations

import html
import json
import re
import shutil
import tempfile
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

import yaml

from aida.arms.health.audio import to_wav
from aida.arms.health.audit import AuditLog
from aida.arms.health.capture import ModelUnavailableError, TranscriptionError
from aida.arms.health.consent import ConsentRecord, ConsentTexts, load_consent_texts
from aida.arms.health.consultation import (
    Consultation,
    ConsultationResult,
    PreparedConsultation,
)
from aida.arms.health.identity import IdentityError, Professional, ProfileStore
from aida.arms.health.live import LiveSession, LiveTranscriber, line_kind
from aida.arms.health.note import RenderedNote
from aida.arms.health.review import ReviewItem, review_items
from aida.arms.health.review_ui import (
    _CSS,
    _ICONS,
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
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import Event, EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.resources import data_root, user_dir

_RAIZ = data_root()
_UI_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "ui.yaml"

_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
"""Un código, no un nombre: sin espacios ni tildes. Aparece en la caché, en las
métricas y en el nombre de los archivos exportados, y lo que no es un nombre no es
un dato que haya que proteger."""

_MAX_BYTES = 512 * 1024 * 1024

_WORKDIR_PREFIX = "aida-sesion-"
"""Las sesiones viven en carpetas con este prefijo, y al arrancar se borran las que
hayan quedado: si el proceso murió sin cerrarse, el audio en claro de la última
sesión seguiría en disco."""

_CLAVES_APP = (
    "header_place",
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
    "missing_owner",
    "missing_audio",
    "missing_document",
    "missing_consent",
    "consent_version",
    "ready",
    "start_button",
    "record_button",
    "exit_label",
    "closed_text",
    "recording_pill",
    "stop_button",
    "working_lead",
    "working_count",
    "pill_ready",
    "pill_working",
    "pill_problem",
    "retry_label",
    "back_label",
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
    "settings_label",
    "title_settings",
    "settings_lead",
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
    "title_first_run",
    "pill_preparing",
    "problem_model_title",
    "who_title",
    "who_lead",
    "who_add",
    "who_add_hint",
    "registry_short",
    "pin_label",
    "enter_button",
    "first_title",
    "first_lead",
    "first_name",
    "first_registry",
    "first_pin",
    "first_pin2",
    "first_hint",
    "first_button",
    "first_pin_mismatch",
    "locked_title",
    "locked_lead",
    "unlock_button",
    "other_professional",
    "title_locked",
    "title_who",
    "pill_locked",
    "change_professional",
    "profiles_title",
    "profiles_text",
    "profiles_add",
    "profiles_added",
    "identity_missing",
    "identity_pin_format",
    "identity_duplicate",
    "identity_unknown",
    "identity_bad_pin",
    "identity_wait",
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

_ICONS_APP = _ICONS.replace("</defs></svg>", _ICONS_EXTRA + "\n</defs></svg>")

_CSS_SETTINGS = """/* La 78: ajustes y primer arranque */
.head-link {
  display: inline-flex; align-items: center; gap: .4rem; padding: .35rem .75rem;
  border-radius: 999px; font-size: .8rem; color: var(--head-ink);
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.1);
}
.head-link svg { width: 15px; height: 15px; }
.settings { display: flex; flex-direction: column; gap: 1rem; margin-top: 1.25rem; }
.setting {
  border: 1px solid var(--line); border-radius: var(--r); padding: 1rem 1.15rem;
  background: var(--surface);
}
.setting-head { display: flex; align-items: center; gap: .6rem; }
.setting-head h3 { margin: 0; font-size: .95rem; font-weight: 650; }
.setting-head svg { width: 18px; height: 18px; color: var(--brand); }
.setting p { margin: .35rem 0 0; color: var(--muted); font-size: .85rem; }
.status {
  margin-left: auto; display: inline-flex; align-items: center; gap: .35rem; font-size: .78rem;
  font-weight: 600; padding: .15rem .6rem; border-radius: 999px;
}
.status.ok { background: var(--ok-soft, #e6f4ec); color: #1a7f4b; }
.status.warn { background: var(--warn-soft); color: var(--warn); }
.status.fail { background: var(--off-soft); color: var(--off); }
.key-row { display: flex; gap: .6rem; margin-top: .8rem; }
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

_ICONS_SETTINGS = _ICONS_APP.replace(
    "</defs></svg>", _ICONS_SETTINGS_EXTRA + "\n</defs></svg>"
)

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
.consent-card.done { border-color: color-mix(in srgb, var(--brand) 45%, var(--line)); }"""

_ICONS_PAGES = _ICONS_SETTINGS.replace("</defs></svg>", _ICONS_CONSENT + "\n</defs></svg>")

_CSS_IDENTITY = """/* La 83: quién atiende */
.who {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: .75rem;
  margin-top: 1.25rem;
}
.person {
  display: flex; align-items: center; gap: .75rem; padding: .85rem 1rem; text-align: left;
  border: 1px solid var(--line); border-radius: var(--r); background: var(--surface);
  cursor: pointer;
  font: inherit; color: var(--ink);
}
.person:hover { border-color: var(--line-2); }
.person[aria-pressed="true"] {
  border-color: var(--brand);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--brand) 18%, transparent);
}
.person .avatar {
  width: 38px; height: 38px; border-radius: 50%; display: grid; place-items: center; flex: none;
  background: color-mix(in srgb, var(--brand) 14%, var(--surface)); color: var(--brand);
  font-weight: 700;
}
.person strong { display: block; font-size: .92rem; }
.person small { color: var(--muted); font-size: .78rem; }
.person.add { border-style: dashed; color: var(--muted); }
.person.add .avatar { background: var(--surface-2); color: var(--muted); }
.pin-row { display: flex; gap: .6rem; align-items: center; margin-top: 1.25rem; }
.pin-row .input { max-width: 220px; letter-spacing: .35em; font-size: 1.1rem; text-align: center; }
.pin-error { color: var(--off); font-size: .85rem; }
.pin-error[hidden] { display: none; }
.head-user {
  display: inline-flex; align-items: center; gap: .45rem; padding: .35rem .75rem;
  border-radius: 999px;
  font-size: .8rem; color: var(--head-ink); background: rgba(255,255,255,.07);
  border: 1px solid rgba(255,255,255,.1);
}
.head-user svg { width: 15px; height: 15px; }
.lock-icon {
  width: 56px; height: 56px; border-radius: 16px; display: grid; place-items: center;
  margin: 0 auto .75rem;
  background: var(--surface-2); color: var(--brand);
}
.lock-icon svg { width: 28px; height: 28px; }
.center { text-align: center; }
.center .pin-row { justify-content: center; }
.head-user[hidden], .head-link[hidden] { display: none; }
button.head-user { cursor: pointer; font: inherit; font-size: .8rem; }
.fields-2 { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem 1rem; margin-top: 1rem; }"""

_ICONS_IDENTITY = """  <symbol id="i-user" viewBox="0 0 24 24">
    <circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>
  </symbol>
  <symbol id="i-lock" viewBox="0 0 24 24">
    <rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>
  </symbol>
  <symbol id="i-plus" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></symbol>"""

_ICONS_ALL = _ICONS_PAGES.replace("</defs></svg>", _ICONS_IDENTITY + "\n</defs></svg>")

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

_ICONS_LIVE = _ICONS_APP.replace("</defs></svg>", _ICONS_LIVE_EXTRA + "\n</defs></svg>")


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


def load_app_wording(path: Path | None = None) -> AppWording:
    ruta = path if path is not None else _UI_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    seccion = raw.get("app") if isinstance(raw, dict) else None
    if not isinstance(seccion, dict):
        raise ValueError(f"falta la sección 'app' en {ruta}")
    faltan = [clave for clave in _CLAVES_APP if not isinstance(seccion.get(clave), str)]
    if faltan:
        raise ValueError(f"faltan claves en la sección 'app' de {ruta}: {faltan}")
    pasos = seccion.get("steps")
    if not isinstance(pasos, list) or len(pasos) != 5 or not all(
        isinstance(p, str) for p in pasos
    ):
        raise ValueError(f"'app.steps' debe ser una lista de cinco cadenas en {ruta}")
    minutos = seccion.get("idle_lock_minutes")
    if not isinstance(minutos, int | float) or isinstance(minutos, bool) or minutos <= 0:
        raise ValueError(f"'app.idle_lock_minutes' debe ser un número positivo en {ruta}")
    return AppWording(
        texts={clave: seccion[clave] for clave in _CLAVES_APP},
        steps=tuple(pasos),
        idle_lock_minutes=float(minutos),
    )


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


@dataclass(frozen=True, slots=True)
class ReviewDraft:
    prepared: PreparedConsultation
    note: RenderedNote
    items: tuple[ReviewItem, ...]


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
    return ReviewDraft(prepared=prepared, note=note, items=items)


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


@dataclass
class _Session:
    ready: AppReady | AppProblem
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
    professional: Professional | None = None
    """Quién atiende (plan 83). Sobrevive a `clean`: termina una consulta, no la sesión
    del profesional."""
    last_activity: float = 0.0
    history_seen: bool = False
    download: str = "idle"
    """`idle`, `running`, `done` o `failed`: la descarga del modelo sobrevive a cambiar
    de pantalla, y `clean` no la toca."""
    download_detail: str = ""

    def clean(self) -> None:
        """Lo subido o grabado no se queda en disco: el audio en claro es lo más
        sensible de una sesión, y la transcripción ya vive en la caché cifrada."""
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


def _safe_suffix(name: str) -> str:
    sufijo = Path(name).suffix.lower()
    return sufijo if re.fullmatch(r"\.[a-z0-9]{1,8}", sufijo) else ""


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
) -> None:
    """Levanta la aplicación y bloquea hasta que alguien pulsa «Salir» (o Ctrl-C).
    Una sesión cada vez: mientras una se procesa, no empieza otra."""
    for resto in sorted(Path(tempfile.gettempdir()).glob(f"{_WORKDIR_PREFIX}*")):
        shutil.rmtree(resto, ignore_errors=True)
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
    token = token_urlsafe(16)
    lock = threading.Lock()
    perfiles = (
        profiles if profiles is not None else ProfileStore(user_dir("config") / "profiles.json")
    )
    auditoria = audit if audit is not None else AuditLog()
    inactividad = wording.idle_lock_minutes * 60
    session = _Session(ready=build())
    # Nadie entra sin decir quién es (plan 83): lo primero es elegir perfil, o crear el
    # primero si el equipo es nuevo.
    session.state = "who" if perfiles.professionals() else "first_profile"

    def tras_identificarse() -> None:
        """Con el candado tomado: adonde se va después de entrar o desbloquear."""
        session.last_activity = clock()
        if isinstance(session.ready, AppProblem):
            session.state = "problem"
            session.problem = session.ready
        else:
            session.state = "start"
            session.problem = None

    def vigilar() -> None:
        """Con el candado tomado, en cada petición. Bloquea si pasó el plazo sin uso, y
        solo donde no hay nada a medias: en una grabación, un proceso o una revisión,
        nunca. Lo que estuviera subido se descarta, como al recargar."""
        if session.professional is None:
            return
        ahora = clock()
        reposo = session.state in ("start", "settings", "problem")
        if reposo and ahora - session.last_activity > inactividad:
            session.clean()
            session.state = "locked"
            auditoria.record("locked", actor=session.professional)
            return
        if session.state != "locked":
            session.last_activity = ahora

    def identificado() -> bool:
        return session.professional is not None and session.state not in (
            "who", "first_profile", "locked",
        )

    def reconstruir() -> None:
        """Con el candado tomado. Vuelve a montar todo: una clave guardada o un modelo
        descargado se notan sin cerrar AIDA (`I-152`)."""
        session.clean()
        session.ready = build()
        if isinstance(session.ready, AppProblem):
            session.state = "problem"
            session.problem = session.ready
        else:
            session.state = "start"
            session.problem = None

    def descargar(herramientas: AppTools) -> None:
        try:
            herramientas.download_model()
        except Exception as exc:  # frontera: una descarga fallida no tumba el http_server
            with lock:
                session.download = "failed"
                session.download_detail = f"{type(exc).__name__}: {exc}"
            return
        with lock:
            session.download = "done"
            session.download_detail = ""

    def procesar(
        ready: AppReady,
        owner: str,
        audio: Path,
        document: Path,
        consent: ConsentRecord | None,
        mine: Mapping[str, str] | None = None,
        excluded: frozenset[str] = frozenset(),
    ) -> None:
        try:
            wav = audio.with_name("audio-16k.wav")
            to_wav(audio, wav)
            prepared = ready.consultation.prepare(
                owner=AgentId(owner),
                audio_path=wav,
                history_text=lambda: ready.reader.read(document),
                consent=consent,
            )
            draft = _keep_mine(
                draft_for_screen(ready.consultation, prepared), mine or {}, secciones
            )
            vista = history_view(document, prepared.history)
        except (TranscriptionError, ModelUnavailableError) as exc:
            with lock:
                session.clean()
                session.state = "problem"
                session.problem = AppProblem("transcription", str(exc))
            return
        except Exception as exc:  # frontera del adaptador: un fallo no tumba el http_server
            with lock:
                session.clean()
                session.state = "problem"
                session.problem = AppProblem("processing", f"{type(exc).__name__}: {exc}")
            return
        with lock:
            # Lo que propuso AIDA, sin los textos (van cifrados con la revisión
            # guardada): qué códigos y qué campos con contenido.
            auditoria.record(
                "draft_generated",
                actor=session.professional,
                owner=owner,
                data={
                    "codes": ",".join(sorted(m.diagnostic.code for m in prepared.matches)),
                    "fields": ",".join(
                        sorted(spec.section_id for spec, texto in draft.note.fields if texto)
                    ),
                },
            )
            session.drafted = True
            session.mine = dict(mine or {})
            session.excluded = excluded
            session.draft = draft
            session.view = vista
            session.started = clock()
            session.state = "review"

    def pagina() -> str:
        momento = now()
        marca = settings.wording.brand
        if session.state == "first_profile":
            return _first_profile_page(wording, marca, token, momento)
        if session.state == "who":
            return _who_page(wording, marca, token, perfiles.professionals(), momento)
        if session.state == "locked" and session.professional is not None:
            return _locked_page(wording, marca, token, session.professional, momento)
        if session.state == "review" and session.draft is not None and session.view:
            borrador = session.draft
            return _render_page(
                borrador.prepared.matches,
                borrador.note,
                borrador.prepared.transcript,
                session.view,
                session.owner,
                momento,
                settings.wording,
                token,
                overrides=session.mine,
                excluded=session.excluded,
                consent_line=textos.line(session.consent) if session.consent else None,
            )
        if session.state == "live" and session.live is not None:
            return _live_page(
                wording, settings, token, session.owner, momento, secciones, session.view
            )
        if session.state == "working":
            return _working_page(wording, marca, token, session.owner, momento)
        if session.state == "settings" and tools is not None:
            return _settings_page(
                wording, marca, token, tools, momento, session.download,
                profiles=perfiles.professionals(), professional=session.professional,
            )
        if session.state == "problem" and session.problem is not None:
            if session.problem.kind == "model" and tools is not None:
                return _model_page(wording, marca, token, tools, momento, session.download)
            return _problem_page(wording, marca, token, session.problem, momento)
        if isinstance(session.ready, AppReady):
            return _start_page(
                wording, marca, token, session.ready, momento,
                settings=tools is not None, consent=textos,
                professional=session.professional,
            )
        return _problem_page(wording, marca, token, AppProblem("processing"), momento)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, body: dict[str, Any]) -> None:
            datos = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self._send(status, datos, "application/json; charset=utf-8")

        def _404(self) -> None:
            self._send(404, b"", "text/plain")

        def _body(self) -> bytes | None:
            longitud = int(self.headers.get("Content-Length", "0"))
            if longitud > _MAX_BYTES:
                return None
            return self.rfile.read(longitud)

        def do_GET(self) -> None:
            with lock:
                vigilar()
                if self.path == f"/{token}":
                    self._send(200, pagina().encode("utf-8"), "text/html; charset=utf-8")
                    return
                if not identificado():
                    self._json(401, {"error": ""})
                    return
                if self.path == f"/{token}/estado":
                    pasos: tuple[bool, ...] = ()
                    if isinstance(session.ready, AppReady):
                        eventos = list(session.ready.log)[session.start_index :]
                        pasos = progress(eventos, drafted=session.drafted)
                    self._json(200, {"state": session.state, "steps": list(pasos)})
                    return
                if self.path.startswith(f"/{token}/vivo"):
                    self._json(200, _live_status(session, self.path))
                    return
                if self.path == f"/{token}/modelo" and tools is not None:
                    self._json(200, {
                        "state": session.download,
                        "bytes": tools.model_bytes(),
                        "total": tools.model_total_mb * 1024 * 1024,
                        "detail": session.download_detail,
                    })
                    return
                vista = session.view
                if (
                    self.path == f"/{token}/historia"
                    and session.state in ("review", "live")
                    and vista is not None
                    and vista.content_type is not None
                ):
                    # Ver la historia original es un acceso a datos de la persona
                    # atendida: queda en la auditoría, una vez por consulta.
                    if not session.history_seen:
                        session.history_seen = True
                        auditoria.record(
                            "history_viewed", actor=session.professional, owner=session.owner
                        )
                    self._send(200, vista.data, vista.content_type)
                    return
            self._404()

        def do_POST(self) -> None:
            ruta = self.path
            if not ruta.startswith(f"/{token}/"):
                self._404()
                return
            accion = ruta[len(token) + 2 :]
            with lock:
                vigilar()
                permitido = identificado()
            if accion in ("perfil", "entrar", "desbloquear", "cambiar", "salir"):
                self._identity(accion)
                return
            if not permitido:
                self._json(401, {"error": ""})
                return
            if accion == "actividad":
                self._json(200, {"ok": True})
            elif accion in ("audio", "documento"):
                self._upload(accion)
            elif accion == "empezar":
                self._start()
            elif accion == "guardar":
                self._save()
            elif accion in ("pcm", "campo", "codigo", "pausa"):
                self._live(accion)
            elif accion == "terminar":
                self._finish()
            elif accion in ("ajustes", "clave-probar", "clave-guardar", "modelo") and tools:
                self._settings(accion, tools)
            elif accion == "volver":
                with lock:
                    if session.state == "settings":
                        reconstruir()
                    else:
                        session.clean()
                        if isinstance(session.ready, AppReady):
                            session.state = "start"
                            session.problem = None
                self._json(200, {"ok": True})
            elif accion == "reintentar":
                with lock:
                    reconstruir()
                self._json(200, {"ok": True})
            else:
                self._404()

        def _identity(self, accion: str) -> None:
            """Entrar, crear un perfil, desbloquear, cambiar de profesional y salir: lo
            único que se atiende sin estar identificado."""
            if accion == "salir":
                with lock:
                    if session.professional is not None:
                        auditoria.record("session_closed", actor=session.professional)
                    session.clean()
                    session.state = "closed"
                self._json(200, {"ok": True})
                threading.Thread(target=http_server.shutdown, daemon=True).start()
                return
            if accion == "cambiar":
                with lock:
                    if session.state in ("live", "working", "review"):
                        self._json(409, {"error": ""})
                        return
                    if session.professional is not None:
                        auditoria.record("session_closed", actor=session.professional)
                    session.clean()
                    session.professional = None
                    session.state = "who"
                self._json(200, {"ok": True})
                return
            datos = self._body() or b"{}"
            try:
                pedido = json.loads(datos)
            except ValueError:
                pedido = None
            if not isinstance(pedido, dict):
                self._json(400, {"error": ""})
                return
            with lock:
                try:
                    if accion == "perfil":
                        self._new_profile(pedido)
                    elif accion == "entrar" and session.state == "who":
                        perfil = str(pedido.get("profile", ""))
                        session.professional = perfiles.verify(perfil, str(pedido.get("pin", "")))
                        auditoria.record("session_opened", actor=session.professional)
                        tras_identificarse()
                        self._json(200, {"ok": True})
                    elif (
                        accion == "desbloquear"
                        and session.state == "locked"
                        and session.professional is not None
                    ):
                        perfiles.verify(session.professional.id, str(pedido.get("pin", "")))
                        auditoria.record("unlocked", actor=session.professional)
                        tras_identificarse()
                        self._json(200, {"ok": True})
                    else:
                        self._json(409, {"error": ""})
                except IdentityError as exc:
                    restantes = (
                        perfiles.attempts_left(
                            str(pedido.get("profile", ""))
                            if accion == "entrar"
                            else session.professional.id if session.professional else ""
                        )
                        if exc.code == "bad_pin"
                        else 0
                    )
                    self._json(
                        400, {"error": wording[f"identity_{exc.code}"].format(restantes)}
                    )

        def _new_profile(self, pedido: dict[str, Any]) -> None:
            """Con el candado tomado. El primero, en el primer arranque, y entra con él;
            los demás, desde Ajustes y por alguien ya identificado (plan 83): nadie se da
            de alta solo en un equipo que ya tiene dueños."""
            primero = session.state == "first_profile"
            if not primero and not (identificado() and session.state == "settings"):
                self._json(409, {"error": ""})
                return
            pin = str(pedido.get("pin", ""))
            if pin != str(pedido.get("pin2", "")):
                self._json(400, {"error": wording["first_pin_mismatch"]})
                return
            nuevo = perfiles.create(
                name=str(pedido.get("name", "")),
                registry=str(pedido.get("registry", "")),
                pin=pin,
            )
            auditoria.record(
                "profile_created",
                actor=session.professional if not primero else nuevo,
                data={"profile": nuevo.id, "name": nuevo.label()},
            )
            if primero:
                session.professional = nuevo
                auditoria.record("session_opened", actor=nuevo)
                tras_identificarse()
            self._json(200, {"ok": True})

        def _settings(self, accion: str, herramientas: AppTools) -> None:
            """Ajustes solo se abre desde el inicio o desde un problema: nunca con una
            consulta a medias, que se perdería al reconstruir."""
            if accion == "ajustes":
                with lock:
                    if session.state not in ("start", "problem"):
                        self._json(409, {"ok": False})
                        return
                    session.clean()
                    session.state = "settings"
                self._json(200, {"ok": True})
                return
            if accion == "modelo":
                with lock:
                    if session.download == "running":
                        self._json(200, {"ok": True})
                        return
                    session.download = "running"
                    session.download_detail = ""
                threading.Thread(target=descargar, args=(herramientas,), daemon=True).start()
                self._json(200, {"ok": True})
                return
            datos = self._body()
            try:
                clave = str(json.loads(datos or b"{}").get("key", "")).strip()
            except (ValueError, AttributeError):
                clave = ""
            if not clave:
                self._json(400, {"ok": False, "reason": ""})
                return
            # Sin el candado: probar la clave sale a la red y puede tardar segundos.
            sirve, motivo = herramientas.check_cloud_key(clave)
            if accion == "clave-guardar" and sirve:
                herramientas.save_cloud_key(clave)
            self._json(200, {"ok": sirve, "reason": motivo})

        def _live(self, accion: str) -> None:
            datos = self._body()
            with lock:
                vivo = session.live if session.state == "live" else None
            if vivo is None or datos is None:
                self._json(409, {"error": "no hay una grabación en curso"})
                return
            if accion == "pcm":
                vivo.feed(datos)
                self._json(200, {"ok": True})
                return
            try:
                pedido = json.loads(datos or b"{}")
            except ValueError:
                pedido = None
            if not isinstance(pedido, dict):
                self._json(400, {"error": "cuerpo inválido"})
                return
            if accion == "campo":
                ok = isinstance(pedido.get("id"), str) and isinstance(pedido.get("text"), str)
                if not ok or not vivo.edit_field(pedido["id"], pedido["text"]):
                    self._json(400, {"error": "campo desconocido"})
                    return
            elif accion == "codigo":
                if not isinstance(pedido.get("code"), str):
                    self._json(400, {"error": "código inválido"})
                    return
                vivo.set_code(pedido["code"], pedido.get("included") is not False)
            else:
                vivo.pause(pedido.get("paused") is True)
            self._json(200, {"ok": True})

        def _finish(self) -> None:
            with lock:
                ready = session.ready
                vivo = session.live
                document = session.document
                if (
                    session.state != "live"
                    or vivo is None
                    or document is None
                    or not isinstance(ready, AppReady)
                ):
                    self._json(409, {"error": "no hay una grabación en curso"})
                    return
                session.state = "working"
                session.start_index = len(ready.log)
                session.drafted = False
            audio = vivo.stop()
            mine, excluded = vivo.mine(), vivo.rejected()
            with lock:
                session.live = None
            threading.Thread(
                target=procesar,
                args=(ready, session.owner, audio, document, vivo.consent, mine, excluded),
                daemon=True,
            ).start()
            self._json(200, {"ok": True})

        def _upload(self, accion: str) -> None:
            datos = self._body()
            if datos is None:
                self._json(413, {"error": "archivo demasiado grande"})
                return
            with lock:
                if session.state != "start":
                    self._json(409, {"error": "hay una sesión en curso"})
                    return
                if session.workdir is None:
                    session.workdir = Path(tempfile.mkdtemp(prefix=_WORKDIR_PREFIX))
                sufijo = _safe_suffix(self.headers.get("X-Name", ""))
                destino = session.workdir / f"{accion}{sufijo}"
                destino.write_bytes(datos)
                if accion == "audio":
                    session.audio = destino
                else:
                    session.document = destino
            self._json(200, {"ok": True})

        def _start(self) -> None:
            datos = self._body() or b"{}"
            try:
                pedido = json.loads(datos)
            except ValueError:
                pedido = None
            if not isinstance(pedido, dict):
                self._json(400, {"error": "cuerpo inválido"})
                return
            owner = pedido.get("owner")
            with lock:
                ready = session.ready
                if not isinstance(ready, AppReady) or session.state != "start":
                    self._json(409, {"error": "no se puede empezar ahora"})
                    return
                if not isinstance(owner, str) or not valid_owner(owner):
                    self._json(400, {"error": wording["owner_invalid"]})
                    return
                grabando = pedido.get("recorded") is True
                if session.document is None or (session.audio is None and not grabando):
                    self._json(400, {"error": wording["missing_document"]})
                    return
                # Al grabar y al subir (plan 82): sin autorización no se procesa nada. En
                # el mismo orden que la pantalla, que la pide después de los archivos.
                if pedido.get("consent") is not True:
                    self._json(400, {"error": wording["missing_consent"]})
                    return
                session.owner = owner
                registro = ready.consultation.record_consent(
                    owner=AgentId(owner),
                    mode="recorded" if grabando else "uploaded",
                    actor=session.professional,
                )
                session.consent = registro
                if grabando and session.workdir is not None:
                    documento = session.document
                    session.live = LiveSession(
                        consultation=ready.consultation,
                        owner=owner,
                        note_sections=secciones,
                        history_text=lambda: ready.reader.read(documento),
                        workdir=session.workdir,
                        transcriber=ready.live_stt,
                        vad=crear_vad(),
                        consent=registro,
                        clock=clock,
                    )
                    session.live.start()
                    session.view = history_view(documento, "")
                    session.state = "live"
                    self._json(200, {"ok": True})
                    return
                session.start_index = len(ready.log)
                session.drafted = False
                session.state = "working"
                audio, document = session.audio, session.document
                assert audio is not None and document is not None
            threading.Thread(
                target=procesar, args=(ready, owner, audio, document, registro), daemon=True
            ).start()
            self._json(200, {"ok": True})

        def _save(self) -> None:
            datos = self._body() or b"{}"
            try:
                pedido = json.loads(datos)
            except ValueError:
                pedido = None
            if not isinstance(pedido, dict):
                self._json(400, {"error": "cuerpo inválido"})
                return
            with lock:
                ready = session.ready
                borrador = session.draft
                if (
                    session.state != "review"
                    or borrador is None
                    or not isinstance(ready, AppReady)
                ):
                    self._json(409, {"error": "no hay nada que guardar"})
                    return
                envio, error = parse_submission(
                    pedido,
                    items=borrador.items,
                    note=borrador.note,
                    measured_elapsed=clock() - session.started,
                )
                if envio is None:
                    self._json(400, {"error": error})
                    return
                close_from_screen(
                    ready.consultation, borrador, envio, actor=session.professional
                )
                session.clean()
                session.state = "start"
            self._json(200, {"ok": True, "next": f"/{token}"})

    http_server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{http_server.server_port}/{token}"
    if open_browser is not None:
        open_browser(url)
    else:
        webbrowser.open(url)
    try:
        http_server.serve_forever()
    finally:
        http_server.server_close()
        with lock:
            session.clean()


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
    e = html.escape
    if live:
        estilo = f"{_CSS}\n{_CSS_EXTRA}\n{_CSS_LIVE}"
        iconos = _ICONS_LIVE
    else:
        estilo = f"{_CSS}\n{_CSS_EXTRA}\n{_CSS_SETTINGS}\n{_CSS_CONSENT}\n{_CSS_IDENTITY}"
        iconos = _ICONS_ALL
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{e(title)}</title>
<style>
{estilo}
</style>
</head>
<body>
{iconos}
<header class="top">
  <div class="mark"><i>{e(brand[:1])}</i>{e(brand)}</div>
  <div class="sep"></div>
  <div class="title"><strong>{e(title)}</strong>
    <span>{e(place)} · {moment.strftime("%d/%m/%Y")}</span></div>
  <div class="head-right">{head_link}
    <span class="{pill_class}" id="head-pill"><span class="dot"></span>
      <span id="pill-text">{e(pill)}</span></span>
  </div>
</header>
{body}
<script>
{script}
</script>
</body>
</html>
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
) -> str:
    e = html.escape
    textos = consent if consent is not None else load_consent_texts()
    chips = (
        (w["check_cloud"] if ready.cloud else w["check_local_only"], ready.cloud),
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
    body = f"""<section class="screen" id="s-start">
  <div class="stage">
    <div class="panel-card">
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
            modo: {"title": t.title, "text": t.text, "check": t.check}
            for modo, t in sorted(textos.modes.items())
        },
        ensure_ascii=False,
    )
    script = (
        f"const TOKEN = {json.dumps(token)};\nconst W = {palabras};\n"
        f"const CONSENT = {autorizaciones};\n{_JS_START}"
    )
    enlace = _user_chip(w, professional)
    script += _identity_script(w)
    if settings:
        enlace += (
            f'\n    <button type="button" class="head-link" id="to-settings">'
            f'<svg><use href="#i-gear"/></svg>{e(w["settings_label"])}</button>'
        )
        script += _JS_TO_SETTINGS
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

function setMode(next) {
  mode = next;
  document.getElementById('mode-upload').setAttribute('aria-pressed', String(mode === 'upload'));
  document.getElementById('mode-record').setAttribute('aria-pressed', String(mode === 'record'));
  document.getElementById('drop-audio').hidden = mode !== 'upload';
  document.getElementById('mic-box').hidden = mode !== 'record';
  // Otro modo es otro texto: lo marcado para uno no vale para el otro.
  const texto = CONSENT[mode === 'record' ? 'recorded' : 'uploaded'];
  document.getElementById('consent-title').textContent = texto.title;
  document.getElementById('consent-text').textContent = texto.text;
  document.getElementById('consent-check').textContent = texto.check;
  document.getElementById('consent').checked = false;
  document.getElementById('consent-box').classList.remove('done');
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
) -> str:
    """Cada tarjeta se consulta al abrir: es una pantalla que se visita poco, y lo que
    enseña tiene que ser lo que hay ahora, no lo que había al arrancar."""
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
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card">
      <h2>{e(w["title_settings"])}</h2>
      <p class="lead">{e(w["settings_lead"])}</p>
      <div class="settings">
        <div class="setting">
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
        {_profiles_block(w, profiles)}
        <div class="setting">
          <div class="setting-head"><svg><use href="#i-key"/></svg><h3>{e(w["cipher_title"])}</h3>
            <span class="status ok">{e(w["cipher_ok"])}</span></div>
          <p>{e(cifrado)}</p>
        </div>
      </div>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="back">
      <svg><use href="#i-back"/></svg>{e(w["back_label"])}</button>
  </footer>
</section>"""
    return _page(
        brand=brand, title=w["title_settings"], place=w["header_place"], moment=moment,
        pill=w["pill_ready"], pill_class="pill", body=body,
        script=_settings_script(w, token, tools) + _identity_script(w),
        head_link=_user_chip(w, professional),
    )


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
    return f"const TOKEN = {json.dumps(token)};\nconst W = {textos};\n{_JS_SETTINGS}"


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
        script=_settings_script(w, token, tools) + _JS_TO_SETTINGS,
    )


def _user_chip(w: AppWording, professional: Professional | None) -> str:
    """Quién atiende, en la cabecera; pulsarlo cambia de profesional (plan 83)."""
    if professional is None:
        return ""
    e = html.escape
    return (
        f'\n    <button type="button" class="head-user" id="head-user" '
        f'title="{e(w["change_professional"])}"><svg><use href="#i-user"/></svg>'
        f"{e(professional.name)}</button>"
    )


def _identity_script(w: AppWording) -> str:
    return f"const IDLE_MS = {int(w.idle_lock_minutes * 60_000)};\n{_JS_IDENTITY_HEAD}"


def _initials(name: str) -> str:
    """Las iniciales del avatar, sin títulos: «Dra. Laura Gómez» es LG, no DL."""
    palabras = [p for p in name.split() if not p.endswith(".")]
    return "".join(p[0] for p in palabras[:2]).upper() or name[:1].upper()


def _person(p: Professional, w: AppWording, *, button: bool) -> str:
    e = html.escape
    etiqueta = "button" if button else "div"
    extra = (
        f' type="button" aria-pressed="false" data-id="{e(p.id)}" data-name="{e(p.name)}"'
        if button
        else ""
    )
    return (
        f'<{etiqueta} class="person"{extra}><span class="avatar">{e(_initials(p.name))}</span>'
        f"<span><strong>{e(p.name)}</strong>"
        f"<small>{e(w['registry_short'])} {e(p.registry)}</small></span></{etiqueta}>"
    )


def _profiles_block(w: AppWording, profiles: Sequence[Professional]) -> str:
    """En Ajustes: quiénes hay y añadir a otro. Lo añade alguien ya identificado."""
    e = html.escape
    personas = "".join(_person(p, w, button=False) for p in profiles)
    return f"""<div class="setting">
          <div class="setting-head"><svg><use href="#i-user"/></svg>
            <h3>{e(w["profiles_title"])}</h3></div>
          <p>{e(w["profiles_text"])}</p>
          <div class="who">{personas}</div>
          <div class="fields-2">
            <div><label class="label" for="np-name">{e(w["first_name"])}</label>
              <input class="input" id="np-name" autocomplete="off"></div>
            <div><label class="label" for="np-registry">{e(w["first_registry"])}</label>
              <input class="input" id="np-registry" autocomplete="off"></div>
            <div><label class="label" for="np-pin">{e(w["first_pin"])}</label>
              <input class="input" id="np-pin" type="password" inputmode="numeric"></div>
            <div><label class="label" for="np-pin2">{e(w["first_pin2"])}</label>
              <input class="input" id="np-pin2" type="password" inputmode="numeric"></div>
          </div>
          <div class="key-row"><button type="button" class="btn" id="np-add">
            <svg><use href="#i-plus"/></svg>{e(w["profiles_add"])}</button></div>
          <div class="result fail" id="np-error" hidden><svg><use href="#i-alert"/></svg>
            <span></span></div>
        </div>"""


def _who_page(
    w: AppWording,
    brand: str,
    token: str,
    profiles: Sequence[Professional],
    moment: datetime,
) -> str:
    e = html.escape
    personas = "".join(_person(p, w, button=True) for p in profiles)
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card">
      <h2>{e(w["who_title"])}</h2>
      <p class="lead">{e(w["who_lead"])}</p>
      <div class="who">{personas}
        <div class="person add"><span class="avatar">
          <svg style="width:18px;height:18px"><use href="#i-plus"/></svg></span>
          <span><strong>{e(w["who_add"])}</strong><small>{e(w["who_add_hint"])}</small></span>
        </div>
      </div>
      <div class="pin-row">
        <input class="input" id="pin" type="password" inputmode="numeric"
          aria-label="{e(w["pin_label"])}" autocomplete="off">
        <span class="pin-error" id="pin-error" hidden></span>
      </div>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="exit">
      <svg><use href="#i-exit"/></svg>{e(w["exit_label"])}</button>
    <span class="stat" id="who-name"></span>
    <button type="button" class="btn btn-primary" id="enter">
      <svg><use href="#i-check"/></svg>{e(w["enter_button"])}</button>
  </footer>
</section>"""
    script = (
        f"const TOKEN = {json.dumps(token)};\n"
        f"const W = {json.dumps({'closed_text': w['closed_text']}, ensure_ascii=False)};\n"
        f"{_JS_WHO}"
    )
    return _page(
        brand=brand, title=w["title_who"], place=w["header_place"], moment=moment,
        pill=w["pill_ready"], pill_class="pill", body=body, script=script,
    )


def _first_profile_page(w: AppWording, brand: str, token: str, moment: datetime) -> str:
    e = html.escape
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card">
      <h2>{e(w["first_title"])}</h2>
      <p class="lead">{e(w["first_lead"])}</p>
      <div class="fields-2">
        <div><label class="label" for="np-name">{e(w["first_name"])}</label>
          <input class="input" id="np-name" autocomplete="off"></div>
        <div><label class="label" for="np-registry">{e(w["first_registry"])}</label>
          <input class="input" id="np-registry" autocomplete="off"></div>
        <div><label class="label" for="np-pin">{e(w["first_pin"])}</label>
          <input class="input" id="np-pin" type="password" inputmode="numeric"></div>
        <div><label class="label" for="np-pin2">{e(w["first_pin2"])}</label>
          <input class="input" id="np-pin2" type="password" inputmode="numeric"></div>
      </div>
      <p class="hint">{e(w["first_hint"])}</p>
      <p class="pin-error" id="np-error" hidden></p>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="exit">
      <svg><use href="#i-exit"/></svg>{e(w["exit_label"])}</button>
    <span class="stat"></span>
    <button type="button" class="btn btn-primary" id="np-add">
      <svg><use href="#i-check"/></svg>{e(w["first_button"])}</button>
  </footer>
</section>"""
    script = (
        f"const TOKEN = {json.dumps(token)};\n"
        f"const W = {json.dumps({'closed_text': w['closed_text']}, ensure_ascii=False)};\n"
        f"{_JS_NEW_PROFILE}"
    )
    return _page(
        brand=brand, title=w["title_first_run"], place=w["header_place"], moment=moment,
        pill=w["pill_ready"], pill_class="pill", body=body, script=script,
    )


def _locked_page(
    w: AppWording, brand: str, token: str, professional: Professional, moment: datetime
) -> str:
    e = html.escape
    entrada = e(w["locked_lead"]).replace(
        "{}", f"<strong>{e(professional.name)}</strong>"
    ).replace("{minutes}", str(int(w.idle_lock_minutes)))
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card center">
      <div class="lock-icon"><svg><use href="#i-lock"/></svg></div>
      <h2>{e(w["locked_title"])}</h2>
      <p class="lead">{entrada}</p>
      <div class="pin-row">
        <input class="input" id="pin" type="password" inputmode="numeric"
          aria-label="{e(w["pin_label"])}" autocomplete="off">
      </div>
      <p class="pin-error" id="pin-error" hidden></p>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="other">
      <svg><use href="#i-user"/></svg>{e(w["other_professional"])}</button>
    <span class="stat"></span>
    <button type="button" class="btn btn-primary" id="unlock">
      <svg><use href="#i-check"/></svg>{e(w["unlock_button"])}</button>
  </footer>
</section>"""
    chip = (
        f'\n    <span class="head-user"><svg><use href="#i-user"/></svg>'
        f"{e(professional.name)}</span>"
    )
    return _page(
        brand=brand, title=w["title_locked"], place=w["header_place"], moment=moment,
        pill=w["pill_locked"], pill_class="pill fallback", body=body,
        script=f"const TOKEN = {json.dumps(token)};\n{_JS_LOCKED}", head_link=chip,
    )


_JS_IDENTITY_HEAD = """
(function () {
  let last = Date.now(), sent = 0;
  const activity = () => {
    last = Date.now();
    if (last - sent > 60000) {
      sent = last;
      fetch('/' + TOKEN + '/actividad', { method: 'POST', body: '' });
    }
  };
  ['input', 'click', 'keydown'].forEach(ev => document.addEventListener(ev, activity, true));
  // El bloqueo lo decide Python; la página solo recarga para enseñarlo.
  setInterval(() => { if (Date.now() - last > IDLE_MS + 2000) location.reload(); }, 5000);
  const me = document.getElementById('head-user');
  if (me) {
    me.addEventListener('click', async () => {
      await fetch('/' + TOKEN + '/cambiar', { method: 'POST', body: '' });
      location.reload();
    });
  }
  const add = document.getElementById('np-add');
  if (add) {
    add.addEventListener('click', async () => {
      const v = id => document.getElementById(id).value;
      const body = JSON.stringify({
        name: v('np-name'), registry: v('np-registry'), pin: v('np-pin'), pin2: v('np-pin2'),
      });
      const r = await fetch('/' + TOKEN + '/perfil', { method: 'POST', body });
      if (r.ok) { location.reload(); return; }
      const box = document.getElementById('np-error');
      box.querySelector('span').textContent = (await r.json()).error;
      box.hidden = false;
    });
  }
})();
"""

_JS_WHO = """
function post(path, body) {
  return fetch('/' + TOKEN + path, { method: 'POST', body: body || '' });
}
let chosen = null;
document.querySelectorAll('.person[data-id]').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.person[data-id]').forEach(
    x => x.setAttribute('aria-pressed', 'false'));
  b.setAttribute('aria-pressed', 'true');
  chosen = b.dataset.id;
  document.getElementById('who-name').textContent = b.dataset.name;
  document.getElementById('pin-error').hidden = true;
  document.getElementById('pin').focus();
}));
const first = document.querySelector('.person[data-id]');
if (first) { first.click(); }
async function enter() {
  const pin = document.getElementById('pin').value;
  if (!chosen || !pin) { return; }
  const r = await post('/entrar', JSON.stringify({ profile: chosen, pin }));
  if (r.ok) { location.reload(); return; }
  const err = document.getElementById('pin-error');
  err.textContent = (await r.json()).error;
  err.hidden = false;
  document.getElementById('pin').value = '';
}
document.getElementById('enter').addEventListener('click', enter);
document.getElementById('pin').addEventListener('keydown', ev => {
  if (ev.key === 'Enter') { enter(); }
});
document.getElementById('exit').addEventListener('click', async () => {
  await post('/salir');
  document.body.textContent = W.closed_text;
});
"""

_JS_NEW_PROFILE = """
function post(path, body) {
  return fetch('/' + TOKEN + path, { method: 'POST', body: body || '' });
}
document.getElementById('np-add').addEventListener('click', async () => {
  const v = id => document.getElementById(id).value;
  const r = await post('/perfil', JSON.stringify({
    name: v('np-name'), registry: v('np-registry'), pin: v('np-pin'), pin2: v('np-pin2') }));
  if (r.ok) { location.reload(); return; }
  const err = document.getElementById('np-error');
  err.textContent = (await r.json()).error;
  err.hidden = false;
});
document.getElementById('exit').addEventListener('click', async () => {
  await post('/salir');
  document.body.textContent = W.closed_text;
});
"""

_JS_LOCKED = """
function post(path, body) {
  return fetch('/' + TOKEN + path, { method: 'POST', body: body || '' });
}
async function unlock() {
  const pin = document.getElementById('pin').value;
  const r = await post('/desbloquear', JSON.stringify({ pin }));
  if (r.ok) { location.reload(); return; }
  const err = document.getElementById('pin-error');
  err.textContent = (await r.json()).error;
  err.hidden = false;
  document.getElementById('pin').value = '';
}
document.getElementById('unlock').addEventListener('click', unlock);
document.getElementById('pin').addEventListener('keydown', ev => {
  if (ev.key === 'Enter') { unlock(); }
});
document.getElementById('other').addEventListener('click', async () => {
  await post('/cambiar');
  location.reload();
});
"""


_JS_TO_SETTINGS = """
document.getElementById('to-settings').addEventListener('click', async () => {
  await fetch('/' + TOKEN + '/ajustes', { method: 'POST', body: '' });
  location.reload();
});
"""

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
