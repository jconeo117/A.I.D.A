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
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
from typing import Any, Protocol

import yaml

from aida.arms.health.audio import to_wav
from aida.arms.health.capture import ModelUnavailableError, TranscriptionError
from aida.arms.health.consultation import (
    Consultation,
    ConsultationResult,
    PreparedConsultation,
)
from aida.arms.health.note import RenderedNote
from aida.arms.health.review import ReviewItem, review_items
from aida.arms.health.review_ui import (
    _CSS,
    _ICONS,
    HistoryView,
    ReviewSubmission,
    UISettings,
    _render_page,
    history_view,
    parse_submission,
    reviewer_from,
)
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import Event, EventKind, EventLog
from aida.engine.ontology import AgentId

_RAIZ = Path(__file__).resolve().parents[4]
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
    "consent_label",
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
    "ready",
    "start_button",
    "record_button",
    "exit_label",
    "closed_text",
    "recording_pill",
    "recording_state",
    "recording_hint",
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


@dataclass(frozen=True, slots=True)
class AppWording:
    """Los textos de la aplicación, validados al cargar: una pantalla a medias se
    descubre con alguien esperando delante."""

    texts: Mapping[str, str]
    steps: tuple[str, ...]

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
    return AppWording(
        texts={clave: seccion[clave] for clave in _CLAVES_APP}, steps=tuple(pasos)
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
    consultation: Consultation, draft: ReviewDraft, submission: ReviewSubmission
) -> ConsultationResult:
    return consultation.review(
        draft.prepared,
        reviewer=reviewer_from(submission),
        elapsed_seconds=submission.elapsed_seconds,
        review_findings=False,
        note=draft.note,
        note_edits=submission.note_fields,
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

    def clean(self) -> None:
        """Lo subido o grabado no se queda en disco: el audio en claro es lo más
        sensible de una sesión, y la transcripción ya vive en la caché cifrada."""
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
) -> None:
    """Levanta la aplicación y bloquea hasta que alguien pulsa «Salir» (o Ctrl-C).
    Una sesión cada vez: mientras una se procesa, no empieza otra."""
    for resto in sorted(Path(tempfile.gettempdir()).glob(f"{_WORKDIR_PREFIX}*")):
        shutil.rmtree(resto, ignore_errors=True)
    token = token_urlsafe(16)
    lock = threading.Lock()
    session = _Session(ready=build())
    if isinstance(session.ready, AppProblem):
        session.state = "problem"
        session.problem = session.ready

    def procesar(ready: AppReady, owner: str, audio: Path, document: Path) -> None:
        try:
            wav = audio.with_name("audio-16k.wav")
            to_wav(audio, wav)
            prepared = ready.consultation.prepare(
                owner=AgentId(owner),
                audio_path=wav,
                history_text=lambda: ready.reader.read(document),
            )
            draft = draft_for_screen(ready.consultation, prepared)
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
            session.drafted = True
            session.draft = draft
            session.view = vista
            session.started = clock()
            session.state = "review"

    def pagina() -> str:
        momento = now()
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
            )
        marca = settings.wording.brand
        if session.state == "working":
            return _working_page(wording, marca, token, session.owner, momento)
        if session.state == "problem" and session.problem is not None:
            return _problem_page(wording, marca, token, session.problem, momento)
        if isinstance(session.ready, AppReady):
            return _start_page(wording, marca, token, session.ready, momento)
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
                if self.path == f"/{token}":
                    self._send(200, pagina().encode("utf-8"), "text/html; charset=utf-8")
                    return
                if self.path == f"/{token}/estado":
                    pasos: tuple[bool, ...] = ()
                    if isinstance(session.ready, AppReady):
                        eventos = list(session.ready.log)[session.start_index :]
                        pasos = progress(eventos, drafted=session.drafted)
                    self._json(200, {"state": session.state, "steps": list(pasos)})
                    return
                vista = session.view
                if (
                    self.path == f"/{token}/historia"
                    and session.state == "review"
                    and vista is not None
                    and vista.content_type is not None
                ):
                    self._send(200, vista.data, vista.content_type)
                    return
            self._404()

        def do_POST(self) -> None:
            ruta = self.path
            if not ruta.startswith(f"/{token}/"):
                self._404()
                return
            accion = ruta[len(token) + 2 :]
            if accion in ("audio", "documento"):
                self._upload(accion)
            elif accion == "empezar":
                self._start()
            elif accion == "guardar":
                self._save()
            elif accion == "volver":
                with lock:
                    session.clean()
                    if isinstance(session.ready, AppReady):
                        session.state = "start"
                        session.problem = None
                self._json(200, {"ok": True})
            elif accion == "reintentar":
                with lock:
                    session.clean()
                    session.ready = build()
                    if isinstance(session.ready, AppProblem):
                        session.state = "problem"
                        session.problem = session.ready
                    else:
                        session.state = "start"
                        session.problem = None
                self._json(200, {"ok": True})
            elif accion == "salir":
                with lock:
                    session.clean()
                    session.state = "closed"
                self._json(200, {"ok": True})
                threading.Thread(target=http_server.shutdown, daemon=True).start()
            else:
                self._404()

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
                if pedido.get("recorded") is True and pedido.get("consent") is not True:
                    self._json(400, {"error": wording["missing_consent"]})
                    return
                if session.audio is None or session.document is None:
                    self._json(400, {"error": wording["missing_document"]})
                    return
                session.owner = owner
                session.start_index = len(ready.log)
                session.drafted = False
                session.state = "working"
                audio, document = session.audio, session.document
            threading.Thread(
                target=procesar, args=(ready, owner, audio, document), daemon=True
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
                close_from_screen(ready.consultation, borrador, envio)
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
) -> str:
    e = html.escape
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{e(title)}</title>
<style>
{_CSS}
{_CSS_EXTRA}
</style>
</head>
<body>
{_ICONS_APP}
<header class="top">
  <div class="mark"><i>{e(brand[:1])}</i>{e(brand)}</div>
  <div class="sep"></div>
  <div class="title"><strong>{e(title)}</strong>
    <span>{e(place)} · {moment.strftime("%d/%m/%Y")}</span></div>
  <div class="head-right">
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
    w: AppWording, brand: str, token: str, ready: AppReady, moment: datetime
) -> str:
    e = html.escape
    chips = (
        (w["check_cloud"] if ready.cloud else w["check_local_only"], ready.cloud),
        (w["check_local"], True),
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
    barras = "".join("<span></span>" for _ in range(12))
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
      <label class="consent" id="consent-box" hidden>
        <input type="checkbox" id="consent">{e(w["consent_label"])}</label>
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
</section>
<section class="screen" id="s-recording" hidden>
  <div class="stage">
    <div class="panel-card">
      <div class="rec">
        <div class="rec-dot"></div>
        <div class="rec-time" id="rec-time">00:00</div>
        <div class="meter">{barras}</div>
        <div class="rec-state" id="rec-state"></div>
      </div>
    </div>
  </div>
  <footer class="bar">
    <span class="stat">{e(w["recording_hint"])}</span>
    <button type="button" class="btn btn-primary btn-stop" id="stop">
      <svg><use href="#i-stop"/></svg>{e(w["stop_button"])}</button>
  </footer>
</section>"""
    claves = (
        "lead_upload", "lead_record", "owner_invalid", "change_file", "mic_denied",
        "missing_owner", "missing_audio", "missing_document", "missing_consent", "ready",
        "start_button", "record_button", "closed_text", "recording_pill", "recording_state",
    )
    textos = json.dumps({c: w[c] for c in claves}, ensure_ascii=False)
    script = f"const TOKEN = {json.dumps(token)};\nconst W = {textos};\n{_JS_START}"
    return _page(
        brand=brand, title=w["title_start"], place=w["header_place"], moment=moment,
        pill=w["pill_ready"], pill_class="pill", body=body, script=script,
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
let recorder = null;
let chunks = [];
let recordedAt = 0;

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
  if (mode === 'record' && !document.getElementById('consent').checked) {
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
document.getElementById('consent').addEventListener('change', update);

function setMode(next) {
  mode = next;
  document.getElementById('mode-upload').setAttribute('aria-pressed', String(mode === 'upload'));
  document.getElementById('mode-record').setAttribute('aria-pressed', String(mode === 'record'));
  document.getElementById('drop-audio').hidden = mode !== 'upload';
  document.getElementById('mic-box').hidden = mode !== 'record';
  document.getElementById('consent-box').hidden = mode !== 'record';
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
  const sent = await post('/audio', audio, { 'X-Name': name });
  if (!sent.ok) { fail((await sent.json()).error); return; }
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

async function record() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    fail(W.mic_denied);
    return;
  }
  recorder = new MediaRecorder(stream);
  chunks = [];
  recorder.ondataavailable = ev => { if (ev.data.size) { chunks.push(ev.data); } };
  recorder.onstop = () => {
    stream.getTracks().forEach(track => track.stop());
    const type = recorder.mimeType || 'audio/webm';
    const name = type.includes('ogg') ? 'rec.ogg' : 'rec.webm';
    send(new Blob(chunks, { type: type }), name);
  };
  recorder.start(1000);
  recordedAt = Date.now();
  document.getElementById('s-start').hidden = true;
  document.getElementById('s-recording').hidden = false;
  document.getElementById('rec-state').textContent =
    W.recording_state.replace('{}', owner.value.trim());
  document.getElementById('pill-text').textContent = W.recording_pill;
  document.getElementById('head-pill').classList.add('rec-pill');
  const context = new AudioContext();
  const analyser = context.createAnalyser();
  analyser.fftSize = 64;
  context.createMediaStreamSource(stream).connect(analyser);
  const bins = new Uint8Array(analyser.frequencyBinCount);
  const bars = document.querySelectorAll('.meter span');
  const tick = () => {
    if (recorder.state !== 'recording') { return; }
    const seconds = Math.floor((Date.now() - recordedAt) / 1000);
    const mm = String(Math.floor(seconds / 60)).padStart(2, '0');
    const ss = String(seconds % 60).padStart(2, '0');
    document.getElementById('rec-time').textContent = mm + ':' + ss;
    analyser.getByteFrequencyData(bins);
    bars.forEach((bar, i) => { bar.style.height = Math.max(8, bins[i + 2] / 2.55) + '%'; });
    requestAnimationFrame(tick);
  };
  tick();
}

document.getElementById('go').addEventListener('click', () => {
  if (mode === 'upload') { send(audioFile, audioFile.name); } else { record(); }
});
document.getElementById('stop').addEventListener('click', () => {
  if (recorder) { recorder.stop(); }
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
