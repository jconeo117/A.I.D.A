"""La pantalla de revisión en el navegador: una página local que pinta los
`ReviewItem` ya construidos, recoge los veredictos del profesional y los
traduce al `Reviewer` que `review_encounter` espera.

No es un servicio: nace para una sola consulta, atiende a una persona y
muere al guardar o al expirar. Nada aquí nombra el dominio de la consulta —
título, etiquetas de botón, textos de aviso— porque todo eso sale de
`ui.yaml`; el guardián de vocabulario (`I-073`) no admite literales de dominio
dentro de `src/`.
"""

from __future__ import annotations

import html
import json
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
from typing import Any

import yaml

from aida.arms.health.note import RenderedNote
from aida.arms.health.review import Answer, ItemKind, Reviewer, ReviewItem, Verdict

_RAIZ = Path(__file__).resolve().parents[4]
_UI_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "ui.yaml"

_CLAVES_OBLIGATORIAS = (
    "title",
    "intro",
    "confirm_label",
    "reject_label",
    "codes_heading",
    "history_heading",
    "transcript_heading",
    "note_heading",
    "copy_label",
    "copied_label",
    "save_label",
    "saved_text",
    "abandoned_text",
    "timeout_seconds",
)


@dataclass(frozen=True, slots=True)
class ReviewWording:
    title: str
    intro: str
    confirm_label: str
    reject_label: str
    codes_heading: str
    history_heading: str
    transcript_heading: str
    note_heading: str
    copy_label: str
    copied_label: str
    save_label: str
    saved_text: str
    abandoned_text: str


@dataclass(frozen=True, slots=True)
class UISettings:
    wording: ReviewWording
    timeout_seconds: float


def load_ui_settings(path: Path | None = None) -> UISettings:
    """Carga y valida. Revienta con la clave que falta: una pantalla a medias
    es peor que ninguna, porque se descubre con el profesional ya esperando."""
    ruta = path if path is not None else _UI_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ruta} debe ser un mapeo YAML")
    faltan = [clave for clave in _CLAVES_OBLIGATORIAS if clave not in raw]
    if faltan:
        raise ValueError(f"faltan claves en {ruta}: {faltan}")
    espera = raw["timeout_seconds"]
    if not isinstance(espera, int | float) or isinstance(espera, bool) or espera <= 0:
        raise ValueError(f"'timeout_seconds' debe ser un número positivo en {ruta}")
    wording = ReviewWording(
        title=str(raw["title"]),
        intro=str(raw["intro"]),
        confirm_label=str(raw["confirm_label"]),
        reject_label=str(raw["reject_label"]),
        codes_heading=str(raw["codes_heading"]),
        history_heading=str(raw["history_heading"]),
        transcript_heading=str(raw["transcript_heading"]),
        note_heading=str(raw["note_heading"]),
        copy_label=str(raw["copy_label"]),
        copied_label=str(raw["copied_label"]),
        save_label=str(raw["save_label"]),
        saved_text=str(raw["saved_text"]),
        abandoned_text=str(raw["abandoned_text"]),
    )
    return UISettings(wording=wording, timeout_seconds=float(espera))


class ReviewAbandoned(RuntimeError):  # noqa: N818
    """La pantalla se cerró o expiró sin guardar. No es un error del programa:
    es una revisión incompleta, y se cuenta como tal."""


@dataclass(frozen=True, slots=True)
class ReviewSubmission:
    answers: Mapping[str, Answer]
    elapsed_seconds: float
    note_fields: Mapping[str, str] = field(default_factory=dict)
    """El texto entregado de cada campo, por `section_id`. Qué cambió lo decide
    `review_note`, no la página."""


def _render_item_row(item: ReviewItem) -> str:
    return (
        f'<div class="item" data-ref="{html.escape(item.ref)}">'
        f'<p class="item-text"><strong>{html.escape(item.label)}</strong> '
        f"{html.escape(item.text)}</p>"
        f'<label class="confirm"><input type="checkbox" class="confirm-box" checked>'
        f'<span class="confirm-label"></span></label>'
        f'<button type="button" class="reject-button"></button>'
        f"</div>"
    )


def _render_codes_block(heading: str, items: list[ReviewItem]) -> str:
    if not items:
        return ""
    filas = "\n".join(_render_item_row(item) for item in items)
    return f'<section class="codes"><h2>{html.escape(heading)}</h2>\n{filas}\n</section>'


def _render_page(
    items: tuple[ReviewItem, ...],
    note: RenderedNote,
    history: str,
    transcript: str,
    wording: ReviewWording,
    token: str,
) -> str:
    """Tres paneles —historia y transcripción para leer, la nota para editar y
    copiar campo a campo— y debajo los códigos. El estado inicial de cada código
    es *confirmado*: el profesional quita lo que sobra, no confirma lo que ya está
    bien."""
    items_json = json.dumps([{"ref": item.ref} for item in items], ensure_ascii=False)
    wording_json = json.dumps(
        {
            "confirm_label": wording.confirm_label,
            "reject_label": wording.reject_label,
            "copy_label": wording.copy_label,
            "copied_label": wording.copied_label,
        },
        ensure_ascii=False,
    )
    campos = "\n".join(
        f'<div class="field" data-field-id="{html.escape(spec.section_id)}">'
        f"<h3>{html.escape(spec.label)}</h3>"
        f'<textarea class="field-text">{html.escape(texto)}</textarea>'
        f'<button type="button" class="copy-button"></button>'
        f"</div>"
        for spec, texto in note.fields
    )
    bloque_codigos = _render_codes_block(wording.codes_heading, list(items))

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{html.escape(wording.title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<h1>{html.escape(wording.title)}</h1>
<p class="intro">{html.escape(wording.intro)}</p>
<form id="review-form">
<div class="panels">
<section class="panel"><h2>{html.escape(wording.history_heading)}</h2>
<pre>{html.escape(history)}</pre></section>
<section class="panel"><h2>{html.escape(wording.transcript_heading)}</h2>
<pre>{html.escape(transcript)}</pre></section>
<section class="panel"><h2>{html.escape(wording.note_heading)}</h2>
{campos}
</section>
</div>
{bloque_codigos}
<button type="button" id="save-button">{html.escape(wording.save_label)}</button>
<p id="save-error" class="error" hidden></p>
</form>
<p id="save-ok" hidden>{html.escape(wording.saved_text)}</p>
<script>
const TOKEN = {json.dumps(token)};
const ITEMS = {items_json};
const WORDING = {wording_json};
const STARTED_AT = Date.now();
{_JS}
</script>
</body>
</html>
"""


_CSS = """
body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #1b1f24; }
h1 { font-size: 1.4rem; }
.intro { color: #44515c; }
.panels { display: grid; grid-template-columns: 1fr 1fr 1.4fr; gap: 1rem; align-items: start; }
.panel {
  border: 1px solid #d7dee5; border-radius: 6px; padding: 0.5rem 1rem;
  max-height: 75vh; overflow-y: auto;
}
.panel h2 { font-size: 1.1rem; color: #0b5fff; }
.panel pre { white-space: pre-wrap; font-family: inherit; margin: 0; }
.field { margin-bottom: 1rem; }
.field h3 { font-size: 1rem; margin: 0.5rem 0 0.3rem; }
.field-text { width: 100%; min-height: 5rem; box-sizing: border-box; font-family: inherit; }
.item {
  border-top: 1px solid #edf1f4; padding: 0.6rem 0; display: flex;
  flex-wrap: wrap; align-items: center; gap: 0.6rem;
}
.item-text { flex: 1 1 100%; margin: 0 0 0.3rem; }
.confirm { display: flex; align-items: center; gap: 0.3rem; }
button {
  background: #0b5fff; color: white; border: none; border-radius: 4px;
  padding: 0.4rem 0.8rem; cursor: pointer;
}
button.reject-button, button.copy-button { background: #eef3ff; color: #0b5fff; }
.codes { margin-top: 1.5rem; }
.codes h2 { font-size: 1.1rem; color: #0b5fff; }
#save-button { margin-top: 1rem; }
.error { color: #c0392b; }
"""

_JS = """
function label(cls, text) { document.querySelectorAll(cls).forEach(el => el.textContent = text); }
label('.confirm-label', WORDING.confirm_label);
label('.reject-button', WORDING.reject_label);
label('.copy-button', WORDING.copy_label);

const state = {};
ITEMS.forEach(item => { state[item.ref] = 'confirmed'; });

document.querySelectorAll('.item').forEach(row => {
  const ref = row.dataset.ref;
  const confirmBox = row.querySelector('.confirm-box');
  confirmBox.addEventListener('change', () => {
    state[ref] = confirmBox.checked ? 'confirmed' : 'rejected';
  });
  row.querySelector('.reject-button').addEventListener('click', () => {
    confirmBox.checked = false;
    state[ref] = 'rejected';
  });
});

document.querySelectorAll('.field').forEach(field => {
  const text = field.querySelector('.field-text');
  const button = field.querySelector('.copy-button');
  button.addEventListener('click', () => {
    navigator.clipboard.writeText(text.value).then(() => {
      button.textContent = WORDING.copied_label;
      setTimeout(() => { button.textContent = WORDING.copy_label; }, 1500);
    });
  });
});

document.getElementById('save-button').addEventListener('click', () => {
  const errorBox = document.getElementById('save-error');
  errorBox.hidden = true;
  const answers = {};
  for (const ref in state) { answers[ref] = { verdict: state[ref] }; }
  const note = {};
  document.querySelectorAll('.field').forEach(field => {
    note[field.dataset.fieldId] = field.querySelector('.field-text').value;
  });
  fetch('/' + TOKEN + '/guardar', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      elapsed_seconds: (Date.now() - STARTED_AT) / 1000,
      answers: answers,
      note: note,
    }),
  }).then(resp => {
    if (resp.ok) {
      document.getElementById('review-form').hidden = true;
      document.getElementById('save-ok').hidden = false;
    } else {
      resp.json().then(data => { errorBox.textContent = data.error; errorBox.hidden = false; });
    }
  });
});
"""


def _parse_answers(
    raw: object, items: tuple[ReviewItem, ...]
) -> tuple[dict[str, Answer], str | None]:
    if not isinstance(raw, dict):
        return {}, "answers debe ser un mapeo"
    por_ref = {item.ref: item for item in items}
    refs_esperadas = set(por_ref)
    refs_recibidas = set(raw)
    if refs_recibidas != refs_esperadas:
        return {}, "las respuestas no cubren exactamente los ítems de la revisión"
    answers: dict[str, Answer] = {}
    for ref, entry in raw.items():
        if not isinstance(entry, dict) or "verdict" not in entry:
            return {}, f"respuesta inválida para {ref!r}"
        try:
            verdict = Verdict(entry["verdict"])
        except ValueError:
            return {}, f"veredicto desconocido para {ref!r}"
        item = por_ref[ref]
        if verdict not in item.allowed:
            return {}, f"veredicto {verdict.value!r} no admitido para {ref!r}"
        text = entry.get("text")
        if verdict is Verdict.EDITED:
            if not isinstance(text, str) or not text.strip():
                return {}, f"edición sin texto para {ref!r}"
        elif text is not None:
            return {}, f"texto sin edición para {ref!r}"
        answers[ref] = Answer(verdict=verdict, text=text if verdict is Verdict.EDITED else None)
    return answers, None


@dataclass
class _Outcome:
    submission: ReviewSubmission | None = None


def collect_review(
    items: tuple[ReviewItem, ...],
    *,
    note: RenderedNote,
    history: str,
    transcript: str,
    settings: UISettings,
    open_browser: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ReviewSubmission:
    """Levanta la pantalla, espera a que se guarde y devuelve lo contestado.

    Bloquea a propósito: quien llama está a mitad de una consulta y no tiene
    nada que hacer hasta que el profesional termine. Un servidor que viviera
    más que esta llamada sería un servicio, y esto no lo es: nace, atiende a
    una persona y muere."""
    if any(item.kind is not ItemKind.CODE for item in items):
        raise ValueError("la pantalla solo confirma códigos: los hallazgos se leen en la nota")
    token = token_urlsafe(16)
    wording = settings.wording
    outcome = _Outcome()
    done = threading.Event()
    started = clock()
    page = _render_page(items, note, history, transcript, wording, token)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def _responder_json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _responder_404(self) -> None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == f"/{token}":
                cuerpo = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)
                return
            if self.path == f"/{token}/listo":
                cuerpo = wording.saved_text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)
                return
            self._responder_404()

        def do_POST(self) -> None:
            if self.path != f"/{token}/guardar" or done.is_set():
                self._responder_404()
                return
            longitud = int(self.headers.get("Content-Length", "0"))
            crudo = self.rfile.read(longitud)
            try:
                payload = json.loads(crudo)
            except ValueError:
                self._responder_json(400, {"error": "cuerpo inválido"})
                return
            if not isinstance(payload, dict):
                self._responder_json(400, {"error": "cuerpo inválido"})
                return
            answers, error = _parse_answers(payload.get("answers"), items)
            if error is not None:
                self._responder_json(400, {"error": error})
                return
            campos = payload.get("note")
            ids = sorted(spec.section_id for spec, _texto in note.fields)
            if (
                not isinstance(campos, dict)
                or sorted(campos) != ids
                or not all(isinstance(v, str) for v in campos.values())
            ):
                self._responder_json(400, {"error": "la nota no cubre exactamente sus campos"})
                return
            declarado_elapsed = payload.get("elapsed_seconds")
            if not isinstance(declarado_elapsed, int | float) or isinstance(
                declarado_elapsed, bool
            ) or declarado_elapsed < 0:
                self._responder_json(400, {"error": "elapsed_seconds inválido"})
                return
            medido_elapsed = clock() - started
            elapsed_seconds = min(float(declarado_elapsed), medido_elapsed)
            outcome.submission = ReviewSubmission(
                answers=answers, elapsed_seconds=elapsed_seconds, note_fields=dict(campos)
            )
            self._responder_json(200, {"ok": True})
            done.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    hilo_http = threading.Thread(target=server.serve_forever, daemon=True)
    hilo_http.start()
    url = f"http://127.0.0.1:{server.server_port}/{token}"
    if open_browser is not None:
        open_browser(url)
    else:
        webbrowser.open(url)

    guardado = done.wait(timeout=settings.timeout_seconds)
    server.shutdown()
    server.server_close()
    hilo_http.join(timeout=5)

    if not guardado or outcome.submission is None:
        raise ReviewAbandoned("la revisión se cerró o expiró sin guardar")
    return outcome.submission


def reviewer_from(submission: ReviewSubmission) -> Reviewer:
    """El `Reviewer` que `review_encounter` espera, leyendo de lo ya
    contestado.

    Sin respuesta por omisión: un ítem que la pantalla no contestó levanta. Un
    `CONFIRMED` inventado aquí sería aprobar un hallazgo que nadie vio, y el
    evento diría que sí se revisó."""

    def _reviewer(item: ReviewItem) -> Answer:
        if item.ref not in submission.answers:
            raise ValueError(f"la pantalla no contestó al ítem {item.ref!r}")
        return submission.answers[item.ref]

    return _reviewer
