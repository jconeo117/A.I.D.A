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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
from typing import Any

import yaml

from aida.arms.health.review import Answer, ItemKind, Reviewer, ReviewItem, Verdict

_RAIZ = Path(__file__).resolve().parents[4]
_UI_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "ui.yaml"

_CLAVES_OBLIGATORIAS = (
    "title",
    "intro",
    "confirm_label",
    "edit_label",
    "reject_label",
    "section_reviewed_label",
    "codes_heading",
    "save_label",
    "saved_text",
    "missing_sections_text",
    "abandoned_text",
    "timeout_seconds",
)


@dataclass(frozen=True, slots=True)
class ReviewWording:
    title: str
    intro: str
    confirm_label: str
    edit_label: str
    reject_label: str
    section_reviewed_label: str
    codes_heading: str
    save_label: str
    saved_text: str
    missing_sections_text: str
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
        edit_label=str(raw["edit_label"]),
        reject_label=str(raw["reject_label"]),
        section_reviewed_label=str(raw["section_reviewed_label"]),
        codes_heading=str(raw["codes_heading"]),
        save_label=str(raw["save_label"]),
        saved_text=str(raw["saved_text"]),
        missing_sections_text=str(raw["missing_sections_text"]),
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


def _section_ids_with_findings(items: tuple[ReviewItem, ...]) -> tuple[str, ...]:
    vistos: list[str] = []
    for item in items:
        if item.kind is not ItemKind.FINDING:
            continue
        section_id = item.ref.rsplit(":", 1)[0]
        if section_id not in vistos:
            vistos.append(section_id)
    return tuple(vistos)


def _render_page(items: tuple[ReviewItem, ...], wording: ReviewWording, token: str) -> str:
    """Un bloque por sección —en acordeón, colapsado salvo el primero— con sus
    hallazgos, y un bloque final con los códigos. El estado inicial de cada
    ítem es *confirmado*: el profesional corrige lo que está mal, no confirma
    lo que ya está bien."""
    secciones: list[tuple[str, str, list[ReviewItem]]] = []
    for item in items:
        if item.kind is not ItemKind.FINDING:
            continue
        section_id = item.ref.rsplit(":", 1)[0]
        if secciones and secciones[-1][0] == section_id:
            secciones[-1][2].append(item)
        else:
            secciones.append((section_id, item.label, [item]))
    codigos = [item for item in items if item.kind is ItemKind.CODE]

    items_json = json.dumps(
        [
            {"ref": item.ref, "text": item.text, "allowed": [v.value for v in item.allowed]}
            for item in items
        ],
        ensure_ascii=False,
    )
    wording_json = json.dumps(
        {
            "confirm_label": wording.confirm_label,
            "edit_label": wording.edit_label,
            "reject_label": wording.reject_label,
            "section_reviewed_label": wording.section_reviewed_label,
            "save_label": wording.save_label,
            "missing_sections_text": wording.missing_sections_text,
        },
        ensure_ascii=False,
    )
    section_ids_json = json.dumps([section_id for section_id, _label, _items in secciones])

    bloques_secciones = "\n".join(
        _render_section_block(section_id, label, bloque_items, abierto=index == 0)
        for index, (section_id, label, bloque_items) in enumerate(secciones)
    )
    bloque_codigos = _render_codes_block(wording.codes_heading, codigos)

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
{bloques_secciones}
{bloque_codigos}
<button type="button" id="save-button">{html.escape(wording.save_label)}</button>
<p id="save-error" class="error" hidden></p>
</form>
<p id="save-ok" hidden>{html.escape(wording.saved_text)}</p>
<script>
const TOKEN = {json.dumps(token)};
const ITEMS = {items_json};
const WORDING = {wording_json};
const SECTION_IDS = {section_ids_json};
const STARTED_AT = Date.now();
{_JS}
</script>
</body>
</html>
"""


def _render_section_block(
    section_id: str, label: str, items: list[ReviewItem], *, abierto: bool
) -> str:
    filas = "\n".join(_render_item_row(item) for item in items)
    abierto_attr = " open" if abierto else ""
    etiqueta = html.escape(label)
    return f"""<details class="section" data-section-id="{section_id}"{abierto_attr}>
<summary>{etiqueta}</summary>
<label class="section-reviewed">
<input type="checkbox" class="section-reviewed-box" data-section-id="{section_id}">
<span class="section-reviewed-text"></span>
</label>
{filas}
</details>"""


def _render_codes_block(heading: str, items: list[ReviewItem]) -> str:
    if not items:
        return ""
    filas = "\n".join(_render_item_row(item) for item in items)
    return f'<section class="codes"><h2>{html.escape(heading)}</h2>\n{filas}\n</section>'


def _render_item_row(item: ReviewItem) -> str:
    return (
        f'<div class="item" data-ref="{html.escape(item.ref)}" data-kind="{item.kind.value}">'
        f'<p class="item-text">{html.escape(item.text)}</p>'
        f'<label class="confirm"><input type="checkbox" class="confirm-box" checked>'
        f'<span class="confirm-label"></span></label>'
        f'<button type="button" class="edit-button" hidden></button>'
        f'<button type="button" class="reject-button" hidden></button>'
        f'<textarea class="edit-text" hidden></textarea>'
        f"</div>"
    )


_CSS = """
body {
  font-family: system-ui, sans-serif; max-width: 48rem; margin: 2rem auto;
  color: #1b1f24;
}
h1 { font-size: 1.4rem; }
.intro { color: #44515c; margin-bottom: 1.5rem; }
details.section {
  border: 1px solid #d7dee5; border-radius: 6px;
  margin-bottom: 0.75rem; padding: 0.5rem 1rem;
}
details.section summary { font-weight: 600; cursor: pointer; color: #0b5fff; }
.section-reviewed {
  display: flex; align-items: center; gap: 0.4rem;
  margin: 0.5rem 0 0.75rem; color: #44515c;
}
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
button.edit-button, button.reject-button { background: #eef3ff; color: #0b5fff; }
.edit-text { flex: 1 1 100%; min-height: 3rem; }
.codes h2 { font-size: 1.1rem; color: #0b5fff; }
#save-button { margin-top: 1rem; }
.error { color: #c0392b; }
"""

_JS = """
function label(cls, text) { document.querySelectorAll(cls).forEach(el => el.textContent = text); }
label('.confirm-label', WORDING.confirm_label);
label('.section-reviewed-text', WORDING.section_reviewed_label);
document.querySelectorAll('.edit-button').forEach(btn => btn.hidden = false);
document.querySelectorAll('.reject-button').forEach(btn => btn.hidden = false);
label('.edit-button', WORDING.edit_label);
label('.reject-button', WORDING.reject_label);

const state = {};
ITEMS.forEach(item => { state[item.ref] = { verdict: 'confirmed', text: null }; });

document.querySelectorAll('.item').forEach(row => {
  const ref = row.dataset.ref;
  const confirmBox = row.querySelector('.confirm-box');
  const editButton = row.querySelector('.edit-button');
  const rejectButton = row.querySelector('.reject-button');
  const editText = row.querySelector('.edit-text');
  const allowed = ITEMS.find(i => i.ref === ref).allowed;
  if (!allowed.includes('edited')) { editButton.hidden = true; }

  confirmBox.addEventListener('change', () => {
    if (confirmBox.checked) {
      state[ref] = { verdict: 'confirmed', text: null };
      editText.hidden = true;
      rejectButton.classList.remove('active');
    } else {
      state[ref] = { verdict: 'rejected', text: null };
      editText.hidden = true;
    }
  });
  editButton.addEventListener('click', () => {
    editText.hidden = false;
    confirmBox.checked = false;
    editText.focus();
    state[ref] = { verdict: 'edited', text: editText.value };
  });
  editText.addEventListener('input', () => {
    state[ref] = { verdict: 'edited', text: editText.value };
  });
  rejectButton.addEventListener('click', () => {
    confirmBox.checked = false;
    editText.hidden = true;
    state[ref] = { verdict: 'rejected', text: null };
  });
});

const sectionsReviewed = new Set();
document.querySelectorAll('.section-reviewed-box').forEach(box => {
  box.addEventListener('change', () => {
    const id = box.dataset.sectionId;
    if (box.checked) { sectionsReviewed.add(id); } else { sectionsReviewed.delete(id); }
  });
});

document.getElementById('save-button').addEventListener('click', () => {
  const missing = SECTION_IDS.filter(id => !sectionsReviewed.has(id));
  const errorBox = document.getElementById('save-error');
  if (missing.length > 0) {
    errorBox.textContent = WORDING.missing_sections_text;
    errorBox.hidden = false;
    return;
  }
  errorBox.hidden = true;
  const answers = {};
  for (const ref in state) {
    const entry = state[ref];
    answers[ref] = entry.text !== null
      ? { verdict: entry.verdict, text: entry.text }
      : { verdict: entry.verdict };
  }
  const payload = {
    elapsed_seconds: (Date.now() - STARTED_AT) / 1000,
    sections_reviewed: Array.from(sectionsReviewed),
    answers: answers,
  };
  fetch('/' + TOKEN + '/guardar', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(resp => {
    if (resp.ok) {
      document.getElementById('review-form').hidden = true;
      document.getElementById('save-ok').hidden = false;
    } else {
      resp.json().then(data => {
        errorBox.textContent = data.error || WORDING.missing_sections_text;
        errorBox.hidden = false;
      });
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
    settings: UISettings,
    open_browser: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ReviewSubmission:
    """Levanta la pantalla, espera a que se guarde y devuelve lo contestado.

    Bloquea a propósito: quien llama está a mitad de una consulta y no tiene
    nada que hacer hasta que el profesional termine. Un servidor que viviera
    más que esta llamada sería un servicio, y esto no lo es: nace, atiende a
    una persona y muere."""
    token = token_urlsafe(16)
    wording = settings.wording
    required_sections = _section_ids_with_findings(items)
    outcome = _Outcome()
    done = threading.Event()
    started = clock()
    page = _render_page(items, wording, token)

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
            secciones_marcadas = payload.get("sections_reviewed")
            if not isinstance(secciones_marcadas, list) or not all(
                isinstance(s, str) for s in secciones_marcadas
            ):
                self._responder_json(400, {"error": wording.missing_sections_text})
                return
            if set(required_sections) - set(secciones_marcadas):
                self._responder_json(400, {"error": wording.missing_sections_text})
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
                answers=answers, elapsed_seconds=elapsed_seconds
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
