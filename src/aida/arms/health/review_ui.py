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
import mimetypes
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
from typing import Any

import yaml

from aida.arms.health.match import CodeOrigin, MatchedCode
from aida.arms.health.note import RenderedNote
from aida.arms.health.review import Answer, ItemKind, Reviewer, ReviewItem, Verdict
from aida.arms.health.sections import SectionSpec
from aida.arms.health.segment import is_interrogative
from aida.arms.health.traceability import AbsenceFlag, ReviewTrace
from aida.resources import data_root

_RAIZ = data_root()
_UI_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "ui.yaml"

_CLAVES_OBLIGATORIAS = (
    "brand",
    "title",
    "note_heading",
    "note_hint",
    "copy_label",
    "copied_label",
    "copy_all_label",
    "edited_label",
    "empty_field_placeholder",
    "codes_tab",
    "transcript_tab",
    "history_tab",
    "origin_today_label",
    "origin_history_label",
    "origin_both_label",
    "include_label",
    "provenance_template",
    "fallback_pill",
    "edited_singular",
    "edited_plural",
    "codes_included",
    "save_label",
    "saved_text",
    "abandoned_text",
    "timeout_seconds",
)


@dataclass(frozen=True, slots=True)
class ReviewWording:
    brand: str
    title: str
    note_heading: str
    note_hint: str
    copy_label: str
    copied_label: str
    copy_all_label: str
    edited_label: str
    empty_field_placeholder: str
    codes_tab: str
    transcript_tab: str
    history_tab: str
    origin_today_label: str
    origin_history_label: str
    origin_both_label: str
    include_label: str
    provenance_template: str
    fallback_pill: str
    edited_singular: str
    edited_plural: str
    codes_included: str
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
        brand=str(raw["brand"]),
        title=str(raw["title"]),
        note_heading=str(raw["note_heading"]),
        note_hint=str(raw["note_hint"]),
        copy_label=str(raw["copy_label"]),
        copied_label=str(raw["copied_label"]),
        copy_all_label=str(raw["copy_all_label"]),
        edited_label=str(raw["edited_label"]),
        empty_field_placeholder=str(raw["empty_field_placeholder"]),
        codes_tab=str(raw["codes_tab"]),
        transcript_tab=str(raw["transcript_tab"]),
        history_tab=str(raw["history_tab"]),
        origin_today_label=str(raw["origin_today_label"]),
        origin_history_label=str(raw["origin_history_label"]),
        origin_both_label=str(raw["origin_both_label"]),
        include_label=str(raw["include_label"]),
        provenance_template=str(raw["provenance_template"]),
        fallback_pill=str(raw["fallback_pill"]),
        edited_singular=str(raw["edited_singular"]),
        edited_plural=str(raw["edited_plural"]),
        codes_included=str(raw["codes_included"]),
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
    absences_confirmed: frozenset[str] = frozenset()
    """Las ausencias sin respaldo que el profesional confirmó (plan 84). Si basta o no
    lo decide `Consultation.review`, no la página."""


@dataclass(frozen=True, slots=True)
class HistoryView:
    """Cómo se enseña la historia. El médico quiere ver su documento, no lo que el
    OCR leyó de él: si el navegador sabe pintar el archivo, se sirve tal cual; si
    no, queda el texto. `content_type is None` significa texto."""

    content_type: str | None
    data: bytes
    text: str


def history_view(path: Path | None, text: str) -> HistoryView:
    """PDF o imagen, el archivo; cualquier otra cosa, el texto del OCR. Se lee una
    vez aquí: la pantalla enseña la historia con la que se cotejó, aunque el archivo
    cambie después en disco."""
    if path is not None:
        tipo, _codificacion = mimetypes.guess_type(path.name)
        if tipo is not None and (tipo == "application/pdf" or tipo.startswith("image/")):
            return HistoryView(content_type=tipo, data=path.read_bytes(), text=text)
    return HistoryView(content_type=None, data=b"", text=text)


def _render_field(
    spec: SectionSpec,
    texto: str,
    propio: str | None,
    wording: ReviewWording,
    copiar: str,
    trace: ReviewTrace | None = None,
    not_recorded: bool = False,
) -> str:
    e = html.escape
    visible = propio if propio is not None else texto
    editado = propio is not None and propio.strip() != texto.strip()
    marcas = [f for f in (trace.flags if trace else ()) if f.section_id == spec.section_id]
    clases = (
        "field"
        + ("" if visible.strip() else " empty")
        + (" edited" if editado else "")
        + (" flagged" if marcas else "")
    )
    # Lo que escribió el motor porque no se habló de ello (plan 84), dicho en el campo.
    del_motor = (
        f'<span class="by-aida">{e(trace.settings.not_recorded_badge)}</span>'
        if trace is not None and not_recorded
        else ""
    )
    avisos = "".join(_render_absence(m, trace) for m in marcas if trace is not None)
    return (
        f'<article class="{clases}" data-field-id="{e(spec.section_id)}" '
        f'data-original="{e(texto)}">'
        f'<div class="field-head"><h3>{e(spec.label)}</h3>{del_motor}'
        f'<span class="badge"{"" if editado else " hidden"}>{e(wording.edited_label)}</span>'
        f"{copiar}</div>"
        f'<textarea placeholder="{e(wording.empty_field_placeholder)}">{e(visible)}</textarea>'
        f"{avisos}</article>"
    )


def _render_absence(flag: AbsenceFlag, trace: ReviewTrace) -> str:
    """El aviso de una ausencia sin respaldo: la frase entre comillas y el botón para
    confirmarla. La frase va en `data-sentence`: si el profesional la corrige, el aviso
    se resuelve solo."""
    e = html.escape
    frase = flag.sentence.rstrip(" .")
    texto = e(trace.settings.absence_flag).replace("{0}", f"<q>{e(frase)}</q>")
    return (
        f'<div class="absence" data-flag="{e(flag.id)}" data-sentence="{e(flag.sentence)}">'
        f'<svg><use href="#i-alert"/></svg><div>{texto}'
        f'<div class="absence-actions"><button type="button" class="absence-ok">'
        f"{e(trace.settings.absence_confirm)}</button></div></div></div>"
    )


def _render_prior(trace: ReviewTrace) -> str:
    """«Según la historia del …»: lo que la historia previa trae, con su fecha, al lado
    de la nota y sin tocarla (plan 84)."""
    if not trace.profile.items:
        return ""
    e = html.escape
    fecha = trace.profile.date
    ajustes = trace.settings
    titulo = ajustes.profile_title.format(fecha) if fecha else ajustes.profile_title_undated
    filas = "".join(
        f"<dt>{e(etiqueta)}</dt><dd>{e(' · '.join(lineas))}</dd>"
        for etiqueta, lineas in trace.profile.items
    )
    return f'<div class="prior"><h4>{e(titulo)}</h4><dl>{filas}</dl></div>'


def _render_page(
    matches: tuple[MatchedCode, ...],
    note: RenderedNote,
    transcript: str,
    history: HistoryView,
    owner: str,
    moment: datetime,
    wording: ReviewWording,
    token: str,
    *,
    overrides: Mapping[str, str] | None = None,
    excluded: frozenset[str] = frozenset(),
    consent_line: str | None = None,
    trace: ReviewTrace | None = None,
) -> str:
    """La nota al centro, que es lo que se copia; a la derecha, en pestañas, lo que
    se consulta para corregirla. Cada código empieza incluido: el profesional quita
    lo que sobra, no confirma lo que ya está bien."""
    e = html.escape
    origenes = {
        CodeOrigin.TODAY: wording.origin_today_label,
        CodeOrigin.HISTORY: wording.origin_history_label,
        CodeOrigin.BOTH: wording.origin_both_label,
    }
    if note.redacted and note.provider is not None:
        proveedor, _sep, modelo = note.provider.partition("/")
        procedencia = wording.provenance_template.format(proveedor, modelo)
        clase_pildora = "pill"
    else:
        procedencia = wording.fallback_pill
        clase_pildora = "pill fallback"
    copiar = (
        f'<button type="button" class="icon-btn copy" data-label="{e(wording.copy_label)}">'
        f'<svg><use href="#i-copy"/></svg><span>{e(wording.copy_label)}</span></button>'
    )
    # Lo que el profesional escribió en vivo (plan 77b) llega ya puesto y marcado
    # como editado; `data-original` guarda lo del modelo, que es contra lo que se
    # compara al guardar (`review_note`).
    propios = overrides or {}
    campos = "\n".join(
        _render_field(
            spec, texto, propios.get(spec.section_id), wording, copiar, trace,
            spec.section_id in note.not_recorded,
        )
        for spec, texto in note.fields
    )
    codigos = "\n".join(
        f'<article class="code{" off" if m.diagnostic.code in excluded else ""}" '
        f'data-ref="{e(m.diagnostic.code)}" data-origin="{m.origin.value}">'
        f'<div class="code-head"><span class="code-id">{e(m.diagnostic.code)}</span>'
        f'<span class="origin">{e(origenes[m.origin])}{_history_date(m.origin, trace)}</span>'
        f'<button type="button" class="switch" role="switch" '
        f'aria-checked="{"false" if m.diagnostic.code in excluded else "true"}" '
        f'aria-label="{e(wording.include_label)}"></button></div>'
        f'<p class="code-desc">{e(m.diagnostic.description)}</p>'
        + (
            f'<p class="code-evidence"><svg><use href="#i-quote"/></svg>{e(m.evidence)}</p>'
            if m.evidence
            else ""
        )
        + "</article>"
        for m in matches
    )
    leyenda = (
        (_render_prior(trace) if trace is not None else "")
        + f'<div class="legend"><span>{e(wording.origin_today_label)}</span>'
        f'<span class="h">{e(wording.origin_history_label)}</span>'
        f'<span class="b">{e(wording.origin_both_label)}</span></div>'
    )
    burbujas = "\n".join(
        f'<p class="{"q" if is_interrogative(linea) else "a"}">{e(linea.strip())}</p>'
        for linea in transcript.split("\n")
        if linea.strip()
    )
    if history.content_type == "application/pdf":
        historia = (
            f'<iframe class="history-frame" src="/{token}/historia" '
            f'title="{e(wording.history_tab)}"></iframe>'
        )
    elif history.content_type is not None:
        historia = (
            f'<img class="history-image" src="/{token}/historia" alt="{e(wording.history_tab)}">'
        )
    else:
        historia = f'<pre class="history-text">{e(history.text)}</pre>'
    wording_json = json.dumps(
        {
            "copy_label": wording.copy_label,
            "copied_label": wording.copied_label,
            "edited_singular": wording.edited_singular,
            "edited_plural": wording.edited_plural,
            "codes_included": wording.codes_included,
        },
        ensure_ascii=False,
    )
    marca = wording.brand[:1]

    # Las ausencias por revisar (plan 84): el contador, y los textos que usa la página.
    pendientes = (
        '\n  <span class="stat warn" id="stat-absence"></span>' if trace and trace.flags else ""
    )
    textos_traza = json.dumps(
        {
            "pending_singular": trace.settings.absences_pending_singular if trace else "",
            "pending_plural": trace.settings.absences_pending_plural if trace else "",
            "confirmed": trace.settings.absence_confirmed if trace else "",
        },
        ensure_ascii=False,
    )
    # Con qué autorización se grabó (plan 82): a la vista de quien revisa, que es quien
    # firma. Sin autorización no se llega aquí desde la aplicación ni desde la CLI.
    autorizacion = (
        f'\n  <span class="stat consent-stat"><svg><use href="#i-shield"/></svg>'
        f"{e(consent_line)}</span>"
        if consent_line
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{e(wording.title)}</title>
<style>
{_CSS}
{_CSS_CONSENT_STAT}
{_CSS_TRACE}
</style>
</head>
<body>
{_ICONS_REVIEW_TRACE}
<header class="top">
  <div class="mark"><i>{e(marca)}</i>{e(wording.brand)}</div>
  <div class="sep"></div>
  <div class="title"><strong>{e(wording.title)}</strong>
    <span>{e(owner)} · {moment.strftime("%d/%m/%Y, %H:%M")}</span></div>
  <div class="head-right">
    <span class="{clase_pildora}"><span class="dot"></span>{e(procedencia)}</span>
  </div>
</header>
<main>
  <section class="column">
    <div class="column-head">
      <div><h2>{e(wording.note_heading)}</h2><p>{e(wording.note_hint)}</p></div>
      <button type="button" class="btn" id="copy-all" data-label="{e(wording.copy_all_label)}">
        <svg><use href="#i-copy"/></svg><span>{e(wording.copy_all_label)}</span>
      </button>
    </div>
    <div class="scroll" id="note">
{campos}
    </div>
  </section>
  <section class="column">
    <div class="tabs" role="tablist">
      <button type="button" class="tab" role="tab" data-pane="codes" aria-selected="true">
        <svg><use href="#i-codes"/></svg>{e(wording.codes_tab)}
        <span class="count" id="codes-count"></span>
      </button>
      <button type="button" class="tab" role="tab" data-pane="transcript" aria-selected="false">
        <svg><use href="#i-chat"/></svg>{e(wording.transcript_tab)}
      </button>
      <button type="button" class="tab" role="tab" data-pane="history" aria-selected="false">
        <svg><use href="#i-doc"/></svg>{e(wording.history_tab)}
      </button>
    </div>
    <div class="pane scroll" id="pane-codes">
{leyenda}
{codigos}
    </div>
    <div class="pane scroll" id="pane-transcript" hidden><div class="chat">
{burbujas}
    </div></div>
    <div class="pane scroll" id="pane-history" hidden>
{historia}
    </div>
  </section>
</main>
<div class="done-panel" id="save-ok" hidden>{e(wording.saved_text)}</div>
<footer class="bar">{pendientes}
  <span class="stat" id="stat-edited"></span>
  <span class="stat" id="stat-codes"></span>{autorizacion}
  <span class="error" id="save-error" hidden></span>
  <button type="button" class="btn btn-primary" id="save">
    <svg><use href="#i-save"/></svg>{e(wording.save_label)}
  </button>
</footer>
<script>
const TOKEN = {json.dumps(token)};
const WORDING = {wording_json};
const STARTED_AT = Date.now();
const TRACE = {textos_traza};
{_JS}
{_JS_TRACE}
</script>
</body>
</html>
"""


_CSS = """
:root {
  --canvas: #eceff3; --surface: #ffffff; --surface-2: #f7f9fb;
  --ink: #111827; --ink-2: #374151; --muted: #6b7280; --faint: #9ca3af;
  --line: #e3e7ec; --line-2: #d5dbe2;
  --brand: #0f766e; --brand-ink: #0b5c56; --brand-soft: #e6f4f2; --brand-ring: rgba(15,118,110,.18);
  --head: #0d1b2a; --head-2: #16283b; --head-ink: #e6edf5; --head-muted: #8fa3b8;
  --today: #0f766e; --history: #4f46e5; --both: #9333ea;
  --warn: #b45309; --warn-soft: #fef3c7; --off: #b91c1c; --off-soft: #fee2e2;
  --r: 12px; --r-s: 8px;
  --e1: 0 1px 2px rgba(15,23,42,.05), 0 1px 1px rgba(15,23,42,.03);
  --e2: 0 4px 14px rgba(15,23,42,.08), 0 1px 3px rgba(15,23,42,.06);
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0; font: 15px/1.6 "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--ink); background: var(--canvas); display: flex; flex-direction: column;
  -webkit-font-smoothing: antialiased;
}
svg {
  width: 16px; height: 16px; flex: none; stroke: currentColor; fill: none; stroke-width: 2;
  stroke-linecap: round; stroke-linejoin: round;
}

/* Cabecera */
header.top {
  display: flex; align-items: center; gap: 1.25rem; padding: 0 1.75rem; height: 60px;
  background: linear-gradient(180deg, var(--head), var(--head-2)); color: var(--head-ink);
}
.mark {
  display: flex; align-items: center; gap: .55rem; font-weight: 700; letter-spacing: .14em;
  font-size: .85rem;
}
.mark i {
  width: 26px; height: 26px; border-radius: 7px;
  background: linear-gradient(135deg, #14b8a6, #0f766e); display: grid; place-items: center;
  font-style: normal; font-size: .75rem; letter-spacing: 0; color: white;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.15);
}
.sep { width: 1px; height: 24px; background: rgba(255,255,255,.12); }
.title { display: flex; flex-direction: column; line-height: 1.2; }
.title strong { font-size: .95rem; font-weight: 600; }
.title span { font-size: .78rem; color: var(--head-muted); }
.pill {
  display: inline-flex; align-items: center; gap: .4rem; padding: .3rem .7rem;
  border-radius: 999px; font-size: .78rem; background: rgba(255,255,255,.07);
  color: var(--head-ink); border: 1px solid rgba(255,255,255,.1);
}
.pill .dot {
  width: 7px; height: 7px; border-radius: 50%; background: #2dd4bf;
  box-shadow: 0 0 0 3px rgba(45,212,191,.2);
}
.pill.fallback .dot { background: #fbbf24; box-shadow: 0 0 0 3px rgba(251,191,36,.2); }
.head-right { margin-left: auto; display: flex; gap: .6rem; }

/* Cuerpo */
main {
  flex: 1; display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr); gap: 1.5rem;
  padding: 1.5rem 1.75rem 1rem; min-height: 0;
}
.column { display: flex; flex-direction: column; min-height: 0; }
.column-head {
  display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: 1rem;
  min-height: 40px;
}
.column-head h2 { margin: 0; font-size: 1.25rem; font-weight: 650; letter-spacing: -.01em; }
.column-head p { margin: .1rem 0 0; color: var(--muted); font-size: .85rem; }
.scroll {
  overflow-y: auto; min-height: 0; padding: 2px 6px 1rem 2px; scrollbar-width: thin;
  scrollbar-color: var(--line-2) transparent;
}
.scroll::-webkit-scrollbar {
  width: 8px; } .scroll::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 8px;
}

/* Campos de la nota */
.field {
  position: relative; background: var(--surface); border-radius: var(--r); box-shadow: var(--e1);
  border: 1px solid var(--line); margin-bottom: .85rem; overflow: hidden;
  transition: box-shadow .15s, border-color .15s;
}
.field::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: transparent;
  transition: background .15s;
}
.field:hover { box-shadow: var(--e2); }
.field:focus-within { border-color: var(--brand); box-shadow: 0 0 0 4px var(--brand-ring); }
.field:focus-within::before { background: var(--brand); }
.field.edited::before { background: var(--warn); }
.field-head { display: flex; align-items: center; gap: .6rem; padding: .7rem 1rem .1rem 1.15rem; }
.field-head h3 {
  margin: 0; font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
  color: var(--brand-ink);
}
.badge {
  font-size: .7rem; font-weight: 650; padding: .08rem .5rem; border-radius: 999px;
  background: var(--warn-soft); color: var(--warn);
}
.badge[hidden] { display: none; }
.field textarea {
  display: block; width: 100%; border: 0; resize: none; overflow: hidden;
  padding: .35rem 1.15rem .95rem; font: inherit; font-size: .97rem; color: var(--ink-2);
  background: transparent; outline: none; min-height: 2.5rem;
}
.field textarea::placeholder { color: var(--faint); font-style: italic; }
.field.empty { background: var(--surface-2); box-shadow: none; border-style: dashed; }

/* Botones */
button { font: inherit; cursor: pointer; }
.icon-btn {
  margin-left: auto; display: inline-flex; align-items: center; gap: .35rem;
  border: 1px solid transparent; background: transparent; color: var(--muted);
  padding: .25rem .55rem; border-radius: var(--r-s); font-size: .8rem; opacity: .75;
  transition: all .12s;
}
.field:hover .icon-btn, .icon-btn:focus {
  opacity: 1; background: var(--surface-2); border-color: var(--line);
}
.icon-btn:hover { color: var(--brand); }
.icon-btn.done { color: var(--brand); opacity: 1; }
.btn {
  display: inline-flex; align-items: center; gap: .45rem; border-radius: var(--r-s);
  padding: .5rem .9rem; font-size: .87rem; font-weight: 550; border: 1px solid var(--line-2);
  background: var(--surface); color: var(--ink-2); box-shadow: var(--e1);
}
.btn:hover { border-color: var(--brand); color: var(--brand); }
.btn-primary {
  border: 0; background: var(--brand); color: white; padding: .65rem 1.3rem; font-weight: 600;
  box-shadow: 0 2px 6px rgba(15,118,110,.35);
}
.btn-primary:hover { background: var(--brand-ink); color: white; }

/* Pestañas */
.tabs {
  display: inline-flex; gap: 2px; padding: 3px; background: #dfe4ea; border-radius: 10px;
  margin-bottom: 1rem; align-self: flex-start;
}
.tab {
  display: inline-flex; align-items: center; gap: .4rem; border: 0; background: transparent;
  padding: .42rem .8rem; color: var(--muted); border-radius: 8px; font-weight: 600;
  font-size: .86rem;
}
.tab[aria-selected="true"] { background: var(--surface); color: var(--ink); box-shadow: var(--e1); }
.tab .count {
  font-size: .72rem; font-variant-numeric: tabular-nums; background: var(--brand-soft);
  color: var(--brand-ink); padding: 0 .45rem; border-radius: 999px;
}
.pane[hidden] { display: none; }
.pane { flex: 1; min-height: 0; }

/* Códigos */
.code {
  position: relative; background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--r); box-shadow: var(--e1); padding: .85rem 1rem .85rem 1.15rem;
  margin-bottom: .7rem; overflow: hidden; transition: opacity .15s, box-shadow .15s;
}
.code::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--today);
}
.code[data-origin="history"]::before { background: var(--history); }
.code[data-origin="both"]::before { background: var(--both); }
.code:hover { box-shadow: var(--e2); }
.code-head { display: flex; align-items: center; gap: .6rem; }
.code-id {
  font: 650 .82rem ui-monospace, "JetBrains Mono", SFMono-Regular, Menlo, monospace;
  letter-spacing: .02em; color: var(--ink); background: var(--surface-2);
  border: 1px solid var(--line); padding: .05rem .45rem; border-radius: 6px;
}
.origin { font-size: .74rem; font-weight: 600; color: var(--today); }
.code[data-origin="history"] .origin { color: var(--history); }
.code[data-origin="both"] .origin { color: var(--both); }
.code-desc {
  margin: .5rem 0 .2rem; font-weight: 600; font-size: .93rem; text-transform: lowercase;
}
.code-desc::first-letter { text-transform: uppercase; }
.code-evidence { margin: 0; display: flex; gap: .4rem; color: var(--muted); font-size: .85rem; }
.code-evidence svg { margin-top: .2rem; width: 14px; height: 14px; color: var(--faint); }
.switch {
  margin-left: auto; position: relative; width: 38px; height: 22px; border-radius: 999px;
  border: 0; background: var(--brand); transition: background .15s; padding: 0;
}
.switch::after {
  content: ""; position: absolute; top: 3px; left: 19px; width: 16px; height: 16px;
  border-radius: 50%; background: white; box-shadow: 0 1px 2px rgba(0,0,0,.25);
  transition: left .15s;
}
.code.off { opacity: .6; background: var(--surface-2); box-shadow: none; }
.code.off::before { background: var(--line-2); }
.code.off .code-desc { text-decoration: line-through; color: var(--muted); }
.code.off .switch { background: #c7cdd4; }
.code.off .switch::after { left: 3px; }
.legend {
  display: flex; gap: 1rem; font-size: .75rem; color: var(--muted); margin: -.3rem 0 .8rem;
}
.legend span::before {
  content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 2px;
  margin-right: .35rem; background: var(--today);
}
.legend .h::before { background: var(--history); } .legend .b::before { background: var(--both); }

/* Transcripción */
.chat { display: flex; flex-direction: column; gap: .45rem; }
.chat p {
  margin: 0; max-width: 88%; padding: .5rem .8rem; border-radius: 12px; font-size: .9rem;
  line-height: 1.5;
}
.chat .q {
  align-self: flex-start; background: #e2e8ef; color: var(--ink-2); border-bottom-left-radius: 4px;
}
.chat .a {
  align-self: flex-end; background: var(--surface); border: 1px solid var(--line);
  box-shadow: var(--e1); border-bottom-right-radius: 4px;
}

/* Historia */
.history-frame {
  width: 100%; height: 100%; min-height: 60vh; border: 1px solid var(--line);
  border-radius: var(--r); background: var(--surface); box-shadow: var(--e1);
}

/* Barra */
footer.bar {
  display: flex; align-items: center; gap: 1.25rem; padding: .85rem 1.75rem;
  background: var(--surface); border-top: 1px solid var(--line);
  box-shadow: 0 -4px 16px rgba(15,23,42,.05);
}
.stat { display: flex; align-items: baseline; gap: .4rem; color: var(--muted); font-size: .85rem; }
.stat b { color: var(--ink); font-size: 1.05rem; font-variant-numeric: tabular-nums; }
.stat.warn b { color: var(--warn); }
footer .btn-primary { margin-left: auto; }
.error { color: var(--off); font-size: .88rem; }

/* Estados que la maqueta no enseña */
.history-image {
  display: block; width: 100%; height: auto; border: 1px solid var(--line);
  border-radius: var(--r); background: var(--surface); box-shadow: var(--e1);
}
.history-text {
  margin: 0; padding: 1rem 1.15rem; white-space: pre-wrap; font: inherit; font-size: .9rem;
  color: var(--ink-2); background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--r); box-shadow: var(--e1);
}
.done-panel {
  flex: 1; display: grid; place-items: center; color: var(--ink-2); font-size: 1.05rem;
}
main[hidden], footer[hidden], .done-panel[hidden] { display: none; }
"""

_ICONS = """<svg style="display:none"><defs>
  <symbol id="i-copy" viewBox="0 0 24 24">
    <rect x="9" y="9" width="12" height="12" rx="2"/>
    <path d="M5 15V5a2 2 0 0 1 2-2h10"/>
  </symbol>
  <symbol id="i-check" viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></symbol>
  <symbol id="i-quote" viewBox="0 0 24 24">
    <path d="M7 7h4v4c0 3-2 5-4 6M15 7h4v4c0 3-2 5-4 6"/>
  </symbol>
  <symbol id="i-codes" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10"/></symbol>
  <symbol id="i-chat" viewBox="0 0 24 24">
    <path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z"/>
  </symbol>
  <symbol id="i-doc" viewBox="0 0 24 24">
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/>
    <path d="M14 3v5h5M9 13h6M9 17h6"/>
  </symbol>
  <symbol id="i-save" viewBox="0 0 24 24"><path d="M5 12l5 5L20 7"/></symbol>
</defs></svg>"""

_CSS_CONSENT_STAT = """/* La 82: la autorización, a la vista en la revisión */
.stat.consent-stat { color: var(--muted); }
.stat.consent-stat svg { width: 15px; height: 15px; color: var(--brand); align-self: center; }"""

_ICONS_CONSENT = """  <symbol id="i-shield" viewBox="0 0 24 24">
    <path d="M12 3 4 6v6c0 4.5 3.4 8.3 8 9 4.6-.7 8-4.5 8-9V6Z"/><path d="m9 12 2 2 4-4"/>
  </symbol>"""

_ICONS_REVIEW = _ICONS.replace("</defs></svg>", _ICONS_CONSENT + "\n</defs></svg>")

_CSS_TRACE = """/* La 84: ausencias sin respaldo, «No consta» y la historia con su fecha */
.field.flagged::before { background: var(--warn); }
.absence {
  margin: 0 1rem .9rem 1.15rem; padding: .65rem .8rem; border-radius: 10px;
  background: var(--warn-soft); color: #78350f; font-size: .85rem; line-height: 1.5;
  display: flex; gap: .6rem; align-items: flex-start;
}
.absence svg { width: 16px; height: 16px; flex: none; margin-top: .15rem; color: var(--warn); }
.absence q { font-weight: 650; quotes: "«" "»"; }
.absence .absence-actions { display: flex; gap: .5rem; margin-top: .5rem; }
.absence button {
  font: inherit; font-size: .8rem; font-weight: 600; padding: .3rem .7rem; border-radius: 8px;
  border: 1px solid rgba(180, 83, 9, .35); background: #fff; color: #78350f; cursor: pointer;
}
.absence.ok { background: var(--surface-2); color: var(--muted); }
.absence.ok svg { color: var(--brand); }
.by-aida {
  font-size: .72rem; font-weight: 600; color: var(--muted); background: var(--surface-2);
  border-radius: 999px; padding: .1rem .55rem;
}
.prior {
  border: 1px solid var(--line); border-radius: var(--r); background: var(--surface);
  padding: .85rem 1rem; margin-bottom: .9rem; box-shadow: var(--e1);
}
.prior h4 { margin: 0 0 .5rem; font-size: .8rem; letter-spacing: .04em; text-transform: uppercase;
  color: var(--history); }
.prior dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: .3rem .8rem;
  font-size: .86rem; }
.prior dt { color: var(--muted); }
.prior dd { margin: 0; color: var(--ink); }
.origin small { font-weight: 500; opacity: .8; }
.btn-primary:disabled { opacity: .45; cursor: not-allowed; box-shadow: none; }"""

_ICONS_TRACE = """  <symbol id="i-alert" viewBox="0 0 24 24">
    <path d="M12 9v4M12 17h.01"/>
    <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>
  </symbol>"""

_ICONS_REVIEW_TRACE = _ICONS_REVIEW.replace("</defs></svg>", _ICONS_TRACE + "\n</defs></svg>")


def _history_date(origin: CodeOrigin, trace: ReviewTrace | None) -> str:
    """La fecha de la historia en los códigos que vienen de ella (plan 84)."""
    if trace is None or not trace.profile.date or origin is CodeOrigin.TODAY:
        return ""
    sufijo = trace.settings.history_date_suffix.format(trace.profile.date)
    return f" <small>· {html.escape(sufijo)}</small>"


_JS_TRACE = """
// Las ausencias sin respaldo (plan 84): una está pendiente mientras no se confirme y
// su frase siga en el campo. Con alguna pendiente no se guarda; `review` lo vuelve a
// comprobar igual.
function plain(s) {
  return s.normalize('NFD').replace(/[\\u0300-\\u036f]/g, '').toLowerCase();
}
function absences() {
  const box = document.getElementById('stat-absence');
  if (!box) { return; }
  let pending = 0;
  document.querySelectorAll('.absence').forEach(a => {
    const text = a.closest('.field').querySelector('textarea').value;
    const open = !a.dataset.confirmed && plain(text).includes(plain(a.dataset.sentence));
    a.classList.toggle('ok', !open);
    if (open) { pending += 1; }
    a.closest('.field').classList.toggle('flagged', open);
  });
  box.classList.toggle('warn', pending > 0);
  setStat(box, pending, pending === 1 ? TRACE.pending_singular : TRACE.pending_plural);
  document.getElementById('save').disabled = pending > 0;
}
document.querySelectorAll('.absence-ok').forEach(b => b.addEventListener('click', () => {
  const a = b.closest('.absence');
  a.dataset.confirmed = '1';
  b.closest('.absence-actions').replaceChildren(document.createTextNode(TRACE.confirmed));
  absences();
}));
document.querySelectorAll('.field textarea').forEach(t => t.addEventListener('input', absences));
absences();
"""

_JS = """
const grow = t => { t.style.height = 'auto'; t.style.height = t.scrollHeight + 'px'; };

function setStat(el, n, text) {
  const b = document.createElement('b');
  b.textContent = n;
  el.replaceChildren(b, document.createTextNode(text));
}

function status() {
  const edited = document.querySelectorAll('.field.edited').length;
  const all = document.querySelectorAll('.code').length;
  const on = document.querySelectorAll('.code:not(.off)').length;
  document.getElementById('codes-count').textContent = on + '/' + all;
  const se = document.getElementById('stat-edited');
  se.classList.toggle('warn', edited > 0);
  setStat(se, edited, edited === 1 ? WORDING.edited_singular : WORDING.edited_plural);
  setStat(document.getElementById('stat-codes'), on, WORDING.codes_included.replace('{}', all));
}

function copy(text, button) {
  navigator.clipboard.writeText(text).then(() => {
    const label = button.querySelector('span');
    button.classList.add('done');
    label.textContent = WORDING.copied_label;
    setTimeout(() => {
      button.classList.remove('done');
      label.textContent = button.dataset.label;
    }, 1500);
  });
}

document.querySelectorAll('.field').forEach(field => {
  const text = field.querySelector('textarea');
  const original = field.dataset.original;
  grow(text);
  text.addEventListener('input', () => {
    grow(text);
    const edited = text.value.trim() !== original.trim();
    field.classList.toggle('edited', edited);
    field.querySelector('.badge').hidden = !edited;
    field.classList.toggle('empty', !text.value.trim());
    status();
  });
  const button = field.querySelector('.copy');
  button.addEventListener('click', () => copy(text.value, button));
});

document.getElementById('copy-all').addEventListener('click', event => {
  const parts = [];
  document.querySelectorAll('.field').forEach(field => {
    const value = field.querySelector('textarea').value.trim();
    if (value) { parts.push(field.querySelector('h3').textContent + '\\n' + value); }
  });
  copy(parts.join('\\n\\n'), event.currentTarget);
});

document.querySelectorAll('.code').forEach(code => {
  const toggle = code.querySelector('.switch');
  toggle.addEventListener('click', () => {
    code.classList.toggle('off');
    toggle.setAttribute('aria-checked', String(!code.classList.contains('off')));
    status();
  });
});

document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(t => {
    t.setAttribute('aria-selected', String(t === tab));
  });
  document.querySelectorAll('.pane').forEach(p => {
    p.hidden = p.id !== 'pane-' + tab.dataset.pane;
  });
}));

document.getElementById('save').addEventListener('click', () => {
  const errorBox = document.getElementById('save-error');
  errorBox.hidden = true;
  const answers = {};
  document.querySelectorAll('.code').forEach(code => {
    const off = code.classList.contains('off');
    answers[code.dataset.ref] = { verdict: off ? 'rejected' : 'confirmed' };
  });
  const note = {};
  document.querySelectorAll('.field').forEach(field => {
    note[field.dataset.fieldId] = field.querySelector('textarea').value;
  });
  fetch('/' + TOKEN + '/guardar', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      elapsed_seconds: (Date.now() - STARTED_AT) / 1000,
      answers: answers,
      note: note,
      absences_confirmed: Array.from(
        document.querySelectorAll('.absence[data-confirmed]'), a => a.dataset.flag),
    }),
  }).then(resp => {
    if (resp.ok) {
      resp.json().then(data => {
        if (data.next) { location.href = data.next; return; }
        document.querySelector('main').hidden = true;
        document.querySelector('footer.bar').hidden = true;
        document.getElementById('save-ok').hidden = false;
      });
    } else {
      resp.json().then(data => { errorBox.textContent = data.error; errorBox.hidden = false; });
    }
  });
});

status();
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


def parse_submission(
    payload: Mapping[str, Any],
    *,
    items: tuple[ReviewItem, ...],
    note: RenderedNote,
    measured_elapsed: float,
) -> tuple[ReviewSubmission | None, str]:
    """Lo que la pantalla entregó, validado, o el motivo para rechazarlo. Es la
    única puerta para eso: la usan la pantalla de un solo uso y la aplicación
    (`app.py`), y con dos copias una acabaría aceptando lo que la otra rechaza.

    El tiempo es el menor entre el que declara la página y el que midió quien
    sirve: la página no puede alargar la revisión, y el servidor no sabe cuándo
    se terminó de leer."""
    answers, error = _parse_answers(payload.get("answers"), items)
    if error is not None:
        return None, error
    campos = payload.get("note")
    ids = sorted(spec.section_id for spec, _texto in note.fields)
    if (
        not isinstance(campos, dict)
        or sorted(campos) != ids
        or not all(isinstance(v, str) for v in campos.values())
    ):
        return None, "la nota no cubre exactamente sus campos"
    declarado = payload.get("elapsed_seconds")
    if not isinstance(declarado, int | float) or isinstance(declarado, bool) or declarado < 0:
        return None, "elapsed_seconds inválido"
    confirmadas = payload.get("absences_confirmed", [])
    if not isinstance(confirmadas, list) or not all(isinstance(c, str) for c in confirmadas):
        return None, "absences_confirmed inválido"
    return (
        ReviewSubmission(
            answers=answers,
            elapsed_seconds=min(float(declarado), measured_elapsed),
            note_fields=dict(campos),
            absences_confirmed=frozenset(confirmadas),
        ),
        "",
    )


@dataclass
class _Outcome:
    submission: ReviewSubmission | None = None


def collect_review(
    items: tuple[ReviewItem, ...],
    *,
    matches: tuple[MatchedCode, ...],
    note: RenderedNote,
    transcript: str,
    history: HistoryView,
    owner: str,
    moment: datetime,
    settings: UISettings,
    open_browser: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    consent_line: str | None = None,
    trace: ReviewTrace | None = None,
) -> ReviewSubmission:
    """Levanta la pantalla, espera a que se guarde y devuelve lo contestado.

    Bloquea a propósito: quien llama está a mitad de una consulta y no tiene
    nada que hacer hasta que el profesional termine. Un servidor que viviera
    más que esta llamada sería un servicio, y esto no lo es: nace, atiende a
    una persona y muere."""
    if any(item.kind is not ItemKind.CODE for item in items):
        raise ValueError("la pantalla solo confirma códigos: los hallazgos se leen en la nota")
    if [item.ref for item in items] != [m.diagnostic.code for m in matches]:
        raise ValueError(
            "los ítems y los códigos no coinciden: la pantalla pintaría unos y validaría otros"
        )
    token = token_urlsafe(16)
    wording = settings.wording
    outcome = _Outcome()
    done = threading.Event()
    started = clock()
    page = _render_page(
        matches, note, transcript, history, owner, moment, wording, token,
        consent_line=consent_line,
        trace=trace,
    )

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
            if self.path == f"/{token}/historia" and history.content_type is not None:
                self.send_response(200)
                self.send_header("Content-Type", history.content_type)
                self.send_header("Content-Length", str(len(history.data)))
                self.end_headers()
                self.wfile.write(history.data)
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
            submission, error = parse_submission(
                payload, items=items, note=note, measured_elapsed=clock() - started
            )
            if submission is None:
                self._responder_json(400, {"error": error})
                return
            outcome.submission = submission
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
