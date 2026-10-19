"""Los estilos y los iconos que pintan las pantallas del armazón (101a).

Lo de la plataforma va aquí: la base visual, los ajustes comunes, quién atiende y la
licencia. Lo de un brazo (la grabación, el consentimiento, la consulta) se queda en el
brazo y entra por `extra_css` y `extra_icons`.
"""
CSS_BASE = """
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
ICONS_BASE = """<svg style="display:none"><defs>
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
CSS_SETTINGS = """/* La 78: ajustes y primer arranque */
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
.status.fail { background: var(--off-soft); color: var(--off); }"""
CSS_IDENTITY = """/* La 83: quién atiende */
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
CSS_LICENSE = """/* La 100: la licencia */
.head-warn {
  display: inline-flex; align-items: center; gap: .4rem; padding: .35rem .75rem;
  border-radius: 999px;
  font-size: .78rem; font-weight: 600; color: #fde68a; background: rgba(180, 83, 9, .28);
  border: 1px solid rgba(253, 230, 138, .25);
}
.head-warn[hidden] { display: none; }
.license-row {
  display: grid; grid-template-columns: auto 1fr; gap: .3rem .9rem; margin-top: .7rem;
  font-size: .86rem;
}
.license-row dt { color: var(--muted); }
.license-row dd { margin: 0; }
.readonly-list {
  margin: .9rem 0 0; padding: 0; list-style: none; display: grid; gap: .45rem; font-size: .88rem;
}
.readonly-list li { display: flex; gap: .5rem; align-items: center; color: var(--ink-2); }
.readonly-list svg { width: 16px; height: 16px; color: var(--brand); }"""
ICONS_SHELL = """  <symbol id="i-user" viewBox="0 0 24 24">
    <circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>
  </symbol>
  <symbol id="i-lock" viewBox="0 0 24 24">
    <rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>
  </symbol>
  <symbol id="i-plus" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></symbol>
  <symbol id="i-badge" viewBox="0 0 24 24">
    <path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5Z"/><path d="M9 12h6M12 9v6"/>
  </symbol>
  <symbol id="i-alert" viewBox="0 0 24 24">
    <path d="M12 9v4M12 17h.01"/>
    <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>
  </symbol>
  <symbol id="i-play" viewBox="0 0 24 24"><path d="M6 4l14 8-14 8Z"/></symbol>
  <symbol id="i-exit" viewBox="0 0 24 24">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>
  </symbol>
  <symbol id="i-back" viewBox="0 0 24 24"><path d="M15 18l-6-6 6-6"/></symbol>"""


def icon_sheet(*symbols: str) -> str:
    """El SVG de símbolos del armazón con los del brazo añadidos al final. Un solo sitio para
    componer la hoja: el brazo no monta sus iconos a mano."""
    return ICONS_BASE.replace("</defs></svg>", "".join(s + "\n" for s in symbols) + "</defs></svg>")
