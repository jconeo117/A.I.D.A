"""Las pantallas del armazón: quién atiende, el primer perfil, el bloqueo, la licencia y el
marco de Ajustes (101a). Nada de aquí sabe de consultas: lo que es de un brazo entra por sus
estilos, sus iconos y sus tarjetas.

El texto de cada pantalla sale de `packs/platform/ui.yaml` (`Textos`): la plataforma no
lee textos de salud.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from aida.platform.identity import Professional
from aida.platform.license import LicenseState
from aida.platform.web.assets import (
    CSS_BASE,
    CSS_IDENTITY,
    CSS_LICENSE,
    CSS_SETTINGS,
    ICONS_SHELL,
    icon_sheet,
)
from aida.platform.web.wording import Textos
from aida.resources import user_dir


def user_chip(w: Textos, professional: Professional | None) -> str:
    """Quién atiende, en la cabecera; pulsarlo cambia de profesional (plan 83)."""
    if professional is None:
        return ""
    e = html.escape
    return (
        f'\n    <button type="button" class="head-user" id="head-user" '
        f'title="{e(w["change_professional"])}"><svg><use href="#i-user"/></svg>'
        f"{e(professional.name)}</button>"
    )


def identity_script(w: Textos) -> str:
    return f"const IDLE_MS = {int(w.idle_lock_minutes * 60_000)};\n{_JS_IDENTITY_HEAD}"


def _initials(name: str) -> str:
    """Las iniciales del avatar, sin títulos: «Dra. Laura Gómez» es LG, no DL."""
    palabras = [p for p in name.split() if not p.endswith(".")]
    return "".join(p[0] for p in palabras[:2]).upper() or name[:1].upper()


def _person(p: Professional, w: Textos, *, button: bool) -> str:
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


def profiles_block(w: Textos, profiles: Sequence[Professional]) -> str:
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


def who_page(
    w: Textos,
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
    return page_frame(
        brand=brand, title=w["title_who"], place=w["header_place"], moment=moment,
        pill=w["pill_ready"], pill_class="pill", body=body, script=script,
    )


def first_profile_page(w: Textos, brand: str, token: str, moment: datetime) -> str:
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
    return page_frame(
        brand=brand, title=w["title_first_run"], place=w["header_place"], moment=moment,
        pill=w["pill_ready"], pill_class="pill", body=body, script=script,
    )


def locked_page(
    w: Textos, brand: str, token: str, professional: Professional, moment: datetime
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
    return page_frame(
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


def license_file() -> Path:
    return user_dir("config") / "licencia.json"


def license_chip(w: Textos, license: LicenseState | None) -> str:
    """El aviso de caducidad en la cabecera, desde `WARN_DAYS` antes (plan 100)."""
    if license is None or license.status != "expiring":
        return ""
    e = html.escape
    return (
        f'\n    <span class="head-warn"><svg style="width:14px;height:14px">'
        f'<use href="#i-alert"/></svg>{e(w["license_warn"].format(license.days_left))}</span>'
    )


def _license_status(w: Textos, license: LicenseState) -> tuple[str, str]:
    if license.status == "valid":
        return "ok", w["license_status_valid"]
    if license.status == "expiring":
        return "warn", w["license_status_expiring"].format(license.days_left)
    return "fail", w[f"license_status_{license.status}"]


def license_block(w: Textos, license: LicenseState | None) -> str:
    """La tarjeta «Licencia» de Ajustes: quién, qué brazos y hasta cuándo, y cargar
    otra."""
    if license is None:
        return ""
    e = html.escape
    clase, estado = _license_status(w, license)
    filas = ""
    if license.license is not None:
        lic = license.license
        filas = (
            f'<dl class="license-row"><dt>{e(w["license_client"])}</dt><dd>{e(lic.client)}</dd>'
            f'<dt>{e(w["license_arms"])}</dt><dd>{e(" · ".join(lic.arms))}</dd>'
            f'<dt>{e(w["license_until"])}</dt><dd>{lic.expires.strftime("%d/%m/%Y")}</dd></dl>'
        )
    return f"""<div class="setting">
          <div class="setting-head"><svg><use href="#i-badge"/></svg>
            <h3>{e(w["license_title"])}</h3>
            <span class="status {clase}">{e(estado)}</span></div>
          {filas}
          <div class="key-row"><label class="btn">{e(w["license_load"])}
            <input type="file" id="license-file" accept=".json,application/json" hidden>
          </label></div>
          <div class="result fail" id="license-error" hidden><svg><use href="#i-alert"/></svg>
            <span></span></div>
        </div>"""


def license_page(
    w: Textos,
    brand: str,
    token: str,
    license: LicenseState,
    moment: datetime,
    professional: Professional | None,
) -> str:
    """Sin una licencia válida no se empieza nada: se carga aquí."""
    e = html.escape
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card">
      <div class="alert-head">
        <span class="alert-icon warn"><svg><use href="#i-badge"/></svg></span>
        <div><h2>{e(w["license_missing_title"])}</h2>
        <p class="lead" style="margin:.2rem 0 0">{e(w["license_missing_lead"])}</p></div>
      </div>
      {license_block(w, license)}
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="exit">
      <svg><use href="#i-exit"/></svg>{e(w["exit_label"])}</button>
  </footer>
</section>"""
    return page_frame(
        brand=brand, title=w["license_title"], place=w["header_place"], moment=moment,
        pill=w["pill_problem"], pill_class="pill fallback", body=body,
        script=f"const TOKEN = {json.dumps(token)};\n{JS_LICENSE}{JS_EXIT}",
        head_link=user_chip(w, professional),
    )


def readonly_page(
    w: Textos,
    brand: str,
    token: str,
    license: LicenseState,
    moment: datetime,
    professional: Professional | None,
    arm_name: str,
    textos: tuple[str, ...],
) -> str:
    """Caducada: nada nuevo, y nada de lo registrado se pierde (debate de la 100)."""
    e = html.escape
    caducidad = license.license.expires.strftime("%d/%m/%Y") if license.license else ""
    filas = "".join(
        f'<li><svg><use href="#i-check"/></svg>{e(texto)}</li>' for texto in textos
    )
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card">
      <div class="alert-head">
        <span class="alert-icon warn"><svg><use href="#i-badge"/></svg></span>
        <div><h2>{e(w["readonly_title"].format(arm_name, caducidad))}</h2>
        <p class="lead" style="margin:.2rem 0 0">{e(w["readonly_lead"])}</p></div>
      </div>
      <ul class="readonly-list">{filas}</ul>
      <p class="hint">{e(w["readonly_hint"])}</p>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="exit">
      <svg><use href="#i-exit"/></svg>{e(w["exit_label"])}</button>
    <button type="button" class="btn btn-primary" id="to-settings">
      <svg><use href="#i-gear"/></svg>{e(w["go_settings"])}</button>
  </footer>
</section>"""
    return page_frame(
        brand=brand, title=arm_name, place=w["header_place"], moment=moment,
        pill=w["pill_readonly"], pill_class="pill fallback", body=body,
        script=f"const TOKEN = {json.dumps(token)};\n{JS_TO_SETTINGS}{JS_EXIT}",
        head_link=user_chip(w, professional),
    )


JS_EXIT = """
document.getElementById('exit').addEventListener('click', async () => {
  await fetch('/' + TOKEN + '/salir', { method: 'POST', body: '' });
  document.body.textContent = '';
});
"""


JS_LICENSE = """
// Cargar la licencia (plan 100): el archivo va tal cual; quien lo valida es AIDA.
const licenseFile = document.getElementById('license-file');
if (licenseFile) {
  licenseFile.addEventListener('change', async () => {
    const file = licenseFile.files[0];
    if (!file) { return; }
    const r = await fetch('/' + TOKEN + '/licencia', { method: 'POST', body: await file.text() });
    if (r.ok) { location.reload(); return; }
    const box = document.getElementById('license-error');
    box.querySelector('span').textContent = (await r.json()).error;
    box.hidden = false;
  });
}
"""


JS_TO_SETTINGS = """
document.getElementById('to-settings').addEventListener('click', async () => {
  await fetch('/' + TOKEN + '/ajustes', { method: 'POST', body: '' });
  location.reload();
});
"""


def page_frame(
    *,
    brand: str,
    title: str,
    place: str,
    moment: datetime,
    pill: str,
    pill_class: str,
    body: str,
    script: str,
    head_link: str = "",
    extra_css: str = "",
    extra_icons: str = "",
) -> str:
    """La página entera con la cabecera común. Los estilos y los iconos del brazo van
    detrás de los de la plataforma: una pantalla del armazón no depende de ningún brazo,
    y un brazo añade lo suyo sin tocar la plataforma (101a)."""
    e = html.escape
    estilo = "\n".join(
        parte
        for parte in (CSS_BASE, extra_css, CSS_SETTINGS, CSS_IDENTITY, CSS_LICENSE)
        if parte
    )
    iconos = icon_sheet(ICONS_SHELL, extra_icons)
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


def settings_frame(
    w: Textos,
    brand: str,
    token: str,
    moment: datetime,
    *,
    cards: str,
    script: str,
    professional: Professional | None,
    license: LicenseState | None,
    extra_css: str = "",
    extra_icons: str = "",
) -> str:
    """El marco de Ajustes: cabecera, «Volver» y el aviso de licencia. Las tarjetas las
    compone el brazo, porque el orden entre ellas es suyo; los bloques de perfiles y de
    licencia se los pide con `profiles_block` y `license_block` (101a, decisión 3)."""
    e = html.escape
    body = f"""<section class="screen">
  <div class="stage">
    <div class="panel-card">
      <h2>{e(w["title_settings"])}</h2>
      <p class="lead">{e(w["settings_lead"])}</p>
      <div class="settings">
{cards}
      </div>
    </div>
  </div>
  <footer class="bar">
    <button type="button" class="btn btn-quiet" id="back">
      <svg><use href="#i-back"/></svg>{e(w["back_label"])}</button>
  </footer>
</section>"""
    return page_frame(
        brand=brand, title=w["title_settings"], place=w["header_place"], moment=moment,
        pill=w["pill_ready"], pill_class="pill", body=body,
        script=script + identity_script(w) + JS_LICENSE,
        head_link=license_chip(w, license) + user_chip(w, professional),
        extra_css=extra_css, extra_icons=extra_icons,
    )
