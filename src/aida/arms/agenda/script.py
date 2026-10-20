"""El JavaScript de la pantalla de la agenda (101b). Vive aparte: es código del navegador,
y lo que pesa en la lectura de Python no debe mezclarse con él. Va como cadena literal.
"""

# ruff: noqa: E501
AGENDA_JS = """
const W = CONFIG.w;
let DIA = CONFIG.dia0;
let DATOS = null;
let SELECCION = null;
let MODO = null;

function el(tag, clase, texto) {
  const nodo = document.createElement(tag);
  if (clase) { nodo.className = clase; }
  if (texto !== undefined && texto !== null) { nodo.textContent = texto; }
  return nodo;
}
function hora(iso) { return iso.slice(11, 16); }
function fecha(iso) { return iso.slice(0, 10); }
function sumarDias(dia, dias) {
  const d = new Date(dia + 'T12:00:00');
  d.setDate(d.getDate() + dias);
  return d.toISOString().slice(0, 10);
}
function minutos(hhmm) { return parseInt(hhmm.slice(0, 2), 10) * 60 + parseInt(hhmm.slice(3, 5), 10); }
function recursoNombre(id) {
  const r = DATOS.resources.find((x) => x.id === id);
  return r ? r.label : id;
}
function textoEstado(estado) {
  const nombres = { booked: W.state_booked, cancelled: W.state_cancelled,
    attended: W.state_attended, no_show: W.state_no_show };
  return nombres[estado] || estado;
}
function claseEstado(estado) {
  if (estado === 'attended') { return 'done'; }
  if (estado === 'no_show') { return 'noshow'; }
  if (estado === 'cancelled') { return 'cancel'; }
  return '';
}
function marcarSucio(sucio) {
  return fetch('/' + TOKEN + '/actividad', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dirty: sucio }),
  });
}
async function cargar(dia) {
  await marcarSucio(false);
  const r = await fetch('/' + TOKEN + '/dia/' + dia);
  DATOS = await r.json();
  DIA = dia;
  SELECCION = null;
  MODO = null;
  pintar();
}
async function actuar(accion, argumentos) {
  const r = await fetch('/' + TOKEN + '/accion', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: accion, arguments: argumentos }),
  });
  const respuesta = await r.json();
  if (respuesta.status === 'done') { await cargar(DIA); return; }
  mostrarAviso(respuesta.reason_text || W.reason_failed);
}
function mostrarAviso(texto) {
  document.getElementById('side').prepend(el('p', 'confirm', texto));
}

function pintarSemana() {
  const semana = document.getElementById('week');
  semana.replaceChildren();
  for (const dia of DATOS.week) {
    const boton = el('button', '');
    boton.type = 'button';
    if (dia.day === DIA) { boton.setAttribute('aria-current', 'date'); }
    boton.append(el('b', '', dia.day.slice(8, 10) + '/' + dia.day.slice(5, 7)));
    boton.append(el('span', '', dia.booked + ' / ' + dia.capacity));
    const barra = el('div', 'bar-mini');
    const relleno = el('i', '');
    relleno.style.width = (dia.capacity ? Math.min(100, Math.round(100 * dia.booked / dia.capacity)) : 0) + '%';
    barra.append(relleno);
    boton.append(barra);
    boton.addEventListener('click', () => cargar(dia.day));
    semana.append(boton);
  }
}

function pintarRejilla() {
  const grid = document.getElementById('grid');
  grid.replaceChildren();
  const recursos = DATOS.resources;
  grid.style.gridTemplateColumns = '64px repeat(' + recursos.length + ', 1fr)';
  grid.append(el('div', 'col-head', ''));
  for (const r of recursos) {
    const cabeza = el('div', 'col-head', r.label);
    cabeza.append(el('small', '', r.place));
    grid.append(cabeza);
  }
  let inicio = null;
  let fin = null;
  for (const r of recursos) {
    for (const [ta, tb] of r.hours) {
      inicio = inicio === null ? minutos(ta) : Math.min(inicio, minutos(ta));
      fin = fin === null ? minutos(tb) : Math.max(fin, minutos(tb));
    }
  }
  if (inicio === null) { return; }
  const slot = DATOS.slot_minutes;
  for (let t = inicio; t < fin; t += slot) {
    const hhmm = String(Math.floor(t / 60)).padStart(2, '0') + ':' + String(t % 60).padStart(2, '0');
    grid.append(el('div', 'hour', hhmm));
    for (const r of recursos) {
      const celda = el('div', 'cell');
      const abierto = r.hours.some(([ta, tb]) => minutos(ta) <= t && t < minutos(tb));
      if (!abierto) {
        celda.classList.add('closed');
      } else {
        const cita = DATOS.appointments.find((c) => c.resource === r.id && hora(c.start) === hhmm);
        if (cita) {
          const bloque = el('div', 'appt ' + claseEstado(cita.state));
          bloque.style.height = (cita.minutes / slot * 52 - 6) + 'px';
          if (SELECCION && SELECCION.id === cita.id) { bloque.classList.add('selected'); }
          bloque.append(el('b', '', cita.client));
          bloque.append(el('span', '', cita.service_label + ' · ' + cita.minutes + ' min'));
          bloque.addEventListener('click', () => { SELECCION = cita; MODO = null; pintar(); });
          celda.append(bloque);
        } else {
          celda.classList.add('free');
          celda.addEventListener('click', () => nuevaCita(r.id, hhmm));
        }
      }
      grid.append(celda);
    }
  }
}

function nuevaCita(recurso, hhmm) {
  MODO = { tipo: 'nueva', recurso: recurso || '', hora: hhmm || '08:00', servicio: null };
  pintarPanel();
}

function botonAccion(texto, accion, id) {
  const b = el('button', 'btn', texto);
  b.type = 'button';
  b.addEventListener('click', () => actuar(accion, { appointment: String(id) }));
  return b;
}
function botonModo(texto, modo) {
  const b = el('button', 'btn', texto);
  b.type = 'button';
  b.addEventListener('click', () => { MODO = { tipo: modo }; pintarPanel(); });
  return b;
}
function botonSimple(texto, alPulsar, clase) {
  const b = el('button', clase || 'btn', texto);
  b.type = 'button';
  b.addEventListener('click', alPulsar);
  return b;
}
function selectorServicio(valor) {
  const s = el('select', '');
  for (const sv of DATOS.services) {
    const o = el('option', '', sv.label);
    o.value = sv.id;
    if (sv.id === valor) { o.selected = true; }
    s.append(o);
  }
  return s;
}
function pintarHuecos(lado, servicio, alElegir) {
  const libres = DATOS.free[servicio] || [];
  if (!libres.length) { lado.append(el('p', 'lead', W.no_slots)); return; }
  const cont = el('div', 'slots');
  for (const h of libres) {
    const b = el('button', 'slot', hora(h.start) + ' · ' + recursoNombre(h.resource));
    b.type = 'button';
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => alElegir(h));
    cont.append(b);
  }
  lado.append(cont);
}

function formularioNueva(lado) {
  lado.append(el('h3', '', W.new_title));
  lado.append(el('p', 'lead', W.new_lead));
  const servicio = MODO.servicio || (DATOS.services[0] ? DATOS.services[0].id : '');
  const selector = selectorServicio(servicio);
  selector.addEventListener('change', () => { MODO.servicio = selector.value; pintarPanel(); });
  const nombre = el('input', 'input');
  nombre.placeholder = W.field_name;
  nombre.addEventListener('input', () => marcarSucio(true));
  const telefono = el('input', 'input');
  telefono.placeholder = W.field_phone;
  telefono.addEventListener('input', () => marcarSucio(true));
  lado.append(el('label', 'label', W.field_service), selector);
  lado.append(el('label', 'label', W.field_name), nombre);
  lado.append(el('label', 'label', W.field_phone), telefono);
  lado.append(el('label', 'label', W.free_slots.replace('{0}', selector.options[selector.selectedIndex].text)));
  pintarHuecos(lado, servicio, (h) => actuar('agendar', {
    service: servicio, resource: h.resource, start: h.start,
    name: nombre.value, phone: telefono.value,
  }));
  lado.append(botonSimple(W.book_button, () => actuar('agendar', {
    service: servicio, resource: MODO.recurso,
    start: DIA + 'T' + MODO.hora + ':00' + DATOS.offset,
    name: nombre.value, phone: telefono.value,
  }), 'btn btn-primary'));
}

function formularioEditar(lado) {
  const c = SELECCION;
  lado.append(el('h3', '', W.edit_button));
  const nombre = el('input', 'input');
  nombre.value = c.client;
  nombre.addEventListener('input', () => marcarSucio(true));
  const telefono = el('input', 'input');
  telefono.value = c.phone;
  telefono.addEventListener('input', () => marcarSucio(true));
  lado.append(el('label', 'label', W.field_name), nombre);
  lado.append(el('label', 'label', W.field_phone), telefono);
  lado.append(botonSimple(W.save_button, () => actuar('editar', {
    appointment: String(c.id), name: nombre.value, phone: telefono.value,
  }), 'btn btn-primary'));
  lado.append(botonSimple(W.close_button, () => { MODO = null; pintarPanel(); }));
}

function formularioMover(lado) {
  const c = SELECCION;
  lado.append(el('h3', '', W.move_title));
  pintarHuecos(lado, c.service, (h) => actuar('reprogramar', {
    appointment: String(c.id), start: h.start, resource: h.resource,
  }));
  lado.append(botonSimple(W.close_button, () => { MODO = null; pintarPanel(); }));
}

function confirmarCancelar(lado) {
  const c = SELECCION;
  const texto = W.cancel_confirm.replace('{0}', c.client)
    .replace('{1}', fecha(c.start)).replace('{2}', hora(c.start));
  const caja = el('div', 'confirm', texto);
  const fila = el('div', 'row');
  fila.append(botonSimple(W.cancel_back, () => { MODO = null; pintarPanel(); }));
  fila.append(botonSimple(W.cancel_yes, () => actuar('cancelar', { appointment: String(c.id) }), 'btn btn-danger'));
  caja.append(fila);
  lado.append(caja);
}

function pintarDetalle(lado) {
  const c = SELECCION;
  lado.append(el('h3', '', c.client));
  const datos = el('dl', 'kv');
  for (const [etiqueta, valor] of [
    [W.detail_service, c.service_label],
    [W.detail_resource, recursoNombre(c.resource)],
    [W.detail_phone, c.phone],
    [W.detail_state, textoEstado(c.state)],
  ]) {
    datos.append(el('dt', '', etiqueta), el('dd', '', valor));
  }
  lado.append(datos);
  lado.append(el('p', 'event-line', W.created_line.replace('{0}', c.created_by).replace('{1}', c.updated_by)));
  const acciones = el('div', 'side-actions');
  if (c.state === 'booked') {
    const fila = el('div', 'row');
    fila.append(botonAccion(W.attended_button, 'atendida', c.id));
    fila.append(botonAccion(W.no_show_button, 'no_asistio', c.id));
    acciones.append(fila);
    const fila2 = el('div', 'row');
    fila2.append(botonModo(W.edit_button, 'editar'));
    fila2.append(botonModo(W.move_button, 'mover'));
    acciones.append(fila2);
    acciones.append(botonSimple(W.cancel_button, () => { MODO = { tipo: 'cancelar' }; pintarPanel(); }, 'btn btn-danger'));
  }
  lado.append(acciones);
}

function pintarPanel() {
  const lado = document.getElementById('side');
  lado.replaceChildren();
  if (MODO && MODO.tipo === 'nueva') { return formularioNueva(lado); }
  if (!SELECCION) { lado.append(el('h3', '', W.title)); lado.append(el('p', 'lead', W.new_lead)); return; }
  if (MODO && MODO.tipo === 'editar') { return formularioEditar(lado); }
  if (MODO && MODO.tipo === 'mover') { return formularioMover(lado); }
  if (MODO && MODO.tipo === 'cancelar') { return confirmarCancelar(lado); }
  return pintarDetalle(lado);
}

function pintar() {
  document.getElementById('day-title').textContent = DIA;
  pintarSemana();
  pintarRejilla();
  pintarPanel();
}

document.getElementById('prev').addEventListener('click', () => cargar(sumarDias(DIA, -1)));
document.getElementById('next').addEventListener('click', () => cargar(sumarDias(DIA, 1)));
document.getElementById('today').addEventListener('click', () => cargar(CONFIG.dia0));
document.getElementById('new-btn').addEventListener('click', () => nuevaCita('', ''));
cargar(DIA);
"""
