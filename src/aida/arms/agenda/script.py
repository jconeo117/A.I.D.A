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
async function actuar(accion, argumentos, despues) {
  const r = await fetch('/' + TOKEN + '/accion', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: accion, arguments: argumentos }),
  });
  const respuesta = await r.json();
  if (respuesta.status === 'done') {
    await cargar(DIA);
    if (despues) { await despues(); }
    return;
  }
  mostrarAviso(respuesta.reason_text || W.reason_failed, respuesta.conflicts || []);
}
function mostrarAviso(texto, conflictos) {
  const aviso = el('div', 'confirm', texto);
  if (conflictos && conflictos.length) {
    aviso.append(el('p', '', W.conflicts_title));
    const lista = el('ul', 'list');
    for (const c of conflictos) {
      lista.append(el('li', '', fecha(c.start) + ' ' + hora(c.start) + ' · ' + c.client));
    }
    aviso.append(lista);
  }
  document.getElementById('side').prepend(aviso);
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

function bloqueoEn(recursoId, hhmm) {
  const desde = new Date(DIA + 'T' + hhmm + ':00' + DATOS.offset).getTime();
  const hasta = desde + DATOS.slot_minutes * 60000;
  return DATOS.blocks.find((b) => (b.resource === '' || b.resource === recursoId)
    && new Date(b.start).getTime() < hasta && desde < new Date(b.end).getTime()) || null;
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
  const vistos = new Set();
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
        const bloqueo = cita ? null : bloqueoEn(r.id, hhmm);
        if (cita) {
          const bloque = el('div', 'appt ' + claseEstado(cita.state));
          bloque.style.height = (cita.minutes / slot * 52 - 6) + 'px';
          if (SELECCION && SELECCION.id === cita.id) { bloque.classList.add('selected'); }
          bloque.append(el('b', '', cita.client));
          bloque.append(el('span', '', cita.service_label + ' · ' + cita.minutes + ' min'));
          bloque.addEventListener('click', () => { SELECCION = cita; MODO = null; pintar(); });
          celda.append(bloque);
        } else if (bloqueo) {
          celda.classList.add('blocked');
          const clave = r.id + ':' + bloqueo.id;
          if (!vistos.has(clave)) {
            vistos.add(clave);
            celda.append(el('span', 'blocked-label', bloqueo.reason || W.blocked_label));
          }
          celda.addEventListener('click', () => { SELECCION = null; MODO = { tipo: 'bloqueo', bloqueo: bloqueo }; pintarPanel(); });
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
  nombre.value = MODO.nombre || '';
  nombre.addEventListener('input', () => marcarSucio(true));
  const telefono = el('input', 'input');
  telefono.placeholder = W.field_phone;
  telefono.value = MODO.telefono || '';
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
  lado.append(botonSimple(W.card_link, () => abrirFicha(c.client_id), 'btn btn-quiet'));
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

function formularioBloquear(lado) {
  lado.append(el('h3', '', W.block_title));
  lado.append(el('p', 'lead', W.block_lead));
  const quien = el('select', '');
  const todos = el('option', '', W.all_resources);
  todos.value = '';
  quien.append(todos);
  for (const r of DATOS.resources) {
    const o = el('option', '', r.label);
    o.value = r.id;
    quien.append(o);
  }
  const desde = el('input', 'input');
  desde.type = 'datetime-local';
  desde.value = DIA + 'T08:00';
  const hasta = el('input', 'input');
  hasta.type = 'datetime-local';
  hasta.value = DIA + 'T09:00';
  const motivo = el('input', 'input');
  for (const campo of [desde, hasta, motivo]) { campo.addEventListener('input', () => marcarSucio(true)); }
  lado.append(el('label', 'label', W.field_who), quien);
  lado.append(el('label', 'label', W.field_from), desde);
  lado.append(el('label', 'label', W.field_until), hasta);
  lado.append(el('label', 'label', W.field_reason), motivo);
  lado.append(botonSimple(W.block_save, () => actuar('bloquear', {
    resource: quien.value,
    start: desde.value + ':00' + DATOS.offset,
    end: hasta.value + ':00' + DATOS.offset,
    reason: motivo.value,
  }), 'btn btn-primary'));
  lado.append(botonSimple(W.close_button, () => { MODO = null; pintarPanel(); }));
}

function pintarBloqueo(lado) {
  const b = MODO.bloqueo;
  lado.append(el('h3', '', b.reason || W.blocked_label));
  const datos = el('dl', 'kv');
  datos.append(el('dt', '', W.field_who), el('dd', '', b.resource ? recursoNombre(b.resource) : W.all_resources));
  datos.append(el('dt', '', W.field_from), el('dd', '', fecha(b.start) + ' ' + hora(b.start)));
  datos.append(el('dt', '', W.field_until), el('dd', '', fecha(b.end) + ' ' + hora(b.end)));
  lado.append(datos);
  lado.append(el('p', 'event-line', W.blocked_by.replace('{0}', b.created_by)));
  lado.append(botonSimple(W.unblock_button, () => actuar('desbloquear', { block: String(b.id) }), 'btn btn-danger'));
  lado.append(botonSimple(W.close_button, () => { MODO = null; pintarPanel(); }));
}

async function buscar(texto) {
  const r = await fetch('/' + TOKEN + '/buscar', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: texto }),
  });
  const respuesta = await r.json();
  SELECCION = null;
  MODO = { tipo: 'buscar', contactos: respuesta.clients || [] };
  pintarPanel();
}

function pintarBusqueda(lado) {
  if (!MODO.contactos.length) { lado.append(el('p', 'lead', W.search_empty)); return; }
  const lista = el('ul', 'list');
  for (const c of MODO.contactos) {
    const fila = el('li', '', c.name + ' · ' + c.phone);
    fila.addEventListener('click', () => abrirFicha(c.id));
    lista.append(fila);
  }
  lado.append(lista);
}

async function abrirFicha(id) {
  const r = await fetch('/' + TOKEN + '/contacto/' + id);
  const ficha = await r.json();
  SELECCION = null;
  MODO = { tipo: 'ficha', ficha: ficha };
  pintarPanel();
}

function pintarFicha(lado) {
  const f = MODO.ficha;
  lado.append(el('h3', '', f.name));
  const datos = el('dl', 'kv');
  datos.append(el('dt', '', W.field_phone), el('dd', '', f.phone));
  lado.append(datos);
  const cifras = el('div', 'stats');
  for (const [n, etiqueta] of [[f.upcoming, W.card_upcoming], [f.attended, W.card_attended], [f.no_show, W.card_no_show]]) {
    const caja = el('div', '');
    caja.append(el('b', '', String(n)), el('span', '', etiqueta));
    cifras.append(caja);
  }
  lado.append(cifras);
  lado.append(el('label', 'label', W.card_appointments));
  const lista = el('ul', 'list');
  for (const c of f.appointments) {
    const fila = el('li', '', fecha(c.start) + ' ' + hora(c.start) + ' · ' + c.service_label
      + ' · ' + recursoNombre(c.resource) + ' · ' + textoEstado(c.state));
    fila.addEventListener('click', () => cargar(fecha(c.start)));
    lista.append(fila);
  }
  lado.append(lista);
  if (CONFIG.canal && f.conversation_phone) {
    lado.append(botonSimple(W.convo_link, () => { mostrarPestana('convo'); abrirConversacion(f.conversation_phone, true); }, 'btn btn-quiet'));
  }
  const acciones = el('div', 'side-actions');
  acciones.append(botonSimple(W.card_edit, () => { MODO = { tipo: 'contacto', ficha: f }; pintarPanel(); }));
  acciones.append(botonSimple(W.new_button, () => {
    MODO = { tipo: 'nueva', recurso: '', hora: '08:00', servicio: null, nombre: f.name, telefono: f.phone };
    pintarPanel();
  }, 'btn btn-primary'));
  lado.append(acciones);
}

function formularioContacto(lado) {
  const f = MODO.ficha;
  lado.append(el('h3', '', W.card_edit));
  const nombre = el('input', 'input');
  nombre.value = f.name;
  const telefono = el('input', 'input');
  telefono.value = f.phone;
  for (const campo of [nombre, telefono]) { campo.addEventListener('input', () => marcarSucio(true)); }
  lado.append(el('label', 'label', W.field_name), nombre);
  lado.append(el('label', 'label', W.field_phone), telefono);
  lado.append(botonSimple(W.save_button, () => actuar('editar_contacto', {
    client: String(f.id), name: nombre.value, phone: telefono.value,
  }, () => abrirFicha(f.id)), 'btn btn-primary'));
  lado.append(botonSimple(W.close_button, () => { MODO = { tipo: 'ficha', ficha: f }; pintarPanel(); }));
}

function pintarPanel() {
  const lado = document.getElementById('side');
  lado.replaceChildren();
  if (MODO && MODO.tipo === 'nueva') { return formularioNueva(lado); }
  if (MODO && MODO.tipo === 'bloquear') { return formularioBloquear(lado); }
  if (MODO && MODO.tipo === 'bloqueo') { return pintarBloqueo(lado); }
  if (MODO && MODO.tipo === 'buscar') { return pintarBusqueda(lado); }
  if (MODO && MODO.tipo === 'ficha') { return pintarFicha(lado); }
  if (MODO && MODO.tipo === 'contacto') { return formularioContacto(lado); }
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
document.getElementById('block-btn').addEventListener('click', () => { SELECCION = null; MODO = { tipo: 'bloquear' }; pintarPanel(); });
document.getElementById('search').addEventListener('change', (ev) => buscar(ev.target.value));
cargar(DIA);
"""

CONVERSATIONS_JS = """
const CONVO = { lista: null, actual: null, borrador: '', leido: {} };

function mostrarPestana(cual) {
  document.getElementById('pane-agenda').hidden = cual !== 'agenda';
  document.getElementById('pane-convo').hidden = cual !== 'convo';
  document.getElementById('tab-agenda').setAttribute('aria-selected', String(cual === 'agenda'));
  document.getElementById('tab-convo').setAttribute('aria-selected', String(cual === 'convo'));
}
function convoVisible() { return !document.getElementById('pane-convo').hidden; }
async function cargarConversaciones(conPanel) {
  const r = await fetch('/' + TOKEN + '/conversaciones', { headers: { 'X-AIDA-Background': '1' } });
  if (!r.ok) { return; }
  CONVO.lista = await r.json();
  pintarConversaciones();
  if (conPanel) { pintarCanal(); }
  if (CONVO.actual && convoVisible()) { await abrirConversacion(CONVO.actual.phone, true, true); }
}
function textoUltimo(c) { return c.last_kind === 'text' ? c.last_text : '[' + c.last_kind + ']'; }
function pintarConversaciones() {
  const total = CONVO.lista.conversations.reduce((s, c) => s + c.unread, 0);
  const marca = document.getElementById('unread-total');
  marca.hidden = total === 0;
  marca.textContent = String(total);
  const hilos = document.getElementById('threads');
  hilos.replaceChildren();
  if (!CONVO.lista.conversations.length) { hilos.append(el('p', 'lead', W.convo_empty)); return; }
  for (const c of CONVO.lista.conversations) {
    const b = el('button', 'thread');
    b.type = 'button';
    if (CONVO.actual && CONVO.actual.phone === c.phone) { b.setAttribute('aria-current', 'true'); }
    if (c.unread) { b.append(el('span', 'count', String(c.unread))); }
    b.append(el('b', '', c.name || ('+' + c.phone)));
    b.append(el('small', '', (c.last_direction === 'out' ? W.convo_you : '') + textoUltimo(c)));
    b.addEventListener('click', () => abrirConversacion(c.phone, true, false));
    hilos.append(b);
  }
}
async function abrirConversacion(telefono, marcar, deFondo) {
  const cabeceras = deFondo ? { 'X-AIDA-Background': '1' } : {};
  const r = await fetch('/' + TOKEN + '/conversacion/' + telefono, { headers: cabeceras });
  if (!r.ok) { return; }
  if (!CONVO.actual || CONVO.actual.phone !== telefono) { CONVO.borrador = ''; }
  CONVO.actual = await r.json();
  pintarConversacion();
  const entradas = CONVO.actual.messages.filter((m) => m.direction === 'in');
  const ultima = entradas.length ? entradas[entradas.length - 1].id : 0;
  if (marcar && ultima > (CONVO.leido[telefono] || 0)) {
    CONVO.leido[telefono] = ultima;
    await actuarCanal('marcar_leida', { phone: telefono, up_to: String(ultima) });
  }
}
async function actuarCanal(accion, argumentos) {
  const r = await fetch('/' + TOKEN + '/accion', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: accion, arguments: argumentos }),
  });
  const respuesta = await r.json();
  if (respuesta.status === 'done') { await cargarConversaciones(false); return true; }
  const aviso = document.getElementById('convo-aviso');
  if (aviso) { aviso.textContent = respuesta.reason_text || W.reason_failed; aviso.hidden = false; }
  return false;
}
function pintarConversacion() {
  const c = CONVO.actual;
  const caja = document.getElementById('convo');
  caja.replaceChildren();
  const cabeza = el('div', 'convo-head');
  const quien = el('div', 'who');
  quien.append(el('b', '', c.name || ('+' + c.phone)), el('small', '', '+' + c.phone));
  cabeza.append(quien);
  if (c.client_id !== null) {
    cabeza.append(botonSimple(W.card_link, () => { mostrarPestana('agenda'); abrirFicha(c.client_id); }, 'btn btn-quiet'));
  }
  caja.append(cabeza);
  const lista = el('div', 'msgs');
  for (const m of c.messages) {
    const burbuja = el('div', 'msg ' + m.direction, m.kind === 'text' ? m.text : '[' + m.kind + ']');
    burbuja.append(el('small', '', (m.direction === 'out' ? m.by + ' · ' : '') + fecha(m.at) + ' ' + hora(m.at)));
    lista.append(burbuja);
  }
  caja.append(lista);
  const aviso = el('p', 'confirm');
  aviso.id = 'convo-aviso';
  aviso.hidden = true;
  caja.append(aviso);
  if (!c.window_open) { caja.append(el('p', 'lead', W.window_closed_hint)); return; }
  const escribir = el('div', 'compose');
  const texto = el('input', 'input');
  texto.placeholder = W.compose_placeholder;
  texto.value = CONVO.borrador;
  texto.addEventListener('input', () => { CONVO.borrador = texto.value; marcarSucio(texto.value !== ''); });
  const enviar = botonSimple(W.send_button, async () => {
    if (await actuarCanal('enviar', { phone: c.phone, text: texto.value })) {
      CONVO.borrador = '';
      marcarSucio(false);
      await abrirConversacion(c.phone, false, false);
    }
  }, 'btn btn-primary');
  escribir.append(texto, enviar);
  caja.append(escribir);
}
function pintarCanal() {
  const p = document.getElementById('channel-panel');
  const e = CONVO.lista;
  p.replaceChildren(el('h3', '', W.channel_title));
  let estado = W.channel_running;
  if (e.missing.length) {
    estado = W.channel_missing + ' ' + e.missing.map((x) => W['channel_missing_' + x] || x).join(', ');
  } else if (!e.running) {
    estado = W.channel_restart;
  } else if (e.poll_error) {
    estado = W.channel_error.replace('{0}', e.poll_error);
  }
  p.append(el('p', 'lead', estado));
  p.append(el('label', 'label', W.channel_public_key));
  const clave = el('input', 'input');
  clave.readOnly = true;
  clave.value = e.public_key;
  p.append(clave);
  p.append(el('label', 'label', W.channel_token));
  const token = el('input', 'input');
  token.type = 'password';
  p.append(token);
  const resultado = el('p', 'event-line');
  p.append(botonSimple(W.channel_token_save, async () => {
    const r = await fetch('/' + TOKEN + '/canal-token', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: token.value }),
    });
    const d = await r.json();
    resultado.textContent = d.ok ? W.channel_token_saved : W.channel_token_bad.replace('{0}', d.reason || '');
  }, 'btn'));
  p.append(resultado);
}

document.getElementById('tab-agenda').addEventListener('click', () => mostrarPestana('agenda'));
document.getElementById('tab-convo').addEventListener('click', () => { mostrarPestana('convo'); cargarConversaciones(true); });
cargarConversaciones(true);
setInterval(() => cargarConversaciones(false), 15000);
"""
