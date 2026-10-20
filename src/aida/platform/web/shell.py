"""El armazón de la plataforma: servidor local con token, quién atiende, el bloqueo por
inactividad, la licencia, Ajustes y el lanzador. Los brazos se montan en él como `ArmApp` (101a,
101b).

Por qué vive aquí y no en el brazo: quién está identificado, cuándo se bloquea, qué licencia
vale y qué brazo está abierto son decisiones de la plataforma. Si cada brazo las tomara por su
lado, dos puertas de PIN acabarían divergiendo, y un registro podría escribirse sin autor en una
de ellas.
"""

from __future__ import annotations

import contextlib
import html
import json
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
from typing import Any, Protocol

from aida.platform.audit import AuditLog
from aida.platform.identity import IdentityError, Professional, ProfileStore
from aida.platform.license import LicenseState, read_license
from aida.platform.web.pages import (
    first_profile_page,
    launcher_page,
    license_file,
    locked_page,
    who_page,
)
from aida.platform.web.wording import Textos

_MAX_BYTES = 512 * 1024 * 1024
_IDENTIDAD = ("perfil", "entrar", "desbloquear", "cambiar", "salir", "recuperar-pin")


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    body: bytes
    content_type: str


def json_response(status: int, body: dict[str, Any]) -> Response:
    datos = json.dumps(body, ensure_ascii=False).encode("utf-8")
    return Response(status, datos, "application/json; charset=utf-8")


@dataclass(frozen=True, slots=True)
class ShellContext:
    """Lo que el armazón presta al brazo en cada petición. La construye con el candado
    tomado; el brazo toma `lock` cuando lo necesita, como hacía antes de la mudanza."""

    token: str
    brand: str
    now: datetime
    professional: Professional | None
    license: LicenseState
    audit: AuditLog
    wording: Textos
    lock: threading.Lock
    profiles: tuple[Professional, ...]
    launcher_link: str = ""
    """El botón «Inicio» de la cabecera, con varios brazos; vacío con uno (101b)."""


class ArmApp(Protocol):
    """Lo que un brazo entrega al armazón. `path` llega sin el prefijo `/{token}/`; `get` y
    `post` devuelven `None` si la ruta no es del brazo (el armazón responde 404)."""

    id: str
    name: str

    def page(self, ctx: ShellContext) -> str: ...
    def get(self, path: str, ctx: ShellContext) -> Response | None: ...
    def post(
        self, path: str, body: bytes | None, headers: Mapping[str, str], ctx: ShellContext
    ) -> Response | None: ...
    def at_rest(self) -> bool: ...
    def in_settings(self) -> bool: ...
    def reset(self) -> None: ...
    def rebuild(self) -> None: ...
    def on_identified(self, actor: Professional) -> None: ...
    def on_unlocked(self) -> None: ...
    def settings_cards(self, ctx: ShellContext) -> str: ...
    def readonly_items(self, ctx: ShellContext) -> tuple[str, ...]: ...


@dataclass
class _Sesion:
    """Quién atiende, desde cuándo y en qué brazo está. El estado del trabajo (una consulta, una
    cita) es del brazo: aquí solo se sabe si hay alguien identificado y dónde está."""

    state: str
    professional: Professional | None = None
    last_activity: float = 0.0
    current: str | None = None
    """El `id` del brazo abierto. Con un solo brazo, siempre el suyo; con varios, `None` en el
    lanzador."""


def _con_enlace_lanzador(pagina_html: str, enlace: str) -> str:
    """El botón «Inicio» en la cabecera de la página de un brazo. Una sola inserción, en la marca
    que pone toda página del marco: así ninguna pantalla de brazo necesita saber si hay lanzador."""
    marca = '<div class="head-right">'
    return pagina_html.replace(marca, marca + enlace, 1) if enlace else pagina_html


def run_shell(
    arms: Sequence[ArmApp],
    *,
    brand: str,
    wording: Textos,
    profiles: ProfileStore,
    audit: AuditLog,
    license: LicenseState,
    license_path: Path | None,
    lock: threading.Lock,
    presentacion: Mapping[str, tuple[str, str]] | None = None,
    open_browser: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = datetime.now,
    today: Callable[[], date] = date.today,
) -> None:
    """Levanta el armazón y bloquea hasta que alguien pulsa «Salir» (o Ctrl-C). Con un solo brazo
    entra directamente en él; con varios, primero el lanzador (101b). `presentacion` da, por id de
    brazo, su descripción y su símbolo: lo que dice su manifiesto."""
    presentacion = presentacion or {}
    if not arms:
        raise ValueError("el armazón necesita al menos un brazo")
    por_id: dict[str, ArmApp] = {}
    for brazo_nuevo in arms:
        if brazo_nuevo.id in por_id:
            raise ValueError(f"dos brazos con el mismo id {brazo_nuevo.id!r}")
        por_id[brazo_nuevo.id] = brazo_nuevo
    unico = len(arms) == 1
    token = token_urlsafe(16)
    sesion = _Sesion(state="who" if profiles.professionals() else "first_profile")
    licencia = [license]
    inactividad = wording.idle_lock_minutes * 60
    enlace = (
        ""
        if unico
        else (
            f'<button type="button" class="head-link" id="to-launcher" onclick="'
            f"fetch('/{token}/lanzador', {{method: 'POST'}}).then(() => location.reload())\">"
            f'<svg><use href="#i-back"/></svg>{html.escape(wording["launcher_link"])}</button>'
        )
    )

    def contexto() -> ShellContext:
        return ShellContext(
            token=token,
            brand=brand,
            now=now(),
            professional=sesion.professional,
            license=licencia[0],
            audit=audit,
            wording=wording,
            lock=lock,
            profiles=tuple(profiles.professionals()),
            launcher_link=enlace,
        )

    def brazo() -> ArmApp | None:
        return por_id[sesion.current] if sesion.current is not None else None

    def todos(accion: Callable[[ArmApp], None]) -> None:
        for b in arms:
            accion(b)

    def identificado() -> bool:
        return sesion.professional is not None and sesion.state not in (
            "who",
            "first_profile",
            "locked",
        )

    def actor() -> Professional:
        """Con el candado tomado y en un estado identificado: quien firma lo que se registra. Si
        alguna vez falta, revienta en vez de escribir un registro sin autor (`I-156`)."""
        if sesion.professional is None:
            raise IdentityError("no_actor", "no hay profesional en la sesión")
        return sesion.professional

    def en_reposo() -> bool:
        """El lanzador es reposo; un brazo, cuando él lo dice."""
        if sesion.state == "launcher":
            return True
        actual = brazo()
        return actual is not None and actual.at_rest()

    def tras_identificarse() -> None:
        sesion.last_activity = clock()
        if unico:
            sesion.current = arms[0].id
            sesion.state = "arm"
        else:
            sesion.current = None
            sesion.state = "launcher"

    def vigilar() -> None:
        """Con el candado tomado, en cada petición. Bloquea si pasó el plazo sin uso, y solo donde
        no hay nada a medias: lo deciden los brazos, y el lanzador siempre es reposo."""
        if sesion.professional is None or sesion.state not in ("arm", "launcher"):
            return
        ahora = clock()
        if en_reposo() and ahora - sesion.last_activity > inactividad:
            todos(lambda b: b.reset())
            sesion.current = None
            sesion.state = "locked"
            audit.record("locked", actor=sesion.professional)
            return
        sesion.last_activity = ahora

    def pagina() -> str:
        if sesion.state == "first_profile":
            return first_profile_page(wording, brand, token, now())
        if sesion.state == "who":
            return who_page(wording, brand, token, profiles.professionals(), now())
        if sesion.state == "locked" and sesion.professional is not None:
            return locked_page(wording, brand, token, sesion.professional, now())
        if sesion.state == "launcher":
            tarjetas = tuple(
                (b.id, b.name, *presentacion.get(b.id, ("", "badge")))
                for b in sorted(arms, key=lambda b: b.id)
            )
            return launcher_page(
                wording,
                brand,
                token,
                tarjetas,
                now(),
                sesion.professional,
                licencia[0],
            )
        actual = brazo()
        assert actual is not None
        return _con_enlace_lanzador(actual.page(contexto()), enlace)

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
            respuesta = json_response(status, body)
            self._send(respuesta.status, respuesta.body, respuesta.content_type)

        def _enviar(self, respuesta: Response | None) -> None:
            if respuesta is None:
                self._send(404, b"", "text/plain")
                return
            self._send(respuesta.status, respuesta.body, respuesta.content_type)

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
                if not self.path.startswith(f"/{token}/"):
                    self._enviar(None)
                    return
                if sesion.state == "launcher":
                    self._json(409, {"error": ""})
                    return
                ruta = self.path[len(token) + 2 :]
                ctx = contexto()
                actual = brazo()
            assert actual is not None
            self._enviar(actual.get(ruta, ctx))

        def do_POST(self) -> None:
            ruta = self.path
            if not ruta.startswith(f"/{token}/"):
                self._enviar(None)
                return
            accion = ruta[len(token) + 2 :]
            with lock:
                vigilar()
                permitido = identificado()
            if accion in _IDENTIDAD:
                self._identity(accion)
                return
            if not permitido:
                self._json(401, {"error": ""})
                return
            if accion == "licencia":
                self._license()
                return
            if accion == "restablecer":
                self._restablecer()
                return
            if accion == "codigo-recuperacion":
                self._codigo()
                return
            if accion == "brazo":
                self._abrir_brazo()
                return
            if accion == "lanzador":
                self._volver_al_lanzador()
                return
            cuerpo = self._body()
            with lock:
                if sesion.state == "launcher":
                    self._json(409, {"error": ""})
                    return
                ctx = contexto()
                actual = brazo()
            cabeceras = {clave.lower(): valor for clave, valor in self.headers.items()}
            assert actual is not None
            self._enviar(actual.post(accion, cuerpo, cabeceras, ctx))

        def _abrir_brazo(self) -> None:
            """Desde el lanzador, abre un brazo por su `id` (101b)."""
            try:
                pedido = json.loads(self._body() or b"{}")
            except ValueError:
                pedido = None
            if not isinstance(pedido, dict):
                self._json(400, {"error": ""})
                return
            with lock:
                if sesion.state != "launcher":
                    self._json(409, {"error": ""})
                    return
                elegido = pedido.get("id")
                if not isinstance(elegido, str) or elegido not in por_id:
                    self._json(404, {"error": ""})
                    return
                sesion.current = elegido
                sesion.state = "arm"
            self._json(200, {"ok": True})

        def _volver_al_lanzador(self) -> None:
            """Vuelve al lanzador si el brazo está en reposo: nunca a mitad de una cita o de una
            consulta (101b)."""
            with lock:
                if unico:
                    self._json(404, {"error": ""})
                    return
                actual = brazo()
                if actual is None or not actual.at_rest():
                    self._json(409, {"error": ""})
                    return
                actual.reset()
                sesion.current = None
                sesion.state = "launcher"
            self._json(200, {"ok": True})

        def _identity(self, accion: str) -> None:
            """Entrar, crear un perfil, desbloquear, cambiar de profesional y salir: lo único que se
            atiende sin estar identificado."""
            if accion == "recuperar-pin":
                self._recuperar()
                return
            if accion == "salir":
                with lock:
                    if sesion.professional is not None:
                        audit.record("session_closed", actor=sesion.professional)
                    todos(lambda b: b.reset())
                    sesion.state = "closed"
                self._json(200, {"ok": True})
                threading.Thread(target=http_server.shutdown, daemon=True).start()
                return
            if accion == "cambiar":
                with lock:
                    if not all(b.at_rest() for b in arms):
                        self._json(409, {"error": ""})
                        return
                    if sesion.professional is not None:
                        audit.record("session_closed", actor=sesion.professional)
                    todos(lambda b: b.reset())
                    sesion.professional = None
                    sesion.current = None
                    sesion.state = "who"
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
                    elif accion == "entrar" and sesion.state == "who":
                        perfil = str(pedido.get("profile", ""))
                        sesion.professional = profiles.verify(perfil, str(pedido.get("pin", "")))
                        audit.record("session_opened", actor=sesion.professional)
                        todos(lambda b: b.on_identified(actor()))
                        tras_identificarse()
                        self._json(200, {"ok": True})
                    elif (
                        accion == "desbloquear"
                        and sesion.state == "locked"
                        and sesion.professional is not None
                    ):
                        profiles.verify(sesion.professional.id, str(pedido.get("pin", "")))
                        audit.record("unlocked", actor=sesion.professional)
                        todos(lambda b: b.on_unlocked())
                        tras_identificarse()
                        self._json(200, {"ok": True})
                    else:
                        self._json(409, {"error": ""})
                except IdentityError as exc:
                    restantes = (
                        profiles.attempts_left(
                            str(pedido.get("profile", ""))
                            if accion == "entrar"
                            else sesion.professional.id
                            if sesion.professional
                            else ""
                        )
                        if exc.code == "bad_pin"
                        else 0
                    )
                    self._json(400, {"error": wording[f"identity_{exc.code}"].format(restantes)})

        def _recuperar(self) -> None:
            """Cambiar el PIN con el código de recuperación del perfil (101b, decisión 1). No abre
            sesión: quien lo cambia entra después con el PIN nuevo."""
            try:
                pedido = json.loads(self._body() or b"{}")
            except ValueError:
                pedido = None
            if not isinstance(pedido, dict):
                self._json(400, {"error": ""})
                return
            pin = str(pedido.get("pin", ""))
            if pin != str(pedido.get("pin2", "")):
                self._json(400, {"error": wording["first_pin_mismatch"]})
                return
            with lock:
                try:
                    perfil = profiles.recover_pin(
                        str(pedido.get("profile", "")), str(pedido.get("code", "")), pin
                    )
                except IdentityError as exc:
                    self._json(400, {"error": wording[f"identity_{exc.code}"]})
                    return
                audit.record("pin_recovered", actor=perfil, data={"profile": perfil.id})
            self._json(200, {"ok": True})

        def _restablecer(self) -> None:
            """Un profesional identificado pone un PIN nuevo a otro perfil del equipo. Queda en la
            auditoría con quien lo hizo (101b, decisión 2)."""
            try:
                pedido = json.loads(self._body() or b"{}")
            except ValueError:
                pedido = None
            if not isinstance(pedido, dict):
                self._json(400, {"error": ""})
                return
            pin = str(pedido.get("pin", ""))
            if pin != str(pedido.get("pin2", "")):
                self._json(400, {"error": wording["first_pin_mismatch"]})
                return
            with lock:
                if sesion.professional is None:
                    self._json(401, {"error": ""})
                    return
                try:
                    otro = profiles.reset_pin(str(pedido.get("profile", "")), pin)
                except IdentityError as exc:
                    self._json(400, {"error": wording[f"identity_{exc.code}"]})
                    return
                audit.record("pin_reset", actor=actor(), owner=otro.id, data={"profile": otro.id})
            self._json(200, {"ok": True})

        def _codigo(self) -> None:
            """Genera un código de recuperación nuevo para quien atiende. Se devuelve una vez."""
            with lock:
                if sesion.professional is None:
                    self._json(401, {"error": ""})
                    return
                codigo = profiles.new_recovery_code(sesion.professional.id)
                audit.record(
                    "recovery_code_created",
                    actor=sesion.professional,
                    data={"profile": sesion.professional.id},
                )
            self._json(200, {"codigo": codigo})

        def _new_profile(self, pedido: dict[str, Any]) -> None:
            """Con el candado tomado. El primero, en el primer arranque, y entra con él; los demás,
            desde Ajustes y por alguien ya identificado (plan 83): nadie se da de alta solo en un
            equipo que ya tiene dueños."""
            primero = sesion.state == "first_profile"
            if not primero and not (identificado() and any(b.in_settings() for b in arms)):
                self._json(409, {"error": ""})
                return
            pin = str(pedido.get("pin", ""))
            if pin != str(pedido.get("pin2", "")):
                self._json(400, {"error": wording["first_pin_mismatch"]})
                return
            nuevo = profiles.create(
                name=str(pedido.get("name", "")),
                registry=str(pedido.get("registry", "")),
                pin=pin,
            )
            audit.record(
                "profile_created",
                actor=nuevo if primero else actor(),
                data={"profile": nuevo.id, "name": nuevo.label()},
            )
            codigo = profiles.new_recovery_code(nuevo.id)
            audit.record("recovery_code_created", actor=nuevo, data={"profile": nuevo.id})
            if primero:
                sesion.professional = nuevo
                audit.record("session_opened", actor=nuevo)
                todos(lambda b: b.on_identified(nuevo))
                tras_identificarse()
            self._json(200, {"ok": True, "codigo": codigo})

        def _license(self) -> None:
            """Cargar una licencia: se verifica antes de guardarla, y una que no sirve no pisa la
            que había. Vale si da derecho a algún brazo de este equipo. Queda en la auditoría, con
            quien la cargó."""
            datos = self._body() or b""
            estado = read_license(datos.decode("utf-8", "replace"), today=today())
            if not any(estado.licensed(b.id) for b in arms) or estado.status == "expired":
                self._json(400, {"error": wording["license_rejected"].format(estado.status)})
                return
            with lock:
                destino = license_path if license_path is not None else license_file()
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(datos)
                with contextlib.suppress(OSError):
                    destino.chmod(0o600)
                licencia[0] = estado
                assert estado.license is not None
                audit.record(
                    "license_loaded",
                    actor=actor(),
                    data={
                        "license": estado.license.id,
                        "client": estado.license.client,
                        "expires": estado.license.expires.isoformat(),
                    },
                )
            self._json(200, {"ok": True})

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
            todos(lambda b: b.reset())
