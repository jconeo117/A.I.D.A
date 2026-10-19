"""El armazón de la plataforma: servidor local con token, quién atiende, el bloqueo por
inactividad, la licencia y Ajustes. Un brazo le entrega sus pantallas y sus rutas (101a).

Por qué vive aquí y no en el brazo: quién está identificado, cuándo se bloquea y qué
licencia vale son decisiones de la plataforma. Si cada brazo las tomara por su lado, dos
puertas de PIN acabarían divergiendo, y un registro podría escribirse sin autor en uno
de ellos.
"""

from __future__ import annotations

import contextlib
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
    license_file,
    locked_page,
    who_page,
)
from aida.platform.web.wording import Textos

_MAX_BYTES = 512 * 1024 * 1024
_IDENTIDAD = ("perfil", "entrar", "desbloquear", "cambiar", "salir")


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
    """Quién atiende y desde cuándo. El estado del trabajo (una consulta, una revisión) es
    del brazo: aquí solo se sabe si hay alguien identificado y en qué pantalla de la
    plataforma está."""

    state: str
    professional: Professional | None = None
    last_activity: float = 0.0


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
    open_browser: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = datetime.now,
    today: Callable[[], date] = date.today,
) -> None:
    """Levanta el armazón y bloquea hasta que alguien pulsa «Salir» (o Ctrl-C). Hoy monta un
    solo brazo: el lanzador entre varios llega con la 101b."""
    if len(arms) != 1:
        raise ValueError("el armazón monta un brazo hasta que llegue el lanzador (101b)")
    arm = arms[0]
    token = token_urlsafe(16)
    sesion = _Sesion(state="who" if profiles.professionals() else "first_profile")
    licencia = [license]
    inactividad = wording.idle_lock_minutes * 60

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
        )

    def identificado() -> bool:
        return sesion.professional is not None and sesion.state not in (
            "who", "first_profile", "locked",
        )

    def actor() -> Professional:
        """Con el candado tomado y en un estado identificado: quien firma lo que se
        registra. Si alguna vez falta, revienta en vez de escribir un registro sin autor
        (`I-156`)."""
        if sesion.professional is None:
            raise IdentityError("no_actor", "no hay profesional en la sesión")
        return sesion.professional

    def tras_identificarse() -> None:
        sesion.last_activity = clock()
        sesion.state = "arm"

    def vigilar() -> None:
        """Con el candado tomado, en cada petición. Bloquea si pasó el plazo sin uso, y solo
        donde no hay nada a medias: lo decide el brazo (`at_rest`), nunca el armazón."""
        if sesion.professional is None or sesion.state != "arm":
            return
        ahora = clock()
        if arm.at_rest() and ahora - sesion.last_activity > inactividad:
            arm.reset()
            sesion.state = "locked"
            audit.record("locked", actor=sesion.professional)
            return
        if sesion.state != "locked":
            sesion.last_activity = ahora

    def pagina() -> str:
        if sesion.state == "first_profile":
            return first_profile_page(wording, brand, token, now())
        if sesion.state == "who":
            return who_page(wording, brand, token, profiles.professionals(), now())
        if sesion.state == "locked" and sesion.professional is not None:
            return locked_page(wording, brand, token, sesion.professional, now())
        return arm.page(contexto())

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
                ruta = self.path[len(token) + 2 :]
                ctx = contexto()
            self._enviar(arm.get(ruta, ctx))

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
            cuerpo = self._body()
            with lock:
                ctx = contexto()
            cabeceras = {clave.lower(): valor for clave, valor in self.headers.items()}
            self._enviar(arm.post(accion, cuerpo, cabeceras, ctx))

        def _identity(self, accion: str) -> None:
            """Entrar, crear un perfil, desbloquear, cambiar de profesional y salir: lo único
            que se atiende sin estar identificado."""
            if accion == "salir":
                with lock:
                    if sesion.professional is not None:
                        audit.record("session_closed", actor=sesion.professional)
                    arm.reset()
                    sesion.state = "closed"
                self._json(200, {"ok": True})
                threading.Thread(target=http_server.shutdown, daemon=True).start()
                return
            if accion == "cambiar":
                with lock:
                    if not arm.at_rest():
                        self._json(409, {"error": ""})
                        return
                    if sesion.professional is not None:
                        audit.record("session_closed", actor=sesion.professional)
                    arm.reset()
                    sesion.professional = None
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
                        arm.on_identified(sesion.professional)
                        tras_identificarse()
                        self._json(200, {"ok": True})
                    elif (
                        accion == "desbloquear"
                        and sesion.state == "locked"
                        and sesion.professional is not None
                    ):
                        profiles.verify(sesion.professional.id, str(pedido.get("pin", "")))
                        audit.record("unlocked", actor=sesion.professional)
                        arm.on_unlocked()
                        tras_identificarse()
                        self._json(200, {"ok": True})
                    else:
                        self._json(409, {"error": ""})
                except IdentityError as exc:
                    restantes = (
                        profiles.attempts_left(
                            str(pedido.get("profile", ""))
                            if accion == "entrar"
                            else sesion.professional.id if sesion.professional else ""
                        )
                        if exc.code == "bad_pin"
                        else 0
                    )
                    self._json(
                        400, {"error": wording[f"identity_{exc.code}"].format(restantes)}
                    )

        def _new_profile(self, pedido: dict[str, Any]) -> None:
            """Con el candado tomado. El primero, en el primer arranque, y entra con él; los
            demás, desde Ajustes y por alguien ya identificado (plan 83): nadie se da de alta
            solo en un equipo que ya tiene dueños."""
            primero = sesion.state == "first_profile"
            if not primero and not (identificado() and arm.in_settings()):
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
            if primero:
                sesion.professional = nuevo
                audit.record("session_opened", actor=nuevo)
                arm.on_identified(nuevo)
                tras_identificarse()
            self._json(200, {"ok": True})

        def _license(self) -> None:
            """Cargar una licencia: se verifica antes de guardarla, y una que no sirve no pisa
            la que había. Queda en la auditoría, con quien la cargó."""
            datos = self._body() or b""
            estado = read_license(datos.decode("utf-8", "replace"), today=today())
            if not estado.licensed(arm.id) or estado.status == "expired":
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
            arm.reset()
