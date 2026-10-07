"""Quién usa AIDA: perfiles de profesional con PIN (plan 83).

Antes no existía: el `owner` de los eventos es la persona atendida, y la autorización
de la 82 registraba `usuario@equipo`. En un consultorio el equipo se comparte, así que
el usuario del sistema no dice quién actuó.

El PIN no se guarda: solo una huella PBKDF2-SHA256 con sal propia por perfil, de la
biblioteca estándar. Un PIN de 4 a 8 dígitos es poco espacio; lo que lo protege es que
cada intento cueste (200 000 iteraciones) y que tras cinco fallos haya que esperar.
El límite queda escrito: protege frente a quien prueba en la pantalla, no frente a quien
se lleve el archivo y lo ataque con tiempo. Para eso hace falta el llavero del sistema
(diferido con la 85).
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PIN_DIGITS = (4, 8)
_ITERACIONES = 200_000
_FALLOS_MAXIMOS = 5
_ESPERA_SEGUNDOS = 30.0


class IdentityError(ValueError):
    """Perfil o PIN no válidos. `code` elige el texto del pack que ve quien está
    delante (`identity_<code>` en `ui.yaml`): el código no escribe mensajes para la
    pantalla."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code


@dataclass(frozen=True, slots=True)
class Professional:
    id: str
    name: str
    registry: str
    """El número de registro profesional: lo que identifica a la persona ante terceros,
    porque dos profesionales pueden llamarse igual."""

    def label(self) -> str:
        """Cómo se nombra en los registros: el nombre y el registro, juntos."""
        return f"{self.name} ({self.registry})"


def valid_pin(pin: str) -> bool:
    return pin.isdigit() and PIN_DIGITS[0] <= len(pin) <= PIN_DIGITS[1]


def _huella(pin: str, sal: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), sal, _ITERACIONES).hex()


class ProfileStore:
    """Los perfiles de este equipo, en un archivo de la carpeta de configuración.

    Los fallos se cuentan en memoria y por perfil: reiniciar AIDA los pone a cero, y es
    aceptable, porque reiniciar ya cuesta más que esperar 30 s."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._path = path
        self._clock = clock
        self._fallos: dict[str, int] = {}
        self._bloqueado_hasta: dict[str, float] = {}

    def _leer(self) -> dict[str, dict[str, str]]:
        if not self._path.is_file():
            return {}
        datos = json.loads(self._path.read_text(encoding="utf-8"))
        return {str(k): {str(a): str(b) for a, b in v.items()} for k, v in datos.items()}

    def _escribir(self, perfiles: dict[str, dict[str, str]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(perfiles, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        with contextlib.suppress(OSError):
            self._path.chmod(0o600)

    def professionals(self) -> tuple[Professional, ...]:
        """Ordenados por nombre y, a igualdad, por id: el orden de la pantalla no puede
        depender del orden del archivo."""
        perfiles = self._leer()
        todos = (
            Professional(id=k, name=v["name"], registry=v["registry"])
            for k, v in perfiles.items()
        )
        return tuple(sorted(todos, key=lambda p: (p.name.casefold(), p.id)))

    def get(self, profile_id: str) -> Professional | None:
        return next((p for p in self.professionals() if p.id == profile_id), None)

    def create(self, *, name: str, registry: str, pin: str) -> Professional:
        nombre, registro = " ".join(name.split()), " ".join(registry.split())
        if not nombre or not registro:
            raise IdentityError("missing")
        if not valid_pin(pin):
            raise IdentityError("pin_format")
        perfiles = self._leer()
        if any(v["registry"] == registro for v in perfiles.values()):
            raise IdentityError("duplicate")
        sal = secrets.token_bytes(16)
        identificador = secrets.token_hex(4)
        perfiles[identificador] = {
            "name": nombre,
            "registry": registro,
            "salt": sal.hex(),
            "pin": _huella(pin, sal),
        }
        self._escribir(perfiles)
        return Professional(id=identificador, name=nombre, registry=registro)

    def delete(self, profile_id: str) -> Professional:
        perfiles = self._leer()
        if profile_id not in perfiles:
            raise IdentityError("unknown")
        v = perfiles.pop(profile_id)
        self._escribir(perfiles)
        return Professional(id=profile_id, name=v["name"], registry=v["registry"])

    def attempts_left(self, profile_id: str) -> int:
        """Cuántos fallos quedan antes de tener que esperar: la pantalla lo dice."""
        return _FALLOS_MAXIMOS - self._fallos.get(profile_id, 0)

    def wait_seconds(self, profile_id: str) -> float:
        """Cuánto falta para poder volver a intentarlo; 0 si se puede ya."""
        return max(0.0, self._bloqueado_hasta.get(profile_id, 0.0) - self._clock())

    def verify(self, profile_id: str, pin: str) -> Professional:
        """El profesional si el PIN es el suyo. Revienta con un mensaje para la pantalla
        si no, o si aún tiene que esperar tras cinco fallos."""
        perfil = self._leer().get(profile_id)
        if perfil is None:
            raise IdentityError("unknown")
        if self.wait_seconds(profile_id) > 0:
            raise IdentityError("wait")
        esperada = perfil["pin"]
        calculada = _huella(pin, bytes.fromhex(perfil["salt"])) if valid_pin(pin) else ""
        if not hmac.compare_digest(esperada, calculada):
            fallos = self._fallos.get(profile_id, 0) + 1
            if fallos >= _FALLOS_MAXIMOS:
                self._bloqueado_hasta[profile_id] = self._clock() + _ESPERA_SEGUNDOS
                fallos = 0
            self._fallos[profile_id] = fallos
            raise IdentityError("bad_pin" if fallos else "wait")
        self._fallos.pop(profile_id, None)
        return Professional(id=profile_id, name=perfil["name"], registry=perfil["registry"])
