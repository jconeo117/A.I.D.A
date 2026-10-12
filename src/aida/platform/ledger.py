"""Un registro solo de añadir y encadenado: cada línea lleva la huella de la anterior, y
la suya se calcula sobre todo lo demás.

Nació en la 82 para las autorizaciones y se saca aquí en la 83, cuando la auditoría
necesita lo mismo: dos copias de «cómo se encadena y cómo se verifica» acabarían
verificando de dos maneras. Una sola puerta.

No impide manipular el archivo —quien lo tenga puede reescribirlo entero—, pero una
manipulación parcial (borrar o cambiar una línea) no pasa desapercibida.

Los valores son cadenas, y solo cadenas: la huella se calcula sobre el JSON con claves
ordenadas, y un número o un booleano podrían escribirse de dos formas.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

GENESIS = "0" * 64


def chain_digest(fields: Mapping[str, str]) -> str:
    cuerpo = json.dumps(dict(fields), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(cuerpo.encode("utf-8")).hexdigest()


class ChainedLog:
    """Con `path=None` vive en memoria: la misma lógica, sin archivo."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lines: list[dict[str, str]] = []
        if path is not None and path.is_file():
            for linea in path.read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    self._lines.append({k: str(v) for k, v in json.loads(linea).items()})
        self._digests = {linea["digest"] for linea in self._lines}

    def append(self, fields: Mapping[str, str]) -> dict[str, str]:
        if "prev" in fields or "digest" in fields:
            raise ValueError("'prev' y 'digest' los pone la cadena, no quien añade")
        campos = {**dict(fields), "prev": self._last_digest()}
        linea = {**campos, "digest": chain_digest(campos)}
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Escrita y llevada a disco antes de devolverla: lo que se da por registrado
            # y se pierde en un corte no es evidencia de nada.
            with self._path.open("a", encoding="utf-8") as archivo:
                archivo.write(json.dumps(linea, ensure_ascii=False, sort_keys=True) + "\n")
                archivo.flush()
                os.fsync(archivo.fileno())
            # Legible solo por el usuario. En Windows los permisos los da la carpeta del
            # perfil.
            with contextlib.suppress(OSError):
                self._path.chmod(0o600)
        self._lines.append(linea)
        self._digests.add(linea["digest"])
        return linea

    def _last_digest(self) -> str:
        """La última huella **del archivo**, no de memoria: la aplicación y la CLI pueden
        escribir en el mismo registro desde dos procesos, y cada uno debe encadenar
        detrás de lo que de verdad hay escrito."""
        if self._path is not None and self._path.is_file():
            for linea in reversed(self._path.read_text(encoding="utf-8").splitlines()):
                if linea.strip():
                    return str(json.loads(linea)["digest"])
            return GENESIS
        return self._lines[-1]["digest"] if self._lines else GENESIS

    def contains(self, digest: str) -> bool:
        return digest in self._digests

    def lines(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(linea) for linea in self._lines)

    def verify(self) -> tuple[str, ...]:
        """Los problemas de la cadena, leída de nuevo del archivo: lo que hay en memoria
        no prueba nada sobre lo que hay en disco. Vacío si está íntegra."""
        if self._path is None or not self._path.is_file():
            return ()
        problemas: list[str] = []
        anterior = GENESIS
        lineas = self._path.read_text(encoding="utf-8").splitlines()
        for numero, linea in enumerate(lineas, start=1):
            if not linea.strip():
                continue
            try:
                datos = json.loads(linea)
                guardada = str(datos.pop("digest"))
            except (ValueError, KeyError, AttributeError):
                problemas.append(f"línea {numero}: no se puede leer")
                anterior = ""
                continue
            if datos.get("prev") != anterior:
                problemas.append(f"línea {numero}: no sigue a la anterior")
            if chain_digest({k: str(v) for k, v in datos.items()}) != guardada:
                problemas.append(f"línea {numero}: su contenido no corresponde a su huella")
            anterior = guardada
        return tuple(problemas)
