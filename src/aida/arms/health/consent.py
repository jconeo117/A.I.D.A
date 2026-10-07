"""La autorización de quien es grabado: el texto que se le lee, y su registro.

Plan 82. Antes la casilla de la aplicación no dejaba rastro, y al subir un audio no
se pedía nada. Ahora cada autorización es una línea de un registro solo de añadir, y
**no se transcribe audio sin una autorización que esté en él** (`Consultation`).

Por qué un registro propio y no el journal: el registro de eventos de la aplicación
vive en memoria, y volcarlo entero llevaría la transcripción en claro. Este solo
guarda quién, cuándo, cómo y con qué texto: nada de lo que se dijo.

Por qué sin cifrar: es una evidencia. Si se cifrara con la clave de la caché y esa
clave se perdiera, se perderían a la vez todas las pruebas de autorización.

Por qué encadenado: cada línea lleva la huella de la anterior, y la suya se calcula
sobre todo lo demás. Borrar o cambiar una línea rompe la cadena, y `verify` lo dice.
No impide manipularlo —quien tenga el archivo puede reescribirlo entero—, pero hace
que una manipulación parcial no pase desapercibida.
"""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import os
import socket
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

from aida.engine.ontology import AgentId
from aida.resources import data_root

_RAIZ = data_root()
_TEXTOS_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "consent.yaml"
_GENESIS = "0" * 64
MODES = ("recorded", "uploaded")


class ConsentMissingError(ValueError):
    """Se intentó transcribir sin una autorización registrada para ese dueño."""


@dataclass(frozen=True, slots=True)
class ConsentText:
    title: str
    text: str
    check: str
    label: str
    """Cómo se nombra el modo en la línea de la revisión («grabada», «subida»)."""


@dataclass(frozen=True, slots=True)
class ConsentTexts:
    version: str
    modes: Mapping[str, ConsentText]
    line_template: str
    """`{0}` el modo, `{1}` la fecha y hora, `{2}` la versión."""

    def line(self, record: ConsentRecord) -> str:
        """La única forma de enseñar una autorización: la usan la revisión y el
        export, y con dos copias acabarían diciendo cosas distintas."""
        momento = datetime.fromisoformat(record.at).strftime("%d/%m/%Y %H:%M")
        return self.line_template.format(
            self.modes[record.mode].label, momento, record.version
        )


def load_consent_texts(path: Path | None = None) -> ConsentTexts:
    """Revienta si falta algo: un texto de autorización a medias no es informado."""
    ruta = path if path is not None else _TEXTOS_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ruta} debe ser un mapeo YAML")
    version = raw.get("version")
    plantilla = raw.get("line_template")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"'version' debe ser una cadena no vacía en {ruta}")
    if not isinstance(plantilla, str) or not plantilla.strip():
        raise ValueError(f"'line_template' debe ser una cadena no vacía en {ruta}")
    modos: dict[str, ConsentText] = {}
    for modo in MODES:
        bloque = raw.get(modo)
        claves = ("title", "text", "check", "label")
        if not isinstance(bloque, dict) or not all(
            isinstance(bloque.get(c), str) and bloque[c].strip() for c in claves
        ):
            raise ValueError(f"'{modo}' necesita {list(claves)} no vacías en {ruta}")
        modos[modo] = ConsentText(**{c: " ".join(bloque[c].split()) for c in claves})
    return ConsentTexts(version=version.strip(), modes=modos, line_template=plantilla)


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    owner: AgentId
    mode: str
    version: str
    text_sha256: str
    """Huella del texto exacto que se leyó: si alguien cambia el texto sin subir la
    versión, la huella lo delata."""
    workstation: str
    """Quién la recogió, hasta que la 83 traiga la identidad del profesional: el
    usuario del sistema y el nombre del equipo."""
    at: str
    """ISO 8601 con zona: una evidencia sin zona horaria no se puede situar."""
    prev: str
    digest: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def _digest(campos: Mapping[str, str]) -> str:
    cuerpo = json.dumps(dict(campos), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(cuerpo.encode("utf-8")).hexdigest()


def workstation() -> str:
    return f"{getpass.getuser()}@{socket.gethostname()}"


class ConsentLedger:
    """El registro. Con `path=None` vive en memoria: sigue siendo la misma puerta
    (hay que registrar para poder transcribir), solo que no deja archivo."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._records: list[ConsentRecord] = []
        if path is not None and path.is_file():
            for linea in path.read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    self._records.append(ConsentRecord(**json.loads(linea)))
        self._digests = {r.digest for r in self._records}

    def append(
        self, *, owner: AgentId, mode: str, version: str, text: str, workstation: str, at: str
    ) -> ConsentRecord:
        if mode not in MODES:
            raise ValueError(f"modo de autorización desconocido: {mode!r}")
        campos = {
            "owner": str(owner),
            "mode": mode,
            "version": version,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "workstation": workstation,
            "at": at,
            "prev": self._records[-1].digest if self._records else _GENESIS,
        }
        record = ConsentRecord(
            owner=AgentId(owner),
            mode=mode,
            version=version,
            text_sha256=campos["text_sha256"],
            workstation=workstation,
            at=at,
            prev=campos["prev"],
            digest=_digest(campos),
        )
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Escrita y llevada a disco antes de devolverla: una autorización que se da
            # por registrada y se pierde en un corte no es evidencia de nada.
            with self._path.open("a", encoding="utf-8") as archivo:
                archivo.write(record.to_json() + "\n")
                archivo.flush()
                os.fsync(archivo.fileno())
            # Lleva códigos de quien fue grabado: legible solo por el usuario, como las
            # claves. En Windows los permisos los da la carpeta del perfil.
            with contextlib.suppress(OSError):
                self._path.chmod(0o600)
        self._records.append(record)
        self._digests.add(record.digest)
        return record

    def contains(self, record: ConsentRecord) -> bool:
        return record.digest in self._digests

    def records(self) -> tuple[ConsentRecord, ...]:
        return tuple(self._records)

    def verify(self) -> tuple[str, ...]:
        """Los problemas de la cadena, leída de nuevo del archivo: lo que hay en
        memoria no prueba nada sobre lo que hay en disco. Vacío si está íntegra."""
        if self._path is None or not self._path.is_file():
            return ()
        problemas: list[str] = []
        anterior = _GENESIS
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
            if _digest({k: str(v) for k, v in datos.items()}) != guardada:
                problemas.append(f"línea {numero}: su contenido no corresponde a su huella")
            anterior = guardada
        return tuple(problemas)
