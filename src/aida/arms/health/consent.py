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

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

from aida.engine.ontology import AgentId
from aida.platform.ledger import ChainedLog
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
    """Quién la recogió: desde la 83, el profesional identificado y el equipo
    («nombre (registro) @ usuario@equipo»); antes, solo el usuario y el equipo."""
    at: str
    """ISO 8601 con zona: una evidencia sin zona horaria no se puede situar."""
    prev: str
    digest: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)




class ConsentLedger:
    """El registro de autorizaciones, sobre la cadena común (`aida.ledger`). Con
    `path=None` vive en memoria: sigue siendo la misma puerta (hay que registrar para
    poder transcribir), solo que no deja archivo. El formato de cada línea es el de la
    82: las autorizaciones ya registradas siguen verificando."""

    def __init__(self, path: Path | None = None) -> None:
        self._chain = ChainedLog(path)

    def append(
        self, *, owner: AgentId, mode: str, version: str, text: str, workstation: str, at: str
    ) -> ConsentRecord:
        if mode not in MODES:
            raise ValueError(f"modo de autorización desconocido: {mode!r}")
        linea = self._chain.append(
            {
                "owner": str(owner),
                "mode": mode,
                "version": version,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "workstation": workstation,
                "at": at,
            }
        )
        return _record(linea)

    def contains(self, record: ConsentRecord) -> bool:
        return self._chain.contains(record.digest)

    def records(self) -> tuple[ConsentRecord, ...]:
        return tuple(_record(linea) for linea in self._chain.lines())

    def verify(self) -> tuple[str, ...]:
        return self._chain.verify()


def _record(linea: Mapping[str, str]) -> ConsentRecord:
    return ConsentRecord(
        owner=AgentId(linea["owner"]),
        mode=linea["mode"],
        version=linea["version"],
        text_sha256=linea["text_sha256"],
        workstation=linea["workstation"],
        at=linea["at"],
        prev=linea["prev"],
        digest=linea["digest"],
    )
