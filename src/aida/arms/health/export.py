"""Sacar la consulta revisada del equipo en un archivo que se pueda pegar en
una historia clínica.

Sin cifrar, a propósito: un archivo cifrado no se pega en ningún sitio, y
cifrarlo daría una sensación de seguridad que el uso real desmiente. Lo que sí
hace esta pieza es **decir dónde lo deja**, para que quien ejecuta sepa que ahí
queda información clínica en claro.

El nombre del archivo no lleva el identificador del paciente (`I-077`): lleva la
fecha y una huella corta de ese identificador, que basta para reencontrar sus
consultas sin escribir quién es en el nombre.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aida.arms.health.consultation import ConsultationResult
from aida.arms.health.note import RenderedNote, load_note_settings
from aida.engine.ontology import AgentId


def default_export_dir() -> Path:
    """Visible y fuera de directorios ocultos: el archivo existe para que una
    persona lo abra y lo copie."""
    return Path.home() / "Documentos" / "aida" / "consultas"


def export_name(owner: AgentId, moment: datetime) -> str:
    """`AAAA-MM-DD-HHMM-<huella>`: ordena por fecha al listar, y la huella
    agrupa las consultas de un mismo sujeto sin nombrarlo."""
    huella = hashlib.blake2b(str(owner).encode("utf-8"), digest_size=3).hexdigest()
    return f"{moment:%Y-%m-%d-%H%M}-{huella}"


def _render_note_markdown(note: RenderedNote) -> str:
    """La nota, con el encabezado por campo y —si hay respaldo o procedencia—
    el aviso correspondiente. Antes del resumen de siempre: es lo que el
    profesional va a copiar primero."""
    ajustes = load_note_settings()
    lineas: list[str] = []
    if not note.redacted:
        lineas.append(ajustes.fallback_notice)
        lineas.append("")
    for spec, texto in note.fields:
        lineas.append(ajustes.field_heading.format(spec.label))
        lineas.append(texto)
        lineas.append("")
    if note.provider is not None:
        proveedor, _separador, modelo = note.provider.partition("/")
        lineas.append(ajustes.provenance_template.format(proveedor, modelo))
    return "\n".join(lineas).rstrip()


@dataclass(frozen=True, slots=True)
class ExportedFiles:
    summary: Path
    model: Path | None


def export_consultation(
    result: ConsultationResult,
    *,
    destination: Path,
    moment: datetime,
    include_model: bool,
) -> ExportedFiles:
    """Escribe el resumen y, si se pide, el modelo. `destination` puede ser un
    archivo concreto o un directorio: si es directorio, el nombre lo pone
    `export_name`.

    La transcripción **no** se exporta: es lo más sensible de la consulta y no
    se pega en ninguna historia clínica.
    """
    if destination.suffix:
        summary_path = destination
    else:
        summary_path = destination / f"{export_name(result.owner, moment)}.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    contenido = result.summary_markdown
    if result.note is not None:
        contenido = _render_note_markdown(result.note) + "\n\n" + contenido
    summary_path.write_text(contenido + "\n", encoding="utf-8")

    model_path = None
    if include_model and result.encounter is not None:
        model_path = summary_path.with_suffix(".json")
        model_path.write_text(
            json.dumps(result.encounter.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return ExportedFiles(summary=summary_path, model=model_path)
