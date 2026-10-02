"""La nota clínica: los hechos de cada apartado, redactados en prosa cuando
hay proveedor de lenguaje, o tal como se dijeron cuando no lo hay.

La nota **no** menciona los códigos: van en su propio bloque del resumen, y
ese desacople es lo que permite redactar la nota antes de la revisión —
rechazar un código no obliga a redactarla de nuevo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from aida.arms.health.sections import SectionSpec
from aida.engine.ports import CognitionProvider

_RAIZ = Path(__file__).resolve().parents[4]
_NOTE_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "note.yaml"

_CLAVES_OBLIGATORIAS = (
    "system_prompt",
    "provider",
    "provider_model",
    "field_heading",
    "provenance_template",
    "fallback_notice",
)


class ClinicalNote(BaseModel):
    """La nota por campos, tal como la devuelve el modelo.

    `extra="forbid"` y las claves validadas contra las secciones declaradas: un
    campo inventado por el modelo no se ignora en silencio —se rechaza—, porque
    un apartado que nadie declaró en el pack es un apartado que nadie va a leer
    en la historia clínica.
    """

    model_config = ConfigDict(extra="forbid")

    fields: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NoteSettings:
    system_prompt: str
    provider: str
    provider_model: str
    field_heading: str
    provenance_template: str
    fallback_notice: str


def load_note_settings(path: Path | None = None) -> NoteSettings:
    """Carga y valida. Revienta con la clave que falta: una nota a medias es
    peor que ninguna, porque se descubre con la consulta ya terminada."""
    ruta = path if path is not None else _NOTE_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ruta} debe ser un mapeo YAML")
    faltan = [
        clave
        for clave in _CLAVES_OBLIGATORIAS
        if clave not in raw or not isinstance(raw[clave], str)
    ]
    if faltan:
        raise ValueError(f"faltan claves en {ruta}: {faltan}")
    return NoteSettings(
        system_prompt=raw["system_prompt"],
        provider=raw["provider"],
        provider_model=raw["provider_model"],
        field_heading=raw["field_heading"],
        provenance_template=raw["provenance_template"],
        fallback_notice=raw["fallback_notice"],
    )


@dataclass(frozen=True, slots=True)
class RenderedNote:
    """La nota ya resuelta: qué campo lleva qué texto, en el orden del catálogo,
    y si la redactó el modelo o es el respaldo literal. Lo segundo importa: quien
    la lea tiene que saber si está leyendo prosa o frases crudas."""

    fields: tuple[tuple[SectionSpec, str], ...]
    redacted: bool
    provider: str | None = None
    """Quién la redactó y con qué modelo, como `"openai/gpt-4.1-mini"`. `None`
    cuando es el respaldo literal. Lo lee el export y, en la unidad 73, la
    pantalla: una nota que acaba en una historia clínica tiene que poder decir
    qué la escribió."""


def _literal_fallback(
    in_note_sections: tuple[tuple[SectionSpec, tuple[str, ...]], ...],
) -> RenderedNote:
    return RenderedNote(
        fields=tuple((spec, "\n".join(texts)) for spec, texts in in_note_sections),
        redacted=False,
        provider=None,
    )


def build_note(
    *,
    grouped_sections: tuple[tuple[SectionSpec, tuple[str, ...]], ...],
    conversation: tuple[str, ...] = (),
    settings: NoteSettings,
    providers: tuple[tuple[str, CognitionProvider], ...] = (),
    on_fallback: Callable[[str, str], None] | None = None,
) -> RenderedNote:
    """Los hechos de cada apartado, redactados.

    El briefing lleva los hechos de las secciones con `in_note`, cada uno bajo
    su apartado, y después la conversación entera, en orden. Ni la historia
    previa ni los códigos: lo que el modelo no recibe no lo puede colar en la
    nota. La conversación sí entra, porque todo lo que contiene se dijo, y sin
    ella la nota omitía: «Sí, a la penicilina.» llegaba sin la pregunta que la
    hace una alergia, y lo que la segmentación mandaba a la reserva no llegaba.

    Recibe los candidatos en vez de un proveedor ya compuesto, y los recorre en
    orden hasta que uno devuelve algo válido. Así la procedencia dice **quién
    respondió** y no quién estaba declarado: con un `FallbackCognition` delante, la
    nota afirmaba «redactado por openai» mientras la había escrito el modelo local
    —medido—. Es el mismo movimiento que resolvió `I-138` en la caché: llevar la
    composición al sitio que necesita saber.

    Cada nombre es el que aparecerá en la nota exportada, así que incluye la versión
    del modelo: `"openai/gpt-4.1-mini"`. Cuando un candidato falla y el siguiente
    responde, `on_fallback` se invoca con el nombre del que falló y el del que
    respondió — quien llama decide si eso se avisa y cómo.

    Sin candidatos, si todos fallan, si ninguno valida o si alguno devuelve un
    apartado que el pack no declara, se devuelve el respaldo literal con
    `redacted=False`. Una consulta no se cae porque la prosa falle.
    """
    in_note_sections = tuple(
        (spec, texts) for spec, texts in grouped_sections if spec.in_note and texts
    )
    if not providers or not in_note_sections:
        return _literal_fallback(in_note_sections)

    identifiers = ", ".join(spec.section_id for spec, _texts in in_note_sections)
    briefing = (
        settings.system_prompt
        + "\n\nResponde con un objeto JSON que tenga EXCLUSIVAMENTE la clave "
        + '"fields", sin ninguna otra clave de nivel superior. Dentro de '
        + '"fields", una clave JSON por cada apartado, usando exactamente estos '
        + f"identificadores y ninguno más: {identifiers}.\n\n"
        + "\n\n".join(
            f"{spec.section_id} ({spec.label}):\n" + "\n".join(texts)
            for spec, texts in in_note_sections
        )
    )
    lineas = tuple(linea for linea in conversation if linea.strip())
    if lineas:
        briefing += "\n\nConversación completa, en orden:\n" + "\n".join(lineas)
    valid_ids = {spec.section_id for spec, _texts in in_note_sections}

    for index, (name, candidate) in enumerate(providers):
        try:
            result: object | None = candidate.think(briefing, ClinicalNote)
        except Exception:
            result = None
        if not isinstance(result, ClinicalNote):
            continue
        if any(section_id not in valid_ids for section_id in result.fields):
            continue

        rendered = tuple(
            (spec, result.fields[spec.section_id])
            for spec, _texts in in_note_sections
            if spec.section_id in result.fields
        )
        if index > 0 and on_fallback is not None:
            failed_name, _failed_candidate = providers[index - 1]
            on_fallback(failed_name, name)
        return RenderedNote(fields=rendered, redacted=True, provider=name)

    return _literal_fallback(in_note_sections)
