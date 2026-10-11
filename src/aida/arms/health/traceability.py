"""«No consta» y trazabilidad (plan 84).

Tres cosas que la nota no puede hacer sola, porque las escribe un modelo de lenguaje y
el modelo solo tiene instrucciones, no reglas:

- **No dejar vacío un apartado crítico.** Vacío no dice si no se habló o si se olvidó
  copiar. El motor escribe «No consta», o que consta en la historia previa, con su
  fecha.
- **No afirmar una ausencia que nadie dijo.** «Niega alergias» es un dato clínico; si en
  la consulta no hay un hallazgo negado que lo respalde, se marca para que el
  profesional lo corrija o lo confirme. El motor no cambia el texto: lo señala.
- **Decir de dónde y de cuándo** viene lo que trae la historia previa.

Todo el vocabulario está en el pack (`traceability.yaml`): qué apartados son críticos,
qué frases son de ausencia, qué encabezados forman el perfil de la historia.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from aida.arms.health.history import HistoryLayout
from aida.arms.health.model import ClinicalFinding
from aida.arms.health.note import RenderedNote
from aida.arms.health.sections import SectionSpec
from aida.resources import data_root
from aida.sentences import split_sentences
from aida.textmatch import fold, tokenize

_RAIZ = data_root()
_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "traceability.yaml"


def _plano(texto: str) -> str:
    """Mismo criterio que `history`: mayúsculas y sin tildes, para comparar encabezados."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.upper()) if unicodedata.category(c) != "Mn"
    ).strip()


@dataclass(frozen=True, slots=True)
class TraceabilitySettings:
    critical: Mapping[str, tuple[str, ...]]
    """`section_id` del apartado crítico ⇒ encabezados de la historia que lo cubren."""
    not_recorded: str
    see_history: str
    not_recorded_badge: str
    absence_markers: tuple[str, ...]
    absence_flag: str
    absence_confirm: str
    absence_confirmed: str
    absences_pending_singular: str
    absences_pending_plural: str
    absences_unresolved: str
    date_prefix: str
    profile_title: str
    profile_title_undated: str
    history_date_suffix: str
    profile: tuple[tuple[str, str], ...]
    """(encabezado, etiqueta corta), en el orden en que se enseñan."""


def load_traceability(path: Path | None = None) -> TraceabilitySettings:
    """Revienta si falta algo: un vigilante a medias avisaría de menos sin decirlo."""
    ruta = path if path is not None else _POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ruta} debe ser un mapeo YAML")
    textos = (
        "not_recorded", "see_history", "not_recorded_badge", "absence_flag", "absence_confirm",
        "absence_confirmed", "absences_pending_singular", "absences_pending_plural",
        "absences_unresolved", "date_prefix", "profile_title", "profile_title_undated",
        "history_date_suffix",
    )
    faltan = [k for k in textos if not isinstance(raw.get(k), str) or not raw[k].strip()]
    if faltan:
        raise ValueError(f"faltan textos en {ruta}: {faltan}")
    criticos = raw.get("critical_sections")
    if not isinstance(criticos, dict) or not all(
        isinstance(v, list) and all(isinstance(h, str) for h in v) for v in criticos.values()
    ):
        raise ValueError(f"'critical_sections' debe mapear apartados a encabezados, en {ruta}")
    marcas = raw.get("absence_markers")
    if not isinstance(marcas, list) or not marcas or not all(isinstance(m, str) for m in marcas):
        raise ValueError(f"'absence_markers' debe ser una lista no vacía, en {ruta}")
    perfil = raw.get("profile")
    if not isinstance(perfil, list) or not all(
        isinstance(p, dict)
        and isinstance(p.get("heading"), str)
        and isinstance(p.get("label"), str)
        for p in perfil
    ):
        raise ValueError(f"'profile' debe ser una lista de encabezado y etiqueta, en {ruta}")
    return TraceabilitySettings(
        critical={str(k): tuple(_plano(h) for h in v) for k, v in sorted(criticos.items())},
        absence_markers=tuple(fold(m) for m in marcas),
        profile=tuple((_plano(p["heading"]), p["label"]) for p in perfil),
        **{k: str(raw[k]) for k in textos if k != "date_prefix"},
        date_prefix=_plano(raw["date_prefix"]),
    )


@dataclass(frozen=True, slots=True)
class HistoryProfile:
    date: str | None
    """La fecha de la historia, tal como viene impresa; `None` si no trae."""
    lines: Mapping[str, tuple[str, ...]]
    """Encabezado (plano) ⇒ sus líneas, para todo encabezado del formato que traiga algo."""
    items: tuple[tuple[str, tuple[str, ...]], ...]
    """(etiqueta, líneas) de los encabezados del perfil, en el orden del pack."""


def history_profile(
    text: str, layout: HistoryLayout, settings: TraceabilitySettings
) -> HistoryProfile:
    """Lee la historia por encabezados, como `codable_history_lines`, pero sin filtrar por
    codificable: aquí interesa todo lo que diga, con su fecha. La primera línea con el
    prefijo de fecha es la fecha de la historia."""
    fecha: str | None = None
    por_encabezado: dict[str, list[str]] = {}
    activo: str | None = None
    for cruda in text.splitlines():
        linea = cruda.strip()
        if not linea:
            continue
        plano = _plano(linea)
        if fecha is None and plano.startswith(settings.date_prefix):
            fecha = linea[len(settings.date_prefix):].strip(" :") or None
            continue
        if any(plano.startswith(p) for p in layout.ignore):
            continue
        encabezado = next((s.heading for s in layout.sections if plano.startswith(s.heading)), None)
        if encabezado is not None:
            activo = encabezado
            continue
        if activo is not None:
            por_encabezado.setdefault(activo, []).append(linea)
    lineas = {k: tuple(v) for k, v in sorted(por_encabezado.items())}
    items = tuple(
        (etiqueta, lineas[encabezado])
        for encabezado, etiqueta in settings.profile
        if lineas.get(encabezado)
    )
    return HistoryProfile(date=fecha, lines=lineas, items=items)


def complete_note(
    note: RenderedNote,
    *,
    sections: Sequence[SectionSpec],
    profile: HistoryProfile,
    settings: TraceabilitySettings,
) -> RenderedNote:
    """Ningún apartado crítico sale vacío. Si la consulta no dijo nada, el motor escribe
    «No consta», o que consta en la historia —con su fecha— si la historia trae algo
    bajo los encabezados de ese apartado. Lo que escribió el motor queda en
    `not_recorded`, para que la revisión lo diga."""
    textos = {spec.section_id: texto for spec, texto in note.fields}
    escritos: list[str] = []
    for spec in sections:
        encabezados = settings.critical.get(spec.section_id)
        if encabezados is None or textos.get(spec.section_id, "").strip():
            continue
        en_historia = any(profile.lines.get(h) for h in encabezados)
        textos[spec.section_id] = (
            settings.see_history.format(profile.date)
            if en_historia and profile.date
            else settings.not_recorded
        )
        escritos.append(spec.section_id)
    if not escritos:
        return note
    # El orden del catálogo, y sin perder ningún campo que ya trajera la nota aunque no
    # esté en `sections`.
    declarados = {s.section_id for s in sections}
    orden = list(sections) + [
        spec for spec, _texto in note.fields if spec.section_id not in declarados
    ]
    campos = tuple((spec, textos[spec.section_id]) for spec in orden if spec.section_id in textos)
    return replace(note, fields=campos, not_recorded=tuple(escritos))


@dataclass(frozen=True, slots=True)
class AbsenceFlag:
    id: str
    """Estable para la misma frase en el mismo apartado: con él se confirma."""
    section_id: str
    sentence: str


def _contenido(texto: str, markers: Sequence[str]) -> set[str]:
    """Las palabras con contenido, sin las de la propia marca de ausencia."""
    palabras = set(tokenize(fold(texto)))
    for marca in markers:
        palabras -= set(marca.split())
    return {p for p in palabras if len(p) > 3}


def absence_flags(
    note: RenderedNote,
    findings: Mapping[str, Sequence[ClinicalFinding]],
    settings: TraceabilitySettings,
) -> tuple[AbsenceFlag, ...]:
    """Las frases de ausencia de la nota que ningún hallazgo negado del mismo apartado
    respalda. Un respaldo comparte al menos una palabra con contenido con la frase: «Niega
    alergias» lo respalda «No, alergias no». Lo que escribió el motor no se vigila."""
    marcas: list[AbsenceFlag] = []
    for spec, texto in note.fields:
        if spec.section_id in note.not_recorded:
            continue
        negados = [
            _contenido(f.text, settings.absence_markers)
            for f in findings.get(spec.section_id, ())
            if f.negated
        ]
        for frase in split_sentences(texto, abbreviations=frozenset()):
            plegada = fold(frase)
            if not any(marca in plegada for marca in settings.absence_markers):
                continue
            propias = _contenido(frase, settings.absence_markers)
            if any(propias & negado for negado in negados):
                continue
            huella = hashlib.blake2b(
                f"{spec.section_id}\n{plegada}".encode(), digest_size=4
            ).hexdigest()
            marcas.append(AbsenceFlag(id=huella, section_id=spec.section_id, sentence=frase))
    return tuple(marcas)


def unresolved(
    flags: Sequence[AbsenceFlag], final: Mapping[str, str], confirmed: frozenset[str]
) -> tuple[AbsenceFlag, ...]:
    """Las marcas que siguen pendientes en lo que se va a guardar: ni confirmadas ni
    corregidas (la frase sigue en el texto final de su apartado)."""
    return tuple(
        f
        for f in flags
        if f.id not in confirmed and fold(f.sentence) in fold(final.get(f.section_id, ""))
    )


@dataclass(frozen=True, slots=True)
class ReviewTrace:
    """Lo que la revisión necesita de esta unidad, junto: las marcas, la historia con su
    fecha y los textos. Una sola cosa que pasar a quien pinta."""

    flags: tuple[AbsenceFlag, ...]
    profile: HistoryProfile
    settings: TraceabilitySettings
