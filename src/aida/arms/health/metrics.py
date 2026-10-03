"""Medir la revisión sin guardar a nadie: por cada revisión, cuántos veredictos
hubo de cada clase, por tipo de ítem y por sección, y cuánto tardó. Sin
identificador del paciente y sin ningún texto — quien lo lee
(`aida --review-stats`) solo necesita conteos, y lo que no se guarda no hay que
cifrarlo.

Los conteos salen de los eventos `ITEM_REVIEWED` del registro, nunca de un
contador paralelo: una sola puerta. Si un veredicto no dejó evento, para la
medición no ocurrió.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from aida.arms.health.review import CODE_VERDICTS, FINDING_VERDICTS, ItemKind, Verdict
from aida.engine.events import Event, EventKind


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    completed: bool
    seconds: float
    findings: Mapping[str, int]
    codes: Mapping[str, int]
    sections: Mapping[str, Mapping[str, int]]
    code_origins: Mapping[str, Mapping[str, int]]
    fields: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    """Por campo de la nota, cuántas veces se entregó sin cambios y cuántas
    editado. Es la señal que la pantalla existe para medir."""

    def to_json(self) -> str:
        """Claves ordenadas a todos los niveles: dos registros iguales dan la
        misma línea, se construyan como se construyan."""
        return json.dumps(
            {
                "code_origins": {
                    o: dict(sorted(c.items())) for o, c in sorted(self.code_origins.items())
                },
                "codes": dict(sorted(self.codes.items())),
                "completed": self.completed,
                "fields": {
                    s: dict(sorted(c.items())) for s, c in sorted(self.fields.items())
                },
                "findings": dict(sorted(self.findings.items())),
                "seconds": self.seconds,
                "sections": {
                    s: dict(sorted(c.items())) for s, c in sorted(self.sections.items())
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> ReviewRecord:
        raw = json.loads(line)
        return cls(
            completed=bool(raw["completed"]),
            seconds=float(raw["seconds"]),
            findings={str(k): int(v) for k, v in sorted(raw["findings"].items())},
            codes={str(k): int(v) for k, v in sorted(raw["codes"].items())},
            sections={
                str(s): {str(k): int(v) for k, v in sorted(c.items())}
                for s, c in sorted(raw["sections"].items())
            },
            code_origins={
                str(o): {str(k): int(v) for k, v in sorted(c.items())}
                for o, c in sorted(raw.get("code_origins", {}).items())
            },
            fields={
                str(s): {str(k): int(v) for k, v in sorted(c.items())}
                for s, c in sorted(raw.get("fields", {}).items())
            },
        )


def review_record(events: Iterable[Event], *, completed: bool, seconds: float) -> ReviewRecord:
    """Solo mira `ITEM_REVIEWED`. Qué tramo del registro recibe —el de una
    sola revisión— lo decide quien llama."""
    findings: dict[str, int] = {}
    codes: dict[str, int] = {}
    sections: dict[str, dict[str, int]] = {}
    origenes: dict[str, dict[str, int]] = {}
    campos: dict[str, dict[str, int]] = {}
    for event in events:
        if event.kind is not EventKind.ITEM_REVIEWED:
            continue
        verdict = str(event.data["verdict"])
        item = event.data["item"]
        if item == ItemKind.CODE.value:
            codes[verdict] = codes.get(verdict, 0) + 1
            # El origen lo escribe `review.py`; si un evento viejo no lo trae,
            # se cuenta como de hoy, que es lo que significaba antes.
            origen = str(event.data.get("origin", "today"))
            por_origen = origenes.setdefault(origen, {})
            por_origen[verdict] = por_origen.get(verdict, 0) + 1
        elif item == ItemKind.FINDING.value:
            findings[verdict] = findings.get(verdict, 0) + 1
            section_id = str(event.data["ref"]).rpartition(":")[0]
            per_section = sections.setdefault(section_id, {})
            per_section[verdict] = per_section.get(verdict, 0) + 1
        elif item == ItemKind.FIELD.value:
            field_id = str(event.data["ref"])
            per_field = campos.setdefault(field_id, {})
            per_field[verdict] = per_field.get(verdict, 0) + 1
        else:
            raise ValueError(f"clase de ítem desconocida en ITEM_REVIEWED: {item!r}")
    return ReviewRecord(
        completed=completed,
        seconds=seconds,
        findings=findings,
        codes=codes,
        sections=sections,
        code_origins=origenes,
        fields=campos,
    )


class ReviewMetricsStore:
    """Un archivo JSONL, una línea por revisión. Solo se añade."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, record: ReviewRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")

    def read(self) -> tuple[ReviewRecord, ...]:
        """Vacío si el archivo no existe. Una línea ilegible revienta con su
        número: una métrica que se salta registros en silencio miente sin
        avisar."""
        if not self._path.exists():
            return ()
        records: list[ReviewRecord] = []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(ReviewRecord.from_json(line))
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"{self._path}:{number}: registro ilegible: {exc}") from exc
        return tuple(records)


_NAMES = {
    Verdict.CONFIRMED.value: "confirmados",
    Verdict.EDITED.value: "editados",
    Verdict.REJECTED.value: "rechazados",
}
_ORIGENES = {"today": "de esta consulta", "history": "solo en la historia", "both": "en ambas"}
_FINDING_ORDER = tuple(v.value for v in FINDING_VERDICTS)
_CODE_ORDER = tuple(v.value for v in CODE_VERDICTS)
_FIELD_ORDER = (Verdict.CONFIRMED.value, Verdict.EDITED.value)


def _add(maps: Iterable[Mapping[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for counts in maps:
        for verdict in sorted(counts):
            total[verdict] = total.get(verdict, 0) + counts[verdict]
    return total


def _line(counts: Mapping[str, int], order: tuple[str, ...]) -> str:
    total = sum(counts.get(v, 0) for v in order)
    if total == 0:
        return "0"
    shares = ", ".join(f"{_NAMES[v]} {100 * counts.get(v, 0) / total:.1f} %" for v in order)
    return f"{total} ({shares})"


def format_stats(records: tuple[ReviewRecord, ...]) -> str:
    """El informe que lee quien decide si una fase avanza (AIDA-health.md
    §24-25). El tiempo medio sale solo de las completas: una revisión
    abandonada no mide lo que cuesta revisar. Los conteos sí incluyen lo
    contestado en las cortadas."""
    if not records:
        return "sin revisiones registradas"
    completed = [r for r in records if r.completed]
    lines = [
        f"revisiones: {len(records)} "
        f"({len(completed)} completas, {len(records) - len(completed)} incompletas)"
    ]
    if completed:
        mean = sum(r.seconds for r in completed) / len(completed)
        lines.append(f"tiempo medio por revisión completa: {mean:.1f} s")
    lines.append("hallazgos: " + _line(_add(r.findings for r in records), _FINDING_ORDER))
    lines.append("códigos: " + _line(_add(r.codes for r in records), _CODE_ORDER))
    por_origen: dict[str, dict[str, int]] = {}
    for record in records:
        for origen in sorted(record.code_origins):
            por_origen[origen] = _add((por_origen.get(origen, {}), record.code_origins[origen]))
    if por_origen:
        lines.append("códigos por origen:")
        for origen in sorted(por_origen):
            etiqueta = _ORIGENES.get(origen, origen)
            lines.append(f"  {etiqueta}: " + _line(por_origen[origen], _CODE_ORDER))
    by_section: dict[str, dict[str, int]] = {}
    for record in records:
        for section_id in sorted(record.sections):
            by_section[section_id] = _add(
                (by_section.get(section_id, {}), record.sections[section_id])
            )
    if by_section:
        lines.append("por sección:")
        for section_id in sorted(by_section):
            lines.append(f"  {section_id}: " + _line(by_section[section_id], _FINDING_ORDER))
    by_field: dict[str, dict[str, int]] = {}
    for record in records:
        for field_id in sorted(record.fields):
            by_field[field_id] = _add((by_field.get(field_id, {}), record.fields[field_id]))
    if by_field:
        lines.append("campos de la nota:")
        for field_id in sorted(by_field):
            lines.append(f"  {field_id}: " + _line(by_field[field_id], _FIELD_ORDER))
    return "\n".join(lines)
