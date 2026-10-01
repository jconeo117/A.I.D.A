"""Calibrar la clasificación por apartados contra una referencia etiquetada a
mano: vuelca las frases de una consulta ya exportada para que alguien las
etiquete, y mide cualquier catálogo de secciones contra esa referencia.

No es parte de una consulta — son herramientas para ajustar `sections.yaml`
antes de confiar en lo que produce, y se miden una vez etiquetada la
referencia, no a ojo.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

from aida.arms.health.sections import SectionCatalog
from aida.arms.health.segment import build_section_index, classify_line


@dataclass(frozen=True, slots=True)
class LabelledPhrase:
    """Una frase, lo que el pack le asignó y lo que le corresponde. `expected`
    vacío significa «sin etiquetar todavía»: el reporte lo salta y lo dice, en
    vez de contarlo como acierto o como fallo."""

    text: str
    assigned: str
    expected: str


def dump_phrases(model_json: Path, destination: Path) -> int:
    """Vuelca las frases de un modelo clínico ya exportado a un CSV editable, con
    la sección que el pack les asignó y una columna vacía al lado.

    Del modelo exportado y no de la transcripción: así las frases son exactamente
    las que se clasificaron, sin volver a transcribir ni a partir —que daría otras
    y haría la referencia inservible—.

    Devuelve cuántas frases se volcaron.
    """
    raw = json.loads(model_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{model_json} debe contener un mapeo JSON (section_id -> hallazgos)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    volcadas = 0
    with destination.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["text", "assigned", "expected"])
        for section_id, hallazgos in raw.items():
            if not isinstance(hallazgos, list):
                raise ValueError(
                    f"'{section_id}' debe mapear a una lista de hallazgos en {model_json}"
                )
            for hallazgo in hallazgos:
                writer.writerow([hallazgo["text"], section_id, ""])
                volcadas += 1
    return volcadas


def _read_reference(path: Path) -> tuple[LabelledPhrase, ...]:
    with path.open(encoding="utf-8", newline="") as archivo:
        reader = csv.DictReader(archivo)
        return tuple(
            LabelledPhrase(
                text=fila["text"], assigned=fila["assigned"], expected=fila["expected"]
            )
            for fila in reader
        )


@dataclass(frozen=True, slots=True)
class SectionScore:
    """Las tres cosas que se miden, y la distinción es el criterio que
    gobierna esta unidad:

    - `correct`: la frase fue al apartado que le toca.
    - `contaminated`: fue a **otro apartado**. Es el error que ensucia la nota
      que el médico copia, y el que hay que bajar aunque cueste aciertos.
    - `lost`: debía ir a un apartado y fue a la reserva. Resta contenido, pero
      no mete nada falso en ningún sitio.

    Un reporte que solo dé el total de aciertos permite «mejorar» subiendo la
    contaminación, que es exactamente lo contrario de lo que se busca.
    """

    correct: int = 0
    contaminated: int = 0
    lost: int = 0


def _recortada(text: str, limite: int = 70) -> str:
    return text if len(text) <= limite else text[: limite - 3] + "..."


def section_report(reference: Path, catalog: SectionCatalog) -> str:
    """Clasifica cada frase de la referencia con el catálogo **que se le pase** y
    la compara con su etiqueta. Recibe el catálogo en vez de cargarlo: es lo que
    permite medir dos versiones de `sections.yaml` en la misma ejecución.

    Agrupa por `expected` y no por `assigned`: es lo único que cubre los tres
    casos a la vez —correcta, contaminada, y una frase de reserva que se coló
    en un apartado también cuenta como contaminación, no como pérdida."""
    frases = _read_reference(reference)
    index = build_section_index(catalog.sections)
    fallback_id = catalog.fallback.section_id
    ids_validos = {spec.section_id for spec in catalog.sections}

    puntuaciones: dict[str, SectionScore] = {
        spec.section_id: SectionScore() for spec in catalog.sections
    }
    fallos: list[tuple[str, str, str, float]] = []
    sin_etiquetar = 0
    aciertos = 0
    contaminadas = 0
    perdidas = 0

    for frase in frases:
        esperada = frase.expected.strip()
        if not esperada:
            sin_etiquetar += 1
            continue
        if esperada not in ids_validos:
            raise ValueError(
                f"'{esperada}' no es un section_id declarado en el catálogo "
                f"(frase: {_recortada(frase.text)!r})"
            )
        ganadora = classify_line(
            frase.text, index, threshold=catalog.threshold, fallback=catalog.fallback
        )
        asignada = ganadora.section.section_id
        if asignada == esperada:
            puntuaciones[esperada] = replace(
                puntuaciones[esperada], correct=puntuaciones[esperada].correct + 1
            )
            aciertos += 1
        elif asignada == fallback_id and esperada != fallback_id:
            puntuaciones[esperada] = replace(
                puntuaciones[esperada], lost=puntuaciones[esperada].lost + 1
            )
            perdidas += 1
            fallos.append((frase.text, asignada, esperada, ganadora.score))
        else:
            puntuaciones[esperada] = replace(
                puntuaciones[esperada], contaminated=puntuaciones[esperada].contaminated + 1
            )
            contaminadas += 1
            fallos.append((frase.text, asignada, esperada, ganadora.score))

    lineas = [
        f"Total: {len(frases)} frases, {aciertos} aciertos, {contaminadas} contaminadas, "
        f"{perdidas} perdidas, {sin_etiquetar} sin etiquetar",
        "",
        "Por apartado (correctas / contaminadas / perdidas):",
    ]
    for spec in catalog.sections:
        marcador = puntuaciones[spec.section_id]
        lineas.append(
            f"  {spec.section_id}: {marcador.correct} / {marcador.contaminated} / {marcador.lost}"
        )
    lineas.append("")
    lineas.append("Fallos (puntuación, asignada, esperada, frase):")
    if not fallos:
        lineas.append("  (ninguno)")
    for texto, asignada, esperada, puntuacion in fallos:
        lineas.append(
            f"  [{puntuacion:.4f}] asignada={asignada} esperada={esperada}: "
            f"{_recortada(texto)}"
        )
    return "\n".join(lineas)
