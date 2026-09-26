"""Cómo dice el paciente lo que la historia clínica escribe de otra forma.

El cotejo compara palabras, no significados: «sufro del azúcar» no llega a
«diabetes mellitus» por sí solo. Esta pieza traduce el habla coloquial a un
término clínico que **sí** se parece a una descripción del catálogo.

El término traducido se añade como **consulta aparte**, no dentro de la frase:
medido sobre el catálogo real, meterlo en la frase alarga el texto y hunde la
puntuación por debajo del umbral, mientras que consultarlo solo acierta.

El vocabulario vive en datos (`packs/<...>/lexicon.yaml`) porque cambia con el
país y con la especialidad, y porque el motor no sabe en qué dominio corre.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from aida.textmatch import fold

_RAIZ = Path(__file__).resolve().parents[4]
_LEXICO_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "lexicon.yaml"

Lexicon = tuple[tuple[str, str], ...]
"""Pares (habla del paciente, término clínico), **ordenados**: el orden decide
en qué orden se añaden las consultas, y dos corridas tienen que dar lo mismo."""


def load_lexicon(path: Path | None = None) -> Lexicon:
    """Carga y valida. Claves y valores, cadenas; nada más."""
    ruta = path if path is not None else _LEXICO_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "terms" not in raw:
        raise ValueError(f"falta 'terms' en {ruta}")
    terms = raw["terms"]
    if not isinstance(terms, dict) or not terms:
        raise ValueError(f"'terms' debe ser un mapeo no vacío en {ruta}")
    for coloquial, clinico in terms.items():
        if not isinstance(coloquial, str) or not isinstance(clinico, str):
            raise ValueError(f"'terms' solo admite cadenas, en {ruta}")
    return tuple(sorted((fold(k), v) for k, v in terms.items()))


def clinical_terms(text: str, lexicon: Lexicon) -> tuple[str, ...]:
    """Los términos clínicos que corresponden a lo dicho, sin repetir y en el
    orden del léxico. Vacío si no reconoce nada, que es lo normal en una frase
    que ya viene en lenguaje clínico."""
    plano = fold(text)
    encontrados: list[str] = []
    for coloquial, clinico in lexicon:
        # Con límites de palabra: sin ellos, «gonartrosis» disparaba «artrosis»
        # y cualquier palabra larga arrastraba términos que nadie dijo.
        if re.search(rf"\b{re.escape(coloquial)}\b", plano) and clinico not in encontrados:
            encontrados.append(clinico)
    return tuple(encontrados)
