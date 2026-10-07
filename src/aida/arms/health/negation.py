"""Qué parte de una frase está negada.

El cotejo compara palabras, y «no refiere dolor torácico» comparte todas sus
palabras útiles con «refiere dolor torácico»: sin esta pieza, negar un síntoma
propone su código igual que afirmarlo. Y no basta con marcar la frase entera:
«tiene dolor de cabeza pero no fiebre» tiene las dos cosas, y tirar la frase
completa cambiaría un código falso por un hallazgo perdido, que es peor.

Lo que abre y lo que cierra una negación vive en datos: cambia con el país y
con la especialidad. Doble negación no se modela —un cue dentro de un ámbito
ya abierto no reabre nada—: adivinar «no está sin fiebre» es peor que no
intentarlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from aida.resources import data_root
from aida.textmatch import fold, token_spans

_RAIZ = data_root()
_EXTRACCION_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "extraction.yaml"


@dataclass(frozen=True, slots=True)
class NegationCatalog:
    """Señales y cortadores como secuencias de palabras ya plegadas, ordenadas
    de más larga a más corta y, a igual longitud, alfabéticamente: el orden
    decide qué coincidencia gana, y dos corridas tienen que dar lo mismo."""

    cues: tuple[tuple[str, ...], ...] = ()
    stoppers: tuple[tuple[str, ...], ...] = ()


NO_NEGATION = NegationCatalog()
"""El catálogo vacío como valor por defecto compartido: el linter no admite
construir uno dentro de un argumento por defecto, y al ser inmutable compartir
la misma instancia no tiene riesgo."""


@dataclass(frozen=True, slots=True)
class PolarityParts:
    """Los dos tramos, **tal como se dijeron**: recortes del texto original, no
    una reconstrucción a partir de los tokens. La procedencia completa importa
    tanto como la categoría."""

    affirmed: str
    negated: str


def _sequences(raw: object, clave: str, ruta: Path) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(v, str) for v in raw):
        raise ValueError(f"'{clave}' debe ser una lista no vacía de cadenas, en {ruta}")
    secuencias = []
    for entrada in raw:
        palabras = tuple(fold(entrada).split())
        if not palabras:
            raise ValueError(f"'{clave}' no admite entradas vacías, en {ruta}")
        secuencias.append(palabras)
    return tuple(sorted(set(secuencias), key=lambda s: (-len(s), s)))


def load_negation(path: Path | None = None) -> NegationCatalog:
    ruta = path if path is not None else _EXTRACCION_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("negation"), dict):
        raise ValueError(f"falta 'negation' (mapeo) en {ruta}")
    bloque = raw["negation"]
    return NegationCatalog(
        cues=_sequences(bloque.get("cues"), "cues", ruta),
        stoppers=_sequences(bloque.get("stoppers"), "stoppers", ruta),
    )


def _matches_at(tokens: tuple[str, ...], position: int, sequence: tuple[str, ...]) -> bool:
    return tokens[position : position + len(sequence)] == sequence


def split_polarity(text: str, catalog: NegationCatalog) -> PolarityParts:
    """Puerta única de la polaridad. Todo el que necesite saber qué está negado
    —el cotejo y la extracción de campos— llama aquí: con dos implementaciones,
    el hallazgo diría una cosa y el código propuesto otra.

    El cortador cierra el ámbito y no entra en ninguno de los dos tramos: lo
    que importa es qué hechos quedan a cada lado, no reconstruir la frase.
    """
    spans = token_spans(text)
    if not spans:
        return PolarityParts(affirmed=text.strip(), negated="")
    plegados = tuple(fold(token) for token, _, _ in spans)

    negado_desde: int | None = None
    afirmados: list[tuple[int, int]] = []
    negados: list[tuple[int, int]] = []
    tramo_desde = 0
    posicion = 0
    while posicion < len(plegados):
        # Los cortadores primero: sin esta precedencia, «sin embargo» abriría
        # ámbito por su primera palabra.
        cortador = next(
            (s for s in catalog.stoppers if _matches_at(plegados, posicion, s)), None
        )
        if cortador is not None:
            destino = negados if negado_desde is not None else afirmados
            if posicion > tramo_desde:
                destino.append((spans[tramo_desde][1], spans[posicion - 1][2]))
            negado_desde = None
            posicion += len(cortador)
            tramo_desde = posicion
            continue
        if negado_desde is None:
            senal = next((s for s in catalog.cues if _matches_at(plegados, posicion, s)), None)
            if senal is not None:
                if posicion > tramo_desde:
                    afirmados.append((spans[tramo_desde][1], spans[posicion - 1][2]))
                negado_desde = posicion
                tramo_desde = posicion
                posicion += len(senal)
                continue
        posicion += 1

    if tramo_desde < len(spans):
        destino = negados if negado_desde is not None else afirmados
        destino.append((spans[tramo_desde][1], spans[len(spans) - 1][2]))

    return PolarityParts(
        affirmed=" ".join(text[inicio:fin].strip() for inicio, fin in afirmados).strip(),
        negated=" ".join(text[inicio:fin].strip() for inicio, fin in negados).strip(),
    )


@dataclass(frozen=True, slots=True)
class RetractionRule:
    """Cuándo una frase borra lo que dijo la anterior. `max_chars` es lo que
    separa una respuesta («No, eso no.») de una descripción que empieza negando
    («No, no me da vuelta, es más como que me voy a caer como flojera.»): la
    primera contesta a lo de antes, la segunda habla por sí misma."""

    max_chars: int


def load_retraction(path: Path | None = None) -> RetractionRule:
    ruta = path if path is not None else _EXTRACCION_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("retraction"), dict):
        raise ValueError(f"falta 'retraction' (mapeo) en {ruta}")
    tope = raw["retraction"].get("max_chars")
    if not isinstance(tope, int) or isinstance(tope, bool) or tope <= 0:
        raise ValueError(f"'retraction.max_chars' debe ser un entero positivo en {ruta}")
    return RetractionRule(max_chars=tope)


def retracted_replies(
    lines: tuple[str, ...], catalog: NegationCatalog, rule: RetractionRule
) -> frozenset[str]:
    """Las frases que la siguiente desmiente.

    En una consulta, el profesional enumera y el paciente descarta: «¿Dolor en
    el pecho, falta de aire, palpitaciones?» / «No, eso no». Medido sobre una
    consulta real, esa pareja proponía dos códigos con puntuación 1.00, porque
    la negación del plan 68 trabaja **dentro** de una frase y aquí está en la
    siguiente.

    El criterio no es «la anterior era una pregunta»: se probó y está refutado
    —el modelo no siempre escribe el signo, y de 59 frases codificables solo 9
    lo llevaban, ninguna la culpable—. Es la forma de la **respuesta**: entera
    negada y corta.

    Devuelve los textos, no sus posiciones: quien filtra los recibe ya agrupados
    por sección y ha perdido el orden. Consecuencia declarada: si la misma frase
    aparece dos veces idéntica en la consulta, se retractan las dos.
    """
    retractadas: list[str] = []
    for position, line in enumerate(lines):
        if position == 0:
            continue
        partes = split_polarity(line, catalog)
        if partes.negated and not partes.affirmed and len(line.strip()) <= rule.max_chars:
            previa = lines[position - 1].strip()
            if previa:
                retractadas.append(previa)
    return frozenset(retractadas)


@dataclass(frozen=True, slots=True)
class ThirdPartyCatalog:
    """Señales de que la frase habla de otra persona, como secuencias de
    palabras ya plegadas — mismo molde que `NegationCatalog.cues`, para que
    el mismo criterio de coincidencia valga en los dos sitios."""

    cues: tuple[tuple[str, ...], ...] = ()


def load_third_party(path: Path | None = None) -> ThirdPartyCatalog:
    ruta = path if path is not None else _EXTRACCION_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("third_party"), dict):
        raise ValueError(f"falta 'third_party' (mapeo) en {ruta}")
    bloque = raw["third_party"]
    return ThirdPartyCatalog(cues=_sequences(bloque.get("cues"), "cues", ruta))


def is_third_party(text: str, catalog: ThirdPartyCatalog) -> bool:
    """Si la frase habla de otra persona. Se compara con el texto plegado
    (`fold`), como el resto del vocabulario declarado, para que «Mi Padre» y «mi
    padre» sean lo mismo."""
    plegados = tuple(fold(token) for token, _, _ in token_spans(text))
    return any(
        _matches_at(plegados, posicion, senal)
        for senal in catalog.cues
        for posicion in range(len(plegados))
    )
