"""Cotejo determinista contra el catálogo CIE-10: los N mejores resultados
por encima de un umbral, no uno solo — a diferencia de `IntentClassifier`
(U6), una consulta puede tocar varios códigos relevantes a la vez.

El catálogo se pre-vectoriza una vez (`build_index`); son más de 14000
entradas y volver a tokenizar cada descripción en cada llamada sería
trabajo repetido sin ninguna razón.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import yaml

from aida.arms.health.codes import DiagnosticCode
from aida.resources import data_root
from aida.textmatch import cosine, fold, tokenize, vectorize

_RAIZ = data_root()
_COTEJO_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "matching.yaml"


@dataclass(frozen=True, slots=True)
class CatalogSpec:
    """Qué catálogo cargar, tal como lo declara el pack. Las rutas son
    relativas a la raíz del repositorio y se resuelven aquí, en un solo sitio:
    quien las use no vuelve a componerlas."""

    kind: str
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class MatchingSettings:
    threshold: float
    top_n: int
    stopwords: frozenset[str]
    absence_markers: frozenset[str]
    catalog: CatalogSpec


def load_matching_settings(path: Path | None = None) -> MatchingSettings:
    """Qué catálogo, con qué umbral, cuántos códigos y qué palabras se
    descartan. Los cuatro juntos porque se calibran juntos: cambiar las palabras
    vacías mueve la escala de las puntuaciones, y un umbral heredado de la
    escala anterior deja de significar lo que decía. Y el catálogo manda sobre
    los tres: 0.50 contra un catálogo no es 0.50 contra otro."""
    ruta = path if path is not None else _COTEJO_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ruta} debe ser un mapeo YAML")
    claves = ("threshold", "top_n", "stopwords", "absence_markers", "catalog")
    faltan = [k for k in claves if k not in raw]
    if faltan:
        raise ValueError(f"faltan claves en {ruta}: {faltan}")
    umbral, cuantos, vacias = raw["threshold"], raw["top_n"], raw["stopwords"]
    if not isinstance(umbral, int | float) or isinstance(umbral, bool):
        raise ValueError(f"'threshold' debe ser numérico en {ruta}")
    if not isinstance(cuantos, int) or isinstance(cuantos, bool) or cuantos <= 0:
        raise ValueError(f"'top_n' debe ser un entero positivo en {ruta}")
    if not isinstance(vacias, list) or not all(isinstance(v, str) for v in vacias):
        raise ValueError(f"'stopwords' debe ser una lista de cadenas en {ruta}")
    ausencia = raw["absence_markers"]
    if not isinstance(ausencia, list) or not all(isinstance(v, str) for v in ausencia):
        raise ValueError(f"'absence_markers' debe ser una lista de cadenas en {ruta}")
    return MatchingSettings(
        threshold=float(umbral),
        top_n=cuantos,
        # Plegadas, no solo en minúsculas: el índice y la consulta se comparan
        # plegados (`build_index`), así que una palabra vacía con tilde no
        # coincidiría nunca con el token que pretende descartar.
        stopwords=frozenset(fold(v) for v in vacias),
        # Plegados por lo mismo: se comparan contra la descripción ya plegada.
        absence_markers=frozenset(fold(v) for v in ausencia),
        catalog=_catalog_spec(raw["catalog"], ruta),
    )


def _catalog_spec(raw: object, ruta: Path) -> CatalogSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"'catalog' debe ser un mapeo en {ruta}")
    kind, files = raw.get("kind"), raw.get("files")
    if kind not in ("reference", "dual"):
        raise ValueError(f"'catalog.kind' debe ser 'reference' o 'dual' en {ruta}")
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise ValueError(f"'catalog.files' debe ser una lista de cadenas en {ruta}")
    esperados = 1 if kind == "reference" else 2
    if len(files) != esperados:
        raise ValueError(
            f"'catalog.kind: {kind}' exige {esperados} archivo(s) y trae {len(files)}, en {ruta}"
        )
    rutas = tuple(_RAIZ / f for f in files)
    faltan = [str(r) for r in rutas if not r.is_file()]
    if faltan:
        raise ValueError(f"el catálogo declarado en {ruta} no existe: {faltan}")
    return CatalogSpec(kind=kind, paths=rutas)


@dataclass(frozen=True, slots=True)
class CatalogIndex:
    """El catálogo pre-vectorizado **y** las palabras que se descartaron al
    hacerlo. Van juntas porque tienen que ser las mismas al indexar y al
    consultar: con dos listas distintas, la mitad de las coincidencias serían
    de mentira."""

    entries: tuple[tuple[DiagnosticCode, dict[str, int]], ...]
    stopwords: frozenset[str]


def build_index(
    catalog: tuple[DiagnosticCode, ...],
    *,
    stopwords: frozenset[str] = frozenset(),
    absence_markers: frozenset[str] = frozenset(),
) -> CatalogIndex:
    """Catálogo ya recibido ordenado por código (los dos cargadores lo
    garantizan) — el índice conserva ese orden, es lo que hace determinista el
    desempate en `match_codes`.

    Las descripciones se **pliegan** (minúsculas y sin tildes) antes de
    vectorizar, y la consulta se pliega igual en `match_codes`. Sin eso, el
    catálogo de una autoridad que publica en mayúsculas sin tildes no empareja
    ni una palabra acentuada: medido, «migraña» y «cólera» daban 0.00 y
    «hipertensión esencial» proponía un código sin relación.

    Y lo que la descripción **niega** no entra en el vector (`_asserted`). Sin
    eso, la variante sencilla de una categoría —«SIN MENCION DE COMPLICACION»—
    cargaba con las palabras de la complicación que niega, y el cotejo proponía
    la variante complicada: «diabetes mellitus» daba E140, CON COMA.
    """
    return CatalogIndex(
        entries=tuple(
            (
                entry,
                vectorize(
                    _asserted(fold(entry.description), absence_markers),
                    stopwords=stopwords,
                ),
            )
            for entry in catalog
        ),
        stopwords=stopwords,
    )


def _asserted(description: str, absence_markers: frozenset[str]) -> str:
    """La parte de la descripción que afirma algo: cada tramo entre comas, hasta
    su primer marcador de ausencia.

    Hasta la coma y no hasta el final: «… TIPO B, SIN AGENTE DELTA, CON COMA
    HEPATICO» sigue afirmando el coma hepático. Y el tramo entero, no solo la
    palabra que sigue al marcador: dejar «complicacion» en «SIN MENCION DE
    COMPLICACION» seguía hundiendo la variante sencilla."""
    if not absence_markers:
        return description
    kept: list[str] = []
    for clause in description.split(","):
        for token in tokenize(clause):
            if token in absence_markers:
                break
            kept.append(token)
    return " ".join(kept)


@dataclass(frozen=True, slots=True)
class CodableText:
    """Lo que se coteja y la frase de la que salió. Van juntas porque no
    siempre coinciden: un término del léxico clínico se consulta como texto
    propio («diabetes mellitus») y la frase que lo motivó es otra («sufro del
    azúcar»). Mostrar el término sintético como evidencia diría al profesional
    algo que nadie dijo."""

    query: str
    source: str


class CodeOrigin(StrEnum):
    """De dónde salió el código. Es la distinción entre un diagnóstico de hoy y
    uno que ya constaba, que `AIDA-health.md` pide en su Fase 4 y que el
    profesional necesita para saber qué está registrando de nuevo."""

    TODAY = "today"
    HISTORY = "history"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class MatchedCode:
    diagnostic: DiagnosticCode
    score: float
    origin: CodeOrigin = CodeOrigin.TODAY
    evidence: str = ""
    """La frase que produjo la mejor puntuación de este código. Vacía solo
    cuando el cotejo se hizo por la puerta de un texto suelto (`match_codes`),
    que no sabe de qué frase viene."""
    matched_words: frozenset[str] = frozenset()
    """Las palabras de la consulta que este código comparte con ella. Lo lee
    `match_codes_each` para distinguir un segundo diagnóstico de la misma frase
    de una alternativa al primero: la alternativa no explica ninguna palabra
    nueva."""


def match_codes_each(
    texts: tuple[CodableText, ...], index: CatalogIndex, *, threshold: float, top_n: int
) -> tuple[MatchedCode, ...]:
    """Coteja **cada texto por separado** y funde los resultados.

    Cotejar un texto largo contra descripciones de seis palabras hunde todas
    las puntuaciones: medido, la consulta entera no alcanzaba ningún umbral
    razonable mientras que sus frases sueltas sí. Un código que aparece en
    varias frases se queda con su mejor puntuación —y con la frase que la
    produjo, que es lo que el profesional necesita para juzgarlo—, y el orden
    final es el de siempre: puntuación descendente y, en empate, código
    ascendente.

    De cada texto entran el mejor código y los que explican alguna palabra que
    los anteriores no explicaban. Sin eso, los huecos libres se llenaban con
    otras lecturas de la misma frase: «hipotiroidismo» proponía también
    HIPOTIROIDISMO CONGENITO SIN BOCIO.
    """
    mejor: dict[str, MatchedCode] = {}
    primeros: list[str] = []
    for codable in texts:
        cubiertas: frozenset[str] = frozenset()
        for posicion, encontrado in enumerate(
            match_codes(codable.query, index, threshold=threshold, top_n=top_n)
        ):
            # Más allá del primero, un código de la misma frase solo cuenta si
            # explica alguna palabra que los anteriores no explicaban. Si no, es
            # otra lectura del mismo diagnóstico —«hipotiroidismo» proponía además
            # el CONGÉNITO—, no un diagnóstico más.
            if posicion > 0 and encontrado.matched_words <= cubiertas:
                continue
            cubiertas = cubiertas | encontrado.matched_words
            codigo = encontrado.diagnostic.code
            previo = mejor.get(codigo)
            if previo is None or encontrado.score > previo.score:
                mejor[codigo] = replace(encontrado, evidence=codable.source)
            if posicion == 0 and codigo not in primeros:
                primeros.append(codigo)

    def orden(codigo: str) -> tuple[float, str]:
        return (-mejor[codigo].score, codigo)

    # Primero, el mejor código de cada texto: sin esto, dos frases con
    # diagnósticos distintos compiten por los mismos huecos y la de puntuación
    # más baja desaparece, aunque sea el único código de ese hallazgo.
    elegidos = sorted(primeros, key=orden)[:top_n]
    resto = sorted((c for c in mejor if c not in elegidos), key=orden)
    elegidos.extend(resto[: top_n - len(elegidos)])
    return tuple(mejor[codigo] for codigo in sorted(elegidos, key=orden))


def match_codes(
    text: str, index: CatalogIndex, *, threshold: float, top_n: int
) -> tuple[MatchedCode, ...]:
    """Los `top_n` códigos con mayor coseno por encima de `threshold`,
    ordenados de mayor a menor puntuación y, en empate, por código
    ascendente (desempate por identificador, nunca solo por el float de la
    puntuación)."""
    query = vectorize(fold(text), stopwords=index.stopwords)
    if not query:
        return ()
    scored = [
        (round(cosine(query, vector), 4), entry, vector)
        for entry, vector in index.entries
    ]
    scored = [item for item in scored if item[0] >= threshold]
    scored.sort(key=lambda item: (-item[0], item[1].code))
    return tuple(
        MatchedCode(
            diagnostic=entry,
            score=score,
            matched_words=frozenset(query) & frozenset(vector),
        )
        for score, entry, vector in scored[:top_n]
    )


def match_by_origin(
    *,
    today: tuple[CodableText, ...],
    history: tuple[CodableText, ...],
    index: CatalogIndex,
    threshold: float,
    top_n: int,
) -> tuple[MatchedCode, ...]:
    """Coteja los dos orígenes **por separado** y los funde marcando cuál es
    cuál. Un código que sale en los dos es `BOTH`: es el caso interesante, un
    diagnóstico que ya constaba y del que hoy se vuelve a hablar.

    El tope se aplica **por origen**, no al total: con un tope global, los
    códigos de la historia desaparecen por puntuación y el profesional deja de
    ver lo que ya estaba registrado.
    """
    de_hoy = {
        m.diagnostic.code: m
        for m in match_codes_each(today, index, threshold=threshold, top_n=top_n)
    }
    de_historia = {
        m.diagnostic.code: m
        for m in match_codes_each(history, index, threshold=threshold, top_n=top_n)
    }
    fundidos: list[MatchedCode] = []
    for codigo, encontrado in de_hoy.items():
        if codigo in de_historia:
            mejor = max(encontrado, de_historia[codigo], key=lambda m: m.score)
            fundidos.append(replace(mejor, origin=CodeOrigin.BOTH))
        else:
            fundidos.append(replace(encontrado, origin=CodeOrigin.TODAY))
    for codigo, encontrado in de_historia.items():
        if codigo not in de_hoy:
            fundidos.append(replace(encontrado, origin=CodeOrigin.HISTORY))
    return tuple(sorted(fundidos, key=lambda m: (-m.score, m.diagnostic.code)))
