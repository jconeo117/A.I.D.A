"""El test ácido: el motor no puede saber en qué mundo está corriendo.

La disciplina personal no aguanta seis meses de desarrollo. Esto sí.
Si falla, alguien filtró ambientación dentro del motor y hay que sacarla al
content pack antes de seguir.

Qué escanea y qué no
--------------------
Escanea lo que puede alterar el comportamiento: **identificadores** y
**literales de cadena reales**. Ignora comentarios y docstrings (ARQUITECTURA §5
y 05-plan-u2 §10), porque un docstring que documenta por qué se descartó algo
(como `travel_discount` o `bond_rate` en `TemperamentKnob`) documenta el
agnosticismo en vez de romperlo. Por eso el análisis va por `ast` y no por
expresión regular sobre el fichero crudo.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

ENGINE_ROOT = Path(__file__).resolve().parents[1] / "src" / "aida"
TEMPLATE_PACK_PATH = Path(__file__).resolve().parents[1] / "packs" / "_plantilla" / "pack.yaml"

# Raíces y términos de vocabulario prohibido (con flexiones comunes y compuestos).
FORBIDDEN_BASE_PATTERNS = [
    # Oficina / asistente generalista
    r"correo(s)?", r"email(s)?", r"mensaje(s)?", r"jefe(s)?", r"reunion(es)?", r"reunión(es)?",
    r"factura(s)?", r"empleado(s)?", r"empresa(s)?", r"cliente(s)?", r"nomina(s)?", r"nómina(s)?",
    r"calendario(s)?",
    # Infraestructura y operaciones (dominio de packs/_plantilla)
    r"disco(s)?", r"servidor(es)?", r"certificado(s)?", r"replica(s)?", r"réplica(s)?",
    r"backup(s)?", r"incidencia(s)?", r"ticket(s)?", r"alerta(s)?", r"cpu", r"memoria(s)?",
    r"ram", r"cluster(s)?", r"clúster(es)?", r"telemetria(s)?", r"telemetría(s)?",
    # Fantasía y simulación social
    r"espada(s)?", r"magia(s)?", r"hechizo(s)?", r"mana", r"maná", r"elfo(s)?", r"orco(s)?",
    r"dragon(es)?", r"dragón(es)?", r"taberna(s)?", r"castillo(s)?", r"caballero(s)?",
    r"poción(es)?", r"pocion(es)?", r"runa(s)?",
    # Ciencia ficción
    r"nave(s)?", r"laser(es)?", r"láser(es)?", r"plasma(s)?", r"androide(s)?", r"hiperespacio",
    r"blaster(s)?", r"oxigeno", r"oxígeno", r"cyborg(s)?", r"warp",
    # Necesidades y recursos concretos de simulación
    r"hambre", r"sueno", r"sueño", r"cordura", r"oro", r"credito(s)?", r"crédito(s)?",
]

# Identificadores explícitos del pack de plantilla de infraestructura
TEMPLATE_PACK_EXPLICIT_TERMS = [
    # Necesidades
    "espacio_libre", "certificados_vigentes", "replicas_al_dia",
    # Acciones
    "purgar_temporales", "escalar_replicas", "rotar_certificados", "auditar_telemetria",
    # Franjas
    "horario_laboral", "ventana_nocturna",
    # Situaciones
    "pico_de_trafico",
    # Rasgos
    "prudente", "proactivo",
    # Agente y pack
    "centinela_01", "infraestructura_base",
    # Etiquetas
    "mantenimiento", "escalabilidad", "alta_disponibilidad", "seguridad", "critico", "diagnostico",
]


def _load_template_pack_identifiers() -> set[str]:
    """Extrae dinámicamente todos los identificadores declarados en el pack de plantilla."""
    if not TEMPLATE_PACK_PATH.exists():
        return set(TEMPLATE_PACK_EXPLICIT_TERMS)

    try:
        raw = yaml.safe_load(TEMPLATE_PACK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set(TEMPLATE_PACK_EXPLICIT_TERMS)

    ids: set[str] = set(TEMPLATE_PACK_EXPLICIT_TERMS)
    if isinstance(raw, dict):
        if "id" in raw:
            ids.add(str(raw["id"]))
        for section in ("needs", "traits", "phases", "situations", "actions", "agents"):
            for item in raw.get(section, []) or []:
                if isinstance(item, dict) and "id" in item:
                    ids.add(str(item["id"]))
                if section == "actions" and isinstance(item, dict) and "tags" in item:
                    for tag in item.get("tags") or []:
                        ids.add(str(tag))
    return ids


def _build_agnosticism_pattern() -> re.Pattern[str]:
    pack_terms = sorted(_load_template_pack_identifiers())
    escaped_pack = [re.escape(t) for t in pack_terms]
    all_patterns = FORBIDDEN_BASE_PATTERNS + escaped_pack
    # El delimitador usa exclusión alfanumérica en lugar de \b para que '_' cuente
    # como separador y atrape términos dentro de identificadores snake_case (e.g. escalar_replicas).
    regex_str = (
        r"(?<![a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ])(?:"
        + "|".join(all_patterns)
        + r")(?![a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ])"
    )
    return re.compile(regex_str, re.IGNORECASE)


PATTERN = _build_agnosticism_pattern()


def _engine_sources() -> list[Path]:
    return sorted(
        path
        for path in ENGINE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
        and "packs" not in path.parts
    )


def _prose_nodes(tree: ast.AST) -> set[int]:
    """Cadenas que son documentación, no datos."""
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _scannable(tree: ast.AST) -> list[tuple[int, str]]:
    """Devuelve (línea, texto) de todo lo que sí puede alterar comportamiento."""
    prose = _prose_nodes(tree)
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        match node:
            case ast.Constant(value=str() as text) if id(node) not in prose:
                found.append((node.lineno, text))
            case ast.Name(id=name):
                found.append((node.lineno, name))
            case ast.Attribute(attr=name):
                found.append((node.lineno, name))
            case ast.arg(arg=name):
                found.append((node.lineno, name))
            case ast.keyword(arg=str() as name):
                found.append((node.lineno, name))
            case ast.FunctionDef(name=name) | ast.ClassDef(name=name):
                found.append((node.lineno, name))

    return found


def test_engine_sources_exist() -> None:
    """Evita el falso verde: un test que no escanea nada siempre pasa."""
    assert _engine_sources(), f"no se encontró código de motor en {ENGINE_ROOT}"


@pytest.mark.parametrize("source", _engine_sources(), ids=lambda p: p.name)
def test_engine_has_no_setting_vocabulary(source: Path) -> None:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    offenders = [
        f"  {source.name}:{lineno} -> {match.group(0)!r} en {text!r}"
        for lineno, text in _scannable(tree)
        if (match := PATTERN.search(text))
    ]

    assert not offenders, (
        "Ambientación filtrada dentro del motor:\n"
        + "\n".join(offenders)
        + "\n\nEso pertenece a un content pack. El motor solo conoce primitivas "
        "abstractas (Need, Action, Tag, etc.)."
    )


def test_agnosticism_catches_all_template_pack_identifiers() -> None:
    """Demuestra que el guardián muerde: todos y cada uno de los identificadores
    del pack de plantilla son detectados si se inyectaran en código.
    """
    pack_ids = _load_template_pack_identifiers()
    assert len(pack_ids) >= 15, (
        f"Se esperaban al menos 15 identificadores de plantilla, hay {len(pack_ids)}"
    )

    escapados = []
    for identifier in sorted(pack_ids):
        if not PATTERN.search(identifier):
            escapados.append(identifier)

    assert not escapados, f"Los siguientes identificadores del pack no son detectados: {escapados}"


def test_agnosticism_catches_compounds_and_plurals() -> None:
    """Verifica que el patrón caza plurales ('certificados', 'replicas') y
    compuestos snake_case ('total_disco', 'escalar_replicas').
    """
    samples = [
        "total_disco",
        "escalar_replicas",
        "certificados",
        "replicas",
        "rotar_certificados",
        "purgar_temporales",
        "mis_servidores",
        "alerta_critica",
    ]
    for sample in samples:
        assert PATTERN.search(sample), (
            f"El detector no cazó el término compuesto/plural: {sample!r}"
        )

