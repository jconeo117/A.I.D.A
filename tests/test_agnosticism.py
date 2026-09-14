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
import sys
from pathlib import Path

import pytest
import yaml

ENGINE_ROOT = Path(__file__).resolve().parents[1] / "src" / "aida"
TEMPLATE_PACK_PATH = Path(__file__).resolve().parents[1] / "packs" / "_plantilla" / "pack.yaml"

# G4: ningún módulo del motor puede importar un cliente de modelo, de
# embeddings o de MCP. Solo la raíz del paquete: un "import openai.foo"
# también cuenta como "openai".
FORBIDDEN_MODEL_IMPORT_ROOTS = frozenset(
    {
        "openai",
        "anthropic",
        "litellm",
        "ollama",
        "transformers",
        "cohere",
        "mistralai",
        "huggingface_hub",
        "sentence_transformers",
        "langchain",
        "llama_index",
        "mcp",
        "modelcontextprotocol",
    }
)

# Directorios donde SÍ está permitido lo anterior — la costura entre el motor
# determinista y el modelo de lenguaje. No existen todavía en esta unidad,
# pero `ENGINE_ROOT` cubre `src/aida` entero y U6/U8 los crearán ahí dentro.
EXEMPT_DIR_NAMES = frozenset({"cognition", "intent"})

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
    """Extrae dinámicamente todos los identificadores declarados en el pack de plantilla.

    Si el YAML no parsea, la excepción se propaga (I-009): antes se
    capturaba con un `except Exception` amplio que degradaba en silencio a
    `TEMPLATE_PACK_EXPLICIT_TERMS`, una lista fija. Cualquier identificador
    añadido al pack después de esa caída quedaba sin cubrir sin que nada
    avisara. Un pack de plantilla que no parsea es un fallo en sí mismo, no
    una razón para que el guardián finja que todo sigue igual.
    """
    if not TEMPLATE_PACK_PATH.exists():
        return set(TEMPLATE_PACK_EXPLICIT_TERMS)

    raw = yaml.safe_load(TEMPLATE_PACK_PATH.read_text(encoding="utf-8"))

    ids: set[str] = set(TEMPLATE_PACK_EXPLICIT_TERMS)
    if isinstance(raw, dict):
        if "id" in raw:
            ids.add(str(raw["id"]))
        for section in (
            "needs", "traits", "phases", "situations", "actions", "agents",
            "goals", "intents",
        ):
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
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                found.append((node.lineno, name))

    return found


def _imported_module_roots(tree: ast.AST) -> list[tuple[int, str]]:
    """(línea, raíz del módulo) de cada `import` — para el guardián de G4.

    Solo la raíz importa: `import openai.types` y `from openai import X`
    dependen los dos del mismo paquete de terceros, así que ambos deben
    marcarse con el mismo nombre.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.append((node.lineno, node.module.split(".")[0]))
    return found


def _is_exempt(path: Path) -> bool:
    """`cognition/` e `intent/` son la costura declarada hacia el modelo."""
    return bool(EXEMPT_DIR_NAMES & set(path.parts))


def _model_import_offenders(source: Path) -> list[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    return [
        f"  {source.name}:{lineno} -> import {root!r}"
        for lineno, root in _imported_module_roots(tree)
        if root in FORBIDDEN_MODEL_IMPORT_ROOTS
    ]


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


@pytest.mark.parametrize(
    "source", [p for p in _engine_sources() if not _is_exempt(p)], ids=lambda p: p.name
)
def test_engine_has_no_model_client_imports(source: Path) -> None:
    """G4: el motor decide sin LLM. Ningún módulo del motor importa un
    cliente de modelo, de embeddings o de MCP — esa costura vive en
    `cognition/` o `intent/`, exentos aquí.
    """
    offenders = _model_import_offenders(source)
    assert not offenders, (
        "Import de un cliente de modelo/embeddings/MCP dentro del motor:\n"
        + "\n".join(offenders)
        + "\n\nG4: si el proveedor cae, AIDA sigue decidiendo; esa dependencia "
        "solo puede vivir en src/aida/cognition/ o src/aida/intent/."
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


def test_agnosticism_catches_all_ids_read_independently_from_yaml() -> None:
    """I-054: ancla independiente del cargador. Si `_load_template_pack_identifiers`
    pierde una sección, el patrón la pierde en los dos lados y el test anterior
    no lo notaría — este lee el YAML por su cuenta con `yaml.safe_load`.
    """
    raw = yaml.safe_load(TEMPLATE_PACK_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)

    known_ids = _load_template_pack_identifiers()
    missing: list[str] = []

    for section in sorted(raw):
        value = raw[section]
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and "id" in item:
                item_id = str(item["id"])
                if item_id not in known_ids:
                    missing.append(item_id)

    assert not missing, f"Ids del pack de plantilla ausentes de known_ids: {sorted(missing)}"


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


# ---------------------------------------------------------------------------
# I-005 — el analizador también cubre `async def`
# ---------------------------------------------------------------------------


def test_scannable_detecta_nombre_de_corrutina_con_termino_prohibido() -> None:
    """Una `async def` con vocabulario de dominio en el nombre no escapa al
    análisis (I-005).

    Hoy el motor es síncrono, pero `ENGINE_ROOT` cubre `src/aida` entero y a
    partir de U8/U9 ahí vivirán adaptadores asíncronos: la rendija se abriría
    sola si el `match` de `_scannable` solo cubriera `ast.FunctionDef`.
    """
    tree = ast.parse("async def servidor_temporal() -> int:\n    return 1\n")

    offenders = [(lineno, text) for lineno, text in _scannable(tree) if PATTERN.search(text)]

    assert offenders, "El nombre de una corrutina con término prohibido escapó al análisis"


def test_scannable_sigue_detectando_def_y_class_ordinarios() -> None:
    """No regresión: `def` y `class` normales se siguen detectando (I-005)."""
    tree_def = ast.parse("def servidor_temporal() -> int:\n    return 1\n")
    tree_class = ast.parse("class Servidor:\n    pass\n")

    assert any(PATTERN.search(text) for _, text in _scannable(tree_def))
    assert any(PATTERN.search(text) for _, text in _scannable(tree_class))


# ---------------------------------------------------------------------------
# I-006 — G4 no tenía guardián: imports de clientes de modelo/embeddings/MCP
# ---------------------------------------------------------------------------


def test_g4_guardian_mutacion_import_openai_ambas_formas(tmp_path: Path) -> None:
    """Mutación: inyectar `import openai` y `from openai import X` en un
    módulo del motor debe marcarse como ofensor en las dos formas (I-006).
    """
    for content in ("import openai\n", "from openai import OpenAI\n"):
        modulo = tmp_path / "modulo_mutado.py"
        modulo.write_text(content, encoding="utf-8")

        offenders = _model_import_offenders(modulo)

        assert offenders, f"El guardián de G4 no detectó: {content!r}"


def test_g4_guardian_no_marca_importes_ajenos_al_modelo(tmp_path: Path) -> None:
    """Un import ordinario (`pydantic`, `yaml`) no dispara el guardián de G4."""
    modulo = tmp_path / "modulo_normal.py"
    modulo.write_text("import yaml\nfrom pydantic import BaseModel\n", encoding="utf-8")

    assert _model_import_offenders(modulo) == []


def test_g4_guardian_exime_cognition_e_intent() -> None:
    """`cognition/` e `intent/` son la costura declarada: `_is_exempt` los reconoce."""
    assert _is_exempt(Path("src/aida/cognition/redactor.py"))
    assert _is_exempt(Path("src/aida/intent/clasificador.py"))
    assert not _is_exempt(Path("src/aida/engine/pack.py"))


# ---------------------------------------------------------------------------
# I-009 — el guardián no degrada en silencio si el pack no parsea
# ---------------------------------------------------------------------------


def test_load_template_pack_identifiers_propaga_error_de_parseo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el YAML del pack de plantilla no parsea, la excepción se propaga en
    vez de degradar en silencio a `TEMPLATE_PACK_EXPLICIT_TERMS` (I-009).
    """
    yaml_roto = tmp_path / "pack_roto.yaml"
    yaml_roto.write_text("id: [sin cerrar\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "TEMPLATE_PACK_PATH", yaml_roto)

    with pytest.raises(yaml.YAMLError):
        _load_template_pack_identifiers()

