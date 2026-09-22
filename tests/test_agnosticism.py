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
import tomllib
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
        "faster_whisper",
        "whisper",
        "ctranslate2",
        "onnxruntime",
        "tokenizers",
        "pytesseract",
        "PIL",
        "pypdfium2",
        "sounddevice",
        "pyannote",
        "torch",
    }
)

# Directorios donde SÍ está permitido lo anterior — la costura entre el motor
# determinista y el modelo de lenguaje. No existen todavía en esta unidad,
# pero `ENGINE_ROOT` cubre `src/aida` entero y U6/U8 los crearán ahí dentro.
EXEMPT_DIR_NAMES = frozenset({"cognition", "intent", "arms", "voice"})

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
            case ast.Call(
                func=ast.Attribute(attr="join"),
                args=[ast.List(elts=elts) | ast.Tuple(elts=elts)],
            ) if elts and all(
                isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                for elt in elts
            ):
                sep = ""
                if isinstance(node.func, ast.Attribute) and isinstance(
                    node.func.value, ast.Constant
                ) and isinstance(node.func.value.value, str):
                    sep = node.func.value.value
                joined = sep.join(
                    elt.value
                    for elt in elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                )
                found.append((node.lineno, joined))
            case ast.BinOp(
                left=ast.Constant(value=str() as left_str),
                op=ast.Add(),
                right=ast.Constant(value=str() as right_str),
            ):
                found.append((node.lineno, left_str + right_str))

    return found


def _detect_concatenated_literal_evasions(tree: ast.AST) -> list[tuple[int, str]]:
    """Detecta concatenación de literales de cadena para formar palabras (como
    `"".join(["c", "pu"])` o `join` con lista/tupla de `ast.Constant`, o suma
    binaria de constantes de cadena).

    Estas construcciones existen únicamente para eludir los guardianes
    sintácticos (vocabulario de ambientación y términos médicos).
    """
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            elts = node.args[0].elts
            if elts and all(
                isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                for elt in elts
            ):
                sep = ""
                if isinstance(node.func.value, ast.Constant) and isinstance(
                    node.func.value.value, str
                ):
                    sep = node.func.value.value
                joined = sep.join(
                    elt.value
                    for elt in elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                )
                offenders.append((node.lineno, joined))
        elif (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)
        ):
            offenders.append((node.lineno, node.left.value + node.right.value))
    return offenders


def _extract_dynamic_import_root(node: ast.Call) -> str | None:
    """Extrae la raíz del módulo de llamadas a importlib.import_module o __import__.

    Cubre argumentos constantes posicionales o por clave (name=...).
    """
    target: str | None = None

    # 1. __import__("...") o __import__(name="...")
    if isinstance(node.func, ast.Name) and node.func.id == "__import__":
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            target = node.args[0].value
        else:
            for kw in node.keywords:
                if (
                    kw.arg == "name"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    target = kw.value.value
                    break

    # 2. importlib.import_module("...") o import_module("...")
    is_import_module = (
        (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
        or (isinstance(node.func, ast.Name) and node.func.id == "import_module")
    )
    if is_import_module:
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            target = node.args[0].value
        else:
            for kw in node.keywords:
                if (
                    kw.arg == "name"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    target = kw.value.value
                    break

    if target is not None:
        return target.lstrip(".").split(".")[0]
    return None


def _imported_module_roots(tree: ast.AST) -> list[tuple[int, str]]:
    """(línea, raíz del módulo) de cada `import` estático o dinámico — para el guardián de G4.

    Solo la raíz importa: `import openai.types`, `from openai import X`,
    `importlib.import_module("openai.types")` y `__import__("openai")`
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
        elif isinstance(node, ast.Call):
            root = _extract_dynamic_import_root(node)
            if root:
                found.append((node.lineno, root))
    return found


def _is_exempt(path: Path, root: Path = ENGINE_ROOT) -> bool:
    """La exención se decide sobre la ruta DENTRO de `src/aida`, y solo por su
    primer directorio. Sobre la ruta absoluta, un repositorio clonado bajo una
    carpeta llamada `arms` apagaba G4 entero sin que nada fallara (I-113).
    Fuera del árbol revienta: una ruta que no es del motor no puede estar ni
    exenta ni vigilada."""
    relative = path.resolve().relative_to(root.resolve())
    return len(relative.parts) > 1 and relative.parts[0] in EXEMPT_DIR_NAMES


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


@pytest.mark.parametrize("source", _engine_sources(), ids=lambda p: p.name)
def test_engine_has_no_concatenated_literal_evasions(source: Path) -> None:
    """I-092 / §4.2: Ningún módulo de `src/aida/` elude los guardianes
    mediante la concatenación de literales de cadena (como `"".join(["c", "pu"])`
    o `join` con lista de `ast.Constant`).
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    offenders = [
        f"  {source.name}:{lineno} -> concatenación elusiva evaluada como {text!r}"
        for lineno, text in _detect_concatenated_literal_evasions(tree)
    ]
    assert not offenders, (
        f"Elusión de guardián detectada en {source.name}:\n"
        + "\n".join(offenders)
        + "\n\nLas palabras deben escribirse completas y en datos si son de dominio, "
        "nunca fragmentadas en código para esconderlas del análisis sintáctico."
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


def test_scannable_reconstituye_literales_partidos_con_join() -> None:
    """Demuestra que la fragmentación de cadenas con join no burla el escaneo."""
    code = 'device = "".join(["c", "pu"])'
    tree = ast.parse(code)
    textos = [text for _, text in _scannable(tree)]
    assert "cpu" in textos
    assert any(PATTERN.search(t) for t in textos)


def test_detecta_concatenacion_de_literales_elusiva() -> None:
    """Demuestra que _detect_concatenated_literal_evasions caza la elusión."""
    code = 'device = "".join(["c", "pu"])'
    tree = ast.parse(code)
    offenders = _detect_concatenated_literal_evasions(tree)
    assert offenders == [(1, "cpu")]


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
    assert _is_exempt(ENGINE_ROOT / "cognition" / "redactor.py")
    assert _is_exempt(ENGINE_ROOT / "intent" / "clasificador.py")
    assert not _is_exempt(ENGINE_ROOT / "engine" / "pack.py")


def test_g4_guardian_exime_arms() -> None:
    """`arms/` es el punto de entrada para brazos de dominio:
    `EXEMPT_DIR_NAMES` lo incluye y `_is_exempt` lo reconoce para cualquier brazo.
    """
    assert "arms" in EXEMPT_DIR_NAMES
    assert "consult" not in EXEMPT_DIR_NAMES
    assert _is_exempt(ENGINE_ROOT / "arms" / "health" / "capture.py")
    assert _is_exempt(ENGINE_ROOT / "arms" / "accounting" / "algo.py") is True
    assert not _is_exempt(ENGINE_ROOT / "textmatch.py")


def test_g4_guardian_exime_voice() -> None:
    """`voice/` es la costura declarada para captura y escucha de audio local:
    `EXEMPT_DIR_NAMES` lo incluye y `_is_exempt` lo reconoce.
    """
    assert "voice" in EXEMPT_DIR_NAMES
    assert _is_exempt(ENGINE_ROOT / "voice" / "listen.py")
    assert _is_exempt(ENGINE_ROOT / "voice" / "commands.py")
    assert not _is_exempt(ENGINE_ROOT / "textmatch.py")


def test_is_exempt_ignora_carpetas_por_encima_de_la_raiz(tmp_path: Path) -> None:
    """I-113: un nombre exento por encima de `src/aida` no exime nada."""
    root = tmp_path / "arms" / "src" / "aida"
    assert not _is_exempt(root / "engine" / "loop.py", root)
    assert _is_exempt(root / "cognition" / "x.py", root)


def test_is_exempt_solo_mira_el_primer_directorio() -> None:
    """I-113: un subdirectorio con nombre exento dentro del motor no lo exime."""
    assert not _is_exempt(ENGINE_ROOT / "engine" / "voice" / "x.py")
    assert not _is_exempt(ENGINE_ROOT / "voice.py")


def test_is_exempt_fuera_del_arbol_revienta(tmp_path: Path) -> None:
    """I-113: fuera de `src/aida` no hay respuesta posible."""
    with pytest.raises(ValueError):
        _is_exempt(tmp_path / "cognition" / "x.py")


def test_g4_parametrizado_no_queda_vacio() -> None:
    """I-113: el síntoma medido fue un parametrizado vacío que pasa en verde."""
    vigilados = [p for p in _engine_sources() if not _is_exempt(p)]
    assert ENGINE_ROOT / "engine" / "loop.py" in vigilados


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


# ---------------------------------------------------------------------------
# I-072 — G4 debe morder ante los runtimes de modelo traídos por U9 (§4.1)
# ---------------------------------------------------------------------------

NEW_RUNTIME_MODEL_IMPORT_ROOTS = (
    "faster_whisper",
    "whisper",
    "ctranslate2",
    "onnxruntime",
    "tokenizers",
    "pytesseract",
    "PIL",
    "pypdfium2",
    "sounddevice",
    "pyannote",
    "torch",
)


def test_g4_guardian_catches_runtime_model_imports(tmp_path: Path) -> None:
    """I-072 (§4.1): Verifica que el guardián G4 muerde ante los runtimes y librerías
    de modelo traídos por U9 (faster_whisper, whisper, ctranslate2, onnxruntime,
    tokenizers, pytesseract, PIL).

    Tanto la forma 'import root' como 'from root import X' deben ser marcadas
    como ofensores en módulos no exentos del motor.
    """
    for root in sorted(NEW_RUNTIME_MODEL_IMPORT_ROOTS):
        for content in (f"import {root}\n", f"from {root} import SomeClass\n"):
            modulo = tmp_path / f"modulo_{root}.py"
            modulo.write_text(content, encoding="utf-8")

            offenders = _model_import_offenders(modulo)
            assert offenders, (
                f"El guardián G4 no detectó import prohibido de {root!r}: {content!r}"
            )


def test_g4_guardian_forbidden_set_contains_all_runtime_roots() -> None:
    """I-072 (§4.1): Verifica que todas las raíces de runtimes de modelo
    agregadas en U9 están registradas en FORBIDDEN_MODEL_IMPORT_ROOTS.
    Si se retira cualquiera de ellas, este test falla indicando el nombre faltante.
    """
    missing = sorted(set(NEW_RUNTIME_MODEL_IMPORT_ROOTS) - FORBIDDEN_MODEL_IMPORT_ROOTS)
    assert not missing, (
        f"Raíces de runtimes de modelo ausentes de FORBIDDEN_MODEL_IMPORT_ROOTS: {missing}"
    )


PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"

DIST_IMPORT_ROOTS = {
    "cryptography": "cryptography",
    "faster-whisper": "faster_whisper",
    "httpx": "httpx",
    "numpy": "numpy",
    "pyannote.audio": "pyannote",
    "pydantic": "pydantic",
    "pypdfium2": "pypdfium2",
    "pytesseract": "pytesseract",
    "pyyaml": "yaml",
    "sounddevice": "sounddevice",
}
"""Nombre de distribución -> raíz de import. Explícito: la regla general
(guion por guion bajo) falla justo en los casos que importan (`pyyaml`,
`pyannote.audio`), y adivinarla sería un falso verde."""

_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _dist_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement.strip())
    assert match is not None, f"requisito ilegible en pyproject.toml: {requirement!r}"
    return match.group(0).lower()


def _declared_dists() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """(`dependencies`, {extra: dependencias}), todo ordenado."""
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    core = tuple(sorted(_dist_name(r) for r in project["dependencies"]))
    extras = {
        extra: tuple(sorted(_dist_name(r) for r in reqs))
        for extra, reqs in sorted(project.get("optional-dependencies", {}).items())
    }
    return core, extras


def _src_import_roots() -> set[str]:
    roots: set[str] = set()
    for source in _engine_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        roots.update(root for _lineno, root in _imported_module_roots(tree))
    return roots


def test_toda_dependencia_declarada_tiene_mapeo_de_import() -> None:
    """I-089: sin mapeo no se puede comprobar nada — falla en vez de adivinar."""
    core, extras = _declared_dists()
    declared = set(core).union(*extras.values())
    missing = sorted(declared - DIST_IMPORT_ROOTS.keys())
    assert not missing, f"añade su raíz de import a DIST_IMPORT_ROOTS: {missing}"


def test_toda_dependencia_declarada_tiene_consumidor_en_src() -> None:
    """I-089: una dependencia que nadie importa es superficie muerta."""
    core, extras = _declared_dists()
    roots = _src_import_roots()
    declared = set(core).union(*extras.values())
    unused = sorted(d for d in declared if DIST_IMPORT_ROOTS[d] not in roots)
    assert not unused, f"declaradas en pyproject.toml y sin import en src/aida: {unused}"


def test_g4_vigila_toda_raiz_de_los_extras() -> None:
    """I-101: la cobertura de G4 queda atada a lo que el proyecto instala de
    verdad, no a una lista que la propia entrega puede recortar."""
    _core, extras = _declared_dists()
    extra_roots = {DIST_IMPORT_ROOTS[d] for reqs in extras.values() for d in reqs}
    assert not sorted(extra_roots - FORBIDDEN_MODEL_IMPORT_ROOTS)
    assert not sorted(extra_roots - set(NEW_RUNTIME_MODEL_IMPORT_ROOTS))



# ---------------------------------------------------------------------------
# I-095 — Detección de imports dinámicos en el guardián G4 (§4.2)
# ---------------------------------------------------------------------------


def test_g4_guardian_detecta_importlib_import_module_con_raiz_prohibida(tmp_path: Path) -> None:
    """I-095 (§4.2): El guardián G4 detecta llamadas a `importlib.import_module("...")`
    y `from importlib import import_module; import_module("...")` con argumento constante
    cuya raíz pertenezca a FORBIDDEN_MODEL_IMPORT_ROOTS.
    """
    variantes = (
        'import importlib\nimportlib.import_module("faster_whisper")\n',
        'from importlib import import_module\nimport_module("faster_whisper")\n',
        'import importlib\nimportlib.import_module("openai.types")\n',
        'import importlib\nimportlib.import_module(name="faster_whisper")\n',
    )
    for content in variantes:
        modulo = tmp_path / "modulo_mutado_importlib.py"
        modulo.write_text(content, encoding="utf-8")

        offenders = _model_import_offenders(modulo)
        assert offenders, f"El guardián de G4 no detectó llamada a import_module: {content!r}"


def test_g4_guardian_detecta_dunder_import_con_raiz_prohibida(tmp_path: Path) -> None:
    """I-095 (§4.2): El guardián G4 detecta llamadas a `__import__("...")`
    con argumento constante cuya raíz pertenezca a FORBIDDEN_MODEL_IMPORT_ROOTS.
    """
    variantes = (
        '__import__("faster_whisper")\n',
        '__import__("openai.types")\n',
        '__import__(name="faster_whisper")\n',
    )
    for content in variantes:
        modulo = tmp_path / "modulo_mutado_dunder.py"
        modulo.write_text(content, encoding="utf-8")

        offenders = _model_import_offenders(modulo)
        assert offenders, f"El guardián de G4 no detectó llamada a __import__: {content!r}"


def test_g4_guardian_permite_imports_dinamicos_no_prohibidos(tmp_path: Path) -> None:
    """I-095 (§4.2): Llamadas a importlib.import_module o __import__ de paquetes
    no prohibidos (como 'json' o 'math') no activan el guardián G4.
    """
    variantes = (
        'import importlib\nimportlib.import_module("json")\n',
        '__import__("math")\n',
    )
    for content in variantes:
        modulo = tmp_path / "modulo_legitimo_dinamico.py"
        modulo.write_text(content, encoding="utf-8")

        offenders = _model_import_offenders(modulo)
        assert not offenders, f"Falso positivo en import dinámico permitido: {content!r}"



# ---------------------------------------------------------------------------
# I-073 — Agnóstico de dominio: src/aida/arms/health/ no fija términos médicos en código
# ---------------------------------------------------------------------------

FORBIDDEN_MEDICAL_PATTERNS = [
    r"paciente(s)?",
    r"consulta(s)?\s+m[eé]dica(s)?",
    r"diagn[oó]stico(s)?",
    r"cefalea(s)?",
    r"historia(s)?\s+cl[ií]nica(s)?",
]

MEDICAL_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ])(?:"
    + "|".join(FORBIDDEN_MEDICAL_PATTERNS)
    + r")(?![a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ])",
    re.IGNORECASE,
)


def _health_sources() -> list[Path]:
    """`arms/health/`, más `cognition/` y `voice/` (I-088): los dos sitios
    donde viven textos que llegan al modelo o que reconocen los comandos del
    operador, que es justo por donde un vertical tiende a filtrar vocabulario.
    Y `run.py` (I-111)."""
    sources: list[Path] = []
    for parts in (("arms", "health"), ("cognition",), ("voice",)):
        directory = ENGINE_ROOT.joinpath(*parts)
        if not directory.exists():
            raise FileNotFoundError(f"Directorio no encontrado en {directory}")
        sources.extend(
            sorted(path for path in directory.rglob("*.py") if "__pycache__" not in path.parts)
        )
    run_file = ENGINE_ROOT / "run.py"
    if not run_file.exists():
        raise FileNotFoundError(f"Archivo run.py no encontrado en {run_file}")
    sources.append(run_file)
    return sources


def test_health_sources_exist() -> None:
    """Evita falsos verdes asegurando que se encuentran archivos en src/aida/arms/health."""
    sources = [p for p in _health_sources() if p != ENGINE_ROOT / "run.py"]
    assert sources, f"No se encontró código del brazo health en {ENGINE_ROOT / 'arms' / 'health'}"


def test_guardian_medico_escanea_run_py() -> None:
    """I-111: El guardián médico escanea explícitamente run.py."""
    nombres = {p.name for p in _health_sources()}
    assert "run.py" in nombres


def test_guardian_medico_escanea_cognition_y_voice() -> None:
    """I-088: el guardián médico recorre también `cognition/` y `voice/`."""
    rutas = {p.relative_to(ENGINE_ROOT).as_posix() for p in _health_sources()}
    assert "cognition/briefing.py" in rutas
    assert "voice/commands.py" in rutas



@pytest.mark.parametrize("source", _health_sources(), ids=lambda p: p.name)
def test_health_has_no_medical_domain_literals(source: Path) -> None:
    """I-073 (§4.2): El código del módulo `arms/health/` no debe contener vocabulario
    fijo del dominio médico ('paciente', 'consulta médica', 'diagnóstico',
    'cefalea', 'historia clínica').

    NOTA: Este guardián es de vocabulario médico para el brazo `health`, no
    genérico de todos los brazos. Cada futuro brazo (p. ej. accounting) requerirá
    su propio guardián con su lista de términos prohibidos correspondiente.

    El agnosticismo de AIDA exige que los términos de dominio sean inyectados
    como datos o configuración (e.g. vía `SummaryWording`), nunca codificados
    como literales en la tubería ni identificadores internos.
    Se utiliza una lista explícita de patrones médicos y se escanea vía `_scannable`
    para inspeccionar identificadores y literales de cadena reales excluyendo
    docstrings y comentarios.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    offenders = [
        f"  {source.name}:{lineno} -> {match.group(0)!r} en {text!r}"
        for lineno, text in _scannable(tree)
        if (match := MEDICAL_PATTERN.search(text))
    ]
    assert not offenders, (
        f"Literales de dominio médico encontrados en código de {source.name}:\n"
        + "\n".join(offenders)
        + "\n\nEl vocabulario médico debe ser inyectado (vía SummaryWording), no codificado."
    )


