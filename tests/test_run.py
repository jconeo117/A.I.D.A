"""Pruebas del punto de entrada de la línea de comandos (`aida.run.main`).

Cubren la unidad U0 — Esqueleto del repositorio: la única superficie pública
en esta fase es `aida.__version__` y `aida.run.main`, así que estas pruebas
son, por ahora, toda la batería de la unidad.
"""

from __future__ import annotations

import io
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

import aida.engine.loop as loop
from aida import __version__
from aida.run import main

_TEMPLATE_PACK = Path(__file__).resolve().parents[1] / "packs" / "_plantilla" / "pack.yaml"


def test_version_imprime_la_version_declarada_y_devuelve_cero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main(["--version"])` imprime "aida <version>" y termina con éxito."""
    # Arrange: no hay estado previo que preparar, la versión es la del paquete.

    # Act
    codigo = main(["--version"])

    # Assert
    salida = capsys.readouterr().out
    assert codigo == 0
    assert salida == f"aida {__version__}\n"


def test_sin_argumentos_imprime_algo_y_devuelve_cero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sin argumentos, `main` informa de algo (no se comprueba el texto,
    solo que hay salida) y termina con éxito.
    """
    # Arrange: no hay estado previo.

    # Act
    codigo = main([])

    # Assert
    salida = capsys.readouterr().out
    assert codigo == 0
    assert salida.strip() != ""


def test_sin_argumento_explicito_se_comporta_como_lista_vacia(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main()` sin argumento delega en `sys.argv`, que aquí se sustituye por
    uno equivalente a no pasar ningún argumento de usuario.
    """
    # Arrange: se sustituye sys.argv para no depender de con qué argumentos
    # se invocó pytest realmente.
    monkeypatch.setattr("sys.argv", ["aida"])

    # Act
    codigo = main()

    # Assert
    salida = capsys.readouterr().out
    assert codigo == 0
    assert salida.strip() != ""


def test_version_del_paquete_coincide_con_la_de_pyproject() -> None:
    """La versión de `aida.__version__` es la única fuente de verdad y debe
    coincidir con la declarada en `pyproject.toml`.
    """
    # Arrange: se localiza pyproject.toml a partir de este mismo archivo, no
    # del directorio de trabajo actual, para que la prueba sea determinista
    # sin importar desde dónde se lance pytest.
    ruta_pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    contenido = tomllib.loads(ruta_pyproject.read_text(encoding="utf-8"))

    # Act
    version_declarada = contenido["project"]["version"]

    # Assert
    assert __version__ == version_declarada


def test_argumento_desconocido_termina_con_codigo_dos() -> None:
    """Una opción no reconocida por `argparse` levanta `SystemExit(2)` en
    lugar de tragarse el error en silencio.
    """
    # Arrange: una opción que el parser no declara.

    # Act / Assert
    with pytest.raises(SystemExit) as excinfo:
        main(["--opcion-inexistente"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# CLI --pack (Unidad U2)
# ---------------------------------------------------------------------------


def _pack_yaml_valido() -> str:
    return """
id: "pack_cli_test"
label: "Pack CLI válido"
needs:
  - id: "n1"
    label: "Necesidad 1"
    decay_per_tick: 0.01
actions:
  - id: "a1"
    label: "Acción 1"
    duration_ticks: 1
    reversibility: "reversible"
    satisfies:
      n1: 0.5
    effects:
      n1: 0.5
agents:
  - id: "ag1"
    label: "Agente 1"
"""


def test_run_pack_opcion_valida_imprime_resumen_y_devuelve_cero(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`main(["--pack", <ruta>])` con pack válido devuelve 0 e imprime el resumen."""
    # Arrange
    pack_file = tmp_path / "valido.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    # Act
    codigo = main(["--pack", str(pack_file)])

    # Assert
    captured = capsys.readouterr()
    assert codigo == 0
    assert "pack 'pack_cli_test'" in captured.out
    assert "needs: 1" in captured.out
    assert "actions: 1" in captured.out
    assert "agents: 1" in captured.out
    assert captured.err == ""


def test_run_pack_opcion_invalida_imprime_errores_en_stderr_y_devuelve_uno(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`main(["--pack", <ruta>])` con pack inválido devuelve 1 e imprime error por stderr."""
    # Arrange: pack con error de validación (efecto negativo sin coste declarado)
    invalido_yaml = """
id: "pack_invalido"
label: "Pack inválido"
needs:
  - id: "n1"
    label: "Necesidad 1"
    decay_per_tick: 0.01
actions:
  - id: "a1"
    label: "Acción 1"
    duration_ticks: 1
    reversibility: "reversible"
    effects:
      n1: -0.5
agents:
  - id: "ag1"
    label: "Agente 1"
"""
    pack_file = tmp_path / "invalido.yaml"
    pack_file.write_text(invalido_yaml, encoding="utf-8")

    # Act
    codigo = main(["--pack", str(pack_file)])

    # Assert
    captured = capsys.readouterr()
    assert codigo == 1
    assert captured.err != ""
    assert "no se pudo cargar" in captured.err or "inválido" in captured.err


def test_run_pack_archivo_inexistente_devuelve_uno_y_reporta_en_stderr(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`main(["--pack", <ruta_inexistente>])` devuelve 1 e informa por stderr."""
    # Arrange
    ruta_inexistente = tmp_path / "no_existe.yaml"

    # Act
    codigo = main(["--pack", str(ruta_inexistente)])

    # Assert
    captured = capsys.readouterr()
    assert codigo == 1
    assert "no se pudo cargar" in captured.err


def test_run_pack_no_invoca_sys_exit_directamente(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`main` no llama a `sys.exit` para --pack: retorna el código como entero."""
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    def mock_exit(code: int = 0) -> None:
        raise AssertionError(f"sys.exit({code}) fue llamado dentro de main()")

    monkeypatch.setattr(sys, "exit", mock_exit)

    # Tanto en éxito como en fallo no debe llamar a sys.exit
    assert main(["--pack", str(pack_file)]) == 0
    assert main(["--pack", str(tmp_path / "no_existe.yaml")]) == 1


def test_explain_applies_agent_tastes(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """§6.2: `--explain` utiliza `AgentState.from_spec` y aplica los gustos del agente (I-044).

    Con un gusto de 1.50 hacia `atender_b`, la deliberación mostrada en `--explain`
    refleja la utilidad modulada (+0.112500 frente a +0.075000) y margen (+0.037500).
    """
    pack_yaml = """id: "pack_gusto"
label: "Pack con gusto declarado"
needs:
  - id: "eje_a"
    label: "Eje A"
    decay_per_tick: 0.05
    initial: 0.5
    cap: 1.0
  - id: "eje_b"
    label: "Eje B"
    decay_per_tick: 0.05
    initial: 0.5
    cap: 1.0
actions:
  - id: "atender_a"
    label: "Atender A"
    duration_ticks: 1
    reversibility: "reversible"
    satisfies:
      eje_a: 0.30
    effects:
      eje_a: 0.30
  - id: "atender_b"
    label: "Atender B"
    duration_ticks: 1
    reversibility: "reversible"
    satisfies:
      eje_b: 0.30
    effects:
      eje_b: 0.30
agents:
  - id: "agente_01"
    label: "Agente 01"
    initial_needs:
      eje_a: 0.50
      eje_b: 0.50
    tastes:
      - target: "atender_b"
        multiplier: 1.50
tuning:
  candidate_pool: 2
  temperature: 0.01
"""
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(pack_yaml, encoding="utf-8")

    codigo = main(["--pack", str(pack_file), "--explain"])

    captured = capsys.readouterr()
    assert codigo == 0
    assert "atender_b" in captured.out
    assert "+0.112500" in captured.out
    assert "+0.037500" in captured.out


# ---------------------------------------------------------------------------
# CLI --converse (Unidad U7)
# ---------------------------------------------------------------------------


def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limpia variables de entorno de LLM para garantizar aislamiento total de red."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AIDA_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def test_converse_flujo_entrada_stdin_termina_con_cero_e_imprime_respuesta(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§6.4 Caso 1 / I-064 / I-066: --converse sin credenciales debe limpiar el entorno para
    no realizar llamadas remotas accidentales y anunciar 'sin voz' con transparencia.
    """
    _clean_llm_env(monkeypatch)
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    monkeypatch.setattr("sys.stdin", io.StringIO("hola\nsalir\n"))

    codigo = main(["--pack", str(pack_file), "--converse"])

    captured = capsys.readouterr()
    assert codigo == 0
    assert "sin voz" in captured.out
    assert "(voz: " not in captured.out
    assert "> " in captured.out
    assert captured.err == ""


def test_converse_sin_pack_muestra_uso_breve_y_termina_con_cero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§6.4 Caso 2: --converse sin --pack documenta el comportamiento actual
    mostrando el mensaje de uso breve y retornando código 0.
    """
    codigo = main(["--converse"])

    captured = capsys.readouterr()
    assert codigo == 0
    assert "aida: asistente determinista. Usa --version o --pack <ruta>" in captured.out


def test_converse_pack_sin_agentes_informa_en_stderr_y_devuelve_uno(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """§6.4 Caso 3: --converse sobre un pack sin agentes emite error controlado en stderr
    y devuelve código 1 (protege M12).
    """
    pack_yaml = """id: "pack_sin_agentes"
label: "Pack sin agentes"
needs:
  - id: "n1"
    label: "Necesidad 1"
    decay_per_tick: 0.01
actions:
  - id: "a1"
    label: "Acción 1"
    duration_ticks: 1
    reversibility: "reversible"
    satisfies:
      n1: 0.5
    effects:
      n1: 0.5
agents: []
"""
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(pack_yaml, encoding="utf-8")

    codigo = main(["--pack", str(pack_file), "--converse"])

    captured = capsys.readouterr()
    assert codigo == 1
    assert "El pack no declara ningún agente." in captured.err


def test_converse_prints_assistant_response_in_output(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-062 / V3: el bucle conversacional interactivo debe imprimir por stdout
    la respuesta redactada por el asistente tras procesar cada turno del usuario.
    """
    _clean_llm_env(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("hola\nsalir\n"))

    codigo = main(["--pack", str(_TEMPLATE_PACK), "--converse"])

    captured = capsys.readouterr()
    assert codigo == 0
    assert "Empezó a hacer:" in captured.out


def test_converse_llm_flag_takes_precedence_over_env_var(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-062 / V4: la opción explícita de CLI --llm debe tener prioridad estricta
    sobre la variable de entorno AIDA_LLM_PROVIDER al seleccionar el proveedor.
    """
    _clean_llm_env(monkeypatch)
    monkeypatch.setenv("AIDA_LLM_PROVIDER", "claude")

    recorded_providers: list[str] = []

    def _spy_build_provider(name: str) -> None:
        recorded_providers.append(name)
        return None

    monkeypatch.setattr("aida.run.build_provider", _spy_build_provider)
    monkeypatch.setattr("sys.stdin", io.StringIO("salir\n"))

    codigo = main(["--pack", str(_TEMPLATE_PACK), "--converse", "--llm", "ollama"])
    assert codigo == 0
    assert recorded_providers == ["ollama"]


def test_converse_uses_aida_llm_provider_env_when_flag_absent(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-062 / V5: en ausencia de bandera --llm, la variable AIDA_LLM_PROVIDER
    debe definir el proveedor; si tampoco está definida, el valor por defecto es gemini.
    """
    _clean_llm_env(monkeypatch)
    recorded_providers: list[str] = []

    def _spy_build_provider(name: str) -> None:
        recorded_providers.append(name)
        return None

    monkeypatch.setattr("aida.run.build_provider", _spy_build_provider)
    monkeypatch.setattr("sys.stdin", io.StringIO("salir\n"))

    # Con AIDA_LLM_PROVIDER=claude
    monkeypatch.setenv("AIDA_LLM_PROVIDER", "claude")
    codigo_claude = main(["--pack", str(_TEMPLATE_PACK), "--converse"])
    assert codigo_claude == 0
    assert recorded_providers == ["claude"]

    # Sin variable de entorno: valor por defecto "gemini"
    recorded_providers.clear()
    monkeypatch.delenv("AIDA_LLM_PROVIDER", raising=False)
    codigo_gemini = main(["--pack", str(_TEMPLATE_PACK), "--converse"])
    assert codigo_gemini == 0
    assert recorded_providers == ["gemini"]


def test_converse_built_provider_speaks_in_output(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-062 / V8: el proveedor retornado por build_provider debe ser el inyectado
    a la Session para que sus respuestas orales sean las emitidas por la CLI.
    """
    _clean_llm_env(monkeypatch)

    class _MockVoiceProvider:
        def think(self, briefing: str, schema: type) -> object | None:
            return cast(object, schema(text="VOZ DE PRUEBA"))

    monkeypatch.setattr("aida.run.build_provider", lambda name: _MockVoiceProvider())
    monkeypatch.setattr("sys.stdin", io.StringIO("hola\nsalir\n"))

    codigo = main(["--pack", str(_TEMPLATE_PACK), "--converse"])

    captured = capsys.readouterr()
    assert codigo == 0
    assert "VOZ DE PRUEBA" in captured.out


def test_converse_seed_propagates_to_loop_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-062 / V6: el parámetro --seed de la línea de comandos debe propagarse
    fielmente a Loop.start para garantizar la reproducibilidad determinista del bucle.
    """
    _clean_llm_env(monkeypatch)
    recorded_seeds: list[int] = []
    original_start = loop.Loop.start

    def _spy_start(
        pack: Any,
        spec: Any,
        *,
        seed: int,
        inbox_capacity: int,
        log_window: int | None = None,
        intent: Any = None,
    ) -> loop.Loop:
        recorded_seeds.append(seed)
        return original_start(
            pack,
            spec,
            seed=seed,
            inbox_capacity=inbox_capacity,
            log_window=log_window,
            intent=intent,
        )

    monkeypatch.setattr("aida.run.Loop.start", _spy_start)
    monkeypatch.setattr("sys.stdin", io.StringIO("salir\n"))

    codigo = main(["--pack", str(_TEMPLATE_PACK), "--converse", "--seed", "7"])
    assert codigo == 0
    assert recorded_seeds == [7]


def test_converse_classifier_is_wired_and_recognizes_intent(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-062 / V7: Loop.start debe recibir el clasificador de intenciones construido
    desde el pack para reconocer peticiones en lenguaje natural del usuario.
    """
    _clean_llm_env(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("libera espacio en disco\nsalir\n"))

    codigo = main(["--pack", str(_TEMPLATE_PACK), "--converse"])

    captured = capsys.readouterr()
    assert codigo == 0
    assert "Se reconoció la petición" in captured.out


def test_converse_banner_indicates_provider_availability(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-066: el banner inicial de --converse debe indicar con veracidad si el
    proveedor está activo ('(voz: ...)') o indisponible ('(sin voz: ... el proveedor ...)').
    """
    _clean_llm_env(monkeypatch)

    # Caso 1: proveedor es None -> 'sin voz', no '(voz: '
    monkeypatch.setattr("aida.run.build_provider", lambda name: None)
    monkeypatch.setattr("sys.stdin", io.StringIO("salir\n"))
    codigo_none = main(["--pack", str(_TEMPLATE_PACK), "--converse"])
    captured_none = capsys.readouterr()
    assert codigo_none == 0
    assert "sin voz" in captured_none.out
    assert "(voz: " not in captured_none.out

    # Caso 2: proveedor disponible -> '(voz: ', no 'sin voz'
    class _MockProvider:
        def think(self, briefing: str, schema: type) -> object | None:
            return None

    monkeypatch.setattr("aida.run.build_provider", lambda name: _MockProvider())
    monkeypatch.setattr("sys.stdin", io.StringIO("salir\n"))
    codigo_mock = main(["--pack", str(_TEMPLATE_PACK), "--converse"])
    captured_mock = capsys.readouterr()
    assert codigo_mock == 0
    assert "(voz: " in captured_mock.out
    assert "sin voz" not in captured_mock.out



