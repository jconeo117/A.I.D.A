"""Pruebas del punto de entrada de la línea de comandos (`aida.run.main`).

Cubren la unidad U0 — Esqueleto del repositorio: la única superficie pública
en esta fase es `aida.__version__` y `aida.run.main`, así que estas pruebas
son, por ahora, toda la batería de la unidad.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tomllib
from dataclasses import replace
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

import aida.engine.loop as loop
from aida import __version__
from aida.cognition.briefing import SpokenReply, fallback_free_chat_reply
from aida.run import main
from aida.voice.diarize import TranscriptSegment

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


_AVISO_FALTA_KEY = "aida: aviso: falta AIDA_STT_API_KEY; se transcribe en local\n"


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


def test_converse_free_chat_sin_credenciales_devuelve_fallback_charla_libre(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§6.4 Caso 1 / M9: --converse --free-chat sin credenciales devuelve fallback_free_chat_reply,
    distinta de la respuesta operativa de fallback.
    """
    _clean_llm_env(monkeypatch)
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    monkeypatch.setattr("sys.stdin", io.StringIO("hola, qué tal\nsalir\n"))

    codigo = main(["--pack", str(pack_file), "--converse", "--free-chat"])

    captured = capsys.readouterr()
    assert codigo == 0
    frase_charla = fallback_free_chat_reply()
    assert frase_charla in captured.out
    assert "Empezó a hacer" not in captured.out
    assert "No hay novedades" not in captured.out


# ---------------------------------------------------------------------------
# CLI --consultation (Unidad U9 / 31-plan-consultorio-cli)
# ---------------------------------------------------------------------------


def test_consultation_sin_audio_ni_history_devuelve_uno_y_menciona_banderas_en_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§6.1 Caso 1: `--consultation` sin `--audio` ni `--history` debe devolver
    código 1 e informar por stderr mencionando ambas banderas.
    """
    # Sin ninguna de las dos banderas
    codigo = main(["--consultation"])
    captured = capsys.readouterr()
    assert codigo == 1
    assert "--audio" in captured.err
    assert "--history" in captured.err

    # Con solo --audio
    codigo_solo_audio = main(["--consultation", "--audio", "consulta.wav"])
    captured_audio = capsys.readouterr()
    assert codigo_solo_audio == 1
    assert "--history" in captured_audio.err

    # Con solo --history
    codigo_solo_history = main(["--consultation", "--history", "historia.png"])
    captured_history = capsys.readouterr()
    assert codigo_solo_history == 1
    assert "--audio" in captured_history.err


def test_consultation_sin_cache_key_devuelve_uno_y_menciona_variable_en_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§6.1 Caso 2: `--consultation --audio x --history y` sin AIDA_CONSULT_CACHE_KEY
    en el entorno devuelve código 1 y menciona la variable por stderr.
    """
    monkeypatch.delenv("AIDA_CONSULT_CACHE_KEY", raising=False)

    codigo = main(["--consultation", "--audio", "audio.wav", "--history", "historia.png"])

    captured = capsys.readouterr()
    assert codigo == 1
    assert "AIDA_CONSULT_CACHE_KEY" in captured.err


def test_consultation_clave_invalida_falla_sin_cargar_catalogo_ni_whisper(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§6.1 Caso 3: `--consultation --audio x --history y` con clave Fernet no válida
    devuelve código 1, menciona por stderr que la clave no es válida, y nunca
    intenta cargar el catálogo CIE-10 ni el modelo de Whisper.
    """
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", "no-es-una-clave-valida")

    catalogo_llamado = False

    def spy_load_catalog(*args: Any, **kwargs: Any) -> Any:
        nonlocal catalogo_llamado
        catalogo_llamado = True
        raise AssertionError("load_catalog no debe ser llamado con clave inválida")

    monkeypatch.setattr("aida.arms.health.codes.load_catalog", spy_load_catalog)

    whisper_llamado = False

    def spy_whisper_model(*args: Any, **kwargs: Any) -> Any:
        nonlocal whisper_llamado
        whisper_llamado = True
        raise AssertionError("WhisperModel no debe ser llamado con clave inválida")

    try:
        import importlib

        faster_whisper = importlib.import_module("faster_whisper")
        monkeypatch.setattr(faster_whisper, "WhisperModel", spy_whisper_model)
    except ImportError:
        pass

    codigo = main(["--consultation", "--audio", "audio.wav", "--history", "historia.png"])

    captured = capsys.readouterr()
    assert codigo == 1
    assert "AIDA_CONSULT_CACHE_KEY" in captured.err
    assert "no es una clave" in captured.err or "válida" in captured.err
    assert not catalogo_llamado
    assert not whisper_llamado


def test_consultation_diarize_sin_hf_token_devuelve_uno_y_menciona_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--consultation --diarize --audio x --history y sin AIDA_HF_TOKEN devuelve
    código 1 y mensaje 'aida: error: --diarize requiere AIDA_HF_TOKEN' por stderr.
    """
    monkeypatch.delenv("AIDA_HF_TOKEN", raising=False)

    codigo = main(
        ["--consultation", "--diarize", "--audio", "audio.wav", "--history", "historia.png"]
    )

    captured = capsys.readouterr()
    assert codigo == 1
    assert "aida: error: --diarize requiere AIDA_HF_TOKEN" in captured.err


def test_consultation_ejecucion_completa_con_extra_capture_y_clave_real(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§6.1 Caso 4: Con el extra `capture` instalado y clave Fernet real,
    una ejecución completa contra un audio y un documento de prueba da
    código 0 e imprime el resumen en Markdown por stdout.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Mock del backend de Whisper para aislar la prueba de descargas y latencia
    class _MockWhisperBackend:
        def __init__(
            self, text: str = "El paciente refiere cefalea intensa y dolor de cabeza."
        ) -> None:
            self.text = text

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            mock_seg = MagicMock()
            mock_seg.start = 0.0
            mock_seg.end = 1.0
            mock_seg.text = self.text
            mock_seg.no_speech_prob = 0.0
            return [mock_seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackend(),
    )

    # Mock de pytesseract para no depender de archivos traineddata del sistema
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa: cefalea tensional recurrente.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    codigo = main(
        [
            "--consultation",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_test_cli",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 0
    assert "- El paciente dijo:" in captured.out
    assert "- La historia registra:" in captured.out
    assert "El paciente refiere cefalea" in captured.out
    assert "cefalea tensional" in captured.out
    assert captured.err.replace(_AVISO_FALTA_KEY, "") == ""

    # Comprueba que el caché cifrado fue escrito en disco y es descifrable con la clave
    cache_file = tmp_path / ".cache" / "aida" / "consultas" / "paciente_test_cli.cache"
    assert cache_file.exists()
    contenido_descifrado = Fernet(key.encode()).decrypt(cache_file.read_bytes()).decode("utf-8")
    assert "- El paciente dijo:" in contenido_descifrado


def test_consultation_segment_ejecucion_completa_muestra_secciones_en_stdout(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§6.3 CLI: `--consultation --segment --audio x --history y` ejecuta la segmentación
    clínica y muestra las secciones en el resumen por stdout.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackendMultiSegment:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg1 = MagicMock()
            seg1.start = 0.0
            seg1.end = 2.0
            seg1.text = "consulta hoy por dolor de cabeza intenso"
            seg1.no_speech_prob = 0.0

            seg2 = MagicMock()
            seg2.start = 2.0
            seg2.end = 4.0
            seg2.text = "se receta ibuprofeno cada 8 horas"
            seg2.no_speech_prob = 0.0

            return [seg1, seg2], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackendMultiSegment(),
    )

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa: sin antecedentes de interés.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_test_cli_segment",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 0
    assert "### Motivo de consulta" in captured.out
    assert "consulta hoy por dolor de cabeza intenso" in captured.out
    assert "### Medicamentos" in captured.out
    assert "se receta ibuprofeno cada 8 horas" in captured.out
    assert "- El paciente dijo:" not in captured.out
    assert captured.err.replace(_AVISO_FALTA_KEY, "") == ""


def test_consultation_cli_segment_print_model(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§5.3 CLI: `--consultation --segment --print-model` con datos mockeados:
    el JSON impreso en stdout tras el resumen parsea con `json.loads` y coincide
    con `result.encounter.to_dict()`.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackendMultiSegment:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg1 = MagicMock()
            seg1.start = 0.0
            seg1.end = 2.0
            seg1.text = "consulta hoy por dolor de cabeza intenso"
            seg1.no_speech_prob = 0.0

            seg2 = MagicMock()
            seg2.start = 2.0
            seg2.end = 4.0
            seg2.text = "se receta ibuprofeno cada 8 horas"
            seg2.no_speech_prob = 0.0

            return [seg1, seg2], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackendMultiSegment(),
    )

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa: sin antecedentes de interés.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--print-model",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_test_cli_print_model",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 0
    assert captured.err.replace(_AVISO_FALTA_KEY, "") == ""

    # Extraer el JSON tras el resumen en stdout
    json_start = captured.out.find("{")
    assert json_start != -1, "No se encontró JSON en stdout"
    summary_part = captured.out[:json_start]
    json_part = captured.out[json_start:]

    # El resumen markdown está presente antes del JSON
    assert "### Motivo de consulta" in summary_part
    assert "### Medicamentos" in summary_part

    # El JSON parsea con json.loads y coincide con el encounter esperado
    parsed = json.loads(json_part)
    expected = {
        "motivo_consulta": [
            {
                "text": "consulta hoy por dolor de cabeza intenso",
                "duration": None,
                "laterality": None,
                "reported_by": None,
                "negated": None,
            }
        ],
        "medicamentos": [
            {
                "text": "se receta ibuprofeno cada 8 horas",
                "duration": None,
                "laterality": None,
                "reported_by": None,
                "negated": None,
            }
        ],
    }
    assert parsed == expected


def test_consultation_cli_print_model_sin_segment_avisa_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§5.3 CLI: `--consultation --print-model` sin `--segment`:
    termina con código 0, imprime el resumen en stdout y muestra
    "aida: aviso: --print-model requiere --segment" en stderr.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackend:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg = MagicMock()
            seg.start = 0.0
            seg.end = 2.0
            seg.text = "consulta por control general"
            seg.no_speech_prob = 0.0
            return [seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackend(),
    )

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia previa normal.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    codigo = main(
        [
            "--consultation",
            "--print-model",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_test_sin_segment",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 0
    # Resumen impreso en stdout
    assert "consulta por control general" in captured.out
    # Aviso en stderr
    assert "aida: aviso: --print-model requiere --segment" in captured.err


def test_consultation_cli_extract_sin_segment_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§7.6 CLI: `--consultation --extract` sin `--segment`:
    código 1, mensaje de error en stderr ("aida: error: --extract requiere --segment"),
    sin construir nada del extractor.
    """

    def _fail_build() -> Any:
        raise AssertionError("build_health_extractor no debió ser llamado")

    monkeypatch.setattr(
        "aida.arms.health.extract.build_health_extractor", _fail_build, raising=False
    )

    codigo = main(
        ["--consultation", "--extract", "--audio", "dummy.wav", "--history", "dummy.png"]
    )
    captured = capsys.readouterr()
    assert codigo == 1
    assert "aida: error: --extract requiere --segment" in captured.err


def test_consultation_cli_segment_extract_print_model(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§7.6 CLI: `--consultation --segment --extract --print-model` con datos mockeados:
    el JSON impreso trae los 4 campos con los valores extraídos según el audio/texto simulado.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackendExtract:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg1 = MagicMock()
            seg1.start = 0.0
            seg1.end = 2.0
            seg1.text = "consulta hoy por dolor en rodilla derecha desde hace dos semanas"
            seg1.no_speech_prob = 0.0

            seg2 = MagicMock()
            seg2.start = 2.0
            seg2.end = 4.0
            seg2.text = "se receta ibuprofeno cada 8 horas"
            seg2.no_speech_prob = 0.0

            return [seg1, seg2], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackendExtract(),
    )

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa: sin antecedentes de interés.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--extract",
            "--print-model",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_test_cli_extract_print_model",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 0
    assert captured.err.replace(_AVISO_FALTA_KEY, "") == ""

    # Extraer el JSON tras el resumen en stdout
    json_start = captured.out.find("{")
    assert json_start != -1, "No se encontró JSON en stdout"
    json_part = captured.out[json_start:]

    parsed = json.loads(json_part)
    expected = {
        "enfermedad_actual": [
            {
                "text": "consulta hoy por dolor en rodilla derecha desde hace dos semanas",
                "duration": "2 semanas",
                "laterality": "derecho",
                "reported_by": None,
                "negated": None,
            }
        ],
        "medicamentos": [
            {
                "text": "se receta ibuprofeno cada 8 horas",
                "duration": None,
                "laterality": None,
                "reported_by": None,
                "negated": None,
            }
        ],
    }
    assert parsed == expected


def test_consultation_con_proveedor_de_voz_emite_resumen_del_modelo(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """I-092 (§4.1): `--consultation` cablea el proveedor devuelto por
    `build_provider`. Si el proveedor devuelve un `SpokenReply`, su texto
    debe aparecer al inicio de la salida en stdout.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackend:
        def __init__(
            self, text: str = "El paciente refiere cefalea intensa y dolor de cabeza."
        ) -> None:
            self.text = text

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            mock_seg = MagicMock()
            mock_seg.start = 0.0
            mock_seg.end = 1.0
            mock_seg.text = self.text
            mock_seg.no_speech_prob = 0.0
            return [mock_seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackend(),
    )

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa: cefalea tensional recurrente.",
    )

    mock_provider = MagicMock()
    mock_provider.think.return_value = SpokenReply(text="VOZ DEL MODELO EN EL RESUMEN")
    monkeypatch.setattr("aida.run.build_provider", lambda *args, **kwargs: mock_provider)

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    codigo = main(
        [
            "--consultation",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_test_cli",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 0
    assert captured.out.startswith("VOZ DEL MODELO EN EL RESUMEN")
    assert captured.err.replace(_AVISO_FALTA_KEY, "") == ""


def test_default_summary_wording_desde_cwd_arbitrario_no_lanza_filenotfound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """I-094 (§2): `default_summary_wording()` carga el YAML de redacción por defecto
    sin importar el directorio de trabajo (cwd).
    """
    from aida.arms.health.wording import default_summary_wording

    monkeypatch.chdir(tmp_path)
    wording = default_summary_wording()
    assert wording.system_prompt
    assert wording.said_template


def test_consultation_con_cwd_arbitrario_clave_invalida_falla_por_clave_y_no_por_archivo(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """I-094 (§4.1): Desde un directorio de trabajo ajeno a la raíz del repositorio,
    `--consultation` con clave Fernet inválida debe fallar con código 1 indicando
    que la clave no es válida (y no con FileNotFoundError).
    """
    otro_directorio = tmp_path / "otro_cwd"
    otro_directorio.mkdir()
    monkeypatch.chdir(otro_directorio)

    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", "clave-invalida-no-fernet")

    codigo = main(
        [
            "--consultation",
            "--audio",
            "audio.wav",
            "--history",
            "historia.png",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 1
    assert "AIDA_CONSULT_CACHE_KEY no es una clave Fernet válida" in captured.err


def test_consultation_con_cwd_arbitrario_ejecucion_completa_tiene_exito(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """I-094 (§4.1): Desde un directorio de trabajo ajeno a la raíz del repositorio
    (simulado con monkeypatch.chdir(tmp_path)), `--consultation` con clave válida
    y mocks completos debe completar con código 0 e imprimir el markdown en stdout,
    sin fallar con FileNotFoundError por las rutas del catálogo CIE-10 ni por las
    del archivo de redacción (wording).
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    otro_directorio = tmp_path / "otro_cwd"
    otro_directorio.mkdir()
    monkeypatch.chdir(otro_directorio)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackend:
        def __init__(
            self, text: str = "El paciente refiere cefalea intensa y dolor de cabeza."
        ) -> None:
            self.text = text

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            mock_seg = MagicMock()
            mock_seg.start = 0.0
            mock_seg.end = 1.0
            mock_seg.text = self.text
            mock_seg.no_speech_prob = 0.0
            return [mock_seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackend(),
    )

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa: cefalea tensional recurrente.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    codigo = main(
        [
            "--consultation",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_test_cwd",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 0
    assert "- El paciente dijo:" in captured.out
    assert "- La historia registra:" in captured.out
    assert "El paciente refiere cefalea" in captured.out
    assert "cefalea tensional" in captured.out
    assert captured.err.replace(_AVISO_FALTA_KEY, "") == ""


# ---------------------------------------------------------------------------
# CLI --converse --mic (Unidad U10 / 33-plan-consultorio-voz)
# ---------------------------------------------------------------------------


def _modulo_doble(nombre: str, **atributos: Any) -> ModuleType:
    """Módulo falso con `__spec__`, para que `importlib.util.find_spec` lo
    acepte sin que exista el paquete real. Sustituir por el nombre en texto
    (`monkeypatch.setattr("faster_whisper.X", ...)`) importa el módulo de
    verdad, y en la instalación por defecto no está: es I-132."""
    modulo = ModuleType(nombre)
    modulo.__spec__ = ModuleSpec(nombre, loader=None)
    for clave, valor in atributos.items():
        setattr(modulo, clave, valor)
    return modulo


class _DummyMicrophoneSource:
    def __init__(self, max_chunks: int = 500) -> None:
        self._chunks_yielded = 0
        self._max_chunks = max_chunks

    def __enter__(self) -> _DummyMicrophoneSource:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def __call__(self) -> Any:
        import numpy as np

        self._chunks_yielded += 1
        if self._chunks_yielded > self._max_chunks:
            raise AssertionError(
                "Flujo de micrófono excedido: el bucle no terminó tras procesar "
                "los trozos esperados"
            )
        return np.zeros(512, dtype=np.float32)


def _scripted_vad_factory() -> Any:
    import numpy as np

    turn_chunks = 0

    def vad(tail: np.ndarray) -> np.ndarray:
        nonlocal turn_chunks
        if len(tail) == 512:
            turn_chunks = 0
        turn_chunks += 1
        if turn_chunks <= 2:
            return np.array([0.9])
        return np.array([0.0])

    return vad


def test_converse_mic_salir_termina_con_cero_sin_llamar_session_turn(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§6.4 Caso 1: --converse --mic sin --history: un guion que transcribe
    a 'salir' termina con código 0 sin llamar a Session.turn.
    """
    pytest.importorskip("sounddevice", reason="requiere el extra 'capture' (sounddevice)")
    pytest.importorskip("faster_whisper", reason="requiere el extra 'capture' (faster_whisper)")
    _clean_llm_env(monkeypatch)
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    monkeypatch.setattr("aida.voice.listen.MicrophoneSource", _DummyMicrophoneSource)
    monkeypatch.setattr("aida.voice.listen.vad_from_faster_whisper", _scripted_vad_factory)

    class MockSegment:
        def __init__(
            self, text: str, start: float = 0.0, end: float = 1.0, no_speech_prob: float = 0.0
        ) -> None:
            self.text = text
            self.start = start
            self.end = end
            self.no_speech_prob = no_speech_prob

    script = iter(["salir"])

    class MockWhisper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[MockSegment], None]:
            return [MockSegment(next(script))], None

    monkeypatch.setattr("faster_whisper.WhisperModel", MockWhisper)

    turn_called = False

    def spy_turn(self: Any, utterance: str) -> str:
        nonlocal turn_called
        turn_called = True
        return "respuesta"

    monkeypatch.setattr("aida.cognition.session.Session.turn", spy_turn)

    codigo = main(["--pack", str(pack_file), "--converse", "--mic"])

    captured = capsys.readouterr()
    assert codigo == 0
    assert not turn_called
    assert "(escuchando; di 'salir' para terminar)" in captured.out


def test_converse_mic_hola_llama_session_turn_una_vez(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§6.4 Caso 2: --converse --mic: un guion que transcribe a 'hola'
    llama a Session.turn('hola') una vez antes de recibir 'salir'.
    """
    pytest.importorskip("sounddevice", reason="requiere el extra 'capture' (sounddevice)")
    pytest.importorskip("faster_whisper", reason="requiere el extra 'capture' (faster_whisper)")
    _clean_llm_env(monkeypatch)
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    monkeypatch.setattr("aida.voice.listen.MicrophoneSource", _DummyMicrophoneSource)
    monkeypatch.setattr("aida.voice.listen.vad_from_faster_whisper", _scripted_vad_factory)

    class MockSegment:
        def __init__(
            self, text: str, start: float = 0.0, end: float = 1.0, no_speech_prob: float = 0.0
        ) -> None:
            self.text = text
            self.start = start
            self.end = end
            self.no_speech_prob = no_speech_prob

    script = iter(["hola", "salir"])

    class MockWhisper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[MockSegment], None]:
            return [MockSegment(next(script))], None

    monkeypatch.setattr("faster_whisper.WhisperModel", MockWhisper)

    recorded_turns: list[str] = []

    def spy_turn(self: Any, utterance: str) -> str:
        recorded_turns.append(utterance)
        return f"Respuesta a {utterance}"

    monkeypatch.setattr("aida.cognition.session.Session.turn", spy_turn)

    codigo = main(["--pack", str(pack_file), "--converse", "--mic"])

    captured = capsys.readouterr()
    assert codigo == 0
    assert recorded_turns == ["hola"]
    assert "Respuesta a hola" in captured.out


def test_converse_mic_history_inexistente_falla_antes_de_cargar_whisper(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§6.4 Caso 3 / Mata M10: --converse --mic --history inexistente.pdf
    devuelve código 1, mensaje con la ruta, antes de cargar Whisper.
    """
    pytest.importorskip("sounddevice", reason="requiere el extra 'capture' (sounddevice)")
    pytest.importorskip("faster_whisper", reason="requiere el extra 'capture' (faster_whisper)")
    _clean_llm_env(monkeypatch)
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    def crashing_whisper(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("WhisperModel no debe construirse si la ruta de historia no existe")

    monkeypatch.setattr("faster_whisper.WhisperModel", crashing_whisper)

    inexistente = tmp_path / "historia_inexistente.pdf"
    codigo = main(
        ["--pack", str(pack_file), "--converse", "--mic", "--history", str(inexistente)]
    )

    captured = capsys.readouterr()
    assert codigo == 1
    assert "no existe" in captured.err
    assert str(inexistente) in captured.err


def test_converse_sin_mic_no_importa_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§6.4 Caso 4: --converse sin --mic no importa sounddevice."""
    _clean_llm_env(monkeypatch)
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    monkeypatch.setattr("sys.stdin", io.StringIO("salir\n"))
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)

    codigo = main(["--pack", str(pack_file), "--converse"])
    assert codigo == 0
    assert "sounddevice" not in sys.modules


def test_converse_mic_modo_consulta_delega_en_consultation_no_en_session_turn(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """§6.4 Caso 5 / Mata M9: Con --history y comando 'modo consulta',
    se entra en modo consulta llamando a consultation.run_with_history y no a
    session.turn.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    pytest.importorskip("sounddevice", reason="requiere el extra 'capture' (sounddevice)")
    pytest.importorskip("faster_whisper", reason="requiere el extra 'capture' (faster_whisper)")
    _clean_llm_env(monkeypatch)
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    historia_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(historia_path)

    monkeypatch.setattr("aida.voice.listen.MicrophoneSource", _DummyMicrophoneSource)
    monkeypatch.setattr("aida.voice.listen.vad_from_faster_whisper", _scripted_vad_factory)

    # El operador dice "modo consulta", el consultado dice "dolor de cabeza",
    # luego operador dice "salir"
    script = iter(["modo consulta", "dolor de cabeza", "salir"])

    class MockSegment:
        def __init__(
            self, text: str, start: float = 0.0, end: float = 1.0, no_speech_prob: float = 0.0
        ) -> None:
            self.text = text
            self.start = start
            self.end = end
            self.no_speech_prob = no_speech_prob

    class MockWhisper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[MockSegment], None]:
            return [MockSegment(next(script))], None

    monkeypatch.setattr("faster_whisper.WhisperModel", MockWhisper)
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa del paciente.",
    )

    turn_calls: list[str] = []

    def spy_turn(self: Any, utterance: str) -> str:
        turn_calls.append(utterance)
        return f"Turn: {utterance}"

    monkeypatch.setattr("aida.cognition.session.Session.turn", spy_turn)

    codigo = main(
        [
            "--pack",
            str(pack_file),
            "--converse",
            "--mic",
            "--history",
            str(historia_path),
            "--owner",
            "paciente_voz_01",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 0
    # Session.turn no debe ser llamado para "modo consulta" ni en la consulta
    assert "modo consulta" not in turn_calls
    assert len(turn_calls) == 0
    # La salida debe mostrar el cambio de modo y el resumen
    assert "(modo consulta: escuchando)" in captured.out
    assert "- El paciente dijo:" in captured.out
    assert "dolor de cabeza" in captured.out
    assert "(de vuelta en conversación)" in captured.out


def test_converse_mic_sin_extra_capture_devuelve_uno_con_mensaje_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """I-102: si el extra capture no está disponible, --mic devuelve 1
    e imprime el mensaje de error tratado por stderr."""
    _clean_llm_env(monkeypatch)
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    monkeypatch.setattr("aida.voice.listen.MicrophoneSource", _DummyMicrophoneSource)
    monkeypatch.setattr("aida.voice.listen.vad_from_faster_whisper", _scripted_vad_factory)

    class MockSegment:
        def __init__(
            self, text: str, start: float = 0.0, end: float = 1.0, no_speech_prob: float = 0.0
        ) -> None:
            self.text = text
            self.start = start
            self.end = end
            self.no_speech_prob = no_speech_prob

    script = iter(["salir"])

    class MockWhisper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[MockSegment], None]:
            return [MockSegment(next(script))], None

    monkeypatch.setitem(
        sys.modules, "faster_whisper", _modulo_doble("faster_whisper", WhisperModel=MockWhisper)
    )
    monkeypatch.setitem(sys.modules, "sounddevice", _modulo_doble("sounddevice"))
    monkeypatch.setattr("aida.voice.listen.check_capture_available", lambda: False)

    codigo = main(["--pack", str(pack_file), "--converse", "--mic"])

    captured = capsys.readouterr()
    assert codigo == 1
    assert (
        "aida: error: --mic requiere el extra 'capture' (uv sync --extra capture)"
        in captured.err
    )


def test_converse_mic_keyboard_interrupt_termina_con_cero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """I-106: KeyboardInterrupt durante la escucha en --converse --mic termina con código 0."""
    pytest.importorskip("sounddevice", reason="requiere el extra 'capture' (sounddevice)")
    pytest.importorskip("faster_whisper", reason="requiere el extra 'capture' (faster_whisper)")
    _clean_llm_env(monkeypatch)
    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    class _MockWhisper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            return [], None

    monkeypatch.setattr("faster_whisper.WhisperModel", _MockWhisper)

    class _InterruptMicrophoneSource:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def __call__(self) -> Any:
            raise KeyboardInterrupt

    monkeypatch.setattr("aida.voice.listen.MicrophoneSource", _InterruptMicrophoneSource)
    monkeypatch.setattr("aida.voice.listen.vad_from_faster_whisper", _scripted_vad_factory)

    codigo = main(["--pack", str(pack_file), "--converse", "--mic"])
    assert codigo == 0


def test_switch_to_consult_phrases_uses_subject_noun(tmp_path: Path) -> None:
    """I-103: switch_to_consult_phrases utiliza el subject_noun del wording."""
    from aida.arms.health.wording import switch_to_consult_phrases

    wording_file = tmp_path / "wording.yaml"
    wording_file.write_text("subject_noun: cliente\n", encoding="utf-8")
    phrases = switch_to_consult_phrases(wording_file)
    assert "escucha al cliente" in phrases


def test_switch_to_consult_phrases_missing_subject_noun_raises(tmp_path: Path) -> None:
    """I-103: si falta subject_noun en el wording, se lanza KeyError o ValueError."""
    from aida.arms.health.wording import switch_to_consult_phrases

    wording_file = tmp_path / "wording.yaml"
    wording_file.write_text("system_prompt: algo\n", encoding="utf-8")
    with pytest.raises((KeyError, ValueError)):
        switch_to_consult_phrases(wording_file)


def test_conversation_silence_seconds_is_one_point_five() -> None:
    """I-107: ancla que CONVERSATION_SILENCE_SECONDS es 1.5."""
    from aida.run import CONVERSATION_SILENCE_SECONDS

    assert CONVERSATION_SILENCE_SECONDS == 1.5


# ---------------------------------------------------------------------------
# Pruebas §B.3: Revisión por CLI (51-plan-revision-profesional.md)
# ---------------------------------------------------------------------------


def test_consultation_cli_review_sin_segment_devuelve_error(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Caso 16 (§B.3): --review sin --segment devuelve 1 y stderr contiene
    '--review requiere --segment'.
    """
    audio_path = tmp_path / "dummy.wav"
    audio_path.write_bytes(b"RIFF dummy")
    hist_path = tmp_path / "dummy.txt"
    hist_path.write_text("historia")

    codigo = main(
        [
            "--consultation",
            "--review",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            "doc_test_review_no_segment",
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 1
    assert "--review requiere --segment" in captured.err


def test_consultation_cli_review_edicion_y_rechazo_completo(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 17 (§B.3): --consultation --segment --extract --review --print-model
    con builtins.input sustituido:
    - '[c] confirmar / [e] editar / [r] rechazar: ' -> 'e' la primera vez, 'c' siguientes;
    - 'texto corregido: ' -> texto nuevo con lateralidad distinta (izquierda en vez de derecha);
    - '[c] confirmar / [r] rechazar: ' -> siempre 'r'.
    Resultado: código 0; el JSON impreso lleva el texto editado con sus campos
    recalculados; el resumen impreso lleva no_confirmed_text del pack real.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackendExtract:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg1 = MagicMock()
            seg1.start = 0.0
            seg1.end = 2.0
            seg1.text = "consulta hoy por dolor en rodilla derecha desde hace dos semanas"
            seg1.no_speech_prob = 0.0

            seg2 = MagicMock()
            seg2.start = 2.0
            seg2.end = 4.0
            seg2.text = "se receta ibuprofeno cada 8 horas"
            seg2.no_speech_prob = 0.0

            return [seg1, seg2], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackendExtract(),
    )

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa: sin antecedentes de interés.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    first_finding = True

    def mock_input(prompt: str) -> str:
        nonlocal first_finding
        if "[c] confirmar / [e] editar / [r] rechazar:" in prompt:
            if first_finding:
                first_finding = False
                return "e"
            return "c"
        if "texto corregido:" in prompt:
            return "consulta hoy por dolor en rodilla izquierda desde hace dos semanas"
        if "[c] confirmar / [r] rechazar:" in prompt:
            return "r"
        raise AssertionError(f"Prompt inesperado: {prompt}")

    monkeypatch.setattr("builtins.input", mock_input)

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--extract",
            "--review",
            "--print-model",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_cli_review_test",
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 0
    assert captured.err.replace(_AVISO_FALTA_KEY, "") == ""

    # Extraer el JSON tras el resumen en stdout
    json_start = captured.out.find("{")
    assert json_start != -1, "No se encontró JSON en stdout"
    json_part = captured.out[json_start:]

    parsed = json.loads(json_part)
    expected = {
        "enfermedad_actual": [
            {
                "text": "consulta hoy por dolor en rodilla izquierda desde hace dos semanas",
                "duration": "2 semanas",
                "laterality": "izquierdo",
                "reported_by": None,
                "negated": None,
            }
        ],
        "medicamentos": [
            {
                "text": "se receta ibuprofeno cada 8 horas",
                "duration": None,
                "laterality": None,
                "reported_by": None,
                "negated": None,
            }
        ],
    }
    assert parsed == expected

    # El resumen impreso lleva no_confirmed_text del pack real
    from aida.arms.health.wording import default_summary_wording

    real_wording = default_summary_wording()
    assert real_wording.no_confirmed_text is not None
    assert real_wording.no_confirmed_text in captured.out


def test_consultation_cli_review_eof_cancela_sin_guardar_cache(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 18 (§B.3): input levanta EOFError en la primera llamada:
    código 1, stderr contiene 'revisión incompleta', y no existe
    tmp_path / '.cache' / 'aida' / 'consultas' / '<owner>.cache'.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackendExtract:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg1 = MagicMock()
            seg1.start = 0.0
            seg1.end = 2.0
            seg1.text = "consulta hoy por dolor en rodilla derecha desde hace dos semanas"
            seg1.no_speech_prob = 0.0
            return [seg1], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackendExtract(),
    )

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    def mock_input(prompt: str) -> str:
        raise EOFError("interrupción simulada")

    monkeypatch.setattr("builtins.input", mock_input)

    owner = "paciente_cli_review_eof"
    codigo = main(
        [
            "--consultation",
            "--segment",
            "--extract",
            "--review",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            owner,
        ]
    )

    captured = capsys.readouterr()
    assert codigo == 1
    assert "revisión incompleta" in captured.err

    cache_file = tmp_path / ".cache" / "aida" / "consultas" / f"{owner}.cache"
    assert not cache_file.exists()


# ---------------------------------------------------------------------------
# CLI --check-capture y robustez de captura (§B.4 / 56-plan-captura-robusta.md)
# ---------------------------------------------------------------------------


def test_cli_check_capture_todo_correcto_devuelve_cero(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 16 (§B.4): --check-capture con todas las comprobaciones en True
    devuelve 0 y muestra 'ok   ' en cada línea."""
    monkeypatch.setattr(
        "aida.arms.health.capture.capture_diagnosis",
        lambda: (
            ("módulo faster_whisper", True),
            ("módulo sounddevice", True),
            ("módulo pytesseract", True),
            ("modelo small en cpu", True),
            ("tesseract 5.3.0, español", True),
            ("entrada de audio: Micrófono", True),
        ),
    )
    codigo = main(["--check-capture"])
    captured = capsys.readouterr()
    assert codigo == 0
    lineas = captured.out.strip().splitlines()
    assert len(lineas) == 6
    assert all(linea.startswith("ok   ") for linea in lineas)
    assert captured.err == ""


def test_cli_check_capture_con_fallo_devuelve_uno(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 17 (§B.4): --check-capture con al menos una comprobación en False
    devuelve 1 y muestra 'FALTA' en la línea afectada."""
    monkeypatch.setattr(
        "aida.arms.health.capture.capture_diagnosis",
        lambda: (
            ("módulo faster_whisper", True),
            ("modelo small en cpu: Libcublas not found", False),
            ("módulo sounddevice", True),
        ),
    )
    codigo = main(["--check-capture"])
    captured = capsys.readouterr()
    assert codigo == 1
    assert "FALTA modelo small en cpu: Libcublas not found" in captured.out


def test_consultation_cli_audio_sin_voz_devuelve_uno_sin_traza(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 18 (§B.4): --consultation con audio sin voz devuelve 1,
    mensaje 'no se detectó voz' en stderr y sin traceback."""
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperSinVoz:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            return [], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperSinVoz(),
    )
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa.",
    )

    audio_path = tmp_path / "consulta_silencio.wav"
    audio_path.write_bytes(b"RIFF silence")
    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    codigo = main(
        [
            "--consultation",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_silencio",
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 1
    assert "aida: error: no se detectó voz en" in captured.err
    assert "Traceback" not in captured.err


def test_converse_mic_modo_consulta_audio_sin_voz_imprime_error_y_sigue_en_bucle(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 19 (§B.4): en --converse --mic, si la consulta levantada por 'modo consulta'
    falla por audio sin voz, imprime el error y continúa en conversación hasta 'salir'."""
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    pytest.importorskip("sounddevice", reason="requiere el extra 'capture' (sounddevice)")
    pytest.importorskip("faster_whisper", reason="requiere el extra 'capture' (faster_whisper)")
    _clean_llm_env(monkeypatch)
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    pack_file = tmp_path / "pack.yaml"
    pack_file.write_text(_pack_yaml_valido(), encoding="utf-8")

    historia_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(historia_path)

    monkeypatch.setattr("aida.voice.listen.MicrophoneSource", _DummyMicrophoneSource)
    monkeypatch.setattr("aida.voice.listen.vad_from_faster_whisper", _scripted_vad_factory)

    script = iter(["modo consulta", "saludo cualquiera", "salir"])

    class _MockSegment:
        def __init__(self, text: str, no_speech_prob: float = 0.0) -> None:
            self.text = text
            self.start = 0.0
            self.end = 1.0
            self.no_speech_prob = no_speech_prob

    class _MockWhisper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(
            self, audio_path: str, *args: Any, **kwargs: Any
        ) -> tuple[list[Any], None]:
            if "consultado.wav" in audio_path:
                return [], None
            return [_MockSegment(next(script))], None

    monkeypatch.setattr("faster_whisper.WhisperModel", _MockWhisper)
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa.",
    )

    codigo = main(
        [
            "--pack",
            str(pack_file),
            "--converse",
            "--mic",
            "--history",
            str(historia_path),
            "--owner",
            "paciente_voz_silencio",
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 0
    assert "aida: error: no se detectó voz en" in captured.err
    assert "(de vuelta en conversación)" in captured.out


def test_consultation_cli_review_registra_completed_true(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 13 (§B.3): --review registra:
    tras la ejecución, tmp_path/.local/share/aida/revisiones.jsonl tiene una línea
    con '"completed":true'.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackend:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg = MagicMock()
            seg.start = 0.0
            seg.end = 2.0
            seg.text = "consulta hoy por dolor en rodilla derecha"
            seg.no_speech_prob = 0.0
            return [seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackend(),
    )
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")
    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    monkeypatch.setattr("builtins.input", lambda prompt: "c")

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--review",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_cli_review_metrics",
        ]
    )
    capsys.readouterr()
    assert codigo == 0

    metrics_path = tmp_path / ".local" / "share" / "aida" / "revisiones.jsonl"
    assert metrics_path.exists()
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"completed":true' in lines[0]


def test_consultation_cli_review_cortado_registra_completed_false_y_sale_con_uno(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 14 (§B.3): --review cortado (input levanta EOFError):
    salida 1, y el archivo tiene una línea con '"completed":false'.
    """
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackend:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg = MagicMock()
            seg.start = 0.0
            seg.end = 2.0
            seg.text = "consulta hoy por dolor en rodilla derecha"
            seg.no_speech_prob = 0.0
            return [seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackend(),
    )
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")
    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    def mock_input_eof(prompt: str) -> str:
        raise EOFError("corte con Ctrl-D")

    monkeypatch.setattr("builtins.input", mock_input_eof)

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--review",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_cli_review_cut",
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 1
    assert "revisión incompleta" in captured.err

    metrics_path = tmp_path / ".local" / "share" / "aida" / "revisiones.jsonl"
    assert metrics_path.exists()
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"completed":false' in lines[0]


def test_cli_review_stats_sin_nada_y_con_registros(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 15 (§B.3): --review-stats sin nada:
    Sin AIDA_CONSULT_CACHE_KEY y sin archivo: imprime 'sin revisiones registradas' y sale con 0.
    Con el archivo escrito a partir de los dos registros del caso 6:
    imprime la salida literal del caso 6.
    """
    monkeypatch.delenv("AIDA_CONSULT_CACHE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # 1. Sin archivo
    codigo = main(["--review-stats"])
    captured = capsys.readouterr()
    assert codigo == 0
    assert captured.out == "sin revisiones registradas\n"

    # 2. Con archivo escrito a partir de los dos registros del caso 6
    from aida.arms.health.metrics import ReviewMetricsStore, ReviewRecord

    metrics_path = tmp_path / ".local" / "share" / "aida" / "revisiones.jsonl"
    store = ReviewMetricsStore(metrics_path)
    r1 = ReviewRecord(
        completed=True,
        seconds=42.5,
        findings={"confirmed": 2, "edited": 1, "rejected": 1},
        codes={"rejected": 2},
        sections={
            "enfermedad_actual": {"edited": 1},
            "examen_fisico": {"confirmed": 1},
            "motivo_consulta": {"confirmed": 1},
            "signos": {"rejected": 1},
        },
        code_origins={"today": {"rejected": 2}},
    )
    r2 = ReviewRecord(
        completed=False,
        seconds=7.0,
        findings={"confirmed": 1},
        codes={},
        sections={"motivo_consulta": {"confirmed": 1}},
        code_origins={},
    )
    store.append(r1)
    store.append(r2)

    codigo2 = main(["--review-stats"])
    captured2 = capsys.readouterr()
    assert codigo2 == 0
    expected = (
        "revisiones: 2 (1 completas, 1 incompletas)\n"
        "tiempo medio por revisión completa: 42.5 s\n"
        "hallazgos: 5 (confirmados 60.0 %, editados 20.0 %, rechazados 20.0 %)\n"
        "códigos: 2 (confirmados 0.0 %, rechazados 100.0 %)\n"
        "códigos por origen:\n"
        "  de esta consulta: 2 (confirmados 0.0 %, rechazados 100.0 %)\n"
        "por sección:\n"
        "  enfermedad_actual: 1 (confirmados 0.0 %, editados 100.0 %, rechazados 0.0 %)\n"
        "  examen_fisico: 1 (confirmados 100.0 %, editados 0.0 %, rechazados 0.0 %)\n"
        "  motivo_consulta: 2 (confirmados 100.0 %, editados 0.0 %, rechazados 0.0 %)\n"
        "  signos: 1 (confirmados 0.0 %, editados 0.0 %, rechazados 100.0 %)\n"
    )
    assert captured2.out == expected


def test_cli_review_stats_ruta_con_linea_ilegible_devuelve_uno_con_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 16 (§B.3): --review-stats RUTA lee ese archivo;
    si tiene una línea ilegible, sale con 1 y stderr contiene la ruta y el número de línea.
    """
    monkeypatch.delenv("AIDA_CONSULT_CACHE_KEY", raising=False)
    custom_path = tmp_path / "custom_revisiones.jsonl"
    bad_content = (
        '{"completed":true,"seconds":10.0,"findings":{},"codes":{},"sections":{}}\n'
        "{no es json valido\n"
    )
    custom_path.write_text(bad_content, encoding="utf-8")

    codigo = main(["--review-stats", str(custom_path)])
    captured = capsys.readouterr()
    assert codigo == 1
    assert str(custom_path) in captured.err
    assert ":2:" in captured.err
    assert "registro ilegible" in captured.err


def test_cli_sin_review_no_mide(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 17 (§B.3): --consultation --segment (sin --review) no crea el archivo de métricas."""
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackend:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg = MagicMock()
            seg.start = 0.0
            seg.end = 2.0
            seg.text = "consulta hoy por dolor en rodilla derecha"
            seg.no_speech_prob = 0.0
            return [seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackend(),
    )
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")
    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_cli_sin_review",
        ]
    )
    capsys.readouterr()
    assert codigo == 0

    metrics_path = tmp_path / ".local" / "share" / "aida" / "revisiones.jsonl"
    assert not metrics_path.exists()


# ---------------------------------------------------------------------------
# Plan 60: Exportar la consulta revisada (§B.2)
# ---------------------------------------------------------------------------


def _preparar_entorno_consultation_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _MockWhisperBackend:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            seg = MagicMock()
            seg.start = 0.0
            seg.end = 2.0
            seg.text = "consulta hoy por dolor en rodilla derecha"
            seg.no_speech_prob = 0.0
            return [seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _MockWhisperBackend(),
    )
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio bytes")
    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)
    return audio_path, img_path


def test_cli_consultation_export_con_ruta_escribe_resumen_y_aviso_en_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 8 (§B.2): --consultation ... --export RUTA escribe el resumen
    y emite avisos en stderr sin ensuciar stdout."""
    audio_path, img_path = _preparar_entorno_consultation_cli(monkeypatch, tmp_path)
    custom_export = tmp_path / "carpeta_salida" / "resumen_exportado.md"

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_export_ruta",
            "--export",
            str(custom_export),
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 0
    assert custom_export.exists()
    assert custom_export.read_text(encoding="utf-8").strip() != ""
    assert f"aida: consulta escrita en {custom_export}" in captured.err
    assert "aida: aviso: queda en claro, sin cifrar; dónde vive es cosa tuya" in captured.err
    # Los avisos van a stderr, no a stdout
    assert "aida: consulta escrita en" not in captured.out


def test_cli_consultation_export_sin_ruta_escribe_en_carpeta_por_defecto(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 9 (§B.2): --export sin ruta escribe bajo
    Path.home() / 'Documentos' / 'aida' / 'consultas'."""
    audio_path, img_path = _preparar_entorno_consultation_cli(monkeypatch, tmp_path)

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_export_defecto",
            "--export",
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 0

    export_dir = tmp_path / "Documentos" / "aida" / "consultas"
    assert export_dir.exists()
    archivos = list(export_dir.glob("*.md"))
    assert len(archivos) == 1
    assert archivos[0].read_text(encoding="utf-8").strip() != ""
    assert f"aida: consulta escrita en {archivos[0]}" in captured.err
    assert "aida: aviso: queda en claro, sin cifrar; dónde vive es cosa tuya" in captured.err


def test_cli_consultation_sin_export_no_escribe_en_carpeta_por_defecto(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 10 (§B.2): Sin --export no se crea ningún archivo bajo esa carpeta."""
    audio_path, img_path = _preparar_entorno_consultation_cli(monkeypatch, tmp_path)

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_sin_export",
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 0

    export_dir = tmp_path / "Documentos" / "aida" / "consultas"
    assert not export_dir.exists() or len(list(export_dir.glob("*.md"))) == 0
    assert "aida: consulta escrita en" not in captured.err
    assert "aida: consulta escrita en" not in captured.out


def test_cli_consultation_export_con_print_model_escribe_json_al_lado(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 11 (§B.2): --export junto con --print-model escribe también el .json."""
    audio_path, img_path = _preparar_entorno_consultation_cli(monkeypatch, tmp_path)
    custom_export = tmp_path / "carpeta_salida" / "resumen_modelo.md"

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--audio",
            str(audio_path),
            "--history",
            str(img_path),
            "--owner",
            "paciente_export_model",
            "--export",
            str(custom_export),
            "--print-model",
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 0
    assert custom_export.exists()

    model_path = custom_export.with_suffix(".json")
    assert model_path.exists()
    model_data = json.loads(model_path.read_text(encoding="utf-8"))
    assert isinstance(model_data, dict)
    assert f"aida: consulta escrita en {custom_export}" in captured.err
    assert f"aida: modelo escrito en {model_path}" in captured.err


def test_consultation_cli_cached_stt_segunda_ejecucion_no_llama_modelo(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 12 (§B.2): --consultation dos veces con el mismo audio y la misma clave:
    la segunda no llama al modelo de transcripción."""
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    transcribe_calls = 0

    class _CountingWhisperBackend:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            nonlocal transcribe_calls
            transcribe_calls += 1
            mock_seg = MagicMock()
            mock_seg.start = 0.0
            mock_seg.end = 2.0
            mock_seg.text = "El paciente refiere cefalea intensa y dolor de cabeza."
            mock_seg.no_speech_prob = 0.0
            return [mock_seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _CountingWhisperBackend(),
    )
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa: cefalea recurrente.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio content for cache test")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    cmd = [
        "--consultation",
        "--audio",
        str(audio_path),
        "--history",
        str(img_path),
        "--owner",
        "paciente_cached_stt",
    ]

    # Primera ejecución: transcribe
    codigo1 = main(cmd)
    assert codigo1 == 0
    assert transcribe_calls == 1

    # Segunda ejecución con la misma clave y mismo audio: usa caché y NO transcribe
    codigo2 = main(cmd)
    assert codigo2 == 0
    assert transcribe_calls == 1


def test_consultation_cli_cached_stt_clave_distinta_vuelve_a_transcribir(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 13 (§B.2): Con AIDA_CONSULT_CACHE_KEY distinta entre las dos ejecuciones,
    la segunda vuelve a transcribir porque no puede descifrar la caché anterior."""
    image_mod = pytest.importorskip(
        "PIL.Image",
        reason="requiere el extra 'capture' (faster-whisper, pytesseract, Pillow)",
    )
    _clean_llm_env(monkeypatch)
    key1 = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key1)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    transcribe_calls = 0

    class _CountingWhisperBackend:
        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None]:
            nonlocal transcribe_calls
            transcribe_calls += 1
            mock_seg = MagicMock()
            mock_seg.start = 0.0
            mock_seg.end = 2.0
            mock_seg.text = "El paciente refiere cefalea intensa y dolor de cabeza."
            mock_seg.no_speech_prob = 0.0
            return [mock_seg], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: _CountingWhisperBackend(),
    )
    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *args, **kwargs: "Historia clínica previa: cefalea recurrente.",
    )

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio content for key test")

    img_path = tmp_path / "historia.png"
    img = image_mod.new("RGB", (20, 20), color="white")
    img.save(img_path)

    cmd = [
        "--consultation",
        "--audio",
        str(audio_path),
        "--history",
        str(img_path),
        "--owner",
        "paciente_diff_key",
    ]

    # Primera ejecución con key1
    codigo1 = main(cmd)
    assert codigo1 == 0
    assert transcribe_calls == 1

    # Segunda ejecución con key2 distinta: no puede leer caché de key1 y vuelve a transcribir
    key2 = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key2)

    codigo2 = main(cmd)
    assert codigo2 == 0
    assert transcribe_calls == 2


# ---------------------------------------------------------------------------
# Plan 67: Puerto de transcripción (nube y respaldo local) (§B.3)
# ---------------------------------------------------------------------------


def test_transcription_provider_local_no_construye_cliente_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 12: Con provider: local, no se construye ningún cliente HTTP:
    el transcriptor que llega a la consulta es el local envuelto en CachedSTT.
    """
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcripts import CachedSTT
    from aida.run import _transcription

    cliente_construido = False

    def mock_httpx_client(*args: Any, **kwargs: Any) -> Any:
        nonlocal cliente_construido
        cliente_construido = True
        raise AssertionError("No debió construirse ningún httpx.Client con provider: local")

    monkeypatch.setattr("httpx.Client", mock_httpx_client, raising=False)
    monkeypatch.setenv("AIDA_STT_API_KEY", "clave-prueba-para-test")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    ajustes = replace(load_capture_settings(), provider="local")
    monkeypatch.setattr(
        "aida.arms.health.capture.load_capture_settings",
        lambda: ajustes,
    )
    key = Fernet.generate_key().decode()
    stt_base = MagicMock()
    resultado = _transcription(stt_base, key)
    assert isinstance(resultado, CachedSTT)
    assert resultado._stt is stt_base
    assert not cliente_construido


def test_transcription_provider_cloud_sin_api_key_usa_local_y_avisa_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 13: Con provider: cloud y sin AIDA_STT_API_KEY: se usa el local
    y stderr lo avisa.
    """
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcripts import CachedSTT
    from aida.run import _transcription

    monkeypatch.delenv("AIDA_STT_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    ajustes = replace(load_capture_settings(), provider="cloud")
    monkeypatch.setattr(
        "aida.arms.health.capture.load_capture_settings",
        lambda: ajustes,
    )
    key = Fernet.generate_key().decode()
    stt_base = MagicMock()
    resultado = _transcription(stt_base, key)

    assert isinstance(resultado, CachedSTT)
    assert resultado._stt is stt_base
    captured = capsys.readouterr()
    assert "aida: aviso: falta AIDA_STT_API_KEY; se transcribe en local" in captured.err


def test_identidad_cache_distingue_nube_de_local_no_reutiliza(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 14: La identidad de la caché distingue nube de local: el mismo audio
    transcrito con provider: cloud y luego con local no reutiliza la primera transcripción.
    """
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcription import FallbackTranscription
    from aida.arms.health.transcripts import CachedSTT
    from aida.run import _transcription

    key = Fernet.generate_key().decode()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AIDA_STT_API_KEY", "clave-prueba")

    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"RIFF dummy audio bytes para distincion de cache")

    # 1. Ejecución con provider: cloud
    ajustes_cloud = replace(load_capture_settings(), provider="cloud")
    monkeypatch.setattr(
        "aida.arms.health.capture.load_capture_settings",
        lambda: ajustes_cloud,
    )

    mock_cloud_backend = MagicMock()
    mock_seg_cloud = TranscriptSegment(0.0, 2.0, "Texto desde nube")
    mock_cloud_backend.transcribe_segments.return_value = (mock_seg_cloud,)

    stt_cloud = cast(Any, _transcription(MagicMock(), key))
    assert isinstance(stt_cloud, FallbackTranscription)
    primario_cloud: Any = stt_cloud._primary
    primario_cloud._stt = mock_cloud_backend
    res_cloud = stt_cloud.transcribe_segments(audio)
    assert res_cloud[0].text == "Texto desde nube"
    assert mock_cloud_backend.transcribe_segments.call_count == 1

    # 2. Ejecución con provider: local sobre el mismo audio
    ajustes_local = replace(load_capture_settings(), provider="local")
    monkeypatch.setattr(
        "aida.arms.health.capture.load_capture_settings",
        lambda: ajustes_local,
    )

    mock_local_backend = MagicMock()
    mock_seg_local = TranscriptSegment(0.0, 2.0, "Texto desde local")
    mock_local_backend.transcribe_segments.return_value = (mock_seg_local,)

    stt_local = _transcription(mock_local_backend, key)
    assert isinstance(stt_local, CachedSTT)
    res_local = stt_local.transcribe_segments(audio)
    assert res_local[0].text == "Texto desde local"
    assert mock_local_backend.transcribe_segments.call_count == 1


# ---------------------------------------------------------------------------
# Pruebas B.8: Inyección de catálogo de negación en la CLI (68-plan-negacion-evidencia.md)
# ---------------------------------------------------------------------------


def test_build_consultation_inyecta_catalogo_negacion_real(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 38 (§B.8): Con --consultation, la CLI inyecta un NegationCatalog
    no vacío cargado del pack."""
    import argparse
    from unittest.mock import MagicMock

    from aida.arms.health.negation import NegationCatalog
    from aida.run import _build_consultation

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    args = argparse.Namespace(llm=None)
    consultation = _build_consultation(
        args,
        stt=MagicMock(),
        ocr=MagicMock(),
        key=key,
    )
    assert consultation is not None
    assert isinstance(consultation._negation, NegationCatalog)
    assert len(consultation._negation.cues) > 0
    assert len(consultation._negation.stoppers) > 0
    assert ("no",) in consultation._negation.cues
    assert ("pero",) in consultation._negation.stoppers


# ---------------------------------------------------------------------------
# Plan 69: Catálogo declarado en el pack (§B.3)
# ---------------------------------------------------------------------------


def test_build_consultation_carga_catalogo_referencia(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 13 (§B.3): Con --consultation, la CLI carga el catálogo que declara el pack
    (por defecto reference) llamando a load_reference_catalog y NO a load_catalog."""
    import argparse
    from unittest.mock import MagicMock

    import aida.arms.health.codes as codes_mod
    from aida.run import _build_consultation

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    mock_load_ref = MagicMock(return_value=())
    mock_load_dual = MagicMock(return_value=())
    monkeypatch.setattr(codes_mod, "load_reference_catalog", mock_load_ref)
    monkeypatch.setattr(codes_mod, "load_catalog", mock_load_dual)

    args = argparse.Namespace(llm=None)
    consultation = _build_consultation(args, stt=MagicMock(), ocr=MagicMock(), key=key)
    assert consultation is not None
    assert mock_load_ref.called
    assert not mock_load_dual.called


def test_build_consultation_camino_dual_sigue_vivo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 14 (§B.3): Con matching.yaml declarando kind: dual y dos archivos,
    _build_consultation llama a load_catalog con las dos rutas en orden (.json primero)."""
    import argparse
    from unittest.mock import MagicMock

    import aida.arms.health.codes as codes_mod
    import aida.arms.health.match as match_mod
    from aida.arms.health.match import CatalogSpec, MatchingSettings
    from aida.run import _build_consultation

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    f_json = tmp_path / "test_codes.json"
    f_csv = tmp_path / "test_cie10.csv"
    f_json.touch()
    f_csv.touch()

    fake_settings = MatchingSettings(
        threshold=0.5,
        top_n=5,
        stopwords=frozenset(),
        catalog=CatalogSpec(kind="dual", paths=(f_json, f_csv)),
    )
    monkeypatch.setattr(match_mod, "load_matching_settings", lambda *a, **kw: fake_settings)

    mock_load_dual = MagicMock(return_value=())
    mock_load_ref = MagicMock(return_value=())
    monkeypatch.setattr(codes_mod, "load_catalog", mock_load_dual)
    monkeypatch.setattr(codes_mod, "load_reference_catalog", mock_load_ref)

    args = argparse.Namespace(llm=None)
    consultation = _build_consultation(args, stt=MagicMock(), ocr=MagicMock(), key=key)
    assert consultation is not None
    assert mock_load_dual.called
    assert not mock_load_ref.called
    assert mock_load_dual.call_args[0] == (f_json, f_csv)


def test_run_py_no_tiene_rutas_catalogo_hardcodeadas() -> None:
    """Caso 15 (§B.3): Ninguna ruta de catálogo queda escrita en run.py:
    comprobar que no aparecen 'codes.json' ni 'cie-10.csv' en src/aida/run.py."""
    run_path = Path(__file__).resolve().parents[1] / "src" / "aida" / "run.py"
    contenido = run_path.read_text(encoding="utf-8")
    assert "codes.json" not in contenido
    assert "cie-10.csv" not in contenido


# ---------------------------------------------------------------------------
# Plan 70: CLI y composición de transcripción (§B.5)
# ---------------------------------------------------------------------------


def test_build_consultation_inyecta_regla_retractacion_del_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 22 (§B.5): La CLI inyecta la regla de retractación real del pack."""
    import argparse

    from aida.arms.health.negation import RetractionRule
    from aida.run import _build_consultation

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    consultation = _build_consultation(
        argparse.Namespace(llm=None),
        stt=MagicMock(),
        ocr=MagicMock(),
        key=key,
    )
    assert consultation is not None
    assert isinstance(consultation._retraction, RetractionRule)
    assert consultation._retraction.max_chars == 24


def test_transcription_pasa_prompt_a_transcriptores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caso 23 (§B.5): La CLI pasa el prompt del pack a los dos transcriptores."""
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcription import FallbackTranscription
    from aida.run import _transcription

    key = Fernet.generate_key().decode()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AIDA_STT_API_KEY", "test-key")

    ajustes = replace(
        load_capture_settings(),
        provider="cloud",
        transcription_prompt="Prompt único de prueba",
    )
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: ajustes)

    fb = cast(Any, _transcription(MagicMock(), key))
    assert isinstance(fb, FallbackTranscription)
    primario: Any = fb._primary
    respaldo: Any = fb._fallback
    assert primario._stt._prompt == "Prompt único de prueba"
    prompt_hex = hashlib.blake2b(
        "Prompt único de prueba".encode(), digest_size=8
    ).hexdigest()
    assert prompt_hex in primario._model
    assert prompt_hex in respaldo._model


def test_cached_stt_huella_cambia_con_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caso 24 (§B.5): Dos capture.yaml idénticos salvo en prompt producen identidades distintas."""
    from aida.arms.health.capture import load_capture_settings
    from aida.run import _transcription

    key = Fernet.generate_key().decode()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    aj1 = replace(load_capture_settings(), provider="local", transcription_prompt="prompt A")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: aj1)
    stt1 = cast(Any, _transcription(MagicMock(), key))

    aj2 = replace(load_capture_settings(), provider="local", transcription_prompt="prompt B")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: aj2)
    stt2 = cast(Any, _transcription(MagicMock(), key))

    assert stt1._model != stt2._model

    monkeypatch.setenv("AIDA_STT_API_KEY", "key-test")
    aj_c1 = replace(load_capture_settings(), provider="cloud", transcription_prompt="prompt A")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: aj_c1)
    fb1 = cast(Any, _transcription(MagicMock(), key))

    aj_c2 = replace(load_capture_settings(), provider="cloud", transcription_prompt="prompt B")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: aj_c2)
    fb2 = cast(Any, _transcription(MagicMock(), key))

    assert fb1._primary._model != fb2._primary._model


def test_transcription_timeout_separa_connect_y_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caso 25 (§B.5): El cliente se construye con httpx.Timeout y valores
    separados de connect y read."""
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcription import FallbackTranscription
    from aida.run import _transcription

    key = Fernet.generate_key().decode()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AIDA_STT_API_KEY", "test-key")

    ajustes = replace(
        load_capture_settings(),
        provider="cloud",
        connect_timeout_seconds=3.5,
        timeout_seconds=95.0,
    )
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: ajustes)

    fb = cast(Any, _transcription(MagicMock(), key))
    assert isinstance(fb, FallbackTranscription)
    primario: Any = fb._primary
    cliente = primario._stt._client
    assert cliente.timeout.connect == 3.5
    assert cliente.timeout.read == 95.0


def test_load_capture_settings_connect_timeout_valida(tmp_path: Path) -> None:
    """Caso 26 (§B.5): connect_timeout_seconds es obligatorio y valida número positivo."""
    from aida.arms.health.capture import load_capture_settings

    base_yaml = (
        "model: small\ndevice: cpu\nno_speech_threshold: 0.6\nabbreviations: []\n"
        "provider: local\ncloud_model: m\ncloud_language: es\ncloud_base_url: http://x\n"
        "timeout_seconds: 10\ncompute_type: int8\ncloud_formats: ['.wav']\n"
        "transcription_prompt: 'p'\n"
    )

    def _probar(contenido: str) -> None:
        p = tmp_path / "c.yaml"
        p.write_text(contenido, encoding="utf-8")
        with pytest.raises(ValueError, match="connect_timeout_seconds"):
            load_capture_settings(p)

    _probar(base_yaml)  # clave ausente
    _probar(f"{base_yaml}connect_timeout_seconds: 0\n")
    _probar(f"{base_yaml}connect_timeout_seconds: -2.0\n")
    _probar(f"{base_yaml}connect_timeout_seconds: true\n")
    _probar(f"{base_yaml}connect_timeout_seconds: cinco\n")


def test_transcription_sin_clave_devuelve_cached_stt_local_y_avisa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Caso 27 (§B.5): Sin clave devuelve CachedSTT local con identidad local:... y aviso."""
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcripts import CachedSTT
    from aida.run import _transcription

    key = Fernet.generate_key().decode()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("AIDA_STT_API_KEY", raising=False)

    ajustes = replace(load_capture_settings(), provider="cloud")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: ajustes)

    res = _transcription(MagicMock(), key)
    assert isinstance(res, CachedSTT)
    assert res._model.startswith("local:")
    assert "falta AIDA_STT_API_KEY" in capsys.readouterr().err


def test_transcription_con_clave_devuelve_fallback_con_caches_distintas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caso 28 (§B.5): Con clave devuelve FallbackTranscription con identidades
    distintas en primario y respaldo."""
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcription import FallbackTranscription
    from aida.arms.health.transcripts import CachedSTT
    from aida.run import _transcription

    key = Fernet.generate_key().decode()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AIDA_STT_API_KEY", "test-key")

    ajustes = replace(load_capture_settings(), provider="cloud")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: ajustes)

    fb = cast(Any, _transcription(MagicMock(), key))
    assert isinstance(fb, FallbackTranscription)
    assert isinstance(fb._primary, CachedSTT)
    assert isinstance(fb._fallback, CachedSTT)
    assert fb._primary._model.startswith("cloud:")
    assert fb._fallback._model.startswith("local:")
    assert fb._primary._model != fb._fallback._model


def test_transcription_caida_respaldo_y_recuperacion_genera_dos_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caso 29 (§B.5, I-138): Mismo audio con caída y luego recuperación
    genera 2 entradas en caché."""
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcription import CloudTranscriptionError
    from aida.run import _transcription

    key = Fernet.generate_key().decode()
    cache_dir = tmp_path / ".cache" / "aida" / "transcripciones"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AIDA_STT_API_KEY", "test-key")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF audio data")

    # 1. Nube falla -> responde respaldo
    mock_local = MagicMock()
    mock_local.transcribe_segments.return_value = (TranscriptSegment(0.0, 1.0, "Texto local"),)
    ajustes = replace(load_capture_settings(), provider="cloud")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: ajustes)

    fb1 = cast(Any, _transcription(mock_local, key))
    fb1._primary._stt.transcribe_segments = MagicMock(
        side_effect=CloudTranscriptionError("nube caida")
    )
    res1 = fb1.transcribe_segments(audio)
    assert res1[0].text == "Texto local"
    assert len(list(cache_dir.glob("*.transcript"))) == 1

    # 2. Nube responde -> debe llamar a la nube y generar segunda entrada
    fb2 = cast(Any, _transcription(mock_local, key))
    fb2._primary._stt.transcribe_segments = MagicMock(
        return_value=(TranscriptSegment(0.0, 1.0, "Texto nube"),)
    )
    res2 = fb2.transcribe_segments(audio)
    assert res2[0].text == "Texto nube"
    assert fb2._primary._stt.transcribe_segments.call_count == 1
    assert len(list(cache_dir.glob("*.transcript"))) == 2


def test_transcription_reanudar_sin_red_usa_cache_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caso 30 (§B.5): Reanudar sin red (plan 65): con nube cayendo dos veces,
    la segunda no retranscribe en local."""
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcription import CloudTranscriptionError
    from aida.run import _transcription

    key = Fernet.generate_key().decode()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AIDA_STT_API_KEY", "test-key")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF audio data")

    mock_local = MagicMock()
    mock_local.transcribe_segments.return_value = (TranscriptSegment(0.0, 1.0, "Texto local"),)
    ajustes = replace(load_capture_settings(), provider="cloud")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: ajustes)

    fb1 = cast(Any, _transcription(mock_local, key))
    fb1._primary._stt.transcribe_segments = MagicMock(
        side_effect=CloudTranscriptionError("nube caida")
    )
    fb1.transcribe_segments(audio)
    assert mock_local.transcribe_segments.call_count == 1

    # Segunda corrida sin red
    fb2 = cast(Any, _transcription(mock_local, key))
    fb2._primary._stt.transcribe_segments = MagicMock(
        side_effect=CloudTranscriptionError("nube caida")
    )
    fb2.transcribe_segments(audio)
    assert mock_local.transcribe_segments.call_count == 1


def test_cached_stt_identidad_incluye_device_y_compute_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caso 31 (§B.5): La identidad local incluye device y compute_type."""
    from aida.arms.health.capture import load_capture_settings
    from aida.run import _transcription

    key = Fernet.generate_key().decode()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    aj1 = replace(load_capture_settings(), provider="local", device="cpu", compute_type="int8")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: aj1)
    stt1 = cast(Any, _transcription(MagicMock(), key))

    aj2 = replace(load_capture_settings(), provider="local", device="cpu", compute_type="float32")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: aj2)
    stt2 = cast(Any, _transcription(MagicMock(), key))

    aj3 = replace(load_capture_settings(), provider="local", device="cuda", compute_type="int8")
    monkeypatch.setattr("aida.arms.health.capture.load_capture_settings", lambda: aj3)
    stt3 = cast(Any, _transcription(MagicMock(), key))

    assert stt1._model != stt2._model
    assert stt1._model != stt3._model


def test_run_no_expone_cached_stt() -> None:
    """Caso 32 (§B.5): aida.run no expone _cached_stt."""
    import aida.run

    assert not hasattr(aida.run, "_cached_stt")


def test_consultation_cli_ui_sin_review_devuelve_error_dos(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Caso 24 (§B.4): --ui sin --review imprime error y devuelve 2."""
    audio_path = tmp_path / "dummy.wav"
    audio_path.write_bytes(b"RIFF dummy")
    hist_path = tmp_path / "dummy.txt"
    hist_path.write_text("historia")

    codigo = main(
        [
            "--consultation",
            "--ui",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            "doc_test_ui_no_review",
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 2
    assert "aida: error: --ui requiere --review" in captured.err


def test_consultation_cli_ui_con_review_llama_collect_review_y_pasa_elapsed(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 25 (§B.4): Con --ui, la CLI llama a collect_review y pasa
    elapsed_seconds a Consultation.review.
    """
    from aida.arms.health.consultation import Consultation
    from aida.arms.health.review import Answer, ReviewItem, Verdict
    from aida.arms.health.review_ui import ReviewSubmission

    _clean_llm_env(monkeypatch)
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _DummySTT:
        def transcribe_segments(self, path: Path) -> tuple[TranscriptSegment, ...]:
            return (
                TranscriptSegment(0.0, 2.0, "consulta hoy por dolor en rodilla derecha"),
            )

    class _DummyOCR:
        def read(self, path: Path) -> str:
            return "Historia clínica previa: sin antecedentes."

    monkeypatch.setattr(
        "aida.arms.health.capture.build_local_capture",
        lambda: (_DummySTT(), _DummyOCR()),
    )

    captured_items: list[ReviewItem] = []

    def mock_collect(
        items: tuple[ReviewItem, ...], *, settings: Any
    ) -> ReviewSubmission:
        captured_items.extend(items)
        answers = {item.ref: Answer(Verdict.CONFIRMED) for item in items}
        return ReviewSubmission(answers=answers, elapsed_seconds=42.5)

    monkeypatch.setattr("aida.arms.health.review_ui.collect_review", mock_collect)

    original_review = Consultation.review
    recorded_elapsed: list[float | None] = []

    def spy_review(
        self: Any,
        prepared: Any,
        *,
        reviewer: Any,
        elapsed_seconds: float | None = None,
    ) -> Any:
        recorded_elapsed.append(elapsed_seconds)
        return original_review(
            self, prepared, reviewer=reviewer, elapsed_seconds=elapsed_seconds
        )

    monkeypatch.setattr(Consultation, "review", spy_review)

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio")
    hist_path = tmp_path / "historia.txt"
    hist_path.write_text("Historia clínica previa.")

    codigo = main(
        [
            "--consultation",
            "--segment",
            "--review",
            "--ui",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            "doc_cli_ui_success",
        ]
    )
    capsys.readouterr()
    assert codigo == 0
    assert len(captured_items) > 0
    assert recorded_elapsed == [42.5]

    metrics_path = tmp_path / ".local" / "share" / "aida" / "revisiones.jsonl"
    assert metrics_path.exists()
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["seconds"] == 42.5
    assert data["completed"] is True


def test_consultation_cli_ui_review_abandoned_avisa_y_no_escribe_cache_ni_export(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 26 (§B.4): ReviewAbandoned imprime aviso en stderr, sale con 1,
    y no escribe caché ni archivos de export.
    """
    from aida.arms.health.review_ui import ReviewAbandoned, load_ui_settings

    _clean_llm_env(monkeypatch)
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _DummySTT:
        def transcribe_segments(self, path: Path) -> tuple[TranscriptSegment, ...]:
            return (
                TranscriptSegment(0.0, 2.0, "consulta hoy por dolor en rodilla derecha"),
            )

    class _DummyOCR:
        def read(self, path: Path) -> str:
            return "Historia clínica previa."

    monkeypatch.setattr(
        "aida.arms.health.capture.build_local_capture",
        lambda: (_DummySTT(), _DummyOCR()),
    )

    def mock_abandoned(items: Any, *, settings: Any) -> Any:
        raise ReviewAbandoned("sesión cerrada por el usuario")

    monkeypatch.setattr("aida.arms.health.review_ui.collect_review", mock_abandoned)

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio")
    hist_path = tmp_path / "historia.txt"
    hist_path.write_text("Historia clínica previa.")
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    owner = "doc_cli_ui_abandoned"
    codigo = main(
        [
            "--consultation",
            "--segment",
            "--review",
            "--ui",
            "--export",
            str(export_dir),
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            owner,
        ]
    )
    captured = capsys.readouterr()
    assert codigo == 1

    ui_settings = load_ui_settings()
    assert ui_settings.wording.abandoned_text in captured.err

    cache_path = tmp_path / ".cache" / "aida" / "consultas" / f"{owner}.cache"
    assert not cache_path.exists()
    assert list(export_dir.iterdir()) == []


def test_consultation_cli_sin_ui_usa_stdin_y_no_llama_collect_review(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caso 27 (§B.4): Sin --ui, se usa _review_from_stdin y nunca se llama collect_review."""
    _clean_llm_env(monkeypatch)
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("AIDA_CONSULT_CACHE_KEY", key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    class _DummySTT:
        def transcribe_segments(self, path: Path) -> tuple[TranscriptSegment, ...]:
            return (
                TranscriptSegment(0.0, 2.0, "consulta hoy por dolor en rodilla derecha"),
            )

    class _DummyOCR:
        def read(self, path: Path) -> str:
            return "Historia clínica previa."

    monkeypatch.setattr(
        "aida.arms.health.capture.build_local_capture",
        lambda: (_DummySTT(), _DummyOCR()),
    )

    collect_called = False

    def mock_collect(items: Any, *, settings: Any) -> Any:
        nonlocal collect_called
        collect_called = True
        raise AssertionError("collect_review no debió llamarse sin --ui")

    monkeypatch.setattr("aida.arms.health.review_ui.collect_review", mock_collect)
    monkeypatch.setattr("builtins.input", lambda prompt: "c")

    audio_path = tmp_path / "consulta.wav"
    audio_path.write_bytes(b"RIFF dummy audio")
    hist_path = tmp_path / "historia.txt"
    hist_path.write_text("Historia clínica previa.")

    owner = "doc_cli_stdin_review"
    codigo = main(
        [
            "--consultation",
            "--segment",
            "--review",
            "--audio",
            str(audio_path),
            "--history",
            str(hist_path),
            "--owner",
            owner,
        ]
    )
    capsys.readouterr()
    assert codigo == 0
    assert not collect_called

    cache_path = tmp_path / ".cache" / "aida" / "consultas" / f"{owner}.cache"
    assert cache_path.exists()















