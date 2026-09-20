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
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

import aida.engine.loop as loop
from aida import __version__
from aida.cognition.briefing import SpokenReply, fallback_free_chat_reply
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
            mock_seg.text = self.text
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
    assert captured.err == ""

    # Comprueba que el caché cifrado fue escrito en disco y es descifrable con la clave
    cache_file = tmp_path / ".cache" / "aida" / "consultas" / "paciente_test_cli.cache"
    assert cache_file.exists()
    contenido_descifrado = Fernet(key.encode()).decrypt(cache_file.read_bytes()).decode("utf-8")
    assert "- El paciente dijo:" in contenido_descifrado


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
            mock_seg.text = self.text
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
    assert captured.err == ""


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
            mock_seg.text = self.text
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
    assert captured.err == ""


# ---------------------------------------------------------------------------
# CLI --converse --mic (Unidad U10 / 33-plan-consultorio-voz)
# ---------------------------------------------------------------------------


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
        def __init__(self, text: str) -> None:
            self.text = text

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
        def __init__(self, text: str) -> None:
            self.text = text

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
        def __init__(self, text: str) -> None:
            self.text = text

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









