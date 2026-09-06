"""Pruebas del punto de entrada de la línea de comandos (`aida.run.main`).

Cubren la unidad U0 — Esqueleto del repositorio: la única superficie pública
en esta fase es `aida.__version__` y `aida.run.main`, así que estas pruebas
son, por ahora, toda la batería de la unidad.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from aida import __version__
from aida.run import main


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

