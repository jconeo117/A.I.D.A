"""Pruebas del punto de entrada de la línea de comandos (`aida.run.main`).

Cubren la unidad U0 — Esqueleto del repositorio: la única superficie pública
en esta fase es `aida.__version__` y `aida.run.main`, así que estas pruebas
son, por ahora, toda la batería de la unidad.
"""

from __future__ import annotations

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
