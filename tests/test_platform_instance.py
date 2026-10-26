"""Tests de instancia única y control de ejecución (Plan 78b §B).

Cubre T1, T2, T3, T4, T5, T6 y T11:
- T1: acquire dos veces en la misma carpeta (la segunda da None, tras release se readquiere).
- T2: acquire borra aida.url previo.
- T3: record y running_url devuelven la URL; release borra aida.url.
- T4: subproceso que muere con kill() libera el bloqueo del SO.
- T5: _open_platform con bloqueo previo devuelve 0 y abre URL viva en navegador.
- T6: _open_platform registra URL viva durante ejecución y la limpia al cerrar.
- T11: coherencia de versión entre pyproject.toml y aida.__version__.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aida import __version__
from aida.platform.instance import URL_NAME, acquire, running_url


def test_acquire_dos_veces_misma_carpeta_segunda_da_none_y_tras_release_se_relee(
    tmp_path: Path,
) -> None:
    """T1 (§B): acquire dos veces en la misma carpeta: la segunda da None;
    tras release, se puede volver a adquirir."""
    candado1 = acquire(tmp_path)
    assert candado1 is not None

    candado2 = acquire(tmp_path)
    assert candado2 is None

    candado1.release()
    candado3 = acquire(tmp_path)
    assert candado3 is not None
    candado3.release()


def test_acquire_borra_aida_url_previo(tmp_path: Path) -> None:
    """T2 (§B): acquire borra un aida.url previo."""
    url_file = tmp_path / URL_NAME
    url_file.write_text("http://127.0.0.1:9999/sesion-previa", encoding="utf-8")
    assert url_file.exists()

    candado = acquire(tmp_path)
    assert candado is not None
    assert not url_file.exists()
    candado.release()


def test_record_y_running_url_devuelven_url_y_release_borra_aida_url(
    tmp_path: Path,
) -> None:
    """T3 (§B): record y luego running_url devuelven la URL; release borra aida.url."""
    candado = acquire(tmp_path)
    assert candado is not None
    url_esperada = "http://127.0.0.1:4567/app"

    candado.record(url_esperada)
    assert running_url(tmp_path) == url_esperada
    assert (tmp_path / URL_NAME).read_text(encoding="utf-8") == url_esperada

    candado.release()
    assert not (tmp_path / URL_NAME).exists()
    assert running_url(tmp_path) is None


def test_subproceso_bloquea_y_al_morir_se_libera(tmp_path: Path) -> None:
    """T4 (§B): Un subproceso (sys.executable -c) adquiere el bloqueo y se mata con kill();
    después, el padre adquiere."""
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from aida.platform.instance import acquire\n"
        "lock = acquire(Path(sys.argv[1]))\n"
        "if lock is None: sys.exit(1)\n"
        "sys.stdout.write('LISTO\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout is not None
        linea = proc.stdout.readline()
        assert "LISTO" in linea

        candado_padre_intento = acquire(tmp_path)
        assert candado_padre_intento is None

        proc.kill()
        proc.wait(timeout=5.0)

        candado_padre_exito = acquire(tmp_path)
        assert candado_padre_exito is not None
        candado_padre_exito.release()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_open_platform_con_bloqueo_ya_tomado_y_url_registrada(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T5 (§B): _open_platform con el bloqueo ya tomado y una URL registrada:
    devuelve 0, el open_browser espía recibe la URL, y ni run_shell ni health_arm se llaman."""
    from aida.run import _open_platform

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "aida.run.user_dir",
        lambda kind: data_dir if kind == "data" else tmp_path / kind,
    )

    candado = acquire(data_dir)
    assert candado is not None
    url_viva = "http://127.0.0.1:12345/sesion-viva"
    candado.record(url_viva)

    urls_abiertas: list[str] = []
    run_shell_llamado = False

    arm_mock = MagicMock()
    arm_mock.id = "dummy"
    arm_mock.description = "Dummy"
    arm_mock.icon = "badge"
    arm_mock.entry = lambda: (lambda *a, **kw: MagicMock())
    monkeypatch.setattr("aida.platform.arms.discover", lambda: [arm_mock])

    def fake_run_shell(*args: Any, **kwargs: Any) -> None:
        nonlocal run_shell_llamado
        run_shell_llamado = True

    monkeypatch.setattr("aida.platform.web.shell.run_shell", fake_run_shell)

    args = argparse.Namespace()
    codigo = _open_platform(args, open_browser=urls_abiertas.append)

    assert codigo == 0
    assert urls_abiertas == [url_viva]
    assert run_shell_llamado is False

    candado.release()


def test_open_platform_primera_instancia_registra_url_y_libera_al_cerrar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T6 (§B): _open_platform con un run_shell simulado que llama a
    kwargs["open_browser"]("http://127.0.0.1:1/t"): dentro, aida.url existe con esa URL;
    después, no existe y acquire funciona."""
    from aida.run import _open_platform

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "aida.run.user_dir",
        lambda kind: data_dir if kind == "data" else tmp_path / kind,
    )

    url_dentro: str | None = None
    url_simulada = "http://127.0.0.1:1/t"
    navegador_llamado: list[str] = []

    def fake_run_shell(*args: Any, **kwargs: Any) -> None:
        nonlocal url_dentro
        abrir_fn = kwargs.get("open_browser")
        assert abrir_fn is not None
        abrir_fn(url_simulada)
        url_file = data_dir / URL_NAME
        if url_file.exists():
            url_dentro = url_file.read_text(encoding="utf-8")

    monkeypatch.setattr("aida.platform.web.shell.run_shell", fake_run_shell)

    arm_mock = MagicMock()
    arm_mock.id = "dummy"
    arm_mock.description = "Dummy"
    arm_mock.icon = "badge"
    arm_mock.entry = lambda: (lambda *a, **kw: MagicMock())
    monkeypatch.setattr("aida.platform.arms.discover", lambda: [arm_mock])

    args = argparse.Namespace()
    codigo = _open_platform(args, open_browser=navegador_llamado.append)

    assert codigo == 0
    assert navegador_llamado == [url_simulada]
    assert url_dentro == url_simulada, "aida.url no contenía la URL esperada durante run_shell"
    assert not (data_dir / URL_NAME).exists(), "aida.url no se borró al salir de _open_platform"

    nuevo_candado = acquire(data_dir)
    assert nuevo_candado is not None, "El bloqueo no se liberó tras finalizar _open_platform"
    nuevo_candado.release()


def test_pyproject_version_coincide_con_aida_version() -> None:
    """T11 (§B): La versión de pyproject.toml (leída con tomllib) es igual a aida.__version__."""
    raiz = Path(__file__).resolve().parents[1]
    pyproject_path = raiz / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    version_pyproject = data["project"]["version"]

    assert version_pyproject == __version__
