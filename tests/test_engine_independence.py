from __future__ import annotations

import ast
from pathlib import Path


def test_engine_no_importa_arms_ni_platform() -> None:
    engine_dir = Path(__file__).resolve().parents[1] / "src" / "aida" / "engine"
    archivos = list(engine_dir.glob("*.py"))
    assert archivos, "no se encontraron archivos en aida/engine"

    for py_file in archivos:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("aida.arms"), (
                        f"{py_file.name} importa {alias.name}"
                    )
                    assert not alias.name.startswith("aida.platform"), (
                        f"{py_file.name} importa {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                modulo = node.module or ""
                assert not modulo.startswith("aida.arms"), (
                    f"{py_file.name} importa {modulo}"
                )
                assert not modulo.startswith("aida.platform"), (
                    f"{py_file.name} importa {modulo}"
                )
