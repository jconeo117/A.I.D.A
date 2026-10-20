from __future__ import annotations

import ast
import sys
from pathlib import Path


def test_relay_solo_importa_stdlib_cryptography_y_relay() -> None:
    relay_dir = Path(__file__).resolve().parents[1] / "src" / "aida" / "relay"
    assert relay_dir.is_dir()

    modulos_py = list(relay_dir.glob("*.py"))
    assert len(modulos_py) >= 5

    permitidos_base = set(sys.stdlib_module_names) | {"cryptography", "__future__"}

    for archivo in modulos_py:
        arbol = ast.parse(archivo.read_text(encoding="utf-8"), filename=str(archivo))
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                for alias in nodo.names:
                    raiz = alias.name.split(".")[0]
                    assert raiz in permitidos_base or alias.name.startswith("aida.relay"), (
                        f"{archivo.name} importa {alias.name} fuera de "
                        "stdlib/cryptography/aida.relay"
                    )
            elif isinstance(nodo, ast.ImportFrom):
                if nodo.level and nodo.level > 0:
                    # Import relativo dentro de aida.relay
                    continue
                modulo = nodo.module or ""
                raiz = modulo.split(".")[0]
                assert raiz in permitidos_base or modulo.startswith("aida.relay"), (
                    f"{archivo.name} importa desde {modulo} fuera de stdlib/cryptography/aida.relay"
                )
