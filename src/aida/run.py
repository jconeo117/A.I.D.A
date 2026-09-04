"""Punto de entrada de la línea de comandos.

    uv run aida --version

Aquí no hay nada más que ejecutar todavía: el bucle de deliberación llega en
unidades posteriores. `main` no llama a `sys.exit` para poder probarse sin
terminar el proceso del test.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from aida import __version__

USO_BREVE = "aida: asistente determinista. Usa --version para ver la versión instalada."
"""Sin argumentos no hay nada que decidir todavía: se informa y se sale con éxito."""


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aida", description="A.I.D.A.: asistente determinista.")
    parser.add_argument(
        "--version",
        action="store_true",
        help="imprime la versión instalada y termina",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _construir_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"aida {__version__}")
        return 0

    print(USO_BREVE)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
