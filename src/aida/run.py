"""Punto de entrada de la línea de comandos.

    uv run aida --version

Aquí no hay nada más que ejecutar todavía: el bucle de deliberación llega en
unidades posteriores. `main` no llama a `sys.exit` para poder probarse sin
terminar el proceso del test.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from aida import __version__
from aida.engine.pack import PackError, describe_pack, load_pack

USO_BREVE = "aida: asistente determinista. Usa --version o --pack <ruta>."
"""Sin argumentos no hay nada que decidir todavía: se informa y se sale con éxito."""


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aida", description="A.I.D.A.: asistente determinista.")
    parser.add_argument(
        "--version",
        action="store_true",
        help="imprime la versión instalada y termina",
    )
    parser.add_argument(
        "--pack",
        metavar="RUTA",
        help="carga, valida y resume el content pack en la ruta indicada",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _construir_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"aida {__version__}")
        return 0

    if args.pack:
        path = Path(args.pack)
        try:
            pack = load_pack(path)
            resumen = describe_pack(pack)
            elementos = ", ".join(f"{k}: {v}" for k, v in sorted(resumen.items()))
            print(f"pack {pack.id!r} ({elementos})")
            return 0
        except PackError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    print(USO_BREVE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
