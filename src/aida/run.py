"""Punto de entrada de la línea de comandos.

    uv run aida --version
    uv run aida --pack <ruta>
    uv run aida --pack <ruta> --explain

`main` no llama a `sys.exit` para poder probarse sin terminar el proceso
del test.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from aida import __version__
from aida.engine.explain import explain_deliberation
from aida.engine.pack import PackError, describe_pack, load_pack
from aida.engine.state import AgentState, ExecutionState

USO_BREVE = "aida: asistente determinista. Usa --version o --pack <ruta> [--explain]."
"""Sin argumentos no hay nada que decidir todavía: se informa y se sale con éxito."""


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aida", description="A.I.D.A.: asistente determinista."
    )
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
    parser.add_argument(
        "--explain",
        action="store_true",
        help="delibera una vez con el agente del pack e imprime la traza causal",
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
        except PackError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if args.explain:
            if not pack.agents:
                print("El pack no declara ningún agente para deliberar.", file=sys.stderr)
                return 1

            agent_spec = pack.agents[0]
            # `from_spec` es la puerta única de construcción de un agente desde
            # el pack: necesidades iniciales, metas bloqueadas o activas según
            # `requires_goal`, gustos y recomposición, todo en un solo sitio.
            # Rehacerlo aquí a mano fue exactamente lo que dejó a `--explain`
            # deliberando sin los gustos del agente (`I-044`).
            agent = AgentState.from_spec(pack, agent_spec)
            execution = ExecutionState(agent=agent, tick=0)
            situations = execution.active_situations(pack.situations)

            decision, trace = explain_deliberation(
                execution, pack, phase=None, situations=situations
            )
            if decision is None or trace is None:
                print("No hay ninguna opción disponible para deliberar.", file=sys.stderr)
                return 1

            print(trace.format_human())
            return 0

        resumen = describe_pack(pack)
        elementos = ", ".join(f"{k}: {v}" for k, v in sorted(resumen.items()))
        print(f"pack {pack.id!r} ({elementos})")
        return 0

    if args.explain:
        print("aida: error: --explain requiere especificar un pack con --pack", file=sys.stderr)
        return 1

    print(USO_BREVE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
