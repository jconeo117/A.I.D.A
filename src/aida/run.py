"""Punto de entrada de la línea de comandos.

    uv run aida --version
    uv run aida --pack <ruta>
    uv run aida --pack <ruta> --explain

`main` no llama a `sys.exit` para poder probarse sin terminar el proceso
del test.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from aida import __version__
from aida.cognition.providers import build_provider
from aida.cognition.session import Session
from aida.engine.explain import explain_deliberation
from aida.engine.loop import Loop
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
    parser.add_argument(
        "--converse",
        action="store_true",
        help="abre una conversación por línea de comandos con el agente del pack",
    )
    parser.add_argument(
        "--llm",
        metavar="PROVEEDOR",
        default=None,
        help="fuerza el proveedor de voz (claude|gemini|ollama); por defecto, "
        "AIDA_LLM_PROVIDER o 'gemini'",
    )
    parser.add_argument(
        "--free-chat",
        action="store_true",
        help="permite charla libre cuando no hay nada urgente ni una petición reconocida",
    )
    parser.add_argument("--seed", type=int, default=0, help="semilla del generador determinista")
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

        if args.converse:
            if not pack.agents:
                print("El pack no declara ningún agente.", file=sys.stderr)
                return 1
            from aida.intent.classify import build_classifier

            agent_spec = pack.agents[0]
            loop = Loop.start(
                pack,
                agent_spec,
                seed=args.seed,
                inbox_capacity=64,
                intent=build_classifier(pack),
            )
            provider_name = args.llm or os.environ.get("AIDA_LLM_PROVIDER", "gemini")
            # Transparencia: si falta la credencial o el proveedor no existe,
            # advertir que opera en modo sin voz.
            provider = build_provider(provider_name)
            session = Session(
                loop=loop, provider=provider, free_chat_enabled=args.free_chat
            )
            if provider is None:
                print(
                    f"(sin voz: el proveedor {provider_name!r} no está disponible; "
                    "escribe 'salir' para terminar)"
                )
            else:
                print(f"(voz: {provider_name}; escribe 'salir' para terminar)")
            while True:
                try:
                    linea = input("> ")
                except EOFError:
                    break
                if linea.strip().lower() == "salir":
                    break
                print(session.turn(linea))
            return 0

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
