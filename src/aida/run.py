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
from typing import TYPE_CHECKING, Any, cast

from aida import __version__
from aida.cognition.providers import build_provider
from aida.cognition.session import Session
from aida.engine.explain import explain_deliberation
from aida.engine.loop import Loop
from aida.engine.pack import PackError, describe_pack, load_pack
from aida.engine.state import AgentState, ExecutionState

if TYPE_CHECKING:
    from aida.arms.health.consultation import Consultation

USO_BREVE = "aida: asistente determinista. Usa --version o --pack <ruta> [--explain]."
"""Sin argumentos no hay nada que decidir todavía: se informa y se sale con éxito."""

_RAIZ = Path(__file__).resolve().parents[2]


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
        "--consultation",
        action="store_true",
        help="ejecuta el brazo de consulta: transcribe, coteja CIE-10, redacta un resumen",
    )
    parser.add_argument("--audio", metavar="RUTA", help="audio de la consulta (--consultation)")
    parser.add_argument(
        "--history", metavar="RUTA", help="documento de antecedentes previos (--consultation)"
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="activa la diarización de hablantes en la consulta",
    )
    parser.add_argument(
        "--owner",
        metavar="ID",
        default="sujeto_01",
        help="identificador del sujeto, para aislamiento del caché (--consultation)",
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
    parser.add_argument(
        "--mic",
        action="store_true",
        help="con --converse: escucha por micrófono en vez de leer del teclado (extra 'capture')",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=8.0,
        help="segundos de silencio que terminan la escucha en modo consulta",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=180.0, help="tope de escucha en modo consulta"
    )
    parser.add_argument("--seed", type=int, default=0, help="semilla del generador determinista")
    return parser


def _check_consult_cache_key() -> str | None:
    key = os.environ.get("AIDA_CONSULT_CACHE_KEY")
    if not key:
        print(
            "aida: error: falta AIDA_CONSULT_CACHE_KEY (clave Fernet del caché cifrado)",
            file=sys.stderr,
        )
        return None

    from cryptography.fernet import Fernet

    try:
        Fernet(key.encode())
    except ValueError:
        print(
            "aida: error: AIDA_CONSULT_CACHE_KEY no es una clave Fernet válida",
            file=sys.stderr,
        )
        return None
    return key


def _build_consultation(
    args: argparse.Namespace,
    *,
    stt: object,
    ocr: object,
    diarizer: object | None = None,
    key: str | None = None,
) -> Consultation | None:
    """`None` tras imprimir el error si falta o no vale AIDA_CONSULT_CACHE_KEY.
    Recibe STT y OCR ya construidos: quien llama decide si los comparte
    con el bucle de conversación (rama --mic) o los crea solo para esto.
    """
    if key is None:
        key = _check_consult_cache_key()
        if key is None:
            return None

    from aida.arms.health.cache import ConsultationCache
    from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT
    from aida.arms.health.codes import load_catalog
    from aida.arms.health.consultation import Consultation
    from aida.arms.health.match import build_index
    from aida.arms.health.wording import default_summary_wording
    from aida.engine.events import EventLog

    catalog = load_catalog(
        _RAIZ / "data" / "health" / "cie10" / "codes.json",
        _RAIZ / "data" / "health" / "cie10" / "cie-10.csv",
    )
    return Consultation(
        wording=default_summary_wording(),
        stt=stt if isinstance(stt, LocalWhisperSTT) else cast(Any, stt),
        ocr=ocr if isinstance(ocr, LocalTesseractOCR) else cast(Any, ocr),
        catalog_index=build_index(catalog),
        provider=build_provider(args.llm or os.environ.get("AIDA_LLM_PROVIDER", "gemini")),
        cache=ConsultationCache(
            Path.home() / ".cache" / "aida" / "consultas", key=key.encode()
        ),
        log=EventLog(),
        diarizer=cast(Any, diarizer),
    )


CONVERSATION_SILENCE_SECONDS = 1.5
CONVERSATION_MAX_SECONDS = 30.0


def _check_mic_dependencies() -> bool:
    import importlib.util

    for name in ("sounddevice", "faster_whisper"):
        if sys.modules.get(name) is None and name in sys.modules:
            print(
                "aida: error: --mic requiere el extra 'capture' (uv sync --extra capture)",
                file=sys.stderr,
            )
            return False
        try:
            if importlib.util.find_spec(name) is None:
                print(
                    "aida: error: --mic requiere el extra 'capture' (uv sync --extra capture)",
                    file=sys.stderr,
                )
                return False
        except (ValueError, AttributeError):
            print(
                "aida: error: --mic requiere el extra 'capture' (uv sync --extra capture)",
                file=sys.stderr,
            )
            return False

    try:
        from aida.voice.listen import check_capture_available

        if not check_capture_available():
            print(
                "aida: error: --mic requiere el extra 'capture' (uv sync --extra capture)",
                file=sys.stderr,
            )
            return False
    except ImportError:
        print(
            "aida: error: --mic requiere el extra 'capture' (uv sync --extra capture)",
            file=sys.stderr,
        )
        return False
    return True


def _converse_by_voice(args: argparse.Namespace, session: Session) -> int:
    if not _check_mic_dependencies():
        return 1

    import tempfile

    from aida.arms.health.capture import build_local_capture
    from aida.engine.ontology import AgentId
    from aida.voice.commands import CommandCatalog, OperatorCommand
    from aida.voice.listen import (
        MicrophoneSource,
        SilenceGatedListener,
        vad_from_faster_whisper,
        write_wav,
    )

    phrases: dict[OperatorCommand, tuple[str, ...]] = {
        OperatorCommand.EXIT: ("salir", "terminar", "hasta luego"),
    }

    consultation = None
    history_text = ""
    if args.history:
        history_path = Path(args.history)
        if not history_path.exists():
            print(f"aida: error: no existe {history_path}", file=sys.stderr)
            return 1

        from aida.arms.health.wording import switch_to_consult_phrases

        stt, ocr = build_local_capture()
        consultation = _build_consultation(args, stt=stt, ocr=ocr)
        if consultation is None:
            return 1
        history_text = ocr.read(history_path)  # una sola vez, antes de escuchar a nadie
        phrases[OperatorCommand.SWITCH_MODE] = switch_to_consult_phrases()
    else:
        stt, _ocr = build_local_capture()

    commands = CommandCatalog.from_phrases(phrases)
    tmp = Path(tempfile.mkdtemp(prefix="aida-voz-"))

    try:
        with MicrophoneSource() as source:
            listener = SilenceGatedListener(chunk_source=source, vad=vad_from_faster_whisper())
            print("(escuchando; di 'salir' para terminar)")
            while True:
                audio = listener.listen(
                    silence_seconds=CONVERSATION_SILENCE_SECONDS,
                    max_seconds=CONVERSATION_MAX_SECONDS,
                )
                if audio.size == 0:
                    continue
                utterance = stt.transcribe(write_wav(audio, tmp / "operador.wav"))
                command = commands.recognize(utterance)
                if command is OperatorCommand.EXIT:
                    return 0
                if command is OperatorCommand.SWITCH_MODE and consultation is not None:
                    print("(modo consulta: escuchando)")
                    consulted_audio = listener.listen(
                        silence_seconds=args.silence_seconds, max_seconds=args.max_seconds
                    )
                    if consulted_audio.size == 0:
                        print("(no se oyó nada)")
                        continue
                    result = consultation.run_with_history(
                        owner=AgentId(args.owner),
                        audio_path=write_wav(consulted_audio, tmp / "consultado.wav"),
                        history_text=history_text,
                    )
                    print(result.summary_markdown)
                    print("(de vuelta en conversación)")
                    continue
                print(session.turn(utterance))
    except KeyboardInterrupt:
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _construir_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"aida {__version__}")
        return 0

    if args.consultation:
        if not args.audio or not args.history:
            print("aida: error: --consultation requiere --audio y --history", file=sys.stderr)
            return 1

        if args.diarize:
            if not os.environ.get("AIDA_HF_TOKEN"):
                print("aida: error: --diarize requiere AIDA_HF_TOKEN", file=sys.stderr)
                return 1
            from aida.voice.diarize import build_local_diarizer

            diarizer = build_local_diarizer()
        else:
            diarizer = None

        key = _check_consult_cache_key()
        if not key:
            return 1

        from aida.arms.health.capture import build_local_capture
        from aida.engine.ontology import AgentId

        stt, ocr = build_local_capture()
        consultation = _build_consultation(args, stt=stt, ocr=ocr, diarizer=diarizer, key=key)
        if consultation is None:
            return 1
        result = consultation.run(
            owner=AgentId(args.owner),
            audio_path=Path(args.audio),
            history_path=Path(args.history),
        )
        print(result.summary_markdown)
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
            if args.mic:
                return _converse_by_voice(args, session)
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
