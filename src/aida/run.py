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
    from aida.arms.health.review import Answer, ReviewItem

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
        "--segment",
        action="store_true",
        help="agrupa la transcripción en secciones clínicas (--consultation)",
    )
    parser.add_argument(
        "--print-model",
        action="store_true",
        help="imprime el Clinical Model en JSON tras el resumen (requiere --segment)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="extrae duración/lateralidad/quién lo dijo (requiere --segment)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="confirmar, editar o rechazar cada hallazgo y código antes de guardar "
        "(requiere --segment)",
    )
    parser.add_argument(
        "--review-stats",
        nargs="?",
        const="",
        metavar="RUTA",
        help="resume las revisiones registradas (por defecto, el archivo de --review)",
    )
    parser.add_argument(
        "--export",
        nargs="?",
        const="",
        metavar="RUTA",
        help="escribe la consulta revisada en un archivo (sin RUTA, en la carpeta por defecto)",
    )
    parser.add_argument(
        "--check-capture",
        action="store_true",
        help="comprueba modelo, dispositivo, OCR y entrada de audio, y termina",
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


def _review_metrics_path() -> Path:
    """Un solo sitio para la ruta: la escribe --review y la lee --review-stats."""
    return Path.home() / ".local" / "share" / "aida" / "revisiones.jsonl"


def _transcription(stt: object, ajustes: object) -> object:
    """El transcriptor que verá la consulta: la nube si el pack la declara, con
    el modelo local de respaldo.

    Si falta la credencial, no se avisa a medias ni se falla: se usa el local y
    se dice. Un consultorio sin clave tiene que poder trabajar."""
    import os

    from aida.arms.health.transcription import CloudWhisperSTT, FallbackTranscription

    settings: Any = ajustes
    if settings.provider != "cloud":
        return stt
    api_key = os.environ.get("AIDA_STT_API_KEY")
    if not api_key:
        print(
            "aida: aviso: falta AIDA_STT_API_KEY; se transcribe en local",
            file=sys.stderr,
        )
        return stt

    import httpx

    http = httpx.Client(
        base_url=settings.cloud_base_url,
        timeout=settings.timeout_seconds,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    nube = CloudWhisperSTT(
        http, model=settings.cloud_model, language=settings.cloud_language
    )

    def aviso(notice: object) -> None:
        print(
            f"aida: aviso: la transcripción en la nube falló, se usa el modelo local "
            f"({getattr(notice, 'reason', '')[:120]})",
            file=sys.stderr,
        )

    return FallbackTranscription(nube, cast(Any, stt), on_fallback=aviso)


def _cached_stt(stt: object, key: str) -> object:
    """El STT con la caché de transcripciones delante. Medido: transcribir siete
    minutos cuesta ~105 s y el resto de la tubería 0,8, así que reanudar una
    revisión cortada sin esto significa esperar otra vez."""
    from aida.arms.health.capture import load_capture_settings
    from aida.arms.health.transcripts import CachedSTT, TranscriptCache

    ajustes = load_capture_settings()
    stt = _transcription(stt, ajustes)
    cache = TranscriptCache(
        Path.home() / ".cache" / "aida" / "transcripciones", key=key.encode()
    )
    cache.purge_expired()
    return CachedSTT(
        stt,
        cache,
        model=f"{ajustes.provider}:{ajustes.cloud_model}:{ajustes.model}",
        no_speech_threshold=ajustes.no_speech_threshold,
    )


def _build_consultation(
    args: argparse.Namespace,
    *,
    stt: object,
    ocr: object,
    diarizer: object | None = None,
    segmenter: object | None = None,
    extractor: object | None = None,
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
    from aida.arms.health.capture import LocalTesseractOCR
    from aida.arms.health.codes import load_catalog
    from aida.arms.health.consultation import Consultation
    from aida.arms.health.history import load_history_layout
    from aida.arms.health.lexicon import load_lexicon
    from aida.arms.health.match import build_index, load_matching_settings
    from aida.arms.health.metrics import ReviewMetricsStore
    from aida.arms.health.negation import load_negation
    from aida.arms.health.wording import default_summary_wording
    from aida.engine.events import EventLog

    cotejo = load_matching_settings()
    catalog = load_catalog(
        _RAIZ / "data" / "health" / "cie10" / "codes.json",
        _RAIZ / "data" / "health" / "cie10" / "cie-10.csv",
    )
    return Consultation(
        wording=default_summary_wording(),
        stt=cast(Any, _cached_stt(stt, key)),
        ocr=ocr if isinstance(ocr, LocalTesseractOCR) else cast(Any, ocr),
        catalog_index=build_index(catalog, stopwords=cotejo.stopwords),
        match_threshold=cotejo.threshold,
        match_top_n=cotejo.top_n,
        provider=build_provider(args.llm or os.environ.get("AIDA_LLM_PROVIDER", "gemini")),
        cache=ConsultationCache(
            Path.home() / ".cache" / "aida" / "consultas", key=key.encode()
        ),
        log=EventLog(),
        diarizer=cast(Any, diarizer),
        segmenter=cast(Any, segmenter),
        extractor=cast(Any, extractor),
        lexicon=load_lexicon(),
        negation=load_negation(),
        history_layout=load_history_layout(),
        review_metrics=ReviewMetricsStore(_review_metrics_path()),
    )


def _review_from_stdin(item: ReviewItem) -> Answer:
    """El teclado como revisor. Las opciones salen de `item.allowed`, que fija
    `review.py`: aquí solo se pregunta, no se decide qué se admite."""
    from aida.arms.health import review as _review

    keys = {
        _review.Verdict.CONFIRMED: "c",
        _review.Verdict.EDITED: "e",
        _review.Verdict.REJECTED: "r",
    }
    names = {
        _review.Verdict.CONFIRMED: "confirmar",
        _review.Verdict.EDITED: "editar",
        _review.Verdict.REJECTED: "rechazar",
    }
    options = {keys[v]: v for v in item.allowed}
    prompt = " / ".join(f"[{keys[v]}] {names[v]}" for v in item.allowed) + ": "
    print(f"[{item.label}] {item.text}")
    while True:
        choice = input(prompt).strip().lower()
        if choice in options:
            break
        print("(respuesta no válida)")
    verdict = options[choice]
    if verdict is not _review.Verdict.EDITED:
        return _review.Answer(verdict=verdict)
    while True:
        text = input("texto corregido: ").strip()
        if text:
            return _review.Answer(verdict=verdict, text=text)


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
                    from aida.arms.health.capture import (
                        ModelUnavailableError,
                        TranscriptionError,
                    )

                    try:
                        result = consultation.run_with_history(
                            owner=AgentId(args.owner),
                            audio_path=write_wav(consulted_audio, tmp / "consultado.wav"),
                            history_text=history_text,
                        )
                    except (TranscriptionError, ModelUnavailableError) as exc:
                        print(f"aida: error: {exc}", file=sys.stderr)
                        print("(de vuelta en conversación)")
                        continue
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

    if args.review_stats is not None:
        from aida.arms.health.metrics import ReviewMetricsStore, format_stats

        path = Path(args.review_stats) if args.review_stats else _review_metrics_path()
        try:
            records = ReviewMetricsStore(path).read()
        except ValueError as exc:
            print(f"aida: error: {exc}", file=sys.stderr)
            return 1
        print(format_stats(records))
        return 0

    if args.check_capture:
        from aida.arms.health.capture import capture_diagnosis

        lineas = capture_diagnosis()
        for texto, correcto in lineas:
            print(f"{'ok   ' if correcto else 'FALTA'} {texto}")
        return 0 if all(correcto for _texto, correcto in lineas) else 1

    if args.consultation:
        if not args.audio or not args.history:
            print("aida: error: --consultation requiere --audio y --history", file=sys.stderr)
            return 1

        if args.extract and not args.segment:
            print("aida: error: --extract requiere --segment", file=sys.stderr)
            return 1
        if args.review and not args.segment:
            print("aida: error: --review requiere --segment", file=sys.stderr)
            return 1
        if args.diarize:
            if not os.environ.get("AIDA_HF_TOKEN"):
                print("aida: error: --diarize requiere AIDA_HF_TOKEN", file=sys.stderr)
                return 1
            from aida.voice.diarize import build_local_diarizer

            diarizer = build_local_diarizer()
        else:
            diarizer = None

        if args.segment:
            from aida.arms.health.segment import build_health_segmenter

            segmenter = build_health_segmenter()
        else:
            segmenter = None

        if args.extract:
            from aida.arms.health.extract import build_health_extractor

            extractor = build_health_extractor()
        else:
            extractor = None

        key = _check_consult_cache_key()
        if not key:
            return 1

        from aida.arms.health.capture import build_local_capture
        from aida.engine.ontology import AgentId

        stt, ocr = build_local_capture()
        consultation = _build_consultation(
            args,
            stt=stt,
            ocr=ocr,
            diarizer=diarizer,
            segmenter=segmenter,
            extractor=extractor,
            key=key,
        )
        if consultation is None:
            return 1

        from aida.arms.health.capture import ModelUnavailableError, TranscriptionError

        try:
            if args.review:
                prepared = consultation.prepare(
                    owner=AgentId(args.owner),
                    audio_path=Path(args.audio),
                    history_text=lambda: ocr.read(Path(args.history)),
                )
                try:
                    result = consultation.review(prepared, reviewer=_review_from_stdin)
                except (EOFError, KeyboardInterrupt):
                    print(
                        "aida: revisión incompleta: nada se registra como revisado",
                        file=sys.stderr,
                    )
                    return 1
            else:
                result = consultation.run(
                    owner=AgentId(args.owner),
                    audio_path=Path(args.audio),
                    history_path=Path(args.history),
                )
        except (TranscriptionError, ModelUnavailableError) as exc:
            print(f"aida: error: {exc}", file=sys.stderr)
            return 1
        print(result.summary_markdown)
        if args.export is not None:
            from datetime import datetime

            from aida.arms.health.export import default_export_dir, export_consultation

            destino = Path(args.export) if args.export else default_export_dir()
            escritos = export_consultation(
                result,
                destination=destino,
                moment=datetime.now(),
                include_model=bool(args.print_model),
            )
            print(f"aida: consulta escrita en {escritos.summary}", file=sys.stderr)
            if escritos.model is not None:
                print(f"aida: modelo escrito en {escritos.model}", file=sys.stderr)
            print(
                "aida: aviso: queda en claro, sin cifrar; dónde vive es cosa tuya",
                file=sys.stderr,
            )
        if args.print_model:
            if result.encounter is None:
                print("aida: aviso: --print-model requiere --segment", file=sys.stderr)
            else:
                import json

                print(json.dumps(result.encounter.to_dict(), ensure_ascii=False, indent=2))
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
