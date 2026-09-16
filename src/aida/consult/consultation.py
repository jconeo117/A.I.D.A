"""Un caso de consulta: transcribe, lee la historia, coteja códigos,
redacta el resumen, lo cachea. Sin `Loop`, sin `decide()` — no hay
deliberación, es una tubería fija que se ejecuta cuando el médico la pide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aida.cognition.briefing import SpokenReply
from aida.consult.cache import ConsultationCache
from aida.consult.capture import LocalTesseractOCR, LocalWhisperSTT
from aida.consult.match import CatalogIndex, MatchedCode, match_codes
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.engine.ports import CognitionProvider


@dataclass(frozen=True, slots=True)
class SummaryWording:
    """Los textos que nombran el dominio. Se inyectan, no se codifican: el
    módulo hace la tubería, y quién es el hablante y cómo se le llama a cada
    cosa lo decide quien despliega."""

    system_prompt: str
    said_template: str
    history_template: str
    matched_template: str
    no_match_text: str


@dataclass(frozen=True, slots=True)
class ConsultationResult:
    owner: AgentId
    transcript: str
    history_excerpt: str
    matched_codes: tuple[str, ...]
    summary_markdown: str


class Consultation:
    def __init__(
        self,
        *,
        wording: SummaryWording,
        stt: LocalWhisperSTT,
        ocr: LocalTesseractOCR,
        catalog_index: CatalogIndex,
        provider: CognitionProvider | None,
        cache: ConsultationCache,
        log: EventLog,
        match_threshold: float = 0.30,
        match_top_n: int = 5,
    ) -> None:
        self._wording = wording
        self._stt = stt
        self._ocr = ocr
        self._index = catalog_index
        self._provider = provider
        self._cache = cache
        self._log = log
        self._threshold = match_threshold
        self._top_n = match_top_n

    def run(self, *, owner: AgentId, audio_path: Path, history_path: Path) -> ConsultationResult:
        transcript = self._stt.transcribe(audio_path)
        self._log.emit(tick=0, kind=EventKind.INPUT_OBSERVED, owner=owner, content=transcript)

        history = self._ocr.read(history_path)
        self._log.emit(tick=0, kind=EventKind.INPUT_OBSERVED, owner=owner, content=history)

        matches = match_codes(
            f"{transcript}\n{history}", self._index, threshold=self._threshold, top_n=self._top_n
        )
        self._log.emit(
            tick=0,
            kind=EventKind.CATALOG_MATCHED,
            owner=owner,
            data={"codes": ",".join(m.diagnostic.code for m in matches)} if matches else {},
        )

        facts = self._build_facts(transcript, history, matches)
        summary = self._redact(facts)

        result = ConsultationResult(
            owner=owner,
            transcript=transcript,
            history_excerpt=history,
            matched_codes=tuple(m.diagnostic.code for m in matches),
            summary_markdown=summary,
        )
        self._cache.store(owner=owner, summary_markdown=summary)
        return result

    def _build_facts(
        self, transcript: str, history: str, matches: tuple[MatchedCode, ...]
    ) -> list[str]:
        facts = [
            self._wording.said_template.format(transcript),
            self._wording.history_template.format(history),
        ]
        if matches:
            listado = "; ".join(
                f"{m.diagnostic.code} ({m.diagnostic.description})" for m in matches
            )
            facts.append(self._wording.matched_template.format(listado))
        else:
            facts.append(self._wording.no_match_text)
        return facts

    def _redact(self, facts: list[str]) -> str:
        briefing = self._wording.system_prompt + "\n\n" + "\n".join(f"- {f}" for f in facts)
        if self._provider is not None:
            try:
                result: object | None = self._provider.think(briefing, SpokenReply)
            except Exception:
                # Un proveedor que levanta no puede tumbar la consulta: se sigue
                # con el markdown determinista.
                result = None
            # La validación va FUERA del `try`: si se relaja, tiene que romper
            # de forma visible, no quedar tapada por la captura de arriba.
            if isinstance(result, SpokenReply):
                return result.text
        return "\n".join(f"- {f}" for f in facts)

