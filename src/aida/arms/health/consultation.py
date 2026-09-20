"""Un caso de consulta: transcribe, lee la historia, coteja códigos,
redacta el resumen, lo cachea. Sin `Loop`, sin `decide()` — no hay
deliberación, es una tubería fija que se ejecuta cuando el médico la pide.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT
from aida.arms.health.match import CatalogIndex, MatchedCode, match_codes
from aida.arms.health.sections import SectionSpec
from aida.arms.health.segment import ClinicalSegmenter
from aida.cognition.briefing import SpokenReply
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.engine.ports import CognitionProvider
from aida.voice.diarize import DiarizationError, LocalPyannoteDiarizer, merge_by_speaker


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
    professional_label: str | None = None
    patient_label: str | None = None
    section_template: str | None = None


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
        diarizer: LocalPyannoteDiarizer | None = None,
        segmenter: ClinicalSegmenter | None = None,
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
        self._diarizer = diarizer
        self._segmenter = segmenter

    def run(self, *, owner: AgentId, audio_path: Path, history_path: Path) -> ConsultationResult:
        """Lee la historia y delega. Es el camino por archivos (plan 31)."""
        return self.run_with_history(
            owner=owner, audio_path=audio_path, history_text=lambda: self._ocr.read(history_path)
        )

    def run_with_history(
        self, *, owner: AgentId, audio_path: Path, history_text: str | Callable[[], str]
    ) -> ConsultationResult:
        """La historia ya leída. Es el camino cuando la HC se cargó al
        arrancar — hoy a mano; mañana la traerá quien sepa qué paciente
        sigue (Ruta B)."""
        if self._diarizer is not None:
            transcript = self._diarize_and_label(owner, audio_path)
        elif self._segmenter is not None:
            segments = self._stt.transcribe_segments(audio_path)
            transcript = "\n".join(s.text for s in segments)
        else:
            transcript = self._stt.transcribe(audio_path)
        self._log.emit(tick=0, kind=EventKind.INPUT_OBSERVED, owner=owner, content=transcript)

        grouped_sections = None
        if self._segmenter is not None:
            grouped_sections = self._segmenter.segment(transcript)
            self._log.emit(
                tick=0,
                kind=EventKind.TEXT_SEGMENTED,
                owner=owner,
                data={"sections": ",".join(spec.section_id for spec, _ in grouped_sections)}
                if grouped_sections
                else {},
            )

        history = history_text() if callable(history_text) else history_text
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

        facts = self._build_facts(transcript, history, matches, grouped_sections)
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

    def _diarize_and_label(self, owner: AgentId, audio_path: Path) -> str:
        if self._wording.professional_label is None or self._wording.patient_label is None:
            raise ValueError(
                "diarización activa pero wording no trae professional_label/patient_label"
            )
        if self._diarizer is None:
            raise RuntimeError("diarizador no configurado")
        segments = self._stt.transcribe_segments(audio_path)
        turns = self._diarizer.diarize(audio_path)
        pairs = merge_by_speaker(segments, turns)

        distinct: list[str] = []
        for speaker, text in pairs:
            if speaker == "DESCONOCIDO" and not text.strip():
                continue
            if speaker not in distinct:
                distinct.append(speaker)
        if len(distinct) > 2:
            raise DiarizationError(
                f"se detectaron {len(distinct)} hablantes distintos, se esperaban 2"
            )
        professional_id = distinct[0] if distinct else None

        self._log.emit(
            tick=0,
            kind=EventKind.SPEAKERS_DIARIZED,
            owner=owner,
            data={"speakers": str(len(distinct))},
        )

        prof = self._wording.professional_label
        pat = self._wording.patient_label
        lines = [
            f"{prof if speaker == professional_id else pat}: {text}"
            for speaker, text in pairs
        ]
        return "\n".join(lines)

    def _build_facts(
        self,
        transcript: str,
        history: str,
        matches: tuple[MatchedCode, ...],
        grouped_sections: tuple[tuple[SectionSpec, tuple[str, ...]], ...] | None = None,
    ) -> list[str]:
        facts: list[str] = []
        if grouped_sections is not None:
            if self._wording.section_template is None:
                raise ValueError(
                    "segmentación activa pero wording no trae section_template"
                )
            for spec, texts in grouped_sections:
                facts.append(self._wording.section_template.format(spec.label, "\n".join(texts)))
        else:
            facts.append(self._wording.said_template.format(transcript))
        facts.append(self._wording.history_template.format(history))
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

