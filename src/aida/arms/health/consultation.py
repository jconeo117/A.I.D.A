"""Un caso de consulta: transcribe, lee la historia, coteja códigos,
redacta el resumen, lo cachea. Sin `Loop`, sin `decide()` — no hay
deliberación, es una tubería fija que se ejecuta cuando el médico la pide.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT
from aida.arms.health.extract import ClinicalExtractor
from aida.arms.health.lexicon import Lexicon, clinical_terms
from aida.arms.health.match import CatalogIndex, CodeOrigin, MatchedCode, match_by_origin
from aida.arms.health.metrics import ReviewMetricsStore, review_record
from aida.arms.health.model import ClinicalFinding, Encounter, build_encounter
from aida.arms.health.review import Reviewer, review_encounter
from aida.arms.health.sections import SectionSpec
from aida.arms.health.segment import ClinicalSegmenter
from aida.cognition.briefing import SpokenReply
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.engine.ports import CognitionProvider
from aida.voice.diarize import DiarizationError, LocalPyannoteDiarizer, merge_by_speaker


def _listado(codes: list[MatchedCode]) -> str:
    """Un código por su identificador y su descripción, separados por punto y
    coma. Una sola forma para las dos líneas del resumen: la de hoy y la de lo
    que ya constaba."""
    return "; ".join(f"{m.diagnostic.code} ({m.diagnostic.description})" for m in codes)


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
    confirmed_template: str | None = None
    no_confirmed_text: str | None = None
    history_codes_template: str | None = None
    known_code_suffix: str | None = None


@dataclass(frozen=True, slots=True)
class ConsultationResult:
    owner: AgentId
    transcript: str
    history_excerpt: str
    matched_codes: tuple[str, ...]
    summary_markdown: str
    encounter: Encounter | None = None


@dataclass(frozen=True, slots=True)
class PreparedConsultation:
    """Todo lo que la consulta sabe antes de redactar nada. Existe para que la
    revisión parta de lo mismo que el borrador sin que el borrador llegue a
    existir: con revisión, ni se llama al modelo ni se escribe en caché hasta
    que el profesional ha contestado a todo."""

    owner: AgentId
    transcript: str
    history: str
    matches: tuple[MatchedCode, ...]
    grouped_sections: tuple[tuple[SectionSpec, tuple[str, ...]], ...] | None
    encounter: Encounter | None


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
        extractor: ClinicalExtractor | None = None,
        lexicon: Lexicon = (),
        review_metrics: ReviewMetricsStore | None = None,
        clock: Callable[[], float] = time.monotonic,
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
        self._extractor = extractor
        self._lexicon = lexicon
        self._review_metrics = review_metrics
        self._clock = clock

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
        sigue (Ruta B). Es `prepare` más el borrador: los eventos salen
        igual que antes de existir `prepare`."""
        prepared = self.prepare(owner=owner, audio_path=audio_path, history_text=history_text)
        facts = self._build_facts(
            prepared.transcript, prepared.history, prepared.matches, prepared.grouped_sections
        )
        summary = self._redact(facts)

        result = ConsultationResult(
            owner=owner,
            transcript=prepared.transcript,
            history_excerpt=prepared.history,
            matched_codes=tuple(m.diagnostic.code for m in prepared.matches),
            summary_markdown=summary,
            encounter=prepared.encounter,
        )
        self._cache.store(owner=owner, summary_markdown=summary)
        return result

    def prepare(
        self, *, owner: AgentId, audio_path: Path, history_text: str | Callable[[], str]
    ) -> PreparedConsultation:
        """Transcribe, segmenta, extrae, lee la historia y coteja — y se para
        ahí: ni modelo ni caché."""
        if self._extractor is not None and self._segmenter is None:
            raise ValueError("extracción activa pero no hay segmentador configurado")

        if self._diarizer is not None:
            transcript = self._diarize_and_label(owner, audio_path)
        elif self._segmenter is not None:
            segments = self._stt.transcribe_segments(audio_path)
            transcript = "\n".join(s.text for s in segments)
        else:
            transcript = self._stt.transcribe(audio_path)
        self._log.emit(tick=0, kind=EventKind.INPUT_OBSERVED, owner=owner, content=transcript)

        grouped_sections = None
        encounter = None
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
            if grouped_sections is not None:
                encounter = build_encounter(owner=owner, grouped_sections=grouped_sections)
                self._log.emit(
                    tick=0,
                    kind=EventKind.MODEL_BUILT,
                    owner=owner,
                    data={"findings": str(sum(len(v) for v in encounter.findings.values()))},
                )
                if self._extractor is not None:
                    enriched = {
                        section_id: tuple(
                            self._extractor.extract(
                                finding,
                                professional_label=self._wording.professional_label,
                                patient_label=self._wording.patient_label,
                            )
                            for finding in items
                        )
                        for section_id, items in encounter.findings.items()
                    }
                    encounter = Encounter(owner=encounter.owner, findings=enriched)
                    field_names = {
                        name
                        for items in enriched.values()
                        for finding in items
                        for name, value in (
                            ("duration", finding.duration),
                            ("laterality", finding.laterality),
                            ("reported_by", finding.reported_by),
                        )
                        if value is not None
                    }
                    self._log.emit(
                        tick=0,
                        kind=EventKind.FIELDS_EXTRACTED,
                        owner=owner,
                        data={"fields": ",".join(sorted(field_names))} if field_names else {},
                    )

        history = history_text() if callable(history_text) else history_text
        self._log.emit(tick=0, kind=EventKind.INPUT_OBSERVED, owner=owner, content=history)

        de_hoy, de_historia = self._codable_texts(transcript, history, grouped_sections)
        matches = match_by_origin(
            today=de_hoy,
            history=de_historia,
            index=self._index,
            threshold=self._threshold,
            top_n=self._top_n,
        )
        self._log.emit(
            tick=0,
            kind=EventKind.CATALOG_MATCHED,
            owner=owner,
            data={"codes": ",".join(m.diagnostic.code for m in matches)} if matches else {},
        )

        return PreparedConsultation(
            owner=owner,
            transcript=transcript,
            history=history,
            matches=matches,
            grouped_sections=grouped_sections,
            encounter=encounter,
        )

    def _codable_texts(
        self,
        transcript: str,
        history: str,
        grouped_sections: tuple[tuple[SectionSpec, tuple[str, ...]], ...] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Qué se cruza contra el catálogo, como `(hoy, historia)`: los hallazgos
        de las secciones que el pack marca como portadoras de diagnóstico, y por
        otro lado la historia.

        Van separados y no juntos porque el profesional necesita saber si un
        diagnóstico es de hoy o ya constaba.

        Una frase de medicación o de plan no es un diagnóstico, y cotejarla
        produce códigos sin relación — de ahí salía la mitad del ruido de
        `I-125`. Sin segmentación no hay secciones que mirar, así que se cruza
        la transcripción entera, línea a línea."""
        lineas = tuple(linea for linea in history.splitlines() if linea.strip())
        if grouped_sections is None:
            dichos = tuple(t for t in transcript.splitlines() if t.strip())
        else:
            dichos = tuple(
                texto for spec, textos in grouped_sections if spec.codable for texto in textos
            )
        return (
            (*dichos, *self._clinical_equivalents(dichos)),
            (*lineas, *self._clinical_equivalents(lineas)),
        )

    def _clinical_equivalents(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        """El término clínico de lo que el paciente dijo con sus palabras, como
        consulta aparte y sin repetir.

        Aparte y no dentro de la frase: medido, añadirlo a la frase la alarga y
        hunde la puntuación por debajo del umbral; consultarlo solo acierta.

        Se aplica **también a la historia**, aunque venga en lenguaje clínico:
        medido, la historia dice «dislipidemia» y el catálogo «hiperlipidemia»,
        así que sin traducir los dos lados el mismo hecho salía como diagnóstico
        nuevo en vez de como algo que ya constaba."""
        equivalentes: list[str] = []
        for texto in texts:
            for termino in clinical_terms(texto, self._lexicon):
                if termino not in equivalentes:
                    equivalentes.append(termino)
        return tuple(equivalentes)

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

    def review(self, prepared: PreparedConsultation, *, reviewer: Reviewer) -> ConsultationResult:
        """Solo tras contestar a todo se redacta y se guarda. El resumen sale de
        lo revisado sin modelo: si el modelo lo reescribiera, el texto final
        ya no sería lo que el profesional aprobó. Si `reviewer` levanta, se
        propaga y no se guarda nada: callar no es confirmar."""
        if prepared.encounter is None or prepared.grouped_sections is None:
            raise ValueError("revisión activa pero no hay segmentador configurado")
        section_template = self._wording.section_template
        confirmed_template = self._wording.confirmed_template
        no_confirmed_text = self._wording.no_confirmed_text
        history_codes_template = self._wording.history_codes_template
        if section_template is None or confirmed_template is None or no_confirmed_text is None:
            raise ValueError(
                "revisión activa pero wording no trae section_template, "
                "confirmed_template o no_confirmed_text"
            )

        start_seq = self._log.emitted
        started = self._clock()
        try:
            reviewed = review_encounter(
                owner=prepared.owner,
                grouped_sections=prepared.grouped_sections,
                encounter=prepared.encounter,
                matches=prepared.matches,
                reviewer=reviewer,
                log=self._log,
                refine=self._refine_edited,
                known_code_suffix=self._wording.known_code_suffix,
            )
        except BaseException:
            self._record_review(start_seq, started, completed=False)
            raise
        self._record_review(start_seq, started, completed=True)

        facts = [
            section_template.format(spec.label, "\n".join(texts))
            for spec, texts in reviewed.grouped_sections
        ]
        facts.append(self._wording.history_template.format(prepared.history))
        de_hoy = [m for m in reviewed.confirmed_codes if m.origin is not CodeOrigin.HISTORY]
        ya_estaban = [m for m in reviewed.confirmed_codes if m.origin is not CodeOrigin.TODAY]
        if de_hoy:
            facts.append(confirmed_template.format(_listado(de_hoy)))
        if ya_estaban and history_codes_template is not None:
            facts.append(history_codes_template.format(_listado(ya_estaban)))
        if not reviewed.confirmed_codes:
            facts.append(no_confirmed_text)
        summary = "\n".join(f"- {f}" for f in facts)

        result = ConsultationResult(
            owner=prepared.owner,
            transcript=prepared.transcript,
            history_excerpt=prepared.history,
            matched_codes=tuple(m.diagnostic.code for m in reviewed.confirmed_codes),
            summary_markdown=summary,
            encounter=reviewed.encounter,
        )
        self._cache.store(owner=prepared.owner, summary_markdown=summary)
        return result

    def _record_review(self, start_seq: int, started: float, *, completed: bool) -> None:
        """Solo los eventos de esta revisión: el registro puede traer los de
        consultas anteriores del mismo proceso. Arriba se captura
        `BaseException`, y no `Exception`, porque Ctrl-C también es un corte
        que hay que contar."""
        if self._review_metrics is None:
            return
        events = [event for event in self._log if event.seq >= start_seq]
        self._review_metrics.append(
            review_record(events, completed=completed, seconds=self._clock() - started)
        )

    def _refine_edited(self, finding: ClinicalFinding, text: str) -> ClinicalFinding:
        """El texto lo cambia el profesional; quién lo dijo, no — eso es un
        hecho del audio, no de la redacción. Duración y lateralidad sí se
        recalculan, con el mismo extractor que el resto: una sola puerta."""
        edited = replace(finding, text=text)
        if self._extractor is None:
            return edited
        refined = self._extractor.extract(edited, professional_label=None, patient_label=None)
        return replace(refined, reported_by=finding.reported_by)

