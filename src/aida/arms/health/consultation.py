"""Un caso de consulta: transcribe, lee la historia, coteja códigos,
redacta el resumen, lo cachea. Sin `Loop`, sin `decide()` — no hay
deliberación, es una tubería fija que se ejecuta cuando el médico la pide.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT
from aida.arms.health.extract import ClinicalExtractor
from aida.arms.health.history import HistoryLayout, codable_history_lines
from aida.arms.health.lexicon import Lexicon, clinical_terms
from aida.arms.health.match import (
    CatalogIndex,
    CodableText,
    CodeOrigin,
    MatchedCode,
    match_by_origin,
)
from aida.arms.health.metrics import ReviewMetricsStore, review_record
from aida.arms.health.model import ClinicalFinding, Encounter, build_encounter
from aida.arms.health.negation import (
    NO_NEGATION,
    NegationCatalog,
    RetractionRule,
    ThirdPartyCatalog,
    is_third_party,
    retracted_replies,
    split_polarity,
)
from aida.arms.health.note import NoteSettings, RenderedNote, build_note
from aida.arms.health.review import Reviewer, review_encounter, review_note
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
    evidence_template: str | None = None
    negated_suffix: str | None = None


@dataclass(frozen=True, slots=True)
class ConsultationResult:
    owner: AgentId
    transcript: str
    history_excerpt: str
    matched_codes: tuple[str, ...]
    summary_markdown: str
    encounter: Encounter | None = None
    note: RenderedNote | None = None
    """La nota redactada por campos, cuando se pidió (`--note`). `None` cuando
    no se pidió: construirla cuesta una llamada al proveedor de lenguaje, y no
    tiene sentido pagarla cuando nadie va a leerla."""


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
        negation: NegationCatalog = NO_NEGATION,
        third_party: ThirdPartyCatalog | None = None,
        retraction: RetractionRule | None = None,
        history_layout: HistoryLayout | None = None,
        review_metrics: ReviewMetricsStore | None = None,
        note_settings: NoteSettings | None = None,
        note_providers: tuple[tuple[str, CognitionProvider], ...] = (),
        note_on_fallback: Callable[[str, str], None] | None = None,
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
        self._negation = negation
        self._third_party = third_party
        self._retraction = retraction
        self._history_layout = history_layout
        self._review_metrics = review_metrics
        self._note_settings = note_settings
        self._note_providers = note_providers
        self._note_on_fallback = note_on_fallback
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
        return self.finish(prepared)

    def finish(self, prepared: PreparedConsultation) -> ConsultationResult:
        """El borrador a partir de lo ya preparado: separado de
        `run_with_history` para que quien necesite la nota (`render_note`)
        pueda partir del mismo `PreparedConsultation` sin volver a
        transcribir ni a segmentar."""
        facts = self._build_facts(
            prepared.transcript, prepared.history, prepared.matches, prepared.grouped_sections
        )
        summary = self._redact(facts)

        result = ConsultationResult(
            owner=prepared.owner,
            transcript=prepared.transcript,
            history_excerpt=prepared.history,
            matched_codes=tuple(m.diagnostic.code for m in prepared.matches),
            summary_markdown=summary,
            encounter=prepared.encounter,
        )
        self._cache.store(owner=prepared.owner, summary_markdown=summary)
        return result

    def render_note(self, prepared: PreparedConsultation) -> RenderedNote:
        """La nota clínica a partir de lo ya preparado. No depende de la
        revisión ni de qué códigos se confirmen: por eso puede llamarse antes,
        después o en vez de `review`/`finish`, sin volver a segmentar."""
        if prepared.grouped_sections is None:
            raise ValueError("--note requiere --segment")
        if self._note_settings is None:
            raise ValueError("no hay ajustes de nota configurados (note.yaml)")
        return build_note(
            grouped_sections=prepared.grouped_sections,
            conversation=tuple(prepared.transcript.split("\n")),
            settings=self._note_settings,
            providers=self._note_providers,
            on_fallback=self._note_on_fallback,
        )

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

        retractadas: frozenset[str] = frozenset()
        if self._retraction is not None:
            retractadas = retracted_replies(
                tuple(t.strip() for t in transcript.splitlines() if t.strip()),
                self._negation,
                self._retraction,
            )

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
                            ("negated", finding.negated),
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

        de_hoy, de_historia = self._codable_texts(
            transcript, history, grouped_sections, retractadas
        )
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
        retracted: frozenset[str] = frozenset(),
    ) -> tuple[tuple[CodableText, ...], tuple[CodableText, ...]]:
        """Qué se cruza contra el catálogo, como `(hoy, historia)`: los hallazgos
        de las secciones que el pack marca como portadoras de diagnóstico, y por
        otro lado la historia.

        Van separados y no juntos porque el profesional necesita saber si un
        diagnóstico es de hoy o ya constaba.

        Una frase de medicación o de plan no es un diagnóstico, y cotejarla
        produce códigos sin relación — de ahí salía la mitad del ruido de
        `I-125`. Sin segmentación no hay secciones que mirar, así que se cruza
        la transcripción entera, línea a línea.
        """
        lineas = (
            codable_history_lines(history, self._history_layout)
            if self._history_layout is not None
            else tuple(linea for linea in history.splitlines() if linea.strip())
        )
        if grouped_sections is None:
            dichos = tuple(t for t in transcript.splitlines() if t.strip())
        else:
            dichos = tuple(
                texto for spec, textos in grouped_sections if spec.codable for texto in textos
            )
        dichos = tuple(t for t in dichos if t.strip() not in retracted)
        return (self._affirmed(dichos), self._affirmed(lineas))

    def _affirmed(self, texts: tuple[str, ...]) -> tuple[CodableText, ...]:
        """Lo negado no se coteja: «no refiere dolor torácico» comparte todas
        sus palabras útiles con la frase afirmada, así que propondría el mismo
        código. Cada texto conserva la frase completa como procedencia, también
        cuando solo se coteja un tramo de ella: es lo que se le muestra al
        profesional para que juzgue el código.

        Después, el término clínico de lo que se dijo con palabras del
        paciente, como consulta aparte y sin repetir. Aparte y no dentro de la
        frase: medido, añadirlo a la frase la alarga y hunde la puntuación por
        debajo del umbral; consultarlo solo acierta. Y se aplica también a la
        historia, aunque venga en lenguaje clínico: la historia dice
        «dislipidemia» y el catálogo «hiperlipidemia», así que sin traducir los
        dos lados el mismo hecho salía como diagnóstico nuevo.

        El término sale del tramo afirmado, no de la frase entera: traducir «no
        sufro del azúcar» a «diabetes mellitus» volvería a colar por el léxico
        justo lo que la polaridad acaba de excluir.

        Una frase de tercero se descarta aquí, **antes** del léxico: «mi padre
        murió de un infarto» traduciría «infarto» a su término clínico y lo
        consultaría como hecho propio, proponiendo un código del paciente sobre
        algo que le pasó a otra persona. Filtrar después dejaría pasar
        exactamente ese caso, porque el término traducido ya estaría en la
        lista de consultas.
        """
        afirmados: list[CodableText] = []
        for texto in texts:
            if self._third_party is not None and is_third_party(texto, self._third_party):
                continue
            tramo = split_polarity(texto, self._negation).affirmed
            if tramo:
                afirmados.append(CodableText(query=tramo, source=texto))
        vistos: list[str] = []
        equivalentes: list[CodableText] = []
        for codable in afirmados:
            for termino in clinical_terms(codable.query, self._lexicon):
                if termino not in vistos:
                    vistos.append(termino)
                    equivalentes.append(CodableText(query=termino, source=codable.source))
        return (*afirmados, *equivalentes)

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

    def review(
        self,
        prepared: PreparedConsultation,
        *,
        reviewer: Reviewer,
        elapsed_seconds: float | None = None,
        review_findings: bool = True,
        note: RenderedNote | None = None,
        note_edits: Mapping[str, str] | None = None,
    ) -> ConsultationResult:
        """Solo tras contestar a todo se guarda. El resumen —la lista de
        hechos y códigos confirmados— sale sin modelo: eso es lo que el
        profesional aprueba, campo por hallazgo y código por código. La nota
        en prosa (`render_note`) es una redacción de esos mismos hechos, no
        otra cosa que aprobar: copiarla a la historia clínica es el acto de
        aprobación final de esa redacción. Si `reviewer` levanta, se propaga y
        no se guarda nada: callar no es confirmar.

        Con `note` y `note_edits` —la pantalla—, la nota ya redactada se revisa
        aquí y la que vuelve es la entregada, con sus campos editados marcados.
        """
        if prepared.encounter is None or prepared.grouped_sections is None:
            raise ValueError("revisión activa pero no hay segmentador configurado")
        if (note is None) != (note_edits is None):
            raise ValueError("la nota y sus ediciones van juntas, o ninguna de las dos")
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
                evidence_template=self._wording.evidence_template,
                negated_suffix=self._wording.negated_suffix,
                review_findings=review_findings,
            )
            reviewed_note = (
                review_note(owner=prepared.owner, note=note, edits=note_edits, log=self._log)
                if note is not None and note_edits is not None
                else None
            )
        except BaseException:
            self._record_review(
                start_seq, started, completed=False, elapsed_seconds=elapsed_seconds
            )
            raise
        self._record_review(start_seq, started, completed=True, elapsed_seconds=elapsed_seconds)

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
            note=reviewed_note,
        )
        self._cache.store(owner=prepared.owner, summary_markdown=summary)
        return result

    def _record_review(
        self,
        start_seq: int,
        started: float,
        *,
        completed: bool,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Solo los eventos de esta revisión: el registro puede traer los de
        consultas anteriores del mismo proceso. Arriba se captura
        `BaseException`, y no `Exception`, porque Ctrl-C también es un corte que
        hay que contar.

        `elapsed_seconds` llega cuando quien revisó lo hizo **fuera de este
        proceso** —una pantalla aparte—: entonces el tramo que rodea a
        `review_encounter` dura milisegundos y medirlo sería registrar la
        velocidad del bucle en vez del tiempo de la persona, que es justo la
        métrica que el producto promete reducir."""
        if self._review_metrics is None:
            return
        events = [event for event in self._log if event.seq >= start_seq]
        segundos = (
            elapsed_seconds if elapsed_seconds is not None else self._clock() - started
        )
        self._review_metrics.append(
            review_record(events, completed=completed, seconds=segundos)
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

