"""Pruebas unitarias de la revisión del profesional (Fase 6 de AIDA Health).

Cubre §B.1 de `.claude/workflow/51-plan-revision-profesional.md`:
1. Todo confirmado: encounter, grouped_sections, confirmed_codes y secuencia de ITEM_REVIEWED.
2. Hallazgo rechazado: desaparece y sección sin hallazgos se omite de encounter y grouped_sections.
3. Edición: evento INPUT_OBSERVED previo, llamada única a refine y nuevo texto en grouped_sections.
4. Orden: el orden de preguntas sigue grouped_sections y no encounter.findings (dict).
5. Código: ReviewItem.allowed solo contiene CONFIRMED y REJECTED; EDITED levanta ValueError.
6. Validación del veredicto: EDITED sin texto o con espacios, o CONFIRMED/REJECTED con texto
   levantan ValueError.
7. Revisión cortada: revisor levanta EOFError en tercer ítem; la excepción se propaga y
   quedan 2 eventos.
8. Código rechazado: queda excluido de confirmed_codes y confirmado incluido.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.match import CodeOrigin, MatchedCode
from aida.arms.health.model import ClinicalFinding, Encounter
from aida.arms.health.review import (
    CODE_VERDICTS,
    FINDING_VERDICTS,
    Answer,
    ItemKind,
    ReviewItem,
    Verdict,
    code_item,
    finding_item,
    review_encounter,
    review_items,
)
from aida.arms.health.sections import SectionSpec
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId


def _crear_fixture_base() -> tuple[
    AgentId,
    SectionSpec,
    SectionSpec,
    ClinicalFinding,
    ClinicalFinding,
    Encounter,
    tuple[tuple[SectionSpec, tuple[str, ...]], ...],
    MatchedCode,
    MatchedCode,
    tuple[MatchedCode, ...],
]:
    owner = AgentId("dra_martinez")
    spec1 = SectionSpec(
        section_id="motivo", label="Motivo de consulta", trigger_phrases=(), codable=True
    )
    spec2 = SectionSpec(
        section_id="examen", label="Examen físico", trigger_phrases=(), codable=True
    )
    f1 = ClinicalFinding(
        text="Cefalea frontal pulsátil",
        section_id="motivo",
        duration="3 días",
        laterality=None,
        reported_by="PACIENTE",
    )
    f2 = ClinicalFinding(
        text="Tensión arterial 120/80",
        section_id="examen",
        duration=None,
        laterality=None,
        reported_by="PROFESIONAL",
    )
    encounter = Encounter(owner=owner, findings={"motivo": (f1,), "examen": (f2,)})
    grouped_sections = (
        (spec1, ("Cefalea frontal pulsátil",)),
        (spec2, ("Tensión arterial 120/80",)),
    )
    m1 = MatchedCode(
        diagnostic=DiagnosticCode(code="G43.9", description="Migraña", level=4, source="ambas"),
        score=0.85,
    )
    m2 = MatchedCode(
        diagnostic=DiagnosticCode(code="R51", description="Cefalea", level=4, source="ambas"),
        score=0.72,
    )
    matches = (m1, m2)
    return owner, spec1, spec2, f1, f2, encounter, grouped_sections, m1, m2, matches


def test_review_todo_confirmado() -> None:
    """Caso 1: Todo confirmado preserva encounter, grouped_sections y matches
    con orden de eventos.
    """
    owner, _spec1, _spec2, _f1, _f2, encounter, grouped_sections, _m1, _m2, matches = (
        _crear_fixture_base()
    )
    log = EventLog()
    items_recibidos: list[ReviewItem] = []

    def reviewer(item: ReviewItem) -> Answer:
        items_recibidos.append(item)
        return Answer(verdict=Verdict.CONFIRMED)

    def spy_refine(finding: ClinicalFinding, text: str) -> ClinicalFinding:
        raise AssertionError("refine no debe llamarse si no hay edición")

    resultado = review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=matches,
        reviewer=reviewer,
        log=log,
        refine=spy_refine,
    )

    assert resultado.encounter.findings == encounter.findings
    assert resultado.grouped_sections == grouped_sections
    assert resultado.confirmed_codes == matches
    assert len(items_recibidos) == 4

    eventos_reviewed = [e for e in log if e.kind == EventKind.ITEM_REVIEWED]
    assert len(eventos_reviewed) == 4
    assert [e.data for e in eventos_reviewed] == [
        {"item": "finding", "ref": "motivo:0", "verdict": "confirmed"},
        {"item": "finding", "ref": "examen:0", "verdict": "confirmed"},
        {"item": "code", "ref": "G43.9", "verdict": "confirmed", "origin": "today"},
        {"item": "code", "ref": "R51", "verdict": "confirmed", "origin": "today"},
    ]
    assert not any(e.kind == EventKind.INPUT_OBSERVED for e in log)


def test_review_hallazgo_rechazado_omite_seccion_vacia() -> None:
    """Caso 2: Hallazgo rechazado desaparece; sección con todos rechazados se omite."""
    owner, spec1, spec2, f1, _f2, _enc, _gs, _m1, _m2, matches = _crear_fixture_base()
    f1_extra = ClinicalFinding(
        text="Náuseas matutinas",
        section_id="motivo",
        duration=None,
        laterality=None,
        reported_by="PACIENTE",
    )
    f2 = ClinicalFinding(
        text="Afebril",
        section_id="examen",
        duration=None,
        laterality=None,
        reported_by="PROFESIONAL",
    )

    encounter = Encounter(owner=owner, findings={"motivo": (f1, f1_extra), "examen": (f2,)})
    grouped_sections = (
        (spec1, (f1.text, f1_extra.text)),
        (spec2, (f2.text,)),
    )
    log = EventLog()

    respuestas = {
        "motivo:0": Answer(Verdict.CONFIRMED),
        "motivo:1": Answer(Verdict.REJECTED),
        "examen:0": Answer(Verdict.REJECTED),
        "G43.9": Answer(Verdict.CONFIRMED),
        "R51": Answer(Verdict.CONFIRMED),
    }

    resultado = review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=matches,
        reviewer=lambda item: respuestas[item.ref],
        log=log,
        refine=lambda finding, text: finding,
    )

    # motivo conserva f1 pero pierde f1_extra; examen queda vacía y se omite
    assert "examen" not in resultado.encounter.findings
    assert resultado.encounter.findings["motivo"] == (f1,)
    assert len(resultado.grouped_sections) == 1
    assert resultado.grouped_sections[0] == (spec1, (f1.text,))
    assert resultado.confirmed_codes == matches


def test_review_edicion_emite_input_observed_llama_refine_y_actualiza_texto() -> None:
    """Caso 3: Edición emite INPUT_OBSERVED previo, llama refine y actualiza texto."""
    owner, _spec1, _spec2, f1, _f2, encounter, grouped_sections, _m1, _m2, matches = (
        _crear_fixture_base()
    )
    log = EventLog()
    refine_calls: list[tuple[ClinicalFinding, str]] = []

    def spy_refine(finding: ClinicalFinding, text: str) -> ClinicalFinding:
        refine_calls.append((finding, text))
        return replace(finding, text=text, duration="5 días", laterality="izquierdo")

    def reviewer(item: ReviewItem) -> Answer:
        if item.ref == "motivo:0":
            return Answer(Verdict.EDITED, text="Cefalea hemicraneal izquierda intensa")
        return Answer(Verdict.CONFIRMED)

    resultado = review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=matches,
        reviewer=reviewer,
        log=log,
        refine=spy_refine,
    )

    assert len(refine_calls) == 1
    assert refine_calls[0] == (f1, "Cefalea hemicraneal izquierda intensa")

    hallazgo_revisado = resultado.encounter.findings["motivo"][0]
    assert hallazgo_revisado.text == "Cefalea hemicraneal izquierda intensa"
    assert hallazgo_revisado.duration == "5 días"
    assert hallazgo_revisado.laterality == "izquierdo"
    assert resultado.grouped_sections[0][1] == ("Cefalea hemicraneal izquierda intensa",)

    # Verificar que el INPUT_OBSERVED va exactamente antes del ITEM_REVIEWED de ese hallazgo
    eventos = list(log)
    idx_reviewed = next(
        i for i, e in enumerate(eventos)
        if e.kind == EventKind.ITEM_REVIEWED and e.data.get("ref") == "motivo:0"
    )
    evento_previo = eventos[idx_reviewed - 1]
    assert evento_previo.kind == EventKind.INPUT_OBSERVED
    assert evento_previo.content == "Cefalea hemicraneal izquierda intensa"
    assert evento_previo.owner == owner


def test_review_orden_sigue_grouped_sections_y_no_dict_findings() -> None:
    """Caso 4: El orden de preguntas sigue grouped_sections aunque findings esté invertido."""
    owner, spec1, spec2, f1, f2, _enc, _gs, _m1, _m2, matches = _crear_fixture_base()
    # Construir el dict de findings en orden INVERSO a grouped_sections
    encounter_invertido = Encounter(owner=owner, findings={"examen": (f2,), "motivo": (f1,)})
    grouped_sections = (
        (spec1, ("Cefalea frontal pulsátil",)),
        (spec2, ("Tensión arterial 120/80",)),
    )
    refs_preguntados: list[str] = []

    def reviewer(item: ReviewItem) -> Answer:
        refs_preguntados.append(item.ref)
        return Answer(Verdict.CONFIRMED)

    review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter_invertido,
        matches=matches,
        reviewer=reviewer,
        log=EventLog(),
        refine=lambda finding, text: finding,
    )

    assert refs_preguntados == ["motivo:0", "examen:0", "G43.9", "R51"]


def test_review_codigo_solo_admite_confirmar_y_rechazar() -> None:
    """Caso 5: ReviewItem de un código solo admite CONFIRMED y REJECTED;
    EDITED levanta ValueError.
    """
    owner, _spec1, _spec2, _f1, _f2, encounter, grouped_sections, _m1, _m2, matches = (
        _crear_fixture_base()
    )
    log = EventLog()

    def reviewer(item: ReviewItem) -> Answer:
        if item.kind == ItemKind.CODE:
            assert item.allowed == CODE_VERDICTS
            assert Verdict.EDITED not in item.allowed
            return Answer(Verdict.EDITED, text="Otro código no permitido")
        assert item.allowed == FINDING_VERDICTS
        return Answer(Verdict.CONFIRMED)

    with pytest.raises(ValueError, match=r"veredicto 'edited' no admitido para 'G43\.9'"):
        review_encounter(
            owner=owner,
            grouped_sections=grouped_sections,
            encounter=encounter,
            matches=matches,
            reviewer=reviewer,
            log=log,
            refine=lambda finding, text: finding,
        )

    # No se emitió ITEM_REVIEWED para el código inválido
    refs_emitidos = [e.data["ref"] for e in log if e.kind == EventKind.ITEM_REVIEWED]
    assert "G43.9" not in refs_emitidos
    assert refs_emitidos == ["motivo:0", "examen:0"]


def test_review_validacion_veredicto_invalido() -> None:
    """Caso 6: EDITED sin texto o en blanco, y CONFIRMED/REJECTED con texto dan ValueError."""
    owner, _spec1, _spec2, _f1, _f2, encounter, grouped_sections, _m1, _m2, matches = (
        _crear_fixture_base()
    )

    # 1. EDITED sin texto
    with pytest.raises(ValueError, match=r"edición sin texto para 'motivo:0'"):
        review_encounter(
            owner=owner,
            grouped_sections=grouped_sections,
            encounter=encounter,
            matches=matches,
            reviewer=lambda item: Answer(Verdict.EDITED, text=None),
            log=EventLog(),
            refine=lambda finding, text: finding,
        )

    # 2. EDITED con texto solo de espacios
    with pytest.raises(ValueError, match=r"edición sin texto para 'motivo:0'"):
        review_encounter(
            owner=owner,
            grouped_sections=grouped_sections,
            encounter=encounter,
            matches=matches,
            reviewer=lambda item: Answer(Verdict.EDITED, text="   "),
            log=EventLog(),
            refine=lambda finding, text: finding,
        )

    # 3. CONFIRMED con texto
    with pytest.raises(ValueError, match=r"texto sin edición para 'motivo:0'"):
        review_encounter(
            owner=owner,
            grouped_sections=grouped_sections,
            encounter=encounter,
            matches=matches,
            reviewer=lambda item: Answer(Verdict.CONFIRMED, text="texto innecesario"),
            log=EventLog(),
            refine=lambda finding, text: finding,
        )

    # 4. REJECTED con texto
    with pytest.raises(ValueError, match=r"texto sin edición para 'motivo:0'"):
        review_encounter(
            owner=owner,
            grouped_sections=grouped_sections,
            encounter=encounter,
            matches=matches,
            reviewer=lambda item: Answer(Verdict.REJECTED, text="texto innecesario"),
            log=EventLog(),
            refine=lambda finding, text: finding,
        )


def test_review_cortada_propaga_excepcion_y_conserva_eventos_previos() -> None:
    """Caso 7: Si el revisor levanta EOFError en el 3er ítem, se propaga y
    quedan 2 ITEM_REVIEWED.
    """
    owner, _spec1, _spec2, _f1, _f2, encounter, grouped_sections, _m1, _m2, matches = (
        _crear_fixture_base()
    )
    log = EventLog()
    llamadas = 0

    def reviewer(item: ReviewItem) -> Answer:
        nonlocal llamadas
        llamadas += 1
        if llamadas == 3:
            raise EOFError("Revisión cancelada por el usuario")
        return Answer(Verdict.CONFIRMED)

    with pytest.raises(EOFError, match=r"Revisión cancelada"):
        review_encounter(
            owner=owner,
            grouped_sections=grouped_sections,
            encounter=encounter,
            matches=matches,
            reviewer=reviewer,
            log=log,
            refine=lambda finding, text: finding,
        )

    eventos = [e for e in log if e.kind == EventKind.ITEM_REVIEWED]
    assert len(eventos) == 2
    assert [e.data["ref"] for e in eventos] == ["motivo:0", "examen:0"]


def test_review_codigo_rechazado_queda_fuera() -> None:
    """Caso 8: Código rechazado queda fuera de confirmed_codes y confirmado dentro."""
    owner, _spec1, _spec2, _f1, _f2, encounter, grouped_sections, m1, _m2, matches = (
        _crear_fixture_base()
    )
    log = EventLog()

    def reviewer(item: ReviewItem) -> Answer:
        if item.ref == "G43.9":
            return Answer(Verdict.CONFIRMED)
        if item.ref == "R51":
            return Answer(Verdict.REJECTED)
        return Answer(Verdict.CONFIRMED)

    resultado = review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=matches,
        reviewer=reviewer,
        log=log,
        refine=lambda finding, text: finding,
    )

    assert resultado.confirmed_codes == (m1,)
    eventos_codes = [
        e.data for e in log
        if e.kind == EventKind.ITEM_REVIEWED and e.data.get("item") == "code"
    ]
    assert eventos_codes == [
        {"item": "code", "ref": "G43.9", "verdict": "confirmed", "origin": "today"},
        {"item": "code", "ref": "R51", "verdict": "rejected", "origin": "today"},
    ]


# ---------------------------------------------------------------------------
# Plan 62: La revisión (§B.3)
# ---------------------------------------------------------------------------


def test_review_encounter_sufijo_en_history_y_both_no_en_today() -> None:
    """Caso 7 (§B.3): Un código HISTORY o BOTH se muestra con el sufijo del pack;
    uno TODAY, sin él. Sin sufijo configurado, ninguno lo lleva."""
    fixture = _crear_fixture_base()
    owner, encounter, grouped_sections = fixture[0], fixture[5], fixture[6]

    m_today = MatchedCode(
        DiagnosticCode("T01", "Cefalea aguda", 3, "test"), score=1.0, origin=CodeOrigin.TODAY
    )
    m_hist = MatchedCode(
        DiagnosticCode("H01", "Diabetes mellitus", 3, "test"), score=1.0, origin=CodeOrigin.HISTORY
    )
    m_both = MatchedCode(
        DiagnosticCode("B01", "Hipertensión esencial", 3, "test"), score=1.0, origin=CodeOrigin.BOTH
    )

    items_con_sufijo: list[ReviewItem] = []

    def rev_con_sufijo(item: ReviewItem) -> Answer:
        items_con_sufijo.append(item)
        return Answer(Verdict.CONFIRMED)

    # 1. Con sufijo configurado
    log1 = EventLog(window=20)
    review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=(m_today, m_hist, m_both),
        reviewer=rev_con_sufijo,
        log=log1,
        refine=lambda f, t: f,
        known_code_suffix="ya en la historia",
    )
    cod_items = [it for it in items_con_sufijo if it.kind == ItemKind.CODE]
    assert len(cod_items) == 3
    # TODAY: sin sufijo
    assert cod_items[0].ref == "T01"
    assert cod_items[0].text == "Cefalea aguda"
    assert "ya en la historia" not in cod_items[0].text

    # HISTORY: con sufijo
    assert cod_items[1].ref == "H01"
    assert cod_items[1].text == "Diabetes mellitus · ya en la historia"

    # BOTH: con sufijo
    assert cod_items[2].ref == "B01"
    assert cod_items[2].text == "Hipertensión esencial · ya en la historia"

    # 2. Sin sufijo configurado (None): ninguno lo lleva
    items_sin_sufijo: list[ReviewItem] = []

    def rev_sin_sufijo(it: ReviewItem) -> Answer:
        items_sin_sufijo.append(it)
        return Answer(Verdict.CONFIRMED)

    log2 = EventLog(window=20)
    review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=(m_today, m_hist, m_both),
        reviewer=rev_sin_sufijo,
        log=log2,
        refine=lambda f, t: f,
        known_code_suffix=None,
    )
    cod_items_sin = [it for it in items_sin_sufijo if it.kind == ItemKind.CODE]
    assert len(cod_items_sin) == 3
    assert cod_items_sin[0].text == "Cefalea aguda"
    assert cod_items_sin[1].text == "Diabetes mellitus"
    assert cod_items_sin[2].text == "Hipertensión esencial"


def test_review_encounter_evento_codigo_lleva_origin_hallazgo_no() -> None:
    """Caso 8 (§B.3): El evento de un código lleva origin; el de un hallazgo no."""
    fixture = _crear_fixture_base()
    owner, encounter, grouped_sections = fixture[0], fixture[5], fixture[6]

    m_hist = MatchedCode(
        DiagnosticCode("H01", "Diabetes mellitus", 3, "test"), score=1.0, origin=CodeOrigin.HISTORY
    )
    log = EventLog(window=20)

    review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=(m_hist,),
        reviewer=lambda it: Answer(Verdict.CONFIRMED),
        log=log,
        refine=lambda f, t: f,
    )

    eventos = [e for e in log if e.kind == EventKind.ITEM_REVIEWED]
    # Comprobar eventos de hallazgo: NO deben llevar "origin"
    hallazgos = [e for e in eventos if e.data["item"] == "finding"]
    assert len(hallazgos) >= 1
    for h in hallazgos:
        assert "origin" not in h.data

    # Comprobar eventos de código: SÍ llevan "origin"
    codigos = [e for e in eventos if e.data["item"] == "code"]
    assert len(codigos) == 1
    assert "origin" in codigos[0].data
    assert codigos[0].data["origin"] == "history"


def test_review_encounter_sin_textos_nuevos_comportamiento_identico() -> None:
    """Caso 34 (§B.6): review_encounter sin los textos nuevos se comporta exactamente como antes."""
    owner, spec1, _spec2, f1, _f2, _encounter, _grouped_sections, m1, _m2, _matches = (
        _crear_fixture_base()
    )
    f_neg = replace(f1, negated=f1.text)
    m_ev = replace(m1, evidence="frase con migraña")
    enc = Encounter(owner=owner, findings={"motivo": (f_neg,)})

    items: list[ReviewItem] = []

    def _reviewer(it: ReviewItem) -> Answer:
        items.append(it)
        return Answer(Verdict.CONFIRMED)

    review_encounter(
        owner=owner,
        grouped_sections=((spec1, (f_neg.text,)),),
        encounter=enc,
        matches=(m_ev,),
        reviewer=_reviewer,
        log=EventLog(),
        refine=lambda f, t: f,
    )
    assert len(items) == 2
    assert items[0].text == f1.text  # Sin sufijo negado
    assert items[1].text == m1.diagnostic.description  # Sin plantilla de evidencia


# ---------------------------------------------------------------------------
# Pruebas §B.1: Interfaz de revisión — ítems, una sola puerta (Plan 71)
# ---------------------------------------------------------------------------


def test_review_items_y_review_encounter_mismas_refs_y_orden() -> None:
    """Caso 1 (§B.1): review_items y review_encounter producen las mismas ref
    en el mismo orden."""
    owner, _spec1, _spec2, f1, f2, _enc, grouped_sections, _m1, _m2, matches = (
        _crear_fixture_base()
    )
    # findings en orden inverso al catálogo agrupado
    encounter = Encounter(owner=owner, findings={"examen": (f2,), "motivo": (f1,)})
    items_puros = review_items(
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=matches,
    )
    refs_puros = [item.ref for item in items_puros]

    log = EventLog()
    review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=matches,
        reviewer=lambda it: Answer(Verdict.CONFIRMED),
        log=log,
        refine=lambda f, t: f,
    )
    refs_eventos = [e.data["ref"] for e in log if e.kind == EventKind.ITEM_REVIEWED]

    assert refs_puros == refs_eventos
    assert refs_puros == ["motivo:0", "examen:0", "G43.9", "R51"]


def test_review_items_y_review_encounter_mismo_texto_con_sufijos() -> None:
    """Caso 2 (§B.1): El texto de cada ítem es idéntico entre review_items y
    review_encounter con sufijos de negado, historia y plantilla de evidencia."""
    owner, spec1, spec2, f1, f2, _enc, _gs, m1, m2, _m = _crear_fixture_base()
    f1_neg = replace(f1, text="No dolor de pecho", negated="No dolor de pecho")
    encounter = Encounter(owner=owner, findings={"motivo": (f1_neg,), "examen": (f2,)})
    grouped_sections = (
        (spec1, (f1_neg.text,)),
        (spec2, (f2.text,)),
    )
    m1_hist = replace(m1, origin=CodeOrigin.HISTORY, evidence="registro anterior de migraña")
    m2_today = replace(m2, origin=CodeOrigin.TODAY, evidence="cefalea en nuca")
    matches = (m1_hist, m2_today)

    known_suffix = "de la historia"
    evidence_tmpl = "{0} (según «{1}»)"
    negated_sfx = "descartado"

    items_puros = review_items(
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=matches,
        known_code_suffix=known_suffix,
        evidence_template=evidence_tmpl,
        negated_suffix=negated_sfx,
    )

    items_encounter: list[ReviewItem] = []

    def _reviewer(it: ReviewItem) -> Answer:
        items_encounter.append(it)
        return Answer(Verdict.CONFIRMED)

    review_encounter(
        owner=owner,
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=matches,
        reviewer=_reviewer,
        log=EventLog(),
        refine=lambda f, t: f,
        known_code_suffix=known_suffix,
        evidence_template=evidence_tmpl,
        negated_suffix=negated_sfx,
    )

    textos_puros = [i.text for i in items_puros]
    textos_encounter = [i.text for i in items_encounter]
    assert textos_puros == textos_encounter
    assert textos_puros == [
        "No dolor de pecho · descartado",
        "Tensión arterial 120/80",
        "Migraña · de la historia (según «registro anterior de migraña»)",
        "Cefalea (según «cefalea en nuca»)",
    ]


def test_review_items_allowed_por_clase() -> None:
    """Caso 3 (§B.1): allowed por clase — hallazgos traen 3 veredictos y códigos 2."""
    _owner, _spec1, _spec2, _f1, _f2, encounter, grouped_sections, _m1, _m2, matches = (
        _crear_fixture_base()
    )
    items = review_items(
        grouped_sections=grouped_sections,
        encounter=encounter,
        matches=matches,
    )
    hallazgos = [i for i in items if i.kind is ItemKind.FINDING]
    codigos = [i for i in items if i.kind is ItemKind.CODE]

    assert len(hallazgos) == 2
    assert len(codigos) == 2
    for h in hallazgos:
        assert h.allowed == FINDING_VERDICTS
        assert set(h.allowed) == {Verdict.CONFIRMED, Verdict.EDITED, Verdict.REJECTED}
    for c in codigos:
        assert c.allowed == CODE_VERDICTS
        assert set(c.allowed) == {Verdict.CONFIRMED, Verdict.REJECTED}


def test_finding_item_marca_negado_entero_no_mixto() -> None:
    """Caso 4 (§B.1): finding_item marca el sufijo solo si el hallazgo está
    enteramente negado (negated == text.strip()), no en mixto ni afirmado."""
    spec = SectionSpec("s", "Sección", (), True)
    f_afirmado = ClinicalFinding(text="Dolor de espalda", section_id="s", negated=None)
    f_mixto = ClinicalFinding(
        text="Fumar, no, lo dejé", section_id="s", negated="no, lo dejé"
    )
    f_entero = ClinicalFinding(
        text="  Sin fiebre  ", section_id="s", negated="Sin fiebre"
    )

    it_afirmado = finding_item(spec, 0, f_afirmado, negated_suffix="negado")
    it_mixto = finding_item(spec, 1, f_mixto, negated_suffix="negado")
    it_entero = finding_item(spec, 2, f_entero, negated_suffix="negado")

    assert it_afirmado.text == "Dolor de espalda"
    assert it_mixto.text == "Fumar, no, lo dejé"
    assert it_entero.text == "  Sin fiebre   · negado"


def test_code_item_compone_descripcion_sufijo_historia_y_evidencia_en_orden() -> None:
    """Caso 5 (§B.1): code_item compone descripción, sufijo de historia y
    evidencia en ese orden, y omite cada parte cuando no se declara."""
    diag = DiagnosticCode("I10", "Hipertensión", 3, "minsalud")

    # 1. Sin sufijos
    m_base = MatchedCode(diag, 0.9, origin=CodeOrigin.TODAY)
    it_base = code_item(m_base)
    assert it_base.text == "Hipertensión"

    # 2. Con sufijo historia pero origen TODAY -> se omite
    it_today_sfx = code_item(m_base, known_code_suffix="de la historia")
    assert it_today_sfx.text == "Hipertensión"

    # 3. Con sufijo historia y origen HISTORY -> se añade
    m_hist = replace(m_base, origin=CodeOrigin.HISTORY)
    it_hist = code_item(m_hist, known_code_suffix="de la historia")
    assert it_hist.text == "Hipertensión · de la historia"

    # 4. Con sufijo historia y evidencia -> en ese orden
    m_hist_ev = replace(m_hist, evidence="tensión 150/90")
    it_hist_ev = code_item(
        m_hist_ev,
        known_code_suffix="de la historia",
        evidence_template="{0} (motivo: {1})",
    )
    assert it_hist_ev.text == "Hipertensión · de la historia (motivo: tensión 150/90)"


