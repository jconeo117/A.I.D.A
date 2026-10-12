"""Pruebas del módulo de «No consta», ausencias con respaldo y trazabilidad (plan 84).

Cubre:
- §B.1 traceability.py:
  1. load_traceability: lectura de pack, validaciones estrictas y fallos explícitos.
  2. history_profile: lectura de fecha 24/09/2026, encabezados de perfil en orden,
     ignora cabecera (M6).
  3. complete_note: con historia, sin historia, preserva críticos con texto (M4),
     no toca no-críticos, not_recorded exacto.
  4. absence_flags: respaldo en mismo apartado vs otro vs sin hallazgos (M2), sin marca
     si no hay ausencia, texto del motor no vigilado (M3), estabilidad del id.
  5. unresolved: resuelto si confirmado o si frase borrada (M5), pendiente si no.
- §B.2 La puerta (Consultation):
  6. review con ausencia pendiente: AbsenceUnresolvedError, nada en caché, sin
     ITEM_REVIEWED, sin auditoría (M1).
  7. review confirmando: guarda y registra review_saved con absences_flagged y absences_confirmed.
  8. review con frase eliminada en note_edits: guarda sin confirmar (M5).
  9. render_note con traceability completa críticos; sin traceability los deja como antes (M7).
- §B.3 Revisión, aplicación y CLI:
  10. _render_page: con trace muestra aviso, contador, etiqueta del motor, bloque de fecha,
      sufijo HC en historia; sin trace la página clásica.
  11. parse_submission: absences_confirmed ausente => vacío, no-lista o no-cadenas => error,
      lista válida => frozenset (M10).
  12. Aplicación: /guardar con ausencia pendiente => 400; confirmada => 200 (M8, M10).
  13. run._build_consultation pasa traceability y note_sections (M9).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import yaml
from cryptography.fernet import Fernet

from aida import run
from aida.arms.health.cache import ConsultationCache
from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.consultation import (
    AbsenceUnresolvedError,
    Consultation,
    PreparedConsultation,
)
from aida.arms.health.history import HistoryLayout, HistorySection
from aida.arms.health.match import CodeOrigin, MatchedCode, build_index
from aida.arms.health.model import ClinicalFinding, Encounter
from aida.arms.health.note import RenderedNote, load_note_settings
from aida.arms.health.review import Answer, Verdict
from aida.arms.health.review_ui import (
    _render_page,
    history_view,
    load_ui_settings,
    parse_submission,
)
from aida.arms.health.sections import load_sections
from aida.arms.health.traceability import (
    AbsenceFlag,
    HistoryProfile,
    ReviewTrace,
    absence_flags,
    complete_note,
    history_profile,
    load_traceability,
    unresolved,
)
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.platform.audit import AuditLog
from aida.platform.identity import Professional
from tests.conftest import _ORIGINAL_HANDLE_REQUEST
from tests.test_health_app import (
    _crear_app_ready,
    _crear_wav_minimo,
    _lanzar_servidor_app,
)

_ACTOR = Professional(id="doc1", name="Dra. Laura Gómez", registry="52.341.889")

_HISTORIA_EJEMPLO = """ Centro Médico Demostrativo Aurora (ficticio)
HISTORIA CLÍNICA - RESUMEN

Identificador del paciente: SUJETO-01

Edad: 58 años

Sexo: Masculino

Fecha: 24/09/2026

ANTECEDENTES PERSONALES
Hipertensión esencial

Diabetes mellitus tipo 2

Dislipidemia

Obesidad

Hipotiroidismo primario

Enfermedad por reflujo gastroesofágico

Osteoartrosis de rodilla derecha

ALERGIAS
Alergia medicamentosa: penicilina

MEDICACION ACTUAL
Losartán 50 mg - cada 12 horas
Metformina 850 mg - cada 12 horas
Atorvastatina 20 mg - cada 24 horas
Levotiroxina 75 microgramos - cada 24 horas
Omeprazol 20 mg - cada 24 horas

ANTECEDENTES QUIRURGICOS
Apendicectomía - año 1998
Colecistectomía laparoscópica - año 2014
"""


@pytest.fixture(autouse=True)
def _permitir_servidor_app_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restaura el transporte real de httpx para comunicar con el servidor local."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


# ---------------------------------------------------------------------------
# §B.1 — traceability.py
# ---------------------------------------------------------------------------


def test_load_traceability_lee_pack_y_valida_campos(tmp_path: Path) -> None:
    """Caso 1 (§B.1): load_traceability lee el pack por defecto y valida estructura estricta."""
    ajustes = load_traceability()
    assert "antecedentes" in ajustes.critical
    assert "medicamentos" in ajustes.critical
    assert ajustes.not_recorded == "No consta."
    assert "niega" in ajustes.absence_markers
    assert len(ajustes.profile) >= 3

    # No es mapeo YAML
    f_lista = tmp_path / "lista.yaml"
    f_lista.write_text("- item\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="debe ser un mapeo YAML"):
        load_traceability(f_lista)

    # Falta un texto
    base_dict = yaml.safe_load(
        (Path("packs/health/consultorio/traceability.yaml")).read_text(encoding="utf-8")
    )
    sin_texto = dict(base_dict)
    sin_texto.pop("not_recorded")
    f_bad_text = tmp_path / "sin_texto.yaml"
    f_bad_text.write_text(yaml.dump(sin_texto), encoding="utf-8")
    with pytest.raises(ValueError, match="faltan textos"):
        load_traceability(f_bad_text)

    # critical_sections no mapea a listas
    bad_crit = dict(base_dict)
    bad_crit["critical_sections"] = {"antecedentes": "no es lista"}
    f_bad_crit = tmp_path / "bad_crit.yaml"
    f_bad_crit.write_text(yaml.dump(bad_crit), encoding="utf-8")
    with pytest.raises(ValueError, match="'critical_sections' debe mapear apartados a encabezados"):
        load_traceability(f_bad_crit)

    # absence_markers vacío
    bad_markers = dict(base_dict)
    bad_markers["absence_markers"] = []
    f_bad_markers = tmp_path / "bad_markers.yaml"
    f_bad_markers.write_text(yaml.dump(bad_markers), encoding="utf-8")
    with pytest.raises(ValueError, match="'absence_markers' debe ser una lista no vacía"):
        load_traceability(f_bad_markers)

    # profile no trae encabezado y etiqueta
    bad_profile = dict(base_dict)
    bad_profile["profile"] = [{"heading": "ALERGIAS"}]  # falta label
    f_bad_profile = tmp_path / "bad_profile.yaml"
    f_bad_profile.write_text(yaml.dump(bad_profile), encoding="utf-8")
    with pytest.raises(ValueError, match="'profile' debe ser una lista de encabezado y etiqueta"):
        load_traceability(f_bad_profile)


def test_history_profile_extrae_fecha_items_e_ignora_cabecera() -> None:
    """Caso 2 (§B.1, M6): history_profile lee fecha 24/09/2026, items en orden e ignora cabecera."""
    layout = HistoryLayout(
        sections=(
            HistorySection(heading="ANTECEDENTES PERSONALES", codable=True),
            HistorySection(heading="ALERGIAS", codable=True),
            HistorySection(heading="MEDICACION ACTUAL", codable=True),
            HistorySection(heading="ANTECEDENTES QUIRURGICOS", codable=True),
        ),
        ignore=("FECHA:", "IDENTIFICADOR", "EDAD:", "SEXO:"),
    )
    ajustes = load_traceability()
    perfil = history_profile(_HISTORIA_EJEMPLO, layout, ajustes)

    assert perfil.date == "24/09/2026"  # Si M6 se aplica, fecha queda None
    etiquetas = [etiqueta for etiqueta, _lineas in perfil.items]
    assert etiquetas == ["Alergias", "Medicación", "Cirugías"]

    # Lo de la cabecera no entra
    for encabezado in perfil.lines:
        assert "EDAD" not in encabezado
        assert "SEXO" not in encabezado
        assert "AURORA" not in encabezado


def test_complete_note_casos_con_y_sin_historia_preserva_criticos_con_texto() -> None:
    """Caso 3 (§B.1, M4): complete_note con y sin historia, preserva críticos con texto (M4),
    no toca no críticos y not_recorded lista solo lo escrito."""
    ajustes = load_traceability()
    secciones = tuple(s for s in load_sections().sections if s.in_note)
    sec_mot = next(s for s in secciones if s.section_id == "motivo_consulta")
    sec_ant = next(s for s in secciones if s.section_id == "antecedentes")

    layout = HistoryLayout(
        sections=(
            HistorySection(heading="ALERGIAS", codable=True),
            HistorySection(heading="MEDICACION ACTUAL", codable=True),
        ),
        ignore=("FECHA:",),
    )
    perfil_con = history_profile(_HISTORIA_EJEMPLO, layout, ajustes)
    perfil_sin = history_profile("", layout, ajustes)

    # 1. Con historia: apartados críticos vacíos dicen que consta en la historia
    nota_vacia = RenderedNote(
        fields=((sec_mot, "Acude por dolor de cabeza."),),
        redacted=True,
        provider="p/m",
    )
    completa_con = complete_note(
        nota_vacia, sections=secciones, profile=perfil_con, settings=ajustes
    )
    campos_con = {s.section_id: t for s, t in completa_con.fields}
    assert (
        campos_con["antecedentes"]
        == "No se habló en la consulta; consta en la historia del 24/09/2026."
    )
    assert (
        campos_con["medicamentos"]
        == "No se habló en la consulta; consta en la historia del 24/09/2026."
    )
    assert completa_con.not_recorded == ("antecedentes", "medicamentos")

    # 2. Sin historia: apartados críticos dicen 'No consta.'
    completa_sin = complete_note(
        nota_vacia, sections=secciones, profile=perfil_sin, settings=ajustes
    )
    campos_sin = {s.section_id: t for s, t in completa_sin.fields}
    assert campos_sin["antecedentes"] == "No consta."
    assert campos_sin["medicamentos"] == "No consta."
    assert completa_sin.not_recorded == ("antecedentes", "medicamentos")

    # 3. Crítico CON texto no se toca (M4)
    nota_con_ant = RenderedNote(
        fields=(
            (sec_mot, "Acude por dolor."),
            (sec_ant, "Hipertensión arterial crónica controlada."),
        ),
        redacted=True,
        provider="p/m",
    )
    completa_preservada = complete_note(
        nota_con_ant, sections=secciones, profile=perfil_con, settings=ajustes
    )
    campos_preservados = {s.section_id: t for s, t in completa_preservada.fields}
    assert campos_preservados["antecedentes"] == "Hipertensión arterial crónica controlada."
    assert "antecedentes" not in completa_preservada.not_recorded
    assert (
        campos_preservados["medicamentos"]
        == "No se habló en la consulta; consta en la historia del 24/09/2026."
    )
    assert completa_preservada.not_recorded == ("medicamentos",)


def test_absence_flags_respaldo_mismo_apartado_otro_o_vacio_y_motor() -> None:
    """Caso 4 (§B.1, M2, M3): ausencia con respaldo en mismo apartado vs otro
    apartado vs sin hallazgos (M2), frase sin marca no se marca, texto del motor
    no se vigila (M3) e id estable."""
    ajustes = load_traceability()
    secciones = tuple(s for s in load_sections().sections if s.in_note)
    sec_ant = next(s for s in secciones if s.section_id == "antecedentes")

    # 1. Respaldo en el MISMO apartado: comparte palabra con contenido ('asma') => sin marca
    nota = RenderedNote(
        fields=((sec_ant, "Niega asma."),),
        redacted=True,
        provider="p/m",
    )
    findings_mismo = {
        "antecedentes": (
            ClinicalFinding(text="No, asma no refiere", section_id="antecedentes", negated="asma"),
        )
    }
    flags_mismo = absence_flags(nota, findings_mismo, ajustes)
    assert flags_mismo == ()

    # 2. Respaldo en OTRO apartado (en motivo_consulta en vez de antecedentes) => con marca
    findings_otro = {
        "motivo_consulta": (
            ClinicalFinding(
                text="No, asma no refiere", section_id="motivo_consulta", negated="asma"
            ),
        )
    }
    flags_otro = absence_flags(nota, findings_otro, ajustes)
    assert len(flags_otro) == 1
    assert flags_otro[0].section_id == "antecedentes"
    assert flags_otro[0].sentence == "Niega asma."

    # 3. Sin hallazgos negados en antecedentes => con marca (M2)
    flags_sin = absence_flags(nota, {}, ajustes)
    assert len(flags_sin) == 1
    assert flags_sin[0].section_id == "antecedentes"

    # 4. Frase sin marca de ausencia => sin marca
    nota_afirma = RenderedNote(
        fields=((sec_ant, "Presenta hipertensión arterial."),),
        redacted=True,
        provider="p/m",
    )
    assert absence_flags(nota_afirma, {}, ajustes) == ()

    # 5. Texto escrito por el motor (not_recorded) => sin marca (M3)
    nota_motor = RenderedNote(
        fields=((sec_ant, "Niega asma."),),
        redacted=True,
        provider="p/m",
        not_recorded=("antecedentes",),
    )
    assert absence_flags(nota_motor, {}, ajustes) == ()

    # 6. ID estable para la misma frase y apartado
    f1 = absence_flags(nota, {}, ajustes)[0]
    f2 = absence_flags(nota, {}, ajustes)[0]
    assert f1.id == f2.id


def test_unresolved_marca_confirmada_o_borrada_se_resuelve_sino_pendiente() -> None:
    """Caso 5 (§B.1, M5): unresolved se resuelve si confirmada o si frase
    quitada del texto final (M5); sino queda pendiente."""
    flag = AbsenceFlag(id="h1", section_id="antecedentes", sentence="Niega alergias.")
    flags = (flag,)

    # 1. Confirmada => resuelta (vacío)
    res_conf = unresolved(flags, {"antecedentes": "Niega alergias."}, frozenset({"h1"}))
    assert res_conf == ()

    # 2. Frase quitada del texto final (sin confirmar) => resuelta (M5)
    res_borr = unresolved(flags, {"antecedentes": "Sin datos evaluados."}, frozenset())
    assert res_borr == ()  # Si M5 se aplica, no evalúa presencia de la frase y retorna (flag,)

    # 3. Ni una cosa ni otra (sigue en el texto y no confirmada) => pendiente
    res_pend = unresolved(flags, {"antecedentes": "Niega alergias."}, frozenset())
    assert res_pend == (flag,)


# ---------------------------------------------------------------------------
# §B.2 — La puerta (Consultation.review y render_note)
# ---------------------------------------------------------------------------


def _crear_consultation_test(
    tmp_path: Path,
    *,
    traceability_active: bool = True,
) -> tuple[
    Consultation,
    PreparedConsultation,
    RenderedNote,
    AbsenceFlag,
    EventLog,
    AuditLog,
    ConsultationCache,
]:
    log = EventLog()
    cache = ConsultationCache(tmp_path / "cache", key=Fernet.generate_key())
    audit = AuditLog(tmp_path / "audit.jsonl", key=Fernet.generate_key())
    traceability = load_traceability() if traceability_active else None
    note_sections = tuple(s for s in load_sections().sections if s.in_note)

    c = Consultation(
        wording=default_summary_wording(),
        stt=cast(Any, object()),
        ocr=cast(Any, object()),
        catalog_index=build_index((DiagnosticCode("I10X", "Hipertensión esencial", 4, "cat"),)),
        cache=cache,
        log=log,
        audit=audit,
        traceability=traceability,
        note_sections=note_sections,
        note_settings=load_note_settings(),
        provider=None,
    )

    sec_mot = next(s for s in note_sections if s.section_id == "motivo_consulta")
    sec_ant = next(s for s in note_sections if s.section_id == "antecedentes")

    note = RenderedNote(
        fields=((sec_mot, "Cefalea frontal."), (sec_ant, "Niega alergias.")),
        redacted=True,
        provider="p/m",
    )
    encounter = Encounter(
        owner=AgentId("HC-01"),
        findings={
            "motivo_consulta": (
                ClinicalFinding(text="cefalea", section_id="motivo_consulta", negated=None),
            ),
            "antecedentes": (),
        },
    )
    prepared = PreparedConsultation(
        owner=AgentId("HC-01"),
        transcript="paciente con dolor",
        history="",
        matches=(),
        grouped_sections=((sec_mot, ("cefalea",)), (sec_ant, ())),
        encounter=encounter,
    )
    flag = (
        c.absence_flags(prepared, note)[0]
        if traceability_active
        else AbsenceFlag("dummy", "x", "y")
    )
    return c, prepared, note, flag, log, audit, cache


def test_review_con_ausencia_pendiente_lanza_error_sin_cache_ni_eventos_ni_auditoria(
    tmp_path: Path,
) -> None:
    """Caso 6 (§B.2, M1): review con ausencia pendiente lanza AbsenceUnresolvedError y no deja
    nada en caché, ni ITEM_REVIEWED, ni auditoría."""
    c, prepared, note, flag, log, audit, cache = _crear_consultation_test(tmp_path)

    with pytest.raises(AbsenceUnresolvedError) as exc_info:
        c.review(
            prepared,
            reviewer=lambda item: Answer(Verdict.CONFIRMED),
            actor=_ACTOR,
            note=note,
            note_edits={"motivo_consulta": "Cefalea frontal.", "antecedentes": "Niega alergias."},
            absences_confirmed=frozenset(),
        )
    assert exc_info.value.pending == (flag,)

    # Nada en caché
    assert cache.load(owner=AgentId("HC-01")) is None
    # Ni ITEM_REVIEWED
    assert len(log.of_kind(EventKind.ITEM_REVIEWED)) == 0
    # Ni auditoría
    assert len(audit.lines()) == 0


def test_review_confirmando_ausencia_guarda_y_audita_flagged_y_confirmed(
    tmp_path: Path,
) -> None:
    """Caso 7 (§B.2): review confirmando la ausencia guarda y lleva
    absences_flagged y absences_confirmed."""
    c, prepared, note, flag, _log, audit, _cache = _crear_consultation_test(tmp_path)

    res = c.review(
        prepared,
        reviewer=lambda item: Answer(Verdict.CONFIRMED),
        actor=_ACTOR,
        note=note,
        note_edits={"motivo_consulta": "Cefalea frontal.", "antecedentes": "Niega alergias."},
        absences_confirmed=frozenset({flag.id}),
    )
    assert res is not None
    lineas = audit.lines()
    assert len(lineas) == 1
    assert lineas[0]["kind"] == "review_saved"
    datos = json.loads(lineas[0]["data"])
    assert datos["absences_flagged"] == flag.id
    assert datos["absences_confirmed"] == flag.id


def test_review_quitando_frase_del_texto_final_sin_confirmar_guarda(tmp_path: Path) -> None:
    """Caso 8 (§B.2, M5): review quitando la frase del texto final guarda sin confirmar."""
    c, prepared, note, _flag, _log, audit, _cache = _crear_consultation_test(tmp_path)

    res = c.review(
        prepared,
        reviewer=lambda item: Answer(Verdict.CONFIRMED),
        actor=_ACTOR,
        note=note,
        note_edits={
            "motivo_consulta": "Cefalea frontal.",
            "antecedentes": "Alergias no indagadas.",
        },
        absences_confirmed=frozenset(),
    )
    assert res is not None
    lineas = audit.lines()
    assert len(lineas) == 1
    assert lineas[0]["kind"] == "review_saved"


def test_render_note_con_traceability_completa_criticos_y_sin_ella_no(tmp_path: Path) -> None:
    """Caso 9 (§B.2, M7): render_note con traceability completa críticos; sin ella, no."""
    # 1. Con traceability
    c_con, prepared, _note, _flag, _log, _audit, _cache = _crear_consultation_test(
        tmp_path / "con", traceability_active=True
    )
    nota_con = c_con.render_note(prepared)
    campos_con = {s.section_id: t for s, t in nota_con.fields}
    assert "antecedentes" in campos_con
    assert campos_con["antecedentes"] == "No consta."  # Si M7 se aplica, no completa críticos

    # 2. Sin traceability
    c_sin, _p, _n, _f, _l, _a, _c = _crear_consultation_test(
        tmp_path / "sin", traceability_active=False
    )
    nota_sin = c_sin.render_note(prepared)
    campos_sin = {s.section_id: t for s, t in nota_sin.fields}
    assert "antecedentes" not in campos_sin or not campos_sin["antecedentes"].strip()


# ---------------------------------------------------------------------------
# §B.3 — Revisión, aplicación y CLI
# ---------------------------------------------------------------------------


def test_render_page_con_y_sin_trace() -> None:
    """Caso 10 (§B.3): _render_page con trace incluye avisos, contador, badge, bloque de fecha
    y fecha en códigos de historia; sin trace produce la página anterior."""
    settings = load_traceability()
    sections = tuple(s for s in load_sections().sections if s.in_note)
    sec_mot = next(s for s in sections if s.section_id == "motivo_consulta")
    sec_ant = next(s for s in sections if s.section_id == "antecedentes")

    note = RenderedNote(
        fields=((sec_mot, "Cefalea."), (sec_ant, "No consta.")),
        redacted=True,
        provider="p/m",
        not_recorded=("antecedentes",),
    )
    flag = AbsenceFlag(id="fl1", section_id="motivo_consulta", sentence="Niega dolor.")
    profile = HistoryProfile(
        date="24/09/2026",
        lines={"ALERGIAS": ("Penicilina",)},
        items=(("Alergias", ("Penicilina",)),),
    )
    trace = ReviewTrace(flags=(flag,), profile=profile, settings=settings)

    matches = (
        MatchedCode(
            diagnostic=DiagnosticCode("I10", "Hipertensión", 4, "cat"),
            origin=CodeOrigin.TODAY,
            score=0.9,
        ),
        MatchedCode(
            diagnostic=DiagnosticCode("E11", "Diabetes", 4, "cat"),
            origin=CodeOrigin.HISTORY,
            score=0.8,
        ),
    )
    hview = history_view(None, "Historia previa")
    ui_settings = load_ui_settings()
    now = datetime.now(UTC)

    # Con trace
    html_con = _render_page(
        matches, note, "transcripción", hview, "HC-01", now, ui_settings.wording, "tok", trace=trace
    )
    assert '<div class="prior">' in html_con
    assert 'data-sentence="Niega dolor."' in html_con
    assert 'id="stat-absence"' in html_con
    assert '<span class="by-aida">No se habló de ello</span>' in html_con
    assert "· HC 24/09/2026" in html_con
    art_today = re.search(r'data-ref="I10".*?</article>', html_con, re.S)
    assert art_today is not None
    assert "· HC 24/09/2026" not in art_today.group(0)

    # Sin trace
    html_sin = _render_page(
        matches, note, "transcripción", hview, "HC-01", now, ui_settings.wording, "tok", trace=None
    )
    assert '<div class="prior">' not in html_sin
    assert 'id="stat-absence"' not in html_sin
    assert '<span class="by-aida">' not in html_sin
    assert "· HC 24/09/2026" not in html_sin


def test_parse_submission_valida_absences_confirmed() -> None:
    """Caso 11 (§B.3, M10): parse_submission valida absences_confirmed ausente => vacío,
    no lista o con no-cadenas => error, lista válida => frozenset."""
    sections = tuple(s for s in load_sections().sections if s.in_note)
    campos = {s.section_id: "texto" for s in sections}
    base_payload = {
        "answers": {},
        "elapsed_seconds": 1.0,
        "note": campos,
    }
    dummy_note = RenderedNote(
        fields=tuple((s, "texto") for s in sections),
        redacted=True,
        provider="p/m",
    )

    # 1. Ausente => frozenset vacío
    sub1, err1 = parse_submission(base_payload, items=(), note=dummy_note, measured_elapsed=5.0)
    assert err1 is None or err1 == ""
    assert sub1 is not None
    assert sub1.absences_confirmed == frozenset()

    # 2. No lista => error
    bad_type = dict(base_payload)
    bad_type["absences_confirmed"] = "cadena_no_lista"
    sub2, err2 = parse_submission(bad_type, items=(), note=dummy_note, measured_elapsed=5.0)
    assert sub2 is None
    assert "absences_confirmed inválido" in err2

    # 3. Lista con no-cadenas => error
    bad_items = dict(base_payload)
    bad_items["absences_confirmed"] = [123]
    sub3, err3 = parse_submission(bad_items, items=(), note=dummy_note, measured_elapsed=5.0)
    assert sub3 is None
    assert "absences_confirmed inválido" in err3

    # 4. Lista válida => frozenset con los ids (M10)
    good = dict(base_payload)
    good["absences_confirmed"] = ["h1", "h2"]
    sub4, err4 = parse_submission(good, items=(), note=dummy_note, measured_elapsed=5.0)
    assert err4 is None or err4 == ""
    assert sub4 is not None
    assert sub4.absences_confirmed == frozenset({"h1", "h2"})


def test_app_guardar_bloquea_ausencia_pendiente_y_guarda_confirmada(tmp_path: Path) -> None:
    """Caso 12 (§B.3, M8, M10): /guardar con una ausencia pendiente da 400; confirmada da 200."""
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    ready = _crear_app_ready(tmp_path)
    traceability = load_traceability()
    sections = tuple(s for s in load_sections().sections if s.in_note)
    ready.consultation._traceability = traceability
    ready.consultation._note_sections = sections

    original_render = ready.consultation.render_note
    inventada = "Niega antecedentes quirúrgicos."

    def con_ausencia(prep: PreparedConsultation) -> RenderedNote:
        nota = original_render(prep)
        campos = tuple(
            (
                spec,
                f"{texto} {inventada}".strip() if spec.section_id == "motivo_consulta" else texto,
            )
            for spec, texto in nota.fields
        )
        return replace(nota, fields=campos)

    ready.consultation.render_note = con_ausencia  # type: ignore[method-assign,assignment]

    url, _token, _port, _hilo, _info = _lanzar_servidor_app(lambda: ready, identificar=False)
    r_perf = httpx.post(
        f"{url}/perfil",
        json={
            "name": "Dra. Laura Gómez",
            "registry": "52.341.889",
            "pin": "1234",
            "pin2": "1234",
        },
        timeout=5.0,
    )
    assert r_perf.status_code == 200, f"Error en perfil: {r_perf.status_code} {r_perf.text}"
    try:
        httpx.post(
            f"{url}/audio",
            content=_crear_wav_minimo(),
            headers={"X-Name": "consulta.wav"},
            timeout=5.0,
        )
        httpx.post(
            f"{url}/documento",
            content=b"%PDF-1.4 prueba",
            headers={"X-Name": "historia.pdf"},
            timeout=5.0,
        )
        httpx.post(f"{url}/empezar", json={"owner": "HC-01", "consent": True}, timeout=5.0)

        for _ in range(50):
            est = httpx.get(f"{url}/estado", timeout=5.0).json()
            if est.get("state") == "review":
                break
            time.sleep(0.1)

        page = httpx.get(url, timeout=5.0).text
        m = re.search(r'data-flag="([0-9a-f]+)"', page)
        assert m is not None, "debió encontrar la marca en la interfaz"
        flag_id = m.group(1)

        field_ids = re.findall(r'data-field-id="([^"]+)"', page)
        campos_envio = {fid: "texto" for fid in field_ids}
        campos_envio["motivo_consulta"] = f"Cefalea frontal pulsátil. {inventada}"

        # 1. Guardar sin confirmar => 400
        r_fail = httpx.post(
            f"{url}/guardar",
            json={
                "answers": {"I10X": {"verdict": "confirmed"}},
                "note": campos_envio,
                "elapsed_seconds": 1.0,
                "absences_confirmed": [],
            },
            timeout=5.0,
        )
        assert r_fail.status_code == 400
        assert r_fail.json().get("error") == "Hay ausencias sin respaldo por corregir o confirmar."

        # 2. Guardar confirmando => 200 (M8, M10)
        r_ok = httpx.post(
            f"{url}/guardar",
            json={
                "answers": {"I10X": {"verdict": "confirmed"}},
                "note": campos_envio,
                "elapsed_seconds": 1.0,
                "absences_confirmed": [flag_id],
            },
            timeout=5.0,
        )
        assert r_ok.status_code == 200
        assert r_ok.json().get("ok") is True
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)


def test_run_build_consultation_carga_traceability_y_note_sections() -> None:
    """Caso 13 (§B.3, M9): run._build_consultation pasa traceability y note_sections."""
    args = run._construir_parser().parse_args(
        ["--consultation", "--audio", "audio.wav", "--consent", "--as", "000"]
    )
    key = Fernet.generate_key().decode()
    c = run._build_consultation(args, stt=cast(Any, object()), ocr=cast(Any, object()), key=key)
    assert c is not None
    assert c.traceability is not None  # Si M9 se aplica, traceability queda None
    assert c.traceability == load_traceability()
    assert c._note_sections == tuple(s for s in load_sections().sections if s.in_note)
