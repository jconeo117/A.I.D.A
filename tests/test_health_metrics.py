"""Tests para el módulo de métricas de revisión clínica (plan 54, I-124)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aida.arms.health.metrics import ReviewMetricsStore, ReviewRecord, format_stats, review_record
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import AgentId


def test_review_record_serializado_to_json_exacto() -> None:
    """Caso 1 (§B.1): review_record sobre un EventLog con ITEM_REVIEWED de
    hallazgos y códigos más INPUT_OBSERVED da to_json() literal."""
    log = EventLog(window=100)
    owner = AgentId("doc_1")

    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "finding", "ref": "motivo_consulta:0", "verdict": "confirmed"},
    )
    log.emit(
        tick=0,
        kind=EventKind.INPUT_OBSERVED,
        owner=owner,
        content="observación intermedia",
    )
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "finding", "ref": "enfermedad_actual:0", "verdict": "edited"},
    )
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "finding", "ref": "signos:0", "verdict": "rejected"},
    )
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "finding", "ref": "examen_fisico:0", "verdict": "confirmed"},
    )
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "code", "ref": "M234", "verdict": "rejected"},
    )
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "code", "ref": "M546", "verdict": "rejected"},
    )

    record = review_record(log, completed=True, seconds=42.5)

    expected = (
        '{"code_origins":{"today":{"rejected":2}},'
        '"codes":{"rejected":2},'
        '"completed":true,'
        '"fields":{},'
        '"findings":{"confirmed":2,"edited":1,"rejected":1},'
        '"seconds":42.5,'
        '"sections":{'
        '"enfermedad_actual":{"edited":1},'
        '"examen_fisico":{"confirmed":1},'
        '"motivo_consulta":{"confirmed":1},'
        '"signos":{"rejected":1}'
        "}}"
    )
    assert record.to_json() == expected


def test_review_record_clase_desconocida_levanta_value_error() -> None:
    """Caso 2 (§B.1): Un ITEM_REVIEWED con item='otro' da ValueError."""
    log = EventLog(window=10)
    owner = AgentId("doc_unknown")
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "otro", "ref": "seccion:0", "verdict": "confirmed"},
    )

    with pytest.raises(ValueError, match=r"clase de ítem desconocida en ITEM_REVIEWED: 'otro'"):
        review_record(log, completed=True, seconds=1.0)


def test_review_record_sin_datos_del_paciente() -> None:
    """Caso 3 (§B.1): Con owner='secreto_x' y INPUT_OBSERVED con 'secreto_x dijo algo',
    la cadena no aparece en to_json() y las claves son exactamente las esperadas."""
    log = EventLog(window=10)
    owner = AgentId("secreto_x")
    log.emit(
        tick=0,
        kind=EventKind.INPUT_OBSERVED,
        owner=owner,
        content="secreto_x dijo algo",
    )
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "finding", "ref": "motivo_consulta:0", "verdict": "confirmed"},
    )

    record = review_record(log, completed=True, seconds=15.0)
    json_line = record.to_json()

    assert "secreto_x" not in json_line
    assert "dijo algo" not in json_line
    raw = json.loads(json_line)
    assert set(raw.keys()) == {
        "code_origins",
        "codes",
        "completed",
        "fields",
        "findings",
        "seconds",
        "sections",
    }


def test_review_record_estabilidad_y_determinismo() -> None:
    """Caso 4 (§B.1): to_json -> from_json -> to_json da la misma línea;
    dos ReviewRecord construidos con distinto orden de inserción dan la misma línea."""
    r1 = ReviewRecord(
        completed=True,
        seconds=20.0,
        findings={"rejected": 1, "confirmed": 2, "edited": 1},
        codes={"rejected": 2, "confirmed": 1},
        sections={
            "signos": {"rejected": 1},
            "motivo_consulta": {"confirmed": 2, "edited": 1},
        },
        code_origins={"today": {"rejected": 2, "confirmed": 1}},
    )

    line1 = r1.to_json()
    from_line = ReviewRecord.from_json(line1)
    assert from_line.to_json() == line1
    assert from_line == r1

    # Mismo contenido con distinto orden en los dicts
    r2 = ReviewRecord(
        completed=True,
        seconds=20.0,
        findings={"confirmed": 2, "edited": 1, "rejected": 1},
        codes={"confirmed": 1, "rejected": 2},
        sections={
            "motivo_consulta": {"edited": 1, "confirmed": 2},
            "signos": {"rejected": 1},
        },
        code_origins={"today": {"confirmed": 1, "rejected": 2}},
    )
    assert r1.to_json() == r2.to_json()


def test_review_metrics_store_append_read_y_linea_ilegible(tmp_path: Path) -> None:
    """Caso 5 (§B.1): Store añade registros, read() los recupera, archivo inexistente
    da (), y línea ilegible da ValueError con número de línea."""
    path = tmp_path / "metrics" / "rev.jsonl"
    store = ReviewMetricsStore(path)

    # Inexistente
    assert store.read() == ()

    r1 = ReviewRecord(
        completed=True,
        seconds=10.5,
        findings={"confirmed": 1},
        codes={},
        sections={"motivo_consulta": {"confirmed": 1}},
        code_origins={},
    )
    r2 = ReviewRecord(
        completed=False,
        seconds=5.0,
        findings={"rejected": 1},
        codes={"rejected": 1},
        sections={"signos": {"rejected": 1}},
        code_origins={},
    )

    store.append(r1)
    store.append(r2)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert store.read() == (r1, r2)

    # Línea ilegible en línea 2
    path_corrupt = tmp_path / "metrics" / "corrupt.jsonl"
    store_corrupt = ReviewMetricsStore(path_corrupt)
    store_corrupt.append(r1)
    with path_corrupt.open("a", encoding="utf-8") as handle:
        handle.write("{no es json\n")

    with pytest.raises(ValueError, match=r":2: registro ilegible:"):
        store_corrupt.read()


def test_format_stats_salida_literal_y_vacio() -> None:
    """Caso 6 (§B.1): format_stats formatea conteos, tiempo medio y porcentajes;
    vacío produce 'sin revisiones registradas'."""
    assert format_stats(()) == "sin revisiones registradas"

    r1 = ReviewRecord(
        completed=True,
        seconds=42.5,
        findings={"confirmed": 2, "edited": 1, "rejected": 1},
        codes={"rejected": 2},
        sections={
            "enfermedad_actual": {"edited": 1},
            "examen_fisico": {"confirmed": 1},
            "motivo_consulta": {"confirmed": 1},
            "signos": {"rejected": 1},
        },
        code_origins={},
    )
    r2 = ReviewRecord(
        completed=False,
        seconds=7.0,
        findings={"confirmed": 1},
        codes={},
        sections={"motivo_consulta": {"confirmed": 1}},
        code_origins={},
    )

    stats = format_stats((r1, r2))
    expected = (
        "revisiones: 2 (1 completas, 1 incompletas)\n"
        "tiempo medio por revisión completa: 42.5 s\n"
        "hallazgos: 5 (confirmados 60.0 %, editados 20.0 %, rechazados 20.0 %)\n"
        "códigos: 2 (confirmados 0.0 %, rechazados 100.0 %)\n"
        "por sección:\n"
        "  enfermedad_actual: 1 (confirmados 0.0 %, editados 100.0 %, rechazados 0.0 %)\n"
        "  examen_fisico: 1 (confirmados 100.0 %, editados 0.0 %, rechazados 0.0 %)\n"
        "  motivo_consulta: 2 (confirmados 100.0 %, editados 0.0 %, rechazados 0.0 %)\n"
        "  signos: 1 (confirmados 0.0 %, editados 0.0 %, rechazados 100.0 %)"
    )
    assert stats == expected


# ---------------------------------------------------------------------------
# Plan 62: Las métricas (§B.4)
# ---------------------------------------------------------------------------


def test_review_record_cuenta_por_origen_y_veredicto() -> None:
    """Caso 9 (§B.4): review_record cuenta los códigos por origen y por veredicto."""
    log = EventLog(window=20)
    owner = AgentId("doc_metrics_origin")
    # today: 1 confirmed, 1 rejected
    log.emit(
        tick=0, kind=EventKind.ITEM_REVIEWED, owner=owner,
        data={"item": "code", "ref": "T01", "verdict": "confirmed", "origin": "today"},
    )
    log.emit(
        tick=0, kind=EventKind.ITEM_REVIEWED, owner=owner,
        data={"item": "code", "ref": "T02", "verdict": "rejected", "origin": "today"},
    )
    # history: 2 confirmed
    log.emit(
        tick=0, kind=EventKind.ITEM_REVIEWED, owner=owner,
        data={"item": "code", "ref": "H01", "verdict": "confirmed", "origin": "history"},
    )
    log.emit(
        tick=0, kind=EventKind.ITEM_REVIEWED, owner=owner,
        data={"item": "code", "ref": "H02", "verdict": "confirmed", "origin": "history"},
    )
    # both: 1 confirmed, 1 rejected
    log.emit(
        tick=0, kind=EventKind.ITEM_REVIEWED, owner=owner,
        data={"item": "code", "ref": "B01", "verdict": "confirmed", "origin": "both"},
    )
    log.emit(
        tick=0, kind=EventKind.ITEM_REVIEWED, owner=owner,
        data={"item": "code", "ref": "B02", "verdict": "rejected", "origin": "both"},
    )

    record = review_record(log, completed=True, seconds=10.0)
    assert record.codes == {"confirmed": 4, "rejected": 2}
    assert record.code_origins == {
        "today": {"confirmed": 1, "rejected": 1},
        "history": {"confirmed": 2},
        "both": {"confirmed": 1, "rejected": 1},
    }


def test_review_record_evento_sin_origin_cuenta_como_today() -> None:
    """Caso 10 (§B.4): Un evento de código sin origin se cuenta como today."""
    log = EventLog(window=10)
    owner = AgentId("doc_legacy")
    # Evento viejo de versión anterior (sin clave origin)
    log.emit(
        tick=0, kind=EventKind.ITEM_REVIEWED, owner=owner,
        data={"item": "code", "ref": "M01", "verdict": "confirmed"},
    )
    log.emit(
        tick=0, kind=EventKind.ITEM_REVIEWED, owner=owner,
        data={"item": "code", "ref": "M02", "verdict": "rejected"},
    )

    record = review_record(log, completed=True, seconds=5.0)
    assert record.codes == {"confirmed": 1, "rejected": 1}
    assert record.code_origins == {"today": {"confirmed": 1, "rejected": 1}}


def test_review_record_to_json_y_from_json_con_y_sin_code_origins() -> None:
    """Caso 11 (§B.4): to_json incluye code_origins ordenado;
    from_json de una línea sin esa clave da un registro con el desglose vacío."""
    # 1. to_json incluye code_origins ordenado y from_json lo recupera
    record = ReviewRecord(
        completed=True,
        seconds=15.0,
        findings={"confirmed": 1},
        codes={"confirmed": 2, "rejected": 1},
        sections={"motivo": {"confirmed": 1}},
        code_origins={"today": {"confirmed": 1}, "both": {"confirmed": 1, "rejected": 1}},
    )
    json_str = record.to_json()
    esperado_bloque = '"code_origins":{"both":{"confirmed":1,"rejected":1},"today":{"confirmed":1}}'
    assert esperado_bloque in json_str

    parsed = ReviewRecord.from_json(json_str)
    assert parsed.code_origins == {
        "both": {"confirmed": 1, "rejected": 1},
        "today": {"confirmed": 1},
    }
    assert parsed == record

    # 2. from_json de una línea legacy sin 'code_origins'
    legacy_json = json.dumps({
        "completed": True,
        "seconds": 15.0,
        "findings": {"confirmed": 1},
        "codes": {"confirmed": 1},
        "sections": {"motivo": {"confirmed": 1}},
    })
    legacy_record = ReviewRecord.from_json(legacy_json)
    assert legacy_record.code_origins == {}


def test_format_stats_imprime_bloque_codigos_por_origen() -> None:
    """Caso 12 (§B.4): format_stats imprime el bloque «códigos por origen» con
    las etiquetas en castellano, y no lo imprime si no hay ninguno."""
    # 1. Con orígenes: imprime "códigos por origen:" y las 3 etiquetas en castellano
    r1 = ReviewRecord(
        completed=True,
        seconds=10.0,
        findings={"confirmed": 1},
        codes={"confirmed": 3, "rejected": 1},
        sections={"motivo": {"confirmed": 1}},
        code_origins={
            "today": {"confirmed": 1},
            "history": {"confirmed": 1, "rejected": 1},
            "both": {"confirmed": 1},
        },
    )
    salida = format_stats((r1,))
    assert "códigos por origen:\n" in salida
    assert "  de esta consulta: 1 (confirmados 100.0 %, rechazados 0.0 %)" in salida
    assert "  solo en la historia: 2 (confirmados 50.0 %, rechazados 50.0 %)" in salida
    assert "  en ambas: 1 (confirmados 100.0 %, rechazados 0.0 %)" in salida

    # 2. Sin códigos / sin orígenes: no imprime el bloque "códigos por origen:"
    r_vacio = ReviewRecord(
        completed=True,
        seconds=5.0,
        findings={"confirmed": 1},
        codes={},
        sections={"motivo": {"confirmed": 1}},
        code_origins={},
    )
    salida_vacia = format_stats((r_vacio,))
    assert "códigos por origen" not in salida_vacia


# ---------------------------------------------------------------------------
# §B.4 — Métrica (Casos 13 a 15)
# ---------------------------------------------------------------------------


def test_review_record_cuenta_campos_en_fields() -> None:
    """Caso 13 (§B.4): review_record sobre eventos ITEM_REVIEWED de item='field'
    cuenta por campo y veredicto en fields."""
    log = EventLog(window=10)
    owner = AgentId("doc_fields")
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "field", "ref": "motivo", "verdict": "confirmed"},
    )
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "field", "ref": "plan", "verdict": "edited"},
    )
    log.emit(
        tick=0,
        kind=EventKind.ITEM_REVIEWED,
        owner=owner,
        data={"item": "field", "ref": "plan", "verdict": "confirmed"},
    )
    record = review_record(log, completed=True, seconds=10.0)
    assert record.fields == {
        "motivo": {"confirmed": 1},
        "plan": {"confirmed": 1, "edited": 1},
    }


def test_review_record_to_json_from_json_fields_y_retrocompatibilidad() -> None:
    """Caso 14 (§B.4): to_json/from_json conservan fields; una línea vieja sin 'fields'
    se lee con fields == {}."""
    record = ReviewRecord(
        completed=True,
        seconds=12.0,
        findings={},
        codes={},
        sections={},
        code_origins={},
        fields={"motivo": {"confirmed": 2}, "plan": {"edited": 1}},
    )
    json_str = record.to_json()
    assert '"fields":{"motivo":{"confirmed":2},"plan":{"edited":1}}' in json_str
    parsed = ReviewRecord.from_json(json_str)
    assert parsed.fields == {"motivo": {"confirmed": 2}, "plan": {"edited": 1}}
    assert parsed == record

    # Línea vieja sin "fields"
    legacy_json = json.dumps({
        "completed": True,
        "seconds": 10.0,
        "findings": {},
        "codes": {},
        "sections": {},
    })
    legacy_record = ReviewRecord.from_json(legacy_json)
    assert legacy_record.fields == {}


def test_format_stats_imprime_bloque_campos_de_la_nota() -> None:
    """Caso 15 (§B.4): format_stats imprime 'campos de la nota:' y una línea por campo
    con el porcentaje de editados."""
    record = ReviewRecord(
        completed=True,
        seconds=10.0,
        findings={},
        codes={},
        sections={},
        code_origins={},
        fields={
            "motivo": {"confirmed": 2},
            "plan": {"confirmed": 1, "edited": 1},
        },
    )
    salida = format_stats((record,))
    assert "campos de la nota:\n" in salida
    assert "  motivo: 2 (confirmados 100.0 %, editados 0.0 %)" in salida
    assert "  plan: 2 (confirmados 50.0 %, editados 50.0 %)" in salida

