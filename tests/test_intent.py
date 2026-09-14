"""Pruebas unitarias del clasificador de intención (`aida.intent.classify`).

Cubre la unidad U6 según la especificación de `.claude/workflow/20-plan-u6.md` §6.1.
Verifica de forma aislada (sin pasar por `Loop`) el comportamiento de
`BagOfWordsClassifier` y su factoría `build_classifier`:
1. Coincidencia exacta (score == 1.0).
2. Coincidencia parcial por encima o igual al umbral (threshold <= score < 1.0).
3. Disyunción de tokens devuelve None.
4. Consulta por debajo de umbral devuelve None y consulta con score == threshold se reconoce.
5. Insensibilidad a mayúsculas y minúsculas.
6. Desempate determinista por orden lexicográfico de IntentId.
7. Pack sin intenciones devuelve None en build_classifier.
8. Reconocimiento de cada intención declarada usando su primer ejemplo.
9. Determinismo en llamadas repetidas.
10. Conformidad con el protocolo IntentClassifier.
11. Entradas vacías o con espacios en blanco devuelven None sin excepción.
"""

from __future__ import annotations

import math

import aida.engine.ontology as ontology
import aida.engine.pack as pack
from aida.engine.ports import IntentClassifier
from aida.intent.classify import (
    BagOfWordsClassifier,
    _cosine,
    _tokenize,
    _vectorize,
    build_classifier,
)


def _make_intent(
    id_: str,
    need_id: str,
    examples: tuple[str, ...],
    label: str | None = None,
) -> ontology.IntentSpec:
    """Construye un IntentSpec con identificadores nominales tipados."""
    return ontology.IntentSpec(
        id=ontology.IntentId(id_),
        label=label or f"Intent {id_}",
        need=ontology.NeedId(need_id),
        examples=examples,
    )


def _make_pack_for_intents(
    intents: list[ontology.IntentSpec] | None = None,
    *,
    threshold: float = 0.35,
) -> pack.ContentPack:
    """Construye un ContentPack válido para aislar las pruebas de intención.

    Garantiza que toda necesidad referenciada tenga su correspondiente
    acción alcanzable (delta > 0.0) para satisfacer las validaciones del pack.
    """
    declared_intents = list(intents or [])
    unique_needs = sorted({intent.need for intent in declared_intents})
    if not unique_needs:
        unique_needs = [ontology.NeedId("baseline_need")]

    needs = [
        ontology.NeedSpec(
            id=need_id,
            label=f"Need {need_id}",
            decay_per_tick=0.01,
            initial=0.5,
        )
        for need_id in unique_needs
    ]
    actions = [
        ontology.ActionSpec(
            id=ontology.ActionId(f"act_{need_id}"),
            label=f"Act {need_id}",
            reversibility=ontology.Reversibility.REVERSIBLE,
            duration_ticks=1,
            satisfies={need_id: 1.0},
            effects={need_id: 1.0},
        )
        for need_id in unique_needs
    ]
    agents = [
        ontology.AgentSpec(
            id=ontology.AgentId("agent_01"),
            label="Agent 01",
            initial_needs={n.id: n.initial for n in needs},
        )
    ]
    tuning = ontology.TuningSpec(
        intent_confidence_threshold=threshold,
        candidate_pool=max(len(actions), 1),
    )
    return pack.ContentPack(
        id="pack_intent_test",
        label="Pack Intent Test",
        needs=needs,
        actions=actions,
        agents=agents,
        intents=declared_intents,
        tuning=tuning,
    )


# ---------------------------------------------------------------------------
# Casos obligatorios de clasificación (§6.1)
# ---------------------------------------------------------------------------


def test_classify_exact_match_returns_score_one_and_intent_id() -> None:
    """Una frase idéntica a un ejemplo produce similitud coseno 1.0 y se reconoce.

    Protege la condición base de fidelidad: si el usuario escribe exactamente
    lo que el catálogo describe, el clasificador debe asociarlo a la intención.
    """
    intent = _make_intent("intent_alpha", "need_alpha", ("solicitar asistencia alfa",))
    p = _make_pack_for_intents([intent], threshold=0.35)
    classifier = build_classifier(p)
    assert classifier is not None

    # Similitud coseno directa es 1.0
    v1 = _vectorize("solicitar asistencia alfa")
    v2 = _vectorize("solicitar asistencia alfa")
    assert math.isclose(_cosine(v1, v2), 1.0)

    result = classifier.classify("solicitar asistencia alfa")
    assert result == "intent_alpha"


def test_classify_partial_match_above_threshold_returns_intent_id() -> None:
    """Una frase con solapamiento parcial que supera el umbral es reconocida.

    Comprueba que el clasificador tolera variaciones léxicas sin requerir
    coincidencia exacta palabra por palabra, siempre que el coseno >= threshold.
    """
    intent = _make_intent(
        "intent_alpha",
        "need_alpha",
        ("solicitar asistencia urgente",),
    )
    p = _make_pack_for_intents([intent], threshold=0.35)
    classifier = build_classifier(p)
    assert classifier is not None

    # 'solicitar asistencia' comparte 2 palabras de 3 con el ejemplo (coseno < 1.0 y > 0.35)
    query = "solicitar asistencia inmediata por favor"
    score = _cosine(_vectorize(query), _vectorize("solicitar asistencia urgente"))
    assert 0.35 <= score < 1.0

    result = classifier.classify(query)
    assert result == "intent_alpha"


def test_classify_disjoint_tokens_returns_none() -> None:
    """Una frase que no comparte ningún token con el catálogo devuelve None.

    Protege la mutación M1: si el cálculo de similitud utilizara unión en vez
    de intersección, frases disyuntas producirían similitud espuria y no darían None.
    """
    intent = _make_intent("intent_alpha", "need_alpha", ("ejecutar protocolo alfa",))
    p = _make_pack_for_intents([intent], threshold=0.35)
    classifier = build_classifier(p)
    assert classifier is not None

    result = classifier.classify("palabras completamente ajenas al catalogo")
    assert result is None


def test_classify_below_threshold_returns_none_and_exact_threshold_recognized() -> None:
    """Una consulta por debajo de threshold da None; si es igual al threshold se reconoce.

    Protege la frontera estricta y la mutación M4: la condición de aceptación
    es best_score >= threshold (es decir, rechazo si best_score < threshold).
    Si se cambiara a best_score <= threshold, una coincidencia exacta al umbral
    sería descartada erróneamente.
    """
    intent = _make_intent("intent_alpha", "need_alpha", ("alfa beta",))

    # Con query 'alfa' y example 'alfa beta':
    # v_q = {'alfa': 1}, norm=1.0
    # v_ex = {'alfa': 1, 'beta': 1}, norm=sqrt(2)
    # cosine = 1 / sqrt(2)
    exact_cosine = 1.0 / math.sqrt(2)

    # 1. Con threshold exactamente igual al score: debe reconocerse (no entrar en score < threshold)
    p_exact = _make_pack_for_intents([intent], threshold=exact_cosine)
    c_exact = build_classifier(p_exact)
    assert c_exact is not None
    assert c_exact.classify("alfa") == "intent_alpha"

    # 2. Con threshold ligeramente superior: debe ser rechazada (score < threshold)
    p_above = _make_pack_for_intents([intent], threshold=exact_cosine + 0.05)
    c_above = build_classifier(p_above)
    assert c_above is not None
    assert c_above.classify("alfa") is None


def test_classify_case_insensitive() -> None:
    """La tokenización ignora mayúsculas y minúsculas uniformemente.

    Protege la mutación M2: si _tokenize omite .lower(), una consulta en
    mayúsculas fallaría al comparar contra ejemplos normalizados.
    """
    intent = _make_intent("intent_alpha", "need_alpha", ("necesito ayuda",))
    p = _make_pack_for_intents([intent], threshold=0.35)
    classifier = build_classifier(p)
    assert classifier is not None

    lower_result = classifier.classify("necesito ayuda")
    upper_result = classifier.classify("NECESITO AYUDA")
    mixed_result = classifier.classify("NeCeSiTo AyUdA")

    assert lower_result == "intent_alpha"
    assert upper_result == "intent_alpha"
    assert mixed_result == "intent_alpha"


def test_classify_tie_break_lowest_lexicographical_id_wins() -> None:
    """Ante empate exacto de similitud entre dos intenciones, gana el id menor.

    Protege la mutación M3: build_classifier ordena por IntentId y classify
    evalúa con `score > best_score`. Con `score >= best_score`, la última
    intención evaluada pisaría a la primera, alterando el determinismo.
    """
    intent_a = _make_intent("intent_alpha", "need_a", ("frase comun identica",))
    intent_b = _make_intent("intent_beta", "need_b", ("frase comun identica",))

    # Pasamos las intenciones en orden inverso para garantizar que build_classifier ordena
    p = _make_pack_for_intents([intent_b, intent_a], threshold=0.35)
    classifier = build_classifier(p)
    assert classifier is not None

    result = classifier.classify("frase comun identica")
    assert result == "intent_alpha"


def test_build_classifier_empty_intents_returns_none() -> None:
    """Un pack sin intenciones declaradas produce clasificador None.

    Protege la mutación M5: sin catálogo cerrado no hay referencias contra
    las que comparar, y el bucle Loop opera sin clasificador activo.
    """
    p = _make_pack_for_intents([], threshold=0.35)
    assert p.intents == ()
    classifier = build_classifier(p)
    assert classifier is None


def test_build_classifier_recognizes_each_intent_with_own_first_example() -> None:
    """Cada intención del pack se reconoce a sí misma usando su primer ejemplo.

    Verifica la cobertura exhaustiva del catálogo cerrado frente a su propia
    declaración de referencia.
    """
    intent_1 = _make_intent("intent_1", "need_1", ("primer ejemplo uno", "variante uno"))
    intent_2 = _make_intent("intent_2", "need_2", ("primer ejemplo dos", "variante dos"))
    intent_3 = _make_intent("intent_3", "need_3", ("primer ejemplo tres", "variante tres"))

    p = _make_pack_for_intents([intent_1, intent_2, intent_3], threshold=0.35)
    classifier = build_classifier(p)
    assert classifier is not None

    for intent in p.intents:
        first_example = intent.examples[0]
        assert classifier.classify(first_example) == str(intent.id)


def test_classify_determinism_multiple_calls_identical_result() -> None:
    """Llamadas sucesivas con la misma entrada producen resultados idénticos.

    El clasificador es inmutable, carece de efectos secundarios y no altera
    estado interno entre invocaciones.
    """
    intent = _make_intent("intent_alpha", "need_alpha", ("consulta determinista",))
    p = _make_pack_for_intents([intent], threshold=0.35)
    classifier = build_classifier(p)
    assert classifier is not None

    r1 = classifier.classify("consulta determinista")
    r2 = classifier.classify("consulta determinista")
    assert r1 == r2 == "intent_alpha"


def test_build_classifier_satisfies_intent_classifier_protocol() -> None:
    """La instancia devuelta por build_classifier satisface IntentClassifier.

    Garantiza la inversión de dependencias: Loop interactúa exclusivamente con
    la costura runtime_checkable de ports.py sin importar aida.intent.
    """
    intent = _make_intent("intent_alpha", "need_alpha", ("ejemplo",))
    p = _make_pack_for_intents([intent], threshold=0.35)
    classifier = build_classifier(p)
    assert classifier is not None
    assert isinstance(classifier, IntentClassifier)
    assert isinstance(classifier, BagOfWordsClassifier)


def test_tuning_spec_intent_confidence_threshold_default_is_0_35() -> None:
    """I-056.1: el umbral por defecto ancla en 0.35, ni 0.0 ni 0.99."""
    assert ontology.TuningSpec().intent_confidence_threshold == 0.35


def test_tokenize_lowercases_and_keeps_accented_chars() -> None:
    """I-056.2: la tokenización conserva acentos y eñes, no solo ASCII."""
    assert _tokenize("Está en ÁREA ñandú") == ("está", "en", "área", "ñandú")


def test_classify_empty_or_whitespace_returns_none() -> None:
    """Entradas vacías o con espacios en blanco devuelven None sin excepción.

    Un texto sin tokens alfanuméricos produce vector vacío y no debe
    provocar errores por división por cero ni excepciones de tipo.
    """
    intent = _make_intent("intent_alpha", "need_alpha", ("ejemplo valido",))
    p = _make_pack_for_intents([intent], threshold=0.35)
    classifier = build_classifier(p)
    assert classifier is not None

    assert classifier.classify("") is None
    assert classifier.classify("   ") is None
    assert classifier.classify("\t\n  \r\n") is None
