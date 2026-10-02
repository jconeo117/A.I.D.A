"""Pruebas para la segmentación clínica (arms/health/segment.py).

Cubre §6.2 de 44-plan-segmentacion-clinica.md:
- Línea claramente de una sección -> esa sección.
- Línea que no toca ningún gatillo -> fallback, y confirma que el vector vacío
  de otros_comentarios nunca gana por coseno (score 0.0 < threshold).
- Empate exacto de coseno entre dos secciones -> gana el section_id menor alfabéticamente.
- score == threshold exacto -> se reconoce (condición >=).
- group_by_section conserva el orden de declaración del catálogo, no el de aparición en
  el transcript, y omite secciones sin ninguna línea.
- Línea vacía o solo espacios -> no genera ningún SegmentedLine.
- ClinicalSegmenter y build_health_segmenter funcionan de punta a punta.
"""

from __future__ import annotations

from pathlib import Path

from aida.arms.health.sections import SectionSpec, load_sections
from aida.arms.health.segment import (
    ClinicalSegmenter,
    SegmentedLine,
    build_health_segmenter,
    build_section_index,
    group_by_section,
    segment_lines,
)
from aida.textmatch import cosine, vectorize


def test_segment_linea_claramente_de_una_seccion() -> None:
    """Una línea con vocabulario que coincide con gatillos de una sección
    se clasifica en esa sección.
    """
    catalog = load_sections()
    segmenter = ClinicalSegmenter(catalog)

    # "consulta hoy por" es gatillo de motivo_consulta
    resultado = segmenter.segment("consulta hoy por dolor de cabeza intenso")
    secciones = [spec.section_id for spec, _ in resultado]
    assert "motivo_consulta" in secciones

    # "se receta" es gatillo de medicamentos
    resultado_med = segmenter.segment("se receta amoxicilina 500mg cada 8 horas")
    secciones_med = [spec.section_id for spec, _ in resultado_med]
    assert "medicamentos" in secciones_med

    # "se solicita" es gatillo de ordenes
    resultado_ord = segmenter.segment("se solicita resonancia magnética")
    secciones_ord = [spec.section_id for spec, _ in resultado_ord]
    assert "ordenes" in secciones_ord


def test_segment_linea_sin_gatillo_cae_en_fallback_y_vector_vacio_score_cero() -> None:
    """Una línea que no toca ningún gatillo cae en fallback ('otros_comentarios').
    Se confirma que el vector vacío de 'otros_comentarios' nunca gana por coseno
    porque su score contra cualquier texto es 0.0 (< threshold).
    """
    catalog = load_sections()
    index = build_section_index(catalog.sections)

    # 1. Comprobación matemática explícita del vector vacío de otros_comentarios
    spec_fallback = catalog.fallback
    assert spec_fallback.section_id == "otros_comentarios"
    assert spec_fallback.trigger_phrases == ()

    vec_fallback = vectorize(" ".join(spec_fallback.trigger_phrases))
    assert vec_fallback == {}

    linea_irrelevante = "computadora teclado pantalla ratón"
    vec_linea = vectorize(linea_irrelevante)
    assert cosine(vec_linea, vec_fallback) == 0.0
    assert catalog.threshold > 0.0

    # 2. segment_lines clasifica la línea irrelevante en fallback
    segmented = segment_lines(
        (linea_irrelevante,),
        index,
        threshold=catalog.threshold,
        fallback=catalog.fallback,
    )
    assert len(segmented) == 1
    assert segmented[0].section.section_id == "otros_comentarios"
    assert segmented[0].text == linea_irrelevante


def test_segment_empate_exacto_coseno_gana_section_id_menor_alfabeticamente() -> None:
    """En caso de empate exacto de coseno entre dos secciones candidatas,
    el criterio de desempate (-score, spec.section_id) selecciona la sección
    con menor section_id alfabéticamente.
    """
    # Caso 1: 'alfa' vs 'beta', misma frase gatillo -> 'alfa' < 'beta', gana 'alfa'
    spec_alfa = SectionSpec(
        section_id="alfa",
        label="Alfa",
        trigger_phrases=("palabra compartida",),
        codable=True,
        in_note=True,
    )
    spec_beta = SectionSpec(
        section_id="beta",
        label="Beta",
        trigger_phrases=("palabra compartida",),
        codable=True,
        in_note=True,
    )
    spec_fallback = SectionSpec(
        section_id="fallback",
        label="Fallback",
        trigger_phrases=(),
        codable=False,
        in_note=False,
    )

    index_1 = build_section_index((spec_beta, spec_alfa, spec_fallback))
    linea = "palabra compartida por ambas"

    res_1 = segment_lines(
        (linea,),
        index_1,
        threshold=0.1,
        fallback=spec_fallback,
    )
    assert len(res_1) == 1
    assert res_1[0].section.section_id == "alfa"

    # Caso 2: 'zeta' vs 'beta' -> 'beta' < 'zeta', gana 'beta' sin importar orden del catálogo
    spec_zeta = SectionSpec(
        section_id="zeta",
        label="Zeta",
        trigger_phrases=("palabra compartida",),
        codable=True,
        in_note=True,
    )
    index_2 = build_section_index((spec_zeta, spec_beta, spec_fallback))

    res_2 = segment_lines(
        (linea,),
        index_2,
        threshold=0.1,
        fallback=spec_fallback,
    )
    assert len(res_2) == 1
    assert res_2[0].section.section_id == "beta"


def test_segment_score_exactamente_igual_a_threshold_se_reconoce() -> None:
    """Una coincidencia con score == threshold exacto debe reconocerse (condición >=).
    Si fuera solo '>', caería a fallback.
    """
    # Consulta: 1 palabra ("cefalea"). Vector: {"cefalea": 1}, norma = 1.0
    # Gatillo: 4 palabras ("cefalea constante intensa matutina").
    # Vector: {"cefalea": 1, "constante": 1, "intensa": 1, "matutina": 1}, norma = sqrt(4) = 2.0
    # Coseno: 1 / (1.0 * 2.0) = 0.5000 exacto.
    spec_sec = SectionSpec(
        section_id="sec_exacta",
        label="Sección Exacta",
        trigger_phrases=("cefalea constante intensa matutina",),
        codable=True,
        in_note=True,
    )
    spec_fallback = SectionSpec(
        section_id="fallback",
        label="Fallback",
        trigger_phrases=(),
        codable=False,
        in_note=False,
    )
    index = build_section_index((spec_sec, spec_fallback))

    # Con threshold exacto de 0.50: se debe reconocer sec_exacta
    res_exacto = segment_lines(
        ("cefalea",),
        index,
        threshold=0.50,
        fallback=spec_fallback,
    )
    assert len(res_exacto) == 1
    assert res_exacto[0].section.section_id == "sec_exacta"

    # Con threshold ligeramente superior (0.5001): cae a fallback
    res_superado = segment_lines(
        ("cefalea",),
        index,
        threshold=0.5001,
        fallback=spec_fallback,
    )
    assert len(res_superado) == 1
    assert res_superado[0].section.section_id == "fallback"


def test_group_by_section_conserva_orden_catalogo_y_omite_secciones_vacias() -> None:
    """group_by_section:
    1. Agrupa las líneas respetando el orden de declaración en el catálogo,
       NO el orden en que las líneas aparecieron en el transcript.
    2. Omite por completo las secciones del catálogo que no tienen líneas asignadas.
    """
    sec_1 = SectionSpec(
        section_id="primera",
        label="Primera",
        trigger_phrases=("uno",),
        codable=True,
        in_note=True,
    )
    sec_2 = SectionSpec(
        section_id="segunda",
        label="Segunda",
        trigger_phrases=("dos",),
        codable=True,
        in_note=True,
    )
    sec_3 = SectionSpec(
        section_id="tercera",
        label="Tercera",
        trigger_phrases=("tres",),
        codable=True,
        in_note=True,
    )
    sec_4 = SectionSpec(
        section_id="cuarta",
        label="Cuarta",
        trigger_phrases=("cuatro",),
        codable=True,
        in_note=True,
    )

    catalogo_orden = (sec_1, sec_2, sec_3, sec_4)

    # Las líneas aparecen en orden inverso: primero línea de sec_2, luego línea de sec_1
    segmented = (
        SegmentedLine(section=sec_2, text="texto de la segunda sección A"),
        SegmentedLine(section=sec_1, text="texto de la primera sección"),
        SegmentedLine(section=sec_2, text="texto de la segunda sección B"),
    )

    grouped = group_by_section(segmented, catalogo_orden)

    # Solo deben estar primera y segunda (tercera y cuarta omitidas)
    assert len(grouped) == 2

    # El orden debe ser el del catálogo: primera antes que segunda
    assert grouped[0][0].section_id == "primera"
    assert grouped[0][1] == ("texto de la primera sección",)

    assert grouped[1][0].section_id == "segunda"
    assert grouped[1][1] == (
        "texto de la segunda sección A",
        "texto de la segunda sección B",
    )


def test_segment_linea_vacia_o_solo_espacios_se_omite() -> None:
    """Líneas vacías o compuestas exclusivamente por espacios en blanco
    no generan ningún SegmentedLine.
    """
    catalog = load_sections()
    index = build_section_index(catalog.sections)

    # Tupla con solo líneas en blanco
    lineas_blancas = ("", "   ", "\t\t", " \n ", "    ")
    res_vacio = segment_lines(
        lineas_blancas,
        index,
        threshold=catalog.threshold,
        fallback=catalog.fallback,
    )
    assert res_vacio == ()

    # Mezcladas con líneas reales
    lineas_mixtas = (
        "   ",
        "consulta hoy por fiebre",
        "",
        "se receta paracetamol",
        "  \t  ",
    )
    res_mixto = segment_lines(
        lineas_mixtas,
        index,
        threshold=catalog.threshold,
        fallback=catalog.fallback,
    )
    assert len(res_mixto) == 2
    assert res_mixto[0].text == "consulta hoy por fiebre"
    assert res_mixto[0].section.section_id == "motivo_consulta"
    assert res_mixto[1].text == "se receta paracetamol"
    assert res_mixto[1].section.section_id == "medicamentos"


def test_clinical_segmenter_de_punta_a_punta() -> None:
    """ClinicalSegmenter.segment procesa un transcript multilínea completo."""
    catalog = load_sections()
    segmenter = ClinicalSegmenter(catalog)

    transcript = (
        "consulta hoy por dolor de cabeza\n"
        "\n"
        "se receta ibuprofeno\n"
        "   \n"
        "un comentario totalmente libre que no coincide con nada\n"
        "se ordena ecografía abdominal"
    )

    resultado = segmenter.segment(transcript)
    secciones_presentes = [spec.section_id for spec, _ in resultado]

    # Orden del catálogo para estas secciones:
    # motivo_consulta -> medicamentos -> ordenes -> otros_comentarios
    assert secciones_presentes == [
        "motivo_consulta",
        "medicamentos",
        "ordenes",
        "otros_comentarios",
    ]

    dict_resultado = {spec.section_id: texts for spec, texts in resultado}
    assert dict_resultado["motivo_consulta"] == ("consulta hoy por dolor de cabeza",)
    assert dict_resultado["medicamentos"] == ("se receta ibuprofeno",)
    assert dict_resultado["ordenes"] == ("se ordena ecografía abdominal",)
    assert dict_resultado["otros_comentarios"] == (
        "un comentario totalmente libre que no coincide con nada",
    )


def test_build_health_segmenter_por_defecto_y_con_ruta(tmp_path: Path) -> None:
    """build_health_segmenter() sin argumentos usa el catálogo por defecto;
    con ruta carga el catálogo especificado.
    """
    seg_defecto = build_health_segmenter()
    assert isinstance(seg_defecto, ClinicalSegmenter)

    # Con archivo propio
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        "threshold: 0.2\n"
        "fallback: s1\n"
        "skip_interrogative: true\n"
        "sections:\n"
        "  - id: s1\n"
        "    label: S1\n"
        "    trigger_phrases: ['hola']\n"
        "    codable: true\n"
        "    in_note: true\n",
        encoding="utf-8",
    )
    seg_custom = build_health_segmenter(custom_yaml)
    assert isinstance(seg_custom, ClinicalSegmenter)
    res = seg_custom.segment("hola doctor")
    assert len(res) == 1
    assert res[0][0].section_id == "s1"


def test_segment_lines_interrogativa_va_a_reserva() -> None:
    """Caso 15 (§B.4, M9): Una frase que termina en '?' va a la reserva
    (fallback), aunque su similitud con una sección sea alta."""
    spec_motivo = SectionSpec(
        section_id="motivo_consulta",
        label="Motivo de consulta",
        trigger_phrases=("motivo de consulta", "que lo trae por aqui"),
        codable=True,
        in_note=True,
    )
    spec_fallback = SectionSpec(
        section_id="otros_comentarios",
        label="Otros comentarios",
        trigger_phrases=(),
        codable=False,
        in_note=False,
    )
    index = build_section_index((spec_motivo, spec_fallback))

    lineas = ("¿Cuénteme qué lo trae por aquí?", "¿Cuál es el motivo de consulta?")
    resultado = segment_lines(
        lineas,
        index,
        threshold=0.20,
        fallback=spec_fallback,
        skip_interrogative=True,
    )
    assert len(resultado) == 2
    assert resultado[0].section.section_id == "otros_comentarios"
    assert resultado[1].section.section_id == "otros_comentarios"


def test_segment_lines_con_skip_interrogative_false_clasifica_normal() -> None:
    """Caso 16 (§B.4, M11): Con skip_interrogative: false, una frase que
    termina en '?' se clasifica por coseno normalmente hacia su sección."""
    spec_motivo = SectionSpec(
        section_id="motivo_consulta",
        label="Motivo de consulta",
        trigger_phrases=("motivo de consulta", "que lo trae por aqui"),
        codable=True,
        in_note=True,
    )
    spec_fallback = SectionSpec(
        section_id="otros_comentarios",
        label="Otros comentarios",
        trigger_phrases=(),
        codable=False,
        in_note=False,
    )
    index = build_section_index((spec_motivo, spec_fallback))

    lineas = ("¿Cuál es el motivo de consulta?",)
    resultado = segment_lines(
        lineas,
        index,
        threshold=0.20,
        fallback=spec_fallback,
        skip_interrogative=False,
    )
    assert len(resultado) == 1
    assert resultado[0].section.section_id == "motivo_consulta"


def test_segment_lines_signo_en_medio_no_cuenta_como_interrogativa() -> None:
    """Caso 17 (§B.4, M10): Un signo '?' en medio de la frase no se desvía a la
    reserva; solo se envía si la frase termina en '?'."""
    spec_sintomas = SectionSpec(
        section_id="sintomas",
        label="Síntomas",
        trigger_phrases=("dolor de cabeza intenso", "mareos frecuentes"),
        codable=True,
        in_note=True,
    )
    spec_fallback = SectionSpec(
        section_id="otros_comentarios",
        label="Otros comentarios",
        trigger_phrases=(),
        codable=False,
        in_note=False,
    )
    index = build_section_index((spec_sintomas, spec_fallback))

    # Frase que tiene '?' en medio pero no termina en '?'
    frase_intermedia = "Dijo ¿por qué? y luego continuó con dolor de cabeza intenso."
    resultado = segment_lines(
        (frase_intermedia,),
        index,
        threshold=0.20,
        fallback=spec_fallback,
        skip_interrogative=True,
    )
    assert len(resultado) == 1
    assert resultado[0].section.section_id == "sintomas"
