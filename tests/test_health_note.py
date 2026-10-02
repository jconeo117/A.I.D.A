"""Pruebas para la nota clínica redactada (Plan 72 — §B.1).

Casos 1 al 11:
1. Prosa cuando el proveedor contesta: redacted=True y texto devuelto.
2. El orden es el del catálogo, no el del modelo.
3. Una clave no declarada invalida la respuesta entera y cae al respaldo.
4. in_note: false no entra (otros_comentarios no aparece ni en prosa ni en respaldo).
5. Un apartado sin hechos no aparece y no se pide al modelo.
6. Sin proveedor (provider=None) -> respaldo literal con frases tal como se dijeron.
7. Proveedor que levanta -> respaldo literal y no se cae la consulta.
8. Proveedor que devuelve None o tipo inválido -> respaldo literal.
9. El briefing solo lleva los hechos de las secciones in_note
   (sin historia previa, códigos ni otros_comentarios).
10. La instrucción sale del pack (briefing empieza por settings.system_prompt).
11. load_note_settings: claves obligatorias, ValueError que las nombra, y el pack real carga.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aida.arms.health.note import (
    ClinicalNote,
    NoteSettings,
    build_note,
    load_note_settings,
)
from aida.arms.health.sections import SectionSpec


class _SpyProvider:
    def __init__(self, response: object | None = None, raise_exc: Exception | None = None) -> None:
        self.response = response
        self.raise_exc = raise_exc
        self.captured_briefing: str | None = None
        self.captured_schema: type | None = None

    def think(self, briefing: str, schema: type) -> object | None:
        self.captured_briefing = briefing
        self.captured_schema = schema
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


@pytest.fixture
def sample_note_settings() -> NoteSettings:
    return NoteSettings(
        system_prompt="Redacta la nota clínica en español.",
        provider="openai",
        provider_model="gpt-4.1-mini",
        field_heading="### {}",
        provenance_template="_Redactado por {} ({})_",
        fallback_notice="(sin redacción: lo que sigue son las frases tal como se dijeron)",
    )


def test_build_note_prosa_cuando_proveedor_contesta(sample_note_settings: NoteSettings) -> None:
    """Caso 1 (§B.1): Cuando el proveedor contesta un ClinicalNote válido,
    la nota resultante tiene redacted=True, el texto en el campo y el provider."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    grouped = ((spec_motivo, ("dolor agudo de cabeza", "desde hace dos días")),)

    provider = _SpyProvider(
        response=ClinicalNote(
            fields={"motivo_consulta": "El paciente consulta por cefalea de 2 días."}
        )
    )

    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )

    assert rendered.redacted is True
    assert rendered.provider == "openai/gpt-4.1-mini"
    assert len(rendered.fields) == 1
    assert rendered.fields[0] == (spec_motivo, "El paciente consulta por cefalea de 2 días.")


def test_build_note_orden_es_el_del_catalogo_no_el_del_modelo(
    sample_note_settings: NoteSettings,
) -> None:
    """Caso 2 (§B.1): El orden de los campos es el del catálogo (grouped_sections),
    no el devuelto en el dict del modelo."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    spec_enfermedad = SectionSpec(
        "enfermedad_actual", "Enfermedad actual", (), codable=True, in_note=True
    )
    spec_plan = SectionSpec("plan", "Plan", (), codable=False, in_note=True)

    grouped = (
        (spec_motivo, ("motivo dicho",)),
        (spec_enfermedad, ("enfermedad dicha",)),
        (spec_plan, ("plan dicho",)),
    )

    # El modelo devuelve en orden inverso
    provider = _SpyProvider(
        response=ClinicalNote(
            fields={
                "plan": "Plan redactado",
                "enfermedad_actual": "Enfermedad redactada",
                "motivo_consulta": "Motivo redactado",
            }
        )
    )

    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )

    assert rendered.redacted is True
    assert tuple(spec.section_id for spec, _ in rendered.fields) == (
        "motivo_consulta",
        "enfermedad_actual",
        "plan",
    )


def test_build_note_clave_no_declarada_invalida_respuesta_entera(
    sample_note_settings: NoteSettings,
) -> None:
    """Caso 3 (§B.1): Una clave que el pack no declara invalida la respuesta entera
    y cae al respaldo literal con redacted=False. No se queda con los campos buenos."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    spec_plan = SectionSpec("plan", "Plan", (), codable=False, in_note=True)

    grouped = (
        (spec_motivo, ("dolor de cabeza",)),
        (spec_plan, ("ibuprofeno",)),
    )

    # El modelo incluye un campo inventado que no está en las secciones declaradas
    provider = _SpyProvider(
        response=ClinicalNote(
            fields={
                "motivo_consulta": "Motivo redactado",
                "campo_inventado": "Alucinación o campo fuera de catálogo",
            }
        )
    )

    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )

    assert rendered.redacted is False
    assert rendered.provider is None
    # Cae al respaldo literal con las frases crudas
    assert rendered.fields == (
        (spec_motivo, "dolor de cabeza"),
        (spec_plan, "ibuprofeno"),
    )


def test_build_note_in_note_false_no_entra(sample_note_settings: NoteSettings) -> None:
    """Caso 4 (§B.1): in_note: false (como otros_comentarios) no entra ni en la prosa
    ni en el respaldo, aunque tenga hechos."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    spec_otros = SectionSpec(
        "otros_comentarios", "Otros comentarios", (), codable=False, in_note=False
    )

    grouped = (
        (spec_motivo, ("consulta por tos",)),
        (spec_otros, ("comentario suelto no clasificable",)),
    )

    # Proveedor responde prosa
    provider = _SpyProvider(
        response=ClinicalNote(fields={"motivo_consulta": "Consulta por tos seca."})
    )
    rendered_prosa = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )
    assert rendered_prosa.redacted is True
    assert all(spec.section_id != "otros_comentarios" for spec, _ in rendered_prosa.fields)
    assert len(rendered_prosa.fields) == 1
    assert rendered_prosa.fields[0][0].section_id == "motivo_consulta"

    # Respaldo sin proveedor
    rendered_respaldo = build_note(
        grouped_sections=grouped, settings=sample_note_settings, providers=()
    )
    assert rendered_respaldo.redacted is False
    assert all(spec.section_id != "otros_comentarios" for spec, _ in rendered_respaldo.fields)
    assert len(rendered_respaldo.fields) == 1
    assert rendered_respaldo.fields[0][0].section_id == "motivo_consulta"


def test_build_note_apartado_sin_hechos_no_aparece_ni_se_pide(
    sample_note_settings: NoteSettings,
) -> None:
    """Caso 5 (§B.1): Un apartado sin hechos no aparece en la nota y no se pide al modelo."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    spec_examen = SectionSpec("examen_fisico", "Examen físico", (), codable=False, in_note=True)

    grouped = (
        (spec_motivo, ("dolor lumbar",)),
        (spec_examen, ()),  # Vacío, sin hechos
    )

    provider = _SpyProvider(
        response=ClinicalNote(fields={"motivo_consulta": "Consulta por lumbalgia."})
    )
    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )

    assert rendered.redacted is True
    assert len(rendered.fields) == 1
    assert rendered.fields[0][0].section_id == "motivo_consulta"
    # Comprobar que en el briefing no se incluyó examen_fisico
    assert provider.captured_briefing is not None
    assert "examen_fisico" not in provider.captured_briefing


def test_build_note_sin_proveedor_da_respaldo_literal(sample_note_settings: NoteSettings) -> None:
    """Caso 6 (§B.1): Sin proveedores (providers=()) produce respaldo literal con
    redacted=False, provider=None y frases tal como se dijeron unidas por salto de línea."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    grouped = ((spec_motivo, ("dolor agudo", "cefalea constante")),)

    rendered = build_note(grouped_sections=grouped, settings=sample_note_settings, providers=())

    assert rendered.redacted is False
    assert rendered.provider is None
    assert rendered.fields == ((spec_motivo, "dolor agudo\ncefalea constante"),)


def test_build_note_proveedor_levanta_da_respaldo(sample_note_settings: NoteSettings) -> None:
    """Caso 7 (§B.1): Proveedor que levanta excepción cae al respaldo literal y la
    consulta no se cae."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    grouped = ((spec_motivo, ("dolor agudo",)),)

    provider = _SpyProvider(raise_exc=RuntimeError("Servicio de LLM caído"))
    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )

    assert rendered.redacted is False
    assert rendered.provider is None
    assert rendered.fields == ((spec_motivo, "dolor agudo"),)


def test_build_note_proveedor_devuelve_none_o_invalido_da_respaldo(
    sample_note_settings: NoteSettings,
) -> None:
    """Caso 8 (§B.1): Proveedor que devuelve None o un objeto que no valida como ClinicalNote
    cae al respaldo literal."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    grouped = ((spec_motivo, ("dolor agudo",)),)

    # Devuelve None
    p_none = _SpyProvider(response=None)
    r1 = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", p_none),),
    )
    assert r1.redacted is False
    assert r1.fields == ((spec_motivo, "dolor agudo"),)

    # Devuelve objeto no ClinicalNote
    p_invalido = _SpyProvider(response="texto no estructurado")
    r2 = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", p_invalido),),
    )
    assert r2.redacted is False
    assert r2.fields == ((spec_motivo, "dolor agudo"),)


def test_build_note_briefing_solo_lleva_hechos_de_secciones_in_note(
    sample_note_settings: NoteSettings,
) -> None:
    """Caso 9 (§B.1): El briefing capturado lleva SOLO hechos de secciones in_note;
    no contiene historia previa, ni códigos, ni frases de otros_comentarios."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    spec_otros = SectionSpec(
        "otros_comentarios", "Otros comentarios", (), codable=False, in_note=False
    )

    grouped = (
        (spec_motivo, ("hecho clínico motivo",)),
        (spec_otros, ("frase clasificada en otros",)),
    )

    provider = _SpyProvider(response=ClinicalNote(fields={"motivo_consulta": "Motivo redactado."}))
    build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )

    assert provider.captured_briefing is not None
    briefing = provider.captured_briefing

    assert "hecho clínico motivo" in briefing
    assert "frase clasificada en otros" not in briefing
    assert "otros_comentarios" not in briefing
    # No contiene referencias a códigos ni historia previa
    assert "CIE-10" not in briefing
    assert "DiagnosticCode" not in briefing
    assert "historia previa" not in briefing.lower()


def test_build_note_instruccion_sale_del_pack(sample_note_settings: NoteSettings) -> None:
    """Caso 10 (§B.1): La instrucción inicial del briefing proviene exactamente de
    settings.system_prompt (definido en note.yaml)."""
    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    grouped = ((spec_motivo, ("hecho 1",)),)

    provider = _SpyProvider(response=ClinicalNote(fields={"motivo_consulta": "Redacción"}))
    build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )

    assert provider.captured_briefing is not None
    assert provider.captured_briefing.startswith(sample_note_settings.system_prompt)


def test_load_note_settings_valida_claves_y_carga_pack_real(tmp_path: Path) -> None:
    """Caso 11 (§B.1): load_note_settings valida claves obligatorias con ValueError,
    rechaza no-mapeos, y carga con éxito el pack real."""
    # Pack real
    real_settings = load_note_settings()
    assert real_settings.provider in ("openai", "gemini", "claude", "ollama")
    assert real_settings.provider_model != ""
    assert len(real_settings.system_prompt) > 20
    assert "{}" in real_settings.field_heading
    assert "{}" in real_settings.provenance_template
    assert real_settings.fallback_notice != ""

    # No es mapeo
    p_not_dict = tmp_path / "not_dict.yaml"
    p_not_dict.write_text("- una lista\n- no un dict\n")
    with pytest.raises(ValueError, match="debe ser un mapeo YAML"):
        load_note_settings(p_not_dict)

    # Clave faltante
    raw = {
        "system_prompt": "prompt",
        "provider": "openai",
        "provider_model": "gpt-4.1-mini",
        "field_heading": "### {}",
        "provenance_template": "{}",
        # falta fallback_notice
    }
    p_incompleto = tmp_path / "incompleto.yaml"
    p_incompleto.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=r"faltan claves.*fallback_notice"):
        load_note_settings(p_incompleto)


def test_clinical_note_extra_forbid() -> None:
    """M8: ClinicalNote rechaza campos extra no permitidos con ValidationError."""
    with pytest.raises(ValidationError):
        ClinicalNote.model_validate({"fields": {}, "campo_extra": "no_permitido"})


def test_note_prohibicion_preguntas_en_system_prompt_y_briefing() -> None:
    """Caso 13 (§B.3, M9): La prohibición de preguntas está en el system_prompt
    del pack real y viaja en el briefing entregado al proveedor de lenguaje."""
    settings = load_note_settings()
    assert (
        "Prohibido redactar preguntas, interrogatorios o lo que el profesional pregunta"
        in settings.system_prompt
    )
    assert "Si un apartado solo contiene preguntas, déjalo vacío." in settings.system_prompt

    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    grouped = ((spec_motivo, ("dolor de cabeza",)),)
    provider = _SpyProvider(response=ClinicalNote(fields={"motivo_consulta": "Cefalea."}))
    build_note(
        grouped_sections=grouped,
        settings=settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )

    assert provider.captured_briefing is not None
    assert (
        "Prohibido redactar preguntas, interrogatorios o lo que el profesional pregunta"
        in provider.captured_briefing
    )


def test_build_note_procedencia_es_quien_respondio(sample_note_settings: NoteSettings) -> None:
    """Caso 10 (§B.3, M6): La procedencia es quien respondió.
    Dos candidatos: el primero devuelve None, el segundo redacta -> la nota dice el segundo."""
    spec_motivo = SectionSpec("motivo_consulta", "Motivo", (), codable=True, in_note=True)
    grouped = ((spec_motivo, ("dolor agudo",)),)

    candidato_1 = _SpyProvider(response=None)
    candidato_2 = _SpyProvider(
        response=ClinicalNote(fields={"motivo_consulta": "Cefalea aguda."})
    )

    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(
            ("openai/gpt-4.1-mini", candidato_1),
            ("ollama/llama3.2:3b", candidato_2),
        ),
    )
    assert rendered.redacted is True
    assert rendered.provider == "ollama/llama3.2:3b"
    assert rendered.fields[0][1] == "Cefalea aguda."


def test_build_note_primer_candidato_responde_segundo_no_se_toca(
    sample_note_settings: NoteSettings,
) -> None:
    """Caso 11 (§B.3, M7): Si responde el primer candidato, el segundo no se toca."""
    spec_motivo = SectionSpec("motivo_consulta", "Motivo", (), codable=True, in_note=True)
    grouped = ((spec_motivo, ("dolor agudo",)),)

    candidato_1 = _SpyProvider(
        response=ClinicalNote(fields={"motivo_consulta": "Cefalea por candidato 1."})
    )
    candidato_2 = _SpyProvider(
        response=ClinicalNote(fields={"motivo_consulta": "Cefalea por candidato 2."})
    )

    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(
            ("openai/gpt-4.1-mini", candidato_1),
            ("ollama/llama3.2:3b", candidato_2),
        ),
    )
    assert rendered.redacted is True
    assert rendered.provider == "openai/gpt-4.1-mini"
    assert rendered.fields[0][1] == "Cefalea por candidato 1."
    assert candidato_2.captured_briefing is None


def test_build_note_sin_candidatos_o_todos_fallan_respaldo_literal_sin_proveedor(
    sample_note_settings: NoteSettings,
) -> None:
    """Caso 12 (§B.3, M8): Si ninguno responde, respaldo literal con
    provider=None y redacted=False."""
    spec_motivo = SectionSpec("motivo_consulta", "Motivo", (), codable=True, in_note=True)
    grouped = ((spec_motivo, ("dolor agudo",)),)

    candidato_falla_1 = _SpyProvider(response=None)
    candidato_falla_2 = _SpyProvider(raise_exc=RuntimeError("Caída total"))

    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(
            ("openai/gpt-4.1-mini", candidato_falla_1),
            ("ollama/llama3.2:3b", candidato_falla_2),
        ),
    )
    assert rendered.redacted is False
    assert rendered.provider is None
    assert rendered.fields == ((spec_motivo, "dolor agudo"),)


def test_build_note_nombre_viaja_con_version_de_modelo(sample_note_settings: NoteSettings) -> None:
    """Caso 13 (§B.3): El nombre que viaja es el que se pasó con versión de modelo incluida."""
    spec_motivo = SectionSpec("motivo_consulta", "Motivo", (), codable=True, in_note=True)
    grouped = ((spec_motivo, ("dolor",)),)

    provider = _SpyProvider(response=ClinicalNote(fields={"motivo_consulta": "Dolor."}))
    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(("anthropic/claude-3-5-sonnet-20241022", provider),),
    )
    assert rendered.provider == "anthropic/claude-3-5-sonnet-20241022"


def test_build_note_aviso_caida_nombra_ambos_proveedores(
    sample_note_settings: NoteSettings,
) -> None:
    """Caso 14 (§B.3): El aviso de caída on_fallback nombra al que falló y al que respondió."""
    spec_motivo = SectionSpec("motivo_consulta", "Motivo", (), codable=True, in_note=True)
    grouped = ((spec_motivo, ("dolor",)),)

    candidato_1 = _SpyProvider(response=None)
    candidato_2 = _SpyProvider(response=ClinicalNote(fields={"motivo_consulta": "Dolor."}))

    eventos_fallback: list[tuple[str, str]] = []

    def spy_fallback(fallido: str, respondedor: str) -> None:
        eventos_fallback.append((fallido, respondedor))

    rendered = build_note(
        grouped_sections=grouped,
        settings=sample_note_settings,
        providers=(
            ("openai/gpt-4.1-mini", candidato_1),
            ("ollama/llama3.2:3b", candidato_2),
        ),
        on_fallback=spy_fallback,
    )
    assert rendered.redacted is True
    assert len(eventos_fallback) == 1
    assert eventos_fallback[0] == ("openai/gpt-4.1-mini", "ollama/llama3.2:3b")


def test_note_prohibicion_reescribir_nombres_en_system_prompt_y_briefing() -> None:
    """Caso 19 (§B.5, M12): La instrucción de no reescribir nombres está en el
    system_prompt del pack y viaja en el briefing."""
    settings = load_note_settings()
    assert (
        "Prohibido corregir, completar o inventar nombres de fármacos y de enfermedades"
        in settings.system_prompt
    )
    assert "cópialos exactamente como aparecen en los hechos" in settings.system_prompt

    spec_motivo = SectionSpec(
        "motivo_consulta", "Motivo de consulta", (), codable=True, in_note=True
    )
    grouped = ((spec_motivo, ("toma lupusveno",)),)
    provider = _SpyProvider(response=ClinicalNote(fields={"motivo_consulta": "Lupusveno."}))
    build_note(
        grouped_sections=grouped,
        settings=settings,
        providers=(("openai/gpt-4.1-mini", provider),),
    )

    assert provider.captured_briefing is not None
    assert (
        "Prohibido corregir, completar o inventar nombres de fármacos y de enfermedades"
        in provider.captured_briefing
    )

