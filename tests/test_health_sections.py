"""Pruebas para el catálogo de secciones clínicas (arms/health/sections.py).

Cubre §6.1 de 44-plan-segmentacion-clinica.md:
- Carga de YAML válido -> SectionCatalog con orden conservado.
- fallback que no está entre sections -> ValueError.
- id repetido -> ValueError.
- Falta threshold/sections/trigger_phrases mal tipado -> ValueError en cada caso.
- Archivo que no es mapeo dict -> ValueError.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aida.arms.health.sections import SectionCatalog, load_sections


def test_load_sections_por_defecto_conserva_orden() -> None:
    """Carga del YAML por defecto (packs/health/consultorio/sections.yaml).
    Verifica que carga un SectionCatalog válido y conserva el orden exacto de declaración.
    """
    catalog = load_sections()
    assert isinstance(catalog, SectionCatalog)
    assert catalog.threshold == 0.15
    assert catalog.fallback.section_id == "otros_comentarios"

    expected_ids = (
        "motivo_consulta",
        "enfermedad_actual",
        "antecedentes",
        "sintomas",
        "signos",
        "examen_fisico",
        "resultados",
        "evaluacion",
        "diagnosticos",
        "procedimientos",
        "medicamentos",
        "ordenes",
        "plan",
        "recomendaciones",
        "otros_comentarios",
    )
    actual_ids = tuple(s.section_id for s in catalog.sections)
    assert actual_ids == expected_ids
    assert len(catalog.sections) == 15
    assert catalog.fallback == catalog.sections[-1]


def test_load_sections_yaml_valido_personalizado(tmp_path: Path) -> None:
    """Carga de un archivo YAML personalizado con varias secciones y orden conservado."""
    data = {
        "threshold": 0.25,
        "fallback": "sec_b",
        "sections": [
            {"id": "sec_a", "label": "Sección A", "trigger_phrases": ["frase a1", "frase a2"]},
            {"id": "sec_b", "label": "Sección B", "trigger_phrases": []},
            {"id": "sec_c", "label": "Sección C", "trigger_phrases": ["frase c1"]},
        ],
    }
    custom_yaml = tmp_path / "custom_sections.yaml"
    custom_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")

    catalog = load_sections(custom_yaml)
    assert catalog.threshold == 0.25
    assert catalog.fallback.section_id == "sec_b"
    assert tuple(s.section_id for s in catalog.sections) == ("sec_a", "sec_b", "sec_c")
    assert catalog.sections[0].label == "Sección A"
    assert catalog.sections[0].trigger_phrases == ("frase a1", "frase a2")
    assert catalog.sections[1].trigger_phrases == ()


def test_load_sections_no_es_mapeo_dict_lanza_value_error(tmp_path: Path) -> None:
    """Si el archivo YAML no es un mapeo (ej. lista o escalar), lanza ValueError."""
    lista_yaml = tmp_path / "lista.yaml"
    lista_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"debe ser un mapeo YAML"):
        load_sections(lista_yaml)

    escalar_yaml = tmp_path / "escalar.yaml"
    escalar_yaml.write_text("solo una cadena de texto\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"debe ser un mapeo YAML"):
        load_sections(escalar_yaml)


def test_load_sections_falta_threshold_o_invalido_lanza_value_error(tmp_path: Path) -> None:
    """Falta 'threshold' o tiene tipo no numérico (o bool) -> lanza ValueError."""
    sin_threshold = tmp_path / "sin_thresh.yaml"
    sin_threshold.write_text(
        yaml.safe_dump({"fallback": "f", "sections": [{"id": "f", "label": "F"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"falta 'threshold' numérico"):
        load_sections(sin_threshold)

    thresh_str = tmp_path / "thresh_str.yaml"
    thresh_str.write_text(
        yaml.safe_dump({
            "threshold": "0.2",
            "fallback": "f",
            "sections": [{"id": "f", "label": "F"}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"falta 'threshold' numérico"):
        load_sections(thresh_str)

    thresh_bool = tmp_path / "thresh_bool.yaml"
    thresh_bool.write_text(
        yaml.safe_dump({
            "threshold": True,
            "fallback": "f",
            "sections": [{"id": "f", "label": "F"}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"falta 'threshold' numérico"):
        load_sections(thresh_bool)


def test_load_sections_falta_fallback_o_invalido_lanza_value_error(tmp_path: Path) -> None:
    """Falta 'fallback' o no es cadena -> lanza ValueError."""
    sin_fallback = tmp_path / "sin_fallback.yaml"
    sin_fallback.write_text(
        yaml.safe_dump({"threshold": 0.2, "sections": [{"id": "f", "label": "F"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"falta 'fallback' \(id de sección\)"):
        load_sections(sin_fallback)

    fallback_num = tmp_path / "fallback_num.yaml"
    fallback_num.write_text(
        yaml.safe_dump({
            "threshold": 0.2,
            "fallback": 123,
            "sections": [{"id": "f", "label": "F"}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"falta 'fallback' \(id de sección\)"):
        load_sections(fallback_num)


def test_load_sections_fallback_no_esta_en_sections_lanza_value_error(tmp_path: Path) -> None:
    """'fallback' no corresponde a ningún id declarado -> lanza ValueError."""
    archivo = tmp_path / "fallback_inexistente.yaml"
    archivo.write_text(
        yaml.safe_dump({
            "threshold": 0.2,
            "fallback": "inexistente",
            "sections": [
                {"id": "motivo", "label": "Motivo", "trigger_phrases": ["consulta"]},
            ],
        }),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match=r"'fallback' 'inexistente' no está entre las secciones declaradas"
    ):
        load_sections(archivo)


def test_load_sections_falta_sections_o_vacia_lanza_value_error(tmp_path: Path) -> None:
    """Falta 'sections', no es lista o es lista vacía -> lanza ValueError."""
    sin_sections = tmp_path / "sin_sections.yaml"
    sin_sections.write_text(
        yaml.safe_dump({"threshold": 0.2, "fallback": "f"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"falta 'sections' \(lista no vacía\)"):
        load_sections(sin_sections)

    sections_vacia = tmp_path / "sections_vacia.yaml"
    sections_vacia.write_text(
        yaml.safe_dump({"threshold": 0.2, "fallback": "f", "sections": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"falta 'sections' \(lista no vacía\)"):
        load_sections(sections_vacia)

    sections_no_lista = tmp_path / "sections_no_lista.yaml"
    sections_no_lista.write_text(
        yaml.safe_dump({"threshold": 0.2, "fallback": "f", "sections": "no_es_lista"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"falta 'sections' \(lista no vacía\)"):
        load_sections(sections_no_lista)


def test_load_sections_entrada_sections_no_es_dict_lanza_value_error(tmp_path: Path) -> None:
    """Cada entrada de 'sections' debe ser un mapeo (dict) -> lanza ValueError si no."""
    archivo = tmp_path / "entrada_no_dict.yaml"
    archivo.write_text(
        yaml.safe_dump({"threshold": 0.2, "fallback": "f", "sections": ["no_es_un_dict"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"cada entrada de 'sections' debe ser un mapeo"):
        load_sections(archivo)


def test_load_sections_id_o_label_mal_tipado_lanza_value_error(tmp_path: Path) -> None:
    """'id' o 'label' faltante o no string -> lanza ValueError."""
    sin_id = tmp_path / "sin_id.yaml"
    sin_id.write_text(
        yaml.safe_dump({
            "threshold": 0.2,
            "fallback": "f",
            "sections": [{"label": "Solo Label"}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'id' y 'label' deben ser cadenas"):
        load_sections(sin_id)

    id_no_str = tmp_path / "id_no_str.yaml"
    id_no_str.write_text(
        yaml.safe_dump({
            "threshold": 0.2,
            "fallback": "f",
            "sections": [{"id": 42, "label": "Label"}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'id' y 'label' deben ser cadenas"):
        load_sections(id_no_str)

    sin_label = tmp_path / "sin_label.yaml"
    sin_label.write_text(
        yaml.safe_dump({"threshold": 0.2, "fallback": "f", "sections": [{"id": "f"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'id' y 'label' deben ser cadenas"):
        load_sections(sin_label)


def test_load_sections_trigger_phrases_mal_tipado_lanza_value_error(tmp_path: Path) -> None:
    """'trigger_phrases' no es lista o contiene elementos no string -> lanza ValueError."""
    triggers_no_lista = tmp_path / "triggers_no_lista.yaml"
    triggers_no_lista.write_text(
        yaml.safe_dump({
            "threshold": 0.2,
            "fallback": "f",
            "sections": [{"id": "f", "label": "F", "trigger_phrases": "frase suelta"}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'trigger_phrases' debe ser una lista de cadenas"):
        load_sections(triggers_no_lista)

    triggers_elementos_no_str = tmp_path / "triggers_elementos_no_str.yaml"
    triggers_elementos_no_str.write_text(
        yaml.safe_dump({
            "threshold": 0.2,
            "fallback": "f",
            "sections": [{"id": "f", "label": "F", "trigger_phrases": ["buena", 123]}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'trigger_phrases' debe ser una lista de cadenas"):
        load_sections(triggers_elementos_no_str)


def test_load_sections_id_repetido_lanza_value_error(tmp_path: Path) -> None:
    """IDs de sección duplicados -> lanza ValueError."""
    archivo = tmp_path / "id_repetido.yaml"
    archivo.write_text(
        yaml.safe_dump({
            "threshold": 0.2,
            "fallback": "duplicado",
            "sections": [
                {"id": "duplicado", "label": "Primera vez"},
                {"id": "duplicado", "label": "Segunda vez"},
            ],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"id de sección repetido: 'duplicado'"):
        load_sections(archivo)
