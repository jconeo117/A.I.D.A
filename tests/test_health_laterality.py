"""Pruebas para arms/health/laterality.py (§7.2 de 46-plan-extraccion-determinista.md).

Cubre:
- Carga válida conserva el orden de declaración.
- `find` con término de una sola categoría -> esa categoría.
- Frase con "ambos" y "derecho" a la vez -> gana bilateral (orden declarado,
  no la posición dentro del texto).
- Sin ningún término -> None.
- YAML sin laterality, con una lista vacía, o con un valor no-lista -> ValueError en cada caso.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aida.arms.health.laterality import LateralityCatalog, load_laterality


def test_carga_valida_conserva_orden_declaracion() -> None:
    catalog = load_laterality()
    assert isinstance(catalog, LateralityCatalog)
    category_names = [cat for cat, _ in catalog.categories]
    assert category_names == ["bilateral", "derecho", "izquierdo"]


def test_carga_valida_archivo_personalizado_conserva_orden(tmp_path: Path) -> None:
    custom_yaml = tmp_path / "custom_lat.yaml"
    custom_yaml.write_text(
        "laterality:\n"
        "  izquierdo: ['izquierdo']\n"
        "  derecho: ['derecho']\n"
        "  bilateral: ['ambos']\n",
        encoding="utf-8",
    )
    catalog = load_laterality(custom_yaml)
    category_names = [cat for cat, _ in catalog.categories]
    assert category_names == ["izquierdo", "derecho", "bilateral"]


def test_find_termino_unica_categoria() -> None:
    catalog = load_laterality()
    assert catalog.find("dolor en el ojo derecho") == "derecho"
    assert catalog.find("molestia en la pierna izquierda") == "izquierdo"
    assert catalog.find("afecta a ambos lados del cuerpo") == "bilateral"


def test_find_frase_con_ambos_y_derecho_gana_bilateral_por_orden_declarado() -> None:
    catalog = load_laterality()
    # Frase donde "ambos" aparece primero:
    assert catalog.find("dolor en ambos ojos, más en el derecho") == "bilateral"
    # Frase donde "derecho" aparece primero en el texto, pero gana "bilateral" por orden declarado:
    assert catalog.find("presenta dolor derecho pero también en ambos miembros") == "bilateral"


def test_find_sin_termino_devuelve_none() -> None:
    catalog = load_laterality()
    assert catalog.find("dolor abdominal epigástrico") is None
    assert catalog.find("") is None


@pytest.mark.parametrize(
    "yaml_content",
    [
        "{}",
        "secciones:\n  motivo: []\n",
        "laterality: 'no es un mapeo'\n",
        "laterality: 42\n",
        "laterality: {}\n",
        "laterality:\n  derecho: []\n",
        "laterality:\n  derecho: 'no es lista'\n",
        "laterality:\n  derecho: 123\n",
        "laterality:\n  derecho: [123, 'derecha']\n",
    ],
)
def test_load_laterality_yaml_invalido_lanza_value_error(
    tmp_path: Path, yaml_content: str
) -> None:
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_laterality(yaml_file)
