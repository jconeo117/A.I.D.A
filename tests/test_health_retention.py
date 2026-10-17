"""Pruebas del cargador de retención declarada en el pack (Plan 85a, §B.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aida.arms.health.retention import Retention, load_retention


def test_load_retention_pack_real_y_validaciones(tmp_path: Path) -> None:
    """B.2 §8: load_retention: lee el pack; revienta con un plazo ausente,
    cero, negativo o booleano.
    """
    real = load_retention()
    assert isinstance(real, Retention)
    assert real.audit_texts_days == 5475
    assert real.transcripts_hours == 24
    assert real.summaries_hours == 1
    assert real.interrupted_hours == 72

    archivo = tmp_path / "retention.yaml"
    archivo.write_text(
        yaml.safe_dump(
            {
                "audit_texts_days": 30,
                "transcripts_hours": 12,
                "summaries_hours": 2,
                "interrupted_hours": 24,
            }
        )
    )
    propia = load_retention(archivo)
    assert propia.audit_texts_days == 30
    assert propia.transcripts_hours == 12
    assert propia.summaries_hours == 2
    assert propia.interrupted_hours == 24

    # Plazo ausente
    archivo.write_text(
        yaml.safe_dump({"audit_texts_days": 30, "transcripts_hours": 12, "summaries_hours": 2})
    )
    with pytest.raises(ValueError, match="enteros positivos"):
        load_retention(archivo)

    # Plazo cero
    archivo.write_text(
        yaml.safe_dump(
            {
                "audit_texts_days": 0,
                "transcripts_hours": 12,
                "summaries_hours": 2,
                "interrupted_hours": 24,
            }
        )
    )
    with pytest.raises(ValueError, match="enteros positivos"):
        load_retention(archivo)

    # Plazo negativo
    archivo.write_text(
        yaml.safe_dump(
            {
                "audit_texts_days": 30,
                "transcripts_hours": -5,
                "summaries_hours": 2,
                "interrupted_hours": 24,
            }
        )
    )
    with pytest.raises(ValueError, match="enteros positivos"):
        load_retention(archivo)

    # Plazo booleano
    archivo.write_text(
        yaml.safe_dump(
            {
                "audit_texts_days": True,
                "transcripts_hours": 12,
                "summaries_hours": 2,
                "interrupted_hours": 24,
            }
        )
    )
    with pytest.raises(ValueError, match="enteros positivos"):
        load_retention(archivo)
