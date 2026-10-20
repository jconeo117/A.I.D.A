from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aida.arms.agenda.setup import load_setup


def test_load_setup_copia_plantilla_por_defecto_si_no_existe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(tmp_path))
    destino = tmp_path / "agenda.yaml"
    assert not destino.exists()
    setup = load_setup(destino)
    assert destino.exists()
    assert setup.sector in ("consultorio", "peluqueria")
    assert setup.slot_minutes > 0
    assert len(setup.services) > 0
    assert len(setup.resources) > 0


def test_load_setup_falla_si_no_es_mapeo(tmp_path: Path) -> None:
    f = tmp_path / "invalido.yaml"
    f.write_text("- lista\n- en vez\n- de dict", encoding="utf-8")
    with pytest.raises(ValueError, match="debe ser un mapeo"):
        load_setup(f)


def test_load_setup_falla_slot_minutes_invalido(tmp_path: Path) -> None:
    f = tmp_path / "invalido.yaml"
    f.write_text("sector: s\nclient_label: c\nslot_minutes: -10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="slot_minutes"):
        load_setup(f)


def test_load_setup_falla_minutos_servicio_invalido(tmp_path: Path) -> None:
    data = {
        "sector": "s",
        "client_label": "c",
        "slot_minutes": 15,
        "services": [{"id": "s1", "label": "S1", "minutes": 0}],
    }
    f = tmp_path / "invalido.yaml"
    f.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="minutos positivos"):
        load_setup(f)


def test_load_setup_falla_recurso_con_servicio_desconocido(tmp_path: Path) -> None:
    data = {
        "sector": "s",
        "client_label": "c",
        "slot_minutes": 15,
        "services": [{"id": "s1", "label": "S1", "minutes": 15}],
        "resources": [{"id": "r1", "label": "R1", "place": "P1", "services": ["fantasma"]}],
    }
    f = tmp_path / "invalido.yaml"
    f.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="servicios que no existen"):
        load_setup(f)


def test_load_setup_falla_dia_desconocido(tmp_path: Path) -> None:
    data = {
        "sector": "s",
        "client_label": "c",
        "slot_minutes": 15,
        "services": [{"id": "s1", "label": "S1", "minutes": 15}],
        "resources": [{
            "id": "r1", "label": "R1", "place": "P1", "services": ["s1"],
            "hours": {"funday": [["08:00", "12:00"]]},
        }],
    }
    f = tmp_path / "invalido.yaml"
    f.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="día desconocido"):
        load_setup(f)


def test_load_setup_falla_hora_mal_formateada(tmp_path: Path) -> None:
    data = {
        "sector": "s",
        "client_label": "c",
        "slot_minutes": 15,
        "services": [{"id": "s1", "label": "S1", "minutes": 15}],
        "resources": [{
            "id": "r1", "label": "R1", "place": "P1", "services": ["s1"],
            "hours": {"mon": [["ocho", "12:00"]]},
        }],
    }
    f = tmp_path / "invalido.yaml"
    f.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="hora"):
        load_setup(f)


def test_load_setup_falla_tramo_al_reves(tmp_path: Path) -> None:
    data = {
        "sector": "s",
        "client_label": "c",
        "slot_minutes": 15,
        "services": [{"id": "s1", "label": "S1", "minutes": 15}],
        "resources": [{
            "id": "r1", "label": "R1", "place": "P1", "services": ["s1"],
            "hours": {"mon": [["12:00", "08:00"]]},
        }],
    }
    f = tmp_path / "invalido.yaml"
    f.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="al revés"):
        load_setup(f)


def test_load_setup_falla_tramo_no_multiplo_slot(tmp_path: Path) -> None:
    data = {
        "sector": "s",
        "client_label": "c",
        "slot_minutes": 30,
        "services": [{"id": "s1", "label": "S1", "minutes": 30}],
        "resources": [{
            "id": "r1", "label": "R1", "place": "P1", "services": ["s1"],
            "hours": {"mon": [["08:00", "08:45"]]},
        }],
    }
    f = tmp_path / "invalido.yaml"
    f.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="múltiplo"):
        load_setup(f)
