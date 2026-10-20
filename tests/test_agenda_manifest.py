from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aida.platform.arms import load_manifest
from aida.resources import data_root


def test_load_manifest_exige_arm_e_icon(tmp_path: Path) -> None:
    base = {
        "id": "mi_brazo",
        "name": "Mi Brazo",
        "description": "Desc",
        "version": "1.0.0",
        "arm": "aida.run:health_arm",
        "icon": "stetho",
    }
    f_ok = tmp_path / "arm_ok.yaml"
    f_ok.write_text(yaml.dump(base), encoding="utf-8")
    m = load_manifest(f_ok)
    assert m.id == "mi_brazo"
    assert m.arm == "aida.run:health_arm"
    assert m.icon == "stetho"

    sin_arm = dict(base)
    del sin_arm["arm"]
    f_sin_arm = tmp_path / "sin_arm.yaml"
    f_sin_arm.write_text(yaml.dump(sin_arm), encoding="utf-8")
    with pytest.raises(ValueError, match="arm"):
        load_manifest(f_sin_arm)

    sin_icon = dict(base)
    del sin_icon["icon"]
    f_sin_icon = tmp_path / "sin_icon.yaml"
    f_sin_icon.write_text(yaml.dump(sin_icon), encoding="utf-8")
    with pytest.raises(ValueError, match="icon"):
        load_manifest(f_sin_icon)


def test_load_manifest_captura_extra(tmp_path: Path) -> None:
    data = {
        "id": "agenda",
        "name": "Agenda",
        "description": "Desc",
        "version": "1.0.0",
        "arm": "aida.arms.agenda.app:build_arm",
        "icon": "calendar",
        "default_sector": "consultorio",
    }
    f = tmp_path / "arm.yaml"
    f.write_text(yaml.dump(data), encoding="utf-8")
    m = load_manifest(f)
    assert m.extra.get("default_sector") == "consultorio"


def test_manifiestos_reales_de_agenda_y_health() -> None:
    p_agenda = data_root() / "packs" / "agenda" / "arm.yaml"
    m_agenda = load_manifest(p_agenda)
    assert m_agenda.id == "agenda"
    assert m_agenda.arm == "aida.arms.agenda.app:build_arm"
    assert m_agenda.icon == "calendar"
    assert m_agenda.extra.get("default_sector") == "consultorio"

    p_health = data_root() / "packs" / "health" / "arm.yaml"
    m_health = load_manifest(p_health)
    assert m_health.id == "health"
    assert m_health.arm == "aida.run:health_arm"
    assert m_health.icon == "stetho"
