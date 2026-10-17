"""Tests de localización de datos y carpetas de usuario (plan 78, §B.1).

Cubre:
1. data_root(): AIDA_DATA_DIR > sys._MEIPASS > raíz del repositorio.
2. Lectura real desde AIDA_DATA_DIR con recarga de módulo (match.py).
3. user_dir en Linux con las rutas estándar bajo HOME.
4. user_dir en Windows (sys.platform = "win32") con APPDATA, LOCALAPPDATA y HOME.
5. AIDA_CONFIG_DIR pisa únicamente la carpeta de configuración.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

import pytest

from aida.resources import data_root, user_dir


def test_data_root_prioridad_entorno_meipass_y_repositorio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 1 (§B.1, M1): data_root() con AIDA_DATA_DIR devuelve esa ruta; con
    sys._MEIPASS simulado y sin variable devuelve _MEIPASS; sin nada devuelve la
    raíz del repositorio donde vive packs/health/consultorio/pack.yaml.
    """
    # 1. Con AIDA_DATA_DIR
    custom_dir = tmp_path / "custom_data"
    monkeypatch.setenv("AIDA_DATA_DIR", str(custom_dir))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert data_root() == custom_dir

    # 2. Con sys._MEIPASS simulado y sin variable
    monkeypatch.delenv("AIDA_DATA_DIR", raising=False)
    meipass_dir = str(tmp_path / "meipass_bundle")
    monkeypatch.setattr(sys, "_MEIPASS", meipass_dir, raising=False)
    assert data_root() == Path(meipass_dir)

    # 3. Sin variable ni _MEIPASS: raíz del repositorio
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    raiz = data_root()
    assert (raiz / "packs" / "health" / "consultorio" / "pack.yaml").is_file()


def test_datos_se_leen_de_verdad_desde_aida_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 2 (§B.1, M1, M2): Los datos se leen desde AIDA_DATA_DIR al recargar
    aida.arms.health.match con importlib.reload.
    """
    import aida.arms.health.match as match_mod

    # Copiamos packs y data/health/cie10 a tmp_path
    repo_root = Path(__file__).resolve().parents[1]
    shutil.copytree(repo_root / "packs", tmp_path / "packs")
    (tmp_path / "data" / "health").mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root / "data" / "health" / "cie10", tmp_path / "data" / "health" / "cie10")

    matching_yaml = tmp_path / "packs" / "health" / "consultorio" / "matching.yaml"
    texto = matching_yaml.read_text(encoding="utf-8")
    # Modificamos threshold a un valor distintivo
    nuevo_texto = texto.replace("threshold: 0.50", "threshold: 0.42")
    assert "threshold: 0.42" in nuevo_texto
    matching_yaml.write_text(nuevo_texto, encoding="utf-8")

    monkeypatch.setenv("AIDA_DATA_DIR", str(tmp_path))
    try:
        importlib.reload(match_mod)
        settings = match_mod.load_matching_settings()
        assert settings.threshold == 0.42
    finally:
        monkeypatch.delenv("AIDA_DATA_DIR", raising=False)
        importlib.reload(match_mod)

    # Verificamos restauración
    assert match_mod.load_matching_settings().threshold == 0.50



def test_user_dir_en_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Caso 3 (§B.1): user_dir en Linux devuelve las 4 rutas estándar bajo HOME."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("AIDA_CONFIG_DIR", raising=False)
    monkeypatch.delenv("AIDA_DATA_DIR", raising=False)

    assert user_dir("config") == tmp_path / ".config" / "aida"
    assert user_dir("cache") == tmp_path / ".cache" / "aida"
    assert user_dir("data") == tmp_path / ".local" / "share" / "aida"
    assert user_dir("documents") == tmp_path / "Documentos" / "aida"


def test_user_dir_en_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Caso 4 (§B.1, M3): user_dir con sys.platform='win32', APPDATA y LOCALAPPDATA
    fijados ubica config en Roaming, cache y data en Local, y documents en Documents.
    """
    roaming = tmp_path / "AppData" / "Roaming"
    local = tmp_path / "AppData" / "Local"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.delenv("AIDA_CONFIG_DIR", raising=False)
    monkeypatch.delenv("AIDA_DATA_DIR", raising=False)

    assert user_dir("config") == roaming / "AIDA"
    assert user_dir("cache") == local / "AIDA" / "cache"
    assert user_dir("data") == local / "AIDA"
    assert user_dir("documents") == tmp_path / "Documents" / "AIDA"


def test_aida_config_dir_pisa_solo_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 5 (§B.1): AIDA_CONFIG_DIR pisa config y solo config."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    override = tmp_path / "custom_config_dir"
    monkeypatch.setenv("AIDA_CONFIG_DIR", str(override))
    monkeypatch.delenv("AIDA_DATA_DIR", raising=False)

    assert user_dir("config") == override
    assert user_dir("cache") == tmp_path / ".cache" / "aida"
    assert user_dir("data") == tmp_path / ".local" / "share" / "aida"
    assert user_dir("documents") == tmp_path / "Documentos" / "aida"
