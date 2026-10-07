"""Dónde están los datos de AIDA y dónde escribe, según cómo se ejecute.

Una sola puerta para las dos preguntas. Antes, doce módulos encontraban el pack y el
catálogo subiendo directorios desde su propio archivo, lo que solo funciona dentro del
repositorio (`I-096`); y las carpetas del usuario estaban escritas a mano para Linux.
El ejecutable de Windows (plan 78b) necesita otras rutas para las dos cosas.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def data_root() -> Path:
    """La carpeta que contiene `packs/` y `data/`: `AIDA_DATA_DIR` si se declara; la
    carpeta del ejecutable si AIDA está empaquetada (PyInstaller la deja en
    `sys._MEIPASS`); y la raíz del repositorio en desarrollo."""
    declarada = os.environ.get("AIDA_DATA_DIR")
    if declarada:
        return Path(declarada)
    empaquetada = getattr(sys, "_MEIPASS", None)
    if isinstance(empaquetada, str):
        return Path(empaquetada)
    return Path(__file__).resolve().parents[2]


def user_dir(kind: str) -> Path:
    """Dónde escribe AIDA en la carpeta del usuario: `config` (claves y ajustes),
    `cache` (transcripciones y resúmenes cifrados), `data` (métricas) o `documents`
    (lo exportado).

    En Linux son las de siempre (`~/.config/aida`, `~/.cache/aida`,
    `~/.local/share/aida`, `~/Documentos/aida`), para que una instalación existente
    siga encontrando sus claves y su caché. En Windows, `%APPDATA%` para la
    configuración y `%LOCALAPPDATA%` para lo demás, que es donde el sistema espera
    cada cosa. `AIDA_CONFIG_DIR` pisa la de configuración: la batería lo usa para no
    leer nunca las claves reales de quien la corre."""
    if kind == "config" and os.environ.get("AIDA_CONFIG_DIR"):
        return Path(os.environ["AIDA_CONFIG_DIR"])
    casa = Path.home()
    if sys.platform == "win32":
        roaming = Path(os.environ.get("APPDATA") or casa / "AppData" / "Roaming")
        local = Path(os.environ.get("LOCALAPPDATA") or casa / "AppData" / "Local")
        windows = {
            "config": roaming / "AIDA",
            "cache": local / "AIDA" / "cache",
            "data": local / "AIDA",
            "documents": casa / "Documents" / "AIDA",
        }
        return windows[kind]
    linux = {
        "config": casa / ".config" / "aida",
        "cache": casa / ".cache" / "aida",
        "data": casa / ".local" / "share" / "aida",
        "documents": casa / "Documentos" / "aida",
    }
    return linux[kind]
