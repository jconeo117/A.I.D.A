"""Los textos de la plataforma: `packs/platform/ui.yaml`, sección `shell` (101a).

Un brazo no repite estas claves: el armazón las pinta en su nombre, y un texto duplicado
en dos archivos sería dos verdades. `load_app_wording` de salud lo comprueba al cargar.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from aida.resources import data_root

_POR_DEFECTO = data_root() / "packs" / "platform" / "ui.yaml"

SHELL_KEYS = (
    "who_title", "who_lead", "who_add", "who_add_hint", "registry_short", "pin_label",
    "enter_button", "first_title", "first_lead", "first_name", "first_registry", "first_pin",
    "first_pin2", "first_hint", "first_button", "first_pin_mismatch", "locked_title",
    "locked_lead", "unlock_button", "other_professional", "title_locked", "title_who",
    "pill_locked", "change_professional", "profiles_title", "profiles_text", "profiles_add",
    "profiles_added", "identity_missing", "identity_pin_format", "identity_duplicate",
    "identity_unknown", "identity_bad_pin", "identity_wait", "license_title", "license_client",
    "license_arms", "license_until", "license_load", "license_status_valid",
    "license_status_expiring", "license_status_expired", "license_status_missing",
    "license_status_invalid", "license_warn", "license_missing_title", "license_missing_lead",
    "license_rejected", "readonly_title", "readonly_lead", "readonly_hint", "pill_readonly",
    "license_readonly", "go_settings", "exit_label", "closed_text", "settings_label",
    "title_settings", "settings_lead", "back_label", "title_first_run", "header_place",
    "pill_ready", "pill_problem", "brand", "launcher_title", "launcher_lead", "launcher_link",
    "identity_bad_code", "recovery_alert", "forgot_link", "pin_title", "pin_lead",
    "recover_code_label", "recover_button", "recover_done", "reset_title", "reset_button",
    "reset_done", "code_button",
    "snapshot_title", "snapshot_text", "snapshot_pass", "snapshot_pass2", "snapshot_create",
    "snapshot_created", "snapshot_restore_title", "snapshot_restore_text", "snapshot_file",
    "snapshot_restore_pass", "snapshot_restore", "snapshot_confirm", "snapshot_restored",
    "snapshot_nofile", "snapshot_first_link", "snapshot_error_weak", "snapshot_error_mismatch",
    "snapshot_error_empty", "snapshot_error_too_large", "snapshot_error_bad_file",
    "snapshot_error_bad_passphrase", "snapshot_error_unsafe_path", "snapshot_confirm_needed",
)
"""Las claves que el armazón pinta. Ninguna de ellas puede estar en `app` de salud."""


class Textos(Protocol):
    """Lo que necesita una pantalla: sus textos por clave y el tiempo sin uso."""

    @property
    def idle_lock_minutes(self) -> float: ...

    def __getitem__(self, clave: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ShellWording:
    texts: Mapping[str, str]
    idle_lock_minutes: float
    """Minutos sin uso tras los que se pide el PIN otra vez (plan 83). Cuánto se tolera un
    equipo desatendido es decisión de cada consultorio, y se declara en el pack."""

    def __getitem__(self, clave: str) -> str:
        return self.texts[clave]


def load_shell_wording(path: Path | None = None) -> ShellWording:
    """Revienta si falta una clave o el tiempo sin uso no es positivo: una pantalla a medias
    se descubre con alguien esperando delante."""
    ruta = path if path is not None else _POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    seccion = raw.get("shell") if isinstance(raw, dict) else None
    if not isinstance(seccion, dict):
        raise ValueError(f"falta la sección 'shell' en {ruta}")
    claves_texto = tuple(c for c in SHELL_KEYS if c != "idle_lock_minutes")
    faltan = [c for c in claves_texto if not isinstance(seccion.get(c), str)]
    if faltan:
        raise ValueError(f"faltan claves en la sección 'shell' de {ruta}: {faltan}")
    minutos = seccion.get("idle_lock_minutes")
    if not isinstance(minutos, int | float) or isinstance(minutos, bool) or minutos <= 0:
        raise ValueError(f"'shell.idle_lock_minutes' debe ser un número positivo en {ruta}")
    return ShellWording(
        texts={c: seccion[c] for c in claves_texto},
        idle_lock_minutes=float(minutos),
    )
