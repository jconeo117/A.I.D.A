"""Tests del módulo de copia de seguridad de la configuración (78d §B.1)."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

import pytest

from aida.platform.snapshot import (
    MAGIC,
    MAX_BYTES,
    PENDING,
    SALT_BYTES,
    Snapshot,
    SnapshotError,
    _fernet,
    clear_pending,
    create_snapshot,
    pending_restore,
    restore_snapshot,
)


def test_ida_y_vuelta_identica(tmp_path: Path) -> None:
    # Arrange
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "claves").mkdir()
    (config_dir / "claves" / "cache.key").write_bytes(b"secreto-equipo-12345")
    (config_dir / "perfiles.json").write_text('{"version": 1}', encoding="utf-8")
    (config_dir / "sub" / "anidada").mkdir(parents=True)
    (config_dir / "sub" / "anidada" / "dato.bin").write_bytes(b"\x00\x01\x02\x03")
    now = datetime(2026, 10, 23, 12, 0, 0)
    passphrase = "frase-segura-12345"

    # Act
    snap = create_snapshot(config_dir, passphrase, now=now, version="0.1.0")
    dest_dir = tmp_path / "restored"
    apartada = restore_snapshot(dest_dir, snap.data, passphrase, now=now)

    # Assert
    assert apartada is None
    assert snap.files == 3
    assert (dest_dir / "claves" / "cache.key").read_bytes() == b"secreto-equipo-12345"
    assert (dest_dir / "perfiles.json").read_text(encoding="utf-8") == '{"version": 1}'
    assert (dest_dir / "sub" / "anidada" / "dato.bin").read_bytes() == b"\x00\x01\x02\x03"


def test_frase_corta_weak(tmp_path: Path) -> None:
    # Arrange
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "archivo.txt").write_text("ok", encoding="utf-8")
    now = datetime(2026, 10, 23, 12, 0, 0)

    # Act & Assert - 11 caracteres falla con weak
    with pytest.raises(SnapshotError) as exc_info:
        create_snapshot(config_dir, "12345678901", now=now, version="0.1.0")
    assert exc_info.value.reason == "weak"

    # 12 caracteres pasa
    snap = create_snapshot(config_dir, "123456789012", now=now, version="0.1.0")
    assert isinstance(snap, Snapshot)


def test_frase_incorrecta_no_toca_el_disco(tmp_path: Path) -> None:
    # Arrange
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "datos.txt").write_text("original", encoding="utf-8")
    now = datetime(2026, 10, 23, 12, 0, 0)
    snap = create_snapshot(config_dir, "frase-correcta-123", now=now, version="0.1.0")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    fichero_previo = dest_dir / "previo.txt"
    fichero_previo.write_text("no me toques", encoding="utf-8")

    # Act & Assert
    with pytest.raises(SnapshotError) as exc_info:
        restore_snapshot(dest_dir, snap.data, "frase-incorrecta-123", now=now)
    assert exc_info.value.reason == "bad_passphrase"
    assert fichero_previo.exists()
    assert fichero_previo.read_text(encoding="utf-8") == "no me toques"
    assert list(tmp_path.glob("dest.antes-*")) == []


def test_archivo_ajeno_o_truncado_bad_file(tmp_path: Path) -> None:
    # Arrange
    dest_dir = tmp_path / "dest"
    now = datetime(2026, 10, 23, 12, 0, 0)
    frase = "frase-prueba-123"

    casos = (
        b"OTROFORMATO\n" + b"\x00" * 32,
        MAGIC + b"123",
        MAGIC + b"\x00" * SALT_BYTES,
        b"esta no es una copia valida",
    )

    # Act & Assert
    for caso in casos:
        with pytest.raises(SnapshotError) as exc_info:
            restore_snapshot(dest_dir, caso, frase, now=now)
        assert exc_info.value.reason == "bad_file"


@pytest.mark.parametrize("ruta_insegura", ["../x", "/etc/x", "C:/x", r"a\b", ""])
def test_ruta_insegura_no_escribe_nada(tmp_path: Path, ruta_insegura: str) -> None:
    # Arrange
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    now = datetime(2026, 10, 23, 12, 0, 0)
    frase = "frase-valida-1234"
    contenido = json.dumps(
        {
            "format": 1,
            "created": now.isoformat(timespec="seconds"),
            "aida_version": "0.1.0",
            "files": {ruta_insegura: base64.b64encode(b"malicioso").decode("ascii")},
        }
    ).encode("utf-8")
    sal = b"\x01" * SALT_BYTES
    cifrado = MAGIC + sal + _fernet(frase, sal).encrypt(contenido)

    # Act & Assert
    with pytest.raises(SnapshotError) as exc_info:
        restore_snapshot(dest_dir, cifrado, frase, now=now)
    assert exc_info.value.reason == "unsafe_path"
    assert not (dest_dir / "x").exists()
    assert not (dest_dir / "malicioso").exists()
    assert not (dest_dir / "a").exists()


def test_restaurar_aparta_la_carpeta_anterior(tmp_path: Path) -> None:
    # Arrange
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "viejo.txt").write_text("contenido anterior", encoding="utf-8")
    ahora = datetime(2026, 10, 23, 14, 30, 0)

    nueva_dir = tmp_path / "nueva"
    nueva_dir.mkdir()
    (nueva_dir / "nuevo.txt").write_text("contenido nuevo", encoding="utf-8")
    snap = create_snapshot(nueva_dir, "frase-segura-1234", now=ahora, version="0.1.0")

    # Act
    apartada = restore_snapshot(config_dir, snap.data, "frase-segura-1234", now=ahora)

    # Assert
    assert apartada is not None
    assert apartada.name == f"config.antes-{ahora.strftime('%Y%m%d-%H%M%S')}"
    assert apartada.is_dir()
    assert (apartada / "viejo.txt").read_text(encoding="utf-8") == "contenido anterior"
    assert (config_dir / "nuevo.txt").read_text(encoding="utf-8") == "contenido nuevo"


def test_aviso_pendiente_y_no_entra_en_la_copia(tmp_path: Path) -> None:
    # Arrange
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "archivo.txt").write_text("algo", encoding="utf-8")
    now = datetime(2026, 10, 23, 10, 0, 0)
    snap = create_snapshot(config_dir, "frase-segura-1234", now=now, version="0.1.0")
    dest_dir = tmp_path / "dest"

    # Act
    restore_snapshot(dest_dir, snap.data, "frase-segura-1234", now=now)

    # Assert
    aviso = pending_restore(dest_dir)
    assert aviso is not None
    assert aviso["restored_at"] == now.isoformat(timespec="seconds")
    assert aviso["files"] == "1"
    assert (dest_dir / PENDING).is_file()

    snap2 = create_snapshot(dest_dir, "frase-segura-1234", now=now, version="0.1.0")
    assert snap2.files == 1

    clear_pending(dest_dir)
    assert pending_restore(dest_dir) is None
    assert not (dest_dir / PENDING).exists()


def test_carpeta_vacia_empty(tmp_path: Path) -> None:
    # Arrange
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    inexistente = tmp_path / "no_existe"
    now = datetime(2026, 10, 23, 10, 0, 0)

    # Act & Assert
    with pytest.raises(SnapshotError) as exc_vacia:
        create_snapshot(vacia, "frase-segura-1234", now=now, version="0.1.0")
    assert exc_vacia.value.reason == "empty"

    with pytest.raises(SnapshotError) as exc_inexistente:
        create_snapshot(inexistente, "frase-segura-1234", now=now, version="0.1.0")
    assert exc_inexistente.value.reason == "empty"


def test_tamano_maximo_too_large(tmp_path: Path) -> None:
    # Arrange
    grande = tmp_path / "grande"
    grande.mkdir()
    (grande / "pesado.bin").write_bytes(b"x" * (MAX_BYTES + 1))
    now = datetime(2026, 10, 23, 10, 0, 0)
    frase = "frase-segura-1234"

    # Act & Assert - al crear
    with pytest.raises(SnapshotError) as exc_create:
        create_snapshot(grande, frase, now=now, version="0.1.0")
    assert exc_create.value.reason == "too_large"

    # Act & Assert - al restaurar contenido descifrado que supera MAX_BYTES
    contenido = json.dumps(
        {
            "format": 1,
            "created": now.isoformat(timespec="seconds"),
            "aida_version": "0.1.0",
            "files": {"big.bin": base64.b64encode(b"y" * (MAX_BYTES + 1)).decode("ascii")},
        }
    ).encode("utf-8")
    sal = b"\x02" * SALT_BYTES
    cifrado = MAGIC + sal + _fernet(frase, sal).encrypt(contenido)
    with pytest.raises(SnapshotError) as exc_restore:
        restore_snapshot(tmp_path / "dest", cifrado, frase, now=now)
    assert exc_restore.value.reason == "too_large"


def test_la_copia_no_contiene_claves_en_claro(tmp_path: Path) -> None:
    # Arrange
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    secreto = b"clave-super-secreta-2026-no-debe-aparecer-en-claro"
    (config_dir / "cache.key").write_bytes(secreto)
    now = datetime(2026, 10, 23, 10, 0, 0)

    # Act
    snap = create_snapshot(config_dir, "frase-segura-1234", now=now, version="0.1.0")

    # Assert
    assert secreto not in snap.data
    assert base64.b64encode(secreto) not in snap.data
    assert b'"files"' not in snap.data


def test_enlaces_simbolicos_no_entran(tmp_path: Path) -> None:
    # Arrange
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "real.txt").write_text("real", encoding="utf-8")
    fuera = tmp_path / "fuera.txt"
    fuera.write_text("fuera", encoding="utf-8")
    enlace = config_dir / "enlace.txt"
    try:
        enlace.symlink_to(fuera)
    except OSError:
        pytest.skip("el sistema no permite crear symlinks")
    now = datetime(2026, 10, 23, 10, 0, 0)

    # Act
    snap = create_snapshot(config_dir, "frase-segura-1234", now=now, version="0.1.0")

    # Assert
    assert snap.files == 1
