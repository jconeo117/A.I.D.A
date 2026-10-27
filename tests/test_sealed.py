"""Pruebas unitarias de archivos sellados en reposo y temporales en claro (Plan 85a, §B.2).

Cubre:
1. seal_bytes + open_bytes: integridad, 0o600, sin claro en disco, InvalidToken con otra clave.
2. SealedAppender: tres append ordenados, concatenación con read_appended, robustez
   frente a truncamiento final manual (las anteriores siguen legibles).
3. plain_copy y plain_directory: existencia en claro con 0o600 dentro del with, borrado
   completo al salir (incluso ante excepción), purge_plain_leftovers elimina restos.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
from cryptography.fernet import Fernet, InvalidToken

from aida.platform.sealed import (
    PLAIN_PREFIX,
    SealedAppender,
    open_bytes,
    plain_copy,
    plain_name,
    purge_plain_leftovers,
    read_appended,
    seal_bytes,
)


def test_seal_bytes_open_bytes_cifrado_y_permisos(tmp_path: Path) -> None:
    """B.2 §1: seal_bytes + open_bytes devuelven lo mismo; el archivo sellado
    no contiene los bytes en claro (busca una marca conocida) y tiene 0o600;
    con otra clave, InvalidToken.
    """
    clave = Fernet.generate_key()
    otra_clave = Fernet.generate_key()
    marca_secreta = b"MARCA_DE_PRUEBA_CONFIDENCIAL_12345"
    datos = b"RIFF" + marca_secreta + b"WAVEfmt "
    archivo_sellado = tmp_path / "audio.wav.sealed"

    seal_bytes(archivo_sellado, datos, clave)

    assert archivo_sellado.is_file()
    assert sys.platform == "win32" or (archivo_sellado.stat().st_mode & 0o777) == 0o600
    contenido_disco = archivo_sellado.read_bytes()
    assert marca_secreta not in contenido_disco
    assert b"RIFF" not in contenido_disco

    recuperado = open_bytes(archivo_sellado, clave)
    assert recuperado == datos

    with pytest.raises(InvalidToken):
        open_bytes(archivo_sellado, otra_clave)


def test_sealed_appender_concatenacion_y_tolerancia_truncamiento(tmp_path: Path) -> None:
    """B.2 §2: SealedAppender: tres append => read_appended los concatena en orden;
    una línea final truncada a mano => las anteriores siguen legibles con read_appended
    hasta la rota (documenta lo que pasa con la rota).
    """
    clave = Fernet.generate_key()
    archivo = tmp_path / "stream.pcm.sealed"
    appender = SealedAppender(archivo, clave)

    trozo1 = b"PRIMER_TROZO_DE_AUDIO"
    trozo2 = b"SEGUNDO_TROZO_DE_AUDIO"
    trozo3 = b"TERCER_TROZO_DE_AUDIO"

    appender.append(trozo1)
    appender.append(trozo2)
    appender.append(trozo3)

    assert read_appended(archivo, clave) == trozo1 + trozo2 + trozo3

    lineas = archivo.read_bytes().splitlines()
    assert len(lineas) == 3
    linea_truncada = lineas[2][: len(lineas[2]) // 2]
    archivo_roto = tmp_path / "stream_truncado.pcm.sealed"
    archivo_roto.write_bytes(b"\n".join([lineas[0], lineas[1], linea_truncada]) + b"\n")

    # 1. Con la última línea truncada (corte a mitad de escritura), read_appended tolera
    #    el corte y devuelve los trozos completos anteriores (87).
    assert read_appended(archivo_roto, clave) == trozo1 + trozo2

    # 2. Con una línea rota en medio (daño real), falla con InvalidToken
    archivo_medio_roto = tmp_path / "stream_medio_roto.pcm.sealed"
    archivo_medio_roto.write_bytes(b"\n".join([lineas[0], linea_truncada, lineas[2]]) + b"\n")
    with pytest.raises(InvalidToken):
        read_appended(archivo_medio_roto, clave)


def test_plain_copy_y_purge_leftovers(tmp_path: Path) -> None:
    """B.2 §3: plain_copy: dentro del with, el archivo existe en claro con el
    nombre sin .sealed y 0o600; al salir, la carpeta no existe; también si
    dentro se lanza una excepción. purge_plain_leftovers borra carpetas aida-claro-*.
    """
    clave = Fernet.generate_key()
    sellado = tmp_path / "historia.pdf.sealed"
    datos = b"%PDF-1.4 documento confidencial"
    seal_bytes(sellado, datos, clave)

    assert plain_name(sellado) == "historia.pdf"

    carpeta_guardada: Path | None = None
    with plain_copy(sellado, clave) as claro:
        assert claro.is_file()
        assert claro.name == "historia.pdf"
        assert sys.platform == "win32" or (claro.stat().st_mode & 0o777) == 0o600
        assert claro.read_bytes() == datos
        carpeta_guardada = claro.parent
        assert carpeta_guardada.name.startswith(PLAIN_PREFIX)
        assert carpeta_guardada.exists()

    assert carpeta_guardada is not None
    assert not carpeta_guardada.exists()

    carpeta_err: Path | None = None
    with pytest.raises(ZeroDivisionError), plain_copy(sellado, clave) as claro:
        carpeta_err = claro.parent
        assert carpeta_err.exists()
        _ = 1 / 0

    assert carpeta_err is not None
    assert not carpeta_err.exists()

    resto1 = Path(tempfile.gettempdir()) / f"{PLAIN_PREFIX}huerfana-1"
    resto2 = Path(tempfile.gettempdir()) / f"{PLAIN_PREFIX}huerfana-2"
    resto1.mkdir(parents=True, exist_ok=True)
    resto2.mkdir(parents=True, exist_ok=True)
    (resto1 / "basura.txt").write_text("restos de proceso caido")

    assert resto1.exists()
    assert resto2.exists()
    purge_plain_leftovers()
    assert not resto1.exists()
    assert not resto2.exists()
