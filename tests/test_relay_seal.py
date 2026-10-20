from __future__ import annotations

import pytest

from aida.relay.seal import new_keypair, open_sealed, seal


def test_sello_ida_y_vuelta() -> None:
    privada, publica = new_keypair()
    texto = b"Hola mundo desde WhatsApp"
    sellado = seal(publica, texto)
    assert open_sealed(privada, sellado) == texto


def test_sello_con_otra_clave_no_abre() -> None:
    _, publica1 = new_keypair()
    privada2, _ = new_keypair()
    sellado = seal(publica1, b"secreto")
    try:
        open_sealed(privada2, sellado)
        pytest.fail("debió fallar")
    except ValueError as exc:
        assert "no es para esta clave" in str(exc)


def test_sello_alterado_no_abre() -> None:
    privada, publica = new_keypair()
    sellado = bytearray(seal(publica, b"mensaje confidencial"))
    assert open_sealed(privada, bytes(sellado)) == b"mensaje confidencial"

    # Alterar un byte del cuerpo cifrado (final)
    alterado_cifrado = bytearray(sellado)
    alterado_cifrado[-1] ^= 0xFF
    try:
        open_sealed(privada, bytes(alterado_cifrado))
        pytest.fail("debió fallar alterado en cifrado")
    except ValueError:
        pass

    # Alterar un byte de la cabecera (datos asociados autenticados: efimera o nonce)
    alterado_cabecera = bytearray(sellado)
    alterado_cabecera[5] ^= 0xFF
    try:
        open_sealed(privada, bytes(alterado_cabecera))
        pytest.fail("debió fallar alterado en cabecera")
    except ValueError:
        pass


def test_dos_sellos_del_mismo_texto_difieren() -> None:
    _, publica = new_keypair()
    texto = b"mismo texto"
    s1 = seal(publica, texto)
    s2 = seal(publica, texto)
    assert s1 != s2
