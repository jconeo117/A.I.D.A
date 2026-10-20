"""El sello: lo que el relé guarda solo lo abre el equipo del cliente.

X25519 efímera por entrada, HKDF-SHA256 y ChaCha20-Poly1305. Formato:
`versión (1) | pública efímera (32) | nonce (12) | cifrado con etiqueta`. La versión y la
pública efímera van autenticadas como datos asociados: cambiar un byte de la cabecera hace que
no abra. Vive solo aquí: el relé sella y el equipo abre con el mismo módulo.
"""

from __future__ import annotations

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

VERSION = b"\x01"
_INFO = b"aida-relay-seal-v1"
_HEADER = 1 + 32 + 12


def _raw(public: X25519PublicKey) -> bytes:
    return public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _derive(shared: bytes, ephemeral: bytes, recipient: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=ephemeral + recipient, info=_INFO
    ).derive(shared)


def new_keypair() -> tuple[bytes, bytes]:
    """`(privada, pública)`, 32 bytes cada una."""
    privada = X25519PrivateKey.generate()
    return (
        privada.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        ),
        _raw(privada.public_key()),
    )


def public_of(private: bytes) -> bytes:
    return _raw(X25519PrivateKey.from_private_bytes(private).public_key())


def seal(public: bytes, plaintext: bytes) -> bytes:
    efimera = X25519PrivateKey.generate()
    efimera_pub = _raw(efimera.public_key())
    clave = _derive(
        efimera.exchange(X25519PublicKey.from_public_bytes(public)), efimera_pub, public
    )
    nonce = os.urandom(12)
    cifrado = ChaCha20Poly1305(clave).encrypt(nonce, plaintext, VERSION + efimera_pub)
    return VERSION + efimera_pub + nonce + cifrado


def open_sealed(private: bytes, sealed: bytes) -> bytes:
    """`ValueError` si no es un sello de esta versión, si no es para esta clave o si se alteró:
    quien abre no distingue entre las tres, y no debe."""
    if len(sealed) < _HEADER + 16 or sealed[:1] != VERSION:
        raise ValueError("sello desconocido")
    efimera_pub, nonce, cifrado = sealed[1:33], sealed[33:_HEADER], sealed[_HEADER:]
    propia = X25519PrivateKey.from_private_bytes(private)
    clave = _derive(
        propia.exchange(X25519PublicKey.from_public_bytes(efimera_pub)),
        efimera_pub,
        _raw(propia.public_key()),
    )
    try:
        return ChaCha20Poly1305(clave).decrypt(nonce, cifrado, VERSION + efimera_pub)
    except InvalidTag as exc:
        raise ValueError("el sello no es para esta clave o fue alterado") from exc
