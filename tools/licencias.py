"""Licencias de AIDA: generar la pareja de claves, firmar y enseñar (plan 100).

No viaja con AIDA (queda fuera del instalador de la 78b). La clave privada se guarda
**fuera del repositorio**; quien la tenga puede fabricar licencias.

    python tools/licencias.py keygen DIRECTORIO_PRIVADO
        Crea la pareja. Escribe la privada en DIRECTORIO_PRIVADO/aida-licencias.key
        (0600) y la pública en src/aida/platform/_public_key.py. Se niega a escribir la
        privada dentro del repositorio, y a pisar una pareja ya existente.

    python tools/licencias.py sign CLAVE_PRIVADA --client NOMBRE --arms health[,otro]
                                    --expires AAAA-MM-DD --out ARCHIVO
    python tools/licencias.py show ARCHIVO
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MODULO = RAIZ / "src" / "aida" / "platform" / "_public_key.py"
PLANTILLA = '''"""La clave pública con la que AIDA verifica las licencias (plan 100).

La escribe `tools/licencias.py keygen`. Es código y no un archivo de datos a propósito:
un archivo de datos se sustituye por otro sin tocar nada, y entonces cualquiera firmaría
sus propias licencias. Sin clave (`None`), ninguna licencia es válida: no hay modo
desarrollo que se la salte.
"""

PUBLIC_KEY: str | None = "{clave}"
'''


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def keygen(destino: Path) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    destino = destino.resolve()
    if destino == RAIZ or RAIZ in destino.parents:
        print("error: la clave privada no se guarda dentro del repositorio", file=sys.stderr)
        return 1
    privada = destino / "aida-licencias.key"
    if privada.exists():
        print(f"error: ya hay una clave en {privada}; no se pisa", file=sys.stderr)
        return 1
    clave = Ed25519PrivateKey.generate()
    destino.mkdir(parents=True, exist_ok=True)
    privada.write_bytes(
        clave.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    privada.chmod(0o600)
    publica = clave.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    MODULO.write_text(PLANTILLA.format(clave=base64.b64encode(publica).decode()), encoding="utf-8")
    print(f"privada: {privada}")
    print(f"pública: {MODULO}")
    return 0


def sign(privada: Path, client: str, arms: str, expires: str, out: Path) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    clave = serialization.load_pem_private_key(privada.read_bytes(), password=None)
    if not isinstance(clave, Ed25519PrivateKey):
        print("error: no es una clave Ed25519", file=sys.stderr)
        return 1
    date.fromisoformat(expires)
    payload: dict[str, object] = {
        "id": secrets.token_hex(6),
        "client": client,
        "arms": sorted(a.strip() for a in arms.split(",") if a.strip()),
        "issued": date.today().isoformat(),
        "expires": expires,
    }
    firma = clave.sign(_canonical(payload))
    out.write_text(
        json.dumps({"payload": payload, "signature": base64.b64encode(firma).decode()},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"licencia {payload['id']} para {client}: {', '.join(payload['arms'])} hasta {expires}")
    return 0


def show(archivo: Path) -> int:
    sys.path.insert(0, str(RAIZ / "src"))
    from aida.platform.license import read_license

    if not archivo.is_file():
        print(f"error: no existe {archivo}", file=sys.stderr)
        return 1
    estado = read_license(archivo.read_text(encoding="utf-8"), today=date.today())
    print(f"estado: {estado.status}")
    if estado.license is not None:
        lic = estado.license
        print(f"id {lic.id} · {lic.client} · {', '.join(lic.arms)} · hasta {lic.expires}")
    return 0 if estado.status in ("valid", "expiring") else 1


def main() -> int:
    p = argparse.ArgumentParser(prog="licencias")
    sub = p.add_subparsers(dest="orden", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("destino", type=Path)
    s = sub.add_parser("sign")
    s.add_argument("privada", type=Path)
    s.add_argument("--client", required=True)
    s.add_argument("--arms", required=True)
    s.add_argument("--expires", required=True)
    s.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("show")
    w.add_argument("archivo", type=Path)
    a = p.parse_args()
    if a.orden == "keygen":
        return keygen(a.destino)
    if a.orden == "sign":
        return sign(a.privada, a.client, a.arms, a.expires, a.out)
    return show(a.archivo)


if __name__ == "__main__":
    raise SystemExit(main())
