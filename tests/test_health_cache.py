"""Pruebas unitarias de la caché cifrada en disco (`aida.arms.health.cache`).

Cubre la unidad U9 según la especificación de `.claude/workflow/26-plan-u9.md` §6.4.
Verifica el almacenamiento seguro, aislamiento por paciente y ciclo de expiración:
1. `store` y `load` inmediato recupera el texto Markdown exacto sin alteraciones.
2. Comprobación forense de cifrado en reposo: los bytes crudos del archivo en disco
   no contienen el texto en claro bajo ninguna codificación. (Protege la mutación M4).
3. Expiración real por TTL en lectura: al superar el TTL, `load` devuelve `None`
   y elimina físicamente el archivo del disco (`not path.exists()`).
4. Consulta sobre un `owner` sin registros previos retorna `None` sin excepciones.
5. Aislamiento estricto: registros de pacientes distintos no colisionan ni se sobreescriben.
6. Barrido explícito `purge_expired`: elimina únicamente los archivos caducados,
   conserva los vigentes y reporta el recuento exacto de eliminaciones.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

import aida.engine.ontology as ontology
from aida.arms.health.cache import ConsultationCache


@pytest.fixture
def fernet_key() -> bytes:
    """Genera una clave Fernet válida de 32 bytes en base64 para cada prueba."""
    return Fernet.generate_key()


# ---------------------------------------------------------------------------
# Almacenamiento y recuperación inmediata (§6.4.1)
# ---------------------------------------------------------------------------


def test_cache_store_and_load_immediate_returns_exact_markdown(
    tmp_path: Path, fernet_key: bytes
) -> None:
    """Almacenar un resumen en Markdown y recuperarlo de inmediato devuelve
    el texto íntegro sin truncamientos ni modificaciones.
    """
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    owner = ontology.AgentId("paciente_100")
    markdown_content = (
        "# Resumen Clínico\n\n"
        "- Hallazgo: Cefalea tensional recurrente.\n"
        "- Recomendación: Control en 15 días.\n"
    )

    cache.store(owner=owner, summary_markdown=markdown_content)
    retrieved = cache.load(owner=owner)

    assert retrieved == markdown_content


# ---------------------------------------------------------------------------
# Cifrado en disco y protección de mutación M4 (§6.4.6)
# ---------------------------------------------------------------------------


def test_cache_file_on_disk_does_not_contain_plaintext_summary(
    tmp_path: Path, fernet_key: bytes
) -> None:
    """Inspecciona los bytes crudos en disco para asegurar que el resumen está
    genuinamente cifrado (Fernet) y que ninguna porción del texto en claro
    permanece legible.

    Protege la mutación M4 (§6.4): si en `store` se guarda el texto en claro
    sin invocar `self._fernet.encrypt(...)`, este test falla inmediatamente (RED).
    """
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    owner = ontology.AgentId("paciente_confidencial")
    secret_summary = "DIAGNOSTICO_CONFIDENCIAL_ABC_123: Paciente con dolor toracico agudo"

    cache.store(owner=owner, summary_markdown=secret_summary)

    cache_file = tmp_path / f"{owner}.cache"
    assert cache_file.exists()

    raw_bytes = cache_file.read_bytes()
    # Comprueba que la secuencia de bytes del texto en claro no se encuentra en el archivo
    assert secret_summary.encode("utf-8") not in raw_bytes
    assert b"DIAGNOSTICO_CONFIDENCIAL_ABC_123" not in raw_bytes


# ---------------------------------------------------------------------------
# Expiración por TTL y borrado físico en disco (§6.4.2)
# ---------------------------------------------------------------------------


def test_cache_load_after_ttl_returns_none_and_deletes_file(
    tmp_path: Path, fernet_key: bytes
) -> None:
    """Tras expirar el TTL, `load` debe retornar `None` y eliminar de forma efectiva
    el archivo del sistema de ficheros para evitar fugas de datos huérfanos.
    """
    ttl = 1  # 1 segundo de vida
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=ttl)
    owner = ontology.AgentId("paciente_expirable")

    cache.store(owner=owner, summary_markdown="# Resumen efímero")
    cache_file = tmp_path / f"{owner}.cache"
    assert cache_file.exists()

    # Esperar a que el TTL se agote
    time.sleep(1.2)

    retrieved = cache.load(owner=owner)
    assert retrieved is None
    assert not cache_file.exists(), "El archivo caducado debe eliminarse físicamente del disco"


# ---------------------------------------------------------------------------
# Propietario sin registros (§6.4.3)
# ---------------------------------------------------------------------------


def test_cache_load_unknown_owner_returns_none(
    tmp_path: Path, fernet_key: bytes
) -> None:
    """Consultar la caché para un paciente sobre el que nunca se almacenó nada
    devuelve `None` sin producir errores ni crear archivos vacíos.
    """
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    owner = ontology.AgentId("paciente_inexistente")

    assert cache.load(owner=owner) is None


# ---------------------------------------------------------------------------
# Aislamiento entre pacientes (§6.4.4)
# ---------------------------------------------------------------------------


def test_cache_isolation_between_distinct_owners(
    tmp_path: Path, fernet_key: bytes
) -> None:
    """Verifica que los datos de pacientes diferentes se mantienen completamente
    aislados y no se interfieren ni se sobrescriben mutuamente.
    """
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=3600)
    owner_a = ontology.AgentId("paciente_A")
    owner_b = ontology.AgentId("paciente_B")

    cache.store(owner=owner_a, summary_markdown="# Resumen A")
    cache.store(owner=owner_b, summary_markdown="# Resumen B")

    assert cache.load(owner=owner_a) == "# Resumen A"
    assert cache.load(owner=owner_b) == "# Resumen B"


# ---------------------------------------------------------------------------
# Purga periódica explícita (§6.4.5)
# ---------------------------------------------------------------------------


def test_cache_purge_expired_deletes_only_stale_files_and_returns_count(
    tmp_path: Path, fernet_key: bytes
) -> None:
    """`purge_expired` debe barrer el directorio, suprimir físicamente solo los archivos
    cuyo TTL haya vencido, mantener intactos los vigentes y retornar el número exacto
    de archivos eliminados.
    """
    ttl = 1
    cache = ConsultationCache(tmp_path, key=fernet_key, ttl_seconds=ttl)

    stale_owner_1 = ontology.AgentId("expirado_1")
    stale_owner_2 = ontology.AgentId("expirado_2")
    cache.store(owner=stale_owner_1, summary_markdown="# Resumen viejo 1")
    cache.store(owner=stale_owner_2, summary_markdown="# Resumen viejo 2")

    # Esperar expiración de los dos primeros
    time.sleep(1.2)

    # Almacenar un registro nuevo fresco (vigente)
    fresh_owner = ontology.AgentId("vigente_1")
    cache.store(owner=fresh_owner, summary_markdown="# Resumen reciente")

    # Ejecutar la purga explícita
    deleted_count = cache.purge_expired()

    assert deleted_count == 2
    assert not (tmp_path / f"{stale_owner_1}.cache").exists()
    assert not (tmp_path / f"{stale_owner_2}.cache").exists()
    assert (tmp_path / f"{fresh_owner}.cache").exists()
    assert cache.load(owner=fresh_owner) == "# Resumen reciente"
