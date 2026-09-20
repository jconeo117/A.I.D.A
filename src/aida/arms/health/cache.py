"""Caché del resumen, aislado por paciente (`owner`), cifrado en disco,
con expiración real — no ocultar lo vencido, borrarlo.

La clave de cifrado sale de una variable de entorno
(`AIDA_CONSULT_CACHE_KEY`), igual que las credenciales de proveedor de U7
(`ANTHROPIC_API_KEY`, etc.): quien despliega la gestiona. No es una bóveda
de secretos de producción, pero es honesto sobre lo que protege — cifrado
sin gestión de claves seria es cifrado de cara a la galería, y este
proyecto no finge lo contrario.
"""

from __future__ import annotations

import time
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from aida.engine.ontology import AgentId


class ConsultationCache:
    def __init__(self, directory: Path, *, key: bytes, ttl_seconds: int = 3600) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(key)
        self._ttl = ttl_seconds

    def _path_for(self, owner: AgentId) -> Path:
        # Se toma solo el nombre base para prevenir secuencias de escape de directorio
        # si el AgentId contiene rutas relativas o absolutas.
        safe_name = Path(str(owner)).name
        return self._directory / f"{safe_name}.cache"

    def store(self, *, owner: AgentId, summary_markdown: str) -> None:
        token = self._fernet.encrypt(summary_markdown.encode("utf-8"))
        self._path_for(owner).write_bytes(token)

    def load(self, *, owner: AgentId) -> str | None:
        """`None` si no hay nada cacheado, si caducó (y en ese caso lo
        borra de verdad), o si la clave no puede descifrarlo."""
        path = self._path_for(owner)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self._ttl:
            path.unlink(missing_ok=True)
            return None
        try:
            return self._fernet.decrypt(path.read_bytes()).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError):
            return None

    def purge_expired(self) -> int:
        """Barrido explícito, para llamar periódicamente en vez de confiar
        solo en que alguien vuelva a pedir ese resumen. Devuelve cuántos
        borró."""
        borrados = 0
        for path in sorted(self._directory.glob("*.cache")):
            if time.time() - path.stat().st_mtime > self._ttl:
                path.unlink(missing_ok=True)
                borrados += 1
        return borrados

