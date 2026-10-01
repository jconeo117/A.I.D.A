"""Nube con contención local para la redacción.

AIDA corre en el equipo que tenga el profesional, así que el trabajo pesado
va a un proveedor de nube. Pero un proveedor caído no puede dejar sin voz a
nada que dependa de él: es la misma regla que ya gobierna la transcripción
(`arms/health/transcription.py`) —si el proveedor cae, AIDA sigue trabajando
y solo pierde calidad de prosa—.
"""

from __future__ import annotations

from collections.abc import Callable

from aida.engine.ports import CognitionProvider


class FallbackCognition:
    """Un proveedor primero, otro si el primero no sirve.

    Mismo patrón que `FallbackTranscription` y por el mismo motivo: si el
    proveedor de nube cae, AIDA sigue trabajando y solo pierde calidad de
    prosa. «No sirve» incluye devolver `None`, no solo levantar: un proveedor
    que contesta algo que no valida es un proveedor que no contestó.
    """

    def __init__(
        self,
        primary: CognitionProvider,
        fallback: CognitionProvider,
        *,
        on_fallback: Callable[[str], None] | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._on_fallback = on_fallback

    def think(self, briefing: str, schema: type) -> object | None:
        try:
            result = self._primary.think(briefing, schema)
        except Exception as exc:
            result = None
            razon = str(exc)
        else:
            razon = "sin validar"
        if result is not None:
            return result
        if self._on_fallback is not None:
            self._on_fallback(razon)
        try:
            return self._fallback.think(briefing, schema)
        except Exception:
            return None
