"""Voz: redacción de lo que el motor ya decidió, nunca decisión nueva.

`engine/` no importa nada de aquí — solo el `Protocol CognitionProvider` de
`ports.py`. Quien orquesta una conversación (`session.py`) decide qué
implementación pasar, igual que quien construye un `Loop` decide si le pasa
un `IntentClassifier` (U6).

Directorio exento del guardián de imports de modelo
(`tests/test_agnosticism.py`, `EXEMPT_DIR_NAMES`): aquí, y solo aquí, vive
código que habla con un proveedor externo.
"""

from __future__ import annotations
