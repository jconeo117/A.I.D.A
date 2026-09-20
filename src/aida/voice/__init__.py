"""Audio de entrada: escuchar hasta que la persona deja de hablar, y
reconocer los comandos del operador. Local, sin nube — mismo criterio que
`arms/health/` para STT/OCR (U9).

Todo se inyecta: el origen de trozos de audio y el detector de habla son
llamables que un test sustituye sin micrófono ni modelo. Nada aquí conoce
el dominio: las frases de los comandos las declara quien construye el
catálogo, no este paquete.

Directorio exento del guardián de imports de modelo
(`tests/test_agnosticism.py`, `EXEMPT_DIR_NAMES`), como `cognition/`,
`intent/` y `arms/health/`.
"""

from __future__ import annotations
