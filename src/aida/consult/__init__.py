"""El brazo de consulta: transcribe, coteja contra la historia y contra un
catálogo de códigos diagnósticos, y entrega un resumen en Markdown. No
diagnostica ni recomienda — organiza y estructura.

No implementa `ToolPort`: transcribir y leer un documento no son acciones
que un agente delibere con utilidad, son pasos fijos de una tubería que se
ejecuta siempre igual. Forzarlos a través del protocolo de herramientas
sería estirarlo hasta que quepa. `ToolPort` sigue sin implementación tras
esta unidad.

Directorio exento del guardián de imports de modelo
(`tests/test_agnosticism.py`, `EXEMPT_DIR_NAMES`), igual que `cognition/` e
`intent/`: aquí vive la dependencia con STT y OCR locales.
"""

from __future__ import annotations
