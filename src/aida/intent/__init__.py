"""Puerto de intención: texto de entrada contra el catálogo cerrado del pack.

Bolsa de palabras y coseno, no un modelo entrenado — ver el docstring de
`classify.py` para el porqué. El directorio está exento del guardián de
imports de modelo (`tests/test_agnosticism.py`, `EXEMPT_DIR_NAMES`) para
cuando haga falta cambiar de implementación; hoy no importa ninguno.
"""

from __future__ import annotations
