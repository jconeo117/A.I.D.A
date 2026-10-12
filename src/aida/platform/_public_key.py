"""La clave pública con la que AIDA verifica las licencias (plan 100).

La escribe `tools/licencias.py keygen`. Es código y no un archivo de datos a propósito:
un archivo de datos se sustituye por otro sin tocar nada, y entonces cualquiera firmaría
sus propias licencias. Sin clave (`None`), ninguna licencia es válida: no hay modo
desarrollo que se la salte.
"""

PUBLIC_KEY: str | None = None
