"""Punto de entrada del ejecutable (plan 78b): PyInstaller necesita un script, y usar
`run.py` como tal lo cargaría dos veces, como `__main__` y como `aida.run`."""

from aida.run import main

raise SystemExit(main())
