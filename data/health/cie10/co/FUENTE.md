# Catálogo CIE-10 de Colombia

- **Archivo original:** `TablaReferencia_CIE10__1.xlsx`, hoja `Table`.
- **Origen:** Tabla de Referencia CIE-10 de MinSalud / SISPRO (Colombia).
- **Fecha de actualización que declara el propio archivo:** 2026-09-15.
- **Convertido el:** 2026-09-26, con `scripts/convert_cie10_co.py`.
- **Códigos:** 12 634, todos con `Habilitado: SI`, todos de cuatro caracteres.
- **Qué columna se conserva:** `Codigo` y `Nombre` (el nombre específico). La
  columna `Descripcion` es la categoría de tres caracteres y no se usa: alarga
  el texto cotejado y el coseno castiga la longitud (medido en el plan 59).

Este catálogo **no** se funde con el de `data/health/cie10/` (Chile:
`icdcode.info` + `deis.cl`). Comparten 12 162 códigos y **3 990 de ellos tienen
descripción distinta**. Cuál se usa lo declara el pack, no el código.
