# Conjunto de referencia para calibrar la clasificación por apartados

Qué apartado le corresponde a cada frase, etiquetado a mano. Es lo que permite
decir «acierta 38 de 72» en vez de «se ve mejor», y lo que hace comparable
cualquier ajuste futuro de `sections.yaml`.

**Solo consultas ficticias.** Estas son frases clínicas completas en un archivo
del repositorio: con una consulta real, esto sería una filtración. El contenido
de `consulta-01-secciones.csv` proviene de un guion inventado y dictado por el
propio autor.

Volcado **después** del plan 74 (modelo de transcripción nuevo y partición de
frases unificada entre nube y local): las 122 frases son las que produce el
pipeline actual, no las 72 de antes de esa unidad — etiquetar antes habría
sido etiquetar un material que iba a desaparecer.

| Archivo | Origen | Frases | Etiquetado |
|---|---|---|---|
| `consulta-01-secciones.csv` | guion SUJETO-01, dictado 2026-09-29, transcrito y segmentado 2026-10-01 (post plan 74) | 122 | a mano, por el autor |
