# A.I.D.A.

**Asistente Inteligente Determinista y Autónomo.** Un secretario virtual cuya
motivación es aritmética, no inferencia: un motor determinista decide *si*
actuar, *qué* hacer y *si está autorizado*. El modelo de lenguaje queda
reducido a dos tareas de borde — entender lo que escribe el usuario y redactar
la respuesta—, y nunca decide ni cambia el estado.

Si el proveedor de modelo se cae, AIDA sigue trabajando: solo pierde la voz.

## Por qué está hecho así

Cuatro propiedades que el diseño sostiene, y que la batería vigila:

- **El motor es la única fuente de verdad.** El modelo propone; proponer no es
  decidir.
- **Determinismo reproducible.** Misma semilla y mismas entradas producen el
  mismo registro de eventos, byte a byte, entre procesos distintos. Nada itera
  un conjunto sin ordenarlo, los desempates son por identificador y la
  aleatoriedad se deriva por clave con `blake2b`, nunca de un flujo compartido.
- **El motor no sabe en qué organización corre.** Cero vocabulario de dominio
  en `src/aida/`: lo que nombra el mundo vive en un *pack* de datos. Hay un
  test que lo comprueba analizando el árbol sintáctico.
- **Ninguna acción irreversible sin autorización explícita**, por herramienta y
  por alcance, sin modo experto que lo desactive.

## Cómo se organiza

| Directorio | Qué contiene | Cambia con |
|---|---|---|
| `src/aida/engine/` | El motor: ontología, estado, registro de eventos, pack, disposición, deliberación, reloj, autoridad | nunca |
| `src/aida/agents/` | La deliberación por utilidad | nunca |
| `src/aida/cognition/`, `src/aida/intent/` | La voz y el reconocimiento de intención | el proveedor de modelo |
| `src/aida/arms/<brazo>/` | Verticales de dominio; `health` es el primero | el vertical |
| `src/aida/voice/` | Audio local: micrófono, corte por silencio, quién habla | — |
| `packs/` | Los datos que nombran el mundo: necesidades, acciones, textos, catálogos | el despliegue |
| `data/` | Catálogos externos, como el CIE-10 del brazo de salud | — |

El primer brazo es un **asistente de consultorio**: transcribe la consulta con
un modelo de voz local, la clasifica en secciones clínicas, extrae de forma
determinista lo que se puede extraer sin interpretar —duración, lateralidad,
quién lo dijo—, coteja contra un catálogo CIE-10 y **pide al profesional que
confirme, edite o rechace cada elemento antes de registrar nada**. Nada sale
del equipo: el audio, la historia y el resumen se procesan en local, y lo que
se guarda va cifrado.

## Instalación

```bash
uv sync                      # motor y CLI
uv sync --extra capture      # además, transcripción y OCR locales
uv sync --extra diarization  # además, identificación de hablantes
```

La transcripción necesita el modelo de voz descargado una vez, y el OCR el
binario de `tesseract` con el idioma correspondiente instalado en el sistema.
Para comprobar qué falta:

```bash
uv run aida --check-capture
```

## Uso

```bash
uv run aida --version
uv run aida --pack packs/_plantilla/pack.yaml            # carga y valida un pack
uv run aida --pack <pack> --explain                      # traza causal de una decisión
uv run aida --pack <pack> --converse                     # conversación por teclado
uv run aida --pack <pack> --converse --mic               # conversación por voz

uv run aida --consultation --audio <audio> --history <documento> \
            --segment --extract --review --print-model   # el brazo de consultorio
uv run aida --review-stats                               # métricas de las revisiones
```

## Desarrollo

```bash
uv run pytest        # 1085 pruebas
uv run mypy src tests  # estricto
uv run ruff check
```

La batería tarda menos de 30 segundos y no toca micrófono, red ni modelos
reales: todo lo pesado se inyecta y se sustituye por dobles. Las dos
instalaciones —con extras y sin ellos— deben quedar en verde.

## Estado

El motor determinista está terminado y validado: ontología, estado, registro
con journal, pack con catálogo cerrado, disposición y aprendizaje,
deliberación con traza causal, reloj híbrido, persistencia con reconciliación,
panel de salud y capa de autoridad. Sobre él funcionan el reconocimiento de
intención sin modelo, la voz con tres proveedores intercambiables y el brazo
de consultorio descrito arriba.

Falta el puerto de herramientas, la memoria episódica y un reloj de fondo que
avance el tiempo entre mensajes.

## Licencia

Sin licencia pública por ahora. Todos los derechos reservados.
