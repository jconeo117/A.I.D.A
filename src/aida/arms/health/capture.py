"""Captura de texto desde audio (STT) y desde imagen/PDF (OCR), las dos
locales, las dos sin nube — decisión del debate de alcance de esta unidad.

Ninguna de las dos es determinista bit a bit entre ejecuciones (a
diferencia del resto del motor): no importa — lo que se audita es la
transcripción EXACTA que entró al sistema, vía `content`/`digest` del
journal (`events.py`), no la promesa de que repetir la transcripción dé
lo mismo dos veces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aida.resources import data_root
from aida.sentences import split_sentences
from aida.voice.diarize import TranscriptSegment


class TranscriptionError(RuntimeError):
    """El audio no se pudo transcribir."""


class NoSpeechError(TranscriptionError):
    """El audio no trae voz. Hereda de `TranscriptionError` para que quien ya
    capturaba fallos de transcripción no se quede fuera, pero es un caso
    distinto: aquí el modelo funcionó y lo que falta es el habla."""


class ModelUnavailableError(RuntimeError):
    """El modelo de transcripción no se pudo cargar: no está la copia local, o
    el dispositivo declarado no sirve en esta máquina."""


_RAIZ = data_root()
_CAPTURA_POR_DEFECTO = _RAIZ / "packs" / "health" / "consultorio" / "capture.yaml"


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    """Qué modelo, en qué dispositivo y con qué umbral de voz. En datos y no en
    código: cambiarlos es una decisión de despliegue, y el guardián de
    vocabulario no admite nombres de hardware dentro de `src/`."""

    model: str
    device: str
    no_speech_threshold: float
    abbreviations: tuple[str, ...]
    provider: str
    cloud_model: str
    cloud_language: str
    cloud_base_url: str
    timeout_seconds: float
    compute_type: str
    connect_timeout_seconds: float
    cloud_formats: tuple[str, ...]
    transcription_prompt: str
    cloud_timestamps: bool
    # Opcional: la batería construye ajustes a mano y no lo necesita.
    model_download_mb: int = 0


def load_capture_settings(path: Path | None = None) -> CaptureSettings:
    """Carga y valida. Revienta con la clave que falta: un ajuste de captura a
    medias es peor que ninguno, porque se descubre a mitad de una consulta."""
    ruta = path if path is not None else _CAPTURA_POR_DEFECTO
    raw = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ruta} debe ser un mapeo YAML")
    faltan = [
        k
        for k in (
            "model",
            "device",
            "no_speech_threshold",
            "abbreviations",
            "provider",
            "cloud_model",
            "cloud_language",
            "cloud_base_url",
            "timeout_seconds",
            "compute_type",
            "connect_timeout_seconds",
            "cloud_formats",
            "transcription_prompt",
            "cloud_timestamps",
        )
        if k not in raw
    ]
    if faltan:
        raise ValueError(f"faltan claves en {ruta}: {faltan}")
    umbral = raw["no_speech_threshold"]
    if not isinstance(umbral, int | float) or isinstance(umbral, bool):
        raise ValueError(f"'no_speech_threshold' debe ser numérico en {ruta}")
    abreviaturas = raw["abbreviations"]
    if not isinstance(abreviaturas, list) or not all(isinstance(a, str) for a in abreviaturas):
        raise ValueError(f"'abbreviations' debe ser una lista de cadenas en {ruta}")
    proveedor = str(raw["provider"]).lower()
    if proveedor not in ("cloud", "local"):
        raise ValueError(f"'provider' debe ser 'cloud' o 'local' en {ruta}")
    espera = raw["timeout_seconds"]
    if not isinstance(espera, int | float) or isinstance(espera, bool) or espera <= 0:
        raise ValueError(f"'timeout_seconds' debe ser un número positivo en {ruta}")
    espera_conexion = raw["connect_timeout_seconds"]
    if (
        not isinstance(espera_conexion, int | float)
        or isinstance(espera_conexion, bool)
        or espera_conexion <= 0
    ):
        raise ValueError(f"'connect_timeout_seconds' debe ser un número positivo en {ruta}")
    formatos = raw["cloud_formats"]
    if (
        not isinstance(formatos, list)
        or not formatos
        or not all(isinstance(f, str) and f.strip() for f in formatos)
    ):
        raise ValueError(f"'cloud_formats' debe ser una lista no vacía de extensiones en {ruta}")
    prompt = raw["transcription_prompt"]
    if not isinstance(prompt, str):
        raise ValueError(f"'transcription_prompt' debe ser una cadena en {ruta}")
    compute_type = raw["compute_type"]
    if not isinstance(compute_type, str) or not compute_type.strip():
        raise ValueError(f"'compute_type' debe ser una cadena no vacía en {ruta}")
    timestamps = raw["cloud_timestamps"]
    if not isinstance(timestamps, bool):
        raise ValueError(f"'cloud_timestamps' debe ser un booleano en {ruta}")
    return CaptureSettings(
        model=str(raw["model"]),
        device=str(raw["device"]),
        no_speech_threshold=float(umbral),
        abbreviations=tuple(a.lower() for a in abreviaturas),
        provider=proveedor,
        cloud_model=str(raw["cloud_model"]),
        cloud_language=str(raw["cloud_language"]),
        cloud_base_url=str(raw["cloud_base_url"]),
        timeout_seconds=float(espera),
        compute_type=str(compute_type).strip(),
        connect_timeout_seconds=float(espera_conexion),
        cloud_formats=tuple(str(f).lower().strip() for f in formatos),
        transcription_prompt=str(prompt),
        cloud_timestamps=timestamps,
        model_download_mb=int(raw.get("model_download_mb", 0)),
    )


def split_into_sentences(
    text: str,
    *,
    abbreviations: frozenset[str],
    start: float | None = None,
    end: float | None = None,
) -> tuple[TranscriptSegment, ...]:
    """Qué es una frase, para los dos caminos de transcripción.

    Existía solo dentro del camino local (plan 58, `I-130`), mientras el de
    nube confiaba en los trozos que cortaba el proveedor: dos criterios
    distintos para la misma pregunta, y la clase de duplicación que este
    proyecto trata como invariante.

    Con `start`/`end`, el tramo de tiempo se reparte entre las frases en
    proporción a lo que ocupa cada una — la diarización empareja por
    solapamiento, así que necesita tiempos distintos por frase; cuál sea el
    reparto exacto importa menos que el hecho de que sea determinista.

    `start` y `end` son opcionales porque hay modelos que no devuelven
    tiempos. Sin ellos, los segmentos salen con `start` y `end` a `None`: **no
    se inventan tiempos plausibles**. Quien empareje por solapamiento — la
    diarización — tiene que romper al pedirlos, no recibir ceros que parecen
    datos y emparejar cualquier cosa.
    """
    frases = split_sentences(text, abbreviations=abbreviations)
    if start is None or end is None:
        return tuple(TranscriptSegment(start=None, end=None, text=frase) for frase in frases)
    if len(frases) <= 1:
        return (TranscriptSegment(start=start, end=end, text=text),)
    total = sum(len(frase) for frase in frases)
    duracion = end - start
    trozos: list[TranscriptSegment] = []
    consumido = 0
    for frase in frases:
        inicio = start + duracion * consumido / total
        consumido += len(frase)
        fin = start + duracion * consumido / total
        trozos.append(TranscriptSegment(start=round(inicio, 3), end=round(fin, 3), text=frase))
    return tuple(trozos)


class LocalWhisperSTT:
    """Envuelve `faster_whisper.WhisperModel`. Se recibe el modelo ya
    cargado, no se construye aquí: cargar un modelo de Whisper es lento y
    con estado (pesos en memoria); un test lo sustituye por un doble sin
    tocar disco ni descargar nada."""

    def __init__(
        self,
        model: object,
        *,
        no_speech_threshold: float = 0.6,
        abbreviations: frozenset[str] = frozenset(),
        prompt: str = "",
    ) -> None:
        self._model: Any = model
        self._no_speech_threshold = no_speech_threshold
        self._abbreviations = abbreviations
        self._prompt = prompt

    def transcribe(self, audio_path: Path) -> str:
        return " ".join(segment.text for segment in self._speech_segments(audio_path))

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        return self._speech_segments(audio_path)

    def _speech_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        """Puerta única de las dos formas de transcribir: lo que se descarta por
        no ser habla no puede depender de cuál se llame.

        `vad_filter` se deja apagado a propósito. Medido sobre la grabación de
        referencia, activarlo empeora el resultado —une frases y transcribe
        peor—, mientras que filtrar por `no_speech_prob` quita lo inventado sin
        tocar una palabra de lo dicho.

        El `initial_prompt` no es una sugerencia de vocabulario: es lo que hace
        que el modelo **puntúe**. Medido sobre una consulta real de cinco
        minutos, sin él salen 1 punto y 143 comas —y la partición por frases del
        plan 58 se queda sin nada que partir—; con él, 89 puntos y frases de 41
        caracteres en vez de tramos de 86. Cuesta un 16 % más de tiempo.

        Y quedarse sin segmentos no es una transcripción vacía: es que no había
        voz. Decirlo es más útil que devolver una consulta en blanco."""
        try:
            segments, _info = self._model.transcribe(
                str(audio_path),
                language="es",
                initial_prompt=self._prompt or None,
            )
            kept = tuple(
                TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
                for s in segments
                if s.no_speech_prob <= self._no_speech_threshold and s.text.strip()
            )
        except Exception as exc:
            raise TranscriptionError(f"no se pudo transcribir {audio_path}: {exc}") from exc
        if not kept:
            raise NoSpeechError(f"no se detectó voz en {audio_path}")
        return tuple(
            trozo
            for segmento in kept
            for trozo in split_into_sentences(
                segmento.text,
                abbreviations=self._abbreviations,
                start=segmento.start,
                end=segmento.end,
            )
        )


class OCRError(RuntimeError):
    """El documento no se pudo leer."""


class LocalTesseractOCR:
    """Envuelve `pytesseract`. Requiere el binario `tesseract-ocr` del
    sistema operativo — no lo instala `uv`, es un requisito de despliegue."""

    def __init__(self, pytesseract_module: object) -> None:
        self._pytesseract: Any = pytesseract_module

    def read(self, image_path: Path) -> str:
        """Texto de una imagen o de un PDF. Un PDF se rasteriza página a
        página con `pypdfium2` (PDFium embebido, sin binario del sistema)
        y cada página pasa por el mismo `image_to_string`; las páginas se
        unen con una línea en blanco para que el cotejo de códigos vea el
        documento entero, no solo la primera."""
        try:
            if image_path.suffix.lower() == ".pdf":
                return self._read_pdf(image_path)
            from PIL import Image

            with Image.open(image_path) as img:
                raw: str = self._pytesseract.image_to_string(img, lang="spa")
                return raw.strip()
        except Exception as exc:
            raise OCRError(f"no se pudo leer {image_path}: {exc}") from exc

    def _read_pdf(self, pdf_path: Path) -> str:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        pages: list[str] = []
        for index in range(len(document)):
            bitmap = document[index].render(scale=2)
            raw: str = self._pytesseract.image_to_string(bitmap.to_pil(), lang="spa")
            pages.append(raw.strip())
        return "\n\n".join(pages).strip()


def build_local_capture(
    settings: CaptureSettings | None = None,
) -> tuple[LocalWhisperSTT, LocalTesseractOCR]:
    """Construye STT y OCR locales. Vive aquí y no en `run.py` porque este
    directorio es la costura declarada hacia los runtimes de modelo: el guardián
    de G4 exime `arms/`, y `run.py` no.

    Modelo y dispositivo se **declaran** en el pack, no se adivinan:
    `device="auto"` elegía la GPU en cuanto la veía y reventaba si el runtime no
    coincidía (I-127). Y el modelo se carga solo de la copia local: por defecto,
    cargarlo consulta Hugging Face en cada ejecución y se cuelga si la red no
    responde (I-129) — «local, sin nube» tiene que valer también al arrancar."""
    from faster_whisper import WhisperModel

    ajustes = settings if settings is not None else load_capture_settings()
    try:
        model = WhisperModel(
            ajustes.model,
            device=ajustes.device,
            compute_type=ajustes.compute_type,
            local_files_only=True,
        )
    except Exception as exc:
        raise ModelUnavailableError(
            f"no se pudo cargar {ajustes.model!r} en {ajustes.device!r}: {exc}. "
            "Si falta la copia local, descárgala una vez con: uv run python -c "
            f'"from faster_whisper import WhisperModel; WhisperModel(\'{ajustes.model}\')"'
        ) from exc
    return (
        LocalWhisperSTT(
            model,
            no_speech_threshold=ajustes.no_speech_threshold,
            abbreviations=frozenset(ajustes.abbreviations),
            prompt=ajustes.transcription_prompt,
        ),
        build_document_reader(),
    )


def build_document_reader() -> LocalTesseractOCR:
    """El lector de documentos, separado del modelo de transcripción: AIDA puede
    arrancar sin modelo local, solo con la nube (plan 78), pero sin leer la historia
    no hay códigos que ya constaban.

    `AIDA_TESSERACT` apunta al ejecutable cuando no está en el PATH: en Windows el
    instalador lo deja en su propia carpeta (plan 78b)."""
    import os

    import pytesseract

    ruta = os.environ.get("AIDA_TESSERACT")
    if ruta:
        pytesseract.pytesseract.tesseract_cmd = ruta
    return LocalTesseractOCR(pytesseract)


def document_reader_version() -> str | None:
    """La versión del lector si está y lee español; `None` si le falta el español.
    Revienta si no está: quien llama decide cómo decirlo."""
    import pytesseract

    build_document_reader()  # fija la ruta declarada en AIDA_TESSERACT
    if "spa" not in pytesseract.get_languages():
        return None
    return str(pytesseract.get_tesseract_version())


def _model_repo(model: str) -> str:
    from faster_whisper import utils

    modelos = getattr(utils, "_MODELS", {})
    return str(modelos.get(model, model))


def local_model_available(settings: CaptureSettings | None = None) -> bool:
    """Si la copia del modelo local está en disco, sin cargarlo (cargar tarda) y sin
    salir a la red."""
    from faster_whisper.utils import download_model

    ajustes = settings if settings is not None else load_capture_settings()
    try:
        download_model(ajustes.model, local_files_only=True)
    except Exception:
        return False
    return True


def local_model_bytes(settings: CaptureSettings | None = None) -> int:
    """Cuánto del modelo hay ya en disco: la descarga no informa de su progreso, así
    que la pantalla lo lee del tamaño de su carpeta en la caché de Hugging Face."""
    from huggingface_hub import constants

    ajustes = settings if settings is not None else load_capture_settings()
    repo = _model_repo(ajustes.model).replace("/", "--")
    carpeta = Path(constants.HF_HUB_CACHE) / f"models--{repo}"
    if not carpeta.is_dir():
        return 0
    archivos = sorted(carpeta.rglob("*"))
    return sum(f.stat().st_size for f in archivos if f.is_file() and not f.is_symlink())


def download_local_model(settings: CaptureSettings | None = None) -> None:
    """Descarga el modelo local una vez. Es lo único de AIDA que sale a la red sin
    ser la nube de transcripción: lo pide el usuario con un botón (plan 78).

    Sin el transporte `xet`: con él, el archivo grande no aparece en disco hasta el
    final —la barra se quedaba en cero— y en la prueba de la 78 no avanzó en seis
    minutos. La descarga clásica escribe a medida que llega, y es lo que mide
    `local_model_bytes`. Se lee en cada llamada, así que basta con fijarlo aquí."""
    from faster_whisper.utils import download_model
    from huggingface_hub import constants

    constants.HF_HUB_DISABLE_XET = True
    ajustes = settings if settings is not None else load_capture_settings()
    download_model(ajustes.model)


def capture_diagnosis(
    *, need_microphone: bool = True, need_model: bool = True
) -> tuple[tuple[str, bool], ...]:
    """Qué hay y qué falta para poder capturar. Devuelve datos —texto y si está
    bien—, no imprime: quien llama decide el formato y el código de salida.

    Comprueba **todo** aunque algo falle: un diagnóstico que se rinde en el
    primer error obliga a ejecutarlo cuatro veces para enterarse de cuatro
    cosas.

    Vive aquí y no en `run.py` por la misma razón que `build_local_capture`.

    La aplicación pide `need_microphone=False` —graba con el navegador, no con
    `sounddevice`— y `need_model=False`: el modelo local se comprueba aparte, porque
    sin él se puede trabajar con la nube (plan 78)."""
    import importlib.util

    lineas: list[tuple[str, bool]] = []
    modulos = ("faster_whisper", "sounddevice", "pytesseract")
    for nombre in modulos if need_microphone else ("faster_whisper", "pytesseract"):
        lineas.append((f"módulo {nombre}", importlib.util.find_spec(nombre) is not None))

    try:
        ajustes = load_capture_settings()
        etiqueta = f"modelo {ajustes.model} en {ajustes.device}"
    except Exception as exc:
        ajustes = None
        etiqueta = f"ajustes de captura: {exc}"
    if ajustes is None:
        lineas.append((etiqueta, False))
    elif need_model:
        try:
            build_local_capture(ajustes)
            lineas.append((etiqueta, True))
        except Exception as exc:
            lineas.append((f"{etiqueta}: {exc}", False))

    try:
        version = document_reader_version()
        lineas.append((f"tesseract {version}, español", version is not None))
    except Exception as exc:
        lineas.append((f"tesseract: {exc}", False))

    if not need_microphone:
        return tuple(lineas)
    try:
        import sounddevice

        entrada = sounddevice.query_devices(kind="input")
        lineas.append((f"entrada de audio: {entrada['name']}", True))
    except Exception as exc:
        lineas.append((f"entrada de audio: {exc}", False))

    return tuple(lineas)
