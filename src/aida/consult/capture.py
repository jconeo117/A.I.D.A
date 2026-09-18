"""Captura de texto desde audio (STT) y desde imagen/PDF (OCR), las dos
locales, las dos sin nube — decisión del debate de alcance de esta unidad.

Ninguna de las dos es determinista bit a bit entre ejecuciones (a
diferencia del resto del motor): no importa — lo que se audita es la
transcripción EXACTA que entró al sistema, vía `content`/`digest` del
journal (`events.py`), no la promesa de que repetir la transcripción dé
lo mismo dos veces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TranscriptionError(RuntimeError):
    """El audio no se pudo transcribir."""


class LocalWhisperSTT:
    """Envuelve `faster_whisper.WhisperModel`. Se recibe el modelo ya
    cargado, no se construye aquí: cargar un modelo de Whisper es lento y
    con estado (pesos en memoria); un test lo sustituye por un doble sin
    tocar disco ni descargar nada."""

    def __init__(self, model: object) -> None:
        self._model: Any = model

    def transcribe(self, audio_path: Path) -> str:
        try:
            segments, _info = self._model.transcribe(str(audio_path), language="es")
            return " ".join(segment.text.strip() for segment in segments)
        except Exception as exc:
            raise TranscriptionError(f"no se pudo transcribir {audio_path}: {exc}") from exc


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


def build_local_capture() -> tuple[LocalWhisperSTT, LocalTesseractOCR]:
    """Construye STT y OCR locales. Vive aquí y no en `run.py` porque este
    directorio es la costura declarada hacia los runtimes de modelo: el
    guardián de G4 exime `consult/`, y `run.py` no."""
    import pytesseract
    from faster_whisper import WhisperModel

    return LocalWhisperSTT(WhisperModel("small")), LocalTesseractOCR(pytesseract)

