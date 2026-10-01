"""Guardar la transcripción de un audio para no repetirla.

Medido: transcribir siete minutos de consulta cuesta unos 105 segundos, y todo
lo que viene después —secciones, campos, léxico, cotejo— 0,8. Así que lo único
que merece guardarse es la transcripción; lo demás se recalcula, y de paso un
ajuste del pack se aplica al reanudar en vez de quedar congelado en el archivo.

Cifrada, con la misma clave que el resumen: una transcripción es lo más
sensible de la consulta. Y con expiración, porque no tiene por qué quedarse ahí
para siempre.

La identidad es la huella del **contenido** del audio más el modelo y el umbral
con los que se transcribió: cambiar de modelo tiene que invalidar lo guardado,
porque la transcripción sería otra.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from aida.voice.diarize import TranscriptSegment


def transcript_fingerprint(audio_path: Path, *, model: str, no_speech_threshold: float) -> str:
    """Huella estable de «este audio con este modelo». `blake2b` y no `hash()`,
    por lo mismo que en el resto del proyecto: `hash()` cambia entre procesos."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(audio_path.read_bytes())
    digest.update(f"|{model}|{no_speech_threshold}".encode())
    return digest.hexdigest()


class TranscriptCache:
    """Solo añade y lee. No sabe de consultas ni de pacientes: su clave es el
    audio, no el sujeto, así que el nombre del archivo no revela a quién
    pertenece."""

    def __init__(self, directory: Path, *, key: bytes, ttl_seconds: int = 86400) -> None:
        self._directory = directory
        self._fernet = Fernet(key)
        self._ttl = ttl_seconds

    def _path_for(self, fingerprint: str) -> Path:
        return self._directory / f"{fingerprint}.transcript"

    def store(self, fingerprint: str, segments: tuple[TranscriptSegment, ...]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        crudo = json.dumps(
            [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
            ensure_ascii=False,
        )
        self._path_for(fingerprint).write_bytes(self._fernet.encrypt(crudo.encode("utf-8")))

    def load(self, fingerprint: str) -> tuple[TranscriptSegment, ...] | None:
        """`None` si no está, si caducó —y entonces lo borra— o si la clave no
        lo descifra. Nunca a medias: una transcripción rota se vuelve a hacer."""
        ruta = self._path_for(fingerprint)
        if not ruta.exists():
            return None
        if time.time() - ruta.stat().st_mtime > self._ttl:
            ruta.unlink(missing_ok=True)
            return None
        try:
            crudo = json.loads(self._fernet.decrypt(ruta.read_bytes()).decode("utf-8"))
            return tuple(
                TranscriptSegment(
                    start=float(s["start"]) if s["start"] is not None else None,
                    end=float(s["end"]) if s["end"] is not None else None,
                    text=str(s["text"]),
                )
                for s in crudo
            )
        except (InvalidToken, UnicodeDecodeError, ValueError, KeyError, TypeError):
            return None

    def purge_expired(self) -> int:
        borrados = 0
        if not self._directory.exists():
            return 0
        for ruta in sorted(self._directory.glob("*.transcript")):
            if time.time() - ruta.stat().st_mtime > self._ttl:
                ruta.unlink(missing_ok=True)
                borrados += 1
        return borrados


class CachedSTT:
    """Envuelve el STT real y le pone la caché delante. Envoltorio y no un
    cambio dentro de `LocalWhisperSTT` para que la caché se pueda quitar sin
    tocar la transcripción, y para que los tests del STT sigan probando el STT.
    """

    def __init__(
        self,
        stt: object,
        cache: TranscriptCache,
        *,
        model: str,
        no_speech_threshold: float,
    ) -> None:
        self._stt = stt
        self._cache = cache
        self._model = model
        self._no_speech_threshold = no_speech_threshold

    def transcribe(self, audio_path: Path) -> str:
        return " ".join(s.text for s in self.transcribe_segments(audio_path))

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        huella = transcript_fingerprint(
            audio_path, model=self._model, no_speech_threshold=self._no_speech_threshold
        )
        guardada = self._cache.load(huella)
        if guardada is not None:
            return guardada
        segments: tuple[TranscriptSegment, ...] = self._stt.transcribe_segments(audio_path)  # type: ignore[attr-defined]
        self._cache.store(huella, segments)
        return segments
