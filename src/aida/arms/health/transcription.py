"""La transcripción como costura: en la nube cuando se puede, en local cuando no.

AIDA Health corre en el portátil que tenga el profesional, así que el trabajo
pesado va a un proveedor de nube. Pero un consultorio sin internet, o un
proveedor caído, no pueden dejar la consulta a medias: es la misma regla que ya
gobierna el modelo de lenguaje desde U7 —si el proveedor cae, AIDA sigue
trabajando y solo pierde velocidad—.

`FallbackTranscription` es la puerta única de esa política: intenta la nube y,
ante **cualquier** fallo, transcribe en local y lo dice. Quien la usa no sabe
cuál de las dos respondió.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aida.arms.health.capture import split_into_sentences
from aida.voice.diarize import TranscriptSegment


class TranscriptionProvider(Protocol):
    """Un transcriptor, venga de donde venga.

    Contrato que hace posible cambiar de proveedor sin cambiar el resto: recibe
    la ruta de un audio y devuelve segmentos de texto. Los tiempos son
    opcionales desde el plan 74 —hay modelos de nube que no los devuelven—,
    pero quien los necesite de verdad (la diarización) tiene que fallar
    explícitamente ante su ausencia, no recibir ceros que parecen datos.
    """

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]: ...


@dataclass(frozen=True, slots=True)
class FallbackNotice:
    """Por qué se usó el respaldo. Se devuelve en vez de escribirse por pantalla:
    quien llama decide si lo imprime, lo registra o lo ignora."""

    reason: str


class FallbackTranscription:
    """Nube primero, local después. Y nunca las dos: si la nube responde, el
    local no se toca, que es lo que evita pagar dos veces y tardar el doble."""

    def __init__(
        self,
        primary: TranscriptionProvider,
        fallback: TranscriptionProvider,
        *,
        on_fallback: Callable[[FallbackNotice], None] | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._on_fallback = on_fallback

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        try:
            return self._primary.transcribe_segments(audio_path)
        except Exception as exc:
            # Cualquier fallo lleva al respaldo: red, cuota, tiempo agotado o un
            # error del proveedor. Distinguirlos aquí solo daría más formas de
            # quedarse a medias.
            if self._on_fallback is not None:
                self._on_fallback(FallbackNotice(reason=str(exc)))
            return self._fallback.transcribe_segments(audio_path)

    def transcribe(self, audio_path: Path) -> str:
        return " ".join(s.text for s in self.transcribe_segments(audio_path))


class CloudTranscriptionError(RuntimeError):
    """La nube respondió algo que no se puede usar. Lleva al respaldo local."""


class CloudWhisperSTT:
    """Transcripción en la nube por HTTP, con el cliente inyectado.

    **Aviso de verificación, igual que en U7 y en el plan de la voz de salida:**
    la forma exacta de la petición y el nombre del campo donde viene el texto se
    comprueban contra la documentación vigente del proveedor **al implementar**.
    Lo que sigue es el contrato que debe cumplir, no una copia garantizada de su
    API: el architect no tiene acceso a la red para confirmarlo.

    Lo que sí es contrato y no se negocia:
      - pide segmentos con tiempos cuando `timestamps=True`, porque la
        diarización y el reparto por frases se apoyan en ellos; sin tiempos
        —algunos modelos no los devuelven— parte el `text` en frases con el
        mismo criterio que el camino local (`split_into_sentences`), y los
        segmentos resultantes llevan `start`/`end` en `None`;
      - fija el idioma y la **versión del modelo** desde los datos del pack, para
        que dos mediciones separadas por semanas sigan siendo comparables;
      - cualquier fallo se levanta como excepción: la decisión de caer al
        respaldo no es suya, es de `FallbackTranscription`.
    """

    def __init__(
        self,
        client: object,
        *,
        model: str,
        language: str,
        prompt: str = "",
        formats: frozenset[str] = frozenset(),
        timestamps: bool = True,
        abbreviations: frozenset[str] = frozenset(),
    ) -> None:
        self._client = client
        self._model = model
        self._language = language
        self._prompt = prompt
        self._formats = formats
        self._timestamps = timestamps
        self._abbreviations = abbreviations

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        # Antes de abrir el archivo: subir diez megabytes para que el proveedor
        # los rechace cuesta minutos, y la extensión ya lo dice. `formats` vacío
        # significa «no se declaró ninguna restricción», no «ninguno vale».
        if self._formats and audio_path.suffix.lower() not in self._formats:
            raise CloudTranscriptionError(
                f"{audio_path.suffix or 'sin extensión'} no está entre los formatos "
                f"que acepta el proveedor ({', '.join(sorted(self._formats))})"
            )
        datos: dict[str, str] = {
            "model": self._model,
            "language": self._language,
        }
        if self._timestamps:
            datos["response_format"] = "verbose_json"
            datos["timestamp_granularities[]"] = "segment"
        else:
            datos["response_format"] = "json"
        if self._prompt:
            datos["prompt"] = self._prompt
        with audio_path.open("rb") as audio:
            response = self._client.post(  # type: ignore[attr-defined]
                "/v1/audio/transcriptions",
                files={"file": (audio_path.name, audio, "audio/wav")},
                data=datos,
            )
        response.raise_for_status()
        crudo = response.json()
        if not self._timestamps:
            texto = str(crudo.get("text", "")).strip()
            if not texto:
                raise CloudTranscriptionError(f"la respuesta no trae texto: {list(crudo)}")
            return split_into_sentences(texto, abbreviations=self._abbreviations)
        segmentos = crudo.get("segments")
        if not segmentos:
            raise CloudTranscriptionError(f"la respuesta no trae segmentos: {list(crudo)}")
        return tuple(
            TranscriptSegment(
                start=float(s["start"]), end=float(s["end"]), text=str(s["text"]).strip()
            )
            for s in segmentos
            if str(s.get("text", "")).strip()
        )
