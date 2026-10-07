"""La consulta en vivo (plan 77b): mientras se graba, AIDA transcribe por enunciados,
ordena, propone códigos y redacta el borrador en las pausas.

Todo lo de aquí es **provisional**. El audio completo se guarda siempre, entero, y al
terminar se analiza con el camino de siempre (`Consultation.prepare`); de lo del vivo
solo sobrevive lo que hizo el profesional: los campos que editó y los códigos que
quitó. Por eso el análisis en vivo va a un registro propio y no al de la consulta: se
repite muchas veces y no es una entrada del sistema.

Hilos, no `asyncio`: es un adaptador, como `app.py`, y el motor sigue síncrono.
"""

from __future__ import annotations

import queue
import threading
import time
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from aida.arms.health.consent import ConsentRecord
from aida.arms.health.consultation import Consultation, PreparedConsultation
from aida.arms.health.match import CodeOrigin
from aida.arms.health.sections import SectionSpec
from aida.arms.health.segment import is_interrogative
from aida.engine.events import EventLog
from aida.engine.ontology import AgentId

if TYPE_CHECKING:
    from aida.voice.diarize import TranscriptSegment

RATE = 16000
CHUNK = 512
SILENCE_SECONDS = 0.8
MAX_UTTERANCE_SECONDS = 15.0
SENTENCES_PER_DRAFT = 6
QUIET_SECONDS = 8.0
CONTEXT_CHARS = 300


class LiveTranscriber(Protocol):
    def transcribe_segments(
        self, audio_path: Path, *, context: str = ""
    ) -> tuple[TranscriptSegment, ...]: ...


class _StreamClosedError(Exception):
    """El oyente pide un trozo y la sesión ya terminó."""


@dataclass
class _Field:
    spec: SectionSpec
    text: str = ""
    state: str = "waiting"
    updated: float = 0.0


class LiveSession:
    """Una consulta grabándose. Los métodos públicos los llama el servidor desde sus
    hilos; todo lo que comparten va bajo un candado."""

    def __init__(
        self,
        *,
        consultation: Consultation,
        owner: str,
        note_sections: Sequence[SectionSpec],
        history_text: Callable[[], str],
        workdir: Path,
        transcriber: LiveTranscriber | None,
        vad: Callable[[np.ndarray], np.ndarray],
        consent: ConsentRecord | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # Antes de abrir el WAV: sin autorización no se graba ni un trozo (plan 82).
        consultation.require_consent(AgentId(owner), consent)
        self._consent = consent
        self._consultation = consultation
        self._owner = AgentId(owner)
        self._history_text = history_text
        self._workdir = workdir
        self._transcriber = transcriber
        self._vad = vad
        self._clock = clock
        self._lock = threading.Lock()
        self._chunks: queue.Queue[np.ndarray | None] = queue.Queue()
        self._buffer = np.zeros(0, dtype=np.float32)
        # Abierto toda la sesión y cerrado en `stop`: un `with` no cabe aquí.
        self._wav = wave.open(str(workdir / "audio.wav"), "wb")  # noqa: SIM115
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(RATE)
        self._lines: list[tuple[str, str]] = []
        self._fields = {spec.section_id: _Field(spec=spec) for spec in note_sections}
        self._order = tuple(spec.section_id for spec in note_sections)
        self._rejected: set[str] = set()
        self._codes: tuple[dict[str, str], ...] = ()
        self._fresh: str | None = None
        self._history: str | None = None
        self._prepared: PreparedConsultation | None = None
        self._offline = transcriber is None
        self._paused = False
        self._stopped = False
        self._drafting = False
        self._lines_at_draft = 0
        self._last_speech = clock()
        self._version = 0
        self._threads: list[threading.Thread] = []

    @property
    def audio_path(self) -> Path:
        return self._workdir / "audio.wav"

    @property
    def consent(self) -> ConsentRecord | None:
        """La autorización con la que se abrió: el análisis final del audio completo
        la necesita para pasar por la misma puerta."""
        return self._consent

    def start(self) -> None:
        for objetivo in (self._read_history, self._listen, self._watch):
            hilo = threading.Thread(target=objetivo, daemon=True)
            hilo.start()
            self._threads.append(hilo)

    # --- lo que llama el servidor -------------------------------------------------

    def feed(self, pcm: bytes) -> None:
        """PCM de 16 bits, 16 kHz, mono, tal como lo manda el navegador. Se guarda
        entero y se trocea para el oyente. En pausa no llega nada, y si llega, no se
        guarda: lo dicho en pausa no existe en ningún sitio."""
        with self._lock:
            if self._paused or self._stopped:
                return
            self._wav.writeframes(pcm)
        muestras = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self._buffer = np.concatenate((self._buffer, muestras))
        while self._buffer.size >= CHUNK:
            self._chunks.put(self._buffer[:CHUNK].copy())
            self._buffer = self._buffer[CHUNK:]

    def pause(self, paused: bool) -> None:
        with self._lock:
            if paused and not self._paused:
                self._lines.append(("pause", ""))
                self._version += 1
            self._paused = paused

    def edit_field(self, field_id: str, text: str) -> bool:
        """Lo que el profesional escribió en un campo queda suyo: la redacción en
        vivo ya no lo toca, y al terminar se conserva sobre el análisis definitivo."""
        with self._lock:
            campo = self._fields.get(field_id)
            if campo is None:
                return False
            campo.text = text
            campo.state = "mine"
            campo.updated = self._clock()
            self._version += 1
            return True

    def set_code(self, code: str, included: bool) -> None:
        with self._lock:
            if included:
                self._rejected.discard(code)
            else:
                self._rejected.add(code)
            self._version += 1

    def stop(self) -> Path:
        """Corta el oyente y cierra el WAV. Lo que el oyente tuviera a medias se
        pierde para el vivo, no para el análisis final: está en el WAV."""
        with self._lock:
            self._stopped = True
        self._chunks.put(None)
        for hilo in self._threads:
            hilo.join(timeout=30)
        with self._lock:
            self._wav.close()
        return self.audio_path

    def mine(self) -> Mapping[str, str]:
        with self._lock:
            return {
                field_id: campo.text
                for field_id, campo in sorted(self._fields.items())
                if campo.state == "mine"
            }

    def rejected(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._rejected)

    def history(self) -> str | None:
        with self._lock:
            return self._history

    def snapshot(self, since: int) -> dict[str, Any]:
        """Lo que la pantalla necesita, de una vez: las frases desde `since`, y los
        campos y códigos enteros (son pocos). La pantalla pregunta cada segundo."""
        ahora = self._clock()
        with self._lock:
            return {
                "version": self._version,
                "lines": [
                    {"kind": kind, "text": text} for kind, text in self._lines[since:]
                ],
                "line_count": len(self._lines),
                "fields": [
                    {
                        "id": field_id,
                        "text": self._fields[field_id].text,
                        "state": self._fields[field_id].state,
                        "age": int(ahora - self._fields[field_id].updated),
                    }
                    for field_id in self._order
                ],
                "codes": [
                    {**codigo, "included": codigo["code"] not in self._rejected,
                     "fresh": codigo["code"] == self._fresh}
                    for codigo in self._codes
                ],
                "offline": self._offline,
                "paused": self._paused,
            }

    # --- los hilos ------------------------------------------------------------------

    def _read_history(self) -> None:
        """El documento se lee una vez, al empezar. Si no se puede leer, el vivo sigue
        sin él —cotejando solo lo de hoy— y el fallo se dice al terminar, donde el
        análisis definitivo lo vuelve a intentar y lo enseña como problema."""
        try:
            texto = self._history_text()
        except Exception:  # frontera: un documento ilegible no tumba la grabación
            texto = ""
        with self._lock:
            self._history = texto

    def _next_chunk(self) -> np.ndarray:
        trozo = self._chunks.get()
        if trozo is None:
            raise _StreamClosedError
        return trozo

    def _listen(self) -> None:
        from aida.voice.listen import SilenceGatedListener, write_wav

        oyente = SilenceGatedListener(chunk_source=self._next_chunk, vad=self._vad)
        numero = 0
        while True:
            try:
                audio = oyente.listen(
                    silence_seconds=SILENCE_SECONDS, max_seconds=MAX_UTTERANCE_SECONDS
                )
            except _StreamClosedError:
                return
            if audio.size == 0:
                continue
            numero += 1
            ruta = write_wav(audio, self._workdir / f"enunciado-{numero}.wav")
            try:
                self._transcribe(ruta)
            finally:
                ruta.unlink(missing_ok=True)

    def _transcribe(self, ruta: Path) -> None:
        if self._transcriber is None:
            return
        with self._lock:
            contexto = " ".join(t for k, t in self._lines if k == "line")[-CONTEXT_CHARS:]
        try:
            segmentos = self._transcriber.transcribe_segments(ruta, context=contexto)
        except Exception:  # frontera: sin red, el vivo se pausa y la grabación sigue
            with self._lock:
                self._offline = True
                self._version += 1
            return
        frases = [s.text.strip() for s in segmentos if s.text.strip()]
        with self._lock:
            self._offline = False
            self._lines.extend(("line", f) for f in frases)
            self._last_speech = self._clock()
            self._version += 1
        if frases:
            self._analyze()

    def _analyze(self) -> None:
        with self._lock:
            texto = "\n".join(t for k, t in self._lines if k == "line")
            historia = self._history or ""
        preparado = self._consultation.analyze(
            owner=self._owner, transcript=texto, history_text=historia, log=EventLog()
        )
        origenes = {
            CodeOrigin.TODAY: "today", CodeOrigin.HISTORY: "history", CodeOrigin.BOTH: "both"
        }
        codigos = tuple(
            {
                "code": m.diagnostic.code,
                "description": m.diagnostic.description,
                "origin": origenes[m.origin],
                "evidence": m.evidence,
            }
            for m in preparado.matches
        )
        with self._lock:
            antes = {c["code"] for c in self._codes}
            nuevos = [c["code"] for c in codigos if c["code"] not in antes]
            if nuevos:
                self._fresh = nuevos[-1]
            self._codes = codigos
            self._prepared = preparado
            self._version += 1

    def _watch(self) -> None:
        """Redacta en las pausas: cuando hay frases nuevas suficientes o la
        conversación se calla un rato, y nunca dos redacciones a la vez."""
        while True:
            time.sleep(1.0)
            with self._lock:
                if self._stopped:
                    return
                pendientes = len([1 for k, _ in self._lines if k == "line"]) - self._lines_at_draft
                callado = self._clock() - self._last_speech >= QUIET_SECONDS
                listo = (
                    self._prepared is not None
                    and not self._drafting
                    and pendientes > 0
                    and (pendientes >= SENTENCES_PER_DRAFT or callado)
                )
                if not listo:
                    continue
                preparado = self._prepared
                self._drafting = True
                self._lines_at_draft += pendientes
                for campo in self._fields.values():
                    if campo.state != "mine" and preparado is not None:
                        campo.state = "writing" if campo.text else campo.state
                self._version += 1
            self._draft(preparado)

    def _draft(self, preparado: PreparedConsultation | None) -> None:
        try:
            nota = self._consultation.render_note(preparado) if preparado else None
        except Exception:  # frontera: si no redacta, el borrador se queda como estaba
            nota = None
        with self._lock:
            ahora = self._clock()
            if nota is not None:
                for spec, texto in nota.fields:
                    campo = self._fields.get(spec.section_id)
                    if campo is None or campo.state == "mine" or not texto.strip():
                        continue
                    campo.text = texto
                    campo.state = "ai"
                    campo.updated = ahora
            for campo in self._fields.values():
                if campo.state == "writing":
                    campo.state = "ai"
            self._drafting = False
            self._version += 1


def line_kind(text: str) -> str:
    """La pantalla pinta las preguntas a la izquierda. Lo decide la misma puerta que
    la clasificación de apartados."""
    return "q" if is_interrogative(text) else "a"
