"""Pruebas unitarias de la caché de transcripciones y CachedSTT (Plan 65: I-131).

Cubre §B.1 de `.claude/workflow/65-plan-reanudar.md`:
1. Ida y vuelta: store y load devuelven los mismos segmentos, con sus tiempos y textos exactos.
2. Huella que no existe da None.
3. Caducado: con ttl_seconds=0, load da None y borra el archivo.
4. Con otra clave Fernet, load da None y no revienta.
5. Archivo corrupto (bytes no descifrables) da None.
6. La huella depende del contenido, no del nombre/ruta; un byte distinto cambia la huella.
7. La huella depende del modelo y del umbral de voz.
8. purge_expired borra solo lo vencido y devuelve cuántos; sin directorio devuelve 0.
9. El nombre del archivo no lleva el identificador del paciente.
10. CachedSTT: primera llamada delega en STT real una vez; segunda no lo llama y devuelve lo mismo.
11. CachedSTT.transcribe devuelve el texto unido y comparte la misma caché con transcribe_segments.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from cryptography.fernet import Fernet

from aida.arms.health.transcripts import (
    CachedSTT,
    TranscriptCache,
    transcript_fingerprint,
)
from aida.voice.diarize import TranscriptSegment


def test_transcript_cache_ida_y_vuelta(tmp_path: Path) -> None:
    """Caso 1 (§B.1): store y load devuelven los mismos segmentos con tiempos y textos exactos."""
    key = Fernet.generate_key()
    cache = TranscriptCache(tmp_path, key=key)
    fp = "abcd1234efgh5678"
    segments = (
        TranscriptSegment(start=0.0, end=1.5, text="Buenos días"),
        TranscriptSegment(start=1.5, end=3.2, text="¿Qué le ocurre hoy?"),
    )
    cache.store(fp, segments)

    cargado = cache.load(fp)
    assert cargado == segments
    assert cargado is not None
    assert cargado[0].start == 0.0
    assert cargado[0].end == 1.5
    assert cargado[0].text == "Buenos días"
    assert cargado[1].start == 1.5
    assert cargado[1].end == 3.2
    assert cargado[1].text == "¿Qué le ocurre hoy?"


def test_transcript_cache_huella_inexistente_devuelve_none(tmp_path: Path) -> None:
    """Caso 2 (§B.1): Una huella que no existe en caché devuelve None."""
    key = Fernet.generate_key()
    cache = TranscriptCache(tmp_path, key=key)
    assert cache.load("no_existe_esta_huella") is None


def test_transcript_cache_caducado_devuelve_none_y_borra_archivo(tmp_path: Path) -> None:
    """Caso 3 (§B.1): Con ttl_seconds=0, load da None y borra el archivo en disco."""
    key = Fernet.generate_key()
    cache = TranscriptCache(tmp_path, key=key, ttl_seconds=0)
    fp = "expira_inmediato"
    segments = (TranscriptSegment(start=0.0, end=1.0, text="Texto corto"),)
    cache.store(fp, segments)

    archivo = tmp_path / f"{fp}.transcript"
    assert archivo.exists()

    # Breve pausa para asegurar mtime < time.time()
    time.sleep(0.01)

    cargado = cache.load(fp)
    assert cargado is None
    # El archivo debe haber sido borrado por load
    assert not archivo.exists()


def test_transcript_cache_otra_clave_devuelve_none_sin_reventar(tmp_path: Path) -> None:
    """Caso 4 (§B.1): Con otra clave Fernet, load da None sin lanzar excepciones."""
    key1 = Fernet.generate_key()
    key2 = Fernet.generate_key()
    cache1 = TranscriptCache(tmp_path, key=key1)
    cache2 = TranscriptCache(tmp_path, key=key2)

    fp = "huella_con_otra_clave"
    segments = (TranscriptSegment(start=0.0, end=2.0, text="Confidencial"),)
    cache1.store(fp, segments)

    # Con la clave equivocada no debe lanzar InvalidToken sino None
    assert cache2.load(fp) is None


def test_transcript_cache_archivo_corrupto_devuelve_none(tmp_path: Path) -> None:
    """Caso 5 (§B.1): Un archivo con bytes corruptos que no son un token Fernet da None."""
    key = Fernet.generate_key()
    cache = TranscriptCache(tmp_path, key=key)
    fp = "huella_corrupta"
    archivo = tmp_path / f"{fp}.transcript"
    archivo.write_bytes(b"bytes totalmente corruptos y no cifrados")

    assert cache.load(fp) is None


def test_transcript_fingerprint_depende_del_contenido_no_de_la_ruta(tmp_path: Path) -> None:
    """Caso 6 (§B.1): Dos rutas distintas con idéntico contenido dan la misma huella;
    un solo byte modificado produce una huella distinta."""
    audio_a = tmp_path / "audio_a.wav"
    audio_b = tmp_path / "sub" / "audio_b.wav"
    audio_b.parent.mkdir(parents=True)
    audio_c = tmp_path / "audio_c.wav"

    contenido_base = b"RIFF 1234 WAVE fmt data audio identico"
    contenido_distinto = b"RIFF 1234 WAVE fmt data audio distintO"

    audio_a.write_bytes(contenido_base)
    audio_b.write_bytes(contenido_base)
    audio_c.write_bytes(contenido_distinto)

    ha = transcript_fingerprint(audio_a, model="small", no_speech_threshold=0.6)
    hb = transcript_fingerprint(audio_b, model="small", no_speech_threshold=0.6)
    hc = transcript_fingerprint(audio_c, model="small", no_speech_threshold=0.6)

    assert ha == hb
    assert ha != hc

    # Determinismo entre procesos (blake2b vs hash)
    codigo = f"""
from pathlib import Path
from aida.arms.health.transcripts import transcript_fingerprint
print(transcript_fingerprint(Path({str(audio_a)!r}), model='small', no_speech_threshold=0.6))
"""
    res1 = subprocess.check_output(
        [sys.executable, "-c", codigo],
        env={**os.environ, "PYTHONHASHSEED": "1234"},
        text=True,
    ).strip()
    res2 = subprocess.check_output(
        [sys.executable, "-c", codigo],
        env={**os.environ, "PYTHONHASHSEED": "9876"},
        text=True,
    ).strip()
    assert res1 == res2 == ha


def test_transcript_fingerprint_depende_del_modelo_y_umbral(tmp_path: Path) -> None:
    """Caso 7 (§B.1): Cambiar de modelo o de umbral de voz altera la huella."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy audio content")

    h_base = transcript_fingerprint(audio, model="small", no_speech_threshold=0.6)
    h_otro_modelo = transcript_fingerprint(audio, model="medium", no_speech_threshold=0.6)
    h_otro_umbral = transcript_fingerprint(audio, model="small", no_speech_threshold=0.5)

    assert h_base != h_otro_modelo
    assert h_base != h_otro_umbral
    assert h_otro_modelo != h_otro_umbral


def test_transcript_cache_purge_expired_borra_solo_vencido(tmp_path: Path) -> None:
    """Caso 8 (§B.1): purge_expired borra solo archivos vencidos y retorna cuántos;
    si el directorio no existe devuelve 0."""
    key = Fernet.generate_key()
    dir_cache = tmp_path / "cache"
    cache = TranscriptCache(dir_cache, key=key, ttl_seconds=10)

    # 1. Sin directorio
    dir_inexistente = tmp_path / "inexistente"
    cache_sin_dir = TranscriptCache(dir_inexistente, key=key)
    assert cache_sin_dir.purge_expired() == 0

    # 2. Archivo fresco y archivo caducado
    cache.store("fresco", (TranscriptSegment(0.0, 1.0, "Fresco"),))
    cache.store("viejo", (TranscriptSegment(0.0, 1.0, "Viejo"),))

    # Modificar mtime del viejo para simular antigüedad de 100 segundos
    archivo_viejo = dir_cache / "viejo.transcript"
    archivo_fresco = dir_cache / "fresco.transcript"
    hace_100s = time.time() - 100.0
    os.utime(archivo_viejo, (hace_100s, hace_100s))

    borrados = cache.purge_expired()
    assert borrados == 1
    assert not archivo_viejo.exists()
    assert archivo_fresco.exists()


def test_transcript_cache_nombre_archivo_no_lleva_identificador_paciente(tmp_path: Path) -> None:
    """Caso 9 (§B.1): El nombre del archivo en disco es la huella hexadecimal del audio,
    sin incluir ningún identificador del paciente ni metadatos clínicos."""
    key = Fernet.generate_key()
    cache = TranscriptCache(tmp_path, key=key)
    fp = "1234567890abcdef1234567890abcdef"
    cache.store(fp, (TranscriptSegment(0.0, 1.0, "Consulta privada"),))

    archivos = list(tmp_path.glob("*.transcript"))
    assert len(archivos) == 1
    nombre = archivos[0].name
    assert nombre == f"{fp}.transcript"
    assert "paciente" not in nombre.lower()
    assert "sujeto" not in nombre.lower()


class _SpySTT:
    def __init__(self, segments: tuple[TranscriptSegment, ...]) -> None:
        self.segments = segments
        self.calls = 0

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        self.calls += 1
        return self.segments


def test_cached_stt_primera_llamada_delega_segunda_usa_cache(tmp_path: Path) -> None:
    """Caso 10 (§B.1): CachedSTT delega en el STT real una sola vez; la segunda llamada
    lee de caché sin invocar al motor de voz."""
    key = Fernet.generate_key()
    cache = TranscriptCache(tmp_path, key=key)
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"audio de siete minutos simulado")

    spy = _SpySTT((
        TranscriptSegment(0.0, 2.0, "El paciente refiere tos"),
        TranscriptSegment(2.0, 4.0, "y fiebre desde ayer"),
    ))
    cached = CachedSTT(spy, cache, model="small", no_speech_threshold=0.6)

    # 1. Primera llamada: delega en el STT real
    res1 = cached.transcribe_segments(audio)
    assert spy.calls == 1
    assert len(res1) == 2

    # 2. Segunda llamada: no invoca al STT real y devuelve lo mismo
    res2 = cached.transcribe_segments(audio)
    assert spy.calls == 1
    assert res2 == res1


def test_cached_stt_transcribe_usa_misma_cache_que_transcribe_segments(tmp_path: Path) -> None:
    """Caso 11 (§B.1): CachedSTT.transcribe devuelve el texto unido y comparte la
    misma caché con transcribe_segments."""
    key = Fernet.generate_key()
    cache = TranscriptCache(tmp_path, key=key)
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"audio para probar transcribe")

    spy = _SpySTT((
        TranscriptSegment(0.0, 1.0, "Hola"),
        TranscriptSegment(1.0, 2.0, "doctor"),
    ))
    cached = CachedSTT(spy, cache, model="small", no_speech_threshold=0.6)

    texto = cached.transcribe(audio)
    assert texto == "Hola doctor"
    assert spy.calls == 1

    # Inmediatamente después, transcribe_segments no debe llamar al STT
    seg = cached.transcribe_segments(audio)
    assert spy.calls == 1
    assert len(seg) == 2
    assert seg[0].text == "Hola"
    assert seg[1].text == "doctor"


def test_transcript_cache_preserva_segmentos_sin_tiempos(tmp_path: Path) -> None:
    """Caso 13 (§B.3, M10): La caché de transcripciones guarda y recupera segmentos sin tiempos
    (start=None y end=None) sin corromperlos a 0.0 ni reventar; None sobrevive al viaje por JSON."""
    key = Fernet.generate_key()
    cache = TranscriptCache(tmp_path, key=key)
    fp = "huella_sin_tiempos"
    segments = (
        TranscriptSegment(start=None, end=None, text="Primera frase sin marcas de tiempo."),
        TranscriptSegment(start=None, end=None, text="Segunda frase sin marcas de tiempo."),
    )
    cache.store(fp, segments)

    cargado = cache.load(fp)
    assert cargado == segments
    assert cargado is not None
    assert cargado[0].start is None
    assert cargado[0].end is None
    assert cargado[0].start != 0.0
    assert cargado[0].end != 0.0
    assert cargado[0].text == "Primera frase sin marcas de tiempo."
    assert cargado[1].start is None
    assert cargado[1].end is None
    assert cargado[1].start != 0.0
    assert cargado[1].end != 0.0
    assert cargado[1].text == "Segunda frase sin marcas de tiempo."
