"""Pruebas unitarias de la captura robusta de audio (I-127, I-128, I-129).

Cubre §B.1 a §B.3 de `.claude/workflow/56-plan-captura-robusta.md`:
1. Filtro de voz: segmento con no_speech_prob alto desaparece, bajos se conservan.
2. Si todos superan el umbral, levanta NoSpeechError con la ruta.
3. Si el modelo devuelve cero segmentos, levanta NoSpeechError.
4. Umbral inyectable: con umbral 0.99 se conserva segmento de 0.944.
5. transcribe y transcribe_segments filtran idéntico con el mismo doble.
6. Segmento de solo espacios se descarta.
7. Error del modelo levanta TranscriptionError y no NoSpeechError.
8. NoSpeechError es subclase de TranscriptionError.
9. vad_filter no se envía al modelo.
10. load_capture_settings carga YAML completo y pack real.
11. Clave faltante o umbral no numérico levanta ValueError nombrando clave.
12. build_local_capture usa modelo, device y local_files_only=True.
13. Fallo de carga de modelo levanta ModelUnavailableError con orden de descarga.
14. capture_diagnosis con todo presente devuelve todas las líneas en True.
15. capture_diagnosis con modelo fallando solo marca esa línea en False.
"""

from __future__ import annotations

import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from aida.arms.health.capture import (
    CaptureSettings,
    LocalWhisperSTT,
    ModelUnavailableError,
    NoSpeechError,
    TranscriptionError,
    build_document_reader,
    build_local_capture,
    capture_diagnosis,
    download_local_model,
    load_capture_settings,
    local_model_bytes,
    split_into_sentences,
)
from aida.voice.diarize import TranscriptSegment


class _DummySegment:
    def __init__(self, start: float, end: float, text: str, no_speech_prob: float) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.no_speech_prob = no_speech_prob


class _DummyWhisperModel:
    def __init__(
        self, segments: list[_DummySegment], raise_exc: Exception | None = None
    ) -> None:
        self.segments = segments
        self.raise_exc = raise_exc
        self.kwargs_received: dict[str, Any] = {}

    def transcribe(
        self, audio_path: str, **kwargs: Any
    ) -> tuple[list[_DummySegment], None]:
        self.kwargs_received = kwargs
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.segments, None


def _modulo_doble(nombre: str, **atributos: Any) -> ModuleType:
    modulo = ModuleType(nombre)
    modulo.__spec__ = ModuleSpec(nombre, loader=None)
    for clave, valor in atributos.items():
        setattr(modulo, clave, valor)
    return modulo


# ---------------------------------------------------------------------------
# B.1 LocalWhisperSTT
# ---------------------------------------------------------------------------


def test_speech_segments_filtra_segmento_alto_y_conserva_bajos(tmp_path: Path) -> None:
    """Caso 1: segmento con no_speech_prob=0.944 entre dos de 0.451 desaparece;
    los otros dos se conservan intactos."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy audio")
    segs = [
        _DummySegment(0.0, 1.0, "primera frase", 0.451),
        _DummySegment(1.0, 2.0, "¡Suscríbete!", 0.944),
        _DummySegment(2.0, 3.0, "segunda frase", 0.451),
    ]
    model = _DummyWhisperModel(segs)
    stt = LocalWhisperSTT(model, no_speech_threshold=0.6)
    result = stt.transcribe_segments(audio)
    assert len(result) == 2
    assert result[0].text == "primera frase"
    assert result[1].text == "segunda frase"
    assert result[0].start == 0.0
    assert result[1].end == 3.0


def test_speech_segments_todos_superan_umbral_levanta_no_speech_error(
    tmp_path: Path,
) -> None:
    """Caso 2: si todos superan el umbral -> NoSpeechError con la ruta."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy audio")
    segs = [
        _DummySegment(0.0, 1.0, "ruido", 0.8),
        _DummySegment(1.0, 2.0, "estática", 0.95),
    ]
    stt = LocalWhisperSTT(_DummyWhisperModel(segs), no_speech_threshold=0.6)
    with pytest.raises(NoSpeechError) as exc_info:
        stt.transcribe_segments(audio)
    assert str(audio) in str(exc_info.value)
    assert "no se detectó voz" in str(exc_info.value)


def test_speech_segments_cero_segmentos_levanta_no_speech_error(
    tmp_path: Path,
) -> None:
    """Caso 3: modelo devuelve cero segmentos -> NoSpeechError, no cadena vacía."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy audio")
    stt = LocalWhisperSTT(_DummyWhisperModel([]), no_speech_threshold=0.6)
    with pytest.raises(NoSpeechError) as exc_info:
        stt.transcribe_segments(audio)
    assert str(audio) in str(exc_info.value)
    assert "no se detectó voz" in str(exc_info.value)


def test_speech_segments_umbral_alto_conserva_segmento(tmp_path: Path) -> None:
    """Caso 4: con no_speech_threshold=0.99, el de 0.944 se conserva: umbral
    inyectable de verdad."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy audio")
    segs = [_DummySegment(0.0, 1.0, "voz tenue", 0.944)]
    stt = LocalWhisperSTT(_DummyWhisperModel(segs), no_speech_threshold=0.99)
    result = stt.transcribe_segments(audio)
    assert len(result) == 1
    assert result[0].text == "voz tenue"


def test_transcribe_y_transcribe_segments_filtran_igual(tmp_path: Path) -> None:
    """Caso 5: transcribe y transcribe_segments filtran igual, con el mismo doble."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy audio")
    segs = [
        _DummySegment(0.0, 1.0, "hola doctor", 0.3),
        _DummySegment(1.0, 2.0, "ruido", 0.9),
        _DummySegment(2.0, 3.0, "tengo fiebre", 0.2),
    ]
    model = _DummyWhisperModel(segs)
    stt = LocalWhisperSTT(model, no_speech_threshold=0.6)
    texto = stt.transcribe(audio)
    segmentos = stt.transcribe_segments(audio)
    assert texto == "hola doctor tengo fiebre"
    assert " ".join(s.text for s in segmentos) == texto


def test_speech_segments_descarta_segmento_de_solo_espacios(tmp_path: Path) -> None:
    """Caso 6: segmento de solo espacios se descarta."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy audio")
    segs = [
        _DummySegment(0.0, 1.0, "   ", 0.1),
        _DummySegment(1.0, 2.0, "dolor de cabeza", 0.2),
    ]
    stt = LocalWhisperSTT(_DummyWhisperModel(segs), no_speech_threshold=0.6)
    result = stt.transcribe_segments(audio)
    assert len(result) == 1
    assert result[0].text == "dolor de cabeza"


def test_transcribe_error_del_modelo_levanta_transcription_error(
    tmp_path: Path,
) -> None:
    """Caso 7: si el modelo levanta, sale TranscriptionError y no NoSpeechError."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy audio")
    model = _DummyWhisperModel([], raise_exc=RuntimeError("CUDA out of memory"))
    stt = LocalWhisperSTT(model)
    with pytest.raises(TranscriptionError) as exc_info:
        stt.transcribe(audio)
    assert not isinstance(exc_info.value, NoSpeechError)
    assert "no se pudo transcribir" in str(exc_info.value)
    assert "CUDA out of memory" in str(exc_info.value)


def test_no_speech_error_es_subclase_de_transcription_error() -> None:
    """Caso 8: NoSpeechError es subclase de TranscriptionError."""
    assert issubclass(NoSpeechError, TranscriptionError)


def test_transcribe_no_envia_vad_filter(tmp_path: Path) -> None:
    """Caso 9: sin vad_filter: el doble comprueba que vad_filter no se envía."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy audio")
    segs = [_DummySegment(0.0, 1.0, "hola", 0.1)]
    model = _DummyWhisperModel(segs)
    stt = LocalWhisperSTT(model)
    stt.transcribe(audio)
    assert "vad_filter" not in model.kwargs_received


# ---------------------------------------------------------------------------
# B.2 Ajustes y construcción
# ---------------------------------------------------------------------------


def test_load_capture_settings_valido_y_pack_real(tmp_path: Path) -> None:
    """Caso 10: load_capture_settings sobre un YAML completo devuelve tipos;
    el del pack real carga con provider == 'cloud' y los campos nuevos."""
    yaml_path = tmp_path / "capture.yaml"
    yaml_path.write_text(
        "provider: cloud\n"
        "cloud_model: whisper-1\n"
        "cloud_timestamps: false\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "timeout_seconds: 120.0\n"
        "model: medium\n"
        "device: cpu\n"
        "no_speech_threshold: 0.55\n"
        "compute_type: int8\n"
        "connect_timeout_seconds: 3.0\n"
        "cloud_formats:\n"
        "  - .wav\n"
        "  - .mp3\n"
        "transcription_prompt: Consulta médica de prueba.\n"
        "abbreviations:\n"
        "  - dr.\n"
        "  - mg.\n",
        encoding="utf-8",
    )
    settings = load_capture_settings(yaml_path)
    assert settings.model == "medium"
    assert settings.device == "cpu"
    assert settings.no_speech_threshold == 0.55
    assert isinstance(settings.no_speech_threshold, float)
    assert settings.abbreviations == ("dr.", "mg.")
    assert settings.provider == "cloud"
    assert settings.cloud_model == "whisper-1"
    assert settings.cloud_timestamps is False
    assert settings.cloud_language == "es"
    assert settings.cloud_base_url == "https://api.openai.com"
    assert settings.timeout_seconds == 120.0
    assert settings.compute_type == "int8"
    assert settings.connect_timeout_seconds == 3.0
    assert settings.cloud_formats == (".wav", ".mp3")
    assert settings.transcription_prompt == "Consulta médica de prueba."

    real = load_capture_settings()
    assert real.model == "small"
    assert real.device == "cpu"
    assert real.no_speech_threshold == 0.6
    assert "dr." in real.abbreviations
    assert real.provider == "cloud"
    assert real.cloud_model == "gpt-4o-mini-transcribe"
    assert real.cloud_timestamps is False
    assert real.cloud_language == "es"
    assert real.cloud_base_url == "https://api.openai.com"
    assert real.timeout_seconds == 120.0
    assert real.compute_type == "int8"
    assert real.connect_timeout_seconds == 3.0
    assert ".wav" in real.cloud_formats
    assert len(real.transcription_prompt) > 0


def test_load_capture_settings_invalido_nombra_clave(tmp_path: Path) -> None:
    """Caso 11 (base): sin device, o con no_speech_threshold: 'alto', levanta ValueError
    nombrando la clave."""
    prefijo = (
        "provider: cloud\n"
        "cloud_model: whisper-1\n"
        "cloud_timestamps: false\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "timeout_seconds: 120.0\n"
        "compute_type: int8\n"
        "connect_timeout_seconds: 3.0\n"
        "cloud_formats: ['.wav']\n"
        "transcription_prompt: Consulta médica.\n"
        "model: small\n"
        "abbreviations:\n"
        "  - dr.\n"
    )
    sin_device = tmp_path / "sin_device.yaml"
    sin_device.write_text(
        f"{prefijo}no_speech_threshold: 0.6\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"faltan claves.*device"):
        load_capture_settings(sin_device)

    umbral_str = tmp_path / "umbral_str.yaml"
    umbral_str.write_text(
        f"{prefijo}device: cpu\nno_speech_threshold: alto\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"no_speech_threshold.*numérico"):
        load_capture_settings(umbral_str)


def test_load_capture_settings_provider_y_timeout_invalidos_nombran_clave(tmp_path: Path) -> None:
    """Caso 11: provider distinto de 'cloud' o 'local' da ValueError nombrando la clave.
    timeout_seconds <= 0 o no numérico da ValueError nombrando la clave."""
    base_yaml = (
        "cloud_model: whisper-1\n"
        "cloud_timestamps: false\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "model: small\n"
        "device: cpu\n"
        "no_speech_threshold: 0.6\n"
        "compute_type: int8\n"
        "connect_timeout_seconds: 3.0\n"
        "cloud_formats: ['.wav']\n"
        "transcription_prompt: Consulta médica.\n"
        "abbreviations:\n  - dr.\n"
    )

    p_invalido = tmp_path / "provider_invalido.yaml"
    p_invalido.write_text(f"provider: nube\ntimeout_seconds: 120.0\n{base_yaml}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"'provider' debe ser 'cloud' o 'local'"):
        load_capture_settings(p_invalido)

    t_cero = tmp_path / "timeout_cero.yaml"
    t_cero.write_text(f"provider: cloud\ntimeout_seconds: 0\n{base_yaml}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"'timeout_seconds' debe ser un número positivo"):
        load_capture_settings(t_cero)

    t_neg = tmp_path / "timeout_negativo.yaml"
    t_neg.write_text(f"provider: cloud\ntimeout_seconds: -10.5\n{base_yaml}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"'timeout_seconds' debe ser un número positivo"):
        load_capture_settings(t_neg)

    t_str = tmp_path / "timeout_str.yaml"
    t_str.write_text(f"provider: cloud\ntimeout_seconds: '120'\n{base_yaml}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"'timeout_seconds' debe ser un número positivo"):
        load_capture_settings(t_str)

    t_bool = tmp_path / "timeout_bool.yaml"
    t_bool.write_text(f"provider: cloud\ntimeout_seconds: true\n{base_yaml}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"'timeout_seconds' debe ser un número positivo"):
        load_capture_settings(t_bool)


def test_build_local_capture_usa_ajustes_declarados_y_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 12 (base): build_local_capture recibe modelo y device del ajuste,
    local_files_only=True, y el STT resultante lleva el umbral del ajuste."""
    recibidos: dict[str, Any] = {}

    class MockWhisperModel:
        def __init__(self, model_size_or_path: str, **kwargs: Any) -> None:
            recibidos["model"] = model_size_or_path
            recibidos.update(kwargs)

    modulo = _modulo_doble("faster_whisper", WhisperModel=MockWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", modulo)
    monkeypatch.setitem(sys.modules, "pytesseract", _modulo_doble("pytesseract"))

    settings = CaptureSettings(
        model="base",
        device="cpu",
        no_speech_threshold=0.75,
        abbreviations=("dr.",),
        provider="cloud",
        cloud_model="whisper-1",
        cloud_language="es",
        cloud_base_url="https://api.openai.com",
        timeout_seconds=120.0,
        compute_type="int8",
        connect_timeout_seconds=3.0,
        cloud_formats=(".wav",),
        transcription_prompt="Consulta médica.",
        cloud_timestamps=True,
    )
    stt, _ocr = build_local_capture(settings)
    assert recibidos["model"] == "base"
    assert recibidos["device"] == "cpu"
    assert recibidos["local_files_only"] is True
    assert stt._no_speech_threshold == 0.75


# ---------------------------------------------------------------------------
# Plan 58: Partición por frases y abreviaturas (§B.2 y §B.3)
# ---------------------------------------------------------------------------


def test_split_by_sentence_reparto_proporcional_y_speech_segments(tmp_path: Path) -> None:
    """Caso 8/9: segmento 0.0-10.0 con dos frases de igual longitud se parte al punto medio;
    end del último trozo coincide con el original. También se comprueba a través de
    transcribe_segments en LocalWhisperSTT."""
    seg = TranscriptSegment(0.0, 10.0, "Cinco uno. Cinco dos.")
    trozos = split_into_sentences(seg.text, abbreviations=frozenset(), start=seg.start, end=seg.end)
    assert len(trozos) == 2
    assert trozos[0].text == "Cinco uno."
    assert trozos[0].start == 0.0
    assert trozos[0].end == 5.0
    assert trozos[1].text == "Cinco dos."
    assert trozos[1].start == 5.0
    assert trozos[1].end == 10.0

    # Comprobación de integración con LocalWhisperSTT._speech_segments
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy")
    model = _DummyWhisperModel([_DummySegment(0.0, 10.0, "Cinco uno. Cinco dos.", 0.1)])
    stt = LocalWhisperSTT(model)
    segs = stt.transcribe_segments(audio)
    assert len(segs) == 2
    assert segs[0].start == 0.0
    assert segs[0].end == 5.0
    assert segs[1].start == 5.0
    assert segs[1].end == 10.0


def test_split_by_sentence_una_sola_frase_devuelve_mismo_objeto() -> None:
    """Caso 9: segmento con una sola frase devuelve un segmento con los mismos datos."""
    seg = TranscriptSegment(0.12345, 9.87654, "Solo una frase sin puntuacion fuerte")
    trozos = split_into_sentences(seg.text, abbreviations=frozenset(), start=seg.start, end=seg.end)
    assert len(trozos) == 1
    assert trozos[0].text == seg.text
    assert trozos[0].start == 0.12345
    assert trozos[0].end == 9.87654


def test_split_by_sentence_tiempos_crecientes_y_sin_huecos() -> None:
    """Caso 10: los tiempos son crecientes y sin huecos (end de cada trozo es start
    del siguiente)."""
    seg = TranscriptSegment(1.5, 12.3, "Frase corta. Otra frase bastante más larga. Y una final.")
    trozos = split_into_sentences(seg.text, abbreviations=frozenset(), start=seg.start, end=seg.end)
    assert len(trozos) == 3
    for i in range(len(trozos) - 1):
        assert trozos[i].end == trozos[i + 1].start
        t_start, t_end = trozos[i].start, trozos[i].end
        assert t_start is not None and t_end is not None
        assert t_start < t_end
    assert trozos[0].start == 1.5
    assert trozos[-1].end == 12.3
    assert trozos[0].end != seg.end


def test_local_whisper_stt_llama_a_split_into_sentences(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 8 (§B.2, M6): LocalWhisperSTT delega la partición de frases en split_into_sentences."""
    import aida.arms.health.capture as cap

    llamadas: list[Any] = []
    real_fn = cap.split_into_sentences

    def spy_split(
        text: str,
        *,
        abbreviations: frozenset[str],
        start: float | None = None,
        end: float | None = None,
    ) -> tuple[TranscriptSegment, ...]:
        llamadas.append((text, abbreviations, start, end))
        return real_fn(text, abbreviations=abbreviations, start=start, end=end)

    monkeypatch.setattr(cap, "split_into_sentences", spy_split)

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy")
    model = _DummyWhisperModel([_DummySegment(0.0, 5.0, "Frase de prueba.", 0.1)])
    stt = LocalWhisperSTT(model)
    stt.transcribe_segments(audio)

    assert len(llamadas) > 0, "LocalWhisperSTT no llamó a split_into_sentences"


def test_split_into_sentences_sin_tiempos_segmentos_con_none() -> None:
    """Caso 10 (§B.2): Sin start ni end, los segmentos salen con start=None y end=None."""
    trozos = split_into_sentences("Primera frase. Segunda frase.", abbreviations=frozenset())
    assert len(trozos) == 2
    assert trozos[0].text == "Primera frase."
    assert trozos[0].start is None
    assert trozos[0].end is None
    assert trozos[1].text == "Segunda frase."
    assert trozos[1].start is None
    assert trozos[1].end is None


def test_load_capture_settings_cloud_timestamps_obligatorio_y_booleano(tmp_path: Path) -> None:
    """Caso 11 (§B.2, M7, M8): cloud_timestamps es obligatorio en capture.yaml y debe ser bool."""
    import yaml

    # Pack real
    real = load_capture_settings()
    assert isinstance(real.cloud_timestamps, bool)
    assert real.cloud_timestamps is False

    # Falta cloud_timestamps
    p1 = tmp_path / "sin_timestamps.yaml"
    data: dict[str, Any] = {
        "model": "base",
        "device": "cpu",
        "no_speech_threshold": 0.6,
        "abbreviations": ["dr."],
        "provider": "cloud",
        "cloud_model": "gpt-4o-mini-transcribe",
        "cloud_language": "es",
        "cloud_base_url": "https://api.openai.com",
        "timeout_seconds": 30.0,
        "compute_type": "default",
        "connect_timeout_seconds": 5.0,
        "cloud_formats": [".wav"],
        "transcription_prompt": "prompt",
    }
    p1.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=r"cloud_timestamps"):
        load_capture_settings(p1)

    # Cadena "false" en vez de booleano
    p2 = tmp_path / "str_timestamps.yaml"
    data["cloud_timestamps"] = "false"
    p2.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=r"cloud_timestamps"):
        load_capture_settings(p2)


def test_load_capture_settings_abreviaturas_minusculas_y_pack_real(tmp_path: Path) -> None:
    """Caso 11: load_capture_settings devuelve abreviaturas en minúsculas y el pack real carga."""
    yaml_path = tmp_path / "capture.yaml"
    yaml_path.write_text(
        "provider: cloud\n"
        "cloud_model: whisper-1\n"
        "cloud_timestamps: false\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "timeout_seconds: 120.0\n"
        "compute_type: int8\n"
        "connect_timeout_seconds: 3.0\n"
        "cloud_formats: ['.wav']\n"
        "transcription_prompt: Consulta médica.\n"
        "model: small\ndevice: cpu\nno_speech_threshold: 0.6\n"
        "abbreviations:\n  - Dr.\n  - MG.\n  - ETC.\n",
        encoding="utf-8",
    )
    settings = load_capture_settings(yaml_path)
    assert settings.abbreviations == ("dr.", "mg.", "etc.")

    real = load_capture_settings()
    assert len(real.abbreviations) > 0
    assert all(a == a.lower() for a in real.abbreviations)
    assert "dr." in real.abbreviations
    assert "mg." in real.abbreviations


def test_load_capture_settings_sin_o_invalido_abbreviations_levanta_value_error(
    tmp_path: Path,
) -> None:
    """Caso 12: sin abbreviations o no siendo lista de cadenas -> ValueError nombrando la clave."""
    prefijo = (
        "provider: cloud\n"
        "cloud_model: whisper-1\n"
        "cloud_timestamps: false\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "timeout_seconds: 120.0\n"
        "compute_type: int8\n"
        "connect_timeout_seconds: 3.0\n"
        "cloud_formats: ['.wav']\n"
        "transcription_prompt: Consulta médica.\n"
    )
    sin_abbrevs = tmp_path / "sin_abbrevs.yaml"
    sin_abbrevs.write_text(
        f"{prefijo}model: small\ndevice: cpu\nno_speech_threshold: 0.6\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"faltan claves.*abbreviations"):
        load_capture_settings(sin_abbrevs)

    no_lista = tmp_path / "no_lista.yaml"
    no_lista.write_text(
        f"{prefijo}model: small\ndevice: cpu\nno_speech_threshold: 0.6\nabbreviations: dr.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'abbreviations' debe ser una lista de cadenas"):
        load_capture_settings(no_lista)

    lista_no_str = tmp_path / "lista_no_str.yaml"
    lista_no_str.write_text(
        f"{prefijo}model: small\ndevice: cpu\nno_speech_threshold: 0.6\nabbreviations:\n  - 123\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'abbreviations' debe ser una lista de cadenas"):
        load_capture_settings(lista_no_str)


def test_build_local_capture_pasa_abreviaturas_a_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 13: build_local_capture pasa las abreviaturas del ajuste al STT."""
    class MockWhisperModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    modulo = _modulo_doble("faster_whisper", WhisperModel=MockWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", modulo)
    monkeypatch.setitem(sys.modules, "pytesseract", _modulo_doble("pytesseract"))

    settings = CaptureSettings(
        model="base",
        device="cpu",
        no_speech_threshold=0.75,
        abbreviations=("dr.", "mg.", "ml."),
        provider="cloud",
        cloud_model="whisper-1",
        cloud_language="es",
        cloud_base_url="https://api.openai.com",
        timeout_seconds=120.0,
        compute_type="int8",
        connect_timeout_seconds=3.0,
        cloud_formats=(".wav",),
        transcription_prompt="Consulta médica.",
        cloud_timestamps=True,
    )
    stt, _ocr = build_local_capture(settings)
    assert stt._abbreviations == frozenset({"dr.", "mg.", "ml."})


# ---------------------------------------------------------------------------
# Plan 70: Puntuación con initial_prompt y compute_type (§B.1)
# ---------------------------------------------------------------------------


def test_speech_segments_pasa_prompt_declarado_al_modelo(tmp_path: Path) -> None:
    """Caso 1 (§B.1): El prompt llega al modelo con initial_prompt."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy")
    model = _DummyWhisperModel([_DummySegment(0.0, 1.0, "Texto clínico", 0.1)])
    stt = LocalWhisperSTT(model, prompt="Prompt clínico de prueba.")
    stt.transcribe_segments(audio)
    assert model.kwargs_received.get("initial_prompt") == "Prompt clínico de prueba."


def test_speech_segments_prompt_vacio_pasa_none(tmp_path: Path) -> None:
    """Caso 2 (§B.1): Prompt vacío pasa None, no ''."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy")
    model = _DummyWhisperModel([_DummySegment(0.0, 1.0, "Texto clínico", 0.1)])
    stt = LocalWhisperSTT(model, prompt="")
    stt.transcribe_segments(audio)
    assert model.kwargs_received.get("initial_prompt") is None


def test_load_capture_settings_sin_transcription_prompt_levanta_value_error(tmp_path: Path) -> None:
    """Caso 3 (§B.1): transcription_prompt es obligatorio en capture.yaml."""
    p = tmp_path / "capture.yaml"
    p.write_text(
        "model: small\ndevice: cpu\nno_speech_threshold: 0.6\nabbreviations: []\n"
        "provider: local\ncloud_model: m\ncloud_language: es\ncloud_base_url: http://x\n"
        "timeout_seconds: 10\ncompute_type: int8\nconnect_timeout_seconds: 3.0\n"
        "cloud_formats: ['.wav']\ncloud_timestamps: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="transcription_prompt"):
        load_capture_settings(p)


def test_build_local_capture_pack_real_propaga_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 4 (§B.1): El pack real trae prompt y build_local_capture la propaga."""
    class MockWhisperModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    modulo_fw = _modulo_doble("faster_whisper", WhisperModel=MockWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", modulo_fw)
    monkeypatch.setitem(sys.modules, "pytesseract", _modulo_doble("pytesseract"))
    stt, _ = build_local_capture()
    assert isinstance(stt, LocalWhisperSTT)
    assert stt._prompt == load_capture_settings().transcription_prompt
    assert len(stt._prompt) > 0


def test_speech_segments_conserva_language_es_con_prompt(tmp_path: Path) -> None:
    """Caso 5 (§B.1): language='es' sigue pasándose junto con el prompt."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy")
    model = _DummyWhisperModel([_DummySegment(0.0, 1.0, "Texto clínico", 0.1)])
    stt = LocalWhisperSTT(model, prompt="Prompt")
    stt.transcribe_segments(audio)
    assert model.kwargs_received.get("language") == "es"
    assert model.kwargs_received.get("initial_prompt") == "Prompt"


def test_build_local_capture_pasa_compute_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 5b (§B.1): compute_type llega a WhisperModel."""
    kwargs_capturados: dict[str, Any] = {}

    class MockWhisperModel:
        def __init__(self, model_size_or_path: str, **kwargs: Any) -> None:
            kwargs_capturados.update(kwargs)

    modulo_fw = _modulo_doble("faster_whisper", WhisperModel=MockWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", modulo_fw)
    monkeypatch.setitem(sys.modules, "pytesseract", _modulo_doble("pytesseract"))
    build_local_capture()
    assert kwargs_capturados.get("compute_type") == load_capture_settings().compute_type


def test_load_capture_settings_sin_compute_type_levanta_value_error(tmp_path: Path) -> None:
    """Caso 5c (§B.1): compute_type es obligatorio en capture.yaml."""
    p = tmp_path / "capture.yaml"
    p.write_text(
        "model: small\ndevice: cpu\nno_speech_threshold: 0.6\nabbreviations: []\n"
        "provider: local\ncloud_model: m\ncloud_language: es\ncloud_base_url: http://x\n"
        "timeout_seconds: 10\nconnect_timeout_seconds: 3.0\n"
        "cloud_formats: ['.wav']\ntranscription_prompt: 'p'\ncloud_timestamps: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="compute_type"):
        load_capture_settings(p)


def test_load_capture_settings_sin_cloud_formats_levanta_value_error(tmp_path: Path) -> None:
    """Caso 29 (§B.4b/M29): cloud_formats es obligatorio en capture.yaml."""
    p = tmp_path / "capture.yaml"
    p.write_text(
        "model: small\ndevice: cpu\nno_speech_threshold: 0.6\nabbreviations: []\n"
        "provider: local\ncloud_model: m\ncloud_language: es\ncloud_base_url: http://x\n"
        "timeout_seconds: 10\nconnect_timeout_seconds: 3.0\ncompute_type: int8\n"
        "transcription_prompt: 'p'\ncloud_timestamps: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cloud_formats"):
        load_capture_settings(p)


def test_build_local_capture_falla_levanta_model_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 13: si el doble levanta al construirse -> ModelUnavailableError,
    con la orden de descarga en el mensaje."""
    class FailingWhisperModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("archivo no encontrado en cache local")

    modulo = _modulo_doble("faster_whisper", WhisperModel=FailingWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", modulo)
    monkeypatch.setitem(sys.modules, "pytesseract", _modulo_doble("pytesseract"))

    with pytest.raises(ModelUnavailableError) as exc_info:
        build_local_capture()
    assert "no se pudo cargar" in str(exc_info.value)
    assert "descárgala una vez con: uv run python -c" in str(exc_info.value)


# ---------------------------------------------------------------------------
# B.3 Diagnóstico
# ---------------------------------------------------------------------------


def test_capture_diagnosis_todo_presente(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 14: con todo presente, todas las líneas en True."""
    class MockWhisperModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    whisper_mod = _modulo_doble("faster_whisper", WhisperModel=MockWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", whisper_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", _modulo_doble("onnxruntime"))
    monkeypatch.setitem(sys.modules, "av", _modulo_doble("av"))
    monkeypatch.setitem(sys.modules, "pypdfium2", _modulo_doble("pypdfium2"))

    tess_mod = _modulo_doble(
        "pytesseract",
        get_tesseract_version=lambda: "5.3.0",
        get_languages=lambda: ["eng", "spa", "osd"],
    )
    monkeypatch.setitem(sys.modules, "pytesseract", tess_mod)

    sd_mod = _modulo_doble(
        "sounddevice", query_devices=lambda kind: {"name": "Microfono USB"}
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd_mod)

    diagnosis = capture_diagnosis()
    assert len(diagnosis) >= 9
    assert all(correcto for _texto, correcto in diagnosis)


def test_capture_diagnosis_modelo_falla_solo_esa_linea_en_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 15: con el modelo levantando, solo esa línea en False y el resto igual."""
    class FailingWhisperModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Libcublas not found")

    whisper_mod = _modulo_doble("faster_whisper", WhisperModel=FailingWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", whisper_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", _modulo_doble("onnxruntime"))
    monkeypatch.setitem(sys.modules, "av", _modulo_doble("av"))
    monkeypatch.setitem(sys.modules, "pypdfium2", _modulo_doble("pypdfium2"))

    tess_mod = _modulo_doble(
        "pytesseract",
        get_tesseract_version=lambda: "5.3.0",
        get_languages=lambda: ["eng", "spa", "osd"],
    )
    monkeypatch.setitem(sys.modules, "pytesseract", tess_mod)

    sd_mod = _modulo_doble(
        "sounddevice", query_devices=lambda kind: {"name": "Microfono USB"}
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd_mod)

    diagnosis = capture_diagnosis()
    assert len(diagnosis) == 9
    falsos = [texto for texto, correcto in diagnosis if not correcto]
    assert len(falsos) == 1
    assert "modelo small en cpu: no se pudo cargar 'small' en 'cpu'" in falsos[0]
    ciertos = [texto for texto, correcto in diagnosis if correcto]
    assert len(ciertos) == 8
    assert any("tesseract" in t for t in ciertos)
    assert any("entrada de audio" in t for t in ciertos)


# ---------------------------------------------------------------------------
# Plan 78: Separación de diagnóstico, tesseract y modelo (§B.3)
# ---------------------------------------------------------------------------


def test_capture_diagnosis_sin_microfono_ni_modelo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 13 (§B.3, M8, M9): capture_diagnosis(need_microphone=False, need_model=False)
    con sounddevice simulado para reventar si se le pregunta: ninguna línea habla de la
    entrada de audio ni del modelo, y sounddevice.query_devices no se llamó.
    """
    sd_llamado = False

    def explode(*args: Any, **kwargs: Any) -> Any:
        nonlocal sd_llamado
        sd_llamado = True
        raise AssertionError("sounddevice no debe ser consultado")

    sd_mod = _modulo_doble("sounddevice", query_devices=explode)
    monkeypatch.setitem(sys.modules, "sounddevice", sd_mod)

    diagnosis = capture_diagnosis(need_microphone=False, need_model=False)
    assert sd_llamado is False
    assert not any("entrada de audio" in texto for texto, _ in diagnosis)
    assert not any("modelo" in texto for texto, _ in diagnosis)
    # Comprueba que tesseract y los módulos sí se reportan
    assert any("tesseract" in texto for texto, _ in diagnosis)
    assert any("módulo faster_whisper" in texto for texto, _ in diagnosis)


def test_capture_diagnosis_sin_argumentos_comprueba_microfono_y_modelo() -> None:
    """Caso 14 (§B.3): capture_diagnosis() sin argumentos sigue comprobando
    micrófono y modelo (utilizado por --check-capture).
    """
    diagnosis = capture_diagnosis()
    assert any("entrada de audio" in texto for texto, _ in diagnosis)
    assert any("modelo" in texto for texto, _ in diagnosis)


def test_build_document_reader_con_y_sin_aida_tesseract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 15 (§B.3): build_document_reader() con AIDA_TESSERACT=/ruta/x deja
    pytesseract.pytesseract.tesseract_cmd == '/ruta/x'; sin la variable no lo toca.
    """
    pytest.importorskip("pytesseract", reason="requiere pytesseract (extra capture)")
    import pytesseract

    cmd_original = pytesseract.pytesseract.tesseract_cmd

    # Con AIDA_TESSERACT
    monkeypatch.setenv("AIDA_TESSERACT", "/opt/custom/bin/tesseract")
    reader = build_document_reader()
    assert reader is not None
    assert pytesseract.pytesseract.tesseract_cmd == "/opt/custom/bin/tesseract"

    # Sin variable: no modifica el cmd actual
    monkeypatch.delenv("AIDA_TESSERACT", raising=False)
    pytesseract.pytesseract.tesseract_cmd = "/ruta/previa"
    build_document_reader()
    assert pytesseract.pytesseract.tesseract_cmd == "/ruta/previa"

    # Restauramos cmd original
    pytesseract.pytesseract.tesseract_cmd = cmd_original


def test_download_local_model_deshabilita_xet_y_pide_modelo_ajustes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 16 (§B.3, M7): download_local_model con faster_whisper.utils.download_model
    simulado: en el momento de la llamada, huggingface_hub.constants.HF_HUB_DISABLE_XET
    es True, y se pide el modelo de los ajustes.
    """
    pytest.importorskip("faster_whisper", reason="requiere faster_whisper (extra capture)")
    import faster_whisper.utils as fw_utils
    from huggingface_hub import constants

    modelo_pedido: str | None = None
    xet_durante_llamada: bool | None = None

    def fake_download_model(model: str, **kwargs: Any) -> Any:
        nonlocal modelo_pedido, xet_durante_llamada
        modelo_pedido = model
        xet_durante_llamada = getattr(constants, "HF_HUB_DISABLE_XET", False)
        return "/tmp/fake-model-dir"

    monkeypatch.setattr(fw_utils, "download_model", fake_download_model)

    settings = load_capture_settings()
    download_local_model(settings)

    assert modelo_pedido == settings.model
    assert xet_durante_llamada is True


def test_local_model_bytes_cuenta_archivos_incluyendo_incomplete_y_no_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 17 (§B.3): local_model_bytes cuenta los archivos de models--<repo> bajo
    HF_HUB_CACHE simulado, incluido un .incomplete, y no cuenta enlaces simbólicos.
    """
    pytest.importorskip("huggingface_hub", reason="requiere huggingface_hub (extra capture)")
    from huggingface_hub import constants

    cache_hub = tmp_path / "hub_cache"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache_hub))

    settings = CaptureSettings(
        model="tiny",
        device="cpu",
        no_speech_threshold=0.6,
        abbreviations=("dr.",),
        provider="cloud",
        cloud_model="whisper-1",
        cloud_language="es",
        cloud_base_url="https://api.openai.com",
        timeout_seconds=30.0,
        compute_type="int8",
        connect_timeout_seconds=3.0,
        cloud_formats=(".wav",),
        transcription_prompt="prompt",
        cloud_timestamps=False,
    )

    from aida.arms.health.capture import _model_repo

    repo_dir_name = f"models--{_model_repo('tiny').replace('/', '--')}"
    repo_dir = cache_hub / repo_dir_name
    repo_dir.mkdir(parents=True)

    # Archivo normal de 150 bytes
    f1 = repo_dir / "model.bin"
    f1.write_bytes(b"x" * 150)

    # Archivo .incomplete de 250 bytes
    f2 = repo_dir / "weights.incomplete"
    f2.write_bytes(b"y" * 250)

    # Symlink a f1 (no debe contarse)
    link = repo_dir / "model_symlink.bin"
    link.symlink_to(f1)

    total_bytes = local_model_bytes(settings)
    assert total_bytes == 400


def test_load_capture_settings_lee_model_download_mb_y_cero_si_falta(
    tmp_path: Path,
) -> None:
    """Caso 18 (§B.3): load_capture_settings lee model_download_mb: 464 del pack;
    un YAML sin la clave da 0.
    """
    real = load_capture_settings()
    assert real.model_download_mb == 464

    yaml_path = tmp_path / "sin_download_mb.yaml"
    yaml_path.write_text(
        "model: small\ndevice: cpu\nno_speech_threshold: 0.6\nabbreviations: []\n"
        "provider: local\ncloud_model: m\ncloud_language: es\ncloud_base_url: http://x\n"
        "timeout_seconds: 10\nconnect_timeout_seconds: 3.0\ncompute_type: int8\n"
        "transcription_prompt: 'p'\ncloud_timestamps: false\ncloud_formats: ['.wav']\n",
        encoding="utf-8",
    )
    s = load_capture_settings(yaml_path)
    assert s.model_download_mb == 0


def test_build_document_reader_bundled_tesseract_y_precedencia(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T7 (§B): Con data_root()/tesseract/tesseract(.exe) creado en el AIDA_DATA_DIR
    del test y sin AIDA_TESSERACT, tesseract_cmd es esa ruta y TESSDATA_PREFIX es su
    tessdata; con AIDA_TESSERACT, manda la variable."""
    pytest.importorskip("pytesseract", reason="requiere pytesseract (extra capture)")
    import pytesseract

    cmd_original = pytesseract.pytesseract.tesseract_cmd
    monkeypatch.setenv("TESSDATA_PREFIX", "previo")

    data_dir = tmp_path / "data"
    monkeypatch.setenv("AIDA_DATA_DIR", str(data_dir))
    tess_dir = data_dir / "tesseract"
    tess_dir.mkdir(parents=True)
    nombre = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    ejecutable = tess_dir / nombre
    ejecutable.write_text("#!/bin/sh\n", encoding="utf-8")
    tessdata_esperado = str(tess_dir / "tessdata")

    try:
        # 1. Sin AIDA_TESSERACT: usa el bundled tesseract y fija TESSDATA_PREFIX
        monkeypatch.delenv("AIDA_TESSERACT", raising=False)
        reader = build_document_reader()
        assert reader is not None
        assert pytesseract.pytesseract.tesseract_cmd == str(ejecutable)
        assert os.environ.get("TESSDATA_PREFIX") == tessdata_esperado

        # 2. Con AIDA_TESSERACT: manda la variable sobre el bundled
        ruta_custom = "/opt/custom/bin/tesseract"
        monkeypatch.setenv("AIDA_TESSERACT", ruta_custom)
        monkeypatch.setenv("TESSDATA_PREFIX", "otro_prefix")
        build_document_reader()
        assert pytesseract.pytesseract.tesseract_cmd == ruta_custom
        assert os.environ.get("TESSDATA_PREFIX") == "otro_prefix"
    finally:
        pytesseract.pytesseract.tesseract_cmd = cmd_original


def test_capture_diagnosis_modulo_av_falla_con_error_de_importacion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T8 (§B): Un av/__init__.py que lanza ImportError("DLL load failed") en tmp_path,
    puesto al frente de sys.path y sin av en sys.modules: la línea de av sale False y su
    texto lleva el error."""
    av_pkg = tmp_path / "av"
    av_pkg.mkdir()
    (av_pkg / "__init__.py").write_text("raise ImportError('DLL load failed')\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "av", raising=False)

    class MockWhisperModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        _modulo_doble("faster_whisper", WhisperModel=MockWhisperModel),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", _modulo_doble("onnxruntime"))
    monkeypatch.setitem(sys.modules, "pypdfium2", _modulo_doble("pypdfium2"))
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        _modulo_doble("sounddevice", query_devices=lambda kind: {"name": "mic"}),
    )
    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        _modulo_doble(
            "pytesseract",
            get_tesseract_version=lambda: "5.3",
            get_languages=lambda: ["spa"],
        ),
    )

    diagnosis = capture_diagnosis()
    lineas_av = [linea for linea in diagnosis if "módulo av" in linea[0]]
    assert len(lineas_av) == 1, "No se encontró la línea del módulo av"
    texto_av, ok_av = lineas_av[0]
    assert ok_av is False
    assert "DLL load failed" in texto_av

