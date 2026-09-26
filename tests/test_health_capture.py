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
    _split_by_sentence,
    build_local_capture,
    capture_diagnosis,
    load_capture_settings,
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
    el del pack real carga con provider == 'cloud' y los cinco campos nuevos."""
    yaml_path = tmp_path / "capture.yaml"
    yaml_path.write_text(
        "provider: cloud\n"
        "cloud_model: whisper-1\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "timeout_seconds: 120.0\n"
        "model: medium\n"
        "device: cpu\n"
        "no_speech_threshold: 0.55\n"
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
    assert settings.cloud_language == "es"
    assert settings.cloud_base_url == "https://api.openai.com"
    assert settings.timeout_seconds == 120.0

    real = load_capture_settings()
    assert real.model == "small"
    assert real.device == "cpu"
    assert real.no_speech_threshold == 0.6
    assert "dr." in real.abbreviations
    assert real.provider == "cloud"
    assert real.cloud_model == "whisper-1"
    assert real.cloud_language == "es"
    assert real.cloud_base_url == "https://api.openai.com"
    assert real.timeout_seconds == 120.0


def test_load_capture_settings_invalido_nombra_clave(tmp_path: Path) -> None:
    """Caso 11 (base): sin device, o con no_speech_threshold: 'alto', levanta ValueError
    nombrando la clave."""
    sin_device = tmp_path / "sin_device.yaml"
    sin_device.write_text(
        "provider: cloud\n"
        "cloud_model: whisper-1\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "timeout_seconds: 120.0\n"
        "model: small\n"
        "no_speech_threshold: 0.6\n"
        "abbreviations:\n"
        "  - dr.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"faltan claves.*device"):
        load_capture_settings(sin_device)

    umbral_str = tmp_path / "umbral_str.yaml"
    umbral_str.write_text(
        "provider: cloud\n"
        "cloud_model: whisper-1\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "timeout_seconds: 120.0\n"
        "model: small\n"
        "device: cpu\n"
        "no_speech_threshold: alto\n"
        "abbreviations:\n"
        "  - dr.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"no_speech_threshold.*numérico"):
        load_capture_settings(umbral_str)


def test_load_capture_settings_provider_y_timeout_invalidos_nombran_clave(tmp_path: Path) -> None:
    """Caso 11: provider distinto de 'cloud' o 'local' da ValueError nombrando la clave.
    timeout_seconds <= 0 o no numérico da ValueError nombrando la clave."""
    base_yaml = (
        "cloud_model: whisper-1\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "model: small\n"
        "device: cpu\n"
        "no_speech_threshold: 0.6\n"
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
    """Caso 8: segmento 0.0-10.0 con dos frases de igual longitud se parte al punto medio;
    end del último trozo coincide con el original. También se comprueba a través de
    transcribe_segments en LocalWhisperSTT."""
    seg = TranscriptSegment(0.0, 10.0, "Cinco uno. Cinco dos.")
    trozos = _split_by_sentence(seg, frozenset())
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
    """Caso 9: segmento con una sola frase devuelve el mismo objeto idéntico
    sin tocar tiempos ni redondear."""
    seg = TranscriptSegment(0.12345, 9.87654, "Solo una frase sin puntuacion fuerte")
    trozos = _split_by_sentence(seg, frozenset())
    assert len(trozos) == 1
    assert trozos[0] is seg
    assert trozos[0].start == 0.12345
    assert trozos[0].end == 9.87654


def test_split_by_sentence_tiempos_crecientes_y_sin_huecos() -> None:
    """Caso 10: los tiempos son crecientes y sin huecos (end de cada trozo es start
    del siguiente)."""
    seg = TranscriptSegment(1.5, 12.3, "Frase corta. Otra frase bastante más larga. Y una final.")
    trozos = _split_by_sentence(seg, frozenset())
    assert len(trozos) == 3
    for i in range(len(trozos) - 1):
        assert trozos[i].end == trozos[i + 1].start
        assert trozos[i].start < trozos[i].end
    assert trozos[0].start == 1.5
    assert trozos[-1].end == 12.3
    assert trozos[0].end != seg.end


def test_load_capture_settings_abreviaturas_minusculas_y_pack_real(tmp_path: Path) -> None:
    """Caso 11: load_capture_settings devuelve abreviaturas en minúsculas y el pack real carga."""
    yaml_path = tmp_path / "capture.yaml"
    yaml_path.write_text(
        "provider: cloud\n"
        "cloud_model: whisper-1\n"
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "timeout_seconds: 120.0\n"
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
        "cloud_language: es\n"
        "cloud_base_url: https://api.openai.com\n"
        "timeout_seconds: 120.0\n"
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
    )
    stt, _ocr = build_local_capture(settings)
    assert stt._abbreviations == frozenset({"dr.", "mg.", "ml."})


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
    assert len(diagnosis) >= 6
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
    assert len(diagnosis) == 6
    falsos = [texto for texto, correcto in diagnosis if not correcto]
    assert len(falsos) == 1
    assert "modelo small en cpu: no se pudo cargar 'small' en 'cpu'" in falsos[0]
    ciertos = [texto for texto, correcto in diagnosis if correcto]
    assert len(ciertos) == 5
    assert any("tesseract" in t for t in ciertos)
    assert any("entrada de audio" in t for t in ciertos)
