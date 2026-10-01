"""Pruebas unitarias del puerto de transcripción y respaldo de nube a local (§B.1).

Cubre los casos 1 a 9 de .claude/workflow/67-plan-puerto-stt.md:
1. La nube responde: devuelve segmentos con tiempos y el respaldo no se llama.
2. Segmentos vacíos o de solo espacios se descartan.
3. Error 500: cae al respaldo y el aviso lleva el motivo.
4. Respuesta sin 'segments': cae al respaldo indicando que falta información de tiempos.
5. Sin red (ConnectError): cae al respaldo.
6. Tiempo agotado (TimeoutException): cae al respaldo.
7. Sin callback on_fallback: la caída opera sin lanzar excepciones.
8. FallbackTranscription.transcribe une texto y usa transcribe_segments.
9. El adaptador envía modelo, idioma y pide segmentos con tiempos en la petición HTTP.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from aida.arms.health.transcription import (
    CloudTranscriptionError,
    CloudWhisperSTT,
    FallbackNotice,
    FallbackTranscription,
)
from aida.voice.diarize import TranscriptSegment


class _SpyFallbackProvider:
    def __init__(self, segments: tuple[TranscriptSegment, ...] = ()) -> None:
        self.segments = segments
        self.call_count = 0
        self.last_audio_path: Path | None = None

    def transcribe_segments(self, audio_path: Path) -> tuple[TranscriptSegment, ...]:
        self.call_count += 1
        self.last_audio_path = audio_path
        return self.segments


def test_nube_responde_devuelve_segmentos_y_respaldo_no_se_llama(tmp_path: Path) -> None:
    """Caso 1: la nube responde con segmentos y tiempos válidos; el respaldo no se invoca."""
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"RIFF dummy audio data")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        return httpx.Response(
            200,
            json={
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "Consulta por dolor articular."},
                    {"start": 2.5, "end": 5.0, "text": "En rodilla derecha."},
                ]
            },
        )

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    nube = CloudWhisperSTT(http, model="whisper-1", language="es")
    respaldo = _SpyFallbackProvider(
        segments=(TranscriptSegment(0.0, 1.0, "texto local"),)
    )

    fallback = FallbackTranscription(nube, respaldo)
    resultado = fallback.transcribe_segments(audio)

    assert len(resultado) == 2
    assert resultado[0].start == 0.0
    assert resultado[0].end == 2.5
    assert resultado[0].text == "Consulta por dolor articular."
    assert resultado[1].start == 2.5
    assert resultado[1].end == 5.0
    assert resultado[1].text == "En rodilla derecha."
    assert respaldo.call_count == 0


def test_segmentos_vacios_o_solo_espacios_se_descartan(tmp_path: Path) -> None:
    """Caso 2: segmentos vacíos, sin texto o con solo espacios son ignorados."""
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"RIFF dummy audio data")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": ""},
                    {"start": 1.0, "end": 2.0, "text": "   \n\t  "},
                    {"start": 2.0, "end": 3.0, "text": "Segmento clínicamente relevante"},
                ]
            },
        )

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    nube = CloudWhisperSTT(http, model="whisper-1", language="es")
    resultado = nube.transcribe_segments(audio)

    assert len(resultado) == 1
    assert resultado[0].start == 2.0
    assert resultado[0].end == 3.0
    assert resultado[0].text == "Segmento clínicamente relevante"


def test_error_500_cae_al_respaldo_y_aviso_lleva_motivo(tmp_path: Path) -> None:
    """Caso 3: error 500 del servidor activa el respaldo local y comunica la razón."""
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"RIFF dummy audio data")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    nube = CloudWhisperSTT(http, model="whisper-1", language="es")
    respaldo_segs = (TranscriptSegment(0.0, 3.0, "Transcripción local de respaldo"),)
    respaldo = _SpyFallbackProvider(segments=respaldo_segs)

    avisos: list[FallbackNotice] = []
    fallback = FallbackTranscription(nube, respaldo, on_fallback=avisos.append)
    resultado = fallback.transcribe_segments(audio)

    assert resultado == respaldo_segs
    assert respaldo.call_count == 1
    assert len(avisos) == 1
    assert "500" in avisos[0].reason


def test_respuesta_sin_segments_cae_al_respaldo_con_aviso(tmp_path: Path) -> None:
    """Caso 4: respuesta sin el campo 'segments' es inservible y activa el respaldo."""
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"RIFF dummy audio data")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "Texto sin segmentos temporales"})

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    nube = CloudWhisperSTT(http, model="whisper-1", language="es")
    respaldo_segs = (TranscriptSegment(0.0, 2.0, "Respaldo local"),)
    respaldo = _SpyFallbackProvider(segments=respaldo_segs)

    avisos: list[FallbackNotice] = []
    fallback = FallbackTranscription(nube, respaldo, on_fallback=avisos.append)
    resultado = fallback.transcribe_segments(audio)

    assert resultado == respaldo_segs
    assert respaldo.call_count == 1
    assert len(avisos) == 1
    assert "la respuesta no trae segmentos" in avisos[0].reason


def test_sin_red_connect_error_cae_al_respaldo(tmp_path: Path) -> None:
    """Caso 5: fallo de red (ConnectError) activa el respaldo local."""
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"RIFF dummy audio data")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("No se pudo resolver el host api.openai.com")

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    nube = CloudWhisperSTT(http, model="whisper-1", language="es")
    respaldo_segs = (TranscriptSegment(0.0, 1.5, "Local sin red"),)
    respaldo = _SpyFallbackProvider(segments=respaldo_segs)

    avisos: list[FallbackNotice] = []
    fallback = FallbackTranscription(nube, respaldo, on_fallback=avisos.append)
    resultado = fallback.transcribe_segments(audio)

    assert resultado == respaldo_segs
    assert respaldo.call_count == 1
    assert len(avisos) == 1
    assert "No se pudo resolver el host" in avisos[0].reason


def test_tiempo_agotado_timeout_cae_al_respaldo(tmp_path: Path) -> None:
    """Caso 6: tiempo de espera agotado (TimeoutException) activa el respaldo local."""
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"RIFF dummy audio data")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Tiempo de lectura agotado tras 120s")

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    nube = CloudWhisperSTT(http, model="whisper-1", language="es")
    respaldo_segs = (TranscriptSegment(0.0, 4.0, "Local tras timeout"),)
    respaldo = _SpyFallbackProvider(segments=respaldo_segs)

    avisos: list[FallbackNotice] = []
    fallback = FallbackTranscription(nube, respaldo, on_fallback=avisos.append)
    resultado = fallback.transcribe_segments(audio)

    assert resultado == respaldo_segs
    assert respaldo.call_count == 1
    assert len(avisos) == 1
    assert "Tiempo de lectura agotado" in avisos[0].reason


def test_sin_on_fallback_la_caida_opera_sin_excepcion(tmp_path: Path) -> None:
    """Caso 7: si on_fallback es None, la caída al respaldo se ejecuta sin errores."""
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"RIFF dummy audio data")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    nube = CloudWhisperSTT(http, model="whisper-1", language="es")
    respaldo_segs = (TranscriptSegment(0.0, 1.0, "Texto ok"),)
    respaldo = _SpyFallbackProvider(segments=respaldo_segs)

    fallback = FallbackTranscription(nube, respaldo, on_fallback=None)
    resultado = fallback.transcribe_segments(audio)

    assert resultado == respaldo_segs
    assert respaldo.call_count == 1


def test_fallback_transcribe_devuelve_texto_unido_usando_mismo_camino(tmp_path: Path) -> None:
    """Caso 8: transcribe devuelve texto unido y delega en transcribe_segments."""
    audio = tmp_path / "consulta.wav"
    audio.write_bytes(b"RIFF dummy audio data")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "Primera oración."},
                    {"start": 2.0, "end": 4.0, "text": "Segunda oración."},
                ]
            },
        )

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    nube = CloudWhisperSTT(http, model="whisper-1", language="es")
    respaldo = _SpyFallbackProvider()

    fallback = FallbackTranscription(nube, respaldo)
    texto = fallback.transcribe(audio)

    assert texto == "Primera oración. Segunda oración."
    assert respaldo.call_count == 0


def test_adaptador_envia_modelo_idioma_y_pide_segmentos_con_tiempos(tmp_path: Path) -> None:
    """Caso 9: comprueba sobre la petición capturada que se envía modelo,
    idioma y timestamp_granularities[]."""
    audio = tmp_path / "paciente_prueba.wav"
    audio.write_bytes(b"RIFF 1234 audio payload")

    peticion_capturada: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal peticion_capturada
        peticion_capturada = request
        return httpx.Response(
            200,
            json={"segments": [{"start": 0.0, "end": 1.0, "text": "Ok"}]},
        )

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
    )
    nube = CloudWhisperSTT(http, model="whisper-1", language="es")
    nube.transcribe_segments(audio)

    assert peticion_capturada is not None
    assert peticion_capturada.url.path == "/v1/audio/transcriptions"
    assert peticion_capturada.method == "POST"

    cuerpo = peticion_capturada.content
    assert b'name="model"\r\n\r\nwhisper-1' in cuerpo
    assert b'name="language"\r\n\r\nes' in cuerpo
    assert b'name="response_format"\r\n\r\nverbose_json' in cuerpo
    assert b'name="timestamp_granularities[]"\r\n\r\nsegment' in cuerpo
    assert b'filename="paciente_prueba.wav"' in cuerpo
    assert b"RIFF 1234 audio payload" in cuerpo


# ---------------------------------------------------------------------------
# Plan 70: Prompt en transcripción nube (§B.2) y puerta de formatos (§B.4b)
# ---------------------------------------------------------------------------


def test_cloud_whisper_envia_prompt_en_multipart(tmp_path: Path) -> None:
    """Caso 6 (§B.2): El prompt viaja en el cuerpo multipart cuando hay texto."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")
    peticion: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal peticion
        peticion = request
        return httpx.Response(200, json={"segments": [{"start": 0.0, "end": 1.0, "text": "Ok"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com")
    stt = CloudWhisperSTT(client, model="m", language="es", prompt="Prompt clínico")
    stt.transcribe_segments(audio)
    assert peticion is not None
    assert b'name="prompt"\r\n\r\nPrompt cl\xc3\xadnico' in peticion.content


def test_cloud_whisper_sin_prompt_no_incluye_campo_prompt(tmp_path: Path) -> None:
    """Caso 7 (§B.2): Sin prompt, la clave prompt no aparece en el cuerpo multipart."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")
    peticion: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal peticion
        peticion = request
        return httpx.Response(200, json={"segments": [{"start": 0.0, "end": 1.0, "text": "Ok"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com")
    stt = CloudWhisperSTT(client, model="m", language="es", prompt="")
    stt.transcribe_segments(audio)
    assert peticion is not None
    assert b'name="prompt"' not in peticion.content


def test_cloud_whisper_cuerpo_multipart_completo_con_prompt(tmp_path: Path) -> None:
    """Caso 8 (§B.2): El resto del cuerpo sigue exactamente igual al mandar prompt."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")
    peticion: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal peticion
        peticion = request
        return httpx.Response(200, json={"segments": [{"start": 0.0, "end": 1.0, "text": "Ok"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com")
    stt = CloudWhisperSTT(client, model="whisper-1", language="es", prompt="Prompt")
    stt.transcribe_segments(audio)
    assert peticion is not None
    cuerpo = peticion.content
    assert b'name="model"\r\n\r\nwhisper-1' in cuerpo
    assert b'name="language"\r\n\r\nes' in cuerpo
    assert b'name="response_format"\r\n\r\nverbose_json' in cuerpo
    assert b'name="timestamp_granularities[]"\r\n\r\nsegment' in cuerpo
    assert b'name="prompt"\r\n\r\nPrompt' in cuerpo


def test_cloud_whisper_formato_no_declarado_levanta_cloud_transcription_error(
    tmp_path: Path,
) -> None:
    """Caso 25 (§B.4b): Una extensión no declarada levanta CloudTranscriptionError
    antes de tocar el cliente HTTP (llamadas a post quedan en 0)."""
    audio = tmp_path / "archivo.m4a"
    audio.write_bytes(b"dummy")
    client = MagicMock()
    stt = CloudWhisperSTT(client, model="m", language="es", formats=frozenset({".wav", ".mp3"}))
    with pytest.raises(CloudTranscriptionError) as exc_info:
        stt.transcribe_segments(audio)
    assert ".m4a" in str(exc_info.value)
    assert client.post.call_count == 0


def test_cloud_whisper_extension_mayusculas_es_insensible(tmp_path: Path) -> None:
    """Caso 26 (§B.4b): La comparación de extensión es insensible a mayúsculas."""
    audio = tmp_path / "archivo.WAV"
    audio.write_bytes(b"dummy")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"segments": [{"start": 0.0, "end": 1.0, "text": "Ok"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com")
    stt = CloudWhisperSTT(client, model="m", language="es", formats=frozenset({".wav"}))
    res = stt.transcribe_segments(audio)
    assert len(res) == 1


def test_cloud_whisper_formats_vacio_no_bloquea_nada(tmp_path: Path) -> None:
    """Caso 27 (§B.4b): formats vacío no bloquea nada (sin restricción)."""
    audio = tmp_path / "archivo.custom"
    audio.write_bytes(b"dummy")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"segments": [{"start": 0.0, "end": 1.0, "text": "Ok"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com")
    stt = CloudWhisperSTT(client, model="m", language="es", formats=frozenset())
    res = stt.transcribe_segments(audio)
    assert len(res) == 1


def test_cloud_whisper_error_formato_nombra_extension_y_admitidas(tmp_path: Path) -> None:
    """Caso 28 (§B.4b): El mensaje de error nombra la extensión y las admitidas."""
    audio = tmp_path / "archivo.ogg"
    stt = CloudWhisperSTT(object(), model="m", language="es", formats=frozenset({".wav", ".mp3"}))
    with pytest.raises(CloudTranscriptionError) as exc_info:
        stt.transcribe_segments(audio)
    msg = str(exc_info.value)
    assert ".ogg" in msg
    assert ".mp3, .wav" in msg


def test_cloud_whisper_sin_extension_levanta_y_lo_nombra(tmp_path: Path) -> None:
    """Caso 29 (§B.4b): Sin extensión levanta y el mensaje dice 'sin extensión'."""
    audio = tmp_path / "audio_sin_ext"
    stt = CloudWhisperSTT(object(), model="m", language="es", formats=frozenset({".wav"}))
    with pytest.raises(CloudTranscriptionError) as exc_info:
        stt.transcribe_segments(audio)
    assert "sin extensión" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Plan 74: Modelo STT sin marcas de tiempo y partición unificada (§B.1)
# ---------------------------------------------------------------------------


def test_cloud_whisper_sin_tiempos_pide_json_sin_verbose_json_ni_granularities(
    tmp_path: Path,
) -> None:
    """Caso 1 (§B.1, M1): Con timestamps: false se pide json y no se mandan
    verbose_json ni timestamp_granularities[]."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")
    peticion: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal peticion
        peticion = request
        return httpx.Response(200, json={"text": "Texto simple."})

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com"
    )
    stt = CloudWhisperSTT(
        client, model="gpt-4o-mini-transcribe", language="es", timestamps=False
    )
    stt.transcribe_segments(audio)

    assert peticion is not None
    cuerpo = peticion.content
    assert b'name="response_format"\r\n\r\njson' in cuerpo
    assert b"verbose_json" not in cuerpo
    assert b"timestamp_granularities[]" not in cuerpo


def test_cloud_whisper_con_tiempos_cuerpo_intacto(tmp_path: Path) -> None:
    """Caso 2 (§B.1): Con timestamps: true, se mandan verbose_json y timestamp_granularities[]."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")
    peticion: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal peticion
        peticion = request
        return httpx.Response(
            200, json={"segments": [{"start": 0.0, "end": 1.0, "text": "Ok"}]}
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com"
    )
    stt = CloudWhisperSTT(client, model="whisper-1", language="es", timestamps=True)
    stt.transcribe_segments(audio)

    assert peticion is not None
    cuerpo = peticion.content
    assert b'name="response_format"\r\n\r\nverbose_json' in cuerpo
    assert b'name="timestamp_granularities[]"\r\n\r\nsegment' in cuerpo


def test_cloud_whisper_sin_tiempos_parte_text_en_frases_con_abreviaturas(
    tmp_path: Path,
) -> None:
    """Caso 3 (§B.1, M2): Sin tiempos, el text se parte en frases con las abreviaturas del pack:
    una respuesta con tres oraciones produce tres segmentos."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "text": (
                    "El paciente refiere cefalea. Niega fiebre en las últimas 24 horas. "
                    "Se indica reposo."
                )
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com"
    )
    stt = CloudWhisperSTT(
        client,
        model="gpt-4o-mini-transcribe",
        language="es",
        timestamps=False,
        abbreviations=frozenset({"dr.", "dña."}),
    )
    segmentos = stt.transcribe_segments(audio)

    assert len(segmentos) == 3
    assert segmentos[0].text == "El paciente refiere cefalea."
    assert segmentos[1].text == "Niega fiebre en las últimas 24 horas."
    assert segmentos[2].text == "Se indica reposo."


def test_cloud_whisper_sin_tiempos_segmentos_tienen_start_y_end_en_none(
    tmp_path: Path,
) -> None:
    """Caso 4 (§B.1, M3): Sin tiempos, los segmentos llevan start y end en None, no en 0.0."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "Frase de prueba sin marcas."})

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com"
    )
    stt = CloudWhisperSTT(
        client, model="gpt-4o-mini-transcribe", language="es", timestamps=False
    )
    segmentos = stt.transcribe_segments(audio)

    assert len(segmentos) == 1
    seg = segmentos[0]
    assert seg.text == "Frase de prueba sin marcas."
    assert seg.start is None
    assert seg.end is None
    assert seg.start != 0.0
    assert seg.end != 0.0


def test_cloud_whisper_con_tiempos_sin_segmentos_levanta(tmp_path: Path) -> None:
    """Caso 5 (§B.1, M4): Con timestamps: true, si la respuesta carece de segments
    levanta CloudTranscriptionError."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "Texto sin segmentos", "usage": {}})

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com"
    )
    stt = CloudWhisperSTT(client, model="whisper-1", language="es", timestamps=True)

    with pytest.raises(CloudTranscriptionError, match="la respuesta no trae segmentos"):
        stt.transcribe_segments(audio)


def test_cloud_whisper_sin_tiempos_sin_text_levanta(tmp_path: Path) -> None:
    """Caso 6 (§B.1): Sin tiempos y sin text (o vacío), levanta CloudTranscriptionError."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")

    def handler_vacio(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "   ", "usage": {}})

    client = httpx.Client(
        transport=httpx.MockTransport(handler_vacio), base_url="https://api.openai.com"
    )
    stt = CloudWhisperSTT(
        client, model="gpt-4o-mini-transcribe", language="es", timestamps=False
    )

    with pytest.raises(CloudTranscriptionError, match="la respuesta no trae texto"):
        stt.transcribe_segments(audio)

    def handler_sin_clave(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {}})

    client2 = httpx.Client(
        transport=httpx.MockTransport(handler_sin_clave), base_url="https://api.openai.com"
    )
    stt2 = CloudWhisperSTT(
        client2, model="gpt-4o-mini-transcribe", language="es", timestamps=False
    )

    with pytest.raises(CloudTranscriptionError, match="la respuesta no trae texto"):
        stt2.transcribe_segments(audio)


def test_cloud_whisper_abreviatura_del_pack_no_parte_frase(tmp_path: Path) -> None:
    """Caso 7 (§B.1, M5): Una abreviatura del pack no parte la frase en el camino de nube,
    igual que en el camino local."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF dummy")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "text": "El Dr. Juan Pérez evaluó la tensión. Recomendó control semanal."
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com"
    )
    stt = CloudWhisperSTT(
        client,
        model="gpt-4o-mini-transcribe",
        language="es",
        timestamps=False,
        abbreviations=frozenset({"dr.", "dra."}),
    )
    segmentos = stt.transcribe_segments(audio)

    assert len(segmentos) == 2
    assert segmentos[0].text == "El Dr. Juan Pérez evaluó la tensión."
    assert segmentos[1].text == "Recomendó control semanal."
