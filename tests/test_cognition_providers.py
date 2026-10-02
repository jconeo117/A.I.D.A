"""Pruebas de los adaptadores de proveedores de cognición (HTTP MockTransport).

Cubre §6.2 de `.claude/workflow/24-plan-u7.md`:
1-4. ClaudeProvider: respuesta válida, error 500, texto no-JSON, esquema inválido.
5. GeminiProvider y OllamaProvider en los mismos escenarios 1-4.
6. build_provider("claude") sin ANTHROPIC_API_KEY devuelve None.
7. build_provider("gemini") con GEMINI_API_KEY devuelve GeminiProvider.
8. build_provider("ollama") sin variables devuelve OllamaProvider.
9. build_provider("no_existe") devuelve None.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from aida.arms.health.consultation import ConsultationResult
from aida.arms.health.export import export_consultation
from aida.arms.health.note import (
    ClinicalNote,
    NoteSettings,
    RenderedNote,
    build_note,
    load_note_settings,
)
from aida.arms.health.sections import SectionSpec
from aida.cognition.briefing import UNIVERSAL_SYSTEM_PROMPT, SpokenReply
from aida.cognition.fallback import FallbackCognition
from aida.cognition.providers import (
    ClaudeProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    _strip_ollama_unsupported_keywords,
    available_providers,
    build_provider,
)
from aida.engine.ontology import AgentId
from aida.run import _cognition


def test_claude_think_valido_devuelve_spoken_reply() -> None:
    """Caso 1: ClaudeProvider procesa una respuesta JSON conforme a SpokenReply."""
    cuerpo_modelo = json.dumps({"text": "Respuesta simulada de Claude."})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        return httpx.Response(200, json={"content": [{"text": cuerpo_modelo}]})

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.anthropic.com"
    )
    provider = ClaudeProvider(client)
    resultado = provider.think("Briefing de prueba", SpokenReply)

    assert isinstance(resultado, SpokenReply)
    assert resultado.text == "Respuesta simulada de Claude."


def test_claude_think_error_500_devuelve_none() -> None:
    """Caso 2: ClaudeProvider devuelve None ante error HTTP 500 del servidor sin levantar."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.anthropic.com"
    )
    provider = ClaudeProvider(client)
    assert provider.think("Briefing de prueba", SpokenReply) is None


def test_claude_think_texto_no_es_json_valido_devuelve_none() -> None:
    """Caso 3: ClaudeProvider devuelve None si el texto recibido no es JSON válido."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"text": "texto plano sin estructura json"}]})

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.anthropic.com"
    )
    provider = ClaudeProvider(client)
    assert provider.think("Briefing de prueba", SpokenReply) is None

    # Cobertura de errores de decodificación JSONDecodeError que heredan de ValueError (M6)
    class _DecodeErrorSchema:
        @classmethod
        def model_validate_json(cls, raw: str) -> None:
            raise json.JSONDecodeError("JSON no válido", raw, 0)

        @classmethod
        def model_json_schema(cls) -> dict[str, str]:
            return {}

    assert provider.think("Briefing de prueba", _DecodeErrorSchema) is None


def test_claude_think_json_no_cumple_spoken_reply_devuelve_none() -> None:
    """Caso 4: ClaudeProvider devuelve None si el JSON viola el esquema SpokenReply."""

    def crear_provider(cuerpo_texto: str) -> ClaudeProvider:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"content": [{"text": cuerpo_texto}]})

        client = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://api.anthropic.com"
        )
        return ClaudeProvider(client)

    # Violación 1: campo extra no permitido (extra=forbid)
    p1 = crear_provider(json.dumps({"text": "ok", "campo_extra": "prohibido"}))
    assert p1.think("Briefing", SpokenReply) is None

    # Violación 2: falta el campo obligatorio 'text'
    p2 = crear_provider(json.dumps({"otro_campo": "valor"}))
    assert p2.think("Briefing", SpokenReply) is None

    # Violación 3: longitud mínima no satisfecha
    p3 = crear_provider(json.dumps({"text": ""}))
    assert p3.think("Briefing", SpokenReply) is None


def test_gemini_think_valido_devuelve_spoken_reply() -> None:
    """Caso 5a: GeminiProvider procesa una respuesta JSON conforme a SpokenReply."""
    cuerpo_modelo = json.dumps({"text": "Respuesta simulada de Gemini."})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": cuerpo_modelo}]}}]},
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com",
    )
    provider = GeminiProvider(client)
    resultado = provider.think("Briefing de prueba", SpokenReply)

    assert isinstance(resultado, SpokenReply)
    assert resultado.text == "Respuesta simulada de Gemini."


def test_gemini_think_errores_devuelven_none() -> None:
    """Caso 5b: GeminiProvider maneja defensivamente 500, no-JSON y esquema inválido."""
    # Sub-caso 500
    c_500 = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(500)),
        base_url="https://generativelanguage.googleapis.com",
    )
    assert GeminiProvider(c_500).think("Briefing", SpokenReply) is None

    # Sub-caso texto no JSON
    c_no_json = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "no es json"}]}}]},
            )
        ),
        base_url="https://generativelanguage.googleapis.com",
    )
    assert GeminiProvider(c_no_json).think("Briefing", SpokenReply) is None

    # Sub-caso esquema inválido
    c_invalido = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps({"invalido": True})}]}}
                    ]
                },
            )
        ),
        base_url="https://generativelanguage.googleapis.com",
    )
    assert GeminiProvider(c_invalido).think("Briefing", SpokenReply) is None


def test_ollama_think_valido_devuelve_spoken_reply() -> None:
    """Caso 5c: OllamaProvider procesa una respuesta JSON conforme a SpokenReply."""
    cuerpo_modelo = json.dumps({"text": "Respuesta simulada de Ollama."})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        return httpx.Response(200, json={"response": cuerpo_modelo})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://localhost:11434",
    )
    provider = OllamaProvider(client)
    resultado = provider.think("Briefing de prueba", SpokenReply)

    assert isinstance(resultado, SpokenReply)
    assert resultado.text == "Respuesta simulada de Ollama."


def test_ollama_think_errores_devuelven_none() -> None:
    """Caso 5d: OllamaProvider maneja defensivamente 500, no-JSON y esquema inválido."""
    # Sub-caso 500
    c_500 = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(500)),
        base_url="http://localhost:11434",
    )
    assert OllamaProvider(c_500).think("Briefing", SpokenReply) is None

    # Sub-caso texto no JSON
    c_no_json = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"response": "no es json"})
        ),
        base_url="http://localhost:11434",
    )
    assert OllamaProvider(c_no_json).think("Briefing", SpokenReply) is None

    # Sub-caso esquema inválido
    c_invalido = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"response": json.dumps({"extra": "invalido"})})
        ),
        base_url="http://localhost:11434",
    )
    assert OllamaProvider(c_invalido).think("Briefing", SpokenReply) is None


def test_build_provider_claude_sin_credencial_devuelve_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 6: build_provider('claude') sin ANTHROPIC_API_KEY devuelve None."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert build_provider("claude") is None


def test_build_provider_gemini_con_credencial_devuelve_proveedor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 7: build_provider('gemini') con GEMINI_API_KEY devuelve un GeminiProvider."""
    monkeypatch.setenv("GEMINI_API_KEY", "clave-gemini-simulada")
    proveedor = build_provider("gemini")
    assert isinstance(proveedor, GeminiProvider)


def test_build_provider_ollama_sin_entorno_devuelve_proveedor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 8: build_provider('ollama') no exige variables de entorno y devuelve OllamaProvider."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    proveedor = build_provider("ollama")
    assert isinstance(proveedor, OllamaProvider)


def test_build_provider_no_existe_devuelve_none() -> None:
    """Caso 9: build_provider con un nombre de proveedor no soportado devuelve None."""
    assert build_provider("no_existe") is None


def test_claude_request_payload_contains_system_prompt_and_briefing() -> None:
    """I-063 / V11 / V15: verifica que la llamada a Claude incluya SYSTEM_PROMPT
    en el campo 'system' y el briefing textual en el contenido de 'messages'.
    """
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"content": [{"text": json.dumps({"text": "OK"})}]},
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.anthropic.com",
    )
    provider = ClaudeProvider(client)
    briefing_text = "Briefing operacional detallado para Claude."
    provider.think(briefing_text, SpokenReply)

    system_field = str(captured_payload.get("system", ""))
    assert UNIVERSAL_SYSTEM_PROMPT in system_field
    messages = captured_payload.get("messages")
    assert isinstance(messages, list)
    assert len(messages) > 0
    assert messages[0].get("content") == briefing_text


def test_gemini_request_payload_contains_system_instruction_path_and_briefing() -> None:
    """I-063 / V10 / V14: verifica que la petición a Gemini viaje a la ruta esperada,
    con SYSTEM_PROMPT en system_instruction y el briefing en contents.
    """
    captured_payload: dict[str, object] = {}
    captured_path: str = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload, captured_path
        captured_path = request.url.path
        captured_payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": json.dumps({"text": "OK"})}]}}]},
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com",
    )
    provider = GeminiProvider(client)
    briefing_text = "Briefing operacional detallado para Gemini."
    provider.think(briefing_text, SpokenReply)

    assert captured_path == "/v1beta/models/gemini-2.0-flash:generateContent"
    system_instruction = captured_payload.get("system_instruction")
    assert isinstance(system_instruction, dict)
    parts = system_instruction.get("parts")
    assert isinstance(parts, list)
    assert parts[0].get("text") == UNIVERSAL_SYSTEM_PROMPT

    contents = captured_payload.get("contents")
    assert isinstance(contents, list)
    content_parts = contents[0].get("parts")
    assert isinstance(content_parts, list)
    assert content_parts[0].get("text") == briefing_text


def test_ollama_request_payload_contains_system_prompt_and_briefing() -> None:
    """I-063 / V12: verifica que la llamada a Ollama integre tanto SYSTEM_PROMPT
    como el briefing en el campo textual prompt.
    """
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"response": json.dumps({"text": "OK"})},
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://localhost:11434",
    )
    provider = OllamaProvider(client)
    briefing_text = "Briefing operacional detallado para Ollama."
    provider.think(briefing_text, SpokenReply)

    prompt_field = str(captured_payload.get("prompt", ""))
    assert UNIVERSAL_SYSTEM_PROMPT in prompt_field
    assert briefing_text in prompt_field


def test_build_provider_gemini_without_key_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-063 / V13: build_provider('gemini') sin variable GEMINI_API_KEY
    debe retornar None de inmediato sin intentar crear clientes incompletos.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert build_provider("gemini") is None


def test_build_provider_gemini_uses_header_instead_of_query_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-065: la credencial de Gemini debe transmitirse exclusivamente mediante
    la cabecera x-goog-api-key para impedir su filtrado en registros HTTP y trazas.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key-123")
    provider = build_provider("gemini")
    assert isinstance(provider, GeminiProvider)
    assert provider._client.headers.get("x-goog-api-key") == "test-secret-key-123"
    assert "key" not in provider._client.params


def test_ollama_strip_quita_longitudes_del_schema_real() -> None:
    """§B.1 Caso 1: _strip_ollama_unsupported_keywords elimina minLength/maxLength
    del schema de SpokenReply en todos los niveles y preserva los demás atributos.
    """
    schema = SpokenReply.model_json_schema()
    assert "minLength" in json.dumps(schema) or "maxLength" in json.dumps(schema)

    resultado = _strip_ollama_unsupported_keywords(schema)
    serializado = json.dumps(resultado)

    assert "minLength" not in serializado
    assert "maxLength" not in serializado
    assert resultado.get("type") == "object"
    assert resultado.get("title") == "SpokenReply"
    assert resultado.get("required") == ["text"]
    assert resultado.get("additionalProperties") is False
    properties = resultado.get("properties")
    assert isinstance(properties, dict)
    assert properties["text"]["type"] == "string"


def test_ollama_strip_recorre_mapas_anidados_y_listas() -> None:
    """§B.1 Caso 2: _strip_ollama_unsupported_keywords recorre mapas anidados y listas,
    preservando elementos no-diccionario y limpiando subclaves minLength/maxLength.
    """
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "minLength": 1}},
        "anyOf": [{"type": "string", "maxLength": 3}, "literal"],
    }
    esperado = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "anyOf": [{"type": "string"}, "literal"],
    }
    assert _strip_ollama_unsupported_keywords(schema) == esperado


def test_ollama_payload_serializado_no_lleva_longitudes() -> None:
    """§B.1 Caso 3: la petición HTTP a Ollama no incluye minLength/maxLength en sus bytes
    y preserva la estructura de schema format requerida.
    """
    contenido_bytes = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal contenido_bytes
        contenido_bytes = request.content
        return httpx.Response(200, json={"response": json.dumps({"text": "ok"})})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://localhost:11434",
    )
    provider = OllamaProvider(client)
    resultado = provider.think("Briefing", SpokenReply)

    assert isinstance(resultado, SpokenReply)
    assert resultado.text == "ok"
    assert b"minLength" not in contenido_bytes
    assert b"maxLength" not in contenido_bytes
    parsed_payload = json.loads(contenido_bytes)
    assert parsed_payload["format"]["properties"]["text"]["type"] == "string"


def test_ollama_think_longitud_fuera_de_rango_devuelve_none() -> None:
    """§B.1 Caso 4: respuestas que violen min_length=1 o max_length=2000 son rechazadas
    por _validate y retornan None a pesar de no estar en la gramática.
    """
    client_vacio = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"response": json.dumps({"text": ""})})
        ),
        base_url="http://localhost:11434",
    )
    assert OllamaProvider(client_vacio).think("Briefing", SpokenReply) is None

    client_largo = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"response": json.dumps({"text": "x" * 2001})})
        ),
        base_url="http://localhost:11434",
    )
    assert OllamaProvider(client_largo).think("Briefing", SpokenReply) is None


def test_openai_request_payload_and_schema_format() -> None:
    """Casos 19 y 20 (§B.3b, M14): El cuerpo capturado de OpenAI trae el modelo declarado,
    UNIVERSAL_SYSTEM_PROMPT como sistema, el briefing como usuario, response_format con
    el esquema, y el esquema va filtrado sin minLength ni maxLength (I-082)."""
    captured_payload: dict[str, Any] = {}
    contenido_bytes = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload, contenido_bytes
        assert request.url.path == "/v1/chat/completions"
        contenido_bytes = request.content
        captured_payload = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"text": "ok"})}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com")
    provider = OpenAIProvider(client, model="gpt-4.1-mini")
    briefing = "Briefing operacional para OpenAI"
    res = provider.think(briefing, SpokenReply)

    assert isinstance(res, SpokenReply)
    assert captured_payload["model"] == "gpt-4.1-mini"
    messages = captured_payload["messages"]
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": UNIVERSAL_SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": briefing}

    rf = captured_payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "SpokenReply"
    assert rf["json_schema"]["strict"] is False

    # Caso 20 / M14: Esquema filtrado sin minLength ni maxLength
    assert b"minLength" not in contenido_bytes
    assert b"maxLength" not in contenido_bytes
    schema_enviado = rf["json_schema"]["schema"]
    assert "minLength" not in json.dumps(schema_enviado)
    assert "maxLength" not in json.dumps(schema_enviado)


def test_openai_reutiliza_filtro_de_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 21 (§B.3b, M15): OpenAIProvider reutiliza la misma función de filtrado
    que Ollama (_strip_ollama_unsupported_keywords). No escribe un segundo filtro propio."""
    import aida.cognition.providers as cp

    llamadas: list[Any] = []
    filtro_original = cp._strip_ollama_unsupported_keywords

    def filtro_espia(schema: Any) -> Any:
        llamadas.append(schema)
        return filtro_original(schema)

    monkeypatch.setattr(cp, "_strip_ollama_unsupported_keywords", filtro_espia)

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps({"text": "ok"})}}]}
            )
        ),
        base_url="https://api.openai.com",
    )
    provider = OpenAIProvider(client)
    res = provider.think("Briefing", SpokenReply)
    assert res is not None
    assert len(llamadas) > 0, "OpenAIProvider no llamó a _strip_ollama_unsupported_keywords"


def test_build_provider_openai_uses_header_instead_of_url_or_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 22 (§B.3b, M16): La credencial de OpenAI va en la cabecera Authorization (Bearer),
    nunca en la URL ni en el cuerpo de la petición (I-065)."""
    monkeypatch.setenv("AIDA_STT_API_KEY", "clave-secreta-openai-999")
    provider = build_provider("openai")
    assert isinstance(provider, OpenAIProvider)
    assert provider._client.headers.get("Authorization") == "Bearer clave-secreta-openai-999"
    assert "key" not in provider._client.params
    assert "api_key" not in provider._client.params

    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"text": "ok"})}}]}
        )

    provider._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com",
        headers={"Authorization": "Bearer clave-secreta-openai-999"},
    )
    provider.think("briefing", SpokenReply)
    assert captured_request is not None
    assert captured_request.headers.get("Authorization") == "Bearer clave-secreta-openai-999"
    assert "clave-secreta-openai-999" not in str(captured_request.url)
    assert "clave-secreta-openai-999" not in captured_request.content.decode()


def test_openai_think_valido_y_esquema_invalido_devuelve_none() -> None:
    """Caso 23 (§B.3b): Una respuesta válida se valida y devuelve objeto;
    una que viola el esquema devuelve None sin levantar excepción."""
    # Válido
    client_ok = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps({"text": "ok"})}}]}
            )
        ),
        base_url="https://api.openai.com",
    )
    p_ok = OpenAIProvider(client_ok)
    res_ok = p_ok.think("briefing", SpokenReply)
    assert isinstance(res_ok, SpokenReply)
    assert res_ok.text == "ok"

    # Esquema inválido (falta 'text')
    client_inv = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps({"otro": 123})}}]}
            )
        ),
        base_url="https://api.openai.com",
    )
    p_inv = OpenAIProvider(client_inv)
    assert p_inv.think("briefing", SpokenReply) is None

    # Respuesta no JSON
    client_no_json = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"choices": [{"message": {"content": "no es json"}}]}
            )
        ),
        base_url="https://api.openai.com",
    )
    p_no_json = OpenAIProvider(client_no_json)
    assert p_no_json.think("briefing", SpokenReply) is None


def test_openai_think_error_http_devuelve_none_sin_propagar() -> None:
    """Caso 24 (§B.3b, M17): Un error HTTP (500, 401, 429) devuelve None y no se propaga."""
    for codigo in (401, 429, 500):
        client = httpx.Client(
            transport=httpx.MockTransport(lambda req, c=codigo: httpx.Response(c, text="Error")),
            base_url="https://api.openai.com",
        )
        p = OpenAIProvider(client)
        assert p.think("briefing", SpokenReply) is None


def test_build_provider_openai_credenciales_y_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 25 (§B.3b, M19): La credencial se lee de AIDA_STT_API_KEY y, si falta,
    de OPENAI_API_KEY; sin ninguna de las dos, build_provider('openai') devuelve None."""
    # Sin ninguna credencial
    monkeypatch.delenv("AIDA_STT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert build_provider("openai") is None

    # Solo AIDA_STT_API_KEY
    monkeypatch.setenv("AIDA_STT_API_KEY", "stt-key-123")
    prov1 = build_provider("openai")
    assert isinstance(prov1, OpenAIProvider)
    assert prov1._client.headers.get("Authorization") == "Bearer stt-key-123"

    # Solo OPENAI_API_KEY (AIDA_STT_API_KEY ausente)
    monkeypatch.delenv("AIDA_STT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-456")
    prov2 = build_provider("openai")
    assert isinstance(prov2, OpenAIProvider)
    assert prov2._client.headers.get("Authorization") == "Bearer openai-key-456"

    # Ambas presentes: AIDA_STT_API_KEY tiene prioridad
    monkeypatch.setenv("AIDA_STT_API_KEY", "prioritaria-789")
    prov3 = build_provider("openai")
    assert isinstance(prov3, OpenAIProvider)
    assert prov3._client.headers.get("Authorization") == "Bearer prioritaria-789"


def test_build_provider_openai_recibe_provider_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 26 (§B.3b, M18): provider_model llega al adaptador OpenAIProvider
    y viaja en el payload de la petición HTTP."""
    monkeypatch.setenv("AIDA_STT_API_KEY", "dummy-key")

    captured_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        captured_models.append(data["model"])
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"text": "ok"})}}]}
        )

    # Modelo por defecto vs modelo personalizado
    p1 = build_provider("openai", model="gpt-4.1-mini")
    p2 = build_provider("openai", model="gpt-5.4-custom")
    assert isinstance(p1, OpenAIProvider)
    assert isinstance(p2, OpenAIProvider)

    p1._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com"
    )
    p2._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.com"
    )

    p1.think("b1", SpokenReply)
    p2.think("b2", SpokenReply)

    assert captured_models == ["gpt-4.1-mini", "gpt-5.4-custom"]


def test_available_providers_solo_lista_los_que_pueden_responder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 27 (§B.3b, M20): available_providers solo lista los que tienen
    credencial o son locales."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("AIDA_STT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Sin ninguna clave de nube, solo ollama
    assert available_providers() == ("ollama",)

    # Con clave de OpenAI, aparecen openai y ollama
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert available_providers() == ("openai", "ollama")

    # Con todas las claves
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert available_providers() == ("openai", "claude", "gemini", "ollama")


def test_available_providers_orden_declarado_openai_primero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 28 (§B.3b, M21): El orden devuelto es el declarado (openai primero),
    no ordenado alfabéticamente."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provs = available_providers()
    # Si fuera alfabético, ollama iría antes de openai ('ollama' < 'openai')
    assert provs == ("openai", "ollama")
    assert provs[0] == "openai"


def test_note_settings_provider_decide_quien_redacta_y_env_lo_pisa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso 29 (§B.3b, M22): provider del pack (note.yaml) decide quién redacta,
    y AIDA_LLM_PROVIDER lo pisa para esa ejecución."""
    settings_openai = NoteSettings(
        system_prompt="p",
        provider="openai",
        provider_model="gpt-4.1-mini",
        field_heading="### {}",
        provenance_template="{}",
        fallback_notice="fb",
    )
    settings_gemini = NoteSettings(
        system_prompt="p",
        provider="gemini",
        provider_model="gemini-2.0-flash",
        field_heading="### {}",
        provenance_template="{}",
        fallback_notice="fb",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "key-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "key-gemini")
    monkeypatch.delenv("AIDA_LLM_PROVIDER", raising=False)

    args = argparse.Namespace(llm=None)

    prov1 = _cognition(args, settings_openai)
    if isinstance(prov1, FallbackCognition):
        assert isinstance(prov1._primary, OpenAIProvider)
    else:
        assert isinstance(prov1, OpenAIProvider)

    prov2 = _cognition(args, settings_gemini)
    if isinstance(prov2, FallbackCognition):
        assert isinstance(prov2._primary, GeminiProvider)
    else:
        assert isinstance(prov2, GeminiProvider)

    # Variable de entorno AIDA_LLM_PROVIDER pisa note.yaml
    monkeypatch.setenv("AIDA_LLM_PROVIDER", "gemini")
    prov3 = _cognition(args, settings_openai)
    if isinstance(prov3, FallbackCognition):
        assert isinstance(prov3._primary, GeminiProvider)
    else:
        assert isinstance(prov3, GeminiProvider)


def test_rendered_note_provider_provenance() -> None:
    """Caso 30 (§B.3b, M23): RenderedNote.provider dice quién redactó ('openai/gpt-4.1-mini')
    y es None en el respaldo literal."""
    settings = load_note_settings()
    spec = SectionSpec("motivo_consulta", "Motivo", (), codable=True, in_note=True)
    grouped = ((spec, ("hecho",)),)

    class _DoubleProvider:
        def think(self, briefing: str, schema: type) -> object | None:
            return ClinicalNote(fields={"motivo_consulta": "Redactado"})

    r_prosa = build_note(
        grouped_sections=grouped,
        settings=settings,
        providers=((f"{settings.provider}/{settings.provider_model}", _DoubleProvider()),),
    )
    assert r_prosa.provider == f"{settings.provider}/{settings.provider_model}"

    r_respaldo = build_note(grouped_sections=grouped, settings=settings, providers=())
    assert r_respaldo.provider is None


def test_export_escribe_procedencia_cuando_hay_nota_y_no_en_respaldo(tmp_path: Path) -> None:
    """Caso 31 (§B.3b, M13, M24): El export escribe la procedencia cuando hay nota redactada,
    y no la escribe cuando es respaldo literal (en su lugar lleva fallback_notice)."""
    settings = load_note_settings()
    spec = SectionSpec("motivo_consulta", "Motivo", (), codable=True, in_note=True)

    # Con nota redactada
    note_redactada = RenderedNote(
        fields=((spec, "Texto redactado por modelo"),),
        redacted=True,
        provider="openai/gpt-4.1-mini",
    )
    res_redactada = ConsultationResult(
        owner=AgentId("sujeto_export_prov"),
        transcript="",
        history_excerpt="",
        matched_codes=(),
        summary_markdown="## Resumen",
        note=note_redactada,
    )
    archivos1 = export_consultation(
        res_redactada,
        destination=tmp_path / "redactada.md",
        moment=datetime(2026, 9, 30, 10, 0),
        include_model=False,
    )
    md1 = archivos1.summary.read_text(encoding="utf-8")
    esperado_prov = settings.provenance_template.format("openai", "gpt-4.1-mini")
    assert esperado_prov in md1
    assert settings.fallback_notice not in md1

    # Con respaldo literal
    note_respaldo = RenderedNote(
        fields=((spec, "Frase cruda"),),
        redacted=False,
        provider=None,
    )
    res_respaldo = ConsultationResult(
        owner=AgentId("sujeto_export_prov"),
        transcript="",
        history_excerpt="",
        matched_codes=(),
        summary_markdown="## Resumen",
        note=note_respaldo,
    )
    archivos2 = export_consultation(
        res_respaldo,
        destination=tmp_path / "respaldo.md",
        moment=datetime(2026, 9, 30, 10, 0),
        include_model=False,
    )
    md2 = archivos2.summary.read_text(encoding="utf-8")
    assert "_Redactado por" not in md2
    assert settings.fallback_notice in md2


# ---------------------------------------------------------------------------
# Plan 79 (§B.4): Propiedad model en los cuatro proveedores
# ---------------------------------------------------------------------------


def test_providers_exponen_propiedad_model() -> None:
    """Caso 14 (§B.4): Los cuatro proveedores exponen su modelo a través de .model.
    Con un httpx.Client():
    - OllamaProvider(c).model == "llama3.2" y con model="x" -> "x"
    - ClaudeProvider(c).model == "claude-3-5-haiku-latest" y con model="x" -> "x"
    - GeminiProvider(c).model == "gemini-2.5-flash" y con model="x" -> "x"
    - OpenAIProvider(c).model == "gpt-4.1-mini" y con model="x" -> "x"
    """
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    assert OllamaProvider(client).model == "llama3.2"
    assert OllamaProvider(client, model="custom-ollama").model == "custom-ollama"

    assert ClaudeProvider(client).model == "claude-3-5-haiku-latest"
    assert ClaudeProvider(client, model="custom-claude").model == "custom-claude"

    assert GeminiProvider(client).model == "gemini-2.0-flash"
    assert GeminiProvider(client, model="custom-gemini").model == "custom-gemini"

    assert OpenAIProvider(client).model == "gpt-4.1-mini"
    assert OpenAIProvider(client, model="custom-openai").model == "custom-openai"

