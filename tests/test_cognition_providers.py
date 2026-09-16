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

import json

import httpx
import pytest

from aida.cognition.briefing import UNIVERSAL_SYSTEM_PROMPT, SpokenReply
from aida.cognition.providers import (
    ClaudeProvider,
    GeminiProvider,
    OllamaProvider,
    build_provider,
)


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
            lambda req: httpx.Response(
                200, json={"response": json.dumps({"extra": "invalido"})}
            )
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
