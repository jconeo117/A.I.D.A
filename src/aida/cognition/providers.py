"""Tres proveedores por patrón strategy, todos sobre HTTP con `httpx`.

Cada uno recibe su cliente `httpx.Client` ya construido, no lo crea él
mismo: así un test lo sustituye por un `httpx.MockTransport` sin tocar la
red, y quien compone la aplicación decide timeouts y reintentos en un solo
sitio.

Los nombres exactos de campo de cada API (`model`, el cuerpo de la
petición, dónde viene el texto en la respuesta) se han verificado contra la
documentación pública de cada proveedor a la fecha de escritura de este
plan. Si al implementar difieren, es una desviación a reportar en el
entregable — no se improvisa en silencio.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import cast

import httpx
from pydantic import BaseModel, ValidationError

from aida.cognition.briefing import UNIVERSAL_SYSTEM_PROMPT


def _validate(raw: str, schema: type) -> object | None:
    try:
        return cast(type[BaseModel], schema).model_validate_json(raw)
    except (ValidationError, ValueError, AttributeError, TypeError):
        return None


class ClaudeProvider:
    """`aida.engine.ports.CognitionProvider` sobre la API de mensajes de
    Anthropic."""

    def __init__(self, client: httpx.Client, *, model: str = "claude-3-5-haiku-latest") -> None:
        self._client = client
        self._model = model

    def think(self, briefing: str, schema: type) -> object | None:
        try:
            schema_json = cast(type[BaseModel], schema).model_json_schema()
            response = self._client.post(
                "/v1/messages",
                json={
                    "model": self._model,
                    "max_tokens": 512,
                    "system": (
                        f"{UNIVERSAL_SYSTEM_PROMPT}\n"
                        f"Responde SOLO con JSON válido de esta forma: {schema_json}"
                    ),
                    "messages": [{"role": "user", "content": briefing}],
                },
            )
            response.raise_for_status()
            text = response.json()["content"][0]["text"]
            return _validate(text, schema)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None


class GeminiProvider:
    """`CognitionProvider` sobre la API de generación de contenido de
    Gemini."""

    def __init__(self, client: httpx.Client, *, model: str = "gemini-2.0-flash") -> None:
        self._client = client
        self._model = model

    def think(self, briefing: str, schema: type) -> object | None:
        try:
            response = self._client.post(
                f"/v1beta/models/{self._model}:generateContent",
                json={
                    "system_instruction": {"parts": [{"text": UNIVERSAL_SYSTEM_PROMPT}]},
                    "contents": [{"parts": [{"text": briefing}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": cast(type[BaseModel], schema).model_json_schema(),
                    },
                },
            )
            response.raise_for_status()
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return _validate(text, schema)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None


def _strip_ollama_unsupported_keywords(schema: Mapping[str, object]) -> dict[str, object]:
    """Ollama compila el `format` de una petición a una gramática GBNF, y con
    `minLength`/`maxLength` presentes al menos un modelo real revienta
    (`"Failed to initialize samplers: failed to parse grammar"`, medido el
    2026-09-16 contra `qwen2.5-coder:7b-instruct-q4_K_M`). pydantic las añade
    en cuanto un campo usa `Field(min_length=..., max_length=...)`. Quitarlas
    no relaja el contrato: `_validate` sigue comprobando la respuesta contra el
    modelo completo. Solo se aplica a lo que recibe Ollama.
    """
    cleaned: dict[str, object] = {}
    for key, value in schema.items():
        if key in ("minLength", "maxLength"):
            continue
        if isinstance(value, Mapping):
            cleaned[key] = _strip_ollama_unsupported_keywords(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _strip_ollama_unsupported_keywords(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


class OllamaProvider:
    """`CognitionProvider` sobre un servidor Ollama local. Sin API key: la
    única de las tres que puede correr sin ninguna credencial."""

    def __init__(self, client: httpx.Client, *, model: str = "llama3.2") -> None:
        self._client = client
        self._model = model

    def think(self, briefing: str, schema: type) -> object | None:
        try:
            response = self._client.post(
                "/api/generate",
                json={
                    "model": self._model,
                    "prompt": f"{UNIVERSAL_SYSTEM_PROMPT}\n\n{briefing}",
                    "format": _strip_ollama_unsupported_keywords(
                        cast(type[BaseModel], schema).model_json_schema()
                    ),
                    "stream": False,
                },
            )
            response.raise_for_status()
            text = response.json()["response"]
            return _validate(text, schema)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None


class OpenAIProvider:
    """`CognitionProvider` sobre la API de OpenAI. Se añade para poder medir la
    nota con la credencial que el despliegue ya tiene —la misma que transcribe—,
    no porque sea la elección definitiva: esa es un ajuste de datos.

    **Aviso de verificación, igual que en U7 y en el plan 67:** la forma exacta
    de la petición y el nombre del campo donde viene el texto se comprueban
    contra la documentación vigente del proveedor **al implementar**. Lo que
    sigue es el contrato que debe cumplir.
    """

    def __init__(self, client: httpx.Client, *, model: str = "gpt-4.1-mini") -> None:
        self._client = client
        self._model = model

    def think(self, briefing: str, schema: type) -> object | None:
        try:
            response = self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": UNIVERSAL_SYSTEM_PROMPT},
                        {"role": "user", "content": briefing},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": cast(type[BaseModel], schema).__name__,
                            "schema": _strip_ollama_unsupported_keywords(
                                cast(type[BaseModel], schema).model_json_schema()
                            ),
                            "strict": False,
                        },
                    },
                },
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            return _validate(text, schema)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, AttributeError):
            return None


_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com",
    "claude": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "ollama": "http://localhost:11434",
}


def build_provider(
    name: str, *, model: str | None = None
) -> ClaudeProvider | GeminiProvider | OllamaProvider | OpenAIProvider | None:
    """Construye el proveedor pedido, leyendo su API key del entorno.

    `None` si el proveedor no se reconoce, o si le falta la credencial que
    necesita — nunca revienta: sin proveedor, `Session` cae a
    `fallback_reply` y el asistente sigue funcionando sin voz.

    `model` pisa el modelo por defecto del adaptador cuando el pack declara
    uno propio (`note.yaml:provider_model`): dos mediciones separadas por
    semanas solo son comparables si consta con qué modelo se hicieron.
    """
    base_url = _DEFAULT_BASE_URLS.get(name)
    if base_url is None:
        return None

    if name == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        client = httpx.Client(
            base_url=base_url, headers={"x-api-key": key, "anthropic-version": "2023-06-01"}
        )
        return ClaudeProvider(client) if model is None else ClaudeProvider(client, model=model)

    if name == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return None
        # httpx registra la URL completa en logs INFO y HTTPStatusError;
        # la clave en cabecera evita fugarla.
        client = httpx.Client(base_url=base_url, headers={"x-goog-api-key": key})
        return GeminiProvider(client) if model is None else GeminiProvider(client, model=model)

    if name == "ollama":
        client = httpx.Client(base_url=os.environ.get("OLLAMA_HOST", base_url))
        return OllamaProvider(client) if model is None else OllamaProvider(client, model=model)

    if name == "openai":
        key = os.environ.get("AIDA_STT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        # La clave en cabecera, nunca en la URL (I-065): httpx registra la URL
        # completa en logs INFO y en HTTPStatusError.
        #
        # `timeout` explícito: el valor por defecto de `httpx` es 5 s, y
        # redactar una nota de una consulta real (~1.500 tokens de entrada)
        # lo supera con holgura — medido, el servidor responde en unos
        # segundos, pero no en cinco. Sin esto, `think` caía al respaldo en
        # **toda** consulta real, no por fallo del proveedor sino por un
        # timeout que nadie había declarado.
        client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(60.0),
        )
        return OpenAIProvider(client) if model is None else OpenAIProvider(client, model=model)

    return None


def available_providers() -> tuple[str, ...]:
    """Los proveedores que **hoy** pueden responder, en orden declarado.

    Un proveedor sin credencial no está disponible, y `build_provider` ya lo
    sabe —devuelve `None`—; esto solo lo pregunta sin construir nada que vaya
    a usarse. Existe para que una pantalla ofrezca lo que funciona en vez de
    una lista fija con opciones que fallan.

    `ollama` cuenta como disponible siempre: no necesita credencial, y si el
    servidor local no está arriba se verá al pedirle algo —igual que
    cualquier otro que falle—.

    El orden es el de `_DEFAULT_BASE_URLS`: decláralo ahí si quieres cambiar
    qué aparece primero, no lo ordenes alfabéticamente aquí — un orden que
    depende de quién pregunta es un orden que cambia según quién pregunte.
    """
    return tuple(name for name in _DEFAULT_BASE_URLS if build_provider(name) is not None)
