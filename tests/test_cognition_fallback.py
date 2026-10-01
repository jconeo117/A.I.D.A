"""Pruebas para FallbackCognition (Plan 72 — §B.3).

Casos 14 al 18:
14. El primario responde -> el secundario no se toca.
15. El primario levanta -> responde el secundario y se avisa.
16. El primario devuelve None -> también cae al secundario.
17. Los dos fallan (o levantan) -> think devuelve None sin levantar.
18. El esquema y el briefing llegan intactos a los dos.
"""

from __future__ import annotations

from pydantic import BaseModel

from aida.cognition.fallback import FallbackCognition


class _DummySchema(BaseModel):
    text: str


class _SpyProvider:
    def __init__(self, response: object | None = None, raise_exc: Exception | None = None) -> None:
        self.response = response
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, type]] = []

    def think(self, briefing: str, schema: type) -> object | None:
        self.calls.append((briefing, schema))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


def test_fallback_cognition_primario_responde_secundario_no_se_toca() -> None:
    """Caso 14 (§B.3): Cuando el primario responde con éxito, el secundario no se llama
    y el callback de aviso no se invoca."""
    primary_resp = _DummySchema(text="respuesta primaria")
    primary = _SpyProvider(response=primary_resp)
    fallback = _SpyProvider(response=_DummySchema(text="respuesta secundaria"))
    avisos: list[str] = []

    cognition = FallbackCognition(primary, fallback, on_fallback=avisos.append)
    res = cognition.think("briefing de prueba", _DummySchema)

    assert res == primary_resp
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0
    assert len(avisos) == 0


def test_fallback_cognition_primario_levanta_secundario_responde_y_avisa() -> None:
    """Caso 15 (§B.3): Si el primario levanta una excepción, el secundario responde
    y se notifica la razón al callback on_fallback."""
    primary = _SpyProvider(raise_exc=RuntimeError("conexión rechazada con nube"))
    fallback_resp = _DummySchema(text="respuesta local")
    fallback = _SpyProvider(response=fallback_resp)
    avisos: list[str] = []

    cognition = FallbackCognition(primary, fallback, on_fallback=avisos.append)
    res = cognition.think("briefing con fallo", _DummySchema)

    assert res == fallback_resp
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
    assert len(avisos) == 1
    assert "conexión rechazada con nube" in avisos[0]


def test_fallback_cognition_primario_devuelve_none_cae_a_secundario() -> None:
    """Caso 16 (§B.3): Si el primario devuelve None (p.ej. no validó), también
    se recurre al secundario y se registra el aviso."""
    primary = _SpyProvider(response=None)
    fallback_resp = _DummySchema(text="respuesta local tras none")
    fallback = _SpyProvider(response=fallback_resp)
    avisos: list[str] = []

    cognition = FallbackCognition(primary, fallback, on_fallback=avisos.append)
    res = cognition.think("briefing none", _DummySchema)

    assert res == fallback_resp
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
    assert len(avisos) == 1
    assert "sin validar" in avisos[0]


def test_fallback_cognition_ambos_fallan_devuelve_none_sin_levantar() -> None:
    """Caso 17 (§B.3): Cuando primario y secundario fallan (excepciones o None),
    think devuelve None sin levantar ninguna excepción."""
    # Ambos levantan excepción
    p1 = _SpyProvider(raise_exc=RuntimeError("nube rota"))
    f1 = _SpyProvider(raise_exc=RuntimeError("ollama caído"))
    c1 = FallbackCognition(p1, f1)
    assert c1.think("briefing", _DummySchema) is None

    # Ambos devuelven None
    p2 = _SpyProvider(response=None)
    f2 = _SpyProvider(response=None)
    c2 = FallbackCognition(p2, f2)
    assert c2.think("briefing", _DummySchema) is None


def test_fallback_cognition_esquema_y_briefing_llegan_intactos_a_ambos() -> None:
    """Caso 18 (§B.3): El briefing y el schema llegan idénticos e intactos al primario
    y, en caso de fallback, al secundario."""
    briefing = "Briefing clínico detallado con formato especial\nLínea 2"
    schema = _DummySchema

    primary = _SpyProvider(response=None)
    fallback = _SpyProvider(response=_DummySchema(text="ok"))

    cognition = FallbackCognition(primary, fallback)
    cognition.think(briefing, schema)

    assert primary.calls == [(briefing, schema)]
    assert fallback.calls == [(briefing, schema)]
