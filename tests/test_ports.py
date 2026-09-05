"""Pruebas de las costuras vacías (`aida.engine.ports`).

Son cinco `Protocol` sin implementación en U1b, así que hay poco que
probar y se evita inventar comportamiento que no existe: solo que los
cinco puertos existen, que son `runtime_checkable`, y que una clase que
cumple la firma satisface el protocolo mientras una que no la cumple, no.
"""

from __future__ import annotations

from collections.abc import Mapping

from aida.engine.ports import (
    CognitionProvider,
    IntentClassifier,
    MemoryStore,
    SemanticStore,
    ToolPort,
)


# ---------------------------------------------------------------------------
# 1. Los cinco puertos existen y son `runtime_checkable`.
# ---------------------------------------------------------------------------
def test_los_cinco_puertos_son_runtime_checkable() -> None:
    """`isinstance` contra cada protocolo no revienta con `TypeError`: es lo
    que garantiza `runtime_checkable` y lo que permite comprobar en
    ejecución que una implementación futura cumple la costura."""
    # Arrange
    puertos = (MemoryStore, SemanticStore, IntentClassifier, CognitionProvider, ToolPort)

    # Act / Assert
    for puerto in puertos:
        assert isinstance(object(), puerto) is False


# ---------------------------------------------------------------------------
# 2. `IntentClassifier`: la costura más simple, un único método.
# ---------------------------------------------------------------------------
class _ClasificadorConforme:
    def classify(self, utterance: str) -> str | None:
        return None


class _NoEsClasificador:
    def otra_cosa(self) -> None:
        pass


def test_una_clase_con_classify_satisface_intent_classifier() -> None:
    """Una clase que declara `classify(self, utterance) -> str | None`
    cumple `IntentClassifier` sin heredar de él: `Protocol` es estructural."""
    # Arrange
    clasificador = _ClasificadorConforme()

    # Act / Assert
    assert isinstance(clasificador, IntentClassifier)


def test_una_clase_sin_classify_no_satisface_intent_classifier() -> None:
    """Una clase que no declara `classify` no cumple el protocolo."""
    # Arrange
    otra = _NoEsClasificador()

    # Act / Assert
    assert not isinstance(otra, IntentClassifier)


# ---------------------------------------------------------------------------
# 3. `CognitionProvider`: un único método, `think`.
# ---------------------------------------------------------------------------
class _ProveedorConforme:
    def think(self, briefing: str, schema: type) -> object | None:
        return None


def test_una_clase_con_think_satisface_cognition_provider() -> None:
    """Una clase que declara `think(self, briefing, schema) -> object | None`
    cumple `CognitionProvider`."""
    # Arrange
    proveedor = _ProveedorConforme()

    # Act / Assert
    assert isinstance(proveedor, CognitionProvider)


def test_una_clase_sin_think_no_satisface_cognition_provider() -> None:
    """Una clase sin `think` no cumple el protocolo."""
    # Arrange
    otra = _NoEsClasificador()

    # Act / Assert
    assert not isinstance(otra, CognitionProvider)


# ---------------------------------------------------------------------------
# 4. `ToolPort`: un único método, `execute`.
# ---------------------------------------------------------------------------
class _HerramientaConforme:
    def execute(self, *, action: object, arguments: Mapping[str, object]) -> object:
        return None


def test_una_clase_con_execute_satisface_tool_port() -> None:
    """Una clase que declara `execute(self, *, action, arguments) -> object`
    cumple `ToolPort`."""
    # Arrange
    herramienta = _HerramientaConforme()

    # Act / Assert
    assert isinstance(herramienta, ToolPort)


def test_una_clase_sin_execute_no_satisface_tool_port() -> None:
    """Una clase sin `execute` no cumple el protocolo."""
    # Arrange
    otra = _NoEsClasificador()

    # Act / Assert
    assert not isinstance(otra, ToolPort)


# ---------------------------------------------------------------------------
# 5. `MemoryStore`: dos métodos, `remember` y `recall`.
# ---------------------------------------------------------------------------
class _MemoriaConforme:
    def remember(self, *, owner: object, event: object, importance: float) -> object | None:
        return None

    def recall(self, *, owner: object, cue: str, limit: int) -> tuple[object, ...]:
        return ()


class _MemoriaIncompleta:
    """Declara solo uno de los dos métodos del protocolo: no basta con
    cumplir parte de la firma."""

    def remember(self, *, owner: object, event: object, importance: float) -> object | None:
        return None


def test_una_clase_con_remember_y_recall_satisface_memory_store() -> None:
    """Una clase que declara ambos métodos de `MemoryStore` lo cumple."""
    # Arrange
    memoria = _MemoriaConforme()

    # Act / Assert
    assert isinstance(memoria, MemoryStore)


def test_una_clase_con_solo_uno_de_los_dos_metodos_no_satisface_memory_store() -> None:
    """Faltar `recall` basta para no cumplir el protocolo, aunque
    `remember` esté presente: `runtime_checkable` exige todos los
    miembros declarados, no una parte."""
    # Arrange
    memoria = _MemoriaIncompleta()

    # Act / Assert
    assert not isinstance(memoria, MemoryStore)


# ---------------------------------------------------------------------------
# 6. `SemanticStore`: tres métodos, `reinforce`, `refute` y `rules_for`.
# ---------------------------------------------------------------------------
class _PreferenciasConforme:
    def reinforce(self, *, owner: object, rule: object, amount: float) -> None:
        return None

    def refute(self, *, owner: object, rule: object, amount: float) -> None:
        return None

    def rules_for(self, *, owner: object, cue: str, limit: int) -> tuple[object, ...]:
        return ()


def test_una_clase_con_los_tres_metodos_satisface_semantic_store() -> None:
    """Una clase que declara `reinforce`, `refute` y `rules_for` cumple
    `SemanticStore`."""
    # Arrange
    preferencias = _PreferenciasConforme()

    # Act / Assert
    assert isinstance(preferencias, SemanticStore)


def test_una_clase_sin_ninguno_de_los_tres_metodos_no_satisface_semantic_store() -> None:
    """Una clase ajena por completo no cumple `SemanticStore`."""
    # Arrange
    otra = _NoEsClasificador()

    # Act / Assert
    assert not isinstance(otra, SemanticStore)
