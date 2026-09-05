"""Pruebas del registro de eventos (`aida.engine.events`).

Cubren la unidad U1b. Seis frentes:

- Determinismo entre procesos: dos procesos de Python de verdad, con
  `PYTHONHASHSEED` distinto, que emiten la misma secuencia de eventos
  producen el mismo JSONL byte a byte. Es el criterio central de G6 para
  esta pieza.
- `owner` obligatorio (G7): no se puede construir ni emitir un evento sin
  dueño, y el índice por dueño aísla la consulta de uno del otro.
- El volcado del payload es estable frente al orden de inserción de sus
  claves.
- El journal: una entrada del exterior se reconstruye exacta, con su
  digest, y el digest es estable entre procesos.
- `seq` monótono y global, incluso con ventana y desalojo real.
- El vertido a `sink` conserva lo que la ventana descarta.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from aida.engine.events import (
    NO_DATA,
    PURGE_BATCH_SIZE,
    Event,
    EventKind,
    EventLog,
    EvictedEventError,
    digest_of_content,
)
from aida.engine.ontology import AgentId

_ALICIA = AgentId("alicia")
_BERTO = AgentId("berto")


# ---------------------------------------------------------------------------
# 1. Determinismo entre procesos.
# ---------------------------------------------------------------------------
def test_dos_procesos_con_pythonhashseed_distinto_producen_el_mismo_jsonl() -> None:
    """Dos procesos de Python distintos, con `PYTHONHASHSEED` distinto entre
    ellos, emiten la misma secuencia de eventos —incluida una entrada del
    exterior con caracteres no ASCII— y vuelcan un JSONL idéntico byte a
    byte. Es el test que da sentido a G6 para el registro: si `to_json`
    dependiera del orden de hash de un `dict` o de un `frozenset`, este test
    fallaría de forma intermitente según el `PYTHONHASHSEED` de cada
    proceso, y no dentro de un único intérprete.
    """
    # Arrange
    script = (
        "from aida.engine.events import EventKind, EventLog\n"
        "from aida.engine.ontology import AgentId\n"
        "log = EventLog()\n"
        "log.emit(\n"
        "    tick=0,\n"
        "    kind=EventKind.INPUT_OBSERVED,\n"
        "    owner=AgentId('alicia'),\n"
        "    content='hola, ¿qué tal? \\n con salto y \"comillas\"',\n"
        ")\n"
        "log.emit(\n"
        "    tick=1,\n"
        "    kind=EventKind.ACTION_STARTED,\n"
        "    owner=AgentId('berto'),\n"
        "    data={'zeta': 1, 'alfa': 2, 'eme': 3},\n"
        ")\n"
        "for linea in log.to_jsonl():\n"
        "    print(linea)\n"
    )

    # Act
    salida_a = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONHASHSEED": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    salida_b = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONHASHSEED": "2"},
        capture_output=True,
        text=True,
        check=True,
    )

    # Assert: hash-seeds distintos, salida idéntica byte a byte.
    assert salida_a.stdout == salida_b.stdout
    assert salida_a.stdout.count("\n") == 2


# ---------------------------------------------------------------------------
# 2. `owner` obligatorio (G7).
# ---------------------------------------------------------------------------
def test_construir_un_event_sin_owner_falla_en_ejecucion() -> None:
    """`Event` no admite `owner=None`: el campo no tiene valor por defecto,
    así que omitirlo revienta con `TypeError` al construir, no solo por
    tipado estático.
    """
    # Arrange / Act / Assert
    with pytest.raises(TypeError):
        Event(seq=0, tick=0, kind=EventKind.ACTION_STARTED)  # type: ignore[call-arg]


def test_emitir_un_evento_sin_owner_falla_en_ejecucion() -> None:
    """`EventLog.emit` exige `owner` como argumento con nombre; omitirlo
    revienta en ejecución, no solo confiando en que mypy lo detecte antes.
    """
    # Arrange
    log = EventLog()

    # Act / Assert
    with pytest.raises(TypeError):
        log.emit(tick=0, kind=EventKind.ACTION_STARTED)  # type: ignore[call-arg]


def test_by_owner_de_un_duenio_nunca_devuelve_eventos_de_otro() -> None:
    """El índice por dueño aísla la consulta: emitidos eventos de dos
    dueños distintos, pedir los de uno no devuelve ni uno solo del otro. Es
    aislamiento entre usuarios, no una comodidad de consulta.
    """
    # Arrange
    log = EventLog()
    log.emit(tick=0, kind=EventKind.ACTION_STARTED, owner=_ALICIA)
    log.emit(tick=0, kind=EventKind.ACTION_STARTED, owner=_BERTO)
    log.emit(tick=1, kind=EventKind.ACTION_COMPLETED, owner=_ALICIA)
    log.emit(tick=1, kind=EventKind.ACTION_COMPLETED, owner=_BERTO)
    log.emit(tick=2, kind=EventKind.GOAL_ACHIEVED, owner=_BERTO)

    # Act
    eventos_alicia = log.by_owner(_ALICIA)
    eventos_berto = log.by_owner(_BERTO)

    # Assert
    assert len(eventos_alicia) == 2
    assert all(evento.owner == _ALICIA for evento in eventos_alicia)
    assert len(eventos_berto) == 3
    assert all(evento.owner == _BERTO for evento in eventos_berto)


# ---------------------------------------------------------------------------
# 3. Orden de claves del payload.
# ---------------------------------------------------------------------------
def test_el_payload_serializa_igual_con_las_claves_en_distinto_orden() -> None:
    """Dos eventos lógicamente idénticos, con las claves del payload
    insertadas en distinto orden, producen la misma línea JSON: el volcado
    ordena las claves en vez de conservar el orden de inserción del
    llamante.
    """
    # Arrange
    evento_orden_a = Event(
        seq=0,
        tick=0,
        kind=EventKind.ACTION_STARTED,
        owner=_ALICIA,
        data={"zeta": 1, "alfa": 2, "eme": 3},
    )
    evento_orden_b = Event(
        seq=0,
        tick=0,
        kind=EventKind.ACTION_STARTED,
        owner=_ALICIA,
        data={"eme": 3, "zeta": 1, "alfa": 2},
    )

    # Act
    json_a = evento_orden_a.to_json()
    json_b = evento_orden_b.to_json()

    # Assert
    assert json_a == json_b


def test_el_payload_vacio_no_aparece_en_el_json() -> None:
    """Un evento sin datos extra no vuelca `"data":{}`: se omite por
    completo, porque en la mayoría de los eventos sería ruido puro."""
    # Arrange
    evento = Event(seq=0, tick=0, kind=EventKind.ACTION_STARTED, owner=_ALICIA, data=NO_DATA)

    # Act
    volcado = json.loads(evento.to_json())

    # Assert
    assert "data" not in volcado


# ---------------------------------------------------------------------------
# 4. El journal.
# ---------------------------------------------------------------------------
def test_una_entrada_del_exterior_se_reconstruye_exacta_desde_el_jsonl() -> None:
    """El contenido original —con saltos de línea, comillas y caracteres no
    ASCII— sobrevive intacto al viaje de ida (emitir) y vuelta (parsear la
    línea JSONL): es el requisito sin el que G6 no sería verificable.
    """
    # Arrange
    contenido_original = 'Envía el informe\nmañana, "por favor" — ¿vale? 你好'
    log = EventLog()

    # Act
    log.emit(
        tick=0,
        kind=EventKind.INPUT_OBSERVED,
        owner=_ALICIA,
        content=contenido_original,
    )
    (linea,) = list(log.to_jsonl())
    reconstruido = json.loads(linea)

    # Assert
    assert reconstruido["content"] == contenido_original


def test_el_digest_de_un_evento_de_entrada_corresponde_a_su_contenido() -> None:
    """`emit` calcula el digest a partir del contenido exacto que recibe:
    coincide con llamar `digest_of_content` a mano sobre el mismo texto."""
    # Arrange
    contenido = "contenido de prueba"
    log = EventLog()

    # Act
    evento = log.emit(tick=0, kind=EventKind.INPUT_OBSERVED, owner=_ALICIA, content=contenido)

    # Assert
    assert evento.digest == digest_of_content(contenido)


def test_contenidos_distintos_dan_digests_distintos() -> None:
    """Dos entradas de texto distinto producen digests distintos: es lo que
    permite detectar una divergencia sin comparar el texto entero."""
    # Arrange / Act
    digest_a = digest_of_content("hola")
    digest_b = digest_of_content("hola.")

    # Assert
    assert digest_a != digest_b


def test_el_mismo_contenido_da_el_mismo_digest_entre_procesos() -> None:
    """`digest_of_content` es estable entre procesos distintos con
    `PYTHONHASHSEED` distinto: usa `blake2b`, no `hash()`."""
    # Arrange
    script = (
        "from aida.engine.events import digest_of_content\n"
        "print(digest_of_content('contenido de prueba'))\n"
    )

    # Act
    salida_a = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONHASHSEED": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    salida_b = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONHASHSEED": "2"},
        capture_output=True,
        text=True,
        check=True,
    )

    # Assert
    assert salida_a.stdout == salida_b.stdout
    assert salida_a.stdout.strip() != ""


# ---------------------------------------------------------------------------
# 5. `seq` monótono y la ventana.
# ---------------------------------------------------------------------------
def _log_con_desalojo_real(sink: io.StringIO | None = None) -> EventLog:
    """Un `EventLog` en el que ya ocurrió un desalojo de verdad, no
    simulado: la purga solo se revisa cada `PURGE_BATCH_SIZE` emisiones
    (ver `events.py`), así que hace falta ese volumen para que la ventana
    llegue a descartar algo en vez de solo declarar que podría hacerlo.
    """
    log = EventLog(window=5, sink=sink)
    for tick in range(PURGE_BATCH_SIZE + 5):
        log.emit(tick=tick, kind=EventKind.ACTION_STARTED, owner=_ALICIA)
    return log


def test_seq_crece_de_uno_en_uno_y_nunca_se_reutiliza() -> None:
    """Emitir varios eventos seguidos numera `seq` de forma consecutiva
    empezando en cero."""
    # Arrange
    log = EventLog()

    # Act
    secuencias = [
        log.emit(tick=tick, kind=EventKind.ACTION_STARTED, owner=_ALICIA).seq
        for tick in range(5)
    ]

    # Assert
    assert secuencias == [0, 1, 2, 3, 4]


def test_seq_sigue_siendo_global_tras_un_desalojo_real() -> None:
    """Con ventana activa y un desalojo de verdad ya ocurrido, el siguiente
    `seq` no reinicia la numeración: sigue contando desde
    `evicted + len(eventos)`, nunca desde `len(eventos)` a secas. Es lo que
    permite que la memoria de U9 referencie un evento por `seq` sin
    ambigüedad.
    """
    # Arrange
    log = _log_con_desalojo_real()
    assert log.evicted > 0, "la ventana no desalojó nada: el test no prueba lo que dice probar"
    emitidos_antes = log.emitted

    # Act
    siguiente = log.emit(tick=PURGE_BATCH_SIZE + 100, kind=EventKind.ACTION_STARTED, owner=_ALICIA)

    # Assert: el siguiente `seq` continúa tras todo lo ya ocurrido, se
    # conserve en memoria o no.
    assert siguiente.seq == emitidos_antes


def test_pedir_un_evento_desalojado_por_seq_revienta() -> None:
    """Un `seq` que la ventana ya descartó no se puede leer en silencio con
    un valor distinto: revienta con `EvictedEventError`."""
    # Arrange
    log = _log_con_desalojo_real()
    assert log.evicted > 0

    # Act / Assert
    with pytest.raises(EvictedEventError):
        log[0]


def test_un_evento_desalojado_de_memoria_sigue_en_el_jsonl_del_sink() -> None:
    """Al vertido va todo lo que ocurre, se retenga en memoria o no: los
    eventos que la ventana ya descartó de memoria siguen presentes en el
    JSONL del `sink`, con su `seq` intacto."""
    # Arrange
    volcado = io.StringIO()

    # Act
    log = _log_con_desalojo_real(sink=volcado)
    assert log.evicted > 0

    lineas = volcado.getvalue().splitlines()
    seq_en_sink = [json.loads(linea)["seq"] for linea in lineas]

    # Assert: todos los emitidos están en el sink, aunque en memoria solo
    # queden los de la ventana.
    assert seq_en_sink == list(range(PURGE_BATCH_SIZE + 5))
    assert len(log) < PURGE_BATCH_SIZE + 5


# ---------------------------------------------------------------------------
# Extra: inmutabilidad del evento, ya cubierta por el contrato de U1a pero
# no probada todavía sobre `Event` en concreto.
# ---------------------------------------------------------------------------
def test_un_evento_construido_no_se_puede_mutar() -> None:
    """`Event` es `frozen=True`: asignar a un campo ya construido revienta."""
    # Arrange
    evento = Event(seq=0, tick=0, kind=EventKind.ACTION_STARTED, owner=_ALICIA)

    # Act / Assert
    with pytest.raises(FrozenInstanceError):
        evento.tick = 1  # type: ignore[misc]
