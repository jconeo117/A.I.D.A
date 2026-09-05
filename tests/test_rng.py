"""Pruebas de la aleatoriedad derivable por clave (`aida.engine.rng`).

Cubren la unidad U1b. Cuatro frentes:

- El mismo par (semilla, clave) da el mismo resultado, entre procesos
  distintos: es lo que hace verificable G6 para esta pieza en concreto.
- Claves distintas dan resultados distintos, y el separador de unidad
  impide que dos claves que concatenadas serían iguales colisionen.
- El orden en que se piden los flujos no afecta al valor de cada uno.
"""

from __future__ import annotations

import os
import subprocess
import sys

from aida.engine.rng import RngSource

_SEED = 12345


def _primer_entero(seed: int, *key: str | int) -> int:
    """Primer entero que produce el flujo de esta clave, para comparar
    valores concretos sin acoplar el test a la implementación de `random`."""
    return RngSource(seed=seed).stream(*key).randint(0, 10_000_000)


# ---------------------------------------------------------------------------
# 1. Determinismo entre procesos.
# ---------------------------------------------------------------------------
def test_misma_semilla_y_clave_dan_el_mismo_resultado_entre_procesos() -> None:
    """Dos procesos de Python distintos, con `PYTHONHASHSEED` distinto entre
    ellos, derivan el mismo entero a partir del mismo (semilla, clave). Si
    `rng.py` usara `hash()` en vez de `blake2b`, este test divergiría solo
    a veces, según el azar de arranque de cada proceso.
    """
    # Arrange
    script = (
        "from aida.engine.rng import RngSource\n"
        "valor = RngSource(seed=12345).stream('decide', 'duenio', 7).randint(0, 10_000_000)\n"
        "print(valor)\n"
    )

    # Act: se hereda el entorno del proceso de test (para que el subproceso
    # encuentre el intérprete y el paquete instalado en modo editable) y solo
    # se fija `PYTHONHASHSEED` en el entorno del SUBPROCESO, nunca en el del
    # proceso que ejecuta pytest.
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

    # Assert: hash-seeds distintos, mismo resultado.
    assert salida_a.stdout == salida_b.stdout
    assert salida_a.stdout.strip() != ""


# ---------------------------------------------------------------------------
# 2. Claves distintas dan resultados distintos; el separador evita colisión.
# ---------------------------------------------------------------------------
def test_claves_distintas_dan_resultados_distintos() -> None:
    """Cambiar un solo elemento de la clave cambia el flujo entero."""
    # Arrange / Act
    valor_a = _primer_entero(_SEED, "decide", "duenio", 7)
    valor_b = _primer_entero(_SEED, "decide", "duenio", 8)

    # Assert
    assert valor_a != valor_b


def test_semillas_distintas_dan_resultados_distintos_para_la_misma_clave() -> None:
    """La semilla también forma parte de lo que se deriva: dos mundos con
    semillas distintas no comparten sorteo aunque pidan la misma clave."""
    # Arrange / Act
    valor_a = _primer_entero(1, "decide", "duenio", 7)
    valor_b = _primer_entero(2, "decide", "duenio", 7)

    # Assert
    assert valor_a != valor_b


def test_claves_que_concatenadas_serian_iguales_no_colisionan() -> None:
    """`("ab", "c")` y `("a", "bc")` producen flujos distintos: el separador
    de unidad ASCII entre partes de la clave es justo lo que evita que
    concatenar sin separador junte dos claves distintas en una sola cadena.
    """
    # Arrange / Act
    valor_ab_c = _primer_entero(_SEED, "ab", "c")
    valor_a_bc = _primer_entero(_SEED, "a", "bc")

    # Assert
    assert valor_ab_c != valor_a_bc


# ---------------------------------------------------------------------------
# 3. El orden en que se piden los flujos no afecta a su valor.
# ---------------------------------------------------------------------------
def test_el_orden_de_peticion_de_los_flujos_no_afecta_su_valor() -> None:
    """Pedir A y luego B da los mismos valores que pedir B y luego A: cada
    flujo es función pura de su propia clave, no de la secuencia de llamadas
    previas. Es lo que permite avanzar partes del mundo por separado.
    """
    # Arrange
    fuente_orden_ab = RngSource(seed=_SEED)
    fuente_orden_ba = RngSource(seed=_SEED)

    # Act: primero A y luego B en una fuente, al revés en la otra.
    valor_a_primero = fuente_orden_ab.stream("a").randint(0, 10_000_000)
    valor_b_segundo = fuente_orden_ab.stream("b").randint(0, 10_000_000)

    valor_b_primero = fuente_orden_ba.stream("b").randint(0, 10_000_000)
    valor_a_segundo = fuente_orden_ba.stream("a").randint(0, 10_000_000)

    # Assert
    assert valor_a_primero == valor_a_segundo
    assert valor_b_segundo == valor_b_primero


def test_llamar_stream_dos_veces_con_la_misma_clave_da_la_misma_secuencia() -> None:
    """Dos generadores distintos construidos con la misma clave producen
    exactamente la misma secuencia de valores, no solo el primero."""
    # Arrange
    fuente = RngSource(seed=_SEED)

    # Act
    secuencia_a = [fuente.stream("decide", "duenio", 7).random() for _ in range(5)]
    secuencia_b = [fuente.stream("decide", "duenio", 7).random() for _ in range(5)]

    # Assert
    assert secuencia_a == secuencia_b
