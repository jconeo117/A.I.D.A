"""Pruebas de aprendizaje, sorpresa y saturación firmada (`aida.engine.outcome`).

Cubre la unidad U3 (§8 de 06-plan-u3.md):
- signed_saturation: simetría, cotas estrictas en (-1, 1), comportamiento en cero, monotonía.
- learn: normalización de signos:
  * Ganancias: rendir más de lo creído es sorpresa positiva (+1), rendir menos es negativa.
  * Costes: gastar más de lo creído es sorpresa negativa (-1), gastar menos es positiva.
- learn: sorpresa máxima en valor absoluto (con signo), no suma acumulativa.
- Las 3 exclusiones de aprendizaje (no mutan creencias ni generan sorpresa):
  1. Techo: room < promised.
  2. Suelo: antes < gasto.
  3. Efectos colaterales: efectos fuera de satisfies o costs.
- learn con rate <= 0 no muta ninguna creencia y devuelve 0.0.
- Almacenes independientes de creencias: gain_beliefs y cost_beliefs no se pisan.
"""

from __future__ import annotations

import pytest

from aida.engine.ontology import ActionId, ActionSpec, NeedId, Reversibility
from aida.engine.outcome import learn, signed_saturation


def _make_action(
    id_: str = "act_prueba",
    *,
    satisfies: dict[NeedId, float] | None = None,
    costs: dict[NeedId, float] | None = None,
    effects: dict[NeedId, float] | None = None,
    reversibility: Reversibility = Reversibility.REVERSIBLE,
) -> ActionSpec:
    """Construye un ActionSpec válido coherente para pruebas de aprendizaje."""
    sat = satisfies or {}
    cst = costs or {}
    eff = dict(effects) if effects is not None else {**sat, **{k: -v for k, v in cst.items()}}
    return ActionSpec(
        id=ActionId(id_),
        label="Acción de prueba",
        duration_ticks=1,
        reversibility=reversibility,
        satisfies=sat,
        costs=cst,
        effects=eff,
    )


# -----------------------------------------------------------------------------
# 1. signed_saturation
# -----------------------------------------------------------------------------


def test_signed_saturation_simetria() -> None:
    """signed_saturation(x, scale) == -signed_saturation(-x, scale)."""
    for x in [0.1, 0.5, 1.0, 2.5, 10.0]:
        assert signed_saturation(-x) == pytest.approx(-signed_saturation(x))


def test_signed_saturation_comportamiento_en_cero() -> None:
    """En cero, la saturación es exactamente 0.0."""
    assert signed_saturation(0.0) == 0.0


def test_signed_saturation_cotas_estrictas_en_menos_uno_y_uno() -> None:
    """Para cualquier valor real finito, el resultado está estrictamente en (-1.0, 1.0)."""
    assert -1.0 < signed_saturation(-1e6) < 1.0
    assert -1.0 < signed_saturation(1e6) < 1.0
    assert signed_saturation(100.0) < 1.0
    assert signed_saturation(-100.0) > -1.0


def test_signed_saturation_monotonia() -> None:
    """Es estrictamente creciente: x1 < x2 => signed_saturation(x1) < signed_saturation(x2)."""
    valores = [-5.0, -1.0, -0.2, 0.0, 0.2, 1.0, 5.0]
    resultados = [signed_saturation(v) for v in valores]
    for i in range(len(resultados) - 1):
        assert resultados[i] < resultados[i + 1]


def test_signed_saturation_escala() -> None:
    """Una escala mayor aplana la curva, reduciendo la magnitud del resultado."""
    x = 1.0
    sat_escala1 = signed_saturation(x, scale=1.0)  # 1.0 / (1 + 1) = 0.5
    sat_escala2 = signed_saturation(x, scale=3.0)  # 1.0 / (1 + 3) = 0.25
    assert sat_escala1 == pytest.approx(0.5)
    assert sat_escala2 == pytest.approx(0.25)
    assert sat_escala2 < sat_escala1


# -----------------------------------------------------------------------------
# 2. learn: normalización de signos
# -----------------------------------------------------------------------------


def test_learn_ganancia_mayor_da_sorpresa_positiva() -> None:
    """Rendir más de lo creído en una ganancia perseguida produce sorpresa positiva (+1)."""
    n1 = NeedId("n1")
    action = _make_action(satisfies={n1: 0.4})
    gain_beliefs: dict[NeedId, float] = {n1: 1.0}
    cost_beliefs: dict[NeedId, float] = {}
    before = {n1: 0.2}
    received = {n1: 0.6}  # Rendimiento 0.6 / 0.4 = 1.5 > 1.0
    rate = 0.1

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate)

    # error = 1.5 - 1.0 = +0.5, factor = +1.0 => sorpresa = +0.5
    assert sorpresa > 0.0
    assert sorpresa == pytest.approx(0.5)
    # creencia actualizada = 1.0 + 0.1 * 0.5 = 1.05
    assert gain_beliefs[n1] == pytest.approx(1.05)


def test_learn_ganancia_menor_da_sorpresa_negativa() -> None:
    """Rendir menos de lo creído en una ganancia perseguida produce sorpresa negativa."""
    n1 = NeedId("n1")
    action = _make_action(satisfies={n1: 0.4})
    gain_beliefs: dict[NeedId, float] = {n1: 1.0}
    cost_beliefs: dict[NeedId, float] = {}
    before = {n1: 0.2}
    received = {n1: 0.2}  # Rendimiento 0.2 / 0.4 = 0.5 < 1.0
    rate = 0.2

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate)

    # error = 0.5 - 1.0 = -0.5, factor = +1.0 => sorpresa = -0.5
    assert sorpresa < 0.0
    assert sorpresa == pytest.approx(-0.5)
    # creencia actualizada = 1.0 + 0.2 * (-0.5) = 0.90
    assert gain_beliefs[n1] == pytest.approx(0.90)


def test_learn_gastar_mas_da_sorpresa_negativa() -> None:
    """Gastar más de lo creído (-recibido / gasto > 1.0) produce un error positivo en el almacén,
    pero al salir se normaliza con factor -1.0 dando sorpresa NEGATIVA (peor de lo esperado).
    """
    n2 = NeedId("n2")
    action = _make_action(costs={n2: 0.2})  # promised = -0.2, spend = 0.2
    gain_beliefs: dict[NeedId, float] = {}
    cost_beliefs: dict[NeedId, float] = {n2: 1.0}
    before = {n2: 0.8}
    received = {n2: -0.3}  # Se gastó 0.3 en vez de 0.2 => realized = 0.3 / 0.2 = 1.5
    rate = 0.1

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate)

    # error_crudo = 1.5 - 1.0 = +0.5. Almacén usa error crudo: 1.0 + 0.1 * 0.5 = 1.05
    assert cost_beliefs[n2] == pytest.approx(1.05)
    # Sorpresa normalizada: +0.5 * (-1.0) = -0.5 (gastar de más es peor!)
    assert sorpresa < 0.0
    assert sorpresa == pytest.approx(-0.5)


def test_learn_gastar_menos_da_sorpresa_positiva() -> None:
    """Gastar menos de lo creído (-recibido / gasto < 1.0) produce un error negativo en el almacén,
    pero al salir se normaliza con factor -1.0 dando sorpresa POSITIVA (mejor de lo esperado).
    """
    n2 = NeedId("n2")
    action = _make_action(costs={n2: 0.2})  # promised = -0.2, spend = 0.2
    gain_beliefs: dict[NeedId, float] = {}
    cost_beliefs: dict[NeedId, float] = {n2: 1.0}
    before = {n2: 0.8}
    received = {n2: -0.1}  # Se gastó 0.1 en vez de 0.2 => realized = 0.1 / 0.2 = 0.5
    rate = 0.1

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate)

    # error_crudo = 0.5 - 1.0 = -0.5. Almacén usa error crudo: 1.0 + 0.1 * (-0.5) = 0.95
    assert cost_beliefs[n2] == pytest.approx(0.95)
    # Sorpresa normalizada: -0.5 * (-1.0) = +0.5 (gastar menos es mejor!)
    assert sorpresa > 0.0
    assert sorpresa == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# 3. learn: sorpresa máxima en valor absoluto con signo, no suma
# -----------------------------------------------------------------------------


def test_learn_devuelve_sorpresa_maxima_y_no_la_suma() -> None:
    """Si una acción toca varios ejes, la sorpresa devuelta es el máximo en valor absoluto
    (preservando su signo normalizado), y NO la suma de errores.
    """
    n1 = NeedId("n1")
    n2 = NeedId("n2")
    action = _make_action(
        satisfies={n1: 0.5},
        costs={n2: 0.2},
    )
    gain_beliefs: dict[NeedId, float] = {n1: 1.0}
    cost_beliefs: dict[NeedId, float] = {n2: 1.0}
    before = {n1: 0.2, n2: 0.8}

    # Para n1 (ganancia): prometido 0.5, recibido 0.55 => realized 1.1, error +0.1, sorpresa +0.1
    # Para n2 (coste): prometido -0.2, recibido -0.32 => realized 0.32 / 0.2 = 1.6
    # error +0.6, sorpresa -0.6
    # La suma sería: +0.1 + (-0.6) = -0.5
    # El máximo en valor absoluto es |-0.6| = 0.6 > |+0.1| = 0.1 => sorpresa devuelta = -0.6
    received = {n1: 0.55, n2: -0.32}
    rate = 0.1

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate)

    assert sorpresa == pytest.approx(-0.6)
    assert sorpresa != pytest.approx(-0.5)  # No es la suma!


def test_learn_sorpresa_maxima_cuando_ganancia_es_mayor() -> None:
    """Caso inverso: la ganancia tiene mayor desviación absoluta que el coste."""
    n1 = NeedId("n1")
    n2 = NeedId("n2")
    action = _make_action(
        satisfies={n1: 0.5},
        costs={n2: 0.2},
    )
    gain_beliefs: dict[NeedId, float] = {n1: 1.0}
    cost_beliefs: dict[NeedId, float] = {n2: 1.0}
    before = {n1: 0.1, n2: 0.8}

    # Para n1: prometido 0.5, recibido 0.9 => realized 1.8, error +0.8, sorpresa +0.8
    # Para n2: prometido -0.2, recibido -0.24 => realized 1.2, error +0.2, sorpresa -0.2
    # Máximo en abs es 0.8 => sorpresa +0.8 (no +0.6 de la suma)
    received = {n1: 0.9, n2: -0.24}
    rate = 0.1

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate)
    assert sorpresa == pytest.approx(0.8)


# -----------------------------------------------------------------------------
# 4. Las 3 exclusiones de aprendizaje
# -----------------------------------------------------------------------------


def test_exclusion_techo_no_aprende_si_no_cabia() -> None:
    """Exclusión 1: Si room < promised (el eje estaba casi lleno), el efecto fue recortado
    por el techo y no por culpa de la acción. No se actualiza la creencia ni genera sorpresa.
    """
    n1 = NeedId("n1")
    action = _make_action(satisfies={n1: 0.4})
    gain_beliefs: dict[NeedId, float] = {n1: 1.0}
    cost_beliefs: dict[NeedId, float] = {}
    # before está en 0.85, room = 1.0 - 0.85 = 0.15 < promised (0.4)
    before = {n1: 0.85}
    received = {n1: 0.15}  # Recibió poco porque no cabía más
    rate = 0.5

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate)

    # Creencia queda intacta en 1.0
    assert gain_beliefs[n1] == pytest.approx(1.0)
    # Sin sorpresa
    assert sorpresa == 0.0


def test_exclusion_suelo_no_aprende_si_no_se_podia_pagar() -> None:
    """Exclusión 2: Si before < spend (no había saldo suficiente para pagar el coste),
    el gasto fue recortado por el suelo. No se debe concluir falsamente que es barato.
    """
    n2 = NeedId("n2")
    action = _make_action(costs={n2: 0.3})  # promised = -0.3, spend = 0.3
    gain_beliefs: dict[NeedId, float] = {}
    cost_beliefs: dict[NeedId, float] = {n2: 1.0}
    # before está en 0.05 < spend (0.3)
    before = {n2: 0.05}
    received = {n2: -0.05}  # Solo se pudo cobrar 0.05
    rate = 0.5

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate)

    # Creencia de coste queda intacta
    assert cost_beliefs[n2] == pytest.approx(1.0)
    # Sin sorpresa
    assert sorpresa == 0.0


def test_exclusion_efectos_colaterales_no_aprende_fuera_de_satisfies_o_costs() -> None:
    """Exclusión 3: Efectos presentes en effects pero ausentes en satisfies (si >0)
    o en costs (si <0) son efectos de rebote/colaterales y quedan excluidos del aprendizaje.
    """
    n_colateral_pos = NeedId("n_colateral_pos")
    n_colateral_neg = NeedId("n_colateral_neg")

    # ActionSpec permite efectos adicionales siempre que las promesas requeridas se cumplan
    action = _make_action(
        satisfies={},
        costs={},
        effects={
            n_colateral_pos: 0.5,   # > 0 pero no en satisfies
            n_colateral_neg: -0.3,  # < 0 pero no en costs
        },
    )
    gain_beliefs: dict[NeedId, float] = {n_colateral_pos: 1.0}
    cost_beliefs: dict[NeedId, float] = {n_colateral_neg: 1.0}
    before = {n_colateral_pos: 0.1, n_colateral_neg: 0.8}
    received = {n_colateral_pos: 0.8, n_colateral_neg: -0.5}
    rate = 0.5

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate)

    assert gain_beliefs[n_colateral_pos] == pytest.approx(1.0)
    assert cost_beliefs[n_colateral_neg] == pytest.approx(1.0)
    assert sorpresa == 0.0


# -----------------------------------------------------------------------------
# 5. rate <= 0 desactiva aprendizaje y no muta diccionarios
# -----------------------------------------------------------------------------


def test_learn_con_rate_cero_deja_diccionarios_intactos_y_devuelve_cero() -> None:
    """Si rate == 0.0, learn no debe realizar ninguna mutación sobre los diccionarios
    y debe retornar 0.0 de inmediato.
    """
    n1 = NeedId("n1")
    n2 = NeedId("n2")
    action = _make_action(satisfies={n1: 0.4}, costs={n2: 0.2})
    gain_beliefs: dict[NeedId, float] = {n1: 1.0}
    cost_beliefs: dict[NeedId, float] = {n2: 1.0}
    before = {n1: 0.2, n2: 0.8}
    received = {n1: 0.8, n2: -0.4}  # Grandes desviaciones

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate=0.0)

    assert sorpresa == 0.0
    assert gain_beliefs == {n1: 1.0}
    assert cost_beliefs == {n2: 1.0}


def test_learn_con_rate_negativo_no_muta_y_devuelve_cero() -> None:
    """Tasa negativa también desactiva el aprendizaje sin producir anomalías."""
    n1 = NeedId("n1")
    action = _make_action(satisfies={n1: 0.4})
    gain_beliefs: dict[NeedId, float] = {n1: 1.0}
    cost_beliefs: dict[NeedId, float] = {}
    before = {n1: 0.2}
    received = {n1: 0.8}

    sorpresa = learn(gain_beliefs, cost_beliefs, action, received, before, rate=-0.1)

    assert sorpresa == 0.0
    assert gain_beliefs == {n1: 1.0}


# -----------------------------------------------------------------------------
# 6. Almacenes independientes de creencias
# -----------------------------------------------------------------------------


def test_learn_almacenes_independientes_para_misma_necesidad() -> None:
    """Si una misma necesidad estuviera en ganancias de una acción y costes de otra,
    los dos diccionarios gain_beliefs y cost_beliefs no interfieren entre sí.
    """
    n1 = NeedId("n1")
    act_gana = _make_action(id_="gana", satisfies={n1: 0.5})
    act_gasta = _make_action(id_="gasta", costs={n1: 0.3})

    gains: dict[NeedId, float] = {n1: 1.0}
    costs: dict[NeedId, float] = {n1: 1.0}

    # Aprende en ganancias: rinde más
    learn(gains, costs, act_gana, received={n1: 0.75}, before={n1: 0.1}, rate=0.2)
    assert gains[n1] == pytest.approx(1.0 + 0.2 * (1.5 - 1.0))
    assert costs[n1] == pytest.approx(1.0)  # Intacto

    # Aprende en costes: gasta más
    learn(gains, costs, act_gasta, received={n1: -0.45}, before={n1: 0.8}, rate=0.2)
    assert costs[n1] == pytest.approx(1.0 + 0.2 * (1.5 - 1.0))
    assert gains[n1] == pytest.approx(1.1)  # No se alteró por el aprendizaje en costes


# -----------------------------------------------------------------------------
# 7. Desempate determinista entre canales (I-019)
# -----------------------------------------------------------------------------


def test_learn_desempate_determinista_entre_ganancia_y_coste_mismo_error_absoluto() -> None:
    """Cuando una ganancia y un coste empatan con idéntico error absoluto,
    sorted(action.effects) fija deterministamente el signo de la sorpresa (I-019).
    """
    n_a = NeedId("n_alpha")  # Alfabéticamente primera
    n_b = NeedId("n_beta")   # Alfabéticamente segunda

    # Caso 1: n_a es ganancia (+1), n_b es coste (-1)
    # Ambas tienen abs(error) == 0.5. Como n_a < n_b, n_a fija sorpresa en +0.5.
    act1 = _make_action(
        id_="act1",
        satisfies={n_a: 0.5},
        costs={n_b: 0.5},
        effects={n_b: -0.5, n_a: 0.5},  # Insertamos n_b primero para verificar que sorted manda
    )
    gb1: dict[NeedId, float] = {n_a: 1.0}
    cb1: dict[NeedId, float] = {n_b: 1.0}
    before1 = {n_a: 0.2, n_b: 0.8}
    received1 = {n_a: 0.75, n_b: -0.75}

    s1 = learn(gb1, cb1, act1, received1, before1, rate=0.2)
    assert s1 == pytest.approx(0.5)

    # Caso 2: n_a es coste (-1), n_b es ganancia (+1)
    # Como n_a < n_b, n_a fija sorpresa en -0.5.
    act2 = _make_action(
        id_="act2",
        satisfies={n_b: 0.5},
        costs={n_a: 0.5},
        effects={n_b: 0.5, n_a: -0.5},  # Insertamos n_b primero para verificar que sorted manda
    )
    gb2: dict[NeedId, float] = {n_b: 1.0}
    cb2: dict[NeedId, float] = {n_a: 1.0}
    before2 = {n_a: 0.8, n_b: 0.2}
    received2 = {n_a: -0.75, n_b: 0.75}

    s2 = learn(gb2, cb2, act2, received2, before2, rate=0.2)
    assert s2 == pytest.approx(-0.5)
