"""Evaluación de resultados y actualización de creencias por error de predicción.

Proporciona la matemática para comparar lo obtenido frente a lo anticipado,
ajustar expectativas empíricas y calcular la magnitud de sorpresa resultante.
"""

from __future__ import annotations

from collections.abc import Mapping

from aida.engine.ontology import ActionSpec, NeedId


def signed_saturation(x: float, scale: float = 1.0) -> float:
    """Aplica rendimiento decreciente simétrico en el intervalo (-1, 1).

        saturación(x) = x / (|x| + escala)

    El valor absoluto en el denominador preserva el signo del argumento
    garantizando un comportamiento simétrico para desviaciones opuestas.
    """
    return x / (abs(x) + scale)


def learn(
    gain_beliefs: dict[NeedId, float],
    cost_beliefs: dict[NeedId, float],
    action: ActionSpec,
    received: Mapping[NeedId, float],
    before: Mapping[NeedId, float],
    rate: float,
) -> float:
    """Corrige expectativas empíricas y devuelve la sorpresa normalizada con signo.

    Muta los diccionarios de creencias según la regla delta:
        creencia <- creencia + rate * (realizado - creencia)

    Retorna el error de predicción de mayor magnitud absoluta preservando su signo:
    positivo indica un resultado más favorable de lo esperado; negativo, más desfavorable.
    Gastar más de lo previsto se normaliza con signo negativo, y rendir más con signo positivo.

    Se descartan tres condiciones donde no corresponde aprendizaje:
    - Espacio insuficiente hasta el tope (el recorte provino del límite del eje).
    - Saldo insuficiente para abonar el coste (el recorte provino del suelo del eje).
    - Efectos colaterales ajenos a lo declarado en `satisfies` o `costs`.

    Si la tasa de aprendizaje es menor o igual a 0.0 o no hay sorpresa observable,
    retorna 0.0 sin modificar las creencias.
    """
    if rate <= 0.0:
        return 0.0

    surprise = 0.0
    for need_id in sorted(action.effects):
        promised = action.effects[need_id]
        if promised > 0.0 and need_id in action.satisfies:
            room = 1.0 - before[need_id]
            if room < promised:
                continue  # El tope recortó el efecto; no se atribuye a la acción
            realized = received.get(need_id, 0.0) / promised
            store: dict[NeedId, float] = gain_beliefs
            sign_factor = 1.0  # Mayor rendimiento del esperado es positivo (+1)

        elif promised < 0.0 and need_id in action.costs:
            spend = -promised
            if before[need_id] < spend:
                continue  # El suelo recortó el coste; no refleja el coste real
            realized = -received.get(need_id, 0.0) / spend
            store = cost_beliefs
            sign_factor = -1.0  # Mayor coste del esperado es negativo (-1)

        else:
            continue  # Efecto colateral no perseguido ni anticipado

        current = store.get(need_id, 1.0)
        error = realized - current
        store[need_id] = current + rate * error

        if abs(error) > abs(surprise):
            surprise = error * sign_factor

    return surprise
