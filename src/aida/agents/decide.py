"""Deliberación determinista: selección de acciones y cálculo de utilidad.

Este módulo implementa el mecanismo mediante el cual un agente evalúa su
catálogo de acciones disponibles y selecciona la siguiente a ejecutar. Todo
el proceso se rige por aritmética exacta sobre el estado interno y la
personalidad compuesta, sin intervención de modelos de lenguaje ni factores
azarosos no reproducibles.

Invariantes fundamentales de diseño:
1. Una sola puerta para el cálculo de urgencia: se consume canónicamente
   `urgency` desde `aida.engine.disposition`, garantizando coherencia en la
   evaluación de necesidades en todo el motor.
2. Una sola puerta para la resolución de techos: siempre a través de
   `agent.cap_of(need)`, nunca accediendo directamente al techo nominal.
3. Determinismo estricto: toda iteración sobre diccionarios o conjuntos se
   ordena explícitamente (`sorted()`) para evitar divergencias acumulativas
   por la no-asociatividad de operaciones en coma flotante.
4. Desempate total y estable: `_rank` ordena por utilidad descendente y
   desempata por el identificador de la acción en texto, eliminando cualquier
   dependencia del orden de declaración del pack.
5. Softmax numéricamente estable: se resta el máximo de utilidad antes de
   exponenciar (`d.utility - best`) para prevenir desbordamientos aritméticos.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from aida.engine.disposition import urgency
from aida.engine.ontology import ActionId, ActionSpec, NeedId, PhaseId, SituationId
from aida.engine.pack import ContentPack
from aida.engine.state import AgentState, ExecutionState


class UncomposedAgentStateError(RuntimeError):
    """El agente no ha sido recompuesto antes de deliberar.

    `AgentState.temperament` es `None` hasta que se invoca `recompose(pack)`.
    La deliberación necesita el temperamento compuesto (temperatura, etc.)
    para evaluar y seleccionar acciones; invocar `decide` sin recomponer
    indica un error de programación de quien llama.
    """


@dataclass(frozen=True, slots=True)
class Decision:
    """Acción evaluada y sus componentes de utilidad desglosados.

    Contiene los cinco factores aritméticos individuales además de la utilidad
    neta. Esto permite que el sistema de explicabilidad (`explain.py`) lea
    directamente los valores calculados sin recalcularlos, garantizando fidelidad
    absoluta e impidiendo discrepancias causales.
    """

    action: ActionId
    utility: float
    benefit: float
    cost: float
    novelty: float
    phase_fit: float
    situation_fit: float


def _rank(decision: Decision) -> tuple[float, str]:
    """Criterio de ordenación total y estable para clasificar decisiones.

    Ordena por utilidad descendente y desempata lexicográficamente por el
    identificador de la acción. El desempate por identificador es esencial
    para el determinismo: sin él, dos opciones con idéntica utilidad quedarían
    en el orden accidental del bucle, y cualquier reordenación del pack
    cambiaría la conducta del agente bajo la misma semilla.
    """
    return (-decision.utility, str(decision.action))


def _weigh(
    pack: ContentPack,
    agent: AgentState,
    terms: Mapping[NeedId, float],
    beliefs: Mapping[NeedId, float],
) -> float:
    """Suma ponderada de necesidades por importancia, urgencia y creencia.

    Que tanto el beneficio como el coste pasen por esta función no es ahorro de
    código: es lo que garantiza que ganar y perder se midan con el mismo patrón.
    La urgencia pondera también el coste porque gastar recursos escasos o
    críticos duele más cuando queda poco.

    Detalles de diseño críticos:
    - `urgency` se importa de `disposition.py` (única puerta canónica).
    - Los techos se resuelven con `agent.cap_of(need)` (única puerta de techos).
    - `sorted(terms)`: la suma en coma flotante no es asociativa; sin orden fijo,
      dos ejecuciones diferirían en el último bit según la semilla de hash.
    - Lo ausente en `beliefs` vale 1.0: el agente sin experiencia asume la
      promesa nominal del pack.
    """
    total = 0.0
    for need_id in sorted(terms):
        need = pack.need_by_id[need_id]
        pressure = urgency(
            agent.needs[need_id], need.urgency_exponent, agent.cap_of(need)
        )
        total += (
            agent.need_weights[need_id]
            * pressure
            * terms[need_id]
            * beliefs.get(need_id, 1.0)
        )
    return total


def benefit(pack: ContentPack, agent: AgentState, action: ActionSpec) -> float:
    """Beneficio esperado por el agente al ejecutar la acción.

    Escala las necesidades que la acción satisface (`satisfies`) según las
    creencias aprendidas del agente (`gain_beliefs`). Lo ausente en creencias
    devuelve un mapa vacío que se evalúa a 1.0 por defecto.
    """
    return _weigh(
        pack, agent, action.satisfies, agent.gain_beliefs.get(action.id, {})
    )


def cost(pack: ContentPack, agent: AgentState, action: ActionSpec) -> float:
    """Coste esperado por el agente al ejecutar la acción.

    Espejo exacto de `benefit`: evalúa las necesidades consumidas (`costs`)
    moduladas por las creencias sobre costes (`cost_beliefs`).
    """
    return _weigh(
        pack, agent, action.costs, agent.cost_beliefs.get(action.id, {})
    )


def novelty(agent: AgentState, action: ActionId, weight: float) -> float:
    """Modulador de novedad y hastío: `1.0 - saturación * weight`.

    El suelo en `1.0 - weight` es deliberado: con novedad pura, una acción
    saturada al tope valdría 0.0 y quedaría vetada indefinidamente. El hastío
    debe poder alterar prioridades, no imponer vetos permanentes a tareas
    vitales. Multiplica solo al beneficio, nunca al neto.
    """
    return 1.0 - agent.saturation.get(action, 0.0) * weight


def phase_fit(action: ActionSpec, phase: PhaseId | None) -> float:
    """Ajuste multiplicativo del beneficio según el tramo del ciclo.

    Si no se especifica franja (`None`), retorna 1.0. Si la acción no declara
    multiplicador para la franja activa, se adopta el factor neutro 1.0.
    """
    if phase is None:
        return 1.0
    return action.phase_multipliers.get(phase, 1.0)


def situation_fit(action: ActionSpec, situations: Iterable[SituationId]) -> float:
    """Ajuste multiplicativo del beneficio según las situaciones activas.

    Multiplica acumulativamente los factores de cada situación en orden
    determinista (`sorted(situations)`). Si no hay multiplicadores situacionales
    declarados en la acción, retorna 1.0.
    """
    if not action.situation_multipliers:
        return 1.0
    total = 1.0
    for situation in sorted(situations):
        total *= action.situation_multipliers.get(situation, 1.0)
    return total


def is_available(agent: AgentState, action: ActionSpec) -> str | None:
    """Comprueba si la acción es ejecutable según las puertas duras del sistema.

    Devuelve `None` si la acción es elegible, o el nombre literal de la puerta
    que la descarta: `"requires"` o `"requires_traits"`. El nombre es consumido
    por la traza explicativa para diagnosticar por qué una opción quedó fuera.
    Las opciones con utilidad cero no se filtran aquí para permitir desempate
    estable en situaciones neutrales.
    """
    for need_id in sorted(action.requires):
        if agent.needs.get(need_id, 0.0) < action.requires[need_id]:
            return "requires"
    for trait_id in sorted(action.requires_traits):
        if agent.traits.get(trait_id, 0.0) < action.requires_traits[trait_id]:
            return "requires_traits"
    return None


def candidates(
    execution: ExecutionState,
    pack: ContentPack,
    phase: PhaseId | None = None,
    situations: Iterable[SituationId] = (),
) -> tuple[list[Decision], dict[ActionId, str]]:
    """Evalúa todas las acciones del catálogo y devuelve las puntuadas y descartadas.

    Itera sobre `sorted(pack.action_by_id)` garantizando orden determinista.
    Para cada acción comprueba `is_available`: si no es ejecutable, registra la
    puerta de descarte en el diccionario `discarded`; si es ejecutable, computa
    los cinco factores aritméticos, calcula la utilidad y añade un `Decision`.

    Fórmula:
        utilidad = (beneficio * novedad * phase_fit * situation_fit
                     - aversión_al_coste * coste) * gusto

    `gusto` (`agent.tastes`) multiplica la utilidad ya compuesta entera, no
    un término aislado: es una predilección o aversión hacia ESA acción en
    concreto, por encima de lo que le pasaría a cualquier agente con las
    mismas necesidades. Lo ausente vale 1.0 — sin gusto declarado, neutral.
    """
    agent = execution.agent
    if agent.temperament is None:
        raise UncomposedAgentStateError(
            f"Agent {agent.agent_id!r} has temperament=None. "
            "AgentState.recompose(pack) must be called before evaluating candidates."
        )

    pool: list[Decision] = []
    discarded: dict[ActionId, str] = {}
    sit_tuple = tuple(situations)

    for action_id in sorted(pack.action_by_id):
        action = pack.action_by_id[action_id]
        door = is_available(agent, action)
        if door is not None:
            discarded[action.id] = door
            continue

        b = benefit(pack, agent, action)
        c = cost(pack, agent, action)
        nov = novelty(agent, action.id, agent.temperament.novelty_weight)
        p_fit = phase_fit(action, phase)
        s_fit = situation_fit(action, sit_tuple)
        util = (b * nov * p_fit * s_fit) - agent.temperament.loss_aversion * c
        util *= agent.tastes.get(str(action.id), 1.0)

        pool.append(
            Decision(
                action=action.id,
                utility=util,
                benefit=b,
                cost=c,
                novelty=nov,
                phase_fit=p_fit,
                situation_fit=s_fit,
            )
        )

    return pool, discarded


def _deliberate(
    execution: ExecutionState,
    pack: ContentPack,
    must_address: NeedId | None = None,
    phase: PhaseId | None = None,
    situations: Iterable[SituationId] = (),
    forbidden: frozenset[ActionId] = frozenset(),
) -> tuple[list[Decision], dict[ActionId, str]]:
    """Candidatas y descartes reales que `decide` va a considerar.

    Único lugar donde se calcula el catálogo completo (`candidates`) y se le
    aplica el filtro de `must_address`. `decide()` y `explain_deliberation()`
    llaman aquí en vez de repetir el filtro cada uno por su lado — así la
    traza nunca puede describir como candidata una acción que `must_address`
    ya había descartado, porque las dos funciones ven exactamente la misma
    lista.
    """
    pool, discarded = candidates(execution, pack, phase=phase, situations=situations)
    if forbidden:
        pool = [d for d in pool if d.action not in forbidden]
    if must_address is not None:
        pool = [
            d
            for d in pool
            if pack.action_by_id[d.action].satisfies.get(must_address, 0.0) > 0.0
        ]
    return pool, discarded


def decide(
    execution: ExecutionState,
    pack: ContentPack,
    must_address: NeedId | None = None,
    phase: PhaseId | None = None,
    situations: Iterable[SituationId] = (),
    forbidden: frozenset[ActionId] = frozenset(),
) -> Decision | None:
    """Selecciona la acción a ejecutar mediante deliberación determinista.

    Flujo:
    1. Si `agent.temperament` es `None`, lanza `UncomposedAgentStateError`.
    2. Obtiene las opciones candidatas disponibles.
    3. Si `must_address` está fijado, filtra las opciones que satisfacen
       ese eje (`action.satisfies.get(must_address, 0.0) > 0.0`). Si ninguna
       lo atiende, retorna `None` inmediatamente (sin fallback a elección
       libre para evitar oscilaciones destructivas).
    4. Si el conjunto de candidatas queda vacío, retorna `None`.
    5. Ordena las candidatas por `_rank` y selecciona los mejores
       `candidate_pool` finalistas.
    6. Si hay un único finalista, lo retorna de inmediato.
    7. Aplica softmax numéricamente estable restando la mejor utilidad (`best`)
       antes de exponenciar, y selecciona con el generador derivado
       `execution.rng.stream("decide", agent.agent_id, execution.tick)`.
    """
    if execution.agent.temperament is None:
        raise UncomposedAgentStateError(
            f"Agent {execution.agent.agent_id!r} has temperament=None. "
            "AgentState.recompose(pack) must be called before decide()."
        )

    pool, _ = _deliberate(
        execution,
        pack,
        must_address=must_address,
        phase=phase,
        situations=situations,
        forbidden=forbidden,
    )

    if not pool:
        return None

    pool.sort(key=_rank)
    finalists = pool[: pack.tuning.candidate_pool]
    if len(finalists) == 1:
        return finalists[0]

    best = finalists[0].utility
    temperature = execution.agent.temperament.temperature
    weights = [math.exp((d.utility - best) / temperature) for d in finalists]
    rng = execution.rng.stream("decide", execution.agent.agent_id, execution.tick)
    return rng.choices(finalists, weights=weights, k=1)[0]


def critical_needs(
    execution: ExecutionState, pack: ContentPack
) -> list[NeedId]:
    """Ejes de necesidad por debajo del umbral crítico del agente.

    Filtra los ejes que declaran `urgent_when_low = False` (los que se acumulan
    en vez de agotarse no constituyen emergencias de interrupción) y aquellos
    cuyo valor actual es inferior al umbral crítico (`critical_need_threshold`).
    El umbral se modula por el compromiso en curso (`switch_cost`) si el agente
    está ejecutando una acción.

    Ordena el resultado por urgencia descendente con desempate por identificador
    de necesidad, garantizando estabilidad total.
    """
    agent = execution.agent
    if agent.temperament is None:
        raise UncomposedAgentStateError(
            f"Agent {agent.agent_id!r} has temperament=None. "
            "AgentState.recompose(pack) must be called before evaluating critical needs."
        )

    threshold = agent.temperament.critical_need_threshold
    if agent.performing is not None and agent.performing.total_ticks > 0:
        progress = max(
            0.0,
            min(1.0, 1.0 - agent.performing.ticks_remaining / agent.performing.total_ticks),
        )
        threshold *= 1.0 - progress * agent.temperament.switch_cost

    urgent: list[NeedId] = []
    for need_id in sorted(agent.needs):
        need = pack.need_by_id.get(need_id)
        if need is None:
            continue
        if not need.urgent_when_low:
            continue
        if agent.needs[need_id] < threshold:
            urgent.append(need_id)

    return sorted(
        urgent,
        key=lambda nid: (
            -urgency(
                agent.needs[nid],
                pack.need_by_id[nid].urgency_exponent,
                agent.cap_of(pack.need_by_id[nid]),
            ),
            str(nid),
        ),
    )


def critical_need(
    execution: ExecutionState, pack: ContentPack
) -> NeedId | None:
    """La necesidad crítica más urgente que justifica interrumpir la acción en curso.

    Aplica triaje: se interrumpe si la acción en curso no atiende la peor
    de las emergencias. Si no hay necesidades críticas o si la acción en curso
    ya atiende la peor, devuelve `None`.
    """
    urgent = critical_needs(execution, pack)
    if not urgent:
        return None
    worst = urgent[0]
    performing = execution.agent.performing
    if performing is not None:
        satisfies = pack.action_by_id[performing.action].satisfies
        if satisfies.get(worst, 0.0) > 0.0:
            return None
    return worst

