"""Salud de un pack: mide una corrida y comprueba lo que el pack promete.

`HealthSpec` (en `ontology.py`) no es un parámetro de simulación: es
intención de diseño verificable. Una afirmación como "bajo esta carga, el
agente decide con presión moderada, ni indiferente ni desbordado" es
comprobable, y conviene que falle ruidosamente en cuanto deja de ser cierta.

Las bandas no están aquí: las declara cada pack en su sección `health`. Un
test con los umbrales clavados pelearía contra la calibración —cambiar un
`decay_per_tick` para mejorar el pack rompería el test aunque el cambio
fuera bueno.

Vive por encima de `engine/` y de `agents/` porque usa el bucle, el motor y
la deliberación a la vez.

`days` se usa hoy como el número de instantes que se llama a `step()`, no
como días de calendario: no existe todavía una duración de ciclo declarada
en el pack (`I-035`). El día en que exista, esta pieza tendrá que
multiplicar por ella en vez de usar `days` tal cual.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations

from aida.agents.decide import candidates, critical_needs
from aida.engine.events import EventKind
from aida.engine.loop import Loop
from aida.engine.ontology import ActionId, AgentSpec, NeedId
from aida.engine.pack import ContentPack
from aida.engine.rng import RngSource
from aida.engine.state import AgentState, ExecutionState


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Lo medido. Promedios sobre las semillas y las fichas de personalidad
    que declare el pack."""

    days: int
    seeds: tuple[int, ...]
    agents: int

    crisis_time: float
    """Fracción media del tiempo con al menos una necesidad bajo el umbral
    crítico del agente (`agents.decide.critical_needs` no vacía)."""

    abort_ratio: float
    """Abandonos frente al total de acciones que llegaron a un final
    (abandonadas + completadas). `0.0` si nadie terminó ni abandonó nada."""

    completed: int
    """Total de acciones completadas, sumando todas las fichas y semillas."""

    stalls: int
    """Instantes en los que el agente quedó libre y `decide()` no encontró
    ninguna opción disponible. Universal: no depende de ninguna banda del
    pack — un agente sin ninguna opción está roto se proponga lo que se
    proponga `health`."""

    unused: frozenset[ActionId] = field(default_factory=frozenset)
    """Acciones del catálogo que nunca se completaron, en ninguna ficha ni
    semilla."""

    differentiation: float | None = None
    """Distancia L1 media entre las PREFERENCIAS de cada par de fichas de
    personalidad, evaluadas sobre los mismos estados. `None` si el pack
    declara menos de dos — no hay con qué comparar, y no es lo mismo que un
    `0.0` medido de verdad.

    Mide carácter, no conducta, y esa distinción no es un matiz: la
    aleatoriedad se deriva por `agent_id`, así que dos fichas idénticas
    ACTÚAN distinto por puro azar. Comparar lo que acabaron haciendo daba el
    mismo número para dos personalidades opuestas que para dos gemelas
    (`I-045`). Comparar lo que prefieren en el mismo estado da `0.0` exacto
    entre gemelas."""

    def lines(self) -> list[str]:
        """Panel legible, para mirar mientras se calibra."""
        out = [
            f"  {self.days} instantes · {self.agents} ficha(s) · "
            f"semillas {list(self.seeds)}",
            "",
            f"  tiempo en crisis        {self.crisis_time:6.1%}",
            f"  abandonos               {self.abort_ratio:6.1%} "
            f"({self.completed} completadas)",
            f"  bloqueos                {self.stalls:6}",
            f"  acciones sin usar       "
            f"{sorted(self.unused) if self.unused else '—'}",
        ]
        if self.differentiation is not None:
            out.append(f"  diferenciación          {self.differentiation:6.3f}")
        return out


def _utility_profile(
    pack: ContentPack, spec: AgentSpec, needs: Mapping[NeedId, float]
) -> dict[ActionId, float]:
    """Qué prefiere ESTA ficha de personalidad en ESTE estado, sin azar de por
    medio.

    Nace un agente desde el pack, se le impone el estado dado y se leen las
    utilidades que `candidates` calcula, normalizadas por la suma de sus
    valores absolutos para que lo que se compare sea la forma de la
    preferencia y no su escala.

    No interviene el sorteo. Es lo que distingue esta medida de contar lo que
    cada ficha acabó haciendo: el RNG se deriva por `agent_id`, así que dos
    fichas idénticas actúan distinto por puro azar, y esa diferencia se colaba
    entera en la métrica anterior (`I-045`). Aquí dos fichas idénticas dan
    exactamente el mismo perfil, y cualquier diferencia que aparezca es
    carácter.
    """
    agent = AgentState.from_spec(pack, spec)
    agent.needs = dict(needs)
    agent.refresh_state(pack.need_by_id)
    agent.recompose(pack)
    execution = ExecutionState(agent=agent, tick=0, rng=RngSource(0))
    pool, _ = candidates(execution, pack)
    norm = sum(abs(decision.utility) for decision in pool)
    if norm == 0.0:
        return {decision.action: 0.0 for decision in pool}
    return {decision.action: decision.utility / norm for decision in pool}


def _l1_distance(a: Mapping[ActionId, float], b: Mapping[ActionId, float]) -> float:
    """Cuánto difieren dos perfiles de actividad, sumando la diferencia
    absoluta eje a eje (acción a acción)."""
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in sorted(keys))


def measure(
    pack: ContentPack,
    *,
    days: int | None = None,
    seeds: tuple[int, ...] | None = None,
) -> HealthReport:
    """Corre el pack `days` instantes por cada ficha de personalidad y cada
    semilla, y mide lo que pasó. Ver la cabecera del módulo sobre `days`.
    """
    spec = pack.health
    total_days = days if days is not None else spec.days
    total_seeds = seeds if seeds is not None else spec.seeds

    crisis_fracs: list[float] = []
    stalls = 0
    completed_total = 0
    aborted_total = 0
    ever_completed: set[ActionId] = set()
    fichas = sorted(pack.agents, key=lambda a: a.id)
    sampled_states: list[dict[NeedId, float]] = []

    for agent_spec in fichas:
        for seed in total_seeds:
            loop = Loop.start(
                pack, agent_spec, seed=seed, inbox_capacity=1, log_window=None
            )
            crisis_ticks = 0
            # Los estados de UNA sola corrida, la de la primera ficha con la
            # primera semilla, sirven de banco de pruebas común: todas las
            # fichas se juzgan sobre los mismos estados, así que la
            # comparación no hereda el azar de la corrida de cada una.
            muestrear = agent_spec is fichas[0] and seed == total_seeds[0]

            for _ in range(total_days):
                loop.step()
                if critical_needs(loop.execution, pack):
                    crisis_ticks += 1
                if loop.execution.agent.performing is None:
                    stalls += 1
                if muestrear:
                    sampled_states.append(dict(loop.execution.agent.needs))

            for event in loop.log:
                if event.kind is EventKind.ACTION_COMPLETED and event.action is not None:
                    ever_completed.add(event.action)
                    completed_total += 1
                elif event.kind is EventKind.ACTION_ABORTED:
                    aborted_total += 1

            crisis_fracs.append(crisis_ticks / total_days if total_days else 0.0)

    crisis_time = sum(crisis_fracs) / len(crisis_fracs) if crisis_fracs else 0.0
    denom = completed_total + aborted_total
    abort_ratio = aborted_total / denom if denom else 0.0
    unused = frozenset(pack.action_by_id) - ever_completed

    differentiation: float | None = None
    if len(fichas) >= 2 and sampled_states:
        distances = [
            _l1_distance(
                _utility_profile(pack, ficha_a, estado),
                _utility_profile(pack, ficha_b, estado),
            )
            for estado in sampled_states
            for ficha_a, ficha_b in combinations(fichas, 2)
        ]
        differentiation = sum(distances) / len(distances)

    return HealthReport(
        days=total_days,
        seeds=total_seeds,
        agents=len(pack.agents),
        crisis_time=crisis_time,
        abort_ratio=abort_ratio,
        completed=completed_total,
        stalls=stalls,
        unused=unused,
        differentiation=differentiation,
    )


def check(pack: ContentPack, report: HealthReport | None = None) -> list[str]:
    """Contrasta lo medido con lo que `pack.health` promete. Vacío = en orden.

    Los dos primeros son invariantes universales: no dependen de ninguna
    banda declarada, porque un agente que nunca termina nada o que se queda
    sin opciones está roto se proponga lo que se proponga `health`.
    """
    report = report if report is not None else measure(pack)
    spec = pack.health
    problems: list[str] = []

    if report.completed == 0:
        problems.append("nadie completó ninguna acción en toda la corrida")
    if report.stalls:
        problems.append(
            f"{report.stalls} instantes sin ninguna opción disponible: "
            "revisa `requires`/`requires_traits` del catálogo"
        )

    if spec.min_crisis is not None and report.crisis_time < spec.min_crisis:
        problems.append(
            f"tiempo en crisis {report.crisis_time:.1%} por debajo del mínimo "
            f"{spec.min_crisis:.1%}: sin presión, el carácter deja de notarse"
        )
    if spec.max_crisis is not None and report.crisis_time > spec.max_crisis:
        problems.append(
            f"tiempo en crisis {report.crisis_time:.1%} por encima del máximo "
            f"{spec.max_crisis:.1%}: la emergencia aplasta al carácter"
        )
    if spec.max_abort_ratio is not None and report.abort_ratio > spec.max_abort_ratio:
        problems.append(
            f"abandonos {report.abort_ratio:.1%} por encima del máximo "
            f"{spec.max_abort_ratio:.1%}"
        )
    if (
        spec.min_differentiation is not None
        and report.differentiation is not None
        and report.differentiation < spec.min_differentiation
    ):
        problems.append(
            f"diferenciación {report.differentiation:.3f} por debajo del "
            f"mínimo {spec.min_differentiation:.3f}: dos fichas de "
            "personalidad distintas están decidiendo casi igual"
        )

    orphans = report.unused - spec.may_go_unused
    if orphans:
        problems.append(
            f"acciones declaradas que nunca se completan: {sorted(orphans)}. "
            "Si es intencional, decláralas en `health.may_go_unused`"
        )

    return problems
