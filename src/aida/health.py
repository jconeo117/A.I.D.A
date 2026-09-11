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

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations

from aida.agents.decide import critical_needs
from aida.engine.events import EventKind
from aida.engine.loop import Loop
from aida.engine.ontology import ActionId, ActionSpec, AgentId
from aida.engine.pack import ContentPack


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
    """Distancia L1 media entre los perfiles de actividad de cada par de
    fichas de personalidad que el pack declare. `None` si el pack declara
    menos de dos — no hay con qué comparar, y no es lo mismo que un `0.0`
    medido de verdad."""

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


def _average_profile(
    runs: list[Counter[ActionId]], actions: Mapping[ActionId, ActionSpec]
) -> dict[ActionId, float]:
    """Perfil de actividad normalizado, promediado entre semillas.

    Cada semilla se normaliza por separado ANTES de promediar —la fracción
    que representó cada acción sobre el total de esa semilla, no el conteo
    bruto—, para que una semilla que por azar completó más acciones no pese
    más que otra solo por eso.
    """
    fractions: list[dict[ActionId, float]] = []
    for counts in runs:
        total = sum(counts.values())
        if total == 0:
            continue
        fractions.append({aid: n / total for aid, n in counts.items()})
    result: dict[ActionId, float] = {aid: 0.0 for aid in actions}
    if not fractions:
        return result
    for profile in fractions:
        for aid, frac in profile.items():
            result[aid] = result.get(aid, 0.0) + frac
    return {aid: v / len(fractions) for aid, v in result.items()}


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
    profiles: dict[AgentId, list[Counter[ActionId]]] = {}

    for agent_spec in sorted(pack.agents, key=lambda a: a.id):
        for seed in total_seeds:
            loop = Loop.start(
                pack, agent_spec, seed=seed, inbox_capacity=1, log_window=None
            )
            crisis_ticks = 0
            completions: Counter[ActionId] = Counter()

            for _ in range(total_days):
                loop.step()
                if critical_needs(loop.execution, pack):
                    crisis_ticks += 1
                if loop.execution.agent.performing is None:
                    stalls += 1

            for event in loop.log:
                if event.kind is EventKind.ACTION_COMPLETED and event.action is not None:
                    completions[event.action] += 1
                    ever_completed.add(event.action)
                    completed_total += 1
                elif event.kind is EventKind.ACTION_ABORTED:
                    aborted_total += 1

            crisis_fracs.append(crisis_ticks / total_days if total_days else 0.0)
            profiles.setdefault(agent_spec.id, []).append(completions)

    crisis_time = sum(crisis_fracs) / len(crisis_fracs) if crisis_fracs else 0.0
    denom = completed_total + aborted_total
    abort_ratio = aborted_total / denom if denom else 0.0
    unused = frozenset(pack.action_by_id) - ever_completed

    differentiation: float | None = None
    if len(profiles) >= 2:
        avg_profiles = {
            aid: _average_profile(runs, pack.action_by_id)
            for aid, runs in profiles.items()
        }
        distances = [
            _l1_distance(avg_profiles[a], avg_profiles[b])
            for a, b in combinations(sorted(avg_profiles), 2)
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
