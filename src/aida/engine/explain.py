"""Explicabilidad y traza causal de deliberación.

Este módulo produce la cadena causal de una decisión tomada por el motor:
qué se eligió, qué quedó segunda y por cuánto margen, de dónde procede cada
utilidad (leyendo los cinco factores directamente de `Decision`, nunca
recalculándolos), qué necesidades y pesos empujaron la elección, qué creencias
aprendidas intervinieron y qué opciones quedaron descartadas por puertas duras.

Invariantes de diseño:
1. No recalculación: la traza lee los términos de `Decision` directamente
   (`benefit`, `cost`, `novelty`, `phase_fit`, `situation_fit`, `utility`).
   Recalcularlos arriesgaría divergencias silenciosas si las fórmulas evolucionan.
2. Fidelidad bit a bit: la utilidad de la acción ganadora en la traza es
   estrictamente idéntica a `decision.utility`.
3. Determinismo y serialización: el volcado a JSONL (`to_json`) ordena
   recursivamente todas las claves lexicográficamente, garantizando que dos
   trazas lógicamente idénticas produzcan bytes idénticos entre procesos.
4. Agnosticismo: la traza no utiliza vocabulario de dominio ni asume ningún
   contexto organizativo o de operaciones concreto.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aida.agents.decide import Decision, _rank, candidates, critical_needs, decide
from aida.engine.disposition import urgency
from aida.engine.ontology import ActionId, NeedId, PhaseId, SituationId
from aida.engine.pack import ContentPack
from aida.engine.state import ExecutionState


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Registro inmutable de la deliberación causal de un agente.

    Contiene todos los datos que justifican una elección: la acción ganadora,
    la alternativa inmediata, el desglose aritmético no recalculado de cada
    candidata evaluada, el vector de presiones y pesos de necesidades, las
    desviaciones de creencias aprendidas y las opciones vetadas por puertas duras.
    """

    agent_id: str
    tick: int
    chosen: ActionId
    utility: float
    runner_up: ActionId | None
    margin: float | None
    candidates: tuple[Decision, ...]
    discarded: dict[ActionId, str]
    need_urgencies: dict[NeedId, float]
    need_weights: dict[NeedId, float]
    critical_needs: tuple[NeedId, ...]
    belief_deviations: dict[ActionId, dict[NeedId, float]]

    def to_json(self) -> str:
        """Serializa la traza en una sola línea JSONL con claves ordenadas.

        Reutiliza el criterio estricto de `Event.to_json`: claves ordenadas en
        cada nivel del diccionario, separadores compactos `(",", ":")`,
        `ensure_ascii=False` para legibilidad humana directa y escape explícito
        de separadores de línea Unicode `\\u2028` y `\\u2029`.
        """
        candidates_data = [
            {
                "action": str(c.action),
                "benefit": c.benefit,
                "cost": c.cost,
                "novelty": c.novelty,
                "phase_fit": c.phase_fit,
                "situation_fit": c.situation_fit,
                "utility": c.utility,
            }
            for c in self.candidates
        ]
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "belief_deviations": {
                str(aid): {str(nid): val for nid, val in sorted(devs.items())}
                for aid, devs in sorted(self.belief_deviations.items())
            },
            "candidates": candidates_data,
            "chosen": str(self.chosen),
            "critical_needs": [str(n) for n in self.critical_needs],
            "discarded": {
                str(aid): door for aid, door in sorted(self.discarded.items())
            },
            "margin": self.margin,
            "need_urgencies": {
                str(nid): val for nid, val in sorted(self.need_urgencies.items())
            },
            "need_weights": {
                str(nid): val for nid, val in sorted(self.need_weights.items())
            },
            "runner_up": (
                str(self.runner_up) if self.runner_up is not None else None
            ),
            "tick": self.tick,
            "utility": self.utility,
        }
        sorted_payload = {key: payload[key] for key in sorted(payload)}
        line = json.dumps(sorted_payload, separators=(",", ":"), ensure_ascii=False)
        return line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    def format_human(self) -> str:
        """Formatea la traza para presentación legible en terminal."""
        lines: list[str] = [
            f"=== Deliberación: agente={self.agent_id} (tick={self.tick}) ===",
            "",
            "Decisión:",
            f"  Acción elegida:     {self.chosen}",
            f"  Utilidad ganadora:  {self.utility:+.6f}",
        ]
        if self.runner_up is not None and self.margin is not None:
            lines.append(f"  Segunda opción:     {self.runner_up}")
            lines.append(f"  Margen:             {self.margin:+.6f}")
        else:
            lines.append("  Segunda opción:     (sin alternativa)")
            lines.append("  Margen:             (sin alternativa)")

        lines.append("")
        lines.append(f"Opciones evaluadas ({len(self.candidates)}):")
        lines.append(
            "  Acción              Utilidad   Beneficio     Coste   Novedad    Franja Situación"
        )
        lines.append(
            "  --------------------------------------------------------------------------------"
        )
        for c in self.candidates:
            lines.append(
                f"  {c.action:<18} {c.utility:+10.6f} {c.benefit:11.6f} {c.cost:9.6f} "
                f"{c.novelty:9.4f} {c.phase_fit:9.4f} {c.situation_fit:10.4f}"
            )

        lines.append("")
        lines.append("Presiones y pesos por necesidad:")
        lines.append(
            "  Eje                  Urgencia    Peso        Presión efectiva"
        )
        lines.append(
            "  -------------------------------------------------------------"
        )
        for need_id in sorted(self.need_urgencies):
            urg = self.need_urgencies[need_id]
            wt = self.need_weights.get(need_id, 1.0)
            eff = urg * wt
            lines.append(f"  {need_id:<20} {urg:10.6f} {wt:10.6f}  {eff:16.6f}")

        if self.critical_needs:
            lines.append(
                f"  Necesidades críticas: {', '.join(str(n) for n in self.critical_needs)}"
            )
        else:
            lines.append("  Necesidades críticas: (ninguna bajo el umbral)")

        lines.append("")
        lines.append("Desviación de creencias aprendidas respecto al pack:")
        if self.belief_deviations:
            for action_id in sorted(self.belief_deviations):
                devs = self.belief_deviations[action_id]
                dev_str = ", ".join(
                    f"{nid}: {val:+.4f}" for nid, val in sorted(devs.items())
                )
                lines.append(f"  {action_id}: {dev_str}")
        else:
            lines.append(
                "  (ninguna desviación; el agente utiliza las promesas del pack)"
            )

        lines.append("")
        lines.append(
            f"Opciones descartadas por puertas duras ({len(self.discarded)}):"
        )
        if self.discarded:
            for action_id in sorted(self.discarded):
                door = self.discarded[action_id]
                lines.append(f"  {action_id}: descartada por puerta {door!r}")
        else:
            lines.append("  (ninguna opción descartada)")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_human()


DeliberationTrace = DecisionTrace


def build_trace(
    execution: ExecutionState,
    pack: ContentPack,
    decision: Decision,
    pool: Sequence[Decision],
    discarded: Mapping[ActionId, str],
) -> DecisionTrace:
    """Construye la traza explicativa a partir de la decisión y las candidatas evaluadas.

    Lee los sumandos y factores directamente de los objetos `Decision`. No los
    recalcula en ningún caso.
    """
    agent = execution.agent
    ranked_pool = sorted(pool, key=_rank)

    runner_up: ActionId | None = None
    margin: float | None = None
    if len(ranked_pool) > 1:
        if ranked_pool[0].action == decision.action:
            runner_up = ranked_pool[1].action
            margin = ranked_pool[0].utility - ranked_pool[1].utility
        else:
            runner_up = ranked_pool[0].action
            margin = abs(decision.utility - ranked_pool[0].utility)

    need_urgencies: dict[NeedId, float] = {}
    need_weights: dict[NeedId, float] = {}
    for need_id in sorted(agent.needs):
        need = pack.need_by_id.get(need_id)
        if need is not None:
            need_urgencies[need_id] = urgency(
                agent.needs[need_id], need.urgency_exponent, agent.cap_of(need)
            )
            need_weights[need_id] = agent.need_weights.get(need_id, 1.0)

    active_critical = tuple(critical_needs(execution, pack))

    belief_deviations: dict[ActionId, dict[NeedId, float]] = {}
    for c in pool:
        g_beliefs = agent.gain_beliefs.get(c.action, {})
        c_beliefs = agent.cost_beliefs.get(c.action, {})
        all_needs = sorted(set(g_beliefs) | set(c_beliefs))
        action_devs: dict[NeedId, float] = {}
        for nid in all_needs:
            if nid in g_beliefs and abs(g_beliefs[nid] - 1.0) > 1e-9:
                action_devs[nid] = g_beliefs[nid] - 1.0
            elif nid in c_beliefs and abs(c_beliefs[nid] - 1.0) > 1e-9:
                action_devs[nid] = c_beliefs[nid] - 1.0
        if action_devs:
            belief_deviations[c.action] = action_devs

    return DecisionTrace(
        agent_id=str(agent.agent_id),
        tick=execution.tick,
        chosen=decision.action,
        utility=decision.utility,
        runner_up=runner_up,
        margin=margin,
        candidates=tuple(ranked_pool),
        discarded={aid: door for aid, door in sorted(discarded.items())},
        need_urgencies=need_urgencies,
        need_weights=need_weights,
        critical_needs=active_critical,
        belief_deviations=belief_deviations,
    )


def explain_deliberation(
    execution: ExecutionState,
    pack: ContentPack,
    must_address: NeedId | None = None,
    phase: PhaseId | None = None,
    situations: Iterable[SituationId] = (),
) -> tuple[Decision | None, DecisionTrace | None]:
    """Ejecuta una deliberación y devuelve la decisión tomada junto con su traza.

    Retorna `(None, None)` si ninguna opción resulta elegible o ejecutable.
    """
    pool, discarded = candidates(
        execution, pack, phase=phase, situations=situations
    )
    decision = decide(
        execution,
        pack,
        must_address=must_address,
        phase=phase,
        situations=situations,
    )
    if decision is None:
        return None, None

    trace = build_trace(execution, pack, decision, pool, discarded)
    return decision, trace
