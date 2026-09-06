"""Carga y validación del content pack.

Un pack inválido debe explotar AL CARGAR, con un mensaje que liste todos los
problemas detectados. La alternativa es descubrirlo horas después, con el
asistente tomando decisiones anómalas sin pista de qué regla o catálogo está
mal configurado.

Cada `_check_*` devuelve una lista de problemas (vacía si todo está correcto) en
vez de lanzar de inmediato. El orquestador consolida todos los hallazgos y
levanta una única excepción con la lista completa. Diseñar un pack es un ciclo
de ajuste y prueba: un validador que solo informa del primer error convierte
diez erratas en diez cargas frustradas.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from functools import cached_property
from pathlib import Path

import yaml
from pydantic import Field, ValidationError, model_validator

from aida.engine.ontology import (
    ActionId,
    ActionSpec,
    AgentId,
    AgentSpec,
    GoalId,
    GoalSpec,
    HealthSpec,
    NeedId,
    NeedSpec,
    PhaseId,
    PhaseSpec,
    SituationId,
    SituationSpec,
    Spec,
    TraitId,
    TraitSpec,
    TuningSpec,
)


class PackError(ValueError):
    """El pack está mal formado. El mensaje lista todos los problemas."""


class ContentPack(Spec):
    """Un dominio entero, en datos. El motor no conoce ningún otro."""

    id: str
    label: str

    needs: tuple[NeedSpec, ...]
    actions: tuple[ActionSpec, ...]
    agents: tuple[AgentSpec, ...]

    traits: tuple[TraitSpec, ...] = ()
    phases: tuple[PhaseSpec, ...] = ()
    situations: tuple[SituationSpec, ...] = ()
    goals: tuple[GoalSpec, ...] = ()

    health: HealthSpec = Field(default_factory=HealthSpec)
    tuning: TuningSpec = Field(default_factory=TuningSpec)

    # -------------------------------------------------------------------------
    # Índices.
    #
    # `@cached_property` calcula la primera vez que se accede y se guarda en la
    # instancia. La deliberación busca especificaciones por identificador miles
    # de veces por corrida; recorrer una tupla cada vez sería un cuello de botella.
    #
    # IMPORTANTE: estos índices NO se usan durante la validación. Un dict colapsa
    # claves repetidas, ocultando identificadores duplicados justo cuando queremos
    # detectarlos. Las validaciones trabajan siempre sobre las tuplas crudas.
    # -------------------------------------------------------------------------

    @cached_property
    def need_by_id(self) -> dict[NeedId, NeedSpec]:
        return {need.id: need for need in self.needs}

    @cached_property
    def trait_by_id(self) -> dict[TraitId, TraitSpec]:
        return {trait.id: trait for trait in self.traits}

    @cached_property
    def action_by_id(self) -> dict[ActionId, ActionSpec]:
        return {action.id: action for action in self.actions}

    @cached_property
    def agent_by_id(self) -> dict[AgentId, AgentSpec]:
        return {agent.id: agent for agent in self.agents}

    @cached_property
    def phase_by_id(self) -> dict[PhaseId, PhaseSpec]:
        return {phase.id: phase for phase in self.phases}

    @cached_property
    def situation_by_id(self) -> dict[SituationId, SituationSpec]:
        return {situation.id: situation for situation in self.situations}

    @cached_property
    def goal_by_id(self) -> dict[GoalId, GoalSpec]:
        return {goal.id: goal for goal in self.goals}

    # -------------------------------------------------------------------------
    # Métodos auxiliares de validación
    # -------------------------------------------------------------------------

    @staticmethod
    def _duplicates(kind: str, ids: Iterable[str]) -> list[str]:
        counts = Counter(ids)
        return [
            f"{kind}: id duplicado {dup!r} (aparece {n} veces)"
            for dup, n in sorted(counts.items())
            if n > 1
        ]

    @staticmethod
    def _unknown_keys(
        context: str, kind: str, keys: Iterable[str], known: frozenset[str]
    ) -> list[str]:
        """Reporta claves que no existen en `known`.

        `sorted` es indispensable para el determinismo: sin él el orden de los
        mensajes dependería del orden de iteración de conjuntos o diccionarios,
        que varía entre procesos con distinto hash seed.
        """
        return [
            f"{context}: {kind} desconocida/o {key!r}" for key in sorted(set(keys) - known)
        ]

    # -------------------------------------------------------------------------
    # Las 14 comprobaciones deterministas
    # -------------------------------------------------------------------------

    def _check_unique_ids(self) -> list[str]:
        """Ningún identificador repetido dentro de una colección."""
        problems = [
            *self._duplicates("necesidades", (n.id for n in self.needs)),
            *self._duplicates("rasgos", (t.id for t in self.traits)),
            *self._duplicates("acciones", (a.id for a in self.actions)),
            *self._duplicates("agentes", (a.id for a in self.agents)),
            *self._duplicates("franjas", (p.id for p in self.phases)),
            *self._duplicates("situaciones", (s.id for s in self.situations)),
            *self._duplicates("metas", (g.id for g in self.goals)),
        ]
        for agent in self.agents:
            problems += self._duplicates(f"metas de {agent.id!r}", (g.id for g in agent.goals))
        return problems

    def _check_global_id_collisions(self) -> list[str]:
        """Acciones y agentes no pueden compartir id.

        `TasteSpec.target` puede apuntar a cualquiera de los dos. Si una acción
        y un agente comparten identificador, un gusto hacia ese id sería ambiguo.
        """
        action_ids = {str(a.id) for a in self.actions}
        agent_ids = {str(a.id) for a in self.agents}
        problems: list[str] = []
        for clash in sorted(action_ids & agent_ids):
            problems.append(
                f"id {clash!r} usado a la vez por un/a acción y un/a agente: "
                f"los gustos no podrían distinguirlos"
            )
        return problems

    def _check_need_refs(self) -> list[str]:
        """Toda referencia a una necesidad debe existir en el pack."""
        known = frozenset(str(n.id) for n in self.needs)
        problems: list[str] = []

        for trait in self.traits:
            problems += self._unknown_keys(
                f"rasgo {trait.id!r}", "necesidad", trait.need_weight_multipliers, known
            )
            problems += self._unknown_keys(
                f"rasgo {trait.id!r}.gain_multipliers", "necesidad", trait.gain_multipliers, known
            )
            problems += self._unknown_keys(
                f"rasgo {trait.id!r}.cost_multipliers", "necesidad", trait.cost_multipliers, known
            )
            problems += self._unknown_keys(
                f"rasgo {trait.id!r}.cap_multipliers", "necesidad", trait.cap_multipliers, known
            )

        for action in self.actions:
            for field, mapping in (
                ("satisfies", action.satisfies),
                ("costs", action.costs),
                ("requires", action.requires),
                ("effects", action.effects),
            ):
                problems += self._unknown_keys(
                    f"acción {action.id!r}.{field}", "necesidad", mapping, known
                )

        for situation in self.situations:
            problems += self._unknown_keys(
                f"situación {situation.id!r}.gain_multipliers",
                "necesidad",
                situation.gain_multipliers,
                known,
            )
            problems += self._unknown_keys(
                f"situación {situation.id!r}.effects",
                "necesidad",
                situation.effects,
                known,
            )

        for agent in self.agents:
            problems += self._unknown_keys(
                f"agente {agent.id!r}.initial_needs", "necesidad", agent.initial_needs, known
            )
            for goal in agent.goals:
                if str(goal.need) not in known:
                    problems.append(
                        f"agente {agent.id!r}, meta {goal.id!r}: necesidad "
                        f"desconocida {goal.need!r}"
                    )

        for goal in self.goals:
            if str(goal.need) not in known:
                problems.append(
                    f"meta {goal.id!r}: necesidad desconocida {goal.need!r}"
                )

        return problems

    def _check_trait_refs(self) -> list[str]:
        """Todo rasgo referido debe existir en el pack."""
        known = frozenset(str(t.id) for t in self.traits)
        problems: list[str] = []

        for agent in self.agents:
            problems += self._unknown_keys(
                f"agente {agent.id!r}", "rasgo", agent.traits, known
            )
            for goal in agent.goals:
                problems += self._unknown_keys(
                    f"agente {agent.id!r}, meta {goal.id!r}.at_traits",
                    "rasgo",
                    goal.at_traits,
                    known,
                )

        for action in self.actions:
            problems += self._unknown_keys(
                f"acción {action.id!r}.requires_traits", "rasgo", action.requires_traits, known
            )

        for goal in self.goals:
            problems += self._unknown_keys(
                f"meta {goal.id!r}.at_traits",
                "rasgo",
                goal.at_traits,
                known,
            )

        return problems

    def _check_phase_refs(self) -> list[str]:
        """Toda franja referida en acciones debe existir en el pack."""
        known = frozenset(str(p.id) for p in self.phases)
        problems: list[str] = []
        for action in self.actions:
            problems += self._unknown_keys(
                f"acción {action.id!r}", "franja", action.phase_multipliers, known
            )
        for trait in self.traits:
            if trait.phase_shift and not known:
                problems.append(
                    f"rasgo {trait.id!r}: declara `phase_shift` pero el pack no "
                    f"tiene franjas, así que el desfase no puede hacer nada"
                )
        return problems

    def _check_situation_refs(self) -> list[str]:
        """Toda situación referida en acciones debe existir en el pack."""
        known = frozenset(str(s.id) for s in self.situations)
        problems: list[str] = []
        for action in self.actions:
            problems += self._unknown_keys(
                f"acción {action.id!r}", "situación", action.situation_multipliers, known
            )
        return problems

    def _check_goal_refs(self) -> list[str]:
        """Metas: eje, rasgos y `requires_goal` existen, y no hay ciclos."""
        needs = frozenset(str(n.id) for n in self.needs)
        traits = frozenset(str(t.id) for t in self.traits)
        problems: list[str] = []

        for agent in self.agents:
            suyas = {g.id: g for g in agent.goals}
            for goal in agent.goals:
                if str(goal.need) not in needs:
                    problems.append(
                        f"agente {agent.id!r}, meta {goal.id!r}: necesidad "
                        f"desconocida {goal.need!r}"
                    )
                for campo, derivas in (
                    ("at_traits", goal.at_traits),
                    ("on_success", goal.on_success),
                    ("on_failure", goal.on_failure),
                ):
                    problems += self._unknown_keys(
                        f"agente {agent.id!r}, meta {goal.id!r}.{campo}",
                        "rasgo",
                        derivas,
                        traits,
                    )

                if goal.requires_goal is not None:
                    if goal.requires_goal not in suyas:
                        problems.append(
                            f"agente {agent.id!r}, meta {goal.id!r}: espera a "
                            f"{goal.requires_goal!r}, que no es una meta suya"
                        )
                    else:
                        visto: list[GoalId] = []
                        actual: GoalId | None = goal.id
                        while actual is not None and actual in suyas:
                            if actual in visto:
                                cadena = " -> ".join(str(g) for g in [*visto, actual])
                                problems.append(
                                    f"agente {agent.id!r}: ciclo de metas {cadena}. "
                                    f"Ninguna se abriría jamás"
                                )
                                break
                            visto.append(actual)
                            actual = suyas[actual].requires_goal

        pack_goals = {g.id: g for g in self.goals}
        for goal in self.goals:
            if str(goal.need) not in needs:
                problems.append(
                    f"meta {goal.id!r}: necesidad desconocida {goal.need!r}"
                )
            for campo, derivas in (
                ("at_traits", goal.at_traits),
                ("on_success", goal.on_success),
                ("on_failure", goal.on_failure),
            ):
                problems += self._unknown_keys(
                    f"meta {goal.id!r}.{campo}",
                    "rasgo",
                    derivas,
                    traits,
                )
            if goal.requires_goal is not None:
                if goal.requires_goal not in pack_goals:
                    problems.append(
                        f"meta {goal.id!r}: espera a {goal.requires_goal!r}, que no existe"
                    )
                else:
                    visto = []
                    actual = goal.id
                    while actual is not None and actual in pack_goals:
                        if actual in visto:
                            cadena = " -> ".join(str(g) for g in [*visto, actual])
                            problems.append(
                                f"ciclo de metas {cadena}. Ninguna se abriría jamás"
                            )
                            break
                        visto.append(actual)
                        actual = pack_goals[actual].requires_goal

        return problems

    def _check_goals_are_reachable(self) -> list[str]:
        """Toda meta puede cumplirse con el catálogo declarado."""
        movidas = {
            need
            for action in self.actions
            for need, delta in action.effects.items()
            if delta > 0.0
        }
        problems: list[str] = []

        for agent in self.agents:
            suyas = {g.id: g for g in agent.goals}
            concede: dict[TraitId, float] = {}
            for otra in suyas.values():
                for derivas in (otra.on_success, otra.on_failure):
                    for trait, delta in derivas.items():
                        concede[trait] = concede.get(trait, 0.0) + max(delta, 0.0)

            for goal in agent.goals:
                if goal.need not in movidas:
                    problems.append(
                        f"agente {agent.id!r}, meta {goal.id!r}: ninguna acción sube "
                        f"{goal.need!r}, así que nunca podrá cumplirse"
                    )

                for trait, minimo in sorted(goal.at_traits.items()):
                    alcanzable = agent.traits.get(trait, 0.0) + concede.get(trait, 0.0)
                    if alcanzable < minimo:
                        problems.append(
                            f"agente {agent.id!r}, meta {goal.id!r}: pide {trait!r} a "
                            f"{minimo}, tiene {agent.traits.get(trait, 0.0)} y ninguna "
                            f"de sus metas se lo sube — solo las derivas mueven un rasgo"
                        )

        concede_pack: dict[TraitId, float] = {}
        for otra in self.goals:
            for derivas in (otra.on_success, otra.on_failure):
                for trait, delta in derivas.items():
                    concede_pack[trait] = concede_pack.get(trait, 0.0) + max(delta, 0.0)

        for goal in self.goals:
            if goal.need not in movidas:
                problems.append(
                    f"meta {goal.id!r}: ninguna acción sube {goal.need!r}, "
                    f"así que nunca podrá cumplirse"
                )
            for trait, minimo in sorted(goal.at_traits.items()):
                if concede_pack.get(trait, 0.0) < minimo:
                    problems.append(
                        f"meta {goal.id!r}: pide {trait!r} a {minimo} y ninguna meta se lo sube"
                    )

        return problems

    def _check_taste_refs(self) -> list[str]:
        """Todo gusto apunta a una acción o agente que existe."""
        known = (
            frozenset(str(a.id) for a in self.actions)
            | frozenset(str(a.id) for a in self.agents)
        )
        return [
            f"agente {agent.id!r}: gusto hacia {taste.target!r}, que no es "
            f"ninguna acción ni agente"
            for agent in self.agents
            for taste in agent.tastes
            if str(taste.target) not in known
        ]

    def _check_costs_are_declared(self) -> list[str]:
        """Todo efecto negativo debe tener su coste declarado, o 0.0 explícito."""
        problems: list[str] = []
        for action in self.actions:
            for need_id, delta in sorted(action.effects.items()):
                if delta < 0.0 and need_id not in action.costs:
                    problems.append(
                        f"acción {action.id!r}: el efecto sobre {need_id!r} es negativo "
                        f"({delta}) pero no declara coste. Añádelo a `costs`, o pon "
                        f"{need_id}: 0.0 si el agente debe ignorarlo a propósito"
                    )
        return problems

    def _check_requirements_cover_what_they_spend(self) -> list[str]:
        """Una precondición por debajo de lo que la acción consume no sirve."""
        problems: list[str] = []
        for action in self.actions:
            for need_id, minimum in sorted(action.requires.items()):
                gasto = -action.effects.get(need_id, 0.0)
                if gasto > 0.0 and minimum < gasto:
                    problems.append(
                        f"acción {action.id!r}: exige {minimum} de {need_id!r} pero "
                        f"puede consumir hasta {round(gasto, 4)}. Un "
                        f"agente justo por encima del mínimo seguiría truncando el "
                        f"efecto; sube el requisito o quítalo"
                    )
        return problems

    def _check_actions_are_reachable(self) -> list[str]:
        """Cada acción puede ejecutarse al menos una vez por algún agente."""
        movidas = {
            need
            for a in self.actions
            for need, delta in a.effects.items()
            if delta > 0.0
        }
        need_defaults = {n.id: n.initial for n in self.needs}
        problems: list[str] = []

        for action in self.actions:
            for need_id, minimum in sorted(action.requires.items()):
                if minimum > 1.0:
                    problems.append(
                        f"acción {action.id!r}: exige {minimum} de {need_id!r}, pero "
                        f"el valor máximo de una necesidad es 1.0; jamás será ejecutable"
                    )
                elif need_id not in movidas and self.agents:
                    empieza_alguno = any(
                        agent.initial_needs.get(need_id, need_defaults.get(need_id, 0.0)) >= minimum
                        for agent in self.agents
                    )
                    if not empieza_alguno:
                        problems.append(
                            f"acción {action.id!r}: exige {minimum} de {need_id!r}, pero "
                            f"ningún agente empieza con ese nivel y ninguna acción lo sube; "
                            f"jamás será ejecutable"
                        )

            for trait_id, req_val in sorted(action.requires_traits.items()):
                if req_val > 1.0:
                    problems.append(
                        f"acción {action.id!r}: exige rasgo {trait_id!r} a {req_val}, pero "
                        f"los rasgos no superan 1.0; jamás será ejecutable"
                    )
                elif self.agents:
                    alcanza_alguno = False
                    for agent in self.agents:
                        derivas_max = sum(
                            max(g.on_success.get(trait_id, 0.0), 0.0)
                            + max(g.on_failure.get(trait_id, 0.0), 0.0)
                            for g in agent.goals
                        )
                        if agent.traits.get(trait_id, 0.0) + derivas_max >= req_val:
                            alcanza_alguno = True
                            break
                    if not alcanza_alguno:
                        problems.append(
                            f"acción {action.id!r}: exige rasgo {trait_id!r} a {req_val}, pero "
                            f"ningún agente lo alcanza ni tiene metas que lo suban; "
                            f"jamás será ejecutable"
                        )

        return problems

    def _check_health_refs(self) -> list[str]:
        """health.may_go_unused apunta a acciones existentes en el catálogo."""
        known = frozenset(str(a.id) for a in self.actions)
        return [
            f"health.may_go_unused: acción desconocida {action_id!r}"
            for action_id in sorted(self.health.may_go_unused)
            if str(action_id) not in known
        ]

    def _check_declared_bounds(self) -> list[str]:
        """Valida cotas de initial_needs en [0.0, 1.0] y at_traits en [-2.0, 2.0]."""
        problems: list[str] = []
        for agent in self.agents:
            for need_id, val in sorted(agent.initial_needs.items()):
                if not 0.0 <= val <= 1.0:
                    problems.append(
                        f"agente {agent.id!r}.initial_needs[{need_id!r}] = {val}, "
                        f"fuera de [0.0, 1.0]"
                    )
            for goal in agent.goals:
                for trait_id, val in sorted(goal.at_traits.items()):
                    if not -2.0 <= val <= 2.0:
                        problems.append(
                            f"agente {agent.id!r}, meta {goal.id!r}."
                            f"at_traits[{trait_id!r}] = {val}, fuera de [-2.0, 2.0]"
                        )
        for goal in self.goals:
            for trait_id, val in sorted(goal.at_traits.items()):
                if not -2.0 <= val <= 2.0:
                    problems.append(
                        f"meta {goal.id!r}.at_traits[{trait_id!r}] = {val}, fuera de [-2.0, 2.0]"
                    )
        return problems

    # -------------------------------------------------------------------------
    # Orquestador
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate(self) -> ContentPack:
        all_problems = [
            *self._check_unique_ids(),
            *self._check_global_id_collisions(),
            *self._check_need_refs(),
            *self._check_trait_refs(),
            *self._check_phase_refs(),
            *self._check_situation_refs(),
            *self._check_goal_refs(),
            *self._check_goals_are_reachable(),
            *self._check_taste_refs(),
            *self._check_costs_are_declared(),
            *self._check_requirements_cover_what_they_spend(),
            *self._check_actions_are_reachable(),
            *self._check_health_refs(),
            *self._check_declared_bounds(),
        ]
        problems = list(dict.fromkeys(all_problems))
        if problems:
            raise ValueError(
                f"pack {self.id!r} inválido:\n" + "\n".join(f"  - {p}" for p in problems)
            )
        return self


def load_pack(path: Path) -> ContentPack:
    """Lee un pack desde disco.

    `safe_load`, nunca `load`: `yaml.load` a secas puede instanciar objetos
    Python arbitrarios desde el fichero. Con packs propios da igual; el día que
    cargues el pack de otra persona, no.

    Gotcha de pydantic: si un validador lanza `ValueError`, pydantic lo captura
    y lo re-lanza envuelto en `ValidationError`. El mensaje sobrevive, el tipo
    no. Por eso se traduce aquí a `PackError`: el resto del programa captura un
    tipo propio y no una excepción de la librería.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ContentPack.model_validate(raw)
    except ValidationError as exc:
        raise PackError(f"no se pudo cargar {path}:\n{exc}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise PackError(f"no se pudo cargar {path}:\n{exc}") from exc


def describe_pack(pack: ContentPack) -> Mapping[str, int]:
    """Resumen numérico por colección del pack."""
    return {
        "needs": len(pack.needs),
        "traits": len(pack.traits),
        "actions": len(pack.actions),
        "agents": len(pack.agents),
        "phases": len(pack.phases),
        "situations": len(pack.situations),
        "goals": len(pack.goals),
    }
