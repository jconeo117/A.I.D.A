"""Estado mutable de una ejecución.

`ontology.py` define cómo ES la ejecución: definiciones que vienen del pack y
no cambian una vez cargadas. Este archivo define cómo ESTÁ ahora mismo: lo que
un tick puede tocar. Es la distinción que `ontology.py` ya declara en su
cabecera (`Spec` frente a `State`), aplicada.

**Derivados y personalidad.** `AgentState` mantiene los rasgos en bruto
y compone pesos, aptitud, temple, techos y temperamento a través de
`engine/disposition.py`. Las expectativas empíricas derivadas de resultados
se actualizan a través de `engine/outcome.py`. El cálculo de urgencia reside
exclusivamente en `disposition.py` como puerta canónica única.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from aida.engine.disposition import (
    Temperament,
    compose_caps,
    compose_costs,
    compose_gains,
    compose_need_weights,
    compose_temperament,
    compute_strain,
)
from aida.engine.disposition import (
    urgency as urgency,
)
from aida.engine.events import EventKind, EventLog
from aida.engine.ontology import (
    ActionId,
    AgentId,
    GoalId,
    GoalStatus,
    NeedId,
    NeedSpec,
    SituationId,
    SituationSpec,
    TraitId,
)
from aida.engine.pack import ContentPack
from aida.engine.rng import RngSource

# -----------------------------------------------------------------------------
# Qué está haciendo el agente ahora mismo.
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerformingAction:
    """Lo que sostiene el compromiso con excepción de urgencia.

    Frozen y no pydantic, a propósito: como el `Event` de `events.py`, lo
    construye el motor con datos que él mismo acaba de calcular al decidir —
    no llega del pack ni del exterior, así que pagar una validación de
    pydantic sería comprobar algo que ya se sabía cierto. `Event` ya sentó
    este criterio; aquí se repite en vez de inventar uno nuevo.

    No lleva a quién se lo hace: no existe otro agente al que apuntar todavía,
    así que esa pieza —`target` en la versión de la que nace— no significa
    nada aquí. Vuelve si el entorno llega a modelarse con las mismas
    primitivas.
    """

    action: ActionId

    ticks_remaining: int
    """Lo que queda. En 0, la acción ya terminó y solo falta que alguien lo
    note y decida qué sigue."""

    total_ticks: int = 0
    """Lo que se planificó al empezar. De aquí sale cuánto se lleva invertido
    —`total_ticks - ticks_remaining`—, que es lo que pesa el coste de cambio
    de contexto: a mitad de camino, soltarlo cuesta más que al principio."""

    def __post_init__(self) -> None:
        if self.ticks_remaining < 0:
            raise ValueError(f"ticks_remaining = {self.ticks_remaining}, no puede ser negativo")
        if self.total_ticks < 0:
            raise ValueError(f"total_ticks = {self.total_ticks}, no puede ser negativo")


# -----------------------------------------------------------------------------
# El libro de calibración.
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class Ledger:
    """Cuánto se ofrece y cuánto se pide de cada eje. Solo para calibrar.

    Es el hermano de `HealthSpec`: un eje que nunca aprieta se esconde detrás
    de los que sí, y un panel agregado puede parecer sano durante toda una
    ejecución mientras un eje entero lleva miles de muestras sin hacer nada.
    Este libro es lo que permite mirar eje por eje en vez de fiarse del
    agregado.

    Se anota lo NOMINAL, no lo que cabe: una reposición que se pierde contra
    el techo cuenta igual, porque la pregunta es cuánto hay en oferta, y el
    excedente contra el techo es justo la señal de que el eje está resuelto.

    Apagado por omisión (`AgentState.ledger is None`): una ejecución normal
    no paga por esto.
    """

    demand: dict[NeedId, float] = field(default_factory=dict)
    """Lo que se pierde: decaimiento bruto más lo que cuesta lo que se hace."""

    passive: dict[NeedId, float] = field(default_factory=dict)
    """Lo que llega sin actuar, por estar bajo el alcance de una situación
    (`SituationSpec.effects`). Aparte de `active` y no restado de él: la
    pregunta que este libro responde es de dónde viene la oferta, y fundir
    las dos borraría esa distinción."""

    active: dict[NeedId, float] = field(default_factory=dict)
    """Lo que llega por actuar. Es lo único que una decisión puede mover."""

    def record(self, book: dict[NeedId, float], need_id: NeedId, amount: float) -> None:
        book[need_id] = book.get(need_id, 0.0) + amount


# -----------------------------------------------------------------------------
# Estado del agente.
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class AgentState:
    """Cómo está el agente ahora mismo, y qué está haciendo.

    `slots=True`: sin `__dict__`, ni memoria ni acceso de más para lo que se
    toca en cada tick. A cambio no se le pueden añadir atributos al vuelo, que
    es justo lo que conviene en el estado caliente.
    """

    agent_id: AgentId
    """Solo el identificador, no la ficha entera. El estado tiene que poder
    guardarse y recargarse sin arrastrar definiciones del pack."""

    needs: dict[NeedId, float]
    """Vive en `[0, 1]` — más exactamente, en `[0, cap]`. Ningún consumidor de
    este diccionario se entera de si hay deuda detrás: eso es justo lo que
    `debt` existe para separar."""

    debt: dict[NeedId, float] = field(default_factory=dict)
    """Lo que el suelo se ha comido, por eje. Siempre >= 0.

    Sin esto, un eje que lleva mucho hundido y uno que acaba de tocar fondo
    son indistinguibles: los dos están en 0.0 y el recorte del suelo se comió
    la diferencia. **Reponer paga esto primero** (`restore`): por eso un eje
    con deuda tarda más en notarse repuesto que uno sin ella, aunque la
    reposición sea idéntica en los dos.

    Disperso: solo aparecen aquí los ejes que de verdad han estado por debajo
    del suelo y cuya `NeedSpec.max_debt` es mayor que 0. Un eje sin deuda
    declarada no ocupa ni una entrada.
    """

    floor_streak: dict[NeedId, int] = field(default_factory=dict)
    """Ticks SEGUIDOS con este eje en el suelo (valor 0.0). Cuenta cuánto
    llevas, no cuán hundido estás — eso es `debt`, y son independientes: se
    puede tocar fondo un instante o pasar mucho tiempo raspando el suelo, y
    solo lo segundo cuenta aquí.

    **Se reinicia de golpe** en cuanto el eje se levanta del suelo
    (`advance_floor_streak`), a diferencia de `debt`, que se paga a plazos.
    Cruzar `NeedSpec.escalates_after` no decide nada por sí mismo — escala:
    emite `EventKind.NEED_ESCALATED` y deja que quien lo consuma decida qué
    hacer con la noticia.

    Disperso como la deuda: solo aparecen los ejes con `escalates_after > 0`
    que de verdad han tocado el suelo alguna vez.
    """

    traits: dict[TraitId, float] = field(default_factory=dict)
    """Rasgo -> intensidad actual, en `[-1, 1]`. Guardado en bruto: qué hace
    cada rasgo lo dice `TraitSpec`, compartido por el catálogo entero; esto es
    solo la ficha de intensidades de este agente en concreto."""

    goals: dict[GoalId, GoalStatus] = field(default_factory=dict)
    """En qué anda cada meta. Solo el estado, no la meta: qué se quiere lo
    dice el pack y no cambia; lo que cambia es si se ha conseguido, se ha
    perdido o sigue en curso."""

    saturation: dict[ActionId, float] = field(default_factory=dict)
    """Hastío por acción, en `[0, 1]`. Sube al completar (`saturate`), se
    disipa cada tick (`recover_novelty`). Uno se cansa de lo que termina, no
    de lo que empieza y abandona.

    Las acciones que nunca han producido hastío no aparecen aquí: no ocupan
    nada."""

    performing: PerformingAction | None = None
    """`None` = ocioso, y el agente delibera en el tick actual."""

    strain: float = 0.0
    """Desgaste cognitivo en `[0, 1]`. DERIVADO de las necesidades, se
    recalcula cada tick (`refresh_state`). No es una emoción: es lo que hace
    que el mismo agente decida peor cuando está mal — vía un modificador de
    CÓMO se decide, no de QUÉ se quiere."""

    ledger: Ledger | None = None
    """Libro de calibración. `None` = apagado, que es la normalidad."""

    need_weights: dict[NeedId, float] = field(default_factory=dict)
    """Cuánto le importa a este agente cada eje. Derivado de rasgos y metas
    activas (`compose_need_weights`)."""

    gains: dict[NeedId, float] = field(default_factory=dict)
    """Aptitud del agente en cada eje. Derivado de rasgos (`compose_gains`)."""

    costs: dict[NeedId, float] = field(default_factory=dict)
    """Temple del agente en cada eje. Derivado de rasgos (`compose_costs`)."""

    caps: dict[NeedId, float] = field(default_factory=dict)
    """Techo efectivo en cada eje. Derivado de rasgos (`compose_caps`)."""

    temperament: Temperament | None = None
    """Configuración de decisión derivada de rasgos, desgaste y tuning
    (`compose_temperament`)."""

    gain_beliefs: dict[ActionId, dict[NeedId, float]] = field(default_factory=dict)
    """Multiplicadores de expectativa sobre lo que entregan las acciones,
    indexados por acción y necesidad. Mutado por `learn`. Arranca vacío; lo
    ausente representa 1.0 (quien no tiene experiencia previa se cree el pack)."""

    cost_beliefs: dict[ActionId, dict[NeedId, float]] = field(default_factory=dict)
    """Multiplicadores de expectativa sobre lo que cuestan las acciones,
    indexados por acción y necesidad. Mutado por `learn`. Arranca vacío; lo
    ausente representa 1.0."""

    # -- necesidades ----------------------------------------------------------

    def cap_of(self, need: NeedSpec | NeedId) -> float:
        """Devuelve el techo efectivo para un eje de necesidad.

        Puerta única para la lectura de techos en el agente. Si el agente ya
        ha recompuesto sus rasgos (`recompose`), lee el techo personalizado de
        `self.caps`. Si la necesidad aún no figura en `self.caps` (por ejemplo,
        un `AgentState` instanciado que no ha sido recompuesto todavía), recurre
        como fallback a `need.cap` si se proporcionó un `NeedSpec`, o a 1.0 por
        defecto.

        **Efecto secundario del fallback**: un `AgentState` sin recomponer
        utilizará el techo nominal del pack (`NeedSpec.cap` o 1.0), lo cual
        difiere del techo efectivo ajustado por rasgos si el agente posee
        modificadores de tope en sus rasgos. Recomponer al inicializar el
        agente garantiza coherencia operativa.
        """
        need_id = need.id if isinstance(need, NeedSpec) else need
        default = need.cap if isinstance(need, NeedSpec) else 1.0
        return self.caps.get(need_id, default)

    @staticmethod
    def _clamp(value: float, cap: float) -> float:
        """El ÚNICO sitio donde se fuerza el rango de un eje.

        Si el recorte se dispersa, un día un eje vale más que su techo y se
        pierden horas buscando quién lo dejó así. El suelo es 0.0 para
        cualquiera; el techo `cap` es el techo efectivo del agente provisto
        por `cap_of(need)` —que ya incorpora la modulación de personalidad
        derivada de los rasgos— o el nominal de `NeedSpec.cap` como fallback.
        """
        return max(0.0, min(cap, value))

    def _spend(self, amount: float, need: NeedSpec) -> None:
        """Baja un eje y anota como deuda lo que el suelo se habría comido.

        Las cuatro líneas que son la razón de ser de esta unidad: lo que el
        recorte se lleva no se pierde, se anota aparte, con tope
        `need.max_debt`. El eje sigue viviendo en `[0, cap]` — quien lea
        `needs` no se entera de que hubo un recorte más grande.
        """
        raw = self.needs[need.id] - amount
        if raw < 0.0 and need.max_debt > 0.0:
            self.debt[need.id] = min(need.max_debt, self.debt.get(need.id, 0.0) - raw)
        cap = self.cap_of(need)
        self.needs[need.id] = self._clamp(raw, cap)

    def restore(self, need_id: NeedId, amount: float, need: NeedSpec) -> float:
        """Repone, saldando la deuda primero. Devuelve lo que llegó al eje.

        Ese retorno importa: es la diferencia entre lo que se repuso y lo que
        se nota. Con deuda pendiente, parte de lo que llega se va en pagar lo
        debido y el eje sube menos de lo que la cantidad bruta sugiere — así
        es como un eje hundido mucho tiempo no se arregla con una reposición
        cualquiera.

        `amount` negativo revienta con `ValueError` en vez de tratarse como
        gasto. Dejarlo pasar rompía el tope `max_debt`: con deuda pendiente,
        `paid = min(owed, amount)` se vuelve negativo si `amount` lo es, y
        entonces `remaining = owed - paid` queda POR ENCIMA de `owed` en vez
        de por debajo — la deuda crece con una llamada que se supone que
        repone. Delegar en `_spend` en vez de rechazar se descartó a
        propósito: fundiría en un solo método dos verbos distintos —reponer
        y gastar—, y quien de verdad quiera bajar un eje ya tiene `_spend`,
        `apply` y `endure` para eso.
        """
        if amount < 0.0:
            raise ValueError(f"amount = {amount}, restore no admite cantidades negativas")
        owed = self.debt.get(need_id, 0.0)
        if owed > 0.0:
            paid = min(owed, amount)
            amount -= paid
            remaining = owed - paid
            if remaining <= 0.0:
                del self.debt[need_id]
            else:
                self.debt[need_id] = remaining
        if amount <= 0.0:
            return 0.0
        before = self.needs[need_id]
        cap = self.cap_of(need)
        self.needs[need_id] = self._clamp(before + amount, cap)
        return self.needs[need_id] - before

    def gain(self, need_id: NeedId, amount: float, need: NeedSpec) -> float:
        """`restore` anotando en el libro como lo que llega por actuar.

        Llama a `restore` antes de anotar, no al revés: si `amount` es
        negativo, `restore` revienta antes de que el ledger vea una entrada
        que de todas formas iba a fallar. Anotar primero dejaría el libro
        con una cifra que nunca llegó a materializarse.
        """
        delivered = self.restore(need_id, amount, need)
        if self.ledger is not None:
            self.ledger.record(self.ledger.active, need_id, amount)
        return delivered

    def deficit(self, need_id: NeedId) -> float:
        """Cuánto falta para estar entero, contando lo debido.

        Contra el entero universal (1.0), no contra el techo de este eje: la
        pregunta es cuánto se ha perdido de verdad, y lo que un recorte de
        techo impide alcanzar no es lo mismo que lo que falta por reponer.
        """
        return 1.0 - self.needs[need_id] + self.debt.get(need_id, 0.0)

    def decay(self, need_specs: Mapping[NeedId, NeedSpec]) -> None:
        """Baja cada eje según su propia tasa. Un eje de progresión, con tasa
        0 forzada por `NeedSpec` (ver `_progression_is_coherent`), no se
        mueve por aquí: solo sube por lo que se hace.

        Se recorre en orden de identificador. No hay acumulador compartido
        entre ejes distintos —cada uno gasta su propio flotante—, pero
        ordenar cuesta nada y evita que este archivo sea la excepción a la
        regla del resto del motor.
        """
        for need_id in sorted(need_specs):
            need = need_specs[need_id]
            if self.ledger is not None:
                self.ledger.record(self.ledger.demand, need_id, need.decay_per_tick)
            self._spend(need.decay_per_tick, need)

    def apply(
        self, effects: Mapping[NeedId, float], need_specs: Mapping[NeedId, NeedSpec]
    ) -> None:
        """Suma los deltas de algo que el agente acaba de completar.

        Antes de anotar, escala por aptitud (`self.gains`) lo que entra y
        por temple (`self.costs`) lo que sale — el mismo canal que compone
        `disposition.compose_gains` y `compose_costs`. Un agente que todavía
        no se recompuso tiene los dos diccionarios vacíos, así que el factor
        cae a 1.0 por defecto y no cambia nada: solo escala una vez que hay
        personalidad compuesta.

        Positivo entra por `gain` —lo que llega por actuar—; negativo por
        `_spend`, anotado como demanda. Ordenado por identificador: sumar en
        coma flotante no es asociativo, y sin un orden fijo dos ejecuciones
        con la misma semilla podrían divergir en el último bit.
        """
        for need_id in sorted(effects):
            delta = effects[need_id]
            need = need_specs[need_id]
            if delta >= 0.0:
                gained = delta * self.gains.get(need_id, 1.0)
                self.gain(need_id, gained, need)
            else:
                cost = -delta * self.costs.get(need_id, 1.0)
                if self.ledger is not None:
                    self.ledger.record(self.ledger.demand, need_id, cost)
                self._spend(cost, need)

    def endure(
        self, effects: Mapping[NeedId, float], need_specs: Mapping[NeedId, NeedSpec]
    ) -> None:
        """Lo que una situación activa le hace al agente por estar bajo su
        alcance, tick a tick. Deltas, como `apply`, pero lo positivo se anota
        como PASIVO y no como activo: llega por estar ahí, no por decidir
        nada, y confundir las dos columnas del libro es exactamente el error
        que `Ledger` existe para no cometer.

        Solo el lado negativo escala por temple (`self.costs`). La aptitud
        (`self.gains`) mide cuánto rinde lo que el agente hace por sí mismo;
        escalar con ella lo que le cae encima sin que decida nada mezclaría
        dos cosas distintas: lo que consigues actuando y lo que te impone el
        entorno.
        """
        for need_id in sorted(effects):
            delta = effects[need_id]
            need = need_specs[need_id]
            if delta >= 0.0:
                if self.ledger is not None:
                    self.ledger.record(self.ledger.passive, need_id, delta)
                self.restore(need_id, delta, need)
            else:
                cost = -delta * self.costs.get(need_id, 1.0)
                if self.ledger is not None:
                    self.ledger.record(self.ledger.demand, need_id, cost)
                self._spend(cost, need)

    def urgency(self, need: NeedSpec) -> float:
        """Cuánto apremia este eje ahora mismo, para este agente.

        Deliberadamente ciega a `debt`: dos agentes con el mismo valor de
        necesidad y distinta deuda dan la misma urgencia. Si la deuda se
        filtrara aquí, habría dos formas de expresar "esto va mal" —el valor
        bajo y la deuda alta— y ninguna se podría calibrar por separado.
        """
        cap = self.cap_of(need)
        return urgency(self.needs[need.id], need.urgency_exponent, cap)

    def refresh_state(
        self, need_specs: ContentPack | Mapping[NeedId, NeedSpec]
    ) -> None:
        """Recalcula el desgaste cognitivo (`strain`) a partir de las necesidades.

        Delega exclusivamente en `disposition.compute_strain`, constituyendo la
        única puerta canónica para el cálculo de desgaste en todo el motor.
        Pasa los techos efectivos del agente (`self.caps`) para que la urgencia
        se normalice respecto a la capacidad real de este agente en cada eje.
        """
        self.strain = compute_strain(need_specs, self.needs, self.caps)

    def advance_floor_streak(
        self,
        need_specs: Mapping[NeedId, NeedSpec],
        *,
        event_log: EventLog,
        owner: AgentId,
        tick: int,
    ) -> None:
        """Al final de un tick: qué ejes siguen en el suelo, cuáles acaban de
        levantar, y cuál acaba de cruzar su umbral de escalado.

        Sube el contador de quien sigue a cero, lo borra de golpe —no lo baja
        a plazos— de quien ha levantado, y emite `NEED_ESCALATED` exactamente
        en el tick en que un eje alcanza `NeedSpec.escalates_after`, nunca
        antes ni en los siguientes: es la transición la que se anuncia, no el
        estado sostenido.

        `escalates_after == 0` desactiva el mecanismo por completo para ese
        eje y no ocupa ni una entrada del diccionario.
        """
        for need_id in sorted(need_specs):
            need = need_specs[need_id]
            if need.escalates_after <= 0:
                continue
            if self.needs[need_id] > 0.0:
                self.floor_streak.pop(need_id, None)
                continue
            streak = self.floor_streak.get(need_id, 0) + 1
            self.floor_streak[need_id] = streak
            if streak == need.escalates_after:
                event_log.emit(
                    tick=tick,
                    kind=EventKind.NEED_ESCALATED,
                    owner=owner,
                    data={"need": need_id},
                )

    # -- hastío -----------------------------------------------------------

    def saturate(self, action: ActionId, gain: float) -> None:
        """Registra que el agente acaba de completar esto. `gain` <= 0 no
        deja rastro."""
        if gain <= 0.0:
            return
        self.saturation[action] = max(0.0, min(1.0, self.saturation.get(action, 0.0) + gain))

    def recover_novelty(self, rate: float) -> None:
        """Disipa el hastío, a la tasa que declara
        `TuningSpec.saturation_recovery`. Lo que llega a cero se olvida: el
        diccionario solo ocupa lo que el agente ha hecho últimamente, no todo
        lo que ha hecho alguna vez.

        Se recorre en orden de identificador. Hoy no hay acumulador
        compartido entre acciones que dependa de en qué orden se procesan,
        pero ordenar cuesta nada y evita que esta sea la única iteración sin
        ordenar de los cuatro módulos nuevos — la misma disciplina que
        `decay` ya aplica arriba.
        """
        if rate <= 0.0:
            return
        for action in sorted(self.saturation):
            remaining = self.saturation[action] - rate
            if remaining <= 0.0:
                del self.saturation[action]
            else:
                self.saturation[action] = remaining

    # -- metas ------------------------------------------------------------

    def pursuing(self) -> tuple[GoalId, ...]:
        """Las metas que empujan ahora mismo. Lo logrado y lo perdido dejan
        de empujar: si siguieran pesando después, serían una necesidad con
        otro nombre. Ordenado por identificador, aunque hoy nada en esta
        función multiplique flotantes: es el mismo criterio que el resto del
        motor aplica a cualquier `dict` que se recorre."""
        return tuple(
            sorted(
                goal_id for goal_id, status in self.goals.items() if status is GoalStatus.ACTIVE
            )
        )

    # -- recomposición y deriva --------------------------------------------

    def recompose(
        self,
        pack: ContentPack,
        active_situations: Iterable[SituationId] = (),
    ) -> None:
        """Rehace todo lo derivado de rasgos y estado actual. Puerta única.

        Recalcula los pesos de necesidades (incluyendo metas activas), aptitud
        (gains), temple (costs), techos (caps) y temperamento a partir de los
        rasgos actuales. Las creencias no se reinician: son acumuladas por
        experiencia.
        """
        active_goals = tuple(
            pack.goal_by_id[gid] for gid in self.pursuing() if gid in pack.goal_by_id
        )
        self.need_weights = compose_need_weights(pack, self.traits, active_goals)
        self.gains = compose_gains(pack, self.traits)
        self.costs = compose_costs(pack, self.traits)
        self.caps = compose_caps(pack, self.traits)
        self.temperament = compose_temperament(
            pack, self.traits, self.strain, active_situations=active_situations
        )

    def drift_trait(self, pack: ContentPack, trait: TraitId, delta: float) -> None:
        """Mueve un rasgo en [-1.0, 1.0] y vuelve a componer todo lo derivado.

        Guarda explícita: el rasgo debe existir obligatoriamente en el pack; si
        no figura en `pack.trait_by_id`, levanta `KeyError` con un mensaje
        propio que identifica el rasgo faltante antes de alterar el estado.
        Modificar el rasgo y recomponer en la misma llamada garantiza que ningún
        consumidor use derivados desfasados.
        """
        if trait not in pack.trait_by_id:
            raise KeyError(
                f"el rasgo {trait!r} no existe en el pack {pack.id!r}; "
                f"declarados: {sorted(pack.trait_by_id)}"
            )
        self.traits[trait] = max(-1.0, min(1.0, self.traits.get(trait, 0.0) + delta))
        self.recompose(pack)


# -----------------------------------------------------------------------------
# Estado de la ejecución.
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class ExecutionState:
    """Qué tick es, con qué semilla, qué está activo y quién decide.

    Nace de recortar lo que en la referencia era el estado de un mundo entero
    —lugares, facciones, propiedad, un censo de varios agentes— a lo que le
    queda cuando solo hay uno: no hay a quién indexar por sitio ni por bando,
    así que no queda nada de eso que conservar. "Ejecución" y no otra cosa,
    porque esto no describe un entorno: describe una corrida concreta del
    motor, con su tick y su semilla.
    """

    agent: AgentState

    tick: int = 0

    rng: RngSource = field(default_factory=lambda: RngSource(0))
    """La fuente de aleatoriedad vive aquí y se pasa explícitamente a quien la
    necesite — nunca un generador global compartido, que es donde muere la
    reproducibilidad."""

    situations: dict[SituationId, int | None] = field(default_factory=dict)
    """Las situaciones EVENTUALES encendidas -> hasta qué tick, o `None` si es
    hasta que algo las apague.

    Las periódicas no están aquí: son una función del tick
    (`SituationSpec.active_at`), así que guardarlas sería guardar algo que ya
    se sabe calcular."""

    @property
    def seed(self) -> int:
        """Se escribe al abrir el registro. Un fallo se reproduce corriendo
        la misma semilla; sin esto, un fallo raro no se investiga."""
        return self.rng.seed

    def active_situations(self, situations: Iterable[SituationSpec]) -> tuple[SituationId, ...]:
        """Lo que está activo ahora mismo, ordenado.

        Tupla ordenada y no `frozenset`: los multiplicadores de una situación
        se componen entre sí, la multiplicación de flotantes no es asociativa,
        y el orden de iteración de un conjunto depende del hash de sus
        elementos — inestable entre procesos con `PYTHONHASHSEED` distinto.
        """
        active = [
            spec.id
            for spec in situations
            if spec.active_at(self.tick) or spec.id in self.situations
        ]
        return tuple(sorted(active))
