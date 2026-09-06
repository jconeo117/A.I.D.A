"""Ontología invariante del motor.

Estas son las ÚNICAS primitivas que el motor conoce. Ninguna sabe en qué
organización corre ni qué tarea resuelve: solo sabe que existen "necesidades",
"metas", "rasgos" y "situaciones" con ciertos números. El dominio entra como
datos (pack), nunca como código.

Distinción central:
    Spec  -> definición inmutable que viene del pack. No cambia en runtime.
    State -> valores mutables de una ejecución concreta. Vive en `state.py`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, NewType

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

# ---------------------------------------------------------------------------
# Identificadores.
#
# `NewType` da a mypy la misma protección que un tipo fuerte: pasar un
# TraitId donde se espera un AgentId es un error de tipos, aunque en runtime
# ambos sean `str`. Coste cero en ejecución.
# ---------------------------------------------------------------------------
NeedId = NewType("NeedId", str)
TraitId = NewType("TraitId", str)
GoalId = NewType("GoalId", str)
PhaseId = NewType("PhaseId", str)
SituationId = NewType("SituationId", str)
ActionId = NewType("ActionId", str)
AgentId = NewType("AgentId", str)
Tag = NewType("Tag", str)

PositiveFactor = Annotated[float, Field(gt=0.0)]
"""Multiplicador estrictamente positivo.

Se usa donde un factor se eleva a una potencia. Un multiplicador de 0 o
negativo haría que `0 ** -1` reviente o que la utilidad cambie de signo, y
ninguna de las dos cosas tiene interpretación en el modelo.
"""

NonNegativeFactor = Annotated[float, Field(ge=0.0)]
"""Magnitud que nunca es negativa.

Se usa en los costes: un coste negativo sería un beneficio, y para eso ya
existe el canal de beneficio de cada pieza. Permitirlo dejaría dos formas de
expresar lo mismo y haría imposible leer una pieza del pack de un vistazo.
"""


class Spec(BaseModel):
    """Base de toda definición que viene del pack. Inmutable y estricta.

    `frozen=True` porque nada por debajo de la carga del pack puede mutar una
    definición: lo que cambia con el tiempo vive en el estado, nunca aquí.
    `extra="forbid"` porque un campo mal escrito en el pack tiene que hacer
    fallar la carga, no ignorarse en silencio y comportarse como si no
    existiera. Un pack malformado no arranca.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Necesidades
# ---------------------------------------------------------------------------
class NeedSpec(Spec):
    """Un eje de presión interna que decae con el tiempo.

    El motor no sabe qué mide este eje ni por qué importa. Solo sabe que baja
    solo y que ciertos efectos lo suben.
    """

    id: NeedId
    label: str
    """Nombre legible, solo para logs. El motor nunca lo interpreta."""

    decay_per_tick: float = Field(ge=0.0, le=1.0)
    """Cuánto baja por tick. Calibrar esto es trabajo central del pack.

    Un 0 describe un eje que no se pierde solo: solo cambia por lo que se
    hace. Ver `progression`.
    """

    progression: bool = False
    """Este eje solo sube: no es una carencia, es cuánto se ha avanzado.

    La diferencia con una carencia no es de grado, es de naturaleza: estar
    cerca de cero en un eje que se agota es una emergencia, y estar cerca de
    cero en un eje que se acumula es simplemente el punto de partida. Por eso
    hace falta el campo, en vez de deducirlo de `decay_per_tick == 0`.

    **Un eje de progresión no puede entrar en crisis.** Tratado como una
    carencia, cualquiera que empezara desde abajo estaría por debajo del
    umbral crítico durante toda su progresión y soltaría cualquier otra cosa
    para atenderlo — una emergencia que no existe, y que el contrato de salud
    del pack (`HealthSpec`) debería poder detectar y no puede si este campo
    falta.

    Lo que sí conserva es la urgencia: `(1 - valor) ** exponente` hace que
    avanzar apremie mientras se está lejos del tope y deje de importar cerca
    de él. Querer progresar no necesita maquinaria nueva — la fórmula ya sabe
    querer, esto solo le da un eje donde hacerlo.
    """

    urgent_when_low: bool = True
    """Si escasear esto es una emergencia que justifica soltarlo todo.

    Hay ejes que no son de progresión y a los que tampoco les pega el pánico:
    algo que se acumula y se gasta, donde no llevarlo encima es lo normal y
    no una crisis. Tratarlo como una carencia produce un agente que
    interrumpe cualquier cosa para reponerlo.

    La distinción es entre lo que se AGOTA y lo que se ACUMULA. Apagar este
    campo no toca la urgencia —sigue apeteciendo llenarlo, con la misma
    curva— solo le quita el poder de interrumpir.
    """

    initial: float = Field(default=1.0, ge=0.0, le=1.0)

    base_weight: float = Field(default=1.0, ge=0.0)
    """Cuánto importa este eje a un agente sin ningún rasgo que lo module.

    Sin esto todo eje partiría igualmente importante, y la única forma de
    decir "esto importa más que aquello" sería inventar un rasgo universal
    que todo agente tuviera igual — lo que convertiría un dato del PACK
    (cuánto pesa este eje en el mundo) en un dato de cada FICHA de
    personalidad. `TraitSpec.need_weight_multipliers` compone sobre este
    valor, nunca lo sustituye: la personalidad reescala una importancia que
    el pack ya fijó, no la crea desde cero.
    """

    strain_weight: float = Field(default=0.0, ge=0.0)
    """Cuánto contribuye este eje al desgaste cognitivo cuando escasea.

    El desgaste no es una emoción: es un modificador de CÓMO se decide, no de
    QUÉ se quiere —de eso ya se encarga la urgencia. Con desgaste alto se
    decide peor: más errático, y los costes pesan más.

    Qué desgasta es propio del pack. Un 0 significa que este eje puede estar
    a cero sin que la cabeza se resienta: aprieta, pero no vuelve torpe.
    """

    escalates_after: int = Field(default=0, ge=0)
    """Cuántos ticks seguidos con este eje en el suelo hacen falta para que
    la presión escale. **0 = nunca escala**, y es el defecto.

    Tocar fondo un instante y recuperarse no es lo mismo que llevar mucho
    tiempo raspando el suelo, y solo lo segundo justifica interrumpir de
    verdad. El contador vive en el estado de la ejecución y se reinicia de
    golpe en cuanto el eje se levanta —a diferencia de la deuda (`max_debt`),
    que se paga a plazos y no perdona de golpe.

    Cruzar el umbral no decide nada por sí mismo: **escala**. Convierte "este
    eje lleva tiempo hundido" en "hay que avisar ahora", sin que el motor
    sepa ni le importe qué significa avisar — eso lo decide quien consuma el
    evento, nunca esta pieza.

    En ticks y no en una unidad de tiempo civil: el reloj lo define el pack,
    y hablar de días obligaría a este módulo a saber cuántos ticks tiene uno.

    Determinista y sin sorteo: una regla que a veces escala a los tres ticks
    y a veces a los cinco no se puede aprender ni auditar.
    """

    max_debt: float = Field(default=0.0, ge=0.0)
    """Cuánto puede acumularse este eje POR DEBAJO del suelo. 0 lo desactiva.

    Sin esto, el motor no distingue estar mal desde hace un instante de
    estar mal desde hace mucho: el suelo recorta el decaimiento en los dos
    casos y el valor llega a 0.0 indistinguible en ambos. Con deuda, lo que
    el suelo recortaría se anota aparte, y **reponer paga la deuda antes de
    subir el valor**. El eje sigue viviendo en `[0, 1]`, así que la urgencia,
    el desgaste y los umbrales no se enteran de que la deuda existe: solo
    cambia cuánto tarda uno en recuperarse de verdad.

    El tope es obligatorio: sin él, una ejecución larga dejaría una deuda
    impagable. Es la perilla que dice cuánto castiga el pack quedarse mucho
    tiempo por debajo del suelo — un tope bajo perdona rápido, uno alto hace
    que arrastrar un déficit cueste caro durante mucho tiempo.
    """

    urgency_exponent: float = Field(default=2.0, gt=0.0)
    """Curva de urgencia: `urgencia = (1 - valor) ** exponente`.

    Exponente 1 = lineal: el eje empuja de forma constante mientras baja.
    Exponente 2 = irrelevante hasta que de pronto es crítico. Subir este
    número por eje concreto permite que un eje apriete de forma distinta a
    otro sin tocar el resto de la fórmula.

    En un eje de progresión conviene lo contrario que en una carencia: por
    debajo de 1 el impulso de seguir avanzando se mantiene alto hasta cerca
    del tope; por encima de 1 se apaga en cuanto se dan los primeros pasos.
    """

    cap: float = Field(default=1.0, gt=0.0, le=1.0)
    """Hasta dónde llega este eje para quien no tiene ninguna ventaja. 1.0 =
    sin techo.

    Existe porque un ritmo no es un techo: con tiempo suficiente, cualquier
    ritmo por bajo que sea termina por llevar a todo el mundo hasta el mismo
    punto, y entonces la diferencia entre quién llega y quién no deja de
    decir nada. Un techo, en cambio, es estable en el tiempo: quién llega
    adónde deja de ser cuestión de paciencia y pasa a ser cuestión de con qué
    se empieza.

    Se declara bajo y son los rasgos quienes lo suben
    (`TraitSpec.cap_multipliers`), nunca al revés: así el sistema dice que
    cualquiera puede progresar hasta cierto punto, y solo con la ventaja
    adecuada se pasa de ahí.
    """

    @property
    def can_be_critical(self) -> bool:
        """Un eje de progresión nunca lo es, digan lo que digan sus otros campos."""
        return self.urgent_when_low and not self.progression

    @model_validator(mode="after")
    def _progression_is_coherent(self) -> NeedSpec:
        """Un eje de progresión no admite las mecánicas de la carencia.

        Ninguna de las tres significa nada aquí, y permitirlas en silencio
        dejaría packs con campos que parecen hacer algo y no lo hacen: no se
        "pierde" progreso con el tiempo, no hay deuda que pagar en algo que
        nunca tiene suelo, y estar lejos del tope no nubla el juicio de
        nadie.
        """
        if not self.progression:
            return self
        vetados = {
            "decay_per_tick": self.decay_per_tick,
            "max_debt": self.max_debt,
            "strain_weight": self.strain_weight,
        }
        puestos = sorted(name for name, value in vetados.items() if value)
        if puestos:
            raise ValueError(
                f"eje de progresión {self.id!r}: {', '.join(puestos)} no significa "
                f"nada en algo que solo sube"
            )
        return self


# ---------------------------------------------------------------------------
# Temperamento
# ---------------------------------------------------------------------------
class TemperamentKnob(StrEnum):
    """Aspectos de CÓMO se decide que un rasgo puede modular.

    Conjunto cerrado a propósito: a diferencia de las necesidades, esto no lo
    define el pack sino el motor. Son los seis puntos donde la maquinaria de
    decisión admite variación individual.

    Descartados los cuatro que necesitan una capa que AIDA no tiene: un
    asistente no tiene geografía que atravesar (`travel_discount`) ni una
    población de la que dejarse atraer o de la que hacerse amigo
    (`companionship_weight`, `bond_rate`, `bond_fade`) — modularían un
    mecanismo que no existe. Si algún día se mide que cambiar de tarea tiene
    un coste real, ya existe `SWITCH_COST` más abajo: es justo esa perilla, no
    una prestada que finja serlo.

    El pack fija el valor del mundo (`TuningSpec`) y el agente lo modula: son
    dos cosas que se COMPONEN, no que se sustituyen. Conservar el valor en
    `TuningSpec` pero quitar la perilla individual convertiría una
    personalidad en una constante global, y una constante global no es una
    personalidad — es justo lo contrario de personalidad como parámetro
    modulable, no como *system prompt*, que es el diferenciador que defiende
    el README en su §7.
    """

    TEMPERATURE = "temperature"
    """Cuánto azar frente a criterio en el sorteo final. Alto: errático. Bajo: metódico."""

    LOSS_AVERSION = "loss_aversion"
    """Cuánto pesa el coste. Alto: calculador. Bajo: vive el momento."""

    CRITICAL_THRESHOLD = "critical_need_threshold"
    """Cuándo cunde la alarma. Alto: se alarma pronto. Bajo: aguanta."""

    SWITCH_COST = "switch_cost"
    """Cuánto protege lo ya invertido. Alto: termina lo que empieza; bajo: lo
    suelta en cuanto aparece algo mejor.

    Es la perilla individual del compromiso con excepción de urgencia: el
    coste de cambio de contexto que el proyecto necesita, con la misma
    semántica que `TuningSpec.switch_cost` fija para el mundo entero.
    """

    NOVELTY = "novelty_weight"
    """Cuánto castiga el hastío. Alto: necesita variedad y se cansa de lo
    repetido; bajo: hace lo mismo cada día sin aburrirse.

    Es lo que evita repetir el mismo aviso indefinidamente.
    """

    LEARNING_RATE = "learning_rate"
    """Cuánto corrige sus expectativas tras cada experiencia. Alto: se fía de
    lo último que le pasó; bajo: dogmático, tarda en aceptar que algo cambió.

    Es el canal individual de las creencias aprendidas que llegan en U2.
    """


# ---------------------------------------------------------------------------
# Rasgos: multiplicadores estables de personalidad
# ---------------------------------------------------------------------------
class TraitSpec(Spec):
    """Un rasgo no "hace" nada: reescala el peso o el rendimiento de ciertos ejes.

    Ahí está la variación de carácter: dos agentes con reglas idénticas y
    pesos distintos deciden distinto sin una sola línea de código específica
    para cada uno.

    Diferido a U9, no descartado: `memory_multipliers` y `regret_multiplier`
    de Agora —cuánto le dura a este agente el recuerdo y el rencor en
    particular— vuelven junto con la memoria episódica, como canales de
    personalidad sobre lo que se recuerda. No tienen consumidor hasta
    entonces.
    """

    id: TraitId
    label: str

    need_weight_multipliers: dict[NeedId, PositiveFactor] = Field(default_factory=dict)
    """Multiplicador a intensidad plena (rasgo = 1.0). Un 1.0 aquí = sin efecto.

    La intensidad del agente (-1..1) se aplica como EXPONENTE, no como
    interpolación lineal:

        factor = multiplicador ** intensidad

    Así el 0 siempre neutraliza (M**0 == 1) y el -1 invierte exactamente
    (M**-1 == 1/M). Con interpolación lineal, un multiplicador de 2.2 y una
    intensidad de -1 daría un factor negativo, que no significa nada.
    """

    gain_multipliers: dict[NeedId, PositiveFactor] = Field(default_factory=dict)
    """Cuánto le cunde a este agente lo que hace, por eje.

    Escala lo que un eje GANA, no lo que le importa:

        need_weight_multipliers  ->  cuánto le IMPORTA
        gain_multipliers         ->  cuánto le CUNDE

    Son cosas distintas y su combinación describe agentes concretos. Sin
    esto, todo agente progresa igual de rápido en todos los ejes, y "se le da
    bien esto" solo puede significar "le importa más", que es la mitad de la
    idea.

    Se compone como los pesos —el multiplicador elevado a la intensidad— y
    multiplica cualquier efecto POSITIVO sobre ese eje. Sigue siendo un
    RITMO, no un tope: modela recuperarse rápido, no tener un límite bajo.
    """

    cost_multipliers: dict[NeedId, PositiveFactor] = Field(default_factory=dict)
    """Cuánto le cuesta a este agente lo que gasta, por eje. Espejo exacto de
    `gain_multipliers`, sobre el otro lado del intercambio:

        need_weight_multipliers  ->  cuánto le IMPORTA
        gain_multipliers         ->  cuánto le CUNDE lo que gana
        cost_multipliers         ->  cuánto le DUELE lo que gasta

    Sin este canal, "esto le cuesta menos a quien tiene más reserva" solo se
    puede fingir subiendo lo que gana, que no es lo mismo y se nota en cuanto
    se mira la contabilidad.

    Se compone igual que los demás —el multiplicador elevado a la
    intensidad— y multiplica cualquier efecto NEGATIVO sobre ese eje: lo que
    cuesta una acción y lo que le hace una situación. No toca el decaimiento,
    que tiene su propio canal en `gain_multipliers`.
    """

    cap_multipliers: dict[NeedId, PositiveFactor] = Field(default_factory=dict)
    """Hasta dónde puede llegar este agente en un eje, por encima del `cap`
    que declara la necesidad:

        need_weight_multipliers  ->  cuánto le IMPORTA
        gain_multipliers         ->  cuánto le CUNDE lo que gana
        cost_multipliers         ->  cuánto le DUELE lo que gasta
        cap_multipliers          ->  hasta DÓNDE puede llegar

    Multiplica el `cap` de la necesidad, y el resultado se recorta a 1.0 —
    nadie pasa del techo del mundo por mucho que el rasgo empuje.

    **Sube un techo bajo; no baja uno alto**, y esa dirección es la que hace
    que el sistema diga lo que quiere decir: la necesidad declara lo que
    alcanza cualquiera con constancia, y el rasgo dice quién llega más lejos.
    Al revés —techo alto que los rasgos bajan— habría que enumerar en cada
    agente todo lo que NO se le da bien, que es infinito.
    """

    phase_shift: int = Field(default=0)
    """Cuánto se desplaza la franja subjetiva de este agente:

        franja_subjetiva = franja_de(tick + phase_shift * intensidad)

    No es un multiplicador, y la diferencia importa. Multiplicar amplifica
    cuando debería invertir: si una franja favorece una acción y el rasgo
    multiplica lo favorecido, el efecto se acentúa siempre en la misma
    dirección en vez de desplazarse. Rotando la franja percibida, todo lo
    demás cae solo: el agente rinde en su propia franja subjetiva aunque el
    reloj objetivo diga otra cosa.

    Lo que NO se desplaza son los horarios que declara el pack — esos son
    objetivos y son los mismos para todos; lo que se desplaza es cómo cada
    agente los vive.
    """

    temperament_multipliers: dict[TemperamentKnob, PositiveFactor] = Field(
        default_factory=dict
    )
    """Segundo canal del rasgo: en vez de QUÉ le importa al agente, cambia
    CÓMO decide.

    Se compone igual que los pesos —el multiplicador elevado a la
    intensidad— sobre el valor que el pack fijó para el mundo entero. Un
    rasgo puede usar los dos canales, uno, o ninguno.
    """


# ---------------------------------------------------------------------------
# Gustos y agentes
# ---------------------------------------------------------------------------
class TasteSpec(Spec):
    """Preferencia hacia un objetivo específico, no hacia una categoría.

    Un rasgo sesga categorías enteras de necesidad; un gusto sesga una
    instancia concreta. Sin esta capa, todo agente con el mismo rasgo se
    comporta exactamente igual frente a cualquier opción de esa categoría.
    """

    target: ActionId | AgentId
    multiplier: float = Field(gt=0.0)
    """<1 aversión, >1 predilección. Multiplica la utilidad final entera, ya
    compuesta, no un término aislado.

    A diferencia de `TraitSpec`, un gusto no tiene un dial de intensidad que
    elevar como exponente: es una asociación directa a UN objetivo concreto,
    no una categoría que varía en grado. Por eso la cota es `gt=0.0` y no un
    rango simétrico alrededor de 1 — un multiplicador negativo invertiría el
    signo de una utilidad que ya puede ser negativa por sí misma (coste mayor
    que beneficio), y esa inversión no tiene lectura: no es "le disgusta más",
    es un signo que ya no significa lo que decía significar.
    """


class GoalStatus(StrEnum):
    """En qué anda una meta. Cuatro estados y no hay más.

    Lo logrado y lo perdido son distintos aunque los dos dejen de empujar:
    mueven rasgos opuestos, y no se puede tratar igual a quien lo consiguió
    que a quien se quedó a las puertas.
    """

    LOCKED = "locked"
    """Todavía no le toca: espera a que se cumpla la que la abre.

    Es lo que convierte metas sueltas en una secuencia con orden. Sin este
    estado, todas las metas de una ambición larga estarían abiertas a la
    vez, empujarían sus ejes desde el primer día y se cerrarían en cualquier
    orden.

    No empuja nada mientras dura: una meta bloqueada no toca ningún peso.
    """

    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"


class GoalSpec(Spec):
    """Algo que este agente quiere y todavía no tiene.

    Una necesidad y una meta no son lo mismo: la necesidad se agota y se
    mantiene, la meta se cumple UNA VEZ. No necesita maquinaria nueva en la
    fórmula de decisión: `need_weight_multipliers` ya es el dial que dice
    cuánto le importa algo a alguien; una meta es un empujón temporal sobre
    ese dial que se apaga al lograrse.

    La regla al escribirlas: una meta sin obstáculo es una rutina. Si el
    umbral está al alcance de lo que ya se hace cada tick, no se ha escrito
    una meta.
    """

    id: GoalId
    label: str

    need: NeedId
    """Qué eje mide el logro. Es la única forma de medirlo hoy: lo que se
    quiere de verdad casi siempre se cifra en algún eje. Lo que no cubre
    todavía es querer algo de otro agente, y eso queda para cuando el entorno
    se modele con las mismas primitivas.
    """

    at_least: float | None = Field(default=None, gt=0.0, le=1.0)
    """Umbral sobre el valor del eje que cuenta como logrado.

    Es un MOMENTO, no una tendencia sostenida, y hay que escribirlo
    sabiéndolo: un eje está hecho para llenarse, así que sobre suficientes
    ticks cualquiera que trabaje ese eje lo toca alguna vez. No se arregla
    subiendo el número —cerca del máximo alcanzable deja de ser un logro y
    pasa a ser una lotería—, salvo que el eje sea de `progression`
    (`NeedSpec.progression`): ese no retrocede solo, así que un `at_least`
    alto sobre él sí describe un objetivo sostenido, sin maquinaria aparte.
    """

    at_traits: dict[TraitId, float] = Field(default_factory=dict)
    """Ser quien hay que ser: cada rasgo, al menos a esta intensidad.

    Es una conjunción: hacen falta todos. Solo es alcanzable si algo mueve
    ese rasgo, y hoy lo único que lo hace es el `on_success`/`on_failure` de
    otra meta — que es exactamente el caso de uso: una meta concede el rasgo
    que abre la siguiente.
    """

    requires_goal: GoalId | None = None
    """No empieza hasta que esta otra se haya LOGRADO. Es lo que ordena
    varias metas en una secuencia con sentido.

    Mientras espera, la meta está en `LOCKED`: no empuja ningún peso y no
    puede fallar — no se puede perder lo que aún no había empezado. Si la que
    la abre se pierde, esta se queda bloqueada para siempre; es correcto: no
    se llega a lo que dependía de un paso anterior que no se dio.
    """

    @model_validator(mode="after")
    def _one_way_to_measure_it(self) -> GoalSpec:
        """Exactamente una condición. Ni cero ni dos.

        Con dos, habría que decidir si se cumple con una o con ambas, y esa
        respuesta sería una convención arbitraria que nadie recuerda al leer
        el pack. Con ninguna, la meta no se puede cerrar jamás.
        """
        cuantas = (self.at_least is not None) + bool(self.at_traits)
        if cuantas != 1:
            raise ValueError(
                f"meta {self.id!r}: hace falta EXACTAMENTE una condición de "
                f"`at_least` (el valor del eje) o `at_traits` (ser quien hay "
                f"que ser), y hay {cuantas}"
            )
        if self.requires_goal == self.id:
            raise ValueError(f"meta {self.id!r}: se abre a sí misma")
        return self

    weight_multiplier: PositiveFactor = 2.0
    """Cuánto empuja mientras dura, sobre el peso que ya tuviera ese eje.

    Se compone como los rasgos, multiplicando: una meta sobre un eje que ya
    importaba aprieta más que sobre uno que daba igual.
    """

    deadline: int | None = Field(default=None, ge=1)
    """Tick absoluto en que se da por perdida. `None` = sin plazo.

    No es adorno: una meta cuyo umbral nunca se alcanza deja al agente
    permanentemente escorado hacia ese eje sin que nada lo resuelva. El plazo
    convierte un empujón indefinido en algo con final, aunque el final sea el
    malo.
    """

    on_success: dict[TraitId, float] = Field(default_factory=dict)
    """Qué le mueve conseguirlo, rasgo a rasgo. Delta en [-2, 2].

    Es lo que separa lograr algo de que no cambie nada: sin esto, el agente
    llega a la meta y sigue siendo exactamente quien era.
    """

    on_failure: dict[TraitId, float] = Field(default_factory=dict)
    """Y qué le mueve perderlo."""

    @model_validator(mode="after")
    def _drifts_are_sane(self) -> GoalSpec:
        """Una deriva mayor que el rango entero del rasgo no significa nada.

        Los rasgos viven en [-1, 1], así que un delta de 2 ya lleva de un
        extremo al otro. Más que eso es un error de calibración pasando por
        intensidad dramática.
        """
        for campo, derivas in (("on_success", self.on_success), ("on_failure", self.on_failure)):
            for trait, delta in sorted(derivas.items()):
                if not -2.0 <= delta <= 2.0:
                    raise ValueError(
                        f"meta {self.id!r}.{campo}[{trait!r}] = {delta}, fuera de [-2, 2]; "
                        f"los rasgos viven en [-1, 1] y un 2 ya cruza el rango entero"
                    )
        return self


class AgentSpec(Spec):
    """Quién es este agente: sus metas, sus rasgos y sus gustos.

    Reducido a propósito. AIDA modela un único agente; cuando el entorno se
    represente con las mismas primitivas —para predecir rutinas en vez de
    reaccionar a ellas— esta pieza crece, y no antes.
    """

    id: AgentId
    label: str

    goals: tuple[GoalSpec, ...] = ()
    """Lo que este agente quiere. Vacío es legítimo."""

    traits: dict[TraitId, float] = Field(default_factory=dict)
    """Qué rasgos tiene ESTE agente y a qué intensidad — no qué hace cada
    rasgo, eso vive en `TraitSpec`, compartido por todo el catálogo del pack.

    Separar la definición del efecto (una vez, en `TraitSpec`) de la ficha de
    intensidades (una por agente, aquí) es lo que permite que cien agentes
    compartan el mismo catálogo de rasgos y decidan distinto sin que cada uno
    repita `need_weight_multipliers` y compañía. Un rasgo ausente de este
    `dict` no es un 0.0 explícito: simplemente no participa en la composición
    de este agente en concreto.
    """

    tastes: tuple[TasteSpec, ...] = ()

    initial_needs: dict[NeedId, float] = Field(default_factory=dict)
    """Override del `initial` de cada `NeedSpec`. Lo que falte usa el
    default declarado en la necesidad.
    """

    @model_validator(mode="after")
    def _traits_in_range(self) -> AgentSpec:
        """Fuera de [-1, 1] la intensidad no significa nada del modelo.

        `TraitSpec` compone sus multiplicadores como `M ** intensidad`: el
        rango es lo que garantiza que 0 neutralice exactamente (`M**0 == 1`) y
        que -1 invierta exactamente (`M**-1 == 1/M`). La potencia en sí no
        revienta con un valor fuera de ese rango, así que sin esta puerta un
        pack podría colar un 3.0 que compila y produce un número que no es
        "invertido" ni "neutral" ni "a tope": solo un artefacto de la fórmula.
        """
        for trait, value in sorted(self.traits.items()):
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"rasgo {trait!r} = {value}, fuera de [-1, 1]")
        return self


# ---------------------------------------------------------------------------
# Franjas
# ---------------------------------------------------------------------------
class PhaseSpec(Spec):
    """Un tramo con nombre dentro del ciclo.

    El motor no sabe cuántos tramos tiene un ciclo ni cómo se llaman: solo
    sabe que el ciclo se parte en tramos y que las cosas rinden distinto en
    cada uno. Cuántos tramos haya y cómo se llamen es contenido del pack.
    """

    id: PhaseId
    label: str
    starts_at: int = Field(ge=0)
    """Tick del ciclo en que empieza. El tramo dura hasta que empieza el
    siguiente.

    No hace falta que ninguno empiece en 0: lo anterior al primer tramo
    pertenece al ÚLTIMO, que es como se modela un tramo que cruza el punto de
    vuelta del ciclo sin partirlo en dos.
    """


# ---------------------------------------------------------------------------
# Situaciones
# ---------------------------------------------------------------------------
class SituationSpec(Spec):
    """Un estado con nombre que va y viene y cambia lo que rinde y lo que
    apetece hacer.

    Una franja es una función del tick: cíclica, siempre hay exactamente una,
    y no se puede empezar ni terminar por decisión de nadie. Una situación es
    lo mismo un escalón más arriba: tiene principio y final, puede solaparse
    con otra, y algunas se sabe cuándo llegan y otras no.

    Dos formas de estar activa, y hacen falta las dos:

    - **Periódica** (`every` + `lasts`): función del tick, igual que una
      franja. No guarda estado, así que sale gratis y no necesita marcarse
      cuando empieza o termina.
    - **Eventual** (ni `every` ni `lasts`): solo está activa si algo externo
      la enciende. Esa sí es estado que se guarda y que emite un evento al
      empezar y al terminar — qué la enciende es una decisión de la cola de
      eventos, no de esta pieza.

    El motor no sabe qué representa una situación concreta: solo sabe que
    multiplica lo que rinde una acción, lo atractiva que resulta y cómo
    decide quien está bajo su efecto.
    """

    id: SituationId
    label: str

    every: int = Field(default=0, ge=0)
    """Cada cuántos ticks vuelve. 0 = no es periódica, algo tiene que
    encenderla."""

    lasts: int = Field(default=0, ge=0)
    """Cuánto dura cada vez que vuelve, en ticks. Solo cuenta si `every` > 0."""

    starts_at: int = Field(default=0, ge=0)
    """En qué punto del ciclo largo empieza, para que no caiga siempre justo
    al principio de la cuenta."""

    gain_multipliers: dict[NeedId, PositiveFactor] = Field(default_factory=dict)
    """Cuánto cunde una ACCIÓN mientras esto está activo. Lo ausente vale 1.0.

    Distinto de `effects`: eso drena o repone de forma pasiva, tick a tick,
    pase lo que pase. Esto solo se nota en lo que el agente hace activamente
    mientras dura la situación, y compone con `TraitSpec.gain_multipliers`
    —situación por rasgo, no uno sustituye al otro. Sin este canal, la única
    forma de que un contexto favorezca hacer algo sería subir el beneficio
    declarado de cada acción del catálogo, lo que obligaría a cada acción a
    conocer de antemano cada situación posible en vez de que la situación
    describa su propio efecto una sola vez.
    """

    effects: dict[NeedId, float] = Field(default_factory=dict)
    """Lo que le hace, cada tick, a quien esté bajo su alcance.

    Deltas por tick: negativo quita, positivo da. Van por tick y no de una
    vez al empezar, porque una situación que castiga mientras dura hace que
    salir de ella importe; aplicar el efecto de golpe al encender premiaría
    llegar tarde.

    Ojo a la escala: se multiplican por la duración. Un delta pequeño durante
    muchos ticks puede sumar más que uno grande aplicado una sola vez.
    """

    temperament_multipliers: dict[TemperamentKnob, PositiveFactor] = Field(
        default_factory=dict
    )
    """Cómo decide un agente mientras esto está activo.

    Hasta aquí una situación solo sabía drenar ejes con `effects`. Esto le
    añade la otra mitad: no solo cambia CUÁNTO cuesta algo, cambia CÓMO se
    decide — el mismo estado que aprieta un eje puede además volver a
    alguien más metódico o más propenso a alarmarse.

    Su consumidor llega en U3/U4; se declara ahora porque es la costura entre
    situación y disposición, y las costuras se declaran pronto.
    """

    @model_validator(mode="after")
    def _periodic_situations_actually_happen(self) -> SituationSpec:
        if self.every and not self.lasts:
            raise ValueError(
                f"situación {self.id!r}: vuelve cada {self.every} ticks pero dura 0, "
                f"así que no llega a ocurrir nunca"
            )
        if self.every and self.lasts >= self.every:
            raise ValueError(
                f"situación {self.id!r}: dura {self.lasts} y vuelve cada {self.every}, "
                f"así que no se apaga jamás; déjala eventual si es permanente"
            )
        if self.lasts and not self.every:
            raise ValueError(
                f"situación {self.id!r}: declara `lasts` sin `every`. Una situación "
                f"eventual dura lo que se le diga al encenderla"
            )
        return self

    @property
    def periodic(self) -> bool:
        return self.every > 0

    def active_at(self, tick: int) -> bool:
        """Si es periódica, si toca ahora. Una eventual nunca lo está por sí
        sola: necesita que algo externo la encienda y la apague, y eso vive
        en el estado de la ejecución, no aquí.
        """
        if not self.periodic:
            return False
        return (tick - self.starts_at) % self.every < self.lasts


# ---------------------------------------------------------------------------
# Acciones
# ---------------------------------------------------------------------------
class Reversibility(StrEnum):
    """Si lo que hace una acción se puede deshacer, y a qué precio."""

    REVERSIBLE = "reversible"
    REVERSIBLE_WITH_COST = "reversible_with_cost"
    IRREVERSIBLE = "irreversible"


class ActionSpec(Spec):
    """Entrada del catálogo cerrado de acciones del asistente.

    Define una operación posible con su latencia prevista, sus requisitos
    de ejecutabilidad, su impacto anticipado al deliberar y su efecto real al
    completarse.

    `satisfies` y `effects` parecen redundantes pero no lo son:
      - satisfies -> se usa ANTES, para puntuar la acción al deliberar.
      - effects   -> se usa DESPUÉS, para aplicar el resultado real.
    Separarlos permite que el agente anticipe un alivio que difiera del efecto
    real, lo que hace posible modelar error de predicción y aprendizaje.
    """

    id: ActionId
    label: str
    reversibility: Reversibility
    """Obligatorio y sin valor por defecto: una acción sin clasificar no
    carga. Un defecto convertiría el olvido en la clasificación más
    permisiva, que es justo la que no se debe poder elegir por descuido.
    """

    duration_ticks: int = Field(ge=1)
    """Latencia esperada en ticks para completar la acción."""

    interruptible: bool = True
    """Si una urgencia crítica puede abortarla a medias. Si es False, la
    acción se completa íntegramente o no se empieza.
    """

    requires: dict[NeedId, NonNegativeFactor] = Field(default_factory=dict)
    """Nivel mínimo por eje de necesidad para poder si quiera intentarla.
    Puerta dura: evita intentar acciones que no se pueden costear.
    """

    requires_traits: dict[TraitId, float] = Field(default_factory=dict)
    """Intensidad mínima de rasgo requerida, acotada a [-1, 1]."""

    satisfies: dict[NeedId, float] = Field(default_factory=dict)
    """Cuánto anticipa aliviar cada necesidad (evaluado al deliberar)."""

    costs: dict[NeedId, NonNegativeFactor] = Field(default_factory=dict)
    """Cuánto anticipa gastar en cada necesidad (evaluado al deliberar)."""

    effects: dict[NeedId, float] = Field(default_factory=dict)
    """Delta real aplicado al estado tras completar la acción."""

    phase_multipliers: dict[PhaseId, PositiveFactor] = Field(default_factory=dict)
    """Multiplicador de beneficio según el tramo del ciclo. Lo ausente vale 1.0."""

    situation_multipliers: dict[SituationId, PositiveFactor] = Field(default_factory=dict)
    """Multiplicador de beneficio según el contexto activo. Lo ausente vale 1.0."""

    saturation_gain: float = Field(default=0.0, ge=0.0)
    """Hastío acumulado por repetición continua de esta acción."""

    tags: frozenset[Tag] = Field(default_factory=frozenset)
    """Etiquetas libres para agrupación o clasificación."""

    @field_serializer("tags", when_used="json")
    def _tags_sorted(self, value: frozenset[Tag]) -> list[Tag]:
        """Fija el orden de volcado a JSON para que sea determinista entre procesos."""
        return sorted(value)

    @model_validator(mode="after")
    def _validate_coherence(self) -> ActionSpec:
        """Comprueba coherencia de cotas en rasgos y de signos entre promesas y efectos.

        - Toda clave en `requires_traits` debe estar acotada a [-1, 1].
        - Toda clave en `satisfies` debe aparecer en `effects` con signo positivo (> 0.0).
        - Toda clave en `costs` debe aparecer en `effects` con signo negativo (< 0.0).
        """
        for trait, intensity in sorted(self.requires_traits.items()):
            if not -1.0 <= intensity <= 1.0:
                raise ValueError(
                    f"acción {self.id!r}.requires_traits[{trait!r}] = {intensity}, fuera de [-1, 1]"
                )
        for need in sorted(self.satisfies):
            effect = self.effects.get(need)
            if effect is None or effect <= 0.0:
                raise ValueError(
                    f"acción {self.id!r}: la clave {need!r} de `satisfies` debe aparecer "
                    f"en `effects` con signo positivo (> 0.0), pero vale {effect!r}"
                )
        for need in sorted(self.costs):
            effect = self.effects.get(need)
            if effect is None or effect >= 0.0:
                raise ValueError(
                    f"acción {self.id!r}: la clave {need!r} de `costs` debe aparecer "
                    f"en `effects` con signo negativo (< 0.0), pero vale {effect!r}"
                )
        return self


# ---------------------------------------------------------------------------
# Salud
# ---------------------------------------------------------------------------
class HealthSpec(Spec):
    """Qué clase de comportamiento quiere tener este pack.

    No son parámetros de simulación: son intención de diseño verificable.
    Una afirmación como "bajo esta carga, el asistente decide con presión
    moderada, ni indiferente ni desbordado" es comprobable, y conviene que
    falle ruidosamente en cuanto deja de ser cierta.

    Están aquí y no en un test porque un test con las bandas clavadas
    pelearía contra la calibración: cambiar un `decay_per_tick` rompería el
    test aunque el cambio fuera bueno. Y cada pack quiere cosas distintas —
    un entorno tranquilo y uno de vigilancia continua no comparten umbral.

    Todo es opcional. Un pack sin `health` solo se somete a los invariantes
    universales: que ningún valor se salga de rango.
    """

    days: int = Field(default=7, ge=1)
    seeds: tuple[int, ...] = (1312, 7, 99)
    """Varias semillas porque una corrida es una muestra, no un resultado."""

    min_crisis: float | None = Field(default=None, ge=0.0, le=1.0)
    """Suelo de presión. Un pack donde nada aprieta nunca es un pack donde el
    carácter no se nota: con todo resuelto, la personalidad deja de influir
    en la decisión.
    """

    max_crisis: float | None = Field(default=None, ge=0.0, le=1.0)
    """Techo de presión. Por encima, la emergencia aplasta al carácter."""

    max_abort_ratio: float | None = Field(default=None, ge=0.0)
    """Abandonos frente a acciones completadas."""

    min_differentiation: float | None = Field(default=None, ge=0.0)
    """Distancia media entre las trayectorias que produce el mismo pack con
    distinta semilla o distinta ficha de personalidad. Es la métrica que caza
    el colapso: configuraciones de carácter distintas que producen la misma
    conducta. Poder demostrarlo en integración continua —que dos
    personalidades deciden distinto sobre el mismo estado, y que el azar no
    se ha comido al carácter— es exactamente lo que defiende el README en su
    §7.
    """

    may_go_unused: frozenset[ActionId] = Field(default_factory=frozenset)
    """Acciones que pueden no aparecer nunca sin que sea un fallo.

    Una red de seguridad es el caso típico: en una ejecución sana nadie
    debería necesitar recurrir a ella. Cualquier OTRA acción sin usar sí es
    un aviso — es contenido declarado que nunca se ejerce.

    Es un `frozenset` y no un `tuple` a propósito: el orden en que el pack
    escribe estas acciones no debería ser parte de la identidad del modelo.
    Con `tuple`, dos `HealthSpec` con las mismas acciones en distinto orden
    dejarían de ser iguales para pydantic, lo que convertiría un detalle de
    redacción del pack en una diferencia semántica que no lo es. Lo único que
    un `frozenset` no da gratis es un orden de iteración estable entre
    procesos —depende de `PYTHONHASHSEED`—, así que la serialización se fija
    aparte en `_may_go_unused_sorted`, sin renunciar a la semántica de
    conjunto del campo.
    """

    @field_serializer("may_go_unused", when_used="json")
    def _may_go_unused_sorted(self, value: frozenset[ActionId]) -> list[ActionId]:
        """Fija el orden de volcado a JSON para que sea igual entre procesos.

        El orden de iteración de un `frozenset` depende de `PYTHONHASHSEED`:
        sin esto, el mismo `HealthSpec` produce un JSON distinto en cada
        proceso aunque el contenido no haya cambiado, lo que rompe G6 en
        cuanto exista un consumidor que registre esta pieza en el log o en un
        pack cargado. El campo en sí sigue siendo un `frozenset` en memoria;
        solo se ordena al volcarlo.
        """
        return sorted(value)


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------
class TuningSpec(Spec):
    """Las perillas que gobiernan CÓMO se decide, no QUÉ existe.

    Viven en el pack y no en el código a propósito: calibrar es un ciclo de
    iteración sobre datos, y una constante enterrada en un módulo de Python
    es una perilla que no se puede girar sin desplegar.
    """

    candidate_pool: int = Field(default=3, ge=1)
    """Cuántas de las mejores opciones entran en el sorteo final.

    1 equivale a elegir siempre la de mayor utilidad (determinista y
    predecible). Subirlo aumenta la variedad, pero también la probabilidad
    de elegir algo claramente peor.
    """

    temperature: float = Field(default=0.15, gt=0.0)
    """Cuánto pesa el azar frente a la utilidad dentro del grupo de
    finalistas.

    Cerca de 0, casi siempre gana la mejor opción. Alta, casi indiferente
    entre las finalistas. Es la perilla que separa "decide con criterio" de
    "decide al azar".
    """

    saturation_recovery: float = Field(default=0.004, ge=0.0)
    """Cuánto se recupera por tick el hastío de repetir la misma acción."""

    novelty_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    """Cuánto castiga el hastío pleno:

        novedad = 1 - saturación * novelty_weight

    Acotado por debajo a `1 - novelty_weight`, y ahí está su razón de ser.
    Con novedad pura, una acción saturada al máximo valdría cero aunque
    hiciera falta con urgencia, y eso produciría un agente que evita lo
    imprescindible por repetido. 0.0 desactiva el mecanismo.
    """

    loss_aversion: float = Field(default=1.0, ge=0.0)
    """Cuánto pesa el coste frente al beneficio, en el término de coste de la
    utilidad.

    1.0 neutral. Por encima, prudente: pesa más lo que se pierde que lo que
    se gana. Por debajo, se decide más por el beneficio inmediato. 0.0
    desactiva los costes.
    """

    learning_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Cuánto corrige el agente sus expectativas tras cada resultado (alfa).

    Nunca alto: a 0.10 hacen falta del orden de veinte resultados para que
    una creencia se asiente, que es lo que se quiere — un agente que cambia
    de opinión con cada resultado aislado no ha aprendido nada, ha
    reaccionado. 0.0 desactiva el aprendizaje y las expectativas se quedan en
    lo que declara el pack.
    """

    strain_multipliers: dict[TemperamentKnob, PositiveFactor] = Field(default_factory=dict)
    """Cómo cambia el temperamento cuando el desgaste está alto.

    Tercera capa sobre las mismas perillas: mundo * rasgo * estado. Se
    compone igual, con el desgaste (0..1) como exponente, así que un agente
    descansado no se ve afectado y uno al límite recibe el multiplicador
    completo. Produce los "días peores" sin tocar la personalidad de base: el
    mismo agente decide distinto según cómo esté, y sigue siendo el mismo.
    """

    duration_discount: float = Field(default=1.0, gt=0.0, le=1.0)
    """Cuánto resta atractivo comprometer tiempo: `descuento ** duración`.

    Cubre el coste de oportunidad de comprometerse: mientras se atiende algo
    largo, no se puede atender nada más, y eso no aparecía en la utilidad por
    ningún otro lado. Se aplica solo al beneficio, nunca al neto: sobre el
    neto, una opción mala y larga parecería menos mala por durar. 1.0 lo
    desactiva y es el valor por defecto.
    """

    switch_cost: float = Field(default=0.0, ge=0.0, le=1.0)
    """Cuánto protege lo ya invertido frente a una interrupción:

        umbral_efectivo = umbral * (1 - progreso * switch_cost)

    A mitad de una tarea el listón para abandonarla baja; casi al final, solo
    algo crítico la corta. Es el mecanismo detrás del compromiso con
    excepción de urgencia: se termina lo empezado salvo que algo cruce el
    umbral crítico. 0.0 lo desactiva: se interrumpe igual con nada hecho que
    con casi todo.
    """

    critical_need_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    """Por debajo de este valor, un eje puede interrumpir lo que se esté
    haciendo. Solo interrumpe si la acción en curso no lo atiende ya.

    0.0 desactiva las interrupciones: se termina siempre lo que se empieza,
    aunque algo se esté hundiendo.
    """
