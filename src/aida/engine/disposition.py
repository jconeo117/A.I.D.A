"""Personalidad compuesta y ponderaciones numéricas del agente.

Define cómo los rasgos inmutables del pack y las intensidades individuales
se traducen en factores operativos: pesos de necesidades, aptitudes,
temples, topes y parámetros de temperamento.

Todo valor en este módulo es derivado de especificaciones y estado:
se recalcula cuando los rasgos, metas o desgaste varían.

**Invariante 1 — Determinismo y no-asociatividad en coma flotante:**
La suma y el producto en aritmética de coma flotante (IEEE 754) no son
asociativos: `(a + b) + c != a + (b + c)` y `(a * b) * c != a * (b * c)`.
En consecuencia, iterar diccionarios o conjuntos cuyo orden dependa del hash
o del orden de inserción genera discrepancias en los bits menos significativos
entre plataformas o ejecuciones con distinta semilla de hash. Este módulo
garantiza determinismo estricto recorriendo siempre colecciones ordenadas por
identificador determinista (`sorted()`) en sus cinco puntos de composición:
rasgos, metas activas, necesidades, situaciones activas y perillas.

**Canales independientes:**
La personalidad opera a través de canales ortogonales:
- `compose_gains` (aptitud): cuánto rinde lo que el agente repone.
- `compose_costs` (temple): cuánto le cuesta o duele lo que gasta.
- `compose_caps` (techos): capacidad máxima estructural efectiva.
- `compose_need_weights` (priorización): importancia subjetiva de cada eje.
- `compose_temperament` (temperamento): modulación de toma de decisiones.
Separar aptitud de temple permite modelar agentes con alta asimilación pero
alta sensibilidad al gasto, combinaciones imposibles bajo un factor único.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from aida.engine.ontology import (
    GoalSpec,
    NeedId,
    NeedSpec,
    SituationId,
    TasteSpec,
    TemperamentKnob,
    TraitId,
)
from aida.engine.pack import ContentPack

_MIN_TEMPERATURE = 1e-6
"""Suelo de temperatura para evitar divisiones por cero en la selección
estocástica posterior manteniendo valores de toma de decisión metódicos."""


def compose_need_weights(
    pack: ContentPack,
    traits: Mapping[TraitId, float],
    pursuing: Iterable[GoalSpec] = (),
) -> dict[NeedId, float]:
    """Calcula la ponderación efectiva de cada necesidad del pack.

    Combina la base declarada en cada necesidad con dos modulaciones
    multiplicativas:
    1. Rasgos individuales (`need_weight_multipliers`), donde la intensidad
       actúa como exponente (`multiplicador ** intensidad`).
    2. Metas en curso (`weight_multiplier`), ordenadas por identificador.

    **Por qué las metas multiplican y no suman:**
    Una meta sobre un eje que el agente ya prioriza ejerce una atracción
    absoluta mayor que sobre un eje indiferente, preservando la coherencia
    del carácter del agente frente a objetivos activos.

    **No-asociatividad en coma flotante:**
    Tanto los rasgos (`sorted(traits)`) como las metas activas
    (`sorted(pursuing, key=lambda g: g.id)`) se iteran en orden estricto de
    identificador para asegurar que la acumulación del producto sea bit a bit
    reproducible e inmune a permutaciones en el diccionario de entrada.

    Devuelve todas las necesidades presentes en el pack para garantizar
    un cálculo uniforme sin requerir valores por defecto externos.
    """
    weights: dict[NeedId, float] = {need.id: need.base_weight for need in pack.needs}
    for trait_id in sorted(traits):
        trait = pack.trait_by_id[trait_id]
        intensity = traits[trait_id]
        for need_id, multiplier in trait.need_weight_multipliers.items():
            weights[need_id] *= multiplier**intensity

    for goal in sorted(pursuing, key=lambda g: g.id):
        weights[goal.need] *= goal.weight_multiplier
    return weights


def compose_gains(
    pack: ContentPack,
    traits: Mapping[TraitId, float],
) -> dict[NeedId, float]:
    """Calcula la aptitud o rendimiento del agente al reponer cada necesidad.

    Canal independiente de asimilación: mide cuánto le cunde al agente lo
    que ingresa en cada eje al actuar. Arranca en 1.0 para todas las
    necesidades del pack y modula mediante los rasgos individuales
    (`gain_multipliers`), elevando el multiplicador a la intensidad del rasgo.

    **Independencia de canal:**
    Es ortogonal a `compose_costs`: un agente puede tener una aptitud muy
    elevada y a la vez un temple frágil ante los costes, o viceversa.

    **No-asociatividad:**
    Itera `sorted(traits)` para garantizar que el producto de factores en
    coma flotante sea determinista e idéntico sin importar el orden de
    inserción en `traits`.
    """
    gains: dict[NeedId, float] = {need.id: 1.0 for need in pack.needs}
    for trait_id in sorted(traits):
        trait = pack.trait_by_id[trait_id]
        intensity = traits[trait_id]
        for need_id, multiplier in trait.gain_multipliers.items():
            gains[need_id] *= multiplier**intensity
    return gains


def compose_costs(
    pack: ContentPack,
    traits: Mapping[TraitId, float],
) -> dict[NeedId, float]:
    """Calcula la tolerancia o coste subjetivo al gastar cada necesidad.

    Canal independiente de temple o resistencia: mide cuánto le duele al
    agente lo que gasta en cada eje. Arranca en 1.0 para todas las necesidades
    del pack y modula mediante los rasgos individuales (`cost_multipliers`),
    elevando el multiplicador a la intensidad del rasgo.

    **Independencia de canal:**
    Es ortogonal a la aptitud (`compose_gains`). Las cuatro combinaciones
    extremas (alta aptitud y alto temple, alta aptitud y bajo temple, baja
    aptitud y alto temple, baja en ambas) describen arquetipos de agente
    completamente diferenciados que un factor único de eficiencia colapsaría.

    **No-asociatividad:**
    Itera `sorted(traits)` para garantizar que el producto de factores en
    coma flotante sea determinista e independiente del orden de los rasgos.
    """
    costs: dict[NeedId, float] = {need.id: 1.0 for need in pack.needs}
    for trait_id in sorted(traits):
        trait = pack.trait_by_id[trait_id]
        intensity = traits[trait_id]
        for need_id, multiplier in trait.cost_multipliers.items():
            costs[need_id] *= multiplier**intensity
    return costs


def compose_caps(
    pack: ContentPack,
    traits: Mapping[TraitId, float],
) -> dict[NeedId, float]:
    """Calcula el límite superior alcanzable en cada necesidad.

    Canal estructural de capacidad: determina el tope efectivo al que el
    agente puede aspirar en cada eje. Arranca en el tope definido por el
    pack (`NeedSpec.cap`) y se modula por los rasgos del agente
    (`cap_multipliers`), elevando cada multiplicador a la intensidad.

    El resultado final se recorta obligatoriamente a 1.0, pues ningún agente
    puede exceder el tope de la escala universal.

    **No-asociatividad:**
    Itera `sorted(traits)` para garantizar que el producto de factores en
    coma flotante sea determinista e independiente del orden de inserción.
    """
    caps: dict[NeedId, float] = {need.id: need.cap for need in pack.needs}
    for trait_id in sorted(traits):
        trait = pack.trait_by_id[trait_id]
        intensity = traits[trait_id]
        for need_id, multiplier in trait.cap_multipliers.items():
            caps[need_id] *= multiplier**intensity
    return {need_id: min(1.0, cap_val) for need_id, cap_val in caps.items()}


def urgency(value: float, exponent: float, cap: float = 1.0) -> float:
    """Calcula el apremio de una necesidad según su nivel respecto al tope.

    Aplica la curva convexa `(1 - valor / tope) ** exponente`.

    **Por qué el tope (`cap`) es del agente:**
    El techo representa la capacidad efectiva de este agente concreto en
    este eje. Si la urgencia evaluara contra 1.0 universal, un agente cuyo
    tope sea 0.5 y se encuentre en 0.5 seguiría sintiendo la mitad del
    apremio posible a pesar de haberlo dado todo. Normalizar respecto a su
    techo personal garantiza que alcanzar su límite efectivo se experimente
    como saciedad plena (urgencia 0.0).

    **Por qué `cap <= 0.0` devuelve 0.0:**
    Evita divisiones por cero y previene valores indefinidos o negativos si
    un eje carece de capacidad operativa.

    **Por qué conversión explícita `float()`:**
    En Python y para el analizador de tipos estático, la exponenciación `**`
    con base negativa y exponente fraccionario produce números complejos.
    Aunque `max(0.0, 1.0 - value / cap)` impide bases negativas en ejecución,
    el cast explícito `float()` garantiza contractualmente al sistema de tipos
    y al motor que la salida es siempre un flotante real.

    **Por qué reside en `disposition.py` y no en deliberación:**
    La urgencia es una función afectiva y de valoración subjetiva compartida
    entre la composición de desgaste (`compute_strain`) y la futura toma de
    decisiones. Alojarla aquí consolida la única puerta canónica de cálculo
    de urgencia en todo el motor.
    """
    if cap <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - value / cap) ** exponent)


def compute_strain(
    pack: ContentPack | Mapping[NeedId, NeedSpec],
    needs: Mapping[NeedId, float],
    caps: Mapping[NeedId, float] | None = None,
) -> float:
    """Calcula el desgaste global como media ponderada convexa de urgencias.

    Constituye la única puerta canónica para el cálculo de desgaste en todo el
    motor. Pondera las urgencias de las necesidades con `strain_weight > 0`
    aplicando la curva convexa de urgencia de cada eje respecto a su techo
    efectivo.

    **Por qué convexidad frente a desgaste lineal:**
    Con desgaste lineal, un sistema sano —ejes en torno a 0.55— reportaría un
    desgaste de ~0.45: permanentemente medio roto y consumiendo casi la mitad
    de la escala en normalidad. El mecanismo perdería su recorrido dinámico útil.
    Con la curva convexa de urgencia, ese mismo estado sano da ~0.15 de desgaste,
    reservando el tramo alto de la escala para situaciones de deterioro crítico
    hacia el suelo.

    **Por qué techos del agente (`caps`):**
    El cálculo de desgaste normaliza cada eje mediante el techo efectivo del
    agente (`caps.get(need.id, need.cap)` si se suministran techos, o
    `need.cap` nominal por omisión). Sin esta normalización, un agente con
    tope reducido que esté en su máximo seguiría sufriendo desgaste por no
    alcanzar el 1.0 universal, distorsionando el propósito del mecanismo.

    **Propiedades:**
    - Devuelve 0.0 si ningún eje declara `strain_weight > 0`, desactivando el
      mecanismo sin requerir código condicional en los consumidores.
    - Ordena las necesidades por identificador determinista (`sorted`) para
      garantizar que la suma ponderada en coma flotante sea bit a bit
      reproducible (Invariante 1 de determinismo).
    """
    total = 0.0
    weights = 0.0
    need_specs: Iterable[NeedSpec] = (
        pack.needs if isinstance(pack, ContentPack) else pack.values()
    )
    for need in sorted(need_specs, key=lambda n: n.id):
        if need.strain_weight <= 0.0:
            continue
        cap = caps.get(need.id, need.cap) if caps is not None else need.cap
        pressure = urgency(needs[need.id], need.urgency_exponent, cap)
        total += need.strain_weight * pressure
        weights += need.strain_weight
    return total / weights if weights else 0.0


@dataclass(frozen=True, slots=True)
class Temperament:
    """Parámetros cuantitativos de decisión ya compuestos para un agente.

    Estructura inmutable que refleja el estado derivado de los factores
    del entorno, la personalidad y las condiciones operativas vigentes.

    **Por qué `dataclass(frozen=True, slots=True)` y no pydantic:**
    Es una regla no negociable del proyecto: pydantic valida lo que escribe un
    humano (packs en YAML); esto lo construye el motor en su ciclo interno
    caliente con datos que él mismo acaba de calcular. Pagar validación de
    pydantic aquí sería gastar ciclos en comprobar lo que ya se sabe cierto.
    `frozen=True` garantiza inmutabilidad y `slots=True` elimina la sobrecarga
    de `__dict__` por instancia.
    """

    temperature: float
    loss_aversion: float
    critical_need_threshold: float
    switch_cost: float
    novelty_weight: float
    learning_rate: float


def compose_temperament(
    pack: ContentPack,
    traits: Mapping[TraitId, float],
    strain: float = 0.0,
    active_situations: Iterable[SituationId] = (),
) -> Temperament:
    """Compone los parámetros de temperamento a partir de tres capas multiplicativas.

    Capa base de calibración (`TuningSpec`), modulada exponencialmente por
    rasgos individuales, por desgaste operativo (`strain_multipliers`) y por
    situaciones activas (`temperament_multipliers`).

    **Cotas estrictas de cada parámetro:**
    - `temperature`: suelo estricto en `_MIN_TEMPERATURE` (`1e-6`) para impedir
      divisiones por cero en la posterior deliberación probabilística, sin tope
      superior.
    - `critical_need_threshold`: techo en 1.0 (umbral de necesidad crítica).
    - `switch_cost`: techo en 1.0 (penalización por cambio de tarea).
    - `novelty_weight`: techo en 1.0 (peso de la búsqueda de variedad).
    - `learning_rate`: techo en 1.0 (tasa de aprendizaje de la regla delta).
    - `loss_aversion`: **sin techo**. Pesar el precio diez o veinte veces más
      describe con coherencia a un agente paralizado ante el coste, lo cual
      constituye un arquetipo psicológico válido que no debe truncarse.

    **Persistencia de perillas en cero:**
    Dado que las tres capas operan mediante multiplicación, si una perilla
    del pack está calibrada en 0.0 en `TuningSpec`, ningún rasgo ni situación
    puede revivirla (`0.0 * x = 0.0`). Lo que el pack apaga permanece apagado.

    **No-asociatividad:**
    Itera `sorted(traits)` y `sorted(active_situations)` para preservar el orden
    de composición multiplicativa en coma flotante con independencia del orden
    de inserción de los diccionarios.
    """
    factors: dict[TemperamentKnob, float] = dict.fromkeys(TemperamentKnob, 1.0)

    for trait_id in sorted(traits):
        trait = pack.trait_by_id[trait_id]
        intensity = traits[trait_id]
        for knob, multiplier in trait.temperament_multipliers.items():
            factors[knob] *= multiplier**intensity

    for knob, multiplier in pack.tuning.strain_multipliers.items():
        factors[knob] *= multiplier**strain

    for situation_id in sorted(active_situations):
        situation = pack.situation_by_id[situation_id]
        for knob, multiplier in situation.temperament_multipliers.items():
            factors[knob] *= multiplier

    tuning = pack.tuning
    return Temperament(
        temperature=max(
            _MIN_TEMPERATURE, tuning.temperature * factors[TemperamentKnob.TEMPERATURE]
        ),
        loss_aversion=tuning.loss_aversion * factors[TemperamentKnob.LOSS_AVERSION],
        critical_need_threshold=min(
            1.0,
            tuning.critical_need_threshold * factors[TemperamentKnob.CRITICAL_THRESHOLD],
        ),
        switch_cost=min(
            1.0, tuning.switch_cost * factors[TemperamentKnob.SWITCH_COST]
        ),
        novelty_weight=min(
            1.0, tuning.novelty_weight * factors[TemperamentKnob.NOVELTY]
        ),
        learning_rate=min(
            1.0, tuning.learning_rate * factors[TemperamentKnob.LEARNING_RATE]
        ),
    )


def compose_tastes(tastes: Iterable[TasteSpec] = ()) -> dict[str, float]:
    """Combina preferencias hacia objetivos específicos multiplicando factores coincidentes.

    **Por qué la clave es `str`:**
    Un gusto puede apuntar a distintos espacios de identificadores (acciones o
    situaciones). Utilizar `str` unifica la búsqueda sin requerir tipos unión
    en el ciclo caliente de decisión. Las validaciones de `pack.py` garantizan
    de antemano que no existan colisiones entre identificadores de distintos tipos.

    Agrupa los multiplicadores por el identificador del objetivo. Objetivos
    duplicados acumulan multiplicativamente sus efectos.
    """
    result: dict[str, float] = {}
    for taste in tastes:
        key = str(taste.target)
        result[key] = result.get(key, 1.0) * taste.multiplier
    return result
