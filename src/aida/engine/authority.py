"""Hasta dónde llega el asistente por su cuenta, y qué pasa con lo demás.

Puerta dura, no rasgo. `decide` puntúa; esta pieza no puntúa nada: dice sí o
no. Una acción por encima del techo del pack no se ejecuta jamás sin un visto
bueno explícito, y no hay utilidad, urgencia ni modo experto que lo cambie.

Lo que queda por encima del techo no se prohíbe: se **propone**. El asistente
no se queda esperando —sigue deliberando y hace lo siguiente que sí puede
hacer solo—, y la propuesta caduca si nadie contesta, porque una propuesta
eterna acaba siendo una autorización que nadie dio.

El permiso entra por llamada explícita (`approve` / `deny`), nunca leyendo el
buzón de entrada. Lo que llega de fuera es dato, nunca instrucción: si un
texto cualquiera pudiera autorizar, el techo lo pondría quien escribiera el
texto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aida.engine.ontology import ActionId, ActionSpec, AuthoritySpec, Reversibility

_RANK: dict[Reversibility, int] = {
    Reversibility.REVERSIBLE: 0,
    Reversibility.REVERSIBLE_WITH_COST: 1,
    Reversibility.IRREVERSIBLE: 2,
}
"""Orden de las tres clasificaciones. Explícito y no `list(Reversibility)`:
el orden de un enum es el de su declaración, y que una reordenación al editar
el archivo cambie quién necesita permiso sería exactamente el fallo silencioso
que esta capa existe para no tener."""


@dataclass(slots=True)
class Authority:
    """Techo de autoridad y propuestas vivas.

    `approved` es de un solo uso: autorizar abre UNA ejecución, no el permiso
    permanente sobre esa acción. Si se quisiera permanente, se sube el techo
    en el pack, que es donde se ve y se audita.
    """

    spec: AuthoritySpec
    pending: dict[ActionId, int] = field(default_factory=dict)
    """Acción propuesta -> instante en que caduca."""

    approved: set[ActionId] = field(default_factory=set)

    def is_gated(self, action: ActionSpec) -> bool:
        """Si empezarla exige permiso que el pack no concede de antemano."""
        return _RANK[action.reversibility] > _RANK[self.spec.autonomous_up_to]

    def may_start(self, action: ActionSpec) -> bool:
        """Sí o no, y consume la autorización si la había.

        Consumir aquí y no en quien llama es deliberado: si el consumo viviera
        fuera, una segunda ejecución con el mismo permiso dependería de que
        todos los llamantes se acordaran de gastarlo.
        """
        if not self.is_gated(action):
            return True
        if action.id in self.approved:
            self.approved.discard(action.id)
            self.pending.pop(action.id, None)
            return True
        return False

    def propose(self, action: ActionSpec, tick: int) -> bool:
        """Registra la propuesta. `False` si ya había una viva para esa acción.

        Devolver `False` en vez de renovarla evita dos cosas: inundar el
        registro con la misma propuesta cada instante, y que una propuesta se
        renueve sola hasta no caducar nunca.
        """
        if action.id in self.pending:
            return False
        self.pending[action.id] = tick + self.spec.confirmation_ttl_ticks
        return True

    def approve(self, action_id: ActionId) -> bool:
        """Concede UNA ejecución. `False` si no había propuesta viva.

        No se puede autorizar lo que nadie ha propuesto: sin esa guarda, un
        permiso concedido de antemano quedaría almacenado esperando a que el
        asistente decidiera usarlo, que es justo lo contrario de una
        confirmación."""
        if action_id not in self.pending:
            return False
        del self.pending[action_id]
        self.approved.add(action_id)
        return True

    def deny(self, action_id: ActionId) -> bool:
        """Retira la propuesta sin conceder nada."""
        return self.pending.pop(action_id, None) is not None

    def expired(self, tick: int) -> list[ActionId]:
        """Las que caducan en este instante, en orden de identificador.

        Ordenado porque quien lo consuma va a emitir un evento por cada una, y
        el registro tiene que salir igual en dos corridas con la misma semilla.
        """
        vencidas = sorted(aid for aid, limite in self.pending.items() if limite <= tick)
        for action_id in vencidas:
            del self.pending[action_id]
        return vencidas
