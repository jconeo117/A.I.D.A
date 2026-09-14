"""Pruebas unitarias de la capa de autoridad (`aida.engine.authority`).

Cubre la unidad U8 según la especificación de `.claude/workflow/19-plan-u8.md` §6.1.
Verifica:
1. Tabla completa de 9 combinaciones para `is_gated`.
2. `may_start` en acción no vetada devuelve True sin mutar estado.
3. `approve` sin propuesta viva devuelve False y no altera `approved`.
4. `deny` sin propuesta viva devuelve False.
5. Doble `propose` devuelve True y luego False, sin renovar el tick de caducidad.
6. Permiso de un solo uso: primer `may_start` True, segundo False.
7. `expired` devuelve únicamente las vencidas en orden alfabético estricto de identificador
   y las elimina de `pending`.
"""

from __future__ import annotations

import pytest

import aida.engine.authority as authority_mod
import aida.engine.ontology as ontology


def _make_action(
    id_: str, reversibility: ontology.Reversibility
) -> ontology.ActionSpec:
    return ontology.ActionSpec(
        id=ontology.ActionId(id_),
        label=f"Action {id_}",
        duration_ticks=1,
        reversibility=reversibility,
        satisfies={},
        costs={},
        effects={},
    )


def _make_authority(
    autonomous_up_to: ontology.Reversibility = ontology.Reversibility.REVERSIBLE,
    ttl: int = 48,
) -> authority_mod.Authority:
    spec = ontology.AuthoritySpec(
        autonomous_up_to=autonomous_up_to, confirmation_ttl_ticks=ttl
    )
    return authority_mod.Authority(spec=spec)


@pytest.mark.parametrize(
    ("reversibility", "autonomous_up_to", "expected_gated"),
    [
        (ontology.Reversibility.REVERSIBLE, ontology.Reversibility.REVERSIBLE, False),
        (
            ontology.Reversibility.REVERSIBLE,
            ontology.Reversibility.REVERSIBLE_WITH_COST,
            False,
        ),
        (ontology.Reversibility.REVERSIBLE, ontology.Reversibility.IRREVERSIBLE, False),
        (
            ontology.Reversibility.REVERSIBLE_WITH_COST,
            ontology.Reversibility.REVERSIBLE,
            True,
        ),
        (
            ontology.Reversibility.REVERSIBLE_WITH_COST,
            ontology.Reversibility.REVERSIBLE_WITH_COST,
            False,
        ),
        (
            ontology.Reversibility.REVERSIBLE_WITH_COST,
            ontology.Reversibility.IRREVERSIBLE,
            False,
        ),
        (ontology.Reversibility.IRREVERSIBLE, ontology.Reversibility.REVERSIBLE, True),
        (
            ontology.Reversibility.IRREVERSIBLE,
            ontology.Reversibility.REVERSIBLE_WITH_COST,
            True,
        ),
        (
            ontology.Reversibility.IRREVERSIBLE,
            ontology.Reversibility.IRREVERSIBLE,
            False,
        ),
    ],
)
def test_authority_is_gated_9_combinations_matrix(
    reversibility: ontology.Reversibility,
    autonomous_up_to: ontology.Reversibility,
    expected_gated: bool,
) -> None:
    """La compuerta evalúa estrictamente el rango de reversibilidad contra el techo.

    La igualdad al techo es autónoma; solo un rango estrictamente mayor
    (según el orden de rango predefinido) exige confirmación explícita.
    """
    auth = _make_authority(autonomous_up_to=autonomous_up_to)
    action = _make_action("test_act", reversibility=reversibility)
    assert auth.is_gated(action) is expected_gated


def test_authority_may_start_ungated_action_does_not_mutate_state() -> None:
    """Acciones dentro del techo son autónomas y no alteran propuestas ni permisos.

    Consultar may_start para una acción permitida de antemano debe retornar True
    de inmediato sin crear entradas en pending ni gastar permisos espurios.
    """
    auth = _make_authority(autonomous_up_to=ontology.Reversibility.REVERSIBLE)
    action = _make_action("libre", reversibility=ontology.Reversibility.REVERSIBLE)

    assert auth.may_start(action) is True
    assert len(auth.approved) == 0
    assert len(auth.pending) == 0


def test_authority_approve_without_proposal_fails_and_preserves_empty_approved() -> None:
    """No se puede autorizar lo que nadie ha propuesto previamente.

    Evita pre-autorizaciones latentes que permitirían saltarse el ciclo reactivo
    de deliberación y propuesta explícita.
    """
    auth = _make_authority(autonomous_up_to=ontology.Reversibility.REVERSIBLE)
    act_id = ontology.ActionId("peligrosa")

    assert auth.approve(act_id) is False
    assert act_id not in auth.approved
    assert len(auth.approved) == 0


def test_authority_deny_without_proposal_returns_false() -> None:
    """Denegar una acción sin propuesta viva no tiene efecto y retorna False.

    La denegación debe ser un rechazo de una propuesta existente; si no existe,
    no debe reportar éxito ni dejar residuos.
    """
    auth = _make_authority(autonomous_up_to=ontology.Reversibility.REVERSIBLE)
    act_id = ontology.ActionId("no_propuesta")

    assert auth.deny(act_id) is False
    assert len(auth.pending) == 0


def test_authority_double_propose_returns_false_and_does_not_renew_expiration() -> None:
    """Re-proponer una acción ya pendiente no reinicia el temporizador TTL.

    Si una propuesta pudiera renovarse automáticamente en cada instante de
    deliberación, nunca caducaría por silencio del supervisor.
    """
    auth = _make_authority(autonomous_up_to=ontology.Reversibility.REVERSIBLE, ttl=10)
    action = _make_action("peligrosa", reversibility=ontology.Reversibility.IRREVERSIBLE)

    # Primera propuesta en tick=1 vence en 1 + 10 = 11
    assert auth.propose(action, tick=1) is True
    assert auth.pending[action.id] == 11

    # Segunda propuesta en tick=5 debe rechazarse y mantener expiración en 11
    assert auth.propose(action, tick=5) is False
    assert auth.pending[action.id] == 11


def test_authority_single_use_permission_consumed_on_first_execution() -> None:
    """Una autorización concede exactamente una ejecución y se gasta de inmediato.

    El consumo atómico en may_start garantiza que la aprobación no actúe como
    un permiso permanente, obligando a re-proponer si se requiere otra vez.
    """
    auth = _make_authority(autonomous_up_to=ontology.Reversibility.REVERSIBLE, ttl=10)
    action = _make_action("peligrosa", reversibility=ontology.Reversibility.IRREVERSIBLE)

    auth.propose(action, tick=1)
    assert auth.approve(action.id) is True
    assert action.id in auth.approved

    # Primer may_start consume la autorización
    assert auth.may_start(action) is True
    assert action.id not in auth.approved
    assert action.id not in auth.pending

    # Segundo may_start queda bloqueado
    assert auth.may_start(action) is False


def test_authority_expired_orders_deterministically_and_purges_pending() -> None:
    """Las acciones vencidas se retornan en orden lexicográfico y se eliminan del estado.

    El ordenamiento estricto garantiza reproducibilidad absoluta en los registros
    de eventos sin depender del orden no determinista de inserción en diccionarios.
    """
    auth = _make_authority(autonomous_up_to=ontology.Reversibility.REVERSIBLE, ttl=5)
    act_zeta = _make_action("zeta", reversibility=ontology.Reversibility.IRREVERSIBLE)
    act_alfa = _make_action("alfa", reversibility=ontology.Reversibility.IRREVERSIBLE)
    act_beta = _make_action("beta", reversibility=ontology.Reversibility.IRREVERSIBLE)

    # Inserción intencionadamente desordenada en tick=1 (vencen en 6)
    auth.propose(act_zeta, tick=1)
    auth.propose(act_alfa, tick=1)
    # act_beta se propone en tick=10 (vence en 15)
    auth.propose(act_beta, tick=10)

    # En tick=6 vencen zeta y alfa; deben salir ordenadas alfabéticamente
    vencidas = auth.expired(6)
    assert vencidas == [ontology.ActionId("alfa"), ontology.ActionId("zeta")]

    # pending ya no contiene las vencidas, pero mantiene la futura
    assert ontology.ActionId("alfa") not in auth.pending
    assert ontology.ActionId("zeta") not in auth.pending
    assert auth.pending[ontology.ActionId("beta")] == 15
