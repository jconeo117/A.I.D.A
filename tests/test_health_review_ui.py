"""Pruebas unitarias de la interfaz de revisión en el navegador (Plan 71).

Cubre §B.2 de `.claude/workflow/71-plan-ui-revision.md`:
7. La página responde 200 con el token y contiene título y textos de ui.yaml.
8. Sin token, 404 (raíz, ruta inventada, token mal escrito).
9. Guardado válido devuelve ReviewSubmission con Answer por ref y veredictos enviados.
10. elapsed_seconds es el menor entre página y servidor (evita falsear métrica).
11. Falta una ref -> 400, y el servidor sigue esperando.
12. Una ref desconocida -> 400.
13. Veredicto no admitido para el ítem (edited sobre código) -> 400.
14. edited sin texto -> 400; texto con veredicto no edited -> 400.
15. Falta marcar una sección -> 400 con missing_sections_text.
16. Tras guardar, el servidor se apaga (segundo intento no conecta).
17. Expiración: timeout_seconds bajo levanta ReviewAbandoned y libera puerto.
18. reviewer_from levanta ante ítem no contestado y devuelve Answer exacta para los demás.
19. load_ui_settings valida claves obligatorias, timeout_seconds numérico y positivo, y pack real.
20. Escucha solo en 127.0.0.1 (dirección del socket).
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from aida.arms.health.review import (
    CODE_VERDICTS,
    FINDING_VERDICTS,
    Answer,
    ItemKind,
    ReviewItem,
    Verdict,
)
from aida.arms.health.review_ui import (
    ReviewAbandoned,
    ReviewSubmission,
    UISettings,
    collect_review,
    load_ui_settings,
    reviewer_from,
)
from tests.conftest import _ORIGINAL_HANDLE_REQUEST


@pytest.fixture(autouse=True)
def _permitir_servidor_revision_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restaura el transporte real de httpx para comunicar con el ThreadingHTTPServer
    efímero que este módulo levanta en localhost para probar la interfaz web (§B.8 #17)."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


def _crear_items_prueba() -> tuple[ReviewItem, ReviewItem]:
    f = ReviewItem(
        kind=ItemKind.FINDING,
        ref="motivo:0",
        label="Motivo de consulta",
        text="Cefalea frontal",
        allowed=FINDING_VERDICTS,
    )
    c = ReviewItem(
        kind=ItemKind.CODE,
        ref="I10X",
        label="I10X",
        text="Hipertensión esencial",
        allowed=CODE_VERDICTS,
    )
    return f, c


def _lanzar_servidor(
    items: tuple[ReviewItem, ...],
    settings: UISettings | None = None,
    clock: Any = None,
) -> tuple[str, str, str, int, threading.Thread, dict[str, Any]]:
    ajustes = settings if settings is not None else load_ui_settings()
    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    kwargs: dict[str, Any] = {"settings": ajustes, "open_browser": _abrir}
    if clock is not None:
        kwargs["clock"] = clock

    def _ejecutar() -> None:
        try:
            info["submission"] = collect_review(items, **kwargs)
        except Exception as exc:
            info["error"] = exc

    hilo = threading.Thread(target=_ejecutar, daemon=True)
    hilo.start()
    assert listo.wait(timeout=5.0), "el servidor no levantó a tiempo"
    url = info["url"]
    base, token = url.rsplit("/", 1)
    puerto = int(base.rsplit(":", 1)[1])
    return url, token, base, puerto, hilo, info


def test_pagina_responde_200_con_token_y_textos_de_ui_yaml() -> None:
    """Caso 7 (§B.2): La página responde 200 con el token y contiene los textos
    declarados en ui.yaml."""
    items = _crear_items_prueba()
    settings = load_ui_settings()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items, settings)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        html = resp.text
        assert settings.wording.title in html
        assert settings.wording.intro in html
        assert settings.wording.codes_heading in html
        assert "motivo:0" in html
        assert "I10X" in html
    finally:
        # Guardar para apagar el servidor
        httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 1.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_sin_token_responde_404() -> None:
    """Caso 8 (§B.2): Sin token devuelve 404 (raíz, ruta inventada, token mal escrito)."""
    items = _crear_items_prueba()
    url, _tok, base, _port, hilo, _info = _lanzar_servidor(items)
    try:
        assert httpx.get(f"{base}/", timeout=5.0).status_code == 404
        assert httpx.get(f"{base}/noexiste", timeout=5.0).status_code == 404
        assert httpx.get(f"{base}/token_invalido", timeout=5.0).status_code == 404
    finally:
        httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 1.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_guardado_valido_devuelve_submission_con_answers() -> None:
    """Caso 9 (§B.2): Un guardado válido devuelve ReviewSubmission con una Answer
    por ref y los veredictos enviados."""
    items = _crear_items_prueba()
    url, _tok, _base, _port, hilo, info = _lanzar_servidor(items)
    resp = httpx.post(
        f"{url}/guardar",
        json={
            "elapsed_seconds": 15.5,
            "sections_reviewed": ["motivo"],
            "answers": {
                "motivo:0": {"verdict": "edited", "text": "Cefalea tensional"},
                "I10X": {"verdict": "rejected"},
            },
        },
        timeout=5.0,
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    hilo.join(timeout=5.0)

    sub: ReviewSubmission = info["submission"]
    assert isinstance(sub, ReviewSubmission)
    assert len(sub.answers) == 2
    assert sub.answers["motivo:0"] == Answer(Verdict.EDITED, text="Cefalea tensional")
    assert sub.answers["I10X"] == Answer(Verdict.REJECTED, text=None)
    assert sub.elapsed_seconds <= 15.5


def test_elapsed_seconds_usa_el_menor_entre_pagina_y_servidor() -> None:
    """Caso 10 (§B.2): elapsed_seconds usa min(declarado, medido) para que el
    navegador no pueda falsear la métrica principal."""
    items = _crear_items_prueba()
    t_actual = [100.0]

    def _reloj() -> float:
        return t_actual[0]

    url, _tok, _base, _port, hilo, info = _lanzar_servidor(items, clock=_reloj)
    # El servidor arrancó en t=100.0; avanzamos el reloj a 110.0 (10 segundos medidos)
    t_actual[0] = 110.0

    resp = httpx.post(
        f"{url}/guardar",
        json={
            "elapsed_seconds": 99999.0,  # Navegador envía número inflado
            "sections_reviewed": ["motivo"],
            "answers": {
                "motivo:0": {"verdict": "confirmed"},
                "I10X": {"verdict": "confirmed"},
            },
        },
        timeout=5.0,
    )
    assert resp.status_code == 200
    hilo.join(timeout=5.0)

    sub: ReviewSubmission = info["submission"]
    assert sub.elapsed_seconds == 10.0  # Menor entre 99999.0 y 10.0


def test_falta_una_ref_da_400_y_servidor_sigue_esperando() -> None:
    """Caso 11 (§B.2): Si falta una ref en answers, 400 y collect_review sigue
    esperando sin devolver a medias."""
    items = _crear_items_prueba()
    url, _tok, _base, _port, hilo, info = _lanzar_servidor(items)
    try:
        resp = httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 5.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"}
                    # Falta "I10X"
                },
            },
            timeout=5.0,
        )
        assert resp.status_code == 400
        assert "no cubren exactamente" in resp.json()["error"]
        assert hilo.is_alive()
        assert "submission" not in info
    finally:
        # Completar guardado para cerrar
        httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 5.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_ref_desconocida_da_400() -> None:
    """Caso 12 (§B.2): Una ref desconocida en answers responde 400."""
    items = _crear_items_prueba()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items)
    try:
        resp = httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 5.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                    "desconocido:99": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        assert resp.status_code == 400
    finally:
        httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 5.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_veredicto_no_admitido_para_item_da_400() -> None:
    """Caso 13 (§B.2): Un veredicto no admitido (ej. edited sobre código) responde 400."""
    items = _crear_items_prueba()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items)
    try:
        resp = httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 5.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "edited", "text": "Texto ilegal en código"},
                },
            },
            timeout=5.0,
        )
        assert resp.status_code == 400
        assert "no admitido" in resp.json()["error"]
    finally:
        httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 5.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_validacion_edited_sin_texto_y_texto_en_no_edited_da_400() -> None:
    """Caso 14 (§B.2): edited sin texto o con espacios da 400; texto en veredicto
    distinto de edited da 400."""
    items = _crear_items_prueba()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items)
    try:
        # 1. edited sin texto
        r1 = httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 1.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "edited", "text": "  "},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        assert r1.status_code == 400
        assert "edición sin texto" in r1.json()["error"]

        # 2. confirmed con texto
        r2 = httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 1.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed", "text": "Sobran datos"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        assert r2.status_code == 400
        assert "texto sin edición" in r2.json()["error"]
    finally:
        httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 1.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_falta_marcar_seccion_da_400_con_missing_sections_text() -> None:
    """Caso 15 (§B.2): Falta marcar una sección en sections_reviewed da 400
    con missing_sections_text."""
    items = _crear_items_prueba()
    settings = load_ui_settings()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items, settings)
    try:
        resp = httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 5.0,
                "sections_reviewed": [],  # No marcó "motivo"
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == settings.wording.missing_sections_text
    finally:
        httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 5.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_servidor_se_apaga_tras_guardar() -> None:
    """Caso 16 (§B.2): Tras un guardado válido, el servidor se apaga; un segundo
    POST no conecta."""
    items = _crear_items_prueba()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items)
    resp = httpx.post(
        f"{url}/guardar",
        json={
            "elapsed_seconds": 1.0,
            "sections_reviewed": ["motivo"],
            "answers": {
                "motivo:0": {"verdict": "confirmed"},
                "I10X": {"verdict": "confirmed"},
            },
        },
        timeout=5.0,
    )
    assert resp.status_code == 200
    hilo.join(timeout=5.0)

    # Segundo POST falla en conexión
    with pytest.raises(httpx.ConnectError):
        httpx.post(
            f"{url}/guardar",
            json={"elapsed_seconds": 1.0, "sections_reviewed": ["motivo"], "answers": {}},
            timeout=1.0,
        )


def test_expiracion_levanta_review_abandoned_y_libera_puerto() -> None:
    """Caso 17 (§B.2): Expiración de timeout_seconds levanta ReviewAbandoned
    y libera el puerto."""
    items = _crear_items_prueba()
    settings = replace(load_ui_settings(), timeout_seconds=0.05)

    _url, _tok, _base, puerto, hilo, info = _lanzar_servidor(items, settings)
    hilo.join(timeout=5.0)

    assert "error" in info
    assert isinstance(info["error"], ReviewAbandoned)

    # El puerto quedó liberado
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{puerto}/", timeout=1.0)


def test_reviewer_from_levanta_ante_item_no_contestado() -> None:
    """Caso 18 (§B.2): reviewer_from levanta ante un ítem no contestado y devuelve
    la Answer exacta para los demás."""
    sub = ReviewSubmission(
        answers={"motivo:0": Answer(Verdict.CONFIRMED)},
        elapsed_seconds=12.0,
    )
    reviewer = reviewer_from(sub)

    f_ok = ReviewItem(ItemKind.FINDING, "motivo:0", "Motivo", "Cefalea", FINDING_VERDICTS)
    f_missing = ReviewItem(ItemKind.FINDING, "examen:0", "Examen", "Presión", FINDING_VERDICTS)

    ans = reviewer(f_ok)
    assert ans == Answer(Verdict.CONFIRMED)

    with pytest.raises(ValueError, match="la pantalla no contestó al ítem 'examen:0'"):
        reviewer(f_missing)


def test_load_ui_settings_valida_claves_y_tipos(tmp_path: Path) -> None:
    """Caso 19 (§B.2): load_ui_settings valida claves obligatorias, timeout_seconds
    numérico y positivo, y el pack real carga correctamente."""
    real = load_ui_settings()
    assert real.timeout_seconds == 1200.0
    assert len(real.wording.title) > 0
    assert len(real.wording.abandoned_text) > 0

    base_yaml = (
        "title: 'T'\nintro: 'I'\nconfirm_label: 'C'\nedit_label: 'E'\nreject_label: 'R'\n"
        "section_reviewed_label: 'S'\ncodes_heading: 'H'\nsave_label: 'G'\nsaved_text: 'OK'\n"
        "missing_sections_text: 'F'\nabandoned_text: 'A'\ntimeout_seconds: 60.0\n"
    )

    def _probar(contenido: str, match: str) -> None:
        p = tmp_path / "ui_test.yaml"
        p.write_text(contenido, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_ui_settings(p)

    # Clave faltante
    _probar(base_yaml.replace("title: 'T'\n", ""), "title")
    # Timeout inválido
    _probar(base_yaml.replace("timeout_seconds: 60.0", "timeout_seconds: 0"), "timeout_seconds")
    _probar(base_yaml.replace("timeout_seconds: 60.0", "timeout_seconds: -10"), "timeout_seconds")
    _probar(base_yaml.replace("timeout_seconds: 60.0", "timeout_seconds: true"), "timeout_seconds")
    _probar(
        base_yaml.replace("timeout_seconds: 60.0", "timeout_seconds: 'diez'"),
        "timeout_seconds",
    )


def test_servidor_escucha_solo_en_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 20 (§B.2): El servidor escucha solo en 127.0.0.1 (evita exposición)."""
    from http.server import ThreadingHTTPServer

    bound_hosts: list[str] = []
    orig_init = ThreadingHTTPServer.__init__

    def spy_init(
        self: Any,
        server_address: Any,
        request_handler_class: Any,
        bind_and_activate: bool = True,
    ) -> None:
        bound_hosts.append(server_address[0])
        orig_init(self, server_address, request_handler_class, bind_and_activate)

    monkeypatch.setattr(ThreadingHTTPServer, "__init__", spy_init)

    items = _crear_items_prueba()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items)
    try:
        assert len(bound_hosts) == 1
        assert bound_hosts[0] == "127.0.0.1"
    finally:
        httpx.post(
            f"{url}/guardar",
            json={
                "elapsed_seconds": 1.0,
                "sections_reviewed": ["motivo"],
                "answers": {
                    "motivo:0": {"verdict": "confirmed"},
                    "I10X": {"verdict": "confirmed"},
                },
            },
            timeout=5.0,
        )
        hilo.join(timeout=5.0)
