"""Pruebas unitarias de la interfaz de revisión en el navegador (Plan 71 / Plan 76).

Cubre §B.5 de `.claude/workflow/76-plan-pantalla.md`:
16. La página responde 200 con el token y contiene la historia, la transcripción
    (escapadas), un textarea por campo con su texto, y los textos de ui.yaml.
17. Sin token, 404 (raíz, ruta inventada, token mal escrito).
18. Un guardado válido (answers de los códigos y note con todos los campos) devuelve
    un ReviewSubmission con note_fields igual a lo enviado.
19. note ausente, con un campo de menos, de más, o con un valor no cadena -> 400 con
    'la nota no cubre exactamente sus campos', y el servidor sigue esperando.
20. collect_review con un ítem de hallazgo -> ValueError antes de abrir el puerto.
21. Siguen: expiración -> ReviewAbandoned y puerto liberado; escucha solo en
    127.0.0.1; elapsed_seconds el menor de los dos; reviewer_from levanta ante un
    ítem no contestado.
22. load_ui_settings exige las claves nuevas y ya no exige edit_label,
    section_reviewed_label ni missing_sections_text.
"""

from __future__ import annotations

import html
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from aida.arms.health.note import RenderedNote
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
from aida.arms.health.sections import SectionSpec
from tests.conftest import _ORIGINAL_HANDLE_REQUEST


@pytest.fixture(autouse=True)
def _permitir_servidor_revision_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restaura el transporte real de httpx para comunicar con el ThreadingHTTPServer
    efímero que este módulo levanta en localhost para probar la interfaz web (§B.8 #17)."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


def _crear_codigos_prueba() -> tuple[ReviewItem, ...]:
    return (
        ReviewItem(
            kind=ItemKind.CODE,
            ref="I10X",
            label="I10X",
            text="Hipertensión esencial",
            allowed=CODE_VERDICTS,
        ),
        ReviewItem(
            kind=ItemKind.CODE,
            ref="E119",
            label="E119",
            text="Diabetes mellitus tipo 2",
            allowed=CODE_VERDICTS,
        ),
    )


def _crear_nota_prueba() -> RenderedNote:
    spec_motivo = SectionSpec(
        section_id="motivo",
        label="Motivo de consulta",
        trigger_phrases=(),
        codable=True,
        in_note=True,
    )
    spec_plan = SectionSpec(
        section_id="plan",
        label="Plan de manejo",
        trigger_phrases=(),
        codable=True,
        in_note=True,
    )
    return RenderedNote(
        fields=(
            (spec_motivo, "Cefalea frontal pulsátil"),
            (spec_plan, "Paracetamol 500mg cada 8 horas"),
        ),
        redacted=True,
        provider="openai/gpt-4.1-mini",
    )


def _payload_valido(
    *,
    elapsed: float = 1.0,
    answers: dict[str, Any] | None = None,
    note: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ans = (
        answers
        if answers is not None
        else {
            "I10X": {"verdict": "confirmed"},
            "E119": {"verdict": "confirmed"},
        }
    )
    nt = (
        note
        if note is not None
        else {
            "motivo": "Cefalea frontal pulsátil",
            "plan": "Paracetamol 500mg cada 8 horas",
        }
    )
    return {
        "elapsed_seconds": elapsed,
        "answers": ans,
        "note": nt,
    }


def _lanzar_servidor(
    items: tuple[ReviewItem, ...] | None = None,
    note: RenderedNote | None = None,
    history: str = "Historia previa del paciente: sin antecedentes.",
    transcript: str = "Transcripción: el paciente refiere dolor.",
    settings: UISettings | None = None,
    clock: Any = None,
) -> tuple[str, str, str, int, threading.Thread, dict[str, Any]]:
    ajustes = settings if settings is not None else load_ui_settings()
    items_rev = items if items is not None else _crear_codigos_prueba()
    nota_rev = note if note is not None else _crear_nota_prueba()
    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    kwargs: dict[str, Any] = {
        "note": nota_rev,
        "history": history,
        "transcript": transcript,
        "settings": ajustes,
        "open_browser": _abrir,
    }
    if clock is not None:
        kwargs["clock"] = clock

    def _ejecutar() -> None:
        try:
            info["submission"] = collect_review(items_rev, **kwargs)
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
    """Caso 16 (§B.5): La página responde 200 con el token y contiene la historia,
    la transcripción (escapadas), un textarea por campo con su texto, y los textos de ui.yaml."""
    items = _crear_codigos_prueba()
    note = _crear_nota_prueba()
    historia_texto = "Historia con caracteres especiales: <alergias> & 'antecedentes'."
    transcripcion_texto = "Transcripción con caracteres especiales: <audio> & 'palabras'."
    settings = load_ui_settings()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(
        items=items,
        note=note,
        history=historia_texto,
        transcript=transcripcion_texto,
        settings=settings,
    )
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        html_content = resp.text

        # Textos de ui.yaml
        assert settings.wording.title in html_content
        assert settings.wording.intro in html_content
        assert settings.wording.codes_heading in html_content
        assert settings.wording.history_heading in html_content
        assert settings.wording.transcript_heading in html_content
        assert settings.wording.note_heading in html_content

        # Historia y transcripción escapadas
        assert html.escape(historia_texto) in html_content
        assert html.escape(transcripcion_texto) in html_content

        # Un textarea por campo con su texto
        assert (
            '<textarea class="field-text">Cefalea frontal pulsátil</textarea>' in html_content
        )
        assert (
            '<textarea class="field-text">Paracetamol 500mg cada 8 horas</textarea>'
            in html_content
        )

        # Códigos presentes
        assert "I10X" in html_content
        assert "E119" in html_content
    finally:
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)


def test_sin_token_responde_404() -> None:
    """Caso 17 (§B.5): Sin token devuelve 404 (raíz, ruta inventada, token mal escrito)."""
    items = _crear_codigos_prueba()
    url, _tok, base, _port, hilo, _info = _lanzar_servidor(items)
    try:
        assert httpx.get(f"{base}/", timeout=5.0).status_code == 404
        assert httpx.get(f"{base}/noexiste", timeout=5.0).status_code == 404
        assert httpx.get(f"{base}/token_invalido", timeout=5.0).status_code == 404
    finally:
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)


def test_guardado_valido_devuelve_submission_con_note_fields() -> None:
    """Caso 18 (§B.5): Un guardado válido (answers de los códigos y note con todos
    los campos) devuelve un ReviewSubmission con note_fields igual a lo enviado."""
    items = _crear_codigos_prueba()
    url, _tok, _base, _port, hilo, info = _lanzar_servidor(items)

    ediciones_nota = {
        "motivo": "Cefalea frontal pulsátil de 3 días",
        "plan": "Paracetamol 500mg cada 8 horas por 3 días",
    }
    resp = httpx.post(
        f"{url}/guardar",
        json=_payload_valido(
            elapsed=15.5,
            answers={
                "I10X": {"verdict": "confirmed"},
                "E119": {"verdict": "rejected"},
            },
            note=ediciones_nota,
        ),
        timeout=5.0,
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    hilo.join(timeout=5.0)

    sub: ReviewSubmission = info["submission"]
    assert isinstance(sub, ReviewSubmission)
    assert len(sub.answers) == 2
    assert sub.answers["I10X"] == Answer(Verdict.CONFIRMED)
    assert sub.answers["E119"] == Answer(Verdict.REJECTED)
    assert sub.note_fields == ediciones_nota
    assert sub.elapsed_seconds <= 15.5


def test_note_invalida_da_400_y_servidor_sigue_esperando() -> None:
    """Caso 19 (§B.5): note ausente, con un campo de menos, de más, o con un valor
    no cadena -> 400 con 'la nota no cubre exactamente sus campos', y el servidor
    sigue esperando."""
    items = _crear_codigos_prueba()
    url, _tok, _base, _port, hilo, info = _lanzar_servidor(items)

    try:
        # 1. note ausente
        p_ausente = {
            "elapsed_seconds": 1.0,
            "answers": {"I10X": {"verdict": "confirmed"}, "E119": {"verdict": "confirmed"}},
        }
        r1 = httpx.post(f"{url}/guardar", json=p_ausente, timeout=5.0)
        assert r1.status_code == 400
        assert r1.json()["error"] == "la nota no cubre exactamente sus campos"
        assert hilo.is_alive()
        assert "submission" not in info

        # 2. un campo de menos (falta "plan")
        p_menos = _payload_valido(note={"motivo": "Cefalea frontal pulsátil"})
        r2 = httpx.post(f"{url}/guardar", json=p_menos, timeout=5.0)
        assert r2.status_code == 400
        assert r2.json()["error"] == "la nota no cubre exactamente sus campos"
        assert hilo.is_alive()

        # 3. un campo de más ("extra")
        p_mas = _payload_valido(
            note={
                "motivo": "Cefalea frontal pulsátil",
                "plan": "Paracetamol",
                "extra": "Infiltrado",
            }
        )
        r3 = httpx.post(f"{url}/guardar", json=p_mas, timeout=5.0)
        assert r3.status_code == 400
        assert r3.json()["error"] == "la nota no cubre exactamente sus campos"
        assert hilo.is_alive()

        # 4. valor no cadena (entero)
        p_nocad = _payload_valido(note={"motivo": 12345, "plan": "Paracetamol"})
        r4 = httpx.post(f"{url}/guardar", json=p_nocad, timeout=5.0)
        assert r4.status_code == 400
        assert r4.json()["error"] == "la nota no cubre exactamente sus campos"
        assert hilo.is_alive()

    finally:
        # Guardado válido para cerrar limpiamente
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)


def test_collect_review_con_item_hallazgo_levanta_value_error_antes_de_abrir_puerto() -> None:
    """Caso 20 (§B.5): collect_review con un ítem de hallazgo -> ValueError
    antes de abrir el puerto."""
    item_finding = ReviewItem(
        kind=ItemKind.FINDING,
        ref="motivo:0",
        label="Motivo de consulta",
        text="Cefalea frontal",
        allowed=FINDING_VERDICTS,
    )
    nota = _crear_nota_prueba()
    settings = replace(load_ui_settings(), timeout_seconds=0.1)

    abierto = False

    def spy_open(_url: str) -> None:
        nonlocal abierto
        abierto = True

    with pytest.raises(
        ValueError,
        match="la pantalla solo confirma códigos: los hallazgos se leen en la nota",
    ):
        collect_review(
            (item_finding,),
            note=nota,
            history="Historia",
            transcript="Transcripción",
            settings=settings,
            open_browser=spy_open,
        )

    assert not abierto


def test_elapsed_seconds_usa_el_menor_entre_pagina_y_servidor() -> None:
    """Caso 21 (§B.5): elapsed_seconds usa min(declarado, medido) para que el
    navegador no pueda falsear la métrica principal."""
    items = _crear_codigos_prueba()
    t_actual = [100.0]

    def _reloj() -> float:
        return t_actual[0]

    url, _tok, _base, _port, hilo, info = _lanzar_servidor(items, clock=_reloj)
    # El servidor arrancó en t=100.0; avanzamos el reloj a 110.0 (10 segundos medidos)
    t_actual[0] = 110.0

    resp = httpx.post(
        f"{url}/guardar",
        json=_payload_valido(elapsed=99999.0),
        timeout=5.0,
    )
    assert resp.status_code == 200
    hilo.join(timeout=5.0)

    sub: ReviewSubmission = info["submission"]
    assert sub.elapsed_seconds == 10.0  # Menor entre 99999.0 y 10.0


def test_falta_una_ref_da_400_y_servidor_sigue_esperando() -> None:
    """Caso 21 (§B.5): Si falta una ref en answers, 400 y collect_review sigue
    esperando sin devolver a medias."""
    items = _crear_codigos_prueba()
    url, _tok, _base, _port, hilo, info = _lanzar_servidor(items)
    try:
        resp = httpx.post(
            f"{url}/guardar",
            json=_payload_valido(
                answers={
                    "I10X": {"verdict": "confirmed"}
                    # Falta "E119"
                }
            ),
            timeout=5.0,
        )
        assert resp.status_code == 400
        assert "no cubren exactamente" in resp.json()["error"]
        assert hilo.is_alive()
        assert "submission" not in info
    finally:
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)


def test_ref_desconocida_da_400() -> None:
    """Caso 21 (§B.5): Una ref desconocida en answers responde 400."""
    items = _crear_codigos_prueba()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items)
    try:
        resp = httpx.post(
            f"{url}/guardar",
            json=_payload_valido(
                answers={
                    "I10X": {"verdict": "confirmed"},
                    "E119": {"verdict": "confirmed"},
                    "desconocido:99": {"verdict": "confirmed"},
                }
            ),
            timeout=5.0,
        )
        assert resp.status_code == 400
    finally:
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)


def test_veredicto_no_admitido_para_item_da_400() -> None:
    """Caso 21 (§B.5): Un veredicto no admitido (ej. edited sobre código) responde 400."""
    items = _crear_codigos_prueba()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items)
    try:
        resp = httpx.post(
            f"{url}/guardar",
            json=_payload_valido(
                answers={
                    "I10X": {"verdict": "confirmed"},
                    "E119": {"verdict": "edited", "text": "Texto ilegal en código"},
                }
            ),
            timeout=5.0,
        )
        assert resp.status_code == 400
        assert "no admitido" in resp.json()["error"]
    finally:
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_se_apaga_tras_guardar() -> None:
    """Caso 21 (§B.5): Tras un guardado válido, el servidor se apaga; un segundo
    POST no conecta."""
    items = _crear_codigos_prueba()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items)
    resp = httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
    assert resp.status_code == 200
    hilo.join(timeout=5.0)

    # Segundo POST falla en conexión
    with pytest.raises(httpx.ConnectError):
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=1.0)


def test_expiracion_levanta_review_abandoned_y_libera_puerto() -> None:
    """Caso 21 (§B.5): Expiración de timeout_seconds levanta ReviewAbandoned
    y libera el puerto."""
    items = _crear_codigos_prueba()
    settings = replace(load_ui_settings(), timeout_seconds=0.05)

    _url, _tok, _base, puerto, hilo, info = _lanzar_servidor(items=items, settings=settings)
    hilo.join(timeout=5.0)

    assert "error" in info
    assert isinstance(info["error"], ReviewAbandoned)

    # El puerto quedó liberado
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{puerto}/", timeout=1.0)


def test_reviewer_from_levanta_ante_item_no_contestado() -> None:
    """Caso 21 (§B.5): reviewer_from levanta ante un ítem no contestado y devuelve
    la Answer exacta para los demás."""
    sub = ReviewSubmission(
        answers={"I10X": Answer(Verdict.CONFIRMED)},
        elapsed_seconds=12.0,
    )
    reviewer = reviewer_from(sub)

    c_ok = ReviewItem(ItemKind.CODE, "I10X", "I10X", "Hipertensión", CODE_VERDICTS)
    c_missing = ReviewItem(ItemKind.CODE, "E119", "E119", "Diabetes", CODE_VERDICTS)

    ans = reviewer(c_ok)
    assert ans == Answer(Verdict.CONFIRMED)

    with pytest.raises(ValueError, match="la pantalla no contestó al ítem 'E119'"):
        reviewer(c_missing)


def test_load_ui_settings_valida_claves_y_tipos(tmp_path: Path) -> None:
    """Caso 22 (§B.5): load_ui_settings exige las claves nuevas y ya no exige
    edit_label, section_reviewed_label ni missing_sections_text."""
    real = load_ui_settings()
    assert real.timeout_seconds == 1200.0
    assert len(real.wording.title) > 0
    assert len(real.wording.abandoned_text) > 0
    assert len(real.wording.history_heading) > 0
    assert len(real.wording.transcript_heading) > 0
    assert len(real.wording.note_heading) > 0
    assert len(real.wording.copy_label) > 0
    assert len(real.wording.copied_label) > 0

    base_yaml = (
        "title: 'T'\nintro: 'I'\nconfirm_label: 'C'\nreject_label: 'R'\n"
        "codes_heading: 'H'\nhistory_heading: 'Hist'\ntranscript_heading: 'Trans'\n"
        "note_heading: 'Nota'\ncopy_label: 'Copiar'\ncopied_label: 'Copiado'\n"
        "save_label: 'G'\nsaved_text: 'OK'\nabandoned_text: 'A'\ntimeout_seconds: 60.0\n"
    )

    # Carga correctamente sin edit_label, section_reviewed_label ni missing_sections_text
    p_ok = tmp_path / "ui_ok.yaml"
    p_ok.write_text(base_yaml, encoding="utf-8")
    loaded = load_ui_settings(p_ok)
    assert loaded.wording.title == "T"
    assert loaded.wording.history_heading == "Hist"

    def _probar(contenido: str, match: str) -> None:
        p = tmp_path / "ui_test.yaml"
        p.write_text(contenido, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_ui_settings(p)

    # Clave faltante de las nuevas
    _probar(base_yaml.replace("note_heading: 'Nota'\n", ""), "note_heading")
    _probar(base_yaml.replace("copy_label: 'Copiar'\n", ""), "copy_label")
    _probar(base_yaml.replace("copied_label: 'Copiado'\n", ""), "copied_label")
    # Timeout inválido
    _probar(base_yaml.replace("timeout_seconds: 60.0", "timeout_seconds: 0"), "timeout_seconds")
    _probar(base_yaml.replace("timeout_seconds: 60.0", "timeout_seconds: -10"), "timeout_seconds")
    _probar(base_yaml.replace("timeout_seconds: 60.0", "timeout_seconds: true"), "timeout_seconds")
    _probar(
        base_yaml.replace("timeout_seconds: 60.0", "timeout_seconds: 'diez'"),
        "timeout_seconds",
    )


def test_servidor_escucha_solo_en_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso 21 (§B.5): El servidor escucha solo en 127.0.0.1 (evita exposición)."""
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

    items = _crear_codigos_prueba()
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(items)
    try:
        assert len(bound_hosts) == 1
        assert bound_hosts[0] == "127.0.0.1"
    finally:
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)
