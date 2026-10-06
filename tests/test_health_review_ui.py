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
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.match import CodeOrigin, MatchedCode
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
    HistoryView,
    ReviewAbandoned,
    ReviewSubmission,
    UISettings,
    collect_review,
    history_view,
    load_ui_settings,
    parse_submission,
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


def _crear_matches_prueba(items: tuple[ReviewItem, ...] | None = None) -> tuple[MatchedCode, ...]:
    if items is None:
        items = _crear_codigos_prueba()
    origs = [CodeOrigin.TODAY, CodeOrigin.HISTORY, CodeOrigin.BOTH]
    return tuple(
        MatchedCode(
            diagnostic=DiagnosticCode(it.ref, it.text, 4, "catalogo"),
            score=1.0,
            origin=origs[i % len(origs)],
            evidence=f"evidencia {it.ref}",
        )
        for i, it in enumerate(items)
    )


def _lanzar_servidor(
    items: tuple[ReviewItem, ...] | None = None,
    note: RenderedNote | None = None,
    history: HistoryView | str = "Historia previa del paciente: sin antecedentes.",
    transcript: str = "Transcripción: el paciente refiere dolor.",
    settings: UISettings | None = None,
    clock: Any = None,
    matches: tuple[MatchedCode, ...] | None = None,
    owner: str = "sujeto-01",
    moment: datetime | None = None,
) -> tuple[str, str, str, int, threading.Thread, dict[str, Any]]:
    ajustes = settings if settings is not None else load_ui_settings()
    items_rev = items if items is not None else _crear_codigos_prueba()
    nota_rev = note if note is not None else _crear_nota_prueba()
    matches_rev = matches if matches is not None else _crear_matches_prueba(items_rev)
    hist_view = (
        history
        if isinstance(history, HistoryView)
        else HistoryView(content_type=None, data=b"", text=history)
    )
    mom = moment if moment is not None else datetime(2026, 10, 3, 10, 30)
    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    kwargs: dict[str, Any] = {
        "matches": matches_rev,
        "note": nota_rev,
        "transcript": transcript,
        "history": hist_view,
        "owner": owner,
        "moment": mom,
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
    """Caso 16 (§B.5 / §B.3): La página responde 200 con el token y contiene la historia,
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
        assert settings.wording.brand in html_content
        assert settings.wording.note_heading in html_content
        assert settings.wording.note_hint in html_content
        assert settings.wording.copy_all_label in html_content
        assert settings.wording.codes_tab in html_content
        assert settings.wording.transcript_tab in html_content
        assert settings.wording.history_tab in html_content
        assert settings.wording.save_label in html_content

        # Historia y transcripción escapadas
        assert html.escape(historia_texto) in html_content
        assert html.escape(transcripcion_texto) in html_content

        # Un textarea por campo con su texto
        assert "Cefalea frontal pulsátil</textarea>" in html_content
        assert "Paracetamol 500mg cada 8 horas</textarea>" in html_content

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
            matches=(),
            note=nota,
            transcript="Transcripción",
            history=HistoryView(content_type=None, data=b"", text="Historia"),
            owner="sujeto-01",
            moment=datetime(2026, 10, 3, 10, 30),
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
    """Caso 12 (§B.3): load_ui_settings exige las claves nuevas y ya no exige
    intro, confirm_label, reject_label ni codes_heading."""
    real = load_ui_settings()
    assert real.timeout_seconds == 1200.0
    assert len(real.wording.title) > 0
    assert len(real.wording.brand) > 0
    assert len(real.wording.note_heading) > 0
    assert len(real.wording.note_hint) > 0
    assert len(real.wording.copy_label) > 0
    assert len(real.wording.copied_label) > 0
    assert len(real.wording.copy_all_label) > 0
    assert len(real.wording.edited_label) > 0
    assert len(real.wording.empty_field_placeholder) > 0
    assert len(real.wording.codes_tab) > 0
    assert len(real.wording.transcript_tab) > 0
    assert len(real.wording.history_tab) > 0
    assert len(real.wording.origin_today_label) > 0
    assert len(real.wording.origin_history_label) > 0
    assert len(real.wording.origin_both_label) > 0
    assert len(real.wording.include_label) > 0
    assert len(real.wording.provenance_template) > 0
    assert len(real.wording.fallback_pill) > 0
    assert len(real.wording.edited_singular) > 0
    assert len(real.wording.edited_plural) > 0
    assert len(real.wording.codes_included) > 0
    assert len(real.wording.save_label) > 0
    assert len(real.wording.saved_text) > 0
    assert len(real.wording.abandoned_text) > 0

    base_yaml = (
        "brand: 'AIDA'\n"
        "title: 'T'\n"
        "note_heading: 'Nota'\n"
        "note_hint: 'Pista'\n"
        "copy_label: 'Copiar'\n"
        "copied_label: 'Copiado'\n"
        "copy_all_label: 'Copiar todo'\n"
        "edited_label: 'Editado'\n"
        "empty_field_placeholder: 'Vacío'\n"
        "codes_tab: 'Códigos'\n"
        "transcript_tab: 'Transcripción'\n"
        "history_tab: 'Historia clínica'\n"
        "origin_today_label: 'Hoy'\n"
        "origin_history_label: 'Previa'\n"
        "origin_both_label: 'Ambas'\n"
        "include_label: 'Incluir'\n"
        "provenance_template: '{} · {}'\n"
        "fallback_pill: 'Sin redacción'\n"
        "edited_singular: 'campo editado'\n"
        "edited_plural: 'campos editados'\n"
        "codes_included: 'de {} incluidos'\n"
        "save_label: 'Guardar'\n"
        "saved_text: 'Guardado'\n"
        "abandoned_text: 'Abandonado'\n"
        "timeout_seconds: 60.0\n"
    )

    # Carga correctamente sin intro, confirm_label, reject_label ni codes_heading
    p_ok = tmp_path / "ui_ok.yaml"
    p_ok.write_text(base_yaml, encoding="utf-8")
    loaded = load_ui_settings(p_ok)
    assert loaded.wording.title == "T"
    assert loaded.wording.brand == "AIDA"
    assert loaded.wording.note_heading == "Nota"

    def _probar(contenido: str, match: str) -> None:
        p = tmp_path / "ui_test.yaml"
        p.write_text(contenido, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_ui_settings(p)

    # Claves faltantes de las nuevas
    _probar(base_yaml.replace("note_heading: 'Nota'\n", ""), "note_heading")
    _probar(base_yaml.replace("copy_all_label: 'Copiar todo'\n", ""), "copy_all_label")
    _probar(base_yaml.replace("history_tab: 'Historia clínica'\n", ""), "history_tab")
    _probar(base_yaml.replace("provenance_template: '{} · {}'\n", ""), "provenance_template")
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


def test_history_view_tipos_mime(tmp_path: Path) -> None:
    """Caso 3 (§B.2, M2): history_view con .pdf -> application/pdf y data con bytes;
    con .png -> image/png; con .txt o None -> content_type is None y data == b""."""
    pdf_file = tmp_path / "historia.pdf"
    pdf_bytes = b"%PDF-1.4 test data"
    pdf_file.write_bytes(pdf_bytes)

    png_file = tmp_path / "historia.png"
    png_bytes = b"\x89PNG\r\n\x1a\n test png"
    png_file.write_bytes(png_bytes)

    txt_file = tmp_path / "historia.txt"
    txt_file.write_text("texto plano")

    # PDF -> application/pdf y bytes
    hv_pdf = history_view(pdf_file, "texto ocr")
    assert hv_pdf.content_type == "application/pdf"
    assert hv_pdf.data == pdf_bytes
    assert hv_pdf.text == "texto ocr"

    # PNG -> image/png y bytes
    hv_png = history_view(png_file, "texto ocr")
    assert hv_png.content_type == "image/png"
    assert hv_png.data == png_bytes

    # TXT -> content_type is None y data b""
    hv_txt = history_view(txt_file, "texto plano")
    assert hv_txt.content_type is None
    assert hv_txt.data == b""
    assert hv_txt.text == "texto plano"

    # None -> content_type is None y data b""
    hv_none = history_view(None, "texto sin archivo")
    assert hv_none.content_type is None
    assert hv_none.data == b""
    assert hv_none.text == "texto sin archivo"


def test_render_historia_pdf_imagen_y_texto(tmp_path: Path) -> None:
    """Caso 4 (§B.2): Con PDF, iframe src="/{token}/historia"; con imagen, img;
    con texto, pre.history-text con texto escapado."""
    pdf_file = tmp_path / "historia.pdf"
    pdf_file.write_bytes(b"%PDF-1.4...")
    hv_pdf = history_view(pdf_file, "texto ocr")

    url, tok, _base, _port, hilo, _info = _lanzar_servidor(history=hv_pdf)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert f'<iframe class="history-frame" src="/{tok}/historia"' in resp.text
    finally:
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)

    png_file = tmp_path / "historia.png"
    png_file.write_bytes(b"\x89PNG...")
    hv_png = history_view(png_file, "texto ocr")

    url2, tok2, _base2, _port2, hilo2, _info2 = _lanzar_servidor(history=hv_png)
    try:
        resp2 = httpx.get(url2, timeout=5.0)
        assert resp2.status_code == 200
        assert f'<img class="history-image" src="/{tok2}/historia"' in resp2.text
    finally:
        httpx.post(f"{url2}/guardar", json=_payload_valido(), timeout=5.0)
        hilo2.join(timeout=5.0)

    texto_raw = "Historia <peligrosa> & 'sensible'"
    hv_txt = HistoryView(content_type=None, data=b"", text=texto_raw)
    url3, _tok3, _base3, _port3, hilo3, _info3 = _lanzar_servidor(history=hv_txt)
    try:
        resp3 = httpx.get(url3, timeout=5.0)
        assert resp3.status_code == 200
        assert f'<pre class="history-text">{html.escape(texto_raw)}</pre>' in resp3.text
    finally:
        httpx.post(f"{url3}/guardar", json=_payload_valido(), timeout=5.0)
        hilo3.join(timeout=5.0)


def test_ruta_historia_con_token_y_sin_token(tmp_path: Path) -> None:
    """Caso 5 (§B.2, M3): GET /{token}/historia devuelve bytes y Content-Type;
    GET /historia (sin token) devuelve 404; con historia de texto devuelve 404."""
    pdf_file = tmp_path / "historia.pdf"
    pdf_bytes = b"%PDF-1.4 test bytes de historia clinica"
    pdf_file.write_bytes(pdf_bytes)
    hv_pdf = history_view(pdf_file, "texto")

    url, tok, base, _port, hilo, _info = _lanzar_servidor(history=hv_pdf)
    try:
        # Con token correcto -> 200, Content-Type application/pdf, y bytes
        resp = httpx.get(f"{base}/{tok}/historia", timeout=5.0)
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "application/pdf"
        assert resp.content == pdf_bytes

        # Sin token (/historia) -> 404
        assert httpx.get(f"{base}/historia", timeout=5.0).status_code == 404

        # Token erróneo -> 404
        assert httpx.get(f"{base}/token_malo/historia", timeout=5.0).status_code == 404
    finally:
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)

    # Con historia de texto (content_type is None) -> 404
    hv_txt = HistoryView(content_type=None, data=b"", text="solo texto")
    url_txt, tok_txt, base_txt, _port_txt, hilo_txt, _info_txt = _lanzar_servidor(history=hv_txt)
    try:
        assert httpx.get(f"{base_txt}/{tok_txt}/historia", timeout=5.0).status_code == 404
    finally:
        httpx.post(f"{url_txt}/guardar", json=_payload_valido(), timeout=5.0)
        hilo_txt.join(timeout=5.0)


def test_tarjetas_de_codigo_orden_y_atributos() -> None:
    """Caso 6 (§B.3, M5): Una tarjeta por código en el orden de matches, con data-ref,
    data-origin, etiqueta de origen, descripción y evidencia (o sin ella)."""
    settings = load_ui_settings()
    matches = (
        MatchedCode(
            diagnostic=DiagnosticCode("I10X", "Hipertensión esencial", 4, "catalogo"),
            score=1.0,
            origin=CodeOrigin.TODAY,
            evidence="PA 140/90",
        ),
        MatchedCode(
            diagnostic=DiagnosticCode("E119", "Diabetes mellitus tipo 2", 4, "catalogo"),
            score=1.0,
            origin=CodeOrigin.HISTORY,
            evidence="",
        ),
        MatchedCode(
            diagnostic=DiagnosticCode("K219", "Reflujo gastroesofágico", 4, "catalogo"),
            score=1.0,
            origin=CodeOrigin.BOTH,
            evidence="pirosis frecuente",
        ),
    )
    items = tuple(
        ReviewItem(
            ItemKind.CODE,
            m.diagnostic.code,
            m.diagnostic.code,
            m.diagnostic.description,
            CODE_VERDICTS,
        )
        for m in matches
    )
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(
        items=items, matches=matches, settings=settings
    )
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        html_content = resp.text

        # Orden estricto de matches
        pos_i10 = html_content.index('data-ref="I10X"')
        pos_e11 = html_content.index('data-ref="E119"')
        pos_k21 = html_content.index('data-ref="K219"')
        assert pos_i10 < pos_e11 < pos_k21

        w = settings.wording
        # Atributos de I10X
        assert 'data-origin="today"' in html_content
        assert f'<span class="origin">{w.origin_today_label}</span>' in html_content
        assert "Hipertensión esencial" in html_content
        assert "PA 140/90" in html_content

        # E119 sin evidencia: no hay párrafo de evidencia con #i-quote para E119
        assert 'data-origin="history"' in html_content
        assert f'<span class="origin">{w.origin_history_label}</span>' in html_content

        # K219 con BOTH
        assert 'data-origin="both"' in html_content
        assert f'<span class="origin">{w.origin_both_label}</span>' in html_content
    finally:
        httpx.post(
            f"{url}/guardar",
            json=_payload_valido(
                answers={
                    "I10X": {"verdict": "confirmed"},
                    "E119": {"verdict": "confirmed"},
                    "K219": {"verdict": "confirmed"},
                }
            ),
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_campos_nota_article_y_clase_empty() -> None:
    """Caso 7 (§B.3): Un article.field por campo con su textarea; campo vacío lleva clase empty."""
    spec_lleno = SectionSpec("lleno", "Lleno", (), codable=True, in_note=True)
    spec_vacio = SectionSpec("vacio", "Vacío", (), codable=True, in_note=True)
    spec_blanco = SectionSpec("blanco", "Blanco", (), codable=True, in_note=True)
    nota = RenderedNote(
        fields=(
            (spec_lleno, "Contenido del campo"),
            (spec_vacio, ""),
            (spec_blanco, "   \n  "),
        ),
        redacted=True,
        provider="openai/gpt-4.1-mini",
    )
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(note=nota)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        html_content = resp.text

        # Campo lleno no tiene clase empty
        assert (
            '<article class="field" data-field-id="lleno" '
            'data-original="Contenido del campo">' in html_content
        )
        assert "Contenido del campo</textarea>" in html_content

        # Campo vacío tiene clase empty
        assert (
            '<article class="field empty" data-field-id="vacio" '
            'data-original="">' in html_content
        )

        # Campo con solo espacios tiene clase empty
        assert (
            '<article class="field empty" data-field-id="blanco" '
            'data-original="   \n  ">' in html_content
        )
    finally:
        httpx.post(
            f"{url}/guardar",
            json=_payload_valido(
                note={
                    "lleno": "a",
                    "vacio": "b",
                    "blanco": "c",
                }
            ),
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_burbujas_transcripcion_preguntas_y_respuestas() -> None:
    """Caso 8 (§B.3): Burbujas: ? al final es class="q", otra es class="a";
    líneas vacías no se pintan."""
    transcript = (
        "¿Cómo ha seguido del dolor?\n"
        "\n"
        "Mucho mejor, doctor.\n"
        "   \n"
        "¿Se tomó las pastillas?\n"
        "Sí, todos los días."
    )
    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(transcript=transcript)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        html_content = resp.text

        # Preguntas -> class="q"
        assert '<p class="q">¿Cómo ha seguido del dolor?</p>' in html_content
        assert '<p class="q">¿Se tomó las pastillas?</p>' in html_content

        # Respuestas -> class="a"
        assert '<p class="a">Mucho mejor, doctor.</p>' in html_content
        assert '<p class="a">Sí, todos los días.</p>' in html_content

        # No hay párrafos vacíos
        assert '<p class="q"></p>' not in html_content
        assert '<p class="a"></p>' not in html_content
    finally:
        httpx.post(f"{url}/guardar", json=_payload_valido(), timeout=5.0)
        hilo.join(timeout=5.0)


def test_todo_se_escapa_contra_xss() -> None:
    """Caso 9 (§B.3, M4): Todo lo que entra de fuera se escapa (transcripción, evidencia,
    historia, nota). Ningún <script>alert(1) sin escapar en el HTML."""
    xss = "<script>alert(1)</script>"
    spec = SectionSpec("xss_field", "Campo XSS", (), codable=True, in_note=True)
    nota = RenderedNote(
        fields=((spec, f"Nota con {xss}"),),
        redacted=True,
        provider="openai/gpt-4.1-mini",
    )
    item = ReviewItem(ItemKind.CODE, "X01", "X01", "Desc", CODE_VERDICTS)
    match = MatchedCode(
        diagnostic=DiagnosticCode("X01", f"Desc {xss}", 4, "cat"),
        score=1.0,
        origin=CodeOrigin.TODAY,
        evidence=f"Evidencia con {xss}",
    )
    historia = f"Historia con {xss}"
    transcripcion = f"Transcripción con {xss}?"

    url, _tok, _base, _port, hilo, _info = _lanzar_servidor(
        items=(item,),
        matches=(match,),
        note=nota,
        history=historia,
        transcript=transcripcion,
    )
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert "<script>alert(1)" not in resp.text
        assert html.escape(xss) in resp.text
    finally:
        httpx.post(
            f"{url}/guardar",
            json=_payload_valido(
                answers={"X01": {"verdict": "confirmed"}},
                note={"xss_field": "ok"},
            ),
            timeout=5.0,
        )
        hilo.join(timeout=5.0)


def test_pildora_procedencia_redactada_y_fallback() -> None:
    """Caso 10 (§B.3, M7): Nota redactada -> class="pill" y proveedor/modelo;
    respaldo literal -> class="pill fallback" y fallback_pill."""
    settings = load_ui_settings()

    # 1. Nota redactada
    nota_redactada = RenderedNote(
        fields=(),
        redacted=True,
        provider="anthropic/claude-3-7-sonnet",
    )
    url1, _tok1, _base1, _port1, hilo1, _info1 = _lanzar_servidor(
        note=nota_redactada, settings=settings
    )
    try:
        resp1 = httpx.get(url1, timeout=5.0)
        assert resp1.status_code == 200
        assert '<span class="pill"><span class="dot"></span>' in resp1.text
        esperado_prov = settings.wording.provenance_template.format(
            "anthropic", "claude-3-7-sonnet"
        )
        assert esperado_prov in resp1.text
    finally:
        httpx.post(f"{url1}/guardar", json=_payload_valido(note={}), timeout=5.0)
        hilo1.join(timeout=5.0)

    # 2. Respaldo literal
    nota_fallback = RenderedNote(
        fields=(),
        redacted=False,
        provider=None,
    )
    url2, _tok2, _base2, _port2, hilo2, _info2 = _lanzar_servidor(
        note=nota_fallback, settings=settings
    )
    try:
        resp2 = httpx.get(url2, timeout=5.0)
        assert resp2.status_code == 200
        assert '<span class="pill fallback"><span class="dot"></span>' in resp2.text
        assert settings.wording.fallback_pill in resp2.text
    finally:
        httpx.post(f"{url2}/guardar", json=_payload_valido(note={}), timeout=5.0)
        hilo2.join(timeout=5.0)


def test_collect_review_items_y_matches_no_coinciden_levanta_value_error_antes_de_puerto() -> None:
    """Caso 11 (§B.3, M6): collect_review con items y matches que no coinciden
    (otro orden, o uno de más) -> ValueError antes de abrir el puerto."""
    c1 = ReviewItem(ItemKind.CODE, "I10X", "I10X", "Hipertensión", CODE_VERDICTS)
    c2 = ReviewItem(ItemKind.CODE, "E119", "E119", "Diabetes", CODE_VERDICTS)
    m1 = MatchedCode(DiagnosticCode("I10X", "Hipertensión", 4, "cat"), 1.0)
    m2 = MatchedCode(DiagnosticCode("E119", "Diabetes", 4, "cat"), 1.0)

    abierto = False

    def spy_open(_url: str) -> None:
        nonlocal abierto
        abierto = True

    nota = _crear_nota_prueba()
    settings = replace(load_ui_settings(), timeout_seconds=0.1)
    hist = HistoryView(content_type=None, data=b"", text="")
    mom = datetime(2026, 10, 3, 10, 30)

    # Orden distinto: (c1, c2) vs (m2, m1)
    with pytest.raises(ValueError, match="los ítems y los códigos no coinciden"):
        collect_review(
            (c1, c2),
            matches=(m2, m1),
            note=nota,
            transcript="",
            history=hist,
            owner="sujeto",
            moment=mom,
            settings=settings,
            open_browser=spy_open,
        )
    assert not abierto

    # Uno de más en items: (c1, c2) vs (m1,)
    with pytest.raises(ValueError, match="los ítems y los códigos no coinciden"):
        collect_review(
            (c1, c2),
            matches=(m1,),
            note=nota,
            transcript="",
            history=hist,
            owner="sujeto",
            moment=mom,
            settings=settings,
            open_browser=spy_open,
        )
    assert not abierto


def test_parse_submission_valida_y_calcula_menor_tiempo() -> None:
    """Caso 1 y 2 (§B.1): parse_submission acepta envíos correctos y fija
    elapsed_seconds al mínimo entre el declarado y el medido."""
    items = _crear_codigos_prueba()
    nota = _crear_nota_prueba()
    payload = _payload_valido(elapsed=5.0)

    # 1. Declarado menor que medido
    sub, err = parse_submission(payload, items=items, note=nota, measured_elapsed=10.0)
    assert err == ""
    assert sub is not None
    assert sub.elapsed_seconds == 5.0
    assert set(sub.answers.keys()) == {"I10X", "E119"}
    assert set(sub.note_fields.keys()) == {"motivo", "plan"}

    # 2. Medido menor que declarado
    sub2, err2 = parse_submission(payload, items=items, note=nota, measured_elapsed=3.0)
    assert err2 == ""
    assert sub2 is not None
    assert sub2.elapsed_seconds == 3.0


@pytest.mark.parametrize(
    ("payload_invalido", "mensaje_esperado"),
    [
        (
            {
                "answers": "no-es-dict",
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": 1.0,
            },
            "answers debe ser un mapeo",
        ),
        (
            {
                "answers": {"I10X": {"verdict": "confirmed"}},
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": 1.0,
            },
            "las respuestas no cubren exactamente los ítems de la revisión",
        ),
        (
            {
                "answers": {
                    "I10X": {"verdict": "confirmed"},
                    "E119": {"verdict": "confirmed"},
                    "EXTRA": {"verdict": "confirmed"},
                },
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": 1.0,
            },
            "las respuestas no cubren exactamente los ítems de la revisión",
        ),
        (
            {
                "answers": {"I10X": "no-dict", "E119": {"verdict": "confirmed"}},
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": 1.0,
            },
            "respuesta inválida para 'I10X'",
        ),
        (
            {
                "answers": {
                    "I10X": {"verdict": "desconocido"},
                    "E119": {"verdict": "confirmed"},
                },
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": 1.0,
            },
            "veredicto desconocido para 'I10X'",
        ),
        (
            {
                "answers": {
                    "I10X": {"verdict": "edited"},
                    "E119": {"verdict": "confirmed"},
                },
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": 1.0,
            },
            "veredicto 'edited' no admitido para 'I10X'",
        ),
        (
            {
                "answers": {
                    "I10X": {"verdict": "confirmed", "text": "sobra"},
                    "E119": {"verdict": "confirmed"},
                },
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": 1.0,
            },
            "texto sin edición para 'I10X'",
        ),
        (
            {
                "answers": {"I10X": {"verdict": "confirmed"}, "E119": {"verdict": "confirmed"}},
                "note": "no-es-dict",
                "elapsed_seconds": 1.0,
            },
            "la nota no cubre exactamente sus campos",
        ),
        (
            {
                "answers": {"I10X": {"verdict": "confirmed"}, "E119": {"verdict": "confirmed"}},
                "note": {"motivo": "m"},
                "elapsed_seconds": 1.0,
            },
            "la nota no cubre exactamente sus campos",
        ),
        (
            {
                "answers": {"I10X": {"verdict": "confirmed"}, "E119": {"verdict": "confirmed"}},
                "note": {"motivo": "m", "plan": "p", "sobra": "s"},
                "elapsed_seconds": 1.0,
            },
            "la nota no cubre exactamente sus campos",
        ),
        (
            {
                "answers": {"I10X": {"verdict": "confirmed"}, "E119": {"verdict": "confirmed"}},
                "note": {"motivo": 123, "plan": "p"},
                "elapsed_seconds": 1.0,
            },
            "la nota no cubre exactamente sus campos",
        ),
        (
            {
                "answers": {"I10X": {"verdict": "confirmed"}, "E119": {"verdict": "confirmed"}},
                "note": {"motivo": "m", "plan": "p"},
            },
            "elapsed_seconds inválido",
        ),
        (
            {
                "answers": {"I10X": {"verdict": "confirmed"}, "E119": {"verdict": "confirmed"}},
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": -0.5,
            },
            "elapsed_seconds inválido",
        ),
        (
            {
                "answers": {"I10X": {"verdict": "confirmed"}, "E119": {"verdict": "confirmed"}},
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": True,
            },
            "elapsed_seconds inválido",
        ),
        (
            {
                "answers": {"I10X": {"verdict": "confirmed"}, "E119": {"verdict": "confirmed"}},
                "note": {"motivo": "m", "plan": "p"},
                "elapsed_seconds": "1.0",
            },
            "elapsed_seconds inválido",
        ),
    ],
)
def test_parse_submission_rechazos(
    payload_invalido: dict[str, Any],
    mensaje_esperado: str,
) -> None:
    """Caso 1 (§B.1): parse_submission rechaza respuestas incompletas, veredictos
    no admitidos, notas que no cubren exactamente sus campos y tiempos inválidos
    con los mismos mensajes que collect_review."""
    items = _crear_codigos_prueba()
    nota = _crear_nota_prueba()
    sub, err = parse_submission(payload_invalido, items=items, note=nota, measured_elapsed=10.0)
    assert sub is None
    assert err == mensaje_esperado


def test_parse_submission_edicion_sin_texto_en_item_que_la_admite() -> None:
    """parse_submission rechaza veredicto EDITED sin texto cuando el ítem lo admite."""
    item = ReviewItem(
        kind=ItemKind.FINDING,
        ref="motivo:0",
        label="motivo",
        text="Cefalea",
        allowed=FINDING_VERDICTS,
    )
    nota = _crear_nota_prueba()
    payload = {
        "answers": {"motivo:0": {"verdict": "edited"}},
        "note": {"motivo": "m", "plan": "p"},
        "elapsed_seconds": 1.0,
    }
    sub, err = parse_submission(payload, items=(item,), note=nota, measured_elapsed=10.0)
    assert sub is None
    assert err == "edición sin texto para 'motivo:0'"

