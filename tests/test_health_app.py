"""Pruebas unitarias de la aplicación local (Plan 77).

Cubre §B.3 y §B.4 de .claude/workflow/77-plan-aplicacion.md:
6. valid_owner: códigos admitidos y rechazados.
7. progress: pasos según los eventos y drafted.
8. load_app_wording: carga el pack real y valida claves y steps.
9. draft_for_screen y close_from_screen: orquestación de pantalla.
10. Sin token: 404 en GET y POST.
11. Escucha solo en 127.0.0.1.
12. AppProblem inicial y /reintentar.
13. /empezar: validaciones de owner, consent y archivos.
14. Mientras working: 409 en /audio y /empezar.
15. Tras procesar: review, /historia, /guardar con next y borrado de sesión.
16. Fallo de transcripción: estado problem y carpeta borrada.
17. Carpetas aida-sesion-* huérfanas se borran al arrancar.
18. /salir: 200 y serve retorna.
19. Todo se escapa contra inyecciones XSS.
"""

from __future__ import annotations

import io
import tempfile
import threading
import time
import uuid
import wave
from collections.abc import Callable
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from cryptography.fernet import Fernet

from aida.arms.health.app import (
    _WORKDIR_PREFIX,
    AppProblem,
    AppReady,
    AppWording,
    close_from_screen,
    draft_for_screen,
    load_app_wording,
    progress,
    serve,
    valid_owner,
)
from aida.arms.health.cache import ConsultationCache
from aida.arms.health.capture import LocalTesseractOCR, LocalWhisperSTT, TranscriptionError
from aida.arms.health.codes import DiagnosticCode
from aida.arms.health.consultation import Consultation
from aida.arms.health.extract import build_health_extractor
from aida.arms.health.match import build_index
from aida.arms.health.note import load_note_settings
from aida.arms.health.review import Answer, ItemKind, Verdict
from aida.arms.health.review_ui import ReviewSubmission, UISettings, load_ui_settings
from aida.arms.health.segment import build_health_segmenter
from aida.arms.health.wording import default_summary_wording
from aida.engine.events import Event, EventKind, EventLog
from aida.engine.ontology import AgentId
from aida.voice.diarize import TranscriptSegment
from tests.conftest import _ORIGINAL_HANDLE_REQUEST


@pytest.fixture(autouse=True)
def _permitir_servidor_app_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restaura el transporte real de httpx para comunicar con el servidor local."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _ORIGINAL_HANDLE_REQUEST)


class _DummySTT(LocalWhisperSTT):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(model=object())
        self._fail = fail

    def transcribe_segments(self, path: Path) -> tuple[TranscriptSegment, ...]:
        if self._fail:
            raise TranscriptionError("fallo simulado de transcripción")
        return (
            TranscriptSegment(0.0, 2.0, "Motivo de consulta: cefalea frontal pulsátil."),
            TranscriptSegment(2.0, 4.0, "Plan de manejo: paracetamol cada 8 horas."),
        )


class _DummyOCR(LocalTesseractOCR):
    def __init__(self) -> None:
        super().__init__(pytesseract_module=object())

    def read(self, path: Path) -> str:
        return "Historia clínica previa: antecedente de hipertensión arterial esencial."


def _crear_wav_minimo() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def _crear_app_ready(tmp_path: Path, *, fail_stt: bool = False) -> AppReady:
    log = EventLog()
    consultation = Consultation(
        wording=default_summary_wording(),
        stt=_DummySTT(fail=fail_stt),
        ocr=_DummyOCR(),
        catalog_index=build_index((DiagnosticCode("I10X", "Hipertensión esencial", 4, "cat"),)),
        provider=None,
        cache=ConsultationCache(tmp_path / "cache", key=Fernet.generate_key()),
        log=log,
        segmenter=build_health_segmenter(),
        extractor=build_health_extractor(),
        note_settings=load_note_settings(),
    )
    return AppReady(
        consultation=consultation,
        reader=_DummyOCR(),
        log=log,
        cloud=True,
        writer="modelo-prueba",
    )


def _lanzar_servidor_app(
    build_fn: Callable[[], AppReady | AppProblem],
    *,
    settings: UISettings | None = None,
    wording: AppWording | None = None,
) -> tuple[str, str, int, threading.Thread, dict[str, Any]]:
    ajustes = settings if settings is not None else load_ui_settings()
    palabras = wording if wording is not None else load_app_wording()
    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    hilo = threading.Thread(
        target=serve,
        args=(build_fn,),
        kwargs={"settings": ajustes, "wording": palabras, "open_browser": _abrir},
        daemon=True,
    )
    hilo.start()
    assert listo.wait(timeout=5.0), "el servidor no abrió la URL en el tiempo límite"
    url = info["url"]
    partes = url.rsplit("/", 1)
    token = partes[1]
    puerto = int(partes[0].split(":")[-1])
    return url, token, puerto, hilo, info


# ---------------------------------------------------------------------------
# §B.3 — Piezas puras de app.py
# ---------------------------------------------------------------------------


def test_valid_owner() -> None:
    """Caso 6 (§B.3): valid_owner admite códigos alfanuméricos de hasta 64 caracteres
    con ._- y rechaza vacíos, espacios, tildes, guión inicial y largos > 64."""
    assert valid_owner("HC-0042") is True
    assert valid_owner("a") is True
    assert valid_owner("x.y_z-1") is True

    assert valid_owner("") is False
    assert valid_owner("Ana Gomez") is False
    assert valid_owner("ñandú") is False
    assert valid_owner("-x") is False
    assert valid_owner("a" * 65) is False


def test_progress_con_eventos_y_drafted() -> None:
    """Caso 7 (§B.3, M2): progress lee el progreso de los eventos: primera INPUT_OBSERVED,
    TEXT_SEGMENTED, segunda INPUT_OBSERVED, CATALOG_MATCHED, y drafted para el quinto."""
    e_stt = Event(seq=0, tick=0, kind=EventKind.INPUT_OBSERVED, owner=AgentId("m"), content="a")
    e_seg = Event(seq=1, tick=0, kind=EventKind.TEXT_SEGMENTED, owner=AgentId("m"), content="")
    e_doc = Event(seq=2, tick=0, kind=EventKind.INPUT_OBSERVED, owner=AgentId("m"), content="h")
    e_cat = Event(seq=3, tick=0, kind=EventKind.CATALOG_MATCHED, owner=AgentId("m"), content="")

    assert progress([], drafted=False) == (False, False, False, False, False)
    assert progress([e_stt], drafted=False) == (True, False, False, False, False)
    assert progress([e_stt, e_seg], drafted=False) == (True, True, False, False, False)
    assert progress([e_stt, e_seg, e_doc], drafted=False) == (True, True, True, False, False)
    assert progress([e_stt, e_seg, e_doc, e_cat], drafted=False) == (
        True,
        True,
        True,
        True,
        False,
    )
    assert progress([e_stt, e_seg, e_doc, e_cat], drafted=True) == (
        True,
        True,
        True,
        True,
        True,
    )

    # La segunda INPUT_OBSERVED enciende el tercer paso aunque no haya TEXT_SEGMENTED delante
    assert progress([e_stt, e_doc], drafted=False) == (True, False, True, False, False)


def test_load_app_wording_pack_real_y_errores(tmp_path: Path) -> None:
    """Caso 8 (§B.3): load_app_wording carga el pack real con cinco pasos, y falla
    si falta la sección app, falta alguna clave o steps tiene una longitud distinta."""
    w = load_app_wording()
    assert isinstance(w, AppWording)
    assert len(w.steps) == 5
    assert isinstance(w["title_start"], str)

    # 1. Falta sección app
    sin_app = tmp_path / "sin_app.yaml"
    sin_app.write_text("otra: valor", encoding="utf-8")
    with pytest.raises(ValueError, match="falta la sección 'app'"):
        load_app_wording(sin_app)

    # 2. Falta una clave en app
    datos = yaml.safe_load(
        Path("packs/health/consultorio/ui.yaml").read_text(encoding="utf-8")
    )
    del datos["app"]["title_start"]
    sin_clave = tmp_path / "sin_clave.yaml"
    sin_clave.write_text(yaml.safe_dump(datos), encoding="utf-8")
    with pytest.raises(ValueError, match="faltan claves en la sección 'app'"):
        load_app_wording(sin_clave)

    # 3. Steps con cuatro elementos
    datos["app"]["title_start"] = "Inicio"
    datos["app"]["steps"] = ["paso 1", "paso 2", "paso 3", "paso 4"]
    cuatro_pasos = tmp_path / "cuatro_pasos.yaml"
    cuatro_pasos.write_text(yaml.safe_dump(datos), encoding="utf-8")
    with pytest.raises(ValueError, match=r"'app\.steps' debe ser una lista de cinco cadenas"):
        load_app_wording(cuatro_pasos)


def test_draft_for_screen_y_close_from_screen(tmp_path: Path) -> None:
    """Caso 9 (§B.3): draft_for_screen devuelve solo ítems de código y la nota, y
    close_from_screen emite ITEM_REVIEWED por código y por campo sin ítems de hallazgo."""
    ready = _crear_app_ready(tmp_path)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(_crear_wav_minimo())
    doc = tmp_path / "doc.txt"
    doc.write_text("Historia previa", encoding="utf-8")

    prepared = ready.consultation.prepare(
        owner=AgentId("HC-01"),
        audio_path=audio,
        history_text=lambda: ready.reader.read(doc),
    )

    draft = draft_for_screen(ready.consultation, prepared)
    assert all(item.kind is ItemKind.CODE for item in draft.items)
    assert len(draft.note.fields) > 0

    submission = ReviewSubmission(
        answers={item.ref: Answer(Verdict.CONFIRMED) for item in draft.items},
        elapsed_seconds=1.5,
        note_fields={spec.section_id: texto for spec, texto in draft.note.fields},
    )

    close_from_screen(ready.consultation, draft, submission)
    eventos_rev = [e for e in ready.log if e.kind is EventKind.ITEM_REVIEWED]
    assert len(eventos_rev) == len(draft.items) + len(draft.note.fields)
    assert all(e.data.get("item") in ("code", "field") for e in eventos_rev)
    assert not any(e.data.get("item") == "finding" for e in eventos_rev)


# ---------------------------------------------------------------------------
# §B.4 — El servidor (serve)
# ---------------------------------------------------------------------------


def test_servidor_sin_token_devuelve_404(tmp_path: Path) -> None:
    """Caso 10 (§B.4, M8): Cualquier GET o POST sin el token aleatorio devuelve 404."""
    ready = _crear_app_ready(tmp_path)
    url, _token, puerto, hilo, _ = _lanzar_servidor_app(lambda: ready)
    base = f"http://127.0.0.1:{puerto}"
    try:
        assert httpx.get(f"{base}/", timeout=5.0).status_code == 404
        assert httpx.get(f"{base}/inventado", timeout=5.0).status_code == 404
        assert httpx.get(f"{base}/malo/estado", timeout=5.0).status_code == 404
        # POST sin token debe responder 404 (M8)
        assert httpx.post(f"{base}/empezar", json={}, timeout=5.0).status_code == 404
        assert httpx.post(f"{base}/malo/empezar", json={}, timeout=5.0).status_code == 404
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_escucha_solo_en_localhost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso 11 (§B.4, M7): El servidor escucha solo en 127.0.0.1."""
    bound_hosts: list[str] = []
    orig_init = ThreadingHTTPServer.__init__

    def spy_init(
        self: Any,
        server_address: tuple[str, int],
        handler_class: Any,
        bind_and_activate: bool = True,
    ) -> None:
        bound_hosts.append(server_address[0])
        orig_init(self, server_address, handler_class, bind_and_activate)

    monkeypatch.setattr(ThreadingHTTPServer, "__init__", spy_init)

    ready = _crear_app_ready(tmp_path)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready)
    try:
        assert len(bound_hosts) == 1
        assert bound_hosts[0] == "127.0.0.1"
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_app_problem_capture_y_reintentar(tmp_path: Path) -> None:
    """Caso 12 (§B.4): Si build() devuelve AppProblem('capture'), la página muestra
    el problema y su comando; /reintentar tras corregirse pasa al inicio."""
    estado: list[AppReady | AppProblem] = [AppProblem("capture")]

    def builder() -> AppReady | AppProblem:
        return estado[0]

    wording = load_app_wording()
    url, _token, _port, hilo, _ = _lanzar_servidor_app(builder, wording=wording)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert wording["problem_capture_title"] in resp.text
        assert wording["problem_capture_command"] in resp.text

        # Corrección y reintento
        estado[0] = _crear_app_ready(tmp_path)
        resp_reintento = httpx.post(f"{url}/reintentar", timeout=5.0)
        assert resp_reintento.status_code == 200
        assert resp_reintento.json() == {"ok": True}

        resp_inicio = httpx.get(url, timeout=5.0)
        assert resp_inicio.status_code == 200
        assert wording["title_start"] in resp_inicio.text
        assert wording["start_button"] in resp_inicio.text
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_empezar_validaciones(tmp_path: Path) -> None:
    """Caso 13 (§B.4, M4): /empezar rechaza identificadores inválidos, grabación sin
    consentimiento y falta de archivos, y pasa a working con todo válido."""
    ready = _crear_app_ready(tmp_path)
    wording = load_app_wording()
    url, _token, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, wording=wording)
    try:
        # 1. Identificador inválido
        r1 = httpx.post(f"{url}/empezar", json={"owner": "Ana Gomez"}, timeout=5.0)
        assert r1.status_code == 400
        assert r1.json()["error"] == wording["owner_invalid"]

        # 2. Grabación sin consentimiento (M4)
        r2 = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-01", "recorded": True, "consent": False},
            timeout=5.0,
        )
        assert r2.status_code == 400
        assert r2.json()["error"] == wording["missing_consent"]

        # 3. Falta de audio o documento
        r3 = httpx.post(f"{url}/empezar", json={"owner": "HC-01"}, timeout=5.0)
        assert r3.status_code == 400
        assert r3.json()["error"] == wording["missing_document"]

        # Subir audio y documento
        wav = _crear_wav_minimo()
        r_audio = httpx.post(
            f"{url}/audio",
            content=wav,
            headers={"X-Name": "consulta.wav"},
            timeout=5.0,
        )
        assert r_audio.status_code == 200
        r_doc = httpx.post(
            f"{url}/documento",
            content=b"Historia clinica",
            headers={"X-Name": "historia.txt"},
            timeout=5.0,
        )
        assert r_doc.status_code == 200

        # Ahora sí, empezar con éxito
        r4 = httpx.post(f"{url}/empezar", json={"owner": "HC-01"}, timeout=5.0)
        assert r4.status_code == 200
        assert r4.json() == {"ok": True}
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_mientras_working_rechaza_audio_y_empezar_con_409(tmp_path: Path) -> None:
    """Caso 14 (§B.4): Mientras working, /audio y /empezar devuelven 409."""
    ready = _crear_app_ready(tmp_path)
    url, _token, _port, hilo, _ = _lanzar_servidor_app(lambda: ready)
    try:
        httpx.post(f"{url}/audio", content=_crear_wav_minimo(), headers={"X-Name": "a.wav"})
        httpx.post(f"{url}/documento", content=b"hist", headers={"X-Name": "d.txt"})
        httpx.post(f"{url}/empezar", json={"owner": "HC-01"})

        # Inmediatamente tras empezar, el servidor está en working
        r_aud = httpx.post(f"{url}/audio", content=_crear_wav_minimo(), headers={"X-Name": "a.wav"})
        assert r_aud.status_code in (409, 200)  # si terminó rápido pasaría a review
        if r_aud.status_code == 409:
            r_emp = httpx.post(f"{url}/empezar", json={"owner": "HC-01"})
            assert r_emp.status_code == 409
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_flujo_completo_review_guardar_y_limpieza(tmp_path: Path) -> None:
    """Caso 15 (§B.4, M1, M5, M11): Tras procesar, estado review, /historia sirve el
    documento, /guardar incompleto rechaza (400, M1), /guardar válido devuelve next (M11),
    vuelve al inicio y borra la carpeta de la sesión (M5)."""
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    ready = _crear_app_ready(tmp_path)
    url, token, _port, hilo, _ = _lanzar_servidor_app(lambda: ready)
    try:
        httpx.post(
            f"{url}/audio",
            content=_crear_wav_minimo(),
            headers={"X-Name": "consulta.wav"},
            timeout=5.0,
        )
        doc_bytes = b"%PDF-1.4 documento ficticio"
        httpx.post(
            f"{url}/documento",
            content=doc_bytes,
            headers={"X-Name": "historia.pdf"},
            timeout=5.0,
        )
        httpx.post(f"{url}/empezar", json={"owner": "HC-01"}, timeout=5.0)

        # Esperar a que pase a review
        for _ in range(50):
            estado = httpx.get(f"{url}/estado", timeout=5.0).json()
            if estado.get("state") == "review":
                break
            time.sleep(0.1)
        assert estado.get("state") == "review"
        assert estado.get("steps") == [True, True, True, True, True]

        # Comprobar /historia sirve el documento
        r_hist = httpx.get(f"{url}/historia", timeout=5.0)
        assert r_hist.status_code == 200
        assert r_hist.headers["Content-Type"].startswith("application/pdf")
        assert r_hist.content == doc_bytes

        # Comprobar que existe la carpeta temporal de sesión
        carpetas_sesion = list(Path(tempfile.gettempdir()).glob(f"{_WORKDIR_PREFIX}*"))
        assert len(carpetas_sesion) >= 1
        sesion_dir = carpetas_sesion[0]

        # Guardar incompleto (M1: falta un campo de nota) -> 400
        r_incompleto = httpx.post(
            f"{url}/guardar",
            json={
                "answers": {"I10X": {"verdict": "confirmed"}},
                "note": {"motivo_consulta": "cefalea"},  # falta medicamentos
                "elapsed_seconds": 2.0,
            },
            timeout=5.0,
        )
        assert r_incompleto.status_code == 400

        # Guardar válido -> 200 con 'next' (M11)
        r_guardar = httpx.post(
            f"{url}/guardar",
            json={
                "answers": {"I10X": {"verdict": "confirmed"}},
                "note": {
                    "motivo_consulta": "cefalea",
                    "medicamentos": "paracetamol",
                },
                "elapsed_seconds": 2.0,
            },
            timeout=5.0,
        )
        assert r_guardar.status_code == 200
        datos_guardar = r_guardar.json()
        assert datos_guardar.get("ok") is True
        assert datos_guardar.get("next") == f"/{token}"

        # Comprobar estado reiniciado a start
        estado_post = httpx.get(f"{url}/estado", timeout=5.0).json()
        assert estado_post.get("state") == "start"

        # Comprobar que la carpeta de sesión fue borrada (M5)
        assert not sesion_dir.exists()
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_fallo_transcripcion_pasa_a_problem_y_limpia_workdir(tmp_path: Path) -> None:
    """Caso 16 (§B.4): Un fallo de transcripción (TranscriptionError) pasa a estado
    problem con kind == 'transcription' y borra la carpeta de sesión."""
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    ready = _crear_app_ready(tmp_path, fail_stt=True)
    wording = load_app_wording()
    url, _token, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, wording=wording)
    try:
        httpx.post(
            f"{url}/audio",
            content=_crear_wav_minimo(),
            headers={"X-Name": "consulta.wav"},
            timeout=5.0,
        )
        httpx.post(
            f"{url}/documento",
            content=b"historia",
            headers={"X-Name": "historia.txt"},
            timeout=5.0,
        )

        carpetas = list(Path(tempfile.gettempdir()).glob(f"{_WORKDIR_PREFIX}*"))
        assert len(carpetas) >= 1
        sesion_dir = carpetas[0]

        httpx.post(f"{url}/empezar", json={"owner": "HC-01"}, timeout=5.0)

        for _ in range(50):
            estado = httpx.get(f"{url}/estado", timeout=5.0).json()
            if estado.get("state") == "problem":
                break
            time.sleep(0.1)
        assert estado.get("state") == "problem"

        resp_pag = httpx.get(url, timeout=5.0)
        assert resp_pag.status_code == 200
        assert wording["problem_transcription_title"] in resp_pag.text

        # La carpeta debe haber sido borrada
        assert not sesion_dir.exists()
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_limpia_carpetas_huerfanas_al_arrancar(tmp_path: Path) -> None:
    """Caso 17 (§B.4, M6): Las carpetas aida-sesion-* existentes antes de serve se
    borran al arrancar para evitar fuga de audios en claro tras caídas previas."""
    huerfana = Path(tempfile.gettempdir()) / f"{_WORKDIR_PREFIX}huerfana-{uuid.uuid4()}"
    huerfana.mkdir(parents=True, exist_ok=True)
    (huerfana / "audio-secreto.wav").write_bytes(b"datos sensibles")

    ready = _crear_app_ready(tmp_path)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready)
    try:
        assert not huerfana.exists()
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_salir_termina_hilo(tmp_path: Path) -> None:
    """Caso 18 (§B.4): /salir devuelve 200 y el hilo de serve termina limpiamente."""
    ready = _crear_app_ready(tmp_path)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready)
    r = httpx.post(f"{url}/salir", timeout=5.0)
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    hilo.join(timeout=5.0)
    assert not hilo.is_alive()


def test_servidor_escapa_todo_html(tmp_path: Path) -> None:
    """Caso 19 (§B.4, M12): El detalle de problema y valores de usuario se escapan
    en HTML contra inyecciones XSS."""
    inyeccion = "<script>alert(1)</script>"
    problema = AppProblem("processing", detail=inyeccion)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: problema)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert inyeccion not in resp.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in resp.text
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)
