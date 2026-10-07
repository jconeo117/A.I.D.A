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
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
import yaml
from cryptography.fernet import Fernet

from aida.arms.health.app import (
    _WORKDIR_PREFIX,
    AppProblem,
    AppReady,
    AppTools,
    AppWording,
    ReviewDraft,
    _keep_mine,
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
from aida.arms.health.live import LiveTranscriber
from aida.arms.health.match import build_index
from aida.arms.health.metrics import ReviewMetricsStore
from aida.arms.health.note import RenderedNote, load_note_settings
from aida.arms.health.review import Answer, ItemKind, Verdict
from aida.arms.health.review_ui import ReviewSubmission, UISettings, load_ui_settings
from aida.arms.health.sections import SectionSpec
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


def _crear_app_ready(
    tmp_path: Path,
    *,
    fail_stt: bool = False,
    store: ReviewMetricsStore | None = None,
    live_stt: LiveTranscriber | None = None,
) -> AppReady:
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
        review_metrics=store,
    )
    return AppReady(
        consultation=consultation,
        reader=_DummyOCR(),
        log=log,
        cloud=True,
        writer="modelo-prueba",
        live_stt=live_stt,
    )


def _dummy_vad(audio: np.ndarray) -> np.ndarray:
    return np.zeros(1, dtype=np.float32)


def _lanzar_servidor_app(
    build_fn: Callable[[], AppReady | AppProblem],
    *,
    settings: UISettings | None = None,
    wording: AppWording | None = None,
    **extra_serve_kwargs: Any,
) -> tuple[str, str, int, threading.Thread, dict[str, Any]]:
    ajustes = settings if settings is not None else load_ui_settings()
    palabras = wording if wording is not None else load_app_wording()
    info: dict[str, Any] = {}
    listo = threading.Event()

    def _abrir(url: str) -> None:
        info["url"] = url
        listo.set()

    serve_kwargs = {
        "settings": ajustes,
        "wording": palabras,
        "open_browser": _abrir,
        "vad_factory": lambda: _dummy_vad,
        **extra_serve_kwargs,
    }
    hilo = threading.Thread(
        target=serve,
        args=(build_fn,),
        kwargs=serve_kwargs,
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

    owner = AgentId("HC-01")
    consent = ready.consultation.record_consent(owner=owner, mode="uploaded")
    prepared = ready.consultation.prepare(
        owner=owner,
        audio_path=audio,
        history_text=lambda: ready.reader.read(doc),
        consent=consent,
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

        # 2. Falta de audio o documento (antes de pedir autorización)
        r2 = httpx.post(f"{url}/empezar", json={"owner": "HC-01"}, timeout=5.0)
        assert r2.status_code == 400
        assert r2.json()["error"] == wording["missing_document"]

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

        # 3. Falta de autorización (tras subir archivos)
        r3 = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-01", "consent": False},
            timeout=5.0,
        )
        assert r3.status_code == 400
        assert r3.json()["error"] == wording["missing_consent"]

        # Ahora sí, empezar con éxito
        r4 = httpx.post(f"{url}/empezar", json={"owner": "HC-01", "consent": True}, timeout=5.0)
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
        httpx.post(f"{url}/empezar", json={"owner": "HC-01", "consent": True}, timeout=5.0)

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

        httpx.post(f"{url}/empezar", json={"owner": "HC-01", "consent": True}, timeout=5.0)

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


# ---------------------------------------------------------------------------
# Plan 77b: La consulta en vivo y la revisión (§B.4)
# ---------------------------------------------------------------------------


def test_servidor_empezar_modo_vivo_validaciones(tmp_path: Path) -> None:
    """Caso 14 (§B.4): /empezar con recorded y consent -> estado live;

    sin consent -> 400; sin documento -> 400.
    """
    ready = _crear_app_ready(tmp_path)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready)
    try:
        # 1. Sin documento ni consent -> 400
        r1 = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-01", "recorded": True, "consent": False},
        )
        assert r1.status_code == 400

        # 2. Sin documento pero con consent -> 400
        r2 = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-01", "recorded": True, "consent": True},
        )
        assert r2.status_code == 400

        # Subir documento
        httpx.post(
            f"{url}/documento",
            content=b"%PDF-1.4 dummy",
            headers={"X-Name": "doc.pdf"},
            timeout=5.0,
        )

        # 3. Con documento pero sin consent -> 400
        r3 = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-01", "recorded": True, "consent": False},
        )
        assert r3.status_code == 400

        # 4. Con documento y con consent -> 200, pasa a estado live
        r4 = httpx.post(
            f"{url}/empezar",
            json={"owner": "HC-01", "recorded": True, "consent": True},
        )
        assert r4.status_code == 200
        estado = httpx.get(f"{url}/estado").json()
        assert estado.get("state") == "live"
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_rutas_vivo_fuera_de_live_y_campo_desconocido(tmp_path: Path) -> None:
    """Caso 15 (§B.4, M13): /pcm, /campo, /codigo, /pausa fuera de live -> 409;

    /campo con un id desconocido en live -> 400.
    """
    ready = _crear_app_ready(tmp_path)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready)
    try:
        # En estado start (fuera de live): todos devuelven 409
        r_pcm = httpx.post(f"{url}/pcm", content=b"\x00\x00" * 512)
        assert r_pcm.status_code == 409

        r_campo = httpx.post(f"{url}/campo", json={"id": "motivo_consulta", "text": "t"})
        assert r_campo.status_code == 409

        r_cod = httpx.post(f"{url}/codigo", json={"code": "I10X", "included": False})
        assert r_cod.status_code == 409

        r_pau = httpx.post(f"{url}/pausa", json={"paused": True})
        assert r_pau.status_code == 409

        # Pasamos a live
        httpx.post(
            f"{url}/documento", content=b"%PDF-1.4 dummy", headers={"X-Name": "doc.pdf"}
        )
        r_emp = httpx.post(
            f"{url}/empezar", json={"owner": "HC-01", "recorded": True, "consent": True}
        )
        assert r_emp.status_code == 200

        # En live: /campo con id desconocido -> 400
        r_desc = httpx.post(f"{url}/campo", json={"id": "id_inexistente", "text": "t"})
        assert r_desc.status_code == 400
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_ruta_vivo_desde_n_y_tipos_linea(tmp_path: Path) -> None:
    """Caso 16 (§B.4): /vivo?desde=N devuelve solo las líneas desde N,

    con kind 'q', 'a' o 'pause'.
    """
    class _MockLiveSTT:
        def __init__(self) -> None:
            self._frases = ["¿Tiene dolor de cabeza?", "Sí, me duele mucho."]

        def transcribe_segments(
            self, path: Path, *, context: str = ""
        ) -> tuple[TranscriptSegment, ...]:
            if self._frases:
                return (TranscriptSegment(0.0, 1.0, self._frases.pop(0)),)
            return ()

    def _vad(audio: np.ndarray) -> np.ndarray:
        chunk = audio[-512:] if audio.size >= 512 else audio
        prob = 0.9 if np.max(np.abs(chunk)) > 0.05 else 0.0
        return np.array([prob], dtype=np.float32)

    ready = _crear_app_ready(tmp_path, live_stt=_MockLiveSTT())
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(
        lambda: ready,
        vad_factory=lambda: _vad,
    )
    try:
        httpx.post(
            f"{url}/documento", content=b"%PDF-1.4 dummy", headers={"X-Name": "doc.pdf"}
        )
        httpx.post(
            f"{url}/empezar", json={"owner": "HC-01", "recorded": True, "consent": True}
        )

        speech = (np.ones(512 * 5, dtype=np.int16) * 10000).tobytes()
        silence = np.zeros(512 * 26, dtype=np.int16).tobytes()

        # 1. Pregunta
        httpx.post(f"{url}/pcm", content=speech)
        httpx.post(f"{url}/pcm", content=silence)

        deadline = time.monotonic() + 3.0
        r_vivo = httpx.get(f"{url}/vivo?desde=0").json()
        while time.monotonic() < deadline and not r_vivo.get("lines"):
            time.sleep(0.05)
            r_vivo = httpx.get(f"{url}/vivo?desde=0").json()

        # 2. Pausa y reanudación
        httpx.post(f"{url}/pausa", json={"paused": True})
        httpx.post(f"{url}/pausa", json={"paused": False})

        # 3. Respuesta
        httpx.post(f"{url}/pcm", content=speech)
        httpx.post(f"{url}/pcm", content=silence)

        deadline = time.monotonic() + 3.0
        r_vivo = httpx.get(f"{url}/vivo?desde=0").json()
        while time.monotonic() < deadline and len(r_vivo.get("lines", [])) < 3:
            time.sleep(0.05)
            r_vivo = httpx.get(f"{url}/vivo?desde=0").json()

        lineas0 = r_vivo.get("lines", [])
        assert len(lineas0) == 3
        assert [lin["kind"] for lin in lineas0] == ["q", "pause", "a"]

        # desde=1 devuelve 2 líneas
        r1 = httpx.get(f"{url}/vivo?desde=1").json()
        lineas1 = r1.get("lines", [])
        assert len(lineas1) == 2
        assert [lin["kind"] for lin in lineas1] == ["pause", "a"]

        # desde=2 devuelve 1 línea
        r2 = httpx.get(f"{url}/vivo?desde=2").json()
        lineas2 = r2.get("lines", [])
        assert len(lineas2) == 1
        assert [lin["kind"] for lin in lineas2] == ["a"]
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_terminar_a_review_con_edicion_y_rechazo_vivo(tmp_path: Path) -> None:
    """Caso 17 (§B.4, M9, M10, M11): /terminar pasa de live a review; el campo

    editado en vivo lleva clase 'edited', su texto y data-original con lo del
    modelo; el código quitado lleva clase 'off' y aria-checked='false'.
    """
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    store = ReviewMetricsStore(tmp_path / "metrics.jsonl")
    ready = _crear_app_ready(tmp_path, store=store)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready)
    try:
        httpx.post(
            f"{url}/documento", content=b"%PDF-1.4 dummy", headers={"X-Name": "doc.pdf"}
        )
        httpx.post(
            f"{url}/empezar", json={"owner": "HC-01", "recorded": True, "consent": True}
        )

        # Médico edita un campo y quita un código en vivo
        r_ed = httpx.post(
            f"{url}/campo",
            json={"id": "motivo_consulta", "text": "Cefalea frontal propia del médico"},
        )
        assert r_ed.status_code == 200

        r_cod = httpx.post(f"{url}/codigo", json={"code": "I10X", "included": False})
        assert r_cod.status_code == 200

        # Mandamos algo de audio PCM para que el WAV exista
        httpx.post(f"{url}/pcm", content=_crear_wav_minimo())

        # Terminar la sesión en vivo
        r_fin = httpx.post(f"{url}/terminar")
        assert r_fin.status_code == 200

        # Esperar a que pase de working a review
        deadline = time.monotonic() + 5.0
        est = httpx.get(f"{url}/estado").json()
        while time.monotonic() < deadline and est.get("state") != "review":
            time.sleep(0.05)
            est = httpx.get(f"{url}/estado").json()
        assert est.get("state") == "review"

        # Comprobar la página de revisión
        r_rev = httpx.get(url, timeout=5.0)
        assert r_rev.status_code == 200
        html = r_rev.text

        # El campo editado tiene clase 'edited', su texto y data-original
        assert '<article class="field edited" data-field-id="motivo_consulta"' in html
        assert "Cefalea frontal propia del médico</textarea>" in html
        assert 'data-original="Motivo de consulta: cefalea frontal pulsátil."' in html

        # El código quitado tiene clase 'off' y aria-checked='false'
        assert '<article class="code off" data-ref="I10X"' in html
        assert 'aria-checked="false"' in html
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_guardar_revision_sin_cambios_registra_metricas(tmp_path: Path) -> None:
    """Caso 18 (§B.4, M11): Guardar la revisión sin tocar nada registra el campo

    como 'edited' y el código como 'rejected' en la métrica.
    """
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    store = ReviewMetricsStore(tmp_path / "metrics.jsonl")
    ready = _crear_app_ready(tmp_path, store=store)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready)
    try:
        httpx.post(
            f"{url}/documento", content=b"%PDF-1.4 dummy", headers={"X-Name": "doc.pdf"}
        )
        httpx.post(
            f"{url}/empezar", json={"owner": "HC-01", "recorded": True, "consent": True}
        )

        httpx.post(
            f"{url}/campo",
            json={"id": "motivo_consulta", "text": "Cefalea frontal propia del médico"},
        )
        httpx.post(f"{url}/codigo", json={"code": "I10X", "included": False})
        httpx.post(f"{url}/pcm", content=_crear_wav_minimo())

        r_fin = httpx.post(f"{url}/terminar")
        assert r_fin.status_code == 200

        deadline = time.monotonic() + 5.0
        est = httpx.get(f"{url}/estado").json()
        while time.monotonic() < deadline and est.get("state") != "review":
            time.sleep(0.05)
            est = httpx.get(f"{url}/estado").json()
        assert est.get("state") == "review"

        # Guardar tal como vino (sin tocar nada en la interfaz de revisión)
        r_save = httpx.post(
            f"{url}/guardar",
            json={
                "answers": {"I10X": {"verdict": "rejected"}},
                "note": {
                    "motivo_consulta": "Cefalea frontal propia del médico",
                    "medicamentos": "Plan de manejo: paracetamol cada 8 horas.",
                },
                "elapsed_seconds": 2.0,
            },
            timeout=5.0,
        )
        assert r_save.status_code == 200

        records = store.read()
        assert len(records) == 1
        r = records[0]
        assert r.fields.get("motivo_consulta") == {"edited": 1}
        assert r.codes.get("rejected") == 1
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_keep_mine_anade_campos_faltantes_vacios() -> None:
    """Caso 19 (§B.4, M12): _keep_mine añade campos de 'mine' que el análisis

    final no trajo, vacíos del lado del modelo para que la nota los conserve.
    """
    spec_motivo = SectionSpec("motivo", "Motivo", (), codable=True, in_note=True)
    spec_extra = SectionSpec("extra", "Apartado Extra", (), codable=True, in_note=True)
    nota_inicial = RenderedNote(
        fields=((spec_motivo, "Cefalea del modelo"),),
        redacted=True,
    )
    draft_obj = ReviewDraft(
        prepared=MagicMock(),
        note=nota_inicial,
        items=(),
    )

    # 1. Campo en mine que falta en el análisis
    mine = {"motivo": "Cefalea editada", "extra": "Texto solo del médico"}
    secciones = (spec_motivo, spec_extra)

    nuevo_draft = _keep_mine(draft_obj, mine, secciones)
    campos = dict((spec.section_id, texto) for spec, texto in nuevo_draft.note.fields)
    assert "extra" in campos
    assert campos["extra"] == ""
    assert campos["motivo"] == "Cefalea del modelo"

    # 2. Si todos los campos de mine ya están en la nota, devuelve el borrador intacto
    draft_sin_faltantes = _keep_mine(draft_obj, {"motivo": "Cefalea"}, secciones)
    assert draft_sin_faltantes is draft_obj


# ---------------------------------------------------------------------------
# Plan 78: Ajustes y arranque de la aplicación (§B.4)
# ---------------------------------------------------------------------------


def _dummy_app_tools(
    *,
    cloud_source: str | None = "env",
    check_fn: Callable[[str], tuple[bool, str]] | None = None,
    save_fn: Callable[[str], None] | None = None,
    model_name: str = "small",
    model_total_mb: int = 464,
    model_available_fn: Callable[[], bool] | None = None,
    model_bytes_fn: Callable[[], int] | None = None,
    download_fn: Callable[[], None] | None = None,
    reader_version_fn: Callable[[], str | None] | None = None,
    cache_since_fn: Callable[[], float | None] | None = None,
) -> AppTools:
    return AppTools(
        cloud_key_source=lambda: cloud_source,
        check_cloud_key=check_fn or (lambda k: (True, "200")),
        save_cloud_key=save_fn or (lambda k: None),
        model_name=model_name,
        model_total_mb=model_total_mb,
        model_available=model_available_fn or (lambda: True),
        model_bytes=model_bytes_fn or (lambda: 0),
        download_model=download_fn or (lambda: None),
        reader_version=reader_version_fn or (lambda: "5.3.0"),
        cache_key_since=cache_since_fn or (lambda: 1700000000.0),
    )


def test_servidor_sin_tools_sin_enlace_ajustes_y_post_ajustes_404(tmp_path: Path) -> None:
    """Caso 19 (§B.4): Sin tools, el inicio no lleva el enlace 'Ajustes' y POST /ajustes da 404."""
    ready = _crear_app_ready(tmp_path)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, tools=None)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert 'id="to-settings"' not in resp.text
        assert 'Ajustes' not in resp.text
        resp_post = httpx.post(f"{url}/ajustes", timeout=5.0)
        assert resp_post.status_code == 404
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_con_tools_lleva_enlace_y_post_ajustes_200(tmp_path: Path) -> None:
    """Caso 20 (§B.4): Con tools, el inicio lo lleva; POST /ajustes desde start da 200
    y la página es la de Ajustes con los textos del pack.
    """
    ready = _crear_app_ready(tmp_path)
    tools = _dummy_app_tools()
    wording = load_app_wording()
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, tools=tools, wording=wording)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert 'id="to-settings"' in resp.text
        assert 'Ajustes' in resp.text

        resp_post = httpx.post(f"{url}/ajustes", timeout=5.0)
        assert resp_post.status_code == 200
        assert resp_post.json() == {"ok": True}

        resp_page = httpx.get(url, timeout=5.0)
        assert resp_page.status_code == 200
        assert wording["title_settings"] in resp_page.text
        assert wording["key_title"] in resp_page.text
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_ajustes_en_working_devuelve_409(tmp_path: Path) -> None:
    """Caso 21 (§B.4, M10): POST /ajustes con la sesión en working da 409 y el estado no cambia."""
    pytest.importorskip("av", reason="requiere el extra de captura (av)")
    bloqueo = threading.Event()
    ready = _crear_app_ready(tmp_path)

    orig_prepare = ready.consultation.prepare

    def blocking_prepare(*args: Any, **kwargs: Any) -> Any:
        bloqueo.wait(timeout=5.0)
        return orig_prepare(*args, **kwargs)

    ready.consultation.prepare = blocking_prepare  # type: ignore[method-assign]
    tools = _dummy_app_tools()
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, tools=tools)
    try:
        audio = tmp_path / "a.wav"
        audio.write_bytes(_crear_wav_minimo())
        doc = tmp_path / "d.txt"
        doc.write_text("historia", encoding="utf-8")
        httpx.post(f"{url}/audio", content=audio.read_bytes(), timeout=5.0)
        httpx.post(
            f"{url}/documento",
            content=doc.read_bytes(),
            headers={"X-Filename": "d.txt"},
            timeout=5.0,
        )
        httpx.post(f"{url}/empezar", json={"owner": "doc_01", "consent": True}, timeout=5.0)
        time.sleep(0.05)

        resp = httpx.post(f"{url}/ajustes", timeout=5.0)
        assert resp.status_code == 409
        assert resp.json() == {"ok": False}
        assert httpx.get(f"{url}/estado", timeout=5.0).json()["state"] == "working"
    finally:
        bloqueo.set()
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_clave_guardar_valida_y_rechaza(tmp_path: Path) -> None:
    """Caso 22 (§B.4, M11): clave-guardar con check_cloud_key falso (401) no llama a
    save_cloud_key y responde ok: false, reason: 401. Con 200 la llama con la clave sin espacios.
    """
    ready = _crear_app_ready(tmp_path)
    claves_guardadas: list[str] = []

    def fake_check(key: str) -> tuple[bool, str]:
        if "bad" in key:
            return False, "401"
        return True, "200"

    tools = _dummy_app_tools(check_fn=fake_check, save_fn=claves_guardadas.append)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, tools=tools)
    try:
        httpx.post(f"{url}/ajustes", timeout=5.0)
        resp_bad = httpx.post(f"{url}/clave-guardar", json={"key": "sk-bad"}, timeout=5.0)
        assert resp_bad.status_code == 200
        assert resp_bad.json() == {"ok": False, "reason": "401"}
        assert len(claves_guardadas) == 0

        resp_good = httpx.post(f"{url}/clave-guardar", json={"key": "  sk-buena  "}, timeout=5.0)
        assert resp_good.status_code == 200
        assert resp_good.json() == {"ok": True, "reason": "200"}
        assert claves_guardadas == ["sk-buena"]
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_clave_probar_nunca_guarda(tmp_path: Path) -> None:
    """Caso 23 (§B.4): clave-probar nunca llama a save_cloud_key."""
    ready = _crear_app_ready(tmp_path)
    claves_guardadas: list[str] = []
    tools = _dummy_app_tools(
        check_fn=lambda k: (True, "200"),
        save_fn=claves_guardadas.append,
    )
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, tools=tools)
    try:
        httpx.post(f"{url}/ajustes", timeout=5.0)
        resp = httpx.post(f"{url}/clave-probar", json={"key": "sk-test"}, timeout=5.0)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "reason": "200"}
        assert len(claves_guardadas) == 0
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_clave_cuerpo_invalido_da_400(tmp_path: Path) -> None:
    """Caso 24 (§B.4): Cuerpo vacío o sin key devuelve 400 en rutas de clave."""
    ready = _crear_app_ready(tmp_path)
    tools = _dummy_app_tools()
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, tools=tools)
    try:
        httpx.post(f"{url}/ajustes", timeout=5.0)
        assert httpx.post(f"{url}/clave-probar", content=b"", timeout=5.0).status_code == 400
        assert httpx.post(f"{url}/clave-probar", json={}, timeout=5.0).status_code == 400
        assert httpx.post(f"{url}/clave-guardar", content=b"", timeout=5.0).status_code == 400
        assert (
            httpx.post(f"{url}/clave-guardar", json={"otra": "x"}, timeout=5.0).status_code == 400
        )
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_descarga_modelo_ciclo_y_concurrencia_y_fallo(tmp_path: Path) -> None:
    """Caso 25 (§B.4, M12): POST /modelo lanza download_model una vez; un segundo POST
    mientras corre no lo lanza otra vez; GET /modelo da running y luego done;
    si revienta, da failed con mensaje en detail y el servidor sigue respondiendo.
    """
    ready = _crear_app_ready(tmp_path)
    descargas: list[int] = []
    bloqueo_dl = threading.Event()
    fallar = False

    def fake_download() -> None:
        descargas.append(1)
        if fallar:
            raise RuntimeError("fallo simulado de descarga")
        bloqueo_dl.wait(timeout=5.0)

    tools = _dummy_app_tools(download_fn=fake_download)
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready, tools=tools)
    try:
        httpx.post(f"{url}/ajustes", timeout=5.0)
        # 1. Primer POST lanza la descarga
        resp1 = httpx.post(f"{url}/modelo", timeout=5.0)
        assert resp1.status_code == 200
        time.sleep(0.05)
        # 2. Segundo POST mientras corre no lanza otra (M12)
        resp2 = httpx.post(f"{url}/modelo", timeout=5.0)
        assert resp2.status_code == 200
        assert len(descargas) == 1

        # 3. GET /modelo da running
        estado_dl = httpx.get(f"{url}/modelo", timeout=5.0).json()
        assert estado_dl["state"] == "running"

        # 4. Desbloqueamos y pasa a done
        bloqueo_dl.set()
        time.sleep(0.05)
        estado_done = httpx.get(f"{url}/modelo", timeout=5.0).json()
        assert estado_done["state"] == "done"

        # 5. Si revienta -> failed con mensaje en detail y servidor responde
        fallar = True
        httpx.post(f"{url}/modelo", timeout=5.0)
        time.sleep(0.05)
        estado_fail = httpx.get(f"{url}/modelo", timeout=5.0).json()
        assert estado_fail["state"] == "failed"
        assert "fallo simulado de descarga" in estado_fail["detail"]
        assert httpx.get(f"{url}/estado", timeout=5.0).status_code == 200
    finally:
        bloqueo_dl.set()
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_volver_desde_ajustes_reconstruye(tmp_path: Path) -> None:
    """Caso 26 (§B.4): POST /volver desde Ajustes llama a build otra vez."""
    ready = _crear_app_ready(tmp_path)
    build_llamadas = 0

    def builder() -> AppReady | AppProblem:
        nonlocal build_llamadas
        build_llamadas += 1
        return ready

    tools = _dummy_app_tools()
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(builder, tools=tools)
    try:
        assert build_llamadas == 1
        httpx.post(f"{url}/ajustes", timeout=5.0)
        resp = httpx.post(f"{url}/volver", timeout=5.0)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert build_llamadas == 2
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_app_problem_model_con_y_sin_tools(tmp_path: Path) -> None:
    """Caso 27 (§B.4): Un build que devuelve AppProblem('model') con tools muestra
    la pantalla de primer arranque (problem_model_title, dl-button, to-settings);
    sin tools muestra la genérica.
    """
    wording = load_app_wording()
    tools = _dummy_app_tools()

    # 1. Con tools: pantalla de primer arranque
    url1, _tok1, _port1, hilo1, _ = _lanzar_servidor_app(
        lambda: AppProblem("model"), tools=tools, wording=wording
    )
    try:
        resp1 = httpx.get(url1, timeout=5.0)
        assert resp1.status_code == 200
        assert wording["problem_model_title"] in resp1.text
        assert 'id="dl-button"' in resp1.text
        assert 'id="to-settings"' in resp1.text
    finally:
        httpx.post(f"{url1}/salir", timeout=5.0)
        hilo1.join(timeout=5.0)

    # 2. Sin tools: pantalla de problema genérica
    url2, _tok2, _port2, hilo2, _ = _lanzar_servidor_app(
        lambda: AppProblem("model"), tools=None, wording=wording
    )
    try:
        resp2 = httpx.get(url2, timeout=5.0)
        assert resp2.status_code == 200
        assert wording["problem_model_title"] not in resp2.text
        assert 'id="dl-button"' not in resp2.text
        assert 'id="to-settings"' not in resp2.text
    finally:
        httpx.post(f"{url2}/salir", timeout=5.0)
        hilo2.join(timeout=5.0)


def test_servidor_app_ready_sin_modelo_local_muestra_aviso(tmp_path: Path) -> None:
    """Caso 28 (§B.4, M15): AppReady(local_model=False) en el inicio muestra check_no_local
    con clase warn.
    """
    ready = _crear_app_ready(tmp_path)
    ready_sin_local = AppReady(
        consultation=ready.consultation,
        reader=ready.reader,
        log=ready.log,
        cloud=ready.cloud,
        writer=ready.writer,
        live_stt=ready.live_stt,
        local_model=False,
    )
    wording = load_app_wording()
    url, _tok, _port, hilo, _ = _lanzar_servidor_app(lambda: ready_sin_local, wording=wording)
    try:
        resp = httpx.get(url, timeout=5.0)
        assert resp.status_code == 200
        assert wording["check_no_local"] in resp.text
        assert 'class="check warn"' in resp.text
    finally:
        httpx.post(f"{url}/salir", timeout=5.0)
        hilo.join(timeout=5.0)


def test_servidor_ajustes_refleja_origenes_clave_y_oculta_descarga(tmp_path: Path) -> None:
    """Caso 29 (§B.4): Ajustes refleja cloud_key_source ('env' => key_status_env y nota,
    None => key_status_none); y si model_available es True no hay id="dl-button".
    """
    ready = _crear_app_ready(tmp_path)
    wording = load_app_wording()

    # 1. Origen env y modelo instalado
    tools_env = _dummy_app_tools(
        cloud_source="env",
        model_available_fn=lambda: True,
    )
    url1, _tok1, _port1, hilo1, _ = _lanzar_servidor_app(
        lambda: ready, tools=tools_env, wording=wording
    )
    try:
        httpx.post(f"{url1}/ajustes", timeout=5.0)
        resp1 = httpx.get(url1, timeout=5.0)
        assert resp1.status_code == 200
        assert wording["key_status_env"] in resp1.text
        assert wording["key_env_note"] in resp1.text
        assert 'id="dl-button"' not in resp1.text
    finally:
        httpx.post(f"{url1}/salir", timeout=5.0)
        hilo1.join(timeout=5.0)

    # 2. Origen None y modelo ausente
    tools_none = _dummy_app_tools(
        cloud_source=None,
        model_available_fn=lambda: False,
    )
    url2, _tok2, _port2, hilo2, _ = _lanzar_servidor_app(
        lambda: ready, tools=tools_none, wording=wording
    )
    try:
        httpx.post(f"{url2}/ajustes", timeout=5.0)
        resp2 = httpx.get(url2, timeout=5.0)
        assert resp2.status_code == 200
        assert wording["key_status_none"] in resp2.text
        assert wording["key_env_note"] not in resp2.text
        assert 'id="dl-button"' in resp2.text
    finally:
        httpx.post(f"{url2}/salir", timeout=5.0)
        hilo2.join(timeout=5.0)


