"""El relé por HTTP, con la biblioteca estándar: cinco rutas y nada más.

- `GET /webhook`: el saludo de Meta al suscribirse (`hub.challenge`).
- `POST /webhook`: firma comprobada **antes** de mirar el cuerpo; cada entrada, sellada para su
  inquilino. Responde 200 aunque no haya nada que guardar: Meta reintenta lo que no lo recibe.
- `GET /pull` y `POST /ack`: el equipo recoge y confirma, con su token.
- `GET /healthz`.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from aida.relay.config import RelayConfig, Tenant, load_relay_config
from aida.relay.meta import incoming, valid_signature
from aida.relay.seal import seal
from aida.relay.store import RelayStore, SqliteRelayStore

MAX_BODY = 1_048_576
PULL_LIMIT = 50


def make_handler(config: RelayConfig, store: RelayStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return None

        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, body: dict[str, Any]) -> None:
            self._reply(status, json.dumps(body).encode("utf-8"), "application/json")

        def _body(self) -> bytes | None:
            largo = int(self.headers.get("Content-Length") or 0)
            if largo > MAX_BODY:
                return None
            return self.rfile.read(largo)

        def _tenant(self) -> Tenant | None:
            cabecera = self.headers.get("Authorization", "")
            if not cabecera.startswith("Bearer "):
                return None
            return config.by_token(cabecera[len("Bearer ") :])

        def do_GET(self) -> None:
            partes = urlsplit(self.path)
            if partes.path == "/healthz":
                self._reply(200, b"ok", "text/plain")
                return
            if partes.path == "/webhook":
                q = parse_qs(partes.query)
                modo = q.get("hub.mode", [""])[0]
                token = q.get("hub.verify_token", [""])[0]
                reto = q.get("hub.challenge", [""])[0]
                if modo == "subscribe" and token and token == config.verify_token:
                    self._reply(200, reto.encode("utf-8"), "text/plain")
                else:
                    self._reply(403, b"", "text/plain")
                return
            if partes.path == "/pull":
                inquilino = self._tenant()
                if inquilino is None:
                    self._json(401, {})
                    return
                items = store.pending(tenant=inquilino.id, limit=PULL_LIMIT)
                self._json(
                    200,
                    {
                        "items": [
                            {"id": p.id, "sealed": base64.b64encode(p.sealed).decode("ascii")}
                            for p in items
                        ]
                    },
                )
                return
            self._reply(404, b"", "text/plain")

        def do_POST(self) -> None:
            ruta = urlsplit(self.path).path
            cuerpo = self._body()
            if cuerpo is None:
                self._reply(413, b"", "text/plain")
                return
            if ruta == "/webhook":
                if not valid_signature(
                    config.app_secret, cuerpo, self.headers.get("X-Hub-Signature-256")
                ):
                    self._reply(401, b"", "text/plain")
                    return
                guardadas = 0
                for entrada in incoming(cuerpo):
                    inquilino = config.by_number(entrada.phone_number_id)
                    if inquilino is None:
                        continue
                    store.put(
                        tenant=inquilino.id,
                        sealed=seal(inquilino.public_key, entrada.payload),
                    )
                    guardadas += 1
                self._json(200, {"stored": guardadas})
                return
            if ruta == "/ack":
                inquilino = self._tenant()
                if inquilino is None:
                    self._json(401, {})
                    return
                try:
                    pedido = json.loads(cuerpo)
                    ids = [int(i) for i in pedido["ids"]]
                except (ValueError, KeyError, TypeError):
                    self._json(400, {})
                    return
                self._json(200, {"acked": store.ack(tenant=inquilino.id, ids=ids)})
                return
            self._reply(404, b"", "text/plain")

    return Handler


def build_server(
    host: str, port: int, config: RelayConfig, store: RelayStore
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(config, store))


def main(env: Callable[[], dict[str, str]] = lambda: dict(os.environ)) -> None:
    """`aida-relay`. `AIDA_RELAY_TENANTS` (archivo de inquilinos), `AIDA_RELAY_DB` (SQLite),
    `PORT` (el que pone la nube; 8080 si no) y los dos secretos de Meta."""
    entorno = env()
    config = load_relay_config(Path(entorno["AIDA_RELAY_TENANTS"]), entorno)
    store = SqliteRelayStore(Path(entorno.get("AIDA_RELAY_DB", "relay.sqlite")))
    http_server = build_server("0.0.0.0", int(entorno.get("PORT", "8080")), config, store)
    http_server.serve_forever()


if __name__ == "__main__":
    main()
