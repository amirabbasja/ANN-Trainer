"""Local, transparent WR-to-pool TCP relay for observing pool messages.

Only point a miner you control at this listener. The relay does not generate
mining requests or submit shares; it forwards exactly what the miner sends.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import socket
import socketserver
import ssl
import sys
import threading
from datetime import datetime, timezone

MAX_LINE = 8 * 1024 * 1024
BUFFER_SIZE = 64 * 1024
SENSITIVE_KEYS = {
    "address", "api_key", "authorization", "login", "pass", "password",
    "secret", "token", "user", "username", "wallet", "worker", "wr_wallet", "kex_uname",
}


class Events:
    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.lock = threading.Lock()

    def emit(self, connection: int, event: str, **details: object) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "connection": connection,
            "event": event,
            **details,
        }
        with self.lock:
            print(json.dumps(record, ensure_ascii=True), file=self.stream, flush=True)


def redact(value: object, *, field_limit: int, hide_values: bool = False) -> object:
    """Preserve JSON structure while hiding known credentials and long fields."""
    if isinstance(value, dict):
        return {
            key: redact(item, field_limit=field_limit,
                        hide_values=hide_values or key.lower() in SENSITIVE_KEYS)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, field_limit=field_limit, hide_values=hide_values)
                for item in value]
    if hide_values and isinstance(value, str):
        return "<redacted>"
    if value == "<redacted>":
        return value
    if isinstance(value, str) and len(value) > field_limit:
        return f"{value[:field_limit]}... [truncated, {len(value)} chars total]"
    return value


class Inspector:
    """Reassemble newline-delimited messages independently in each direction."""

    def __init__(self, events: Events, connection: int, direction: str,
                 pending: dict, pending_lock: threading.Lock, *,
                 raw: bool = False, field_limit: int = 256):
        self.events = events
        self.connection = connection
        self.direction = direction
        self.pending = pending
        self.pending_lock = pending_lock
        self.raw = raw
        self.field_limit = field_limit
        self.buffer = bytearray()
        self.discarded = 0

    def feed(self, data: bytes) -> None:
        self.buffer.extend(data)
        while (end := self.buffer.find(b"\n")) != -1:
            line = bytes(self.buffer[:end]).removesuffix(b"\r")
            del self.buffer[:end + 1]
            if self.discarded or len(line) > MAX_LINE:
                self.events.emit(self.connection, "oversized-line", direction=self.direction,
                                 bytes=self.discarded + len(line) + 1)
                self.discarded = 0
            else:
                self._line(line)
        if len(self.buffer) > MAX_LINE:
            self.discarded += len(self.buffer)
            self.buffer.clear()

    def finish(self) -> None:
        if self.discarded or self.buffer:
            if self.discarded:
                self.events.emit(self.connection, "oversized-line", direction=self.direction,
                                 bytes=self.discarded + len(self.buffer), unterminated=True)
            else:
                self._line(bytes(self.buffer), unterminated=True)
        self.buffer.clear()
        self.discarded = 0

    def _line(self, line: bytes, *, unterminated: bool = False) -> None:
        if not line:
            return
        try:
            message = json.loads(line)
        except (UnicodeDecodeError, ValueError):
            details = {"direction": self.direction, "bytes": len(line),
                       "note": "Not a JSON line; use --show-raw for exact bytes"}
            if self.raw:
                details["hex"] = line.hex()
            self.events.emit(self.connection, "non-json", **details,
                             **({"unterminated": True} if unterminated else {}))
            return

        operation = message.get("method") if isinstance(message, dict) else None
        if isinstance(message, dict) and "id" in message:
            request_id = message["id"]
            # JSON IDs are usually strings/numbers; do not fail on unusual arrays/objects.
            if isinstance(request_id, (str, int, float, bool, type(None))):
                with self.pending_lock:
                    if self.direction == "miner -> pool" and operation is not None:
                        self.pending[request_id] = operation
                    elif self.direction == "pool -> miner" and operation is None:
                        operation = self.pending.pop(request_id, None)

        if self.raw:
            displayed = message
        else:
            displayed = copy.deepcopy(message)
            if isinstance(displayed, dict) and isinstance(operation, str):
                method = operation.lower()
                if "auth" in method or "login" in method:
                    # Positional or keyed Stratum credentials may have unknown names.
                    if isinstance(displayed.get("params"), (list, dict)):
                        displayed["params"] = redact(displayed["params"],
                                                      field_limit=self.field_limit,
                                                      hide_values=True)
                elif "submit" in method and self.direction == "miner -> pool":
                    params = displayed.get("params")
                    if isinstance(params, list) and params:
                        params[0] = redact(params[0], field_limit=self.field_limit,
                                           hide_values=True)
            displayed = redact(displayed, field_limit=self.field_limit)

        self.events.emit(self.connection, "json", direction=self.direction,
                         operation=operation, message=displayed,
                         **({"unterminated": True} if unterminated else {}))


class Relay(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, upstream_host: str, upstream_port: int, tls: bool,
                 ca_file: str | None, connect_timeout: float, raw: bool,
                 field_limit: int, events: Events, listen_port: int):
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.context = ssl.create_default_context(cafile=ca_file) if tls else None
        self.connect_timeout = connect_timeout
        self.raw = raw
        self.field_limit = field_limit
        self.events = events
        self.connection_numbers = itertools.count(1)
        super().__init__(("127.0.0.1", listen_port), RelayHandler)

    def connect_upstream(self) -> socket.socket:
        sock = socket.create_connection((self.upstream_host, self.upstream_port),
                                        timeout=self.connect_timeout)
        try:
            if self.context is not None:
                sock = self.context.wrap_socket(sock, server_hostname=self.upstream_host)
            sock.settimeout(None)  # A pool can remain idle longer than the connect timeout.
            return sock
        except BaseException:
            sock.close()
            raise


class RelayHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        relay: Relay = self.server
        number = next(relay.connection_numbers)
        relay.events.emit(number, "miner-connected", peer=self.client_address[0])
        try:
            upstream = relay.connect_upstream()
        except (OSError, ssl.SSLError) as exc:
            relay.events.emit(number, "upstream-error", error=str(exc))
            return

        relay.events.emit(number, "upstream-connected", host=relay.upstream_host,
                          port=relay.upstream_port, tls=upstream.version() if
                          isinstance(upstream, ssl.SSLSocket) else None)
        pending: dict = {}
        pending_lock = threading.Lock()
        to_pool = Inspector(relay.events, number, "miner -> pool", pending, pending_lock,
                            raw=relay.raw, field_limit=relay.field_limit)
        to_miner = Inspector(relay.events, number, "pool -> miner", pending, pending_lock,
                             raw=relay.raw, field_limit=relay.field_limit)
        with upstream:
            sender = threading.Thread(target=self._forward,
                                      args=(self.request, upstream, to_pool), daemon=True)
            sender.start()
            self._forward(upstream, self.request, to_miner)
            sender.join()
        relay.events.emit(number, "disconnected")

    def _forward(self, source: socket.socket, target: socket.socket,
                 inspector: Inspector) -> None:
        try:
            while data := source.recv(BUFFER_SIZE):
                inspector.feed(data)
                target.sendall(data)
            inspector.finish()
            try:
                target.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        except OSError as exc:
            inspector.finish()
            inspector.events.emit(inspector.connection, "relay-error",
                                  direction=inspector.direction, error=str(exc))
            for sock in (source, target):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-host", required=True, help="Real pool hostname")
    parser.add_argument("--upstream-port", type=int, required=True, help="Real pool port")
    parser.add_argument("--upstream-tls", action="store_true",
                        help="Verify TLS to the real pool (local WR side stays plain TCP)")
    parser.add_argument("--ca-file", help="Custom CA file for upstream TLS")
    parser.add_argument("--listen-port", type=int, default=19000,
                        help="Local plain TCP listener port (default: 19000)")
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--check-only", action="store_true",
                        help="Test TCP/TLS connection without starting the relay or logging in")
    parser.add_argument("--show-raw", action="store_true",
                        help="Show unredacted JSON and non-JSON hex (may expose credentials)")
    parser.add_argument("--field-limit", type=int, default=256,
                        help="Maximum displayed characters per JSON string (default: 256)")
    args = parser.parse_args(argv)
    if not 1 <= args.upstream_port <= 65535 or not 1 <= args.listen_port <= 65535:
        parser.error("ports must be between 1 and 65535")
    if args.ca_file and not args.upstream_tls:
        parser.error("--ca-file requires --upstream-tls")
    if args.field_limit < 1 or args.connect_timeout <= 0:
        parser.error("--field-limit and --connect-timeout must be positive")
    events = Events()
    try:
        with Relay(args.upstream_host, args.upstream_port, args.upstream_tls,
                   args.ca_file, args.connect_timeout, args.show_raw,
                   args.field_limit, events, 0 if args.check_only else args.listen_port) as relay:
            if args.check_only:
                with relay.connect_upstream() as sock:
                    events.emit(0, "connection-ok", host=args.upstream_host,
                                port=args.upstream_port, tls=sock.version() if
                                isinstance(sock, ssl.SSLSocket) else None)
                return 0
            events.emit(0, "listening", host="127.0.0.1", port=args.listen_port,
                        upstream=f"{args.upstream_host}:{args.upstream_port}",
                        upstream_tls=args.upstream_tls, raw=args.show_raw)
            relay.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, ssl.SSLError) as exc:
        print(f"wire probe: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
