"""Local SOCKS integration checks: remote DNS, routing, TLS and no fallback."""

from contextlib import contextmanager
import io
import json
import os
import socket
import ssl
import struct
import threading
import unittest
from unittest.mock import Mock, patch

from LRAEP_trainer import network
from LRAEP_trainer.node import NodeClient
from LRAEP_trainer.pool_transport import PoolTransport


def read_exact(peer, count):
    value = bytearray()
    while len(value) < count:
        part = peer.recv(count - len(value))
        if not part:
            raise EOFError("unexpected test peer close")
        value.extend(part)
    return bytes(value)


@contextmanager
def socks_server(handler):
    failures = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(3)

    def serve():
        try:
            with listener.accept()[0] as peer:
                peer.settimeout(3)
                handler(peer)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield listener.getsockname()[1]
    finally:
        listener.close()
        worker.join(4)
        if worker.is_alive():
            raise AssertionError("test SOCKS server did not terminate")
        if failures:
            raise failures[0]


def handshake(peer, targets, *, credentials=None, fragmented=False):
    greeting = read_exact(peer, 3)
    method = 2 if credentials else 0
    if greeting != bytes([5, 1, method]):
        raise AssertionError("unexpected greeting")
    if fragmented:
        peer.sendall(b"\x05")
        peer.sendall(bytes([method]))
    else:
        peer.sendall(bytes([5, method]))
    if credentials:
        version, length = read_exact(peer, 2)
        username = read_exact(peer, length).decode()
        password = read_exact(peer, read_exact(peer, 1)[0]).decode()
        if version != 1 or (username, password) != credentials:
            raise AssertionError("incorrect SOCKS authentication")
        peer.sendall(b"\x01\x00")
    prefix = read_exact(peer, 4)
    if prefix[:3] != b"\x05\x01\x00":
        raise AssertionError("unexpected CONNECT")
    if prefix[3] == 3:
        host = read_exact(peer, read_exact(peer, 1)[0]).decode("ascii")
    elif prefix[3] == 1:
        host = socket.inet_ntop(socket.AF_INET, read_exact(peer, 4))
    elif prefix[3] == 4:
        host = socket.inet_ntop(socket.AF_INET6, read_exact(peer, 16))
    else:
        raise AssertionError("unexpected address type")
    port = struct.unpack("!H", read_exact(peer, 2))[0]
    targets.append((prefix[3], host, port))
    reply = b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x01"
    for part in (bytes([byte]) for byte in reply) if fragmented else (reply,):
        peer.sendall(part)


class SocksNetworkTests(unittest.TestCase):
    def test_authenticated_binary_exchange_uses_remote_dns_and_partial_replies(self):
        targets = []
        payload = b"\x00binary\xff\n"

        def server(peer):
            handshake(peer, targets, credentials=("test-user", "test-password"), fragmented=True)
            self.assertEqual(read_exact(peer, len(payload)), payload)
            peer.sendall(payload)

        real_getaddrinfo = socket.getaddrinfo

        def only_local_resolution(host, *args, **kwargs):
            self.assertEqual(host, "127.0.0.1", "destination hostname leaked to the local resolver")
            return real_getaddrinfo(host, *args, **kwargs)

        with socks_server(server) as port, patch.object(socket, "getaddrinfo", side_effect=only_local_resolution):
            proxy = network.SocksProxy("127.0.0.1", port, "test-user", "test-password")
            with network.create_connection(("remote-name.invalid", 8048), 2, proxy=proxy) as connection:
                connection.sendall(payload)
                self.assertEqual(read_exact(connection, len(payload)), payload)
        self.assertEqual(targets, [(3, "remote-name.invalid", 8048)])

    def test_literal_destinations_use_the_correct_socks_address_type(self):
        for host, kind in (("192.0.2.12", 1), ("2001:db8::1", 4)):
            with self.subTest(host=host):
                targets = []
                with socks_server(lambda peer: handshake(peer, targets)) as port:
                    with network.create_connection((host, 443), 2,
                            proxy=network.SocksProxy("127.0.0.1", port)):
                        pass
                self.assertEqual(targets, [(kind, host, 443)])

    def test_proxy_failure_closes_socket_and_never_attempts_direct_connection(self):
        for reply in (b"\x05\xff", b"\x05\x00"):
            connection = Mock()
            connection.recv.return_value = reply
            with self.subTest(reply=reply), patch.object(network.socket, "create_connection", return_value=connection) as connect:
                proxy = network.SocksProxy("127.0.0.1", 1080, "private-user", "private-password")
                with self.assertRaises(network.NetworkError) as caught:
                    network.create_connection(("destination.invalid", 443), 2, proxy=proxy)
                self.assertNotIn("private-user", str(caught.exception))
                self.assertNotIn("private-password", str(caught.exception))
                connect.assert_called_once()
                self.assertEqual(connect.call_args.args[0], ("127.0.0.1", 1080))
                connection.close.assert_called_once()

    def test_rejected_authentication_does_not_send_destination(self):
        received = []

        def server(peer):
            self.assertEqual(read_exact(peer, 3), b"\x05\x01\x02")
            peer.sendall(b"\x05\x02")
            self.assertEqual(read_exact(peer, 3), b"\x01\x01u")
            self.assertEqual(read_exact(peer, 2), b"\x01p")
            peer.sendall(b"\x01\x01")
            received.append(peer.recv(1))

        with socks_server(server) as port:
            with self.assertRaisesRegex(network.NetworkError, "credentials"):
                network.create_connection(("must-not-be-sent.invalid", 443), 2,
                    proxy=network.SocksProxy("127.0.0.1", port, "u", "p"))
        self.assertEqual(received, [b""])

    def test_negotiation_uses_one_deadline(self):
        connection = Mock()
        connection.recv.return_value = b"\x05\x00"
        with patch.object(network.socket, "create_connection", return_value=connection) as connect, \
                patch.object(network.time, "monotonic", side_effect=[0, 0.2, 0.4, 1.1]):
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                network.create_connection(("deadline.invalid", 443), 1,
                    proxy=network.SocksProxy("127.0.0.1", 1080))
        connect.assert_called_once()
        connection.close.assert_called_once()

    def test_os_errors_are_sanitized(self):
        with patch.object(network.socket, "create_connection", side_effect=OSError("private-password")) as connect:
            with self.assertRaises(network.NetworkError) as caught:
                network.create_connection(("target.invalid", 443), 2,
                    proxy=network.SocksProxy("127.0.0.1", 1080, "private-user", "private-password"))
        self.assertNotIn("private-password", str(caught.exception))
        connect.assert_called_once()

    def test_route_context_restores_after_nested_exception(self):
        outer = network.SocksProxy("127.0.0.1", 1080)
        inner = network.SocksProxy("127.0.0.1", 1081)
        self.assertIsNone(network.current_proxy())
        with network.routed(outer):
            with self.assertRaises(ValueError), network.routed(inner):
                self.assertIs(network.current_proxy(), inner)
                raise ValueError("stop")
            self.assertIs(network.current_proxy(), outer)
        self.assertIsNone(network.current_proxy())

    def test_invalid_proxy_settings_never_echo_credentials(self):
        cases = [("remote.invalid", 1080, "", ""), ("127.0.0.1", 0, "", ""),
                 ("127.0.0.1", 1080, "secret-user", ""),
                 ("127.0.0.1", 1080, "u", "secret-password" * 30)]
        for values in cases:
            with self.subTest(values=values[:2]), self.assertRaises(network.NetworkError) as caught:
                network.SocksProxy(*values)
            self.assertNotIn("secret-user", str(caught.exception))
            self.assertNotIn("secret-password", str(caught.exception))
        self.assertNotIn("secret-password", repr(network.SocksProxy("127.0.0.1", 1080, "u", "secret-password")))

    def test_invalid_targets_and_timeouts_make_no_connections(self):
        proxy = network.SocksProxy("127.0.0.1", 1080)
        with patch.object(network.socket, "create_connection") as connect:
            for address in (("", 443), ("host", 0), ("host", True), ("bad host", 443)):
                with self.subTest(address=address), self.assertRaises(network.NetworkError):
                    network.create_connection(address, 1, proxy=proxy)
            for timeout in (0, -1, True, float("inf"), float("nan")):
                with self.subTest(timeout=timeout), self.assertRaises(network.NetworkError):
                    network.create_connection(("host", 443), timeout, proxy=proxy)
        connect.assert_not_called()

    def test_pool_transport_captures_route_before_worker_thread(self):
        targets = []

        def server(peer):
            handshake(peer, targets)
            self.assertEqual(read_exact(peer, 4), b"pool")
            peer.sendall(b"done")

        with socks_server(server) as port:
            with network.routed(network.SocksProxy("127.0.0.1", port)):
                transport = PoolTransport("pool-target.invalid", 8048)
            failures = []

            def worker():
                try:
                    with transport:
                        transport.send(b"pool")
                        self.assertEqual(transport.receive(), b"done")
                except BaseException as exc:
                    failures.append(exc)

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(4)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
        self.assertEqual(targets, [(3, "pool-target.invalid", 8048)])

    def test_node_rpc_uses_captured_route_and_ignores_environment_proxies(self):
        targets = []
        requests = []

        def server(peer):
            handshake(peer, targets)
            stream = peer.makefile("rb")
            lines = []
            while True:
                line = stream.readline()
                lines.append(line)
                if line == b"\r\n":
                    break
            length = next(int(line.split(b":", 1)[1]) for line in lines if line.lower().startswith(b"content-length:"))
            requests.append(json.loads(stream.read(length)))
            body = b'{"id":1,"result":"connected"}'
            peer.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            stream.close()

        env = {"http_proxy": "http://must-not-connect.invalid:9999", "https_proxy": "http://must-not-connect.invalid:9999",
               "HTTP_PROXY": "http://must-not-connect.invalid:9999", "NO_PROXY": "*"}
        with socks_server(server) as port, patch.dict(os.environ, env):
            with network.routed(network.SocksProxy("127.0.0.1", port)):
                client = NodeClient("http://rpc-target.invalid:44107", "user", "password", timeout=2)
            self.assertEqual(client.rpc("probe"), "connected")
        self.assertEqual(targets, [(3, "rpc-target.invalid", 44107)])
        self.assertEqual(requests[0]["method"], "probe")

    def test_https_wraps_proxy_socket_with_target_sni_and_closes_on_failure(self):
        proxy = network.SocksProxy("127.0.0.1", 1080)
        raw = Mock()
        context = ssl.create_default_context()
        connection = network._SocksHTTPSConnection("tls-target.invalid:443", proxy=proxy, context=context, timeout=2)
        with patch.object(network, "create_connection", return_value=raw) as connect, \
                patch.object(context, "wrap_socket", side_effect=ssl.SSLError("failed certificate")) as wrap:
            with self.assertRaises(ssl.SSLError):
                connection.connect()
        self.assertEqual(connect.call_args.args[0], ("tls-target.invalid", 443))
        self.assertIs(connect.call_args.kwargs["proxy"], proxy)
        wrap.assert_called_once_with(raw, server_hostname="tls-target.invalid")
        raw.close.assert_called_once()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)


if __name__ == "__main__":
    unittest.main()
