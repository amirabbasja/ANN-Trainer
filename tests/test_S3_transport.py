"""Raw S3 transport tests; no S3 protocol or remote service is assumed."""

import socket
import ssl
import threading
import unittest
from unittest.mock import Mock, patch

from LRAEP_trainer.S3_transport import MAX_IO, S3ConnectionError, S3Transport


class S3TransportTests(unittest.TestCase):
    def test_invalid_hosts(self):
        for host in [None, "", "S3:9000", "https://pool", "user:secret@S3", "a b", "a\n", "-S3", "a..b", "a/", "[invalid]", "::1%scope", "a" * 64 + ".com"]:
            with self.subTest(host=host), self.assertRaises(S3ConnectionError) as caught:
                S3Transport(host, 9000)
            self.assertNotIn("secret", str(caught.exception))

    def test_valid_hosts_and_ipv6(self):
        for host in ["S3.example", "localhost", "127.0.0.1", "::1", "[::1]", "S3.example."]:
            with self.subTest(host=host):
                transport = S3Transport(host, 9000)
                self.assertTrue(transport.endpoint.endswith(":9000"))
        self.assertEqual(S3Transport("[::1]", 9443).endpoint, "[::1]:9443")

    def test_invalid_connection_settings(self):
        for port in [True, 0, -1, 65536, "9000", 9000.0]:
            with self.subTest(port=port), self.assertRaises(S3ConnectionError):
                S3Transport("localhost", port)
        for key in ["connect_timeout", "read_timeout"]:
            for value in [True, False, 0, -1, float("nan"), float("inf"), "15", 10**1000]:
                with self.subTest(key=key, value=value), self.assertRaises(S3ConnectionError):
                    S3Transport("localhost", 9000, **{key: value})
        with self.assertRaises(S3ConnectionError):
            S3Transport("localhost", 9000, tls="yes")
        with self.assertRaises(S3ConnectionError):
            S3Transport("localhost", 9000, ca_file="ca.pem")

    @patch("LRAEP_trainer.S3_transport.socket.create_connection")
    def test_no_handshake_no_framing_and_fragmented_reads(self, connect):
        connection = connect.return_value
        connection.recv.side_effect = [b"\x00", b"ab\n", b"\xff"]
        with S3Transport("S3.example", 9000, connect_timeout=3, read_timeout=7) as transport:
            connect.assert_called_once_with(("S3.example", 9000), timeout=3)
            connection.settimeout.assert_called_once_with(7)
            connection.sendall.assert_not_called()
            payload = b"\x00opaque\n\xff"
            transport.send(payload)
            connection.sendall.assert_called_once_with(payload)
            self.assertEqual(transport.receive(4), b"\x00")
            self.assertEqual(transport.receive(4), b"ab\n")
            self.assertEqual(transport.receive(4), b"\xff")
            self.assertIsNone(transport.tls_version)
        connection.close.assert_called_once()

    @patch("LRAEP_trainer.S3_transport.socket.create_connection")
    def test_verified_tls_context_and_hostname(self, connect):
        # A real default context proves that verification is enabled.
        real_context = ssl.create_default_context()
        self.assertEqual(real_context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(real_context.check_hostname)
        context = Mock(wraps=real_context)
        secured = Mock()
        secured.version.return_value = "TLSv1.3"
        context.wrap_socket = Mock(return_value=secured)
        with patch("LRAEP_trainer.S3_transport.ssl.create_default_context", return_value=context) as create:
            with S3Transport("S3.example", 9443, tls=True, ca_file="custom.pem") as transport:
                create.assert_called_once_with(cafile="custom.pem")
                context.wrap_socket.assert_called_once_with(connect.return_value, server_hostname="S3.example")
                self.assertEqual(transport.tls_version, "TLSv1.3")
                secured.sendall.assert_not_called()
        secured.close.assert_called_once()

    @patch("LRAEP_trainer.S3_transport.socket.create_connection")
    def test_tls_failure_closes_and_does_not_fallback(self, connect):
        context = Mock()
        context.wrap_socket.side_effect = ssl.SSLCertVerificationError("private certificate detail")
        with patch("LRAEP_trainer.S3_transport.ssl.create_default_context", return_value=context):
            with self.assertRaisesRegex(S3ConnectionError, "TLS verification") as caught:
                S3Transport("localhost", 9443, tls=True).open()
        connect.assert_called_once()
        connect.return_value.close.assert_called_once()
        self.assertNotIn("private", str(caught.exception))

    @patch("LRAEP_trainer.S3_transport.socket.create_connection", side_effect=TimeoutError("secret detail"))
    def test_connect_timeout_is_sanitized(self, connect):
        with self.assertRaisesRegex(S3ConnectionError, "timed out") as caught:
            S3Transport("localhost", 9000).open()
        self.assertNotIn("secret", str(caught.exception))

    @patch("LRAEP_trainer.S3_transport.socket.create_connection")
    def test_tls_ca_failure_does_not_connect(self, connect):
        with patch("LRAEP_trainer.S3_transport.ssl.create_default_context", side_effect=OSError("private path")):
            with self.assertRaisesRegex(S3ConnectionError, "certificate authorities") as caught:
                S3Transport("localhost", 9443, tls=True, ca_file="missing.pem").open()
        self.assertNotIn("private", str(caught.exception))
        connect.assert_not_called()

    def test_send_and_receive_require_an_open_connection(self):
        transport = S3Transport("localhost", 9000)
        for operation in [lambda: transport.send(b"data"), transport.receive]:
            with self.assertRaisesRegex(S3ConnectionError, "not open"):
                operation()

    @patch("LRAEP_trainer.S3_transport.socket.create_connection")
    def test_io_failure_closes_connection(self, connect):
        for method in ["send", "receive"]:
            for error in [TimeoutError("secret"), ConnectionResetError("secret")]:
                with self.subTest(method=method, error=type(error)):
                    connection = Mock()
                    connect.return_value = connection
                    getattr(connection, "sendall" if method == "send" else "recv").side_effect = error
                    transport = S3Transport("localhost", 9000).open()
                    with self.assertRaises(S3ConnectionError) as caught:
                        getattr(transport, method)(b"data") if method == "send" else transport.receive()
                    self.assertNotIn("secret", str(caught.exception))
                    connection.close.assert_called_once()
                    with self.assertRaisesRegex(S3ConnectionError, "not open"):
                        transport.receive()

    @patch("LRAEP_trainer.S3_transport.socket.create_connection")
    def test_eof_is_distinct_from_a_fragment(self, connect):
        connect.return_value.recv.return_value = b""
        transport = S3Transport("localhost", 9000).open()
        with self.assertRaisesRegex(S3ConnectionError, "closed the connection"):
            transport.receive()
        connect.return_value.close.assert_called_once()

    @patch("LRAEP_trainer.S3_transport.socket.create_connection")
    def test_io_size_bounds(self, connect):
        with S3Transport("localhost", 9000) as transport:
            for value in [0, -1, True, 1.5, MAX_IO + 1]:
                with self.subTest(value=value), self.assertRaises(S3ConnectionError):
                    transport.receive(value)
            for payload in ["not bytes", bytearray(b"mutable"), b"x" * (MAX_IO + 1)]:
                with self.subTest(kind=type(payload)), self.assertRaises(S3ConnectionError):
                    transport.send(payload)
            connect.return_value.recv.assert_not_called()
            connect.return_value.sendall.assert_not_called()

    @patch("LRAEP_trainer.S3_transport.socket.create_connection")
    def test_open_and_close_are_idempotent(self, connect):
        transport = S3Transport("localhost", 9000)
        self.assertIs(transport.open(), transport)
        self.assertIs(transport.open(), transport)
        connect.assert_called_once()
        transport.close()
        transport.close()
        connect.return_value.close.assert_called_once()

    def test_real_loopback_binary_exchange_and_eof(self):
        # Use an ephemeral loopback port and explicit framing in this test only.
        payload = b"\x00\xffbinary\nbytes"
        failures = []
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(3)

            def server():
                try:
                    with listener.accept()[0] as peer:
                        peer.settimeout(3)
                        received = bytearray()
                        while len(received) < len(payload):
                            part = peer.recv(len(payload) - len(received))
                            if not part:
                                raise AssertionError("client closed before sending test payload")
                            received.extend(part)
                        if bytes(received) != payload:
                            raise AssertionError("transport changed payload")
                        peer.sendall(received)
                except BaseException as error:
                    failures.append(error)

            worker = threading.Thread(target=server, daemon=True)
            worker.start()
            with S3Transport("127.0.0.1", listener.getsockname()[1], read_timeout=3) as transport:
                transport.send(payload)
                received = bytearray()
                while len(received) < len(payload):
                    received.extend(transport.receive(2))
                self.assertEqual(bytes(received), payload)
                with self.assertRaisesRegex(S3ConnectionError, "closed the connection"):
                    transport.receive()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
