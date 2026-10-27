"""VPN link reports and probes stay routed and omit supplied credentials."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import socketserver
import ssl
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from LRAEP_trainer import vpn_check
from LRAEP_trainer.network import SocksProxy
from LRAEP_trainer.vpn import VPNError


class _Managed:
    def __init__(self, proxy):
        self.proxy = proxy
        self.closed = False

    def __enter__(self):
        return self.proxy

    def __exit__(self, *_):
        self.closed = True


class _OptimisticSOCKS(socketserver.BaseRequestHandler):
    """A local core that accepts CONNECT without opening any outbound."""

    def _read(self, count):
        result = b""
        while len(result) < count:
            part = self.request.recv(count - len(result))
            if not part:
                raise EOFError
            result += part
        return result

    def handle(self):
        self.request.settimeout(1)
        try:
            version, methods = self._read(2)
            self._read(methods)
            self.request.sendall(b"\x05\x02")
            version, user_size = self._read(2)
            self._read(user_size)
            self._read(self._read(1)[0])
            self.request.sendall(b"\x01\x00")
            version, command, reserved, address_type = self._read(4)
            length = self._read(1)[0] if address_type == 3 else {1: 4, 4: 16}[address_type]
            self._read(length + 2)
            self.request.sendall(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x00")
            # Plain CONNECT looks successful, but application traffic cannot
            # reach a remote server; a TLS request immediately encounters EOF.
            self.request.recv(4096)
        except (OSError, EOFError, KeyError):
            pass


class VPNCheckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.links = self.directory / "links.txt"
        self.output = self.directory / "results.json"
        self.links.write_text("# do not include in report\n\nfirst-SECRET-URL\nsecond-SECRET-URL\n")
        self.proxy = SocksProxy("127.0.0.1", 19080, "SECRET_USER", "SECRET_PASSWORD")
        self.configuration = {
            "vpn": {"core_path": "/bin/unused", "check_url": "https://check.example/ip"},
            "pool": {"host": "pool.example", "port": 8048, "tls": True, "ca_file": None},
        }

    def run_check(self, **kwargs):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = vpn_check.run_vpn_check(self.configuration, self.links,
                                             output=self.output, **kwargs)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_mixed_links_report_pool_loss_and_https_separately_without_secrets(self):
        managed = [_Managed(self.proxy), _Managed(self.proxy)]
        with patch.object(vpn_check, "parse_link"), \
             patch.object(vpn_check, "ManagedVPN", side_effect=managed) as core, \
             patch.object(vpn_check, "_probe_pool", side_effect=[10, TimeoutError("SECRET"), 30,
                                                               OSError("SECRET"), OSError(), OSError()]), \
             patch.object(vpn_check, "_probe_https", side_effect=[TimeoutError("SECRET"), (15, "203.0.113.7")]):
            result, stdout, stderr = self.run_check()
        self.assertEqual(result, 0)
        self.assertFalse(stderr)
        self.assertEqual(core.call_count, 2)
        self.assertTrue(all(instance.closed for instance in managed))
        report = json.loads(self.output.read_text())
        self.assertEqual(report["working"], 1)
        self.assertEqual([item["line"] for item in report["results"]], [3, 4])
        first, second = report["results"]
        self.assertEqual(first["pool_latency_ms"], 20)
        self.assertEqual(first["pool_loss_percent"], 33.3)
        self.assertEqual(first["https_status"], "failed")
        self.assertTrue(first["usable"])
        self.assertEqual(first["pool_proof_level"], "tls_verified")
        self.assertEqual(second["status"], "POOL_FAIL")
        self.assertEqual(second["https_status"], "ok")
        self.assertIn("POOL_OK", stdout)
        self.assertIn("POOL_FAIL", stdout)
        self.assertIn("FAIL", stdout)
        self.assertNotIn("SECRET", stdout + stderr + self.output.read_text())

    def test_optimistic_socks_ack_with_broken_tunnel_is_unverified(self):
        # Some SOCKS cores acknowledge CONNECT before contacting their remote
        # outbound. A successful acknowledgement alone must never pass a link.
        self.configuration["pool"]["tls"] = False
        self.links.write_text("valid-link\n")
        with patch.object(vpn_check, "parse_link"), \
             patch.object(vpn_check, "ManagedVPN", return_value=_Managed(self.proxy)), \
             patch.object(vpn_check, "_probe_pool", return_value=0.1), \
             patch.object(vpn_check, "_probe_https", side_effect=OSError("SECRET")):
            result, stdout, _ = self.run_check(attempts=1)
        self.assertEqual(result, 1)
        self.assertIn("UNVERIFIED", stdout)
        report = json.loads(self.output.read_text())["results"][0]
        self.assertFalse(report["usable"])
        self.assertEqual(report["pool_proof_level"], "connect_only")
        self.assertNotIn("SECRET", self.output.read_text() + stdout)

    def test_real_socks_ack_without_outbound_cannot_pass_plain_pool_probe(self):
        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True

        self.configuration["pool"]["tls"] = False
        self.links.write_text("valid-link\n")
        with Server(("127.0.0.1", 0), _OptimisticSOCKS) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(thread.join, 2)
            self.addCleanup(server.shutdown)
            proxy = SocksProxy("127.0.0.1", server.server_address[1], "user", "password")
            with patch.object(vpn_check, "parse_link"), \
                 patch.object(vpn_check, "ManagedVPN", return_value=_Managed(proxy)):
                result, stdout, _ = self.run_check(attempts=1, timeout=1)
        self.assertEqual(result, 1)
        self.assertIn("UNVERIFIED", stdout)
        report = json.loads(self.output.read_text())["results"][0]
        self.assertEqual(report["pool_successes"], 1)
        self.assertEqual(report["https_status"], "failed")
        self.assertFalse(report["usable"])

    def test_plain_tcp_without_https_cannot_count_as_working(self):
        self.configuration["pool"]["tls"] = False
        self.configuration["vpn"]["check_url"] = None
        self.links.write_text("valid-link\n")
        with patch.object(vpn_check, "parse_link"), \
             patch.object(vpn_check, "ManagedVPN", return_value=_Managed(self.proxy)), \
             patch.object(vpn_check, "_probe_pool", return_value=0.1), \
             patch.object(vpn_check, "_probe_https") as https:
            result, stdout, _ = self.run_check(attempts=1)
        self.assertEqual(result, 1)
        self.assertIn("UNVERIFIED", stdout)
        self.assertFalse(json.loads(self.output.read_text())["results"][0]["usable"])
        https.assert_not_called()

    def test_plain_tcp_plus_https_is_usable_but_pool_stays_connect_only(self):
        self.configuration["pool"]["tls"] = False
        self.links.write_text("valid-link\n")
        with patch.object(vpn_check, "parse_link"), \
             patch.object(vpn_check, "ManagedVPN", return_value=_Managed(self.proxy)), \
             patch.object(vpn_check, "_probe_pool", return_value=0.1), \
             patch.object(vpn_check, "_probe_https", return_value=(15, "203.0.113.8")):
            result, stdout, _ = self.run_check(attempts=1)
        self.assertEqual(result, 0)
        self.assertIn("CONNECT_OK", stdout)
        report = json.loads(self.output.read_text())["results"][0]
        self.assertTrue(report["usable"])
        self.assertEqual(report["pool_proof_level"], "connect_only")

    def test_invalid_link_does_not_start_core_and_following_link_still_runs(self):
        instance = _Managed(self.proxy)
        with patch.object(vpn_check, "parse_link", side_effect=[VPNError("SECRET"), {}]), \
             patch.object(vpn_check, "ManagedVPN", return_value=instance) as core, \
             patch.object(vpn_check, "_probe_pool", return_value=12) as pool, \
             patch.object(vpn_check, "_probe_https", return_value=(15, "203.0.113.8")):
            result, stdout, stderr = self.run_check(attempts=1)
        self.assertEqual(result, 0)
        core.assert_called_once_with(self.configuration["vpn"], link="second-SECRET-URL")
        pool.assert_called_once_with(self.configuration["pool"], self.proxy, 5.0)
        self.assertNotIn("SECRET", stdout + stderr + self.output.read_text())
        self.assertTrue(instance.closed)

    def test_core_failure_is_redacted_and_does_not_stop_other_links(self):
        instance = _Managed(self.proxy)
        with patch.object(vpn_check, "parse_link"), \
             patch.object(vpn_check, "ManagedVPN", side_effect=[VPNError("SECRET"), instance]), \
             patch.object(vpn_check, "_probe_pool", side_effect=TimeoutError("SECRET")), \
             patch.object(vpn_check, "_probe_https", side_effect=OSError("SECRET")):
            result, stdout, stderr = self.run_check(attempts=1)
        self.assertEqual(result, 1)
        self.assertIn("CORE_FAIL", stdout)
        self.assertIn("POOL_FAIL", stdout)
        self.assertNotIn("SECRET", stdout + stderr + self.output.read_text())
        self.assertTrue(instance.closed)

    def test_https_can_be_disabled_without_any_http_request(self):
        self.configuration["vpn"]["check_url"] = None
        self.links.write_text("valid-link\n")
        with patch.object(vpn_check, "parse_link"), \
             patch.object(vpn_check, "ManagedVPN", return_value=_Managed(self.proxy)), \
             patch.object(vpn_check, "_probe_pool", return_value=12), \
             patch.object(vpn_check, "_probe_https") as http:
            result, stdout, _ = self.run_check(attempts=1)
        self.assertEqual(result, 0)
        http.assert_not_called()
        self.assertIn("disabled", stdout)

    def test_empty_or_missing_links_fail_without_starting_a_core(self):
        for text in ("", "# comments\n  \n"):
            self.links.write_text(text)
            with patch.object(vpn_check, "ManagedVPN") as core:
                result, _, stderr = self.run_check()
            self.assertEqual(result, 1)
            self.assertIn("no configurations", stderr)
            core.assert_not_called()
        self.links.unlink()
        with patch.object(vpn_check, "ManagedVPN") as core:
            result, _, stderr = self.run_check()
        self.assertEqual(result, 1)
        self.assertIn("Could not read", stderr)
        core.assert_not_called()

    def test_probe_options_are_bounded_before_reading_the_file(self):
        for value in (0, 21, True, 1.5):
            with self.subTest(attempts=value), self.assertRaises(ValueError):
                self.run_check(attempts=value)
        for value in (0, 61, -1, float("inf"), float("nan"), True, "5"):
            with self.subTest(timeout=value), self.assertRaises(ValueError):
                self.run_check(timeout=value)

    def test_pool_probe_sends_hostname_to_proxy_and_verifies_target_tls(self):
        raw, secured = Mock(), Mock()
        context = Mock(wrap_socket=Mock(return_value=secured))
        with patch.object(vpn_check.network, "create_connection", return_value=raw) as connect, \
             patch.object(vpn_check.ssl, "create_default_context", return_value=context) as tls:
            latency = vpn_check._probe_pool(self.configuration["pool"], self.proxy, 5)
        connect.assert_called_once_with(("pool.example", 8048), 5, proxy=self.proxy)
        tls.assert_called_once_with(cafile=None)
        context.wrap_socket.assert_called_once_with(raw, server_hostname="pool.example")
        secured.close.assert_called_once()
        self.assertGreaterEqual(latency, 0)

    def test_tls_failure_closes_pool_socket_and_never_retries_directly(self):
        raw = Mock()
        context = Mock()
        context.wrap_socket.side_effect = ssl.SSLCertVerificationError("SECRET")
        with patch.object(vpn_check.network, "create_connection", return_value=raw) as connect, \
             patch.object(vpn_check.ssl, "create_default_context", return_value=context):
            with self.assertRaises(ssl.SSLError):
                vpn_check._probe_pool(self.configuration["pool"], self.proxy, 5)
        raw.close.assert_called_once()
        connect.assert_called_once()

    def test_https_uses_explicit_proxy_verified_context_and_limited_response(self):
        response = Mock()
        response.getcode.return_value = 200
        response.read.return_value = b"203.0.113.9\n"
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock()
        opener = Mock()
        opener.open.return_value = response
        with patch.object(vpn_check.network, "http_opener", return_value=opener) as build:
            latency, exit_ip = vpn_check._probe_https("https://check.example/ip", self.proxy, 5)
        self.assertEqual(exit_ip, "203.0.113.9")
        self.assertGreaterEqual(latency, 0)
        self.assertIs(build.call_args.args[0], self.proxy)
        context = build.call_args.kwargs["context"]
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertIsInstance(build.call_args.kwargs["handlers"][0], vpn_check._RejectRedirects)
        response.read.assert_called_once_with(4097)
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 5)

    def test_http_url_with_credentials_or_insecure_scheme_is_not_opened(self):
        for value in ("http://check.example/ip", "https://user:SECRET@check.example/ip",
                      "https://check.example/ip#SECRET"):
            with self.subTest(url=value), \
                 patch.object(vpn_check.network, "http_opener") as opener, \
                 self.assertRaises(ValueError) as caught:
                vpn_check._probe_https(value, self.proxy, 5)
            self.assertNotIn("SECRET", str(caught.exception))
            opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
