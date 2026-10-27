"""Node protocol tests; no node, GPU, or native PEARL library is required."""

import base64
import io
import json
import ssl
import unittest
import urllib.error
from unittest.mock import patch

from LRAEP_trainer.node import NodeClient, NodeError, _RejectRedirects


class NodeClientTests(unittest.TestCase):
    def setUp(self):
        self.client = NodeClient("http://127.0.0.1:44107", "rpc-name", "secret-password")

    def reply(self, result=None, *, error=None, request_id=1):
        return io.BytesIO(json.dumps({"id": request_id, "result": result, "error": error}).encode())

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_authenticated_request_and_template_shape(self, urlopen):
        urlopen.return_value = self.reply({"height": 123})
        self.assertEqual(self.client.raw_template(), {"height": 123})
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["method"], "getblocktemplate")
        self.assertEqual(body["params"][0]["rules"], ["segwit"])
        self.assertIn("coinbase/append", body["params"][0]["capabilities"])
        authorization = request.get_header("Authorization")
        self.assertEqual(base64.b64decode(authorization.split()[1]), b"rpc-name:secret-password")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 30)

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_null_submit_result_is_accepted(self, urlopen):
        urlopen.return_value = self.reply(None)
        self.assertIsNone(self.client.submit(b"\x01\x02"))
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["method"], "submitblock")
        self.assertEqual(body["params"], ["0102"])

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_every_non_null_submit_result_is_rejection(self, urlopen):
        for result in ["bad-proof", False, "", {}, 0]:
            with self.subTest(result=result):
                self.client = NodeClient("http://127.0.0.1:44107")
                urlopen.return_value = self.reply(result)
                with self.assertRaisesRegex(NodeError, "rejected"):
                    self.client.submit(b"block")

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_rpc_error_redacts_credentials(self, urlopen):
        urlopen.return_value = self.reply(error={"code": -1, "message": "rpc-name secret-password"})
        with self.assertRaises(NodeError) as caught:
            self.client.rpc("getblocktemplate")
        self.assertNotIn("secret-password", str(caught.exception))
        self.assertNotIn("rpc-name", str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_transport_error_redacts_credentials(self, urlopen):
        urlopen.side_effect = urllib.error.URLError("secret-password unavailable")
        with self.assertRaises(NodeError) as caught:
            self.client.rpc("getbestblockhash")
        self.assertNotIn("secret-password", str(caught.exception))

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_http_error_does_not_expose_body_or_reason(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError(
            self.client.url, 401, "secret-password", {}, io.BytesIO(b"secret-password")
        )
        with self.assertRaisesRegex(NodeError, "HTTP error 401") as caught:
            self.client.rpc("getbestblockhash")
        self.assertNotIn("secret-password", str(caught.exception))

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_invalid_responses_are_rejected(self, urlopen):
        bad_replies = [b"not JSON", b"[]", b'{"id": 9, "result": null}', b'{"id": 1}', b'{"id": true,"result":null}']
        for raw in bad_replies:
            with self.subTest(raw=raw):
                self.client = NodeClient("http://localhost:44107")
                urlopen.return_value = io.BytesIO(raw)
                with self.assertRaises(NodeError):
                    self.client.rpc("getbestblockhash")

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_bounded_response(self, urlopen):
        self.client.MAX_RESPONSE_BYTES = 20
        urlopen.return_value = io.BytesIO(b"x" * 21)
        with self.assertRaisesRegex(NodeError, "size limit"):
            self.client.rpc("getbestblockhash")

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_request_ids_increase(self, urlopen):
        urlopen.side_effect = [self.reply("aa" * 32), self.reply("bb" * 32, request_id=2)]
        self.assertEqual(self.client.best_hash(), "aa" * 32)
        self.assertEqual(self.client.best_hash(), "bb" * 32)

    @patch("LRAEP_trainer.node.urllib.request.OpenerDirector.open")
    def test_invalid_best_hash(self, urlopen):
        urlopen.return_value = self.reply("z" * 64)
        with self.assertRaisesRegex(NodeError, "invalid best block hash"):
            self.client.best_hash()

    def test_invalid_urls_do_not_echo_userinfo(self):
        for url in ["https://name:secret@host:44107", "ftp://host", "http://", "http://host:bad", "https://host/#fragment", "http://ho st"]:
            with self.subTest(url=url), self.assertRaises(NodeError) as caught:
                NodeClient(url)
            self.assertNotIn("secret", str(caught.exception))

    def test_https_verifies_certificate_and_hostname(self):
        client = NodeClient()
        self.assertEqual(client._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(client._context.check_hostname)
        self.assertEqual(client.url, "https://127.0.0.1:44107")

    def test_redirects_cannot_forward_credentials(self):
        handler = _RejectRedirects()
        self.assertIsNone(handler.redirect_request(None, None, 302, "redirect", {}, "http://other-node/"))

    def test_invalid_timeouts(self):
        for timeout in [0, -1, float("nan"), float("inf")]:
            with self.subTest(timeout=timeout), self.assertRaises(NodeError):
                NodeClient(timeout=timeout)


if __name__ == "__main__":
    unittest.main()
