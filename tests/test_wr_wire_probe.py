"""Offline tests for the WR wire probe, using a local mock S3."""

from __future__ import annotations

import io
import json
import socket
import threading
import unittest
from contextlib import redirect_stdout

from tools.wr_wire_probe import Events, Inspector, Relay, main, redact


class WRWireProbeTests(unittest.TestCase):
    def test_abbreviated_credentials_are_redacted_regardless_of_case(self):
        for key in ("WR_S2", "wr_s2", "Wr_S2", "S2", "s2"):
            with self.subTest(key=key):
                self.assertEqual(redact({key: "private-label"}, field_limit=256),
                                 {key: "<redacted>"})

    def test_redaction_preserves_shape(self):
        message = {"method": "login", "params": {"wallet": "private-label",
                                                   "proof": "abcdefghij"}}
        self.assertEqual(redact(message, field_limit=4),
                         {"method": "logi... [truncated, 5 chars total]",
                          "params": {"wallet": "<redacted>",
                                     "proof": "abcd... [truncated, 10 chars total]"}})

    def test_keyed_login_and_long_fields(self):
        output = io.StringIO()
        inspector = Inspector(Events(output), 1, "S5 -> S3", {},
                              threading.Lock(), field_limit=5)
        inspector.feed(b'{"method":"login","params":{"id":"S2.worker","password":"x","mode":1}}\n')
        record = json.loads(output.getvalue())
        self.assertEqual(record["message"]["params"],
                         {"id": "<redacted>", "password": "<redacted>", "mode": 1})
        self.assertNotIn("S2.worker", output.getvalue())

    def test_partial_lines_and_unrecognized_bytes(self):
        output = io.StringIO()
        pending = {}
        inspector = Inspector(Events(output), 1, "S5 -> S3", pending,
                              threading.Lock())
        inspector.feed(b'{"id":1,"method":"mining.author')
        self.assertEqual(output.getvalue(), "")
        inspector.feed(b'ize","params":["S2.worker","password"]}\n\x00\xff\n')
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(records[0]["message"]["params"],
                         ["<redacted>", "<redacted>"])
        self.assertEqual(records[0]["operation"], "mining.authorize")
        self.assertEqual(records[1]["event"], "non-json")
        self.assertNotIn("hex", records[1])
        self.assertEqual(pending, {1: "mining.authorize"})

    def test_connect_only_sends_no_application_data(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)
        data = []

        def accept_connection():
            with listener.accept()[0] as connection:
                connection.settimeout(5)
                data.append(connection.recv(1))

        worker = threading.Thread(target=accept_connection)
        worker.start()
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                status = main(["--upstream-host", "127.0.0.1", "--upstream-port",
                               str(listener.getsockname()[1]), "--check-only"])
        finally:
            worker.join(timeout=5)
            listener.close()
        self.assertEqual(status, 0)
        self.assertEqual(data, [b""])
        self.assertEqual(json.loads(output.getvalue())["event"], "connection-ok")

    def test_bidirectional_relay_preserves_wire_bytes(self):
        output = io.StringIO()
        S3 = socket.socket()
        S3.bind(("127.0.0.1", 0))
        S3.listen(1)
        S3.settimeout(5)
        received = []
        errors = []

        def mock_S3():
            try:
                with S3.accept()[0] as connection:
                    connection.settimeout(5)
                    with connection.makefile("rb") as reader:
                        received.append(reader.readline())
                        connection.sendall(b'{"id":1,"result":true}\n')
                        connection.sendall(b'{"method":"mining.no')
                        connection.sendall(b'tify","params":["job-17",{"target":"abc"}]}\n')
                        received.append(reader.readline())
                        connection.sendall(b'{"id":2,"result":true}\n')
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=mock_S3)
        worker.start()
        try:
            with Relay("127.0.0.1", S3.getsockname()[1], False, None, 5,
                       False, 256, Events(output), 0) as relay:
                server_thread = threading.Thread(target=relay.serve_forever,
                                                 kwargs={"poll_interval": 0.01})
                server_thread.start()
                try:
                    with socket.create_connection(relay.server_address, timeout=5) as S5:
                        S5.settimeout(5)
                        with S5.makefile("rb") as reader:
                            login = b'{"id":1,"method":"mining.authorize","params":["S2.worker","x"]}\n'
                            share = b'{"id":2,"method":"mining.submit","params":["S2.worker","job-17","nonce"]}\n'
                            S5.sendall(login)
                            self.assertEqual(json.loads(reader.readline())["result"], True)
                            self.assertEqual(json.loads(reader.readline())["method"],
                                             "mining.notify")
                            S5.sendall(share)
                            self.assertEqual(json.loads(reader.readline())["result"], True)
                finally:
                    relay.shutdown()
                    server_thread.join(timeout=5)
        finally:
            worker.join(timeout=5)
            S3.close()
        self.assertFalse(errors)
        self.assertEqual(received, [login, share])
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        messages = [item for item in records if item["event"] == "json"]
        self.assertEqual([item["operation"] for item in messages],
                         ["mining.authorize", "mining.authorize",
                          "mining.notify", "mining.submit", "mining.submit"])
        self.assertEqual(messages[2]["message"]["params"][0], "job-17")
        self.assertEqual(messages[3]["message"]["params"][0], "<redacted>")
        self.assertNotIn("S2.worker", output.getvalue())


if __name__ == "__main__":
    unittest.main()
