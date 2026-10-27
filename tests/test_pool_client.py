"""KEX session tests using local sockets and no GPU or remote pool."""

import base64
import json
import queue
import socket
import threading
import time
import unittest
from unittest.mock import patch

from LRAEP_trainer.pool_client import (
    PoolAuthenticationError, PoolCompatibilityError, PoolSession, ShareResult, StalePoolJob,
)
from LRAEP_trainer.pool_transport import PoolConnectionError
from LRAEP_trainer.pool_protocol import PoolProtocolError


def job_message(job_id="job-1", *, cert=3, target="1234"):
    params = {"job_id": job_id, "header": "00" * 76, "height": 100, "target": target}
    if cert is not None:
        params["cert_version"] = cert
    return {"method": "mining.notify", "params": params}


class MockPool:
    """One local socket peer whose script is supplied by each test."""

    def __init__(self, script):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(3)
        self.port = self.listener.getsockname()[1]
        self.failures = []
        self.finished = threading.Event()
        self.peer = None
        self.thread = threading.Thread(target=self._serve, args=(script,), daemon=True)

    def _serve(self, script):
        try:
            with self.listener.accept()[0] as self.peer:
                self.peer.settimeout(3)
                script(self.peer)
        except BaseException as error:
            self.failures.append(error)
        finally:
            self.finished.set()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, value, traceback):
        if self.peer is not None:
            try:
                self.peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.listener.close()
        self.thread.join(3)
        if exc_type is None and self.failures:
            raise AssertionError(f"mock pool failed: {self.failures}")
        if self.thread.is_alive():
            raise AssertionError("mock pool thread leaked")

    def settings(self, **extra):
        return {
            "host": "127.0.0.1", "port": self.port, "tls": False,
            "connect_timeout": 2, "read_timeout": 3,
            "KEX_uname": "prl1ptest", "worker": "rig", "password": "secret",
            **extra,
        }


def send_json(peer, message):
    peer.sendall(json.dumps(message, separators=(",", ":")).encode() + b"\n")


def read_json(reader):
    line = reader.readline()
    if not line:
        raise AssertionError("client closed before request")
    return json.loads(line)


def authorize(peer, *, gzip_reply=False, notify=True):
    reader = peer.makefile("rb")
    first = read_json(reader)
    if first["method"] == "mining.subscribe":
        auth = read_json(reader)  # Send no subscribe reply at all.
    else:
        auth = first
    if auth["method"] != "mining.authorize":
        raise AssertionError("expected immediate authorize")
    reply = {"id": auth["id"], "result": True, "error": None}
    if gzip_reply:
        reply["type"] = "v2"
    send_json(peer, reply)
    if notify:
        send_json(peer, job_message())
    return reader, auth


class PoolSessionTests(unittest.TestCase):
    def test_kex_session_uses_only_kex_identity(self):
        settings = {"host": "127.0.0.1", "port": 19000, "KEX_uname": "kex-user",
                    "WR_wallet": "wr-address", "worker": "Rig01"}
        session = PoolSession(settings)
        self.addCleanup(session.close)
        self.assertEqual(session.dialect.auth_requests()[-1]["params"], ["kex-user.Rig01", "x"])

    def test_kex_missing_identity_never_falls_back_to_wr_or_legacy_wallet(self):
        for other_fields in ({"WR_wallet": "wr-address"}, {"wallet": "legacy-address"},
                             {"KEX_uname": "", "WR_wallet": "wr-address"}):
            with self.subTest(other_fields=other_fields), self.assertRaisesRegex(PoolProtocolError, "KEX_uname"):
                PoolSession({"host": "127.0.0.1", "port": 19000, "worker": "Rig01", **other_fields})

    def test_silent_subscribe_auth_and_strict_share_replies(self):
        captured = queue.Queue()

        def script(peer):
            reader, auth = authorize(peer)
            captured.put(auth)
            for result in [True, False, 1, "true", {}]:
                request = read_json(reader)
                captured.put(request)
                send_json(peer, {"id": request["id"], "result": result, "error": None})
            # EOF immediately after the final ack must not erase that ack.
            reader.close()

        with MockPool(script) as pool, PoolSession(pool.settings()) as session:
            snapshot = session.wait_job(1)
            self.assertIsNotNone(snapshot)
            proof = base64.b64encode(b"official-proof-bytes").decode()
            results = [session.submit(snapshot, proof) for _ in range(5)]
            self.assertEqual([r.accepted for r in results], [True, False, False, False, False])
            self.assertIsNone(results[0].error)
            self.assertIsNotNone(results[1].error)
        auth = captured.get_nowait()
        self.assertEqual(auth["params"], ["prl1ptest.rig", "secret"])
        requests = [captured.get_nowait() for _ in range(5)]
        self.assertEqual(len({r["id"] for r in requests}), 5)
        self.assertTrue(all(r["params"]["plain_proof"] == proof for r in requests))

    def test_notify_before_auth_is_held_and_split_frames_are_supported(self):
        sent_notify = threading.Event()
        allow_auth = threading.Event()
        finish = threading.Event()
        opening = []
        errors = []

        def script(peer):
            reader = peer.makefile("rb")
            read_json(reader)
            auth = read_json(reader)
            encoded = json.dumps(job_message()).encode() + b"\n"
            peer.sendall(encoded[:13])
            peer.sendall(encoded[13:])
            sent_notify.set()
            if not allow_auth.wait(2):
                raise AssertionError("auth gate was not released")
            send_json(peer, {"id": auth["id"], "result": True})
            finish.wait(2)
            reader.close()

        with MockPool(script) as pool:
            session = PoolSession(pool.settings())

            def open_session():
                try:
                    opening.append(session.open())
                except BaseException as error:
                    errors.append(error)

            worker = threading.Thread(target=open_session)
            worker.start()
            try:
                self.assertTrue(sent_notify.wait(1))
                self.assertIsNone(session.wait_job(0.03))
                self.assertFalse(opening)
                allow_auth.set()
                worker.join(1)
                self.assertEqual(errors, [])
                self.assertEqual(opening, [session])
                snapshot = session.wait_job(1)
                self.assertEqual(snapshot.job.job_id, "job-1")
            finally:
                allow_auth.set()
                finish.set()
                session.close()
                worker.join(1)

    def test_rotated_job_invalidates_snapshot_but_identical_notify_does_not(self):
        commands = queue.Queue()
        sent = threading.Event()

        def script(peer):
            reader, _ = authorize(peer)
            while True:
                command = commands.get(timeout=2)
                if command == "finish":
                    break
                send_json(peer, command)
                sent.set()
            reader.close()

        with MockPool(script) as pool, PoolSession(pool.settings()) as session:
            old = session.wait_job(1)
            commands.put(job_message())
            self.assertTrue(sent.wait(1))
            sent.clear()
            # A subsequent target-independent notification lets the reader
            # establish that the prior frame has been consumed.
            commands.put(job_message("job-2"))
            self.assertTrue(sent.wait(1))
            deadline = time.monotonic() + 1
            while session.is_current(old) and time.monotonic() < deadline:
                threading.Event().wait(0.005)
            new = session.wait_job(1)
            self.assertFalse(session.is_current(old))
            self.assertEqual(new.job.job_id, "job-2")
            self.assertEqual(new.generation, old.generation + 1)
            with self.assertRaises(StalePoolJob):
                session.submit(old, "cHJvb2Y=")
            commands.put("finish")

    def test_explicit_job_certificate_wins_over_fallback(self):
        finish = threading.Event()

        def script(peer):
            reader, _ = authorize(peer, notify=False)
            send_json(peer, job_message(cert=2))
            finish.wait(2)
            reader.close()

        with MockPool(script) as pool, PoolSession(pool.settings(certificate_version=3)) as session:
            self.assertEqual(session.wait_job(1).job.cert_version, 2)
            finish.set()

    def test_missing_certificate_can_use_explicit_fallback(self):
        finish = threading.Event()

        def script(peer):
            reader, _ = authorize(peer, notify=False)
            send_json(peer, job_message(cert=None))
            finish.wait(2)
            reader.close()

        with MockPool(script) as pool, PoolSession(pool.settings(certificate_version=3)) as session:
            self.assertEqual(session.wait_job(1).job.cert_version, 3)
            finish.set()

    def test_future_certificate_is_fatal_before_any_job_is_released(self):
        def script(peer):
            reader, _ = authorize(peer, notify=False)
            send_json(peer, job_message(cert=4))
            self.assertEqual(peer.recv(1), b"")
            reader.close()

        with MockPool(script) as pool:
            session = PoolSession(pool.settings())
            try:
                with self.assertRaisesRegex(PoolCompatibilityError, "certificate version 4"):
                    session.open()
                    session.wait_job(1)
                self.assertIsNone(session._job)
                self.assertFalse(session._authorized)
            finally:
                session.close()

    def test_auth_requires_literal_true_and_no_error(self):
        for response in [{"result": False}, {"result": 1}, {"result": "true"}, {"result": True, "error": {"message": "secret"}}]:
            with self.subTest(response=response):
                def script(peer):
                    reader = peer.makefile("rb")
                    read_json(reader)
                    auth = read_json(reader)
                    send_json(peer, {"id": auth["id"], **response})
                    reader.close()
                with MockPool(script) as pool:
                    with self.assertRaises(PoolAuthenticationError) as caught:
                        PoolSession(pool.settings()).open()
                    self.assertNotIn("secret", str(caught.exception))

    def test_malformed_json_and_nonobjects_close_session(self):
        for frame in [b"not-json\n", b"[]\n", b'{"id":2,"result":true,"result":false}\n', b'{"id":2,"result":NaN}\n']:
            with self.subTest(frame=frame):
                def script(peer):
                    reader = peer.makefile("rb")
                    read_json(reader)
                    read_json(reader)
                    peer.sendall(frame)
                    reader.close()
                with MockPool(script) as pool:
                    with self.assertRaises(PoolConnectionError):
                        PoolSession(pool.settings()).open()

    def test_oversized_inbound_frame_closes_session(self):
        def script(peer):
            reader = peer.makefile("rb")
            read_json(reader)
            read_json(reader)
            peer.sendall(b"x" * 100)
            reader.close()
        with MockPool(script) as pool, patch.object(PoolSession, "MAX_FRAME_BYTES", 99):
            # The outgoing subscribe/auth still fit the selected cap.
            with self.assertRaises(PoolConnectionError):
                PoolSession(pool.settings(KEX_uname="w", worker="")).open()

    def test_gzip_requires_matching_auth_ack(self):
        for acknowledge in [False, True]:
            with self.subTest(acknowledge=acknowledge):
                finish = threading.Event()
                def script(peer):
                    reader, auth = authorize(peer, gzip_reply=acknowledge)
                    self.assertEqual(auth["params"]["type"], "v2")
                    finish.wait(2)
                    reader.close()
                with MockPool(script) as pool, PoolSession(pool.settings(compression="gzip")) as session:
                    # Wait until the mock's complete initial job was sent
                    # before closing; the auth reply can wake open() first.
                    self.assertIsNotNone(session.wait_job(1))
                    self.assertEqual(session.negotiated_gzip, acknowledge)
                    finish.set()

    def test_concurrent_submissions_match_out_of_order_acknowledgements(self):
        finish = threading.Event()
        outcomes = {}
        errors = []

        def script(peer):
            reader, _ = authorize(peer)
            requests = [read_json(reader), read_json(reader)]
            for request in reversed(requests):
                accepted = request["params"]["plain_proof"] == "eWVz"
                send_json(peer, {"id": request["id"], "result": accepted})
            finish.wait(2)
            reader.close()

        with MockPool(script) as pool, PoolSession(pool.settings()) as session:
            snapshot = session.wait_job(1)

            def submit(proof):
                try:
                    outcomes[proof] = session.submit(snapshot, proof).accepted
                except BaseException as error:
                    errors.append(error)

            workers = [threading.Thread(target=submit, args=(proof,)) for proof in ["eWVz", "bm8="]]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(1)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertEqual(errors, [])
            self.assertEqual(outcomes, {"eWVz": True, "bm8=": False})
            finish.set()

    def test_authentication_deadline_closes_the_reader(self):
        def script(peer):
            reader = peer.makefile("rb")
            read_json(reader)
            read_json(reader)
            self.assertEqual(peer.recv(1), b"")
            reader.close()

        with MockPool(script) as pool:
            session = PoolSession(pool.settings(connect_timeout=0.05))
            with self.assertRaisesRegex(PoolConnectionError, "authentication timed out"):
                session.open()
            self.assertFalse(session._reader.is_alive())

    def test_standalone_difficulty_does_not_change_a_job_target(self):
        finish = threading.Event()

        def script(peer):
            reader, _ = authorize(peer)
            finish.wait(2)
            reader.close()

        with MockPool(script) as pool, PoolSession(pool.settings()) as session:
            snapshot = session.wait_job(1)
            session._handle({"method": "mining.set_difficulty", "params": [999999]})
            self.assertEqual(session.wait_job(0), snapshot)
            self.assertTrue(session.is_current(snapshot))
            finish.set()

    def test_disconnect_invalidates_job_and_unblocks_waiters(self):
        disconnect = threading.Event()
        def script(peer):
            reader, _ = authorize(peer)
            disconnect.wait(2)
            reader.close()
        with MockPool(script) as pool, PoolSession(pool.settings()) as session:
            snapshot = session.wait_job(1)
            disconnect.set()
            self.assertTrue(pool.finished.wait(1))
            deadline = time.monotonic() + 1
            while session.is_current(snapshot) and time.monotonic() < deadline:
                threading.Event().wait(0.005)
            self.assertFalse(session.is_current(snapshot))
            with self.assertRaises(PoolConnectionError):
                session.wait_job(1)

    def test_stop_event_wakes_reader_and_pending_submission(self):
        stop = threading.Event()
        request_received = threading.Event()
        finish = threading.Event()
        results = []
        def script(peer):
            reader, _ = authorize(peer)
            read_json(reader)
            request_received.set()
            finish.wait(2)
            reader.close()
        with MockPool(script) as pool:
            session = PoolSession(pool.settings(), stop).open()
            snapshot = session.wait_job(1)
            def submit():
                try:
                    results.append(session.submit(snapshot, "cHJvb2Y="))
                except PoolConnectionError as error:
                    results.append(error)
            worker = threading.Thread(target=submit)
            worker.start()
            try:
                self.assertTrue(request_received.wait(1))
                stop.set()
                worker.join(1)
                self.assertFalse(worker.is_alive())
                self.assertIsInstance(results[0], PoolConnectionError)
                self.assertFalse(session.is_current(snapshot))
            finally:
                finish.set()
                session.close()
                worker.join(1)
            self.assertFalse(session._reader.is_alive())

    def test_share_timeout_never_credits_late_ack(self):
        finish = threading.Event()
        def script(peer):
            reader, _ = authorize(peer)
            read_json(reader)
            finish.wait(2)
            reader.close()
        with MockPool(script) as pool, PoolSession(pool.settings()) as session:
            snapshot = session.wait_job(1)
            with self.assertRaisesRegex(PoolConnectionError, "acknowledgement timed out"):
                session.submit(snapshot, "cHJvb2Y=", timeout=0.03)
            session._handle({"id": 3, "result": True})
            self.assertEqual(session._pending, {})
            self.assertFalse(session.is_current(snapshot))
            finish.set()

    def test_unknown_subscribe_error_does_not_reject_auth(self):
        finish = threading.Event()
        def script(peer):
            reader = peer.makefile("rb")
            subscribe = read_json(reader)
            auth = read_json(reader)
            data = json.dumps({"id": subscribe["id"], "error": {"code": 20}}).encode() + b"\n"
            data += json.dumps({"id": auth["id"], "result": True}).encode() + b"\n"
            data += json.dumps(job_message()).encode() + b"\n"
            peer.sendall(data)
            finish.wait(2)
            reader.close()
        with MockPool(script) as pool, PoolSession(pool.settings()) as session:
            self.assertIsNotNone(session.wait_job(1))
            finish.set()

    def test_reconnect_does_not_follow_remote_redirect(self):
        finish = threading.Event()
        def script(peer):
            reader, _ = authorize(peer)
            finish.wait(2)
            reader.close()
        with MockPool(script) as pool, PoolSession(pool.settings()) as session:
            snapshot = session.wait_job(1)
            with patch("LRAEP_trainer.pool_client.PoolTransport") as transport:
                session._handle({"method": "client.reconnect", "params": ["untrusted.example", 9999]})
                transport.assert_not_called()
            self.assertFalse(session.is_current(snapshot))
            finish.set()


if __name__ == "__main__":
    unittest.main()
