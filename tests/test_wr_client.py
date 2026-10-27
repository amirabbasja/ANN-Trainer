"""Concurrent PearlHash exchange tests with an in-memory transport, no pool/GPU."""

import json
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from LRAEP_trainer.pearl_codec import decode_frame, encode_frame, encode_payload
from LRAEP_trainer.pool_client import PoolAuthenticationError, PoolCompatibilityError, StalePoolJob
from LRAEP_trainer.pool_transport import PoolConnectionError
from LRAEP_trainer.pool_protocol import PoolProtocolError
from LRAEP_trainer.pool_diagnostics import UnknownRequestRecorder
from LRAEP_trainer.wr_client import WRSession

from test_pool_client import MockPool
from test_wr_protocol import compact_share, job_result


class MemoryTransport:
    connect_timeout = 1
    read_timeout = 1

    def __init__(self):
        self.incoming = queue.Queue()
        self.sent = []
        self.closed = threading.Event()
        self.after_send = None

    def open(self):
        return self

    def close(self):
        self.closed.set()
        self.incoming.put(None)

    def receive(self, maximum):
        try:
            value = self.incoming.get(timeout=2)
        except queue.Empty:
            raise PoolConnectionError("mock transport timed out") from None
        if value is None:
            raise PoolConnectionError("mock transport closed")
        return value

    def send(self, value):
        self.sent.append(decode_frame(value))
        if self.after_send is not None:
            self.after_send(self.sent[-1])

    def feed(self, message):
        self.incoming.put(encode_frame(message, 0x74F4785A))


class WRSessionTests(unittest.TestCase):
    def create_session(self, callback=None, **options):
        settings = dict(host="pool.example", port=9000, WR_wallet="test-wallet",
                        worker="test-worker", password="must-not-be-sent")
        settings.update(options)
        session = WRSession(settings, unknown_handler=callback)
        transport = session.transport = MemoryTransport()

        def normal_send(request):
            transport.feed({"id": request["id"], "status": 1, "result": True})
            if request["method"] == "login":
                # Combine and fragment server messages to exercise actual framing.
                frames = encode_frame({"id": -1, "status": 1,
                                       "result": ["extranonce", "19df092c6c027fee"]}, 0)
                frames += encode_frame({"id": -1, "status": 1, "result": job_result()}, 1)
                transport.incoming.put(frames[:13])
                transport.incoming.put(frames[13:])

        transport.after_send = normal_send
        self.addCleanup(session.close)
        return session, transport

    def test_encoded_login_and_correlated_submission(self):
        session, transport = self.create_session()
        with session:
            snapshot = session.wait_job(1)
            self.assertEqual(transport.sent[0], {"id": 1, "method": "login",
                                               "params": ["test-wallet", "test-worker", "LRAEP-trainer/0.1"]})
            self.assertEqual(snapshot.job.worker_nonce.hex(), "19df092c6c027fee")
            self.assertTrue(session.is_current(snapshot))
            self.assertTrue(session.submit(snapshot, compact_share(snapshot.job)).accepted)
            self.assertEqual(transport.sent[-1]["method"], "submit")
            self.assertFalse(session.negotiated_gzip)

    def test_gpu_report_uses_raw_rate_and_never_waits_for_acknowledgement(self):
        session, transport = self.create_session()
        with session:
            snapshot = session.wait_job(1)
            transport.after_send = None
            started = time.monotonic()
            request_id = session.report_info("NVIDIA GeForce GTX 1660 SUPER ", 7482132916255.0752)
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertEqual(transport.sent[-1], {
                "id": request_id, "method": "report_info",
                "params": [{"hashrate": 7482132916255.0752, "name": "NVIDIA GeForce GTX 1660 SUPER "}],
            })
            self.assertEqual(session.latest_report_id, request_id)
            self.assertIsNone(session.latest_report_accepted)
            self.assertEqual(session._pending, {})
            self.assertIn(request_id, session._report_pending)
            self.assertTrue(session.is_current(snapshot))
            transport.feed({"id": request_id, "status": 1, "result": True})
            with session._condition:
                self.assertTrue(session._condition.wait_for(
                    lambda: session.latest_report_accepted is True, timeout=1,
                ))
            self.assertEqual(session._pending, {})
            self.assertEqual(session._report_pending, {})

    def test_gpu_report_ack_cannot_unblock_or_accept_a_pending_share(self):
        session, transport = self.create_session()
        share_sent = threading.Event()
        requests, results, errors = [], [], []
        with session:
            snapshot = session.wait_job(1)
            transport.after_send = None
            report_id = session.report_info("Test GPU", 4000000000000.0)

            def capture(request):
                requests.append(request)
                share_sent.set()

            transport.after_send = capture

            def submit_share():
                try:
                    results.append(session.submit(snapshot, compact_share(snapshot.job)))
                except Exception as error:
                    errors.append(error)

            submission = threading.Thread(target=submit_share)
            submission.start()
            try:
                self.assertTrue(share_sent.wait(1))
                share_id = requests[-1]["id"]
                self.assertGreater(share_id, report_id)
                transport.feed({"id": report_id, "status": 1, "result": True})
                with session._condition:
                    self.assertTrue(session._condition.wait_for(
                        lambda: session.latest_report_accepted is True, timeout=1,
                    ))
                    self.assertFalse(session._pending[share_id].event.is_set())
                self.assertTrue(submission.is_alive())
                self.assertEqual(results, [])
                transport.feed({"id": share_id, "status": 1, "result": False})
                submission.join(1)
                self.assertEqual(errors, [])
                self.assertEqual([result.accepted for result in results], [False])
            finally:
                session.close()
                submission.join(1)
            self.assertFalse(submission.is_alive())

    def test_report_rejections_are_independent_and_late_older_reply_cannot_replace_latest(self):
        session, transport = self.create_session()
        with session:
            session.wait_job(1)
            transport.after_send = None
            first = session.report_info("GPU", 0.0)
            latest = session.report_info("GPU", 123456789.25)
            session._handle({"id": latest, "status": 1, "result": True})
            with self.assertLogs("LRAEP_trainer", level="WARNING") as messages:
                session._handle({"id": first, "status": 1, "result": False, "error": "PRIVATE_POOL_TEXT"})
            self.assertTrue(session.latest_report_accepted)
            self.assertEqual(session.latest_report_id, latest)
            self.assertEqual(session._report_pending, {})
            self.assertNotIn("PRIVATE_POOL_TEXT", " ".join(messages.output))
            rejected = session.report_info("GPU", 5.0)
            with self.assertLogs("LRAEP_trainer", level="WARNING"):
                session._handle({"id": rejected, "status": 0, "result": False})
            self.assertFalse(session.latest_report_accepted)

    def test_unacknowledged_reports_are_bounded_expire_and_do_not_stop_work(self):
        session, transport = self.create_session()
        with session:
            snapshot = session.wait_job(1)
            transport.after_send = None
            with self.assertLogs("LRAEP_trainer", level="WARNING"):
                ids = [session.report_info("GPU", float(rate)) for rate in range(session.MAX_PENDING_REPORTS + 2)]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(session._report_pending), session.MAX_PENDING_REPORTS)
            self.assertNotIn(ids[0], session._report_pending)
            with session._condition, \
                    patch("LRAEP_trainer.wr_client.time.monotonic", return_value=max(session._report_pending.values()) + 1), \
                    self.assertLogs("LRAEP_trainer", level="WARNING"):
                self.assertIsNone(session.latest_report_accepted)
            self.assertEqual(session._report_pending, {})
            session._handle({"id": ids[-1], "status": 1, "result": True})
            self.assertIsNone(session.latest_report_accepted)
            self.assertTrue(session.is_current(snapshot))
            transport.after_send = lambda request: transport.feed({"id": request["id"], "status": 1, "result": True})
            self.assertTrue(session.submit(snapshot, compact_share(snapshot.job)).accepted)

    def test_rapid_reports_and_out_of_order_acks_keep_latest_state_and_share_credit_separate(self):
        session, transport = self.create_session()
        share_sent = threading.Event()
        shares, results, errors = [], [], []
        with session:
            snapshot = session.wait_job(1)
            transport.after_send = None
            with self.assertLogs("LRAEP_trainer", level="WARNING"):
                report_ids = [session.report_info("GPU", float(rate) * 1e12)
                              for rate in range(session.MAX_PENDING_REPORTS + 2)]
            self.assertNotIn(report_ids[0], session._report_pending)
            self.assertEqual(len(session._report_pending), session.MAX_PENDING_REPORTS)

            def capture_share(request):
                shares.append(request)
                share_sent.set()

            transport.after_send = capture_share

            def submit_share():
                try:
                    results.append(session.submit(snapshot, compact_share(snapshot.job)))
                except Exception as error:
                    errors.append(error)

            submission = threading.Thread(target=submit_share)
            submission.start()
            try:
                self.assertTrue(share_sent.wait(1))
                share_id = shares[-1]["id"]
                self.assertGreater(share_id, report_ids[-1])
                ids = [request["id"] for request in transport.sent]
                self.assertEqual(len(ids), len(set(ids)))
                with self.assertLogs("LRAEP_trainer", level="WARNING"):
                    transport.feed({"id": report_ids[-1], "status": 1, "result": True})
                    transport.feed({"id": report_ids[0], "status": 1, "result": False})
                    transport.feed({"id": report_ids[-2], "status": 1, "result": False})
                    transport.feed({"id": report_ids[-1], "status": 1, "result": False})
                    with session._condition:
                        self.assertTrue(session._condition.wait_for(
                            lambda: session.latest_report_accepted is True, timeout=1,
                        ))
                        self.assertFalse(session._pending[share_id].event.is_set())
                    self.assertEqual(results, [])
                    self.assertTrue(submission.is_alive())
                    transport.feed({"id": share_id, "status": 1, "result": False})
                    submission.join(1)
                self.assertFalse(submission.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual([result.accepted for result in results], [False])
                self.assertEqual(session.latest_report_id, report_ids[-1])
                self.assertTrue(session.latest_report_accepted)
                self.assertTrue(session.is_current(snapshot))
            finally:
                session.close()
                submission.join(1)

    def test_reports_require_authentication_and_invalid_fields_do_not_reserve_ids(self):
        session, transport = self.create_session()
        with self.assertRaisesRegex(PoolConnectionError, "authenticate"):
            session.report_info("GPU", 0.0)
        self.assertEqual(transport.sent, [])
        with session:
            session.wait_job(1)
            before = session._next_id
            count = len(transport.sent)
            for name, rate in (("", 1.0), ("GPU", float("nan")), ("GPU", -1.0)):
                with self.assertRaises(PoolProtocolError):
                    session.report_info(name, rate)
            self.assertEqual(session._next_id, before)
            self.assertEqual(len(transport.sent), count)
            self.assertEqual(session._report_pending, {})

    def test_report_send_failure_closes_session_and_clears_report_pending(self):
        session, transport = self.create_session()
        with session:
            snapshot = session.wait_job(1)

            def fail_send(request):
                raise PoolConnectionError("mock send failed")

            transport.after_send = fail_send
            with self.assertRaises(PoolConnectionError):
                session.report_info("GPU", 12.0)
            self.assertTrue(transport.closed.is_set())
            self.assertEqual(session._report_pending, {})
            self.assertFalse(session.is_current(snapshot))

    def test_wr_session_requires_no_kex_identity(self):
        session, transport = self.create_session()
        self.assertNotIn("KEX_uname", session.settings)
        with session:
            self.assertEqual(transport.sent[0]["params"][0], "test-wallet")

    def test_wr_login_uses_only_wr_wallet_when_both_identities_exist(self):
        session, transport = self.create_session(KEX_uname="not-for-WR")
        with session:
            self.assertEqual(transport.sent[0]["params"][0], "test-wallet")
            self.assertNotIn("not-for-WR", json.dumps(transport.sent))

    def test_missing_wr_wallet_never_uses_kex_or_legacy_identity(self):
        for extra in ({"KEX_uname": "private-kex-name"},
                      {"wallet": "private-old-wallet"},
                      {"WR_wallet": "", "KEX_uname": "private-kex-name"}):
            with self.subTest(fields=tuple(extra)), self.assertRaisesRegex(PoolProtocolError, "WR_wallet") as caught:
                WRSession(dict(host="pool.example", port=9000, worker="test-worker", **extra))
            self.assertNotIn("private-", str(caught.exception))

    def test_rejected_login_is_auth_failure(self):
        session, transport = self.create_session()
        transport.after_send = lambda request: transport.feed({"id": request["id"], "status": 1, "result": False})
        with self.assertRaises(PoolAuthenticationError):
            session.open()
        self.assertTrue(transport.closed.is_set())

    def test_unknown_request_records_exact_frame_cancels_pending_and_stops(self):
        captured = []
        session, transport = self.create_session(lambda *args, **kwargs: captured.append((args, kwargs)))
        with session:
            old = session.wait_job(1)
            message = {"id": -1, "status": 1, "result": ["new_work_scheme", {"x": 1}]}
            frame = encode_frame(message, 15)
            transport.after_send = lambda request: transport.incoming.put(frame)
            with self.assertRaises(PoolCompatibilityError):
                session.submit(old, compact_share(old.job))
            self.assertTrue(session.stop_event.is_set())
            self.assertTrue(transport.closed.is_set())
            self.assertFalse(session.is_current(old))
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0][0][:2], (frame, message))
            self.assertEqual(captured[0][1], {"fatal": True})
            with self.assertRaises(PoolCompatibilityError):
                session.raise_if_failed()

    def test_malformed_frame_and_duplicate_json_are_recorded_and_fatal(self):
        for frame in [b"not-hex\n", encode_payload(b'{"status":1,"id":1,"id":2,"result":true}', 0)]:
            recorded = threading.Event()
            captured = []

            def record(*args, **kwargs):
                captured.append((args, kwargs))
                recorded.set()

            session, transport = self.create_session(record)
            with session:
                session.wait_job(1)
                transport.incoming.put(frame)
                self.assertTrue(recorded.wait(1))
                self.assertTrue(session.stop_event.wait(1))
                with self.assertRaises(PoolCompatibilityError):
                    session.wait_job(0)
                self.assertEqual(captured[0][0][:2], (frame, None))

    def test_disabled_stop_records_extensions_but_malformed_jobs_still_stop(self):
        recorded = threading.Event()
        captured = []

        def record(*args, **kwargs):
            captured.append((args, kwargs))
            recorded.set()

        session, transport = self.create_session(record, stop_on_unknown_request=False)
        with session:
            old = session.wait_job(1)
            transport.feed({"method": "new_extension", "params": []})
            self.assertTrue(recorded.wait(1))
            self.assertEqual(captured[-1][1], {"fatal": False})
            self.assertFalse(session.stop_event.is_set())
            self.assertTrue(session.is_current(old))
            recorded.clear()
            bad_job = list(job_result())
            bad_job[4] = "4"
            transport.feed({"status": 1, "id": -1, "result": bad_job})
            self.assertTrue(recorded.wait(1))
            self.assertTrue(session.stop_event.wait(1))
            self.assertEqual(captured[-1][1], {"fatal": True})

    def test_new_extranonce_invalidates_snapshot_and_binds_new_share(self):
        session, transport = self.create_session()
        with session:
            old = session.wait_job(1)
            transport.feed({"id": -1, "status": 1, "result": ["extranonce", "0102030405060708"]})
            with session._condition:
                self.assertTrue(session._condition.wait_for(
                    lambda: session._worker_nonce == bytes.fromhex("0102030405060708"), timeout=1,
                ))
            new = session.wait_job(1)
            self.assertFalse(session.is_current(old))
            self.assertEqual(new.job.worker_nonce.hex(), "0102030405060708")
            self.assertGreater(new.generation, old.generation)
            with self.assertRaises(StalePoolJob):
                session.submit(old, compact_share(old.job))

    def test_recorder_failure_cannot_reconnect_or_resume_work(self):
        def broken(*args, **kwargs):
            raise OSError("disk is full")

        session, transport = self.create_session(broken)
        with session:
            snapshot = session.wait_job(1)
            transport.after_send = lambda request: transport.feed({"method": "unexpected"})
            with self.assertRaisesRegex(PoolCompatibilityError, "diagnostic recording failed"):
                session.submit(snapshot, compact_share(snapshot.job))
            self.assertTrue(session.stop_event.is_set())

    def test_unterminated_frame_before_disconnect_is_recorded_and_fatal(self):
        recorded = threading.Event()
        captured = []

        def record(*args, **kwargs):
            captured.append((args, kwargs))
            recorded.set()

        session, transport = self.create_session(record)
        with session:
            snapshot = session.wait_job(1)
            partial = encode_frame({"method": "unknown"}, 0)[:-3]
            transport.incoming.put(partial)
            transport.incoming.put(None)
            self.assertTrue(recorded.wait(1))
            self.assertTrue(session.stop_event.wait(1))
            self.assertEqual(captured[0][0][:2], (partial, None))
            with self.assertRaises(PoolCompatibilityError):
                session.wait_job(0)
            self.assertFalse(session.is_current(snapshot))

    def test_work_is_invalidated_before_slow_diagnostic_persistence(self):
        recording, release = threading.Event(), threading.Event()

        def slow_record(*args, **kwargs):
            recording.set()
            if not release.wait(2):
                raise AssertionError("test did not release diagnostic recorder")

        session, transport = self.create_session(slow_record)
        with session:
            snapshot = session.wait_job(1)
            try:
                transport.feed({"method": "unknown"})
                self.assertTrue(recording.wait(1))
                self.assertFalse(session.is_current(snapshot))
                self.assertFalse(session.stop_event.is_set())
                with self.assertRaises(PoolCompatibilityError):
                    session.submit(snapshot, compact_share(snapshot.job))
                release.set()
                self.assertTrue(session.stop_event.wait(1))
                with self.assertRaises(PoolCompatibilityError):
                    session.raise_if_failed()
            finally:
                release.set()

    def test_external_stop_during_recording_cannot_hide_compatibility_failure(self):
        recording, release = threading.Event(), threading.Event()

        def slow_record(*args, **kwargs):
            recording.set()
            if not release.wait(2):
                raise AssertionError("test did not release diagnostic recorder")

        session, transport = self.create_session(slow_record)
        with session:
            snapshot = session.wait_job(1)
            try:
                transport.feed({"method": "unknown"})
                self.assertTrue(recording.wait(1))
                session.stop_event.set()
                self.assertTrue(transport.closed.wait(1))
                self.assertFalse(session.is_current(snapshot))
                release.set()
                with self.assertRaises(PoolCompatibilityError):
                    session.raise_if_failed()
            finally:
                release.set()

    def test_tcp_unknown_message_preserves_evidence_and_cancels_active_work(self):
        active, cancelled = threading.Event(), threading.Event()
        unknown = {"id": -1, "status": 1, "result": ["unfamiliar-request", "payload"]}
        frame = encode_frame(unknown, 19)

        def script(peer):
            reader = peer.makefile("rb")
            try:
                login = decode_frame(reader.readline())
                peer.sendall(encode_frame({"id": login["id"], "status": 1, "result": True}, 0))
                peer.sendall(encode_frame({"id": -1, "status": 1,
                                           "result": ["extranonce", "19df092c6c027fee"]}, 1))
                peer.sendall(encode_frame({"id": -1, "status": 1, "result": job_result()}, 2))
                if not active.wait(2):
                    raise AssertionError("work did not start")
                peer.sendall(frame[:7])
                peer.sendall(frame[7:])
                self.assertEqual(reader.read(1), b"")
            finally:
                reader.close()

        with tempfile.TemporaryDirectory() as directory, MockPool(script) as pool:
            settings = pool.settings()
            settings.pop("KEX_uname", None)
            settings["WR_wallet"] = "test-wallet"
            recorder = UnknownRequestRecorder(directory, settings)
            session = WRSession(settings, unknown_handler=recorder)
            with self.assertLogs("LRAEP_trainer", level="CRITICAL"), session:
                snapshot = session.wait_job(1)

                def simulated_work():
                    active.set()
                    while session.is_current(snapshot) and not session.stop_event.is_set():
                        cancelled.wait(0.001)
                    cancelled.set()

                worker = threading.Thread(target=simulated_work)
                worker.start()
                try:
                    self.assertTrue(session.stop_event.wait(1))
                    self.assertTrue(cancelled.wait(1))
                    with self.assertRaises(PoolCompatibilityError):
                        session.raise_if_failed()
                    self.assertFalse(session.is_current(snapshot))
                    record = json.loads(recorder.last_path.read_text())
                    self.assertEqual(record["raw_frame_hex"], frame.hex())
                    self.assertEqual(record["decoded_message"], unknown)
                    self.assertTrue(record["training_stopped"])
                finally:
                    session.close()
                    worker.join(1)
                self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
