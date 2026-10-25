"""Pool-loop state changes, plus a real GPU-to-local-pool proof exchange."""

import base64
import copy
from contextlib import ExitStack
import gzip
import itertools
import json
import socket
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from pearl_trainer.config import DEFAULT_CONFIG
from pearl_trainer import pool_training as worker
from pearl_trainer.pool_client import JobSnapshot, PoolAuthenticationError, ShareResult
from pearl_trainer.pool_protocol import PoolJob
from pearl_trainer.protocol import TrainingConfig


class PoolLoopTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.configuration = copy.deepcopy(DEFAULT_CONFIG)
        self.configuration["pool"].update(protocol="kryptex", wallet="test-account",
                                           worker="test-rig", reconnect_delay=0)
        self.configuration["training"].update(m=128, n=128, max_batches=1)
        self.job = PoolJob("job-one", bytes(76), 130000, 3, 1234)
        self.snapshot = JobSnapshot(self.job, 1)
        self.session = MagicMock()
        self.session.__enter__.return_value = self.session
        self.session.negotiated_gzip = False
        self.session.wait_job.return_value = self.snapshot
        self.session.is_current.return_value = True
        self.session.submit.return_value = ShareResult(True, None)
        self.factory = self.stack.enter_context(
            patch("pearl_trainer.pool_client.PoolSession", return_value=self.session)
        )
        self.stack.enter_context(patch.object(worker, "load_official_helpers"))
        self.stack.enter_context(patch.object(worker.torch.cuda, "is_available", return_value=True))
        self.stack.enter_context(patch.object(worker.torch.cuda, "get_device_name", return_value="test GPU"))
        self.monitor = self.stack.enter_context(patch.object(worker, "GpuUtilization"))
        self.monitor.return_value.start.return_value.read.return_value = 42
        self.progress = self.stack.enter_context(patch.object(worker, "LiveProgress"))
        self.stack.enter_context(patch.object(worker.native_cuda, "available", return_value=False))
        self.stack.enter_context(patch.object(worker, "pool_share_target", return_value=5678))
        self.prepare = self.stack.enter_context(patch.object(worker, "prepare_batch"))
        self.search = self.stack.enter_context(
            patch.object(worker, "search_batch", return_value=iter([(0, 0, bytes(32))]))
        )
        self.proof = MagicMock()
        self.proof.to_base64.return_value = "cHJvb2Y="
        self.verify = self.stack.enter_context(
            patch.object(worker, "verified_pool_share", return_value=(self.proof, bytes(32)))
        )
        self.header = SimpleNamespace(nbits=0x1D00FFFF)
        native = ModuleType("pearl_mining")
        native.IncompleteBlockHeader = SimpleNamespace(from_bytes=MagicMock(return_value=self.header))
        self.stack.enter_context(patch.dict(sys.modules, {"pearl_mining": native}))

    def test_completed_last_batch_verifies_and_submits_at_share_target(self):
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.search.assert_called_once()
        self.verify.assert_called_once_with(
            self.prepare.return_value, self.header, 3, 0, 0, bytes(32), 5678,
        )
        self.session.submit.assert_called_once_with(self.snapshot, "cHJvb2Y=")
        self.assertEqual(self.header.nbits, 0x1D00FFFF)

    def test_progress_reports_gpu_and_final_share_counters_without_batch_log_spam(self):
        self.configuration["training"]["max_batches"] = 3
        self.search.side_effect = lambda *_: iter([(0, 0, bytes(32))])
        with self.assertLogs(worker.LOG, level="INFO") as messages:
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        output = " ".join(messages.output)
        self.assertNotIn("Pool training batches=", output)
        self.assertIn("accepted=3", output)
        display = self.progress.return_value.__enter__.return_value
        last_status = display.update.call_args.args[0]
        self.assertIn("Hashrate(avg)", last_status)
        self.assertIn("GPU 42%", last_status)
        self.assertIn("accepted=3", last_status)
        self.assertIn("batches=3", last_status)
        self.monitor.return_value.start.return_value.close.assert_called_once()

    def test_telemetry_is_closed_when_pool_authentication_fails(self):
        self.session.__enter__.side_effect = PoolAuthenticationError("invalid account")
        with self.assertRaises(PoolAuthenticationError):
            worker.run_pool_training(self.configuration)
        self.monitor.return_value.start.return_value.close.assert_called_once()
        self.progress.return_value.__exit__.assert_called_once()

    def test_telemetry_uses_callers_current_device_for_implicit_cuda(self):
        self.configuration["device"] = "cuda"
        with patch.object(worker.torch.cuda, "current_device", return_value=2):
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.monitor.assert_called_once_with("cuda:2")

    def test_job_change_during_preparation_skips_old_gpu_work(self):
        next_job = PoolJob("job-two", bytes(76), 130001, 3, 1234)
        newer = JobSnapshot(next_job, 2)
        self.session.wait_job.side_effect = [self.snapshot, newer]
        self.session.is_current.side_effect = lambda item: item is newer
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(self.prepare.call_count, 2)
        self.search.assert_called_once()
        self.session.submit.assert_called_once_with(newer, "cHJvb2Y=")

    def test_job_change_after_gpu_work_discards_found_share(self):
        def search(*_):
            self.session.is_current.return_value = False
            return iter([(0, 0, bytes(32))])

        self.search.side_effect = search
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.verify.assert_not_called()
        self.session.submit.assert_not_called()

    def test_rejection_is_not_logged_as_accepted(self):
        self.session.submit.return_value = ShareResult(False, "low difficulty")
        with self.assertLogs(worker.LOG, level="INFO") as messages:
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        output = " ".join(messages.output)
        self.assertIn("rejected=1", output)
        self.assertIn("accepted=0", output)
        self.assertNotIn("Pool accepted training share", output)

    def test_authentication_failure_stops_without_reconnect_loop(self):
        self.session.__enter__.side_effect = PoolAuthenticationError("invalid account")
        with self.assertRaises(PoolAuthenticationError):
            worker.run_pool_training(self.configuration)
        self.factory.assert_called_once()
        self.prepare.assert_not_called()

    def test_login_check_requires_no_gpu_and_does_not_prepare_or_submit(self):
        with patch.object(worker.torch.cuda, "is_available", return_value=False):
            self.assertEqual(worker.run_pool_training(self.configuration, login_only=True), 0)
        self.prepare.assert_not_called()
        self.search.assert_not_called()
        self.session.submit.assert_not_called()
        self.monitor.assert_not_called()

    def test_pearlhash_never_falls_back_to_kryptex_dialect(self):
        self.configuration["pool"].update(protocol="auto", host="pool.pearlhash.xyz")
        with self.assertRaisesRegex(RuntimeError, "Pearlhash"):
            worker.run_pool_training(self.configuration)
        self.factory.assert_not_called()

    def test_time_limit_interrupts_a_waiting_authentication(self):
        from pearl_trainer.pool_transport import PoolConnectionError

        self.configuration["training"]["max_seconds"] = 0.05

        def wait_until_stopped():
            stop_event = self.factory.call_args.kwargs["stop_event"]
            if not stop_event.wait(1):
                raise AssertionError("time limit did not stop authentication")
            raise PoolConnectionError("stopped")

        self.session.__enter__.side_effect = wait_until_stopped
        started = time.monotonic()
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertLessEqual(self.factory.call_args.args[0]["connect_timeout"], 0.05)
        self.search.assert_not_called()

    def test_unsupported_pool_messages_stop_without_reconnect(self):
        from pearl_trainer.pool_client import PoolCompatibilityError

        self.session.wait_job.side_effect = PoolCompatibilityError("unsupported certificate")
        with self.assertRaises(PoolCompatibilityError):
            worker.run_pool_training(self.configuration)
        self.factory.assert_called_once()
        self.search.assert_not_called()


class PoolStatusTests(unittest.TestCase):
    def test_rate_uses_pearl_mac_work_units_instead_of_tile_attempts_or_double_ops(self):
        cfg = TrainingConfig(m=128, n=128, k=2048)
        stats = worker.PoolTrainingStats(batches=2, attempts=512, submitted=3,
                                         accepted=2, rejected=1, stale=4)
        status = worker.pool_status(stats, cfg, 1, 75)
        self.assertIn("Hashrate(avg) 67.11 MH/s", status)
        self.assertIn("GPU 75%", status)
        self.assertIn("accepted=2 rejected=1", status)
        self.assertIn("submitted=3 batches=2 attempts=512 stale=4", status)

    def test_missing_gpu_utilization_is_unavailable_and_zero_work_has_zero_rate(self):
        status = worker.pool_status(worker.PoolTrainingStats(), TrainingConfig(), 0, None)
        self.assertIn("0.00 H/s", status)
        self.assertIn("GPU N/A", status)


class LocalGPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        import torch
        from pearl_trainer.protocol import load_official_helpers

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        cls.root = Path(__file__).resolve().parents[1]
        try:
            load_official_helpers(cls.root / ".official" / "pearl")
            import pearl_mining as pm
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"official helper unavailable: {exc}") from None
        cls.pm = pm

    def check_exchange(self, compression, *, m=128, n=128, k=2048):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["official_source"] = str(self.root / ".official" / "pearl")
        cfg["training"].update(m=m, n=n, k=k, max_batches=1, max_seconds=15)
        cfg["pool"].update(protocol="kryptex", host="127.0.0.1", tls=False,
                           wallet="local-test-account", worker="gpu-test",
                           compression=compression, connect_timeout=3, read_timeout=5)
        header = self.pm.IncompleteBlockHeader(
            version=1, prev_block=bytes(32), merkle_root=bytes(32),
            timestamp=1234, nbits=0x1D00FFFF,
        )
        factor = 8 * 8 * k
        base_target = (2**256 - 1) // factor
        job = {"header": bytes(header.to_bytes()).hex(), "height": 130000,
               "job_id": "local-gpu-job", "cert_version": 3,
               "target": f"{base_target:064x}"}
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(10)
        cfg["pool"]["port"] = listener.getsockname()[1]
        observed = {}
        errors = []

        def serve():
            try:
                connection, _ = listener.accept()
                with connection, connection.makefile("rb") as stream:
                    connection.settimeout(10)
                    request = json.loads(stream.readline())
                    if request["method"] == "mining.subscribe":
                        # Real Kryptex may never acknowledge this request.
                        request = json.loads(stream.readline())
                    self.assertEqual(request["method"], "mining.authorize")
                    reply = {"id": request["id"], "result": True, "error": None}
                    if compression == "gzip":
                        self.assertEqual(request["params"]["type"], "v2")
                        reply["type"] = "v2"
                    connection.sendall((json.dumps(reply) + "\n" + json.dumps({
                        "method": "mining.notify", "params": job,
                    }) + "\n").encode())
                    share = json.loads(stream.readline())
                    self.assertEqual(share["method"], "mining.submit")
                    self.assertEqual(share["params"]["job_id"], "local-gpu-job")
                    self.assertEqual(share["params"]["worker"], "local-test-account.gpu-test")
                    raw = base64.b64decode(share["params"]["plain_proof"], validate=True)
                    if compression == "gzip":
                        raw = gzip.decompress(raw)
                    proof = self.pm.PlainProof.from_base64(base64.b64encode(raw).decode())
                    valid, message = self.pm.verify_plain_proof_for_cert_version(
                        3, header, proof, nbits_override=0x207FFFFF,
                    )
                    self.assertTrue(valid, message)
                    observed["proof_bytes"] = len(raw)
                    connection.sendall((json.dumps({
                        "id": share["id"], "result": True, "error": None,
                    }) + "\n").encode())
                    # Keep the connection open through the client's clean stop.
                    stream.readline()
            except BaseException as exc:
                errors.append(exc)

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        original_search = worker.search_batch

        def one_candidate(*args):
            # Keep the artificial easiest-target test to one real GPU candidate.
            return itertools.islice(original_search(*args), 1)

        with patch.object(worker, "search_batch", side_effect=one_candidate), \
                self.assertLogs(worker.LOG, level="INFO") as messages:
            result = worker.run_pool_training(cfg)
        server.join(timeout=3)
        self.assertFalse(server.is_alive(), "local pool server did not stop")
        if errors:
            raise errors[0]
        self.assertEqual(result, 0)
        self.assertGreater(observed["proof_bytes"], 1000)
        self.assertIn("accepted=1", " ".join(messages.output))

    def test_gpu_to_local_pool_native_plain_proof(self):
        self.check_exchange("none")

    def test_gpu_to_local_pool_negotiated_gzip_plain_proof(self):
        self.check_exchange("gzip")

    def test_gpu_to_local_pool_large_batch_native_plain_proof(self):
        self.check_exchange("none", m=4096, n=4096, k=16384)


if __name__ == "__main__":
    unittest.main()
