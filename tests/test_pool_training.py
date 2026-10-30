"""Pool-loop state changes, plus a real GPU-to-local-pool proof exchange."""

import base64
import copy
from contextlib import ExitStack, contextmanager
import gzip
import itertools
import json
import socket
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, call, patch

from LRAEP_trainer.config import DEFAULT_CONFIG
import LRAEP_trainer
from LRAEP_trainer import gpu_pacing, pool_training as worker
from LRAEP_trainer.pool_client import JobSnapshot, PoolAuthenticationError, ShareResult
from LRAEP_trainer.pool_protocol import PoolJob
from LRAEP_trainer.protocol import TrainingConfig


class PoolLoopTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.configuration = copy.deepcopy(DEFAULT_CONFIG)
        self.configuration["pool"].update(protocol="KEX", host="pool.example", KEX_uname="test-account",
                                           WR_wallet="wr-test-wallet",
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
            patch("LRAEP_trainer.pool_client.PoolSession", return_value=self.session)
        )
        self.stack.enter_context(patch.object(worker, "load_official_helpers"))
        self.stack.enter_context(patch.object(worker.torch.cuda, "is_available", return_value=True))
        self.stack.enter_context(patch.object(worker.torch.cuda, "get_device_name", return_value="test GPU"))
        self.stack.enter_context(patch.object(worker.torch.cuda, "synchronize"))
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
        native = ModuleType("LRAEP_trainer.primitives")
        native.IncompleteBlockHeader = SimpleNamespace(from_bytes=MagicMock(return_value=self.header))
        self.stack.enter_context(patch.dict(sys.modules, {"LRAEP_trainer.primitives": native}))
        self.stack.enter_context(patch.object(LRAEP_trainer, "primitives", native, create=True))

    def test_completed_last_batch_verifies_and_submits_at_share_target(self):
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.search.assert_called_once()
        self.verify.assert_called_once_with(
            self.prepare.return_value, self.header, 3, 0, 0, bytes(32), 5678,
        )
        self.session.submit.assert_called_once_with(self.snapshot, "cHJvb2Y=")
        self.assertEqual(self.header.nbits, 0x1D00FFFF)

    def test_gpu_pacing_runs_between_batches_after_share_submission(self):
        self.configuration["training"].update(max_batches=2, job_sequence=[("main", 60, 60)])
        order = []
        self.session.submit.side_effect = lambda *_: (order.append("submit"), ShareResult(True, None))[1]
        self.search.side_effect = lambda *_: (order.append("search"), iter([(0, 0, bytes(32))]))[1]
        with patch.object(worker, "GpuPacer") as pacer:
            pacer.return_value.max_utilization = 100
            pacer.return_value.set_utilization.side_effect = lambda value: setattr(pacer.return_value, "max_utilization", value)
            pacer.return_value.wait.side_effect = lambda *_a, **_kw: order.append("wait")
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(order, ["search", "submit", "wait", "search", "submit"])
        pacer.assert_called_once_with(worker.torch.device("cuda:0"), 100)
        pacer.return_value.set_utilization.assert_called_once_with(60)
        self.assertEqual(pacer.return_value.begin.call_count, 2)
        self.assertEqual(pacer.return_value.finish.call_count, 2)
        pacer.return_value.wait.assert_called_once()

    @contextmanager
    def clocked_schedule(self, schedule, *, batches=1, max_seconds=0):
        """Drive the real loop and scheduler without sleeping or touching CUDA."""
        clock = SimpleNamespace(now=0.0, pauses=[], on_wait=lambda: None)
        outer = self

        class ClockEvent:
            stopped = False

            def is_set(self):
                return self.stopped

            def set(self):
                self.stopped = True

            def wait(self, timeout=None):
                if self.stopped:
                    return True
                outer.assertIsNotNone(timeout, "test schedule waited indefinitely")
                clock.on_wait()
                clock.pauses.append((clock.now, timeout, outer.prepare.call_count,
                                     outer.search.call_count, outer.verify.call_count))
                clock.now += timeout
                return self.stopped

        stopped = ClockEvent()
        self.configuration["training"].update(job_sequence=schedule, max_batches=batches,
                                                max_seconds=max_seconds)
        self.search.side_effect = lambda *_: iter([(0, 0, bytes(32))])
        with patch.object(worker.time, "monotonic", side_effect=lambda: clock.now), \
                patch.object(worker.threading, "Event", return_value=stopped), \
                patch.object(worker.threading, "Timer"), \
                patch.object(worker, "GpuPacer") as factory:
            pacer = factory.return_value
            pacer.max_utilization = 100
            pacer.set_utilization.side_effect = lambda value: setattr(pacer, "max_utilization", value)
            yield clock, pacer

    def test_100_percent_prepare_cancellation_drains_cuda_before_pause(self):
        with self.clocked_schedule([("main", 10, 100), ("main", 20, 0), ("main", 10, 100)]) as (clock, _):
            order = []
            clock.on_wait = lambda: order.append("pause")

            def prepare(*_a, **_kw):
                order.append("prepare")
                if self.prepare.call_count == 1:
                    clock.now = 10

            self.prepare.side_effect = prepare
            worker.torch.cuda.synchronize.side_effect = lambda _device: order.append("synchronize")
            with patch.object(worker, "GpuPacer", gpu_pacing.GpuPacer):
                self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(order, ["prepare", "synchronize", "pause", "prepare"])
        worker.torch.cuda.synchronize.assert_called_once_with(worker.torch.device("cuda:0"))
        self.search.assert_called_once()
        self.verify.assert_called_once()
        self.session.submit.assert_called_once()

    def test_100_percent_search_cancellation_drains_cuda_before_exit_cleanup(self):
        with self.clocked_schedule([("main", 10, 100), ("main", 0, -1)]) as (clock, _):
            order = []

            def search(*_):
                order.append("search")
                clock.now = 10
                raise worker.TrainingCancelled("schedule interval ended")

            self.search.side_effect = search
            worker.torch.cuda.synchronize.side_effect = lambda _device: order.append("synchronize")
            self.monitor.return_value.start.return_value.close.side_effect = lambda: order.append("close")
            with patch.object(worker, "GpuPacer", gpu_pacing.GpuPacer):
                self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(order, ["search", "synchronize", "close"])
        worker.torch.cuda.synchronize.assert_called_once_with(worker.torch.device("cuda:0"))
        self.session.__exit__.assert_called_once()
        self.verify.assert_not_called()
        self.session.submit.assert_not_called()

    def test_initial_pause_delays_mining_until_the_positive_interval(self):
        with self.clocked_schedule([("main", 30, 0), ("main", 60, 70)]) as (clock, pacer):
            prepared_at = []
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(prepared_at, [30])
        self.assertEqual(clock.pauses, [(0, 30, 0, 0, 0)])
        pacer.set_utilization.assert_called_once_with(70)
        self.session.submit.assert_called_once()

    def test_pause_between_positive_intervals_performs_no_mining(self):
        with self.clocked_schedule([("main", 10, 80), ("main", 20, 0), ("main", 10, 30)], batches=2) as (clock, pacer):
            prepared_at = []
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)
            pacer.wait.side_effect = lambda *_a, **kw: setattr(clock, "now", kw["deadline"])
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(prepared_at, [0, 30])
        self.assertEqual(clock.pauses, [(10, 20, 1, 1, 1)])
        self.assertEqual(pacer.set_utilization.call_args_list, [call(80), call(30)])
        self.assertEqual(self.session.submit.call_count, 2)

    def test_schedule_switches_effort_and_repeats_from_the_first_interval(self):
        with self.clocked_schedule([("main", 10, 80), ("side-job", 20, 70)], batches=5) as (clock, pacer), \
                patch("LRAEP_trainer.side_job.SideJob") as cnn:
            prepared_at = []
            side_job_at = []
            statuses = []
            display = self.progress.return_value.__enter__.return_value
            display.update.side_effect = lambda text, **_: statuses.append((clock.now, text))
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)
            cnn.return_value.step.side_effect = lambda **_kw: (side_job_at.append(clock.now), True)[1]
            cnn.return_value.status.side_effect = lambda *_: f"Side-job | Epoch 1 | Step {cnn.return_value.step.call_count}/20"

            def pace(*_a, **kw):
                if clock.now % 30 == 10:
                    self.assertEqual(statuses[-1][1],
                                     f"Side-job | Epoch 1 | Step {cnn.return_value.step.call_count}/20")
                clock.now = kw["deadline"]

            pacer.wait.side_effect = pace
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(prepared_at, [0, 30, 60, 90, 120])
        self.assertEqual(side_job_at, [10, 40, 70, 100])
        self.assertEqual(self.session.wait_job.call_count, 5)
        self.assertEqual(pacer.set_utilization.call_args_list,
                         [call(80), call(70), call(80), call(70), call(80),
                          call(70), call(80), call(70), call(80)])
        self.assertEqual(self.search.call_count, 5)
        self.assertEqual(self.session.submit.call_count, 5)
        cnn.assert_called_once_with(self.configuration, before_log=display.clear)
        self.assertTrue(any(now == 10 and text == "Side-job | Epoch 1 | Step 1/20"
                            for now, text in statuses))
        self.assertTrue(any(now == 30 and text.startswith("Main | Step 2/5")
                            for now, text in statuses), "main progress must resume after a side-job interval")
        cnn.return_value.close.assert_called_once()

    def test_initial_side_job_trains_before_pool_connection_or_job_wait(self):
        with self.clocked_schedule([("side-job", 10, 70), ("main", 20, 80)]) as (clock, _), \
                patch("LRAEP_trainer.side_job.SideJob") as cnn:
            def side_job_batch(**_kw):
                self.factory.assert_not_called()
                self.session.wait_job.assert_not_called()
                self.prepare.assert_not_called()
                display = self.progress.return_value.__enter__.return_value
                self.assertEqual(display.update.call_args.args[0], "Side-job | Epoch 1 | Preparing")
                clock.now = 10
                return True

            cnn.return_value.step.side_effect = side_job_batch
            cnn.return_value.status.return_value = "Side-job | Epoch 1 | Preparing"
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.prepare.assert_called_once()
        self.session.submit.assert_called_once()
        cnn.return_value.step.assert_called_once()
        cnn.return_value.close.assert_called_once()

    def test_role_transition_cancels_old_preparation_at_unchanged_gpu_cap(self):
        with self.clocked_schedule([("main", 10, 70), ("side-job", 20, 70)]) as (clock, pacer), \
                patch("LRAEP_trainer.side_job.SideJob") as cnn:
            cnn.return_value.step.side_effect = lambda **_kw: (setattr(clock, "now", 30), True)[1]

            def prepare(*_a, **_kw):
                if self.prepare.call_count == 1:
                    clock.now = 10

            self.prepare.side_effect = prepare
            with self.assertLogs(worker.LOG, level="INFO") as messages:
                self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(self.prepare.call_count, 2)
        self.search.assert_called_once()
        self.session.submit.assert_called_once()
        self.assertEqual(pacer.set_utilization.call_args_list, [call(70)])
        self.assertIn("Main effort GPU duty-cycle target: 70%", " ".join(messages.output))
        self.assertIn("Side-job effort GPU duty-cycle target: 70%", " ".join(messages.output))

    def test_zero_effort_until_time_limit_does_not_initialize_or_mine(self):
        with self.clocked_schedule([("main", 30, 0)], max_seconds=10) as (clock, pacer):
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(clock.now, 10)
        worker.load_official_helpers.assert_not_called()
        self.factory.assert_not_called()
        self.prepare.assert_not_called()
        self.search.assert_not_called()
        self.verify.assert_not_called()
        self.session.submit.assert_not_called()
        self.monitor.assert_not_called()
        pacer.begin.assert_not_called()

    def test_terminal_first_interval_exits_before_helpers_cuda_or_sessions(self):
        self.configuration["training"]["job_sequence"] = [("main", 0, -1), ("main", 60, 80)]
        with patch.object(worker.torch.cuda, "is_available") as cuda:
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        cuda.assert_not_called()
        worker.load_official_helpers.assert_not_called()
        self.factory.assert_not_called()
        self.prepare.assert_not_called()
        self.search.assert_not_called()
        self.verify.assert_not_called()
        self.monitor.assert_not_called()

    def test_terminal_interval_interrupts_blocked_authentication_and_cleans_up(self):
        self.configuration["training"]["job_sequence"] = [("main", 0.3, 80), ("main", 0, -1)]

        def wait_for_exit():
            stopped = self.factory.call_args.kwargs["stop_event"]
            self.assertTrue(stopped.wait(2), "schedule did not stop authentication")
            return self.session

        self.session.__enter__.side_effect = wait_for_exit
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.factory.assert_called_once()
        self.assertLessEqual(self.factory.call_args.args[0]["connect_timeout"], 0.3)
        self.session.__exit__.assert_called_once()
        self.monitor.return_value.start.return_value.close.assert_called_once()
        self.progress.return_value.__exit__.assert_called_once()
        self.prepare.assert_not_called()
        self.search.assert_not_called()
        self.session.submit.assert_not_called()

    def test_terminal_interval_interrupts_blocked_share_submission_and_cleans_up(self):
        from LRAEP_trainer.pool_transport import PoolConnectionError

        self.configuration["training"]["job_sequence"] = [("main", 0.3, 80), ("main", 0, -1)]

        def wait_for_exit(*_):
            stopped = self.factory.call_args.kwargs["stop_event"]
            self.assertTrue(stopped.wait(2), "schedule did not stop share submission")
            raise PoolConnectionError("scheduled exit")

        self.session.submit.side_effect = wait_for_exit
        self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.factory.assert_called_once()
        self.session.submit.assert_called_once()
        self.session.__exit__.assert_called_once()
        self.monitor.return_value.start.return_value.close.assert_called_once()
        self.progress.return_value.__exit__.assert_called_once()

    def test_job_sequence_creates_one_shared_schedule(self):
        self.configuration["training"].update(job_sequence=[("main", 60, 70)])
        with patch.object(worker, "JobSequence", wraps=worker.JobSequence) as schedules:
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        schedules.assert_called_once()
        self.assertEqual(schedules.call_args.args[0], [("main", 60, 70)])
        self.prepare.assert_called_once()
        self.search.assert_called_once()
        self.session.submit.assert_called_once()

    def test_progress_reports_training_steps_and_gpu_without_mining_counter_logs(self):
        self.configuration["training"]["max_batches"] = 3
        self.search.side_effect = lambda *_: iter([(0, 0, bytes(32))])
        with self.assertLogs(worker.LOG, level="INFO") as messages:
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        output = " ".join(messages.output)
        self.assertIn("Main job stopped: steps=3 elapsed=", output)
        self.assertNotIn("Pool accepted training share", output)
        display = self.progress.return_value.__enter__.return_value
        last_status = display.update.call_args.args[0]
        self.assertIn("Main | Step 3/3 (100.0%)", last_status)
        self.assertIn("GPU 42%", last_status)
        self.assertIn("elapsed=", last_status)
        for text in (output, last_status):
            for label in ("Hashrate", "accepted=", "rejected=", "submitted=", "batches=", "attempts=", "stale="):
                self.assertNotIn(label, text)
        self.assertEqual(self.session.submit.call_count, 3)
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

    def test_cancelled_search_is_paced_before_retrying(self):
        self.search.side_effect = [worker.TrainingCancelled("stale"), iter([])]
        with patch.object(worker, "GpuPacer") as pacer:
            pacer.return_value.max_utilization = 80
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        self.assertEqual(pacer.return_value.finish.call_count, 2)
        pacer.return_value.wait.assert_called_once()

    def test_rejection_is_not_logged_as_accepted(self):
        self.session.submit.return_value = ShareResult(False, "low difficulty")
        with self.assertLogs(worker.LOG, level="INFO") as messages:
            self.assertEqual(worker.run_pool_training(self.configuration), 0)
        output = " ".join(messages.output)
        self.assertIn("Pool rejected training share", output)
        self.assertIn("low difficulty", output)
        self.assertNotIn("Pool accepted training share", output)
        self.session.submit.assert_called_once()

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

    def test_pearlhash_login_selects_its_own_adapter_without_gpu(self):
        self.configuration["pool"].update(protocol="WR", host="pool.example")
        with patch("LRAEP_trainer.wr_client.WRSession", return_value=self.session) as wr:
            self.assertEqual(worker.run_pool_training(self.configuration, login_only=True), 0)
        wr.assert_called_once()
        self.factory.assert_not_called()
        self.prepare.assert_not_called()

    def test_wr_alias_and_host_selection(self):
        for protocol, expected in (("WR", "WR"), ("pearlhash", "WR"), ("KEX", "KEX")):
            for host in ("pool.example", "localhost", "127.0.0.1"):
                with self.subTest(protocol=protocol, host=host):
                    self.assertEqual(worker.select_pool_protocol({"host": host, "protocol": protocol}), expected)
        with self.assertRaisesRegex(ValueError, "explicitly select KEX or WR"):
            worker.select_pool_protocol({"host": "pool.example", "protocol": "auto"})
        for protocol in ("kex", "wr", "unknown", None):
            with self.subTest(protocol=protocol), self.assertRaisesRegex(ValueError, "KEX or WR"):
                worker.select_pool_protocol({"host": "pool.example", "protocol": protocol})

    def test_missing_pool_endpoint_stops_before_helpers_or_session(self):
        self.configuration["pool"]["host"] = ""
        with self.assertRaisesRegex(ValueError, "fill pool.host"):
            worker.run_pool_training(self.configuration)
        worker.load_official_helpers.assert_not_called()
        self.factory.assert_not_called()
        self.prepare.assert_not_called()

    def test_unknown_wr_request_stop_cannot_be_reported_as_normal_completion(self):
        from LRAEP_trainer.pool_client import PoolCompatibilityError

        self.configuration["pool"].update(protocol="WR", host="pool.example")
        def start(_settings, stop_event, **_options):
            self.prepare.side_effect = lambda *_a, **_kw: stop_event.set()
            self.session.raise_if_failed.side_effect = PoolCompatibilityError("Unknown PearlHash request; training stopped")
            return self.session
        with patch("LRAEP_trainer.wr_client.WRSession", side_effect=start), \
                patch("LRAEP_trainer.wr_work.prepare_wr_batch", self.prepare), \
                self.assertRaisesRegex(PoolCompatibilityError, "Unknown PearlHash"):
            worker.run_pool_training(self.configuration)
        self.factory.assert_not_called()

    def test_time_limit_interrupts_a_waiting_authentication(self):
        from LRAEP_trainer.pool_transport import PoolConnectionError

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
        from LRAEP_trainer.pool_client import PoolCompatibilityError

        self.session.wait_job.side_effect = PoolCompatibilityError("unsupported certificate")
        with self.assertRaises(PoolCompatibilityError):
            worker.run_pool_training(self.configuration)
        self.factory.assert_called_once()
        self.search.assert_not_called()


class PoolStatusTests(unittest.TestCase):
    def test_unbounded_steps_use_training_format_and_elapsed_time(self):
        cfg = TrainingConfig(m=128, n=128, k=2048)
        stats = worker.PoolTrainingStats(batches=2, attempts=512, submitted=3,
                                         accepted=2, rejected=1, stale=4)
        status = worker.pool_status(stats, cfg, 3723, 75)
        self.assertEqual(status, "Main | Step 2 | elapsed=01:02:03 | GPU 75%")

    def test_configured_batch_limit_has_true_step_progress(self):
        status = worker.pool_status(worker.PoolTrainingStats(batches=2), TrainingConfig(), 1, 75,
                                    max_batches=10)
        self.assertEqual(status, "Main | Step 2/10 (20.0%) | elapsed=00:00:01 | GPU 75%")

    def test_missing_gpu_utilization_and_unstarted_work(self):
        status = worker.pool_status(worker.PoolTrainingStats(), TrainingConfig(), 0, None)
        self.assertEqual(status, "Main | Step 0 | elapsed=00:00:00 | GPU N/A")


class LocalGPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        import torch
        from LRAEP_trainer.protocol import load_official_helpers

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        cls.root = Path(__file__).resolve().parents[1]
        try:
            load_official_helpers(cls.root / "src" / "LRAEP_helpers")
            from LRAEP_trainer import primitives as pm
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"official helper unavailable: {exc}") from None
        cls.pm = pm

    def check_exchange(self, compression, *, m=128, n=128, k=2048):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["official_source"] = str(self.root / "src" / "LRAEP_helpers")
        cfg["training"].update(m=m, n=n, k=k, max_batches=1, max_seconds=15)
        cfg["pool"].update(protocol="KEX", host="127.0.0.1", tls=False,
                           KEX_uname="local-test-account", worker="gpu-test",
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
                        # Real KEX may never acknowledge this request.
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
                self.assertLogs(worker.LOG, level="DEBUG") as messages:
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
