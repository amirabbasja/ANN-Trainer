"""Configuration and duty-cycle checks for the standalone WR trainer."""

import copy
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, call, patch

import LRAEP_trainer
from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, load_config, save_config
from LRAEP_trainer.config_export import write_open_text
from LRAEP_trainer import gpu_pacing, pool_training as worker


class GpuLimitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "config.json"
        write_open_text(self.path.parent / ".env", "unit test text key")
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["vpn"]["link"] = "vless://dummy@example.invalid:443"
        self.config["side_job"]["repo_url"] = "https://huggingface.co/datasets/test-owner/test-thesis"

    def test_encrypted_config_without_effort_fields_receives_default_schedule(self):
        self.config["training"].pop("job_sequence")
        save_config(self.path, self.config)
        training = load_config(self.path)["training"]
        self.assertEqual(training["job_sequence"], [("main", 60, 80), ("main", 120, 70), ("main", 180, 10)])
        self.assertNotIn("side_job_effort", training)
        self.assertNotIn("max_gpu_utilization", training)

    def test_encrypted_config_preserves_mixed_job_sequence(self):
        self.config["training"]["job_sequence"] = [
            ["main", 0.5, 100], ["side-job", 2, 10], ["main", 3, 0], ["side-job", 0, -1]]
        save_config(self.path, self.config)
        training = load_config(self.path)["training"]
        self.assertEqual(training["job_sequence"],
                         [("main", 0.5, 100), ("side-job", 2, 10), ("main", 3, 0), ("side-job", 0, -1)])
        self.assertNotIn("main_effort", training)
        self.assertNotIn("side_job_effort", training)

    def test_legacy_gpu_limit_becomes_constant_schedule(self):
        self.config["training"].pop("job_sequence")
        self.config["training"]["max_gpu_utilization"] = 70
        save_config(self.path, self.config)
        training = load_config(self.path)["training"]
        self.assertEqual(training["job_sequence"], [("main", 60, 70)])
        self.assertNotIn("max_gpu_utilization", training)

    def test_explicit_job_sequence_takes_precedence_over_legacy_limit(self):
        self.config["training"].update(job_sequence=[["main", 30, 40]], max_gpu_utilization=70)
        save_config(self.path, self.config)
        self.assertEqual(load_config(self.path)["training"]["job_sequence"], [("main", 30, 40)])

    def test_invalid_legacy_limits_are_rejected_even_with_main_schedule(self):
        for value in (0, 101, -1, True, 70.5, "70", None):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "training.max_gpu_utilization"):
                self.config["training"]["max_gpu_utilization"] = value
                save_config(self.path, self.config)

    def test_job_sequence_validation_applies_before_encrypted_save(self):
        malformed = (None, True, "PRIVATE_SEQUENCE", {}, [], ["main", 1, 80],
                     [[1, 80]], [["main"]], [["main", 1]], [["main", 1, 80, 2, 0]],
                     [["PRIVATE_SEQUENCE", 1, 80]], [["MAIN", 1, 80]], [["Side-job", 1, 80]],
                     [[True, 1, 80]], [[None, 1, 80]], [[[], 1, 80]],
                     [["main", 0, 80]], [["side-job", -1, -1]], [["main", True, 80]],
                     [["main", "1", 80]], [["main", float("inf"), 80]],
                     [["side-job", float("nan"), 80]], [["main", 1, -2]],
                     [["side-job", 1, 101]], [["main", 1, True]], [["main", 1, 70.5]],
                     [["main", 1, "PRIVATE_SEQUENCE"]])
        for value in malformed:
            config = copy.deepcopy(self.config)
            config["training"]["job_sequence"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "job_sequence") as caught:
                save_config(self.path, config)
            self.assertNotIn("PRIVATE_SEQUENCE", str(caught.exception))

    def test_separate_effort_keys_are_rejected(self):
        for field in ("main_effort", "side_job_effort"):
            config = copy.deepcopy(self.config)
            config["training"][field] = "PRIVATE_SEQUENCE"
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "replaced by training.job_sequence") as caught:
                save_config(self.path, config)
            self.assertNotIn("PRIVATE_SEQUENCE", str(caught.exception))

    def test_effort_defaults_are_independent(self):
        save_config(self.path, self.config)
        training = load_config(self.path)["training"]
        training["job_sequence"].append(("side-job", 1, 100))
        again = load_config(self.path)["training"]
        self.assertEqual(again["job_sequence"], [("main", 60, 80), ("main", 120, 70), ("main", 180, 10)])
        self.assertNotIn("side_job_effort", again)
        exported = copy.deepcopy(DEFAULT_CONFIG["training"])
        try:
            DEFAULT_CONFIG["training"]["job_sequence"][0][2] = 99
            self.config.pop("training")
            save_config(self.path, self.config)
            self.assertEqual(load_config(self.path)["training"]["job_sequence"][0], ("main", 60, 80))
        finally:
            DEFAULT_CONFIG["training"] = exported

    def test_pacer_waits_after_cuda_completion(self):
        stopped = Mock()
        with patch.object(gpu_pacing.time, "monotonic", side_effect=[10, 12, 12]), \
                patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:0", 50)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        synchronize.assert_called_once_with("cuda:0")
        stopped.wait.assert_called_once_with(2)

    def test_pacer_100_percent_skips_sync_and_sleep(self):
        stopped = Mock()
        with patch.object(gpu_pacing.torch.cuda, "synchronize") as synchronize:
            pacer = gpu_pacing.GpuPacer("cuda:0", 100)
            pacer.begin()
            pacer.finish()
            pacer.wait(stopped)
        synchronize.assert_not_called()
        stopped.wait.assert_not_called()


class WRPoolEffortTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["pool"].update(host="pool.example", WR_wallet="test-wallet", reconnect_delay=0)
        self.config["training"].update(m=128, n=128, max_batches=1)
        self.session = MagicMock()
        self.session.__enter__.return_value = self.session
        self.session.report_info.return_value = 1
        self.session.negotiated_gzip = False
        self.session.is_current.return_value = True
        self.session.submit.return_value = SimpleNamespace(accepted=True, error=None)
        self.session.wait_job.return_value = SimpleNamespace(
            job=SimpleNamespace(header_bytes=bytes(76), target=1, job_id="test", cert_version=3),
            generation=1,
        )
        self.factory = self.stack.enter_context(
            patch("LRAEP_trainer.wr_client.WRSession", return_value=self.session))
        self.stack.enter_context(patch.object(worker, "load_official_helpers"))
        self.stack.enter_context(patch.object(worker.torch.cuda, "is_available", return_value=True))
        self.stack.enter_context(patch.object(worker.torch.cuda, "get_device_name", return_value="test GPU"))
        self.stack.enter_context(patch.object(worker.torch.cuda, "synchronize"))
        self.stack.enter_context(patch.object(worker.native_cuda, "available", return_value=False))
        self.monitor = self.stack.enter_context(patch.object(worker, "GpuUtilization"))
        self.monitor.return_value.start.return_value.read.return_value = 42
        self.progress = self.stack.enter_context(patch.object(worker, "LiveProgress"))
        self.stack.enter_context(patch.object(worker, "pool_share_target", return_value=1))
        self.prepare = self.stack.enter_context(patch("LRAEP_trainer.wr_work.prepare_wr_batch"))
        self.verify = self.stack.enter_context(
            patch("LRAEP_trainer.wr_work.verified_wr_share", return_value="verified-share"))
        self.search = self.stack.enter_context(patch.object(
            worker, "search_batch", side_effect=lambda *_: iter([(0, 0, bytes(32))])))
        primitives = ModuleType("LRAEP_trainer.primitives")
        primitives.IncompleteBlockHeader = SimpleNamespace(from_bytes=Mock(return_value="header"))
        self.stack.enter_context(patch.dict(sys.modules, {"LRAEP_trainer.primitives": primitives}))
        self.stack.enter_context(patch.object(LRAEP_trainer, "primitives", primitives, create=True))

    def test_wr_pool_paces_after_submitting_and_before_the_next_batch(self):
        self.config["training"].update(max_batches=2, job_sequence=[("main", 60, 60)])
        order = []
        self.search.side_effect = lambda *_: (order.append("search"), iter([(0, 0, bytes(32))]))[1]
        self.session.submit.side_effect = lambda *_: (
            order.append("submit"), SimpleNamespace(accepted=True, error=None))[1]
        with patch.object(worker, "GpuPacer") as factory:
            pacer = factory.return_value
            pacer.max_utilization = 100
            pacer.set_utilization.side_effect = lambda value: setattr(pacer, "max_utilization", value)
            pacer.wait.side_effect = lambda *_a, **_kw: order.append("wait")
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(order, ["search", "submit", "wait", "search", "submit"])
        factory.assert_called_once_with(worker.torch.device("cuda:0"), 100)
        pacer.set_utilization.assert_called_once_with(60)
        self.assertEqual(pacer.begin.call_count, 2)
        self.assertEqual(pacer.finish.call_count, 2)
        pacer.wait.assert_called_once()

    def test_cancelled_search_is_paced_before_retrying(self):
        self.search.side_effect = [worker.TrainingCancelled("stale"), iter([])]
        with patch.object(worker, "GpuPacer") as factory:
            pacer = factory.return_value
            pacer.max_utilization = 80
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(pacer.finish.call_count, 2)
        pacer.wait.assert_called_once()
        self.assertIn("deadline", pacer.wait.call_args.kwargs)

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
        self.config["training"].update(job_sequence=schedule, max_batches=batches,
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
                self.assertEqual(worker.run_pool_training(self.config), 0)
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
                self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(order, ["search", "synchronize", "close"])
        worker.torch.cuda.synchronize.assert_called_once_with(worker.torch.device("cuda:0"))
        self.session.__exit__.assert_called_once()
        self.verify.assert_not_called()
        self.session.submit.assert_not_called()

    def test_initial_pause_delays_mining_until_the_positive_interval(self):
        with self.clocked_schedule([("main", 30, 0), ("main", 60, 70)]) as (clock, pacer):
            prepared_at = []
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(prepared_at, [30])
        self.assertEqual(clock.pauses, [(0, 30, 0, 0, 0)])
        pacer.set_utilization.assert_called_once_with(70)
        self.session.submit.assert_called_once()

    def test_pause_between_positive_intervals_performs_no_mining(self):
        with self.clocked_schedule([("main", 10, 80), ("main", 20, 0), ("main", 10, 30)], batches=2) as (clock, pacer):
            prepared_at = []
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)
            pacer.wait.side_effect = lambda *_a, **kw: setattr(clock, "now", kw["deadline"])
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(prepared_at, [0, 30])
        self.assertEqual(clock.pauses, [(10, 20, 1, 1, 1)])
        self.assertEqual(pacer.set_utilization.call_args_list, [call(80), call(30)])
        self.assertEqual(self.session.submit.call_count, 2)

    def test_mixed_sequence_switches_effort_and_repeats_from_the_first_interval(self):
        with self.clocked_schedule([("main", 10, 80), ("side-job", 20, 70)], batches=5) as (clock, pacer), \
                patch("LRAEP_trainer.side_job.SideJob") as cnn:
            prepared_at = []
            trained_at = []
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)
            cnn.return_value.step.side_effect = lambda **_kw: (trained_at.append(clock.now), True)[1]
            cnn.return_value.status.return_value = "Side-job | Epoch 1 | Step 1/20"
            pacer.wait.side_effect = lambda *_a, **kw: setattr(clock, "now", kw["deadline"])
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(prepared_at, [0, 30, 60, 90, 120])
        self.assertEqual(trained_at, [10, 40, 70, 100])
        self.assertEqual(pacer.set_utilization.call_args_list,
                         [call(80), call(70), call(80), call(70), call(80),
                          call(70), call(80), call(70), call(80)])
        self.assertEqual(self.search.call_count, 5)
        self.assertEqual(self.session.submit.call_count, 5)
        cnn.assert_called_once()
        cnn.return_value.close.assert_called_once()
        display = self.progress.return_value.__enter__.return_value
        self.assertTrue(any(args[0] == "Side-job | Epoch 1 | Step 1/20"
                            for args, _ in display.update.call_args_list))

    def test_initial_side_job_trains_before_pool_connection_or_job_wait(self):
        with self.clocked_schedule([("side-job", 10, 100), ("main", 20, 100)]) as (clock, _), \
                patch("LRAEP_trainer.side_job.SideJob") as cnn:
            def train_batch(**kwargs):
                self.factory.assert_not_called()
                self.session.wait_job.assert_not_called()
                self.prepare.assert_not_called()
                self.search.assert_not_called()
                self.session.submit.assert_not_called()
                self.assertFalse(kwargs['cancelled']())
                clock.now = 10
                return True
            cnn.return_value.step.side_effect = train_batch
            cnn.return_value.status.return_value = "Side-job | Epoch 1 | Preparing"
            self.assertEqual(worker.run_pool_training(self.config), 0)
        cnn.return_value.step.assert_called_once()
        cnn.return_value.close.assert_called_once()
        self.prepare.assert_called_once()
        self.session.submit.assert_called_once()

    def test_unlimited_sequence_repeats_all_jobs_until_explicitly_stopped(self):
        schedule = [('side-job', 2, 100), ('main', 3, 100), ('main', 1, 50)]
        with self.clocked_schedule(schedule, batches=0, max_seconds=0) as (clock, pacer), \
                patch('LRAEP_trainer.side_job.SideJob') as cnn:
            trained_at = []
            prepared_at = []
            cnn.return_value.step.side_effect = lambda **_kw: (trained_at.append(clock.now), True)[1]
            self.prepare.side_effect = lambda *_a, **_kw: prepared_at.append(clock.now)

            def advance(*_args, deadline):
                clock.now = deadline
                if clock.now >= 18:
                    self.factory.call_args.kwargs['stop_event'].set()

            pacer.wait.side_effect = advance
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(trained_at, [0, 6, 12])
        self.assertEqual(prepared_at, [2, 5, 8, 11, 14, 17])
        self.assertEqual(self.session.submit.call_count, 6)
        cnn.assert_called_once()
        cnn.return_value.close.assert_called_once()
        self.assertEqual(clock.now, 18)

    def test_side_job_only_schedule_never_connects_or_submits_mining_work(self):
        with self.clocked_schedule([("side-job", 10, 100), ("main", 0, -1)], batches=0) as (clock, pacer), \
                patch("LRAEP_trainer.side_job.SideJob") as cnn:
            cnn.return_value.step.return_value = True
            pacer.wait.side_effect = lambda *_a, **kw: setattr(clock, "now", kw["deadline"])
            self.assertEqual(worker.run_pool_training(self.config), 0)
        cnn.return_value.step.assert_called_once()
        cnn.return_value.close.assert_called_once()
        self.factory.assert_not_called()
        self.search.assert_not_called()
        self.verify.assert_not_called()
        self.session.submit.assert_not_called()

    def test_paused_side_job_does_not_construct_a_cnn(self):
        with self.clocked_schedule([("side-job", 10, 0), ("main", 20, 100)]) as (clock, _), \
                patch("LRAEP_trainer.side_job.SideJob") as cnn:
            self.assertEqual(worker.run_pool_training(self.config), 0)
        self.assertEqual(clock.now, 10)
        cnn.assert_not_called()
        self.prepare.assert_called_once()

    def test_side_job_failure_closes_cnn_and_telemetry(self):
        with self.clocked_schedule([("side-job", 10, 100)]) as (_, pacer), \
                patch("LRAEP_trainer.side_job.SideJob") as cnn:
            cnn.return_value.step.side_effect = RuntimeError("fixture training failure")
            with self.assertRaisesRegex(RuntimeError, "fixture training failure"):
                worker.run_pool_training(self.config)
        cnn.return_value.close.assert_called_once()
        pacer.finish.assert_called_once_with(cancelled=True)
        self.monitor.return_value.start.return_value.close.assert_called_once()
        self.factory.assert_not_called()

    def test_zero_effort_until_time_limit_does_not_initialize_or_mine(self):
        with self.clocked_schedule([("main", 30, 0)], max_seconds=10) as (clock, pacer):
            self.assertEqual(worker.run_pool_training(self.config), 0)
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
        self.config["training"]["job_sequence"] = [("main", 0, -1), ("main", 60, 80)]
        with patch.object(worker.torch.cuda, "is_available") as cuda:
            self.assertEqual(worker.run_pool_training(self.config), 0)
        cuda.assert_not_called()
        worker.load_official_helpers.assert_not_called()
        self.factory.assert_not_called()
        self.prepare.assert_not_called()
        self.search.assert_not_called()
        self.verify.assert_not_called()
        self.monitor.assert_not_called()

    def test_terminal_interval_interrupts_blocked_authentication_and_cleans_up(self):
        self.config["training"]["job_sequence"] = [("main", 0.3, 80), ("main", 0, -1)]

        def wait_for_exit():
            stopped = self.factory.call_args.kwargs["stop_event"]
            self.assertTrue(stopped.wait(2), "schedule did not stop authentication")
            return self.session

        self.session.__enter__.side_effect = wait_for_exit
        self.assertEqual(worker.run_pool_training(self.config), 0)
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

        self.config["training"]["job_sequence"] = [("main", 0.3, 80), ("main", 0, -1)]

        def wait_for_exit(*_):
            stopped = self.factory.call_args.kwargs["stop_event"]
            self.assertTrue(stopped.wait(2), "schedule did not stop share submission")
            raise PoolConnectionError("scheduled exit")

        self.session.submit.side_effect = wait_for_exit
        self.assertEqual(worker.run_pool_training(self.config), 0)
        self.factory.assert_called_once()
        self.session.submit.assert_called_once()
        self.session.__exit__.assert_called_once()
        self.monitor.return_value.start.return_value.close.assert_called_once()
        self.progress.return_value.__exit__.assert_called_once()

    def test_main_jobs_use_one_shared_sequence(self):
        self.config["training"].update(job_sequence=[("main", 60, 70)])
        with patch.object(worker, "JobSequence", wraps=worker.JobSequence) as schedules:
            self.assertEqual(worker.run_pool_training(self.config), 0)
        schedules.assert_called_once()
        self.assertEqual(schedules.call_args.args[0], [("main", 60, 70)])
        self.prepare.assert_called_once()
        self.search.assert_called_once()
        self.session.submit.assert_called_once()



if __name__ == "__main__":
    unittest.main()
